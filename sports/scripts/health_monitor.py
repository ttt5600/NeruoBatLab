"""Hourly health check of every quant launchd job (ops_health_monitor, 2026-11-18; notes/ops_health_monitor.md).

Why: two silent failures in one week -- E4 pick logger wrote 0 picks on 2 nights (open-meteo timeouts; launchd exit 0
because run.sh's last command succeeded) and the Q013 RT logger wrote 0 picks after a field rename. launchd's exit code
alone catches neither, so per job (config: scripts/health_jobs.json) this checks:
  (a) launchd: loaded as expected? last exit code; recent error blocks in the job's own log; fresh stderr
  (b) output freshness: the output file grew within its max gap (only while the job is ACTIVE: season / month tail /
      until-date windows), and the last row's own timestamp is inside the gap
  (c) row sanity: key fields non-empty in the last 20 matching rows (a field rename shows up here)
  (d) pick loggers: consecutive complete ACTIVE PT days with 0 pick rows (>= zero_days -> BROKEN)
Statuses (worst first): BROKEN, STALE, WARN, OK; plus IDLE / RETIRED / OFF-SEASON (no output expected).

Outputs (deployed root ~/.claude/quant-sports, launchd cannot read ~/Desktop):
  digests/health-<PT date>.md   rewritten every run (latest status of the day)
  state/health_status.json      machine-readable latest (read by sports/scripts/scorecards.py card_health)
  data/health/history.jsonl     one row per job per run
  macOS notification (osascript, as kalshi_exec.notify) at most once per job per PT day, for STALE/BROKEN only
  (state/health_notified.json).
READ-ONLY w.r.t. every job: never edits, restarts or kills anything. Stdlib only.

  python scripts/health_monitor.py [--no-notify] [--now ISO] [--config PATH] [--root PATH]
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import gzip
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    PT = ZoneInfo("America/Los_Angeles")
except Exception:  # noqa: BLE001
    PT = timezone(timedelta(hours=-8))

HOME = Path.home()
K = HOME / ".claude" / "quant-kalshi"
S = HOME / ".claude" / "quant-sports"
UID = os.getuid()
SEV = {"BROKEN": 5, "STALE": 4, "WARN": 3, "OK": 1, "IDLE": 0, "OFF-SEASON": 0, "RETIRED": 0}
ERR_RE = re.compile(r"Traceback|[A-Za-z]*Error\b|exit [1-9]\d*|FETCH_FAIL|rc=[1-9]\d*|Killed")
HDR_RE = re.compile(r"^=== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
TAIL_BYTES = 256_000
PICK_SCAN_BYTES = 40_000_000
EPOCH_LO, EPOCH_HI = 946_684_800, 4_102_444_800        # 2000..2100 (traps.check_epoch_seconds rule)


def X(p: str) -> Path:
    return Path(p.replace("$K", str(K)).replace("$S", str(S))).expanduser()


# ------------------------------------------------------------------------------------------------ time helpers
def parse_ts(v) -> datetime | None:
    """ISO string (naive = UTC, as every logger here writes UTC) or epoch seconds. Epochs outside 2000..2100 are rejected
    (ms/us mixups; traps.check_epoch_seconds)."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)) or (isinstance(v, str) and re.fullmatch(r"\d{9,13}(\.\d+)?", v)):
        x = float(v)
        if not EPOCH_LO <= x <= EPOCH_HI:
            return None
        return datetime.fromtimestamp(x, timezone.utc)
    try:
        s = str(v).replace("Z", "+00:00")
        if re.fullmatch(r"\d{4}-\d\d-\d\d", s):
            return datetime.fromisoformat(s).replace(tzinfo=PT)
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def in_season(span, d: date) -> bool:
    a, b = (tuple(int(x) for x in s.split("-")) for s in span)
    md = (d.month, d.day)
    return a <= md <= b if a <= b else (md >= a or md <= b)


def active(spec: dict | None, now: datetime, seasons: dict) -> tuple[bool, str]:
    """(active?, why-not). `now` is a datetime; day-level rules use its PT date."""
    if not spec:
        return True, ""
    d = now.astimezone(PT).date()
    if "season" in spec:
        span = seasons.get(spec["season"], spec["season"])
        if not in_season(span, d):
            return False, f"off-season ({spec['season']} {span[0]}..{span[1]})"
    if "weekdays" in spec and d.weekday() not in spec["weekdays"]:
        return False, "not a scheduled weekday"
    if "month_tail" in spec:
        nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
        if (nxt - d).days > spec["month_tail"]:
            return False, f"only active in the last {spec['month_tail']} days of a month"
    if "from" in spec and now < parse_ts(spec["from"]):
        return False, f"starts {spec['from']}"
    if "until" in spec and now >= parse_ts(spec["until"]):
        return False, f"window ended {spec['until']}"
    return True, ""


def age_h(t: datetime | float | None, now: datetime) -> float | None:
    if t is None:
        return None
    if isinstance(t, (int, float)):
        t = datetime.fromtimestamp(t, timezone.utc)
    return (now - t).total_seconds() / 3600


def fmt_h(h: float | None) -> str:
    if h is None:
        return "n/a"
    return f"{h * 60:.0f}m" if h < 1 else (f"{h:.1f}h" if h < 48 else f"{h / 24:.1f}d")


# ------------------------------------------------------------------------------------------------ file readers
def tail_text(p: Path, nbytes: int = TAIL_BYTES) -> list[str]:
    if p.suffix == ".gz":
        try:
            with gzip.open(p, "rt", errors="ignore") as f:
                data = f.read()
        except (OSError, EOFError):        # appended gzip members can be mid-write
            return []
        return data[-nbytes:].splitlines()[1 if len(data) > nbytes else 0:]
    with open(p, "rb") as f:
        f.seek(0, 2)
        n = f.tell()
        f.seek(max(0, n - nbytes))
        lines = f.read().decode(errors="ignore").splitlines()
    return lines[1:] if n > nbytes else lines


def rows(lines: list[str]) -> list[dict]:
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            out.append(json.loads(ln.replace("NaN", "null")))
        except json.JSONDecodeError:
            continue
    return out


def match(r: dict, where: dict | None) -> bool:
    for k, v in (where or {}).items():
        val = r.get(k)
        if isinstance(v, list):
            if not any(fnmatch.fnmatch(str(val), pat) for pat in v):
                return False
        elif val != v:
            return False
    return True


def empty(v) -> bool:
    return v is None or v == "" or (isinstance(v, float) and math.isnan(v)) or str(v).lower() in ("nan", "none", "null")


# ------------------------------------------------------------------------------------------------ checks
def launchd(label: str) -> dict:
    full = f"com.jonathanwang.{label}"
    try:
        r = subprocess.run(["launchctl", "print", f"gui/{UID}/{full}"], capture_output=True, text=True, timeout=10)
    except Exception as e:  # noqa: BLE001
        return {"loaded": None, "err": str(e)}
    if r.returncode != 0:
        return {"loaded": False, "plist": (HOME / "Library" / "LaunchAgents" / f"{full}.plist").exists()}
    g = lambda pat: (m.group(1) if (m := re.search(pat, r.stdout)) else None)  # noqa: E731
    return {"loaded": True, "last_exit": g(r"last exit code = (.+)"), "runs": g(r"\bruns = (\d+)"),
            "state": g(r"\n\tstate = (\S+)")}


def log_errors(p: Path, now: datetime, hours: float = 24, extra: str | None = None, err_re: str | None = None) -> dict:
    """Error blocks in the last `hours` of a '=== <local time>' block log. Logs without headers: scan the tail lines."""
    if not p.exists():
        return {"missing": True}
    lines = tail_text(p, 2_000_000)
    since = now - timedelta(hours=hours)
    blocks, cur = [], None
    for ln in lines:
        m = HDR_RE.match(ln)
        if m:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=PT)
            cur = [t, []]
            blocks.append(cur)
        elif cur is not None:
            cur[1].append(ln)
    out = {"blocks": 0, "err_blocks": 0, "warn_blocks": 0, "last_err": "", "last_err_t": None, "last_block_err": False}
    wre = re.compile(extra) if extra else None
    ERR = re.compile(err_re) if err_re else ERR_RE
    if not blocks:       # headerless log: judge on mtime window only
        if now.timestamp() - p.stat().st_mtime < hours * 3600:
            bad = [ln for ln in lines[-200:] if ERR.search(ln)]
            out.update(blocks=1, err_blocks=int(bool(bad)), last_err=(bad[-1][:200] if bad else ""))
        return out
    for t, body in blocks:
        if not since <= t <= now:
            continue
        out["blocks"] += 1
        bad = [ln for ln in body if ERR.search(ln)]
        if bad:
            out["err_blocks"] += 1
            out["last_err"], out["last_err_t"] = bad[-1].strip()[:200], t.isoformat(timespec="minutes")
        if wre and any(wre.search(ln) for ln in body):
            out["warn_blocks"] += 1
            if not bad:
                out["last_warn"] = next(ln for ln in body if wre.search(ln)).strip()[:200]
        out["last_block_err"] = bool(bad)
    return out


def check_output(o: dict, now: datetime, seasons: dict) -> dict:
    if "glob" in o:
        ms = sorted(glob.glob(str(X(o["glob"]))), key=os.path.getmtime)
        p = Path(ms[-1]) if ms else X(o["glob"])
    else:
        p = X(o["path"])
    on, why = active(o.get("active"), now, seasons)
    res = {"path": str(p).replace(str(HOME), "~"), "active": on, "why_inactive": why}
    if not p.exists():
        res.update(status="BROKEN" if on else "OFF-SEASON", msg="output file missing")
        return res
    st = p.stat()
    res["mtime_age_h"] = age_h(st.st_mtime, now)
    res["size"] = st.st_size
    issues, status = [], "OK"
    if o.get("ts") or o.get("keys"):
        rs = [r for r in rows(tail_text(p)) if match(r, o.get("where"))]
        if o.get("ts"):
            parsed = [t for r in rs if (t := parse_ts(r.get(o["ts"]))) is not None]
            ts = [t for t in parsed if t <= now + timedelta(hours=1)]
            res["last_row_age_h"] = age_h(max(ts), now) if ts else None
            if rs and not parsed:
                issues.append(f"timestamp field '{o['ts']}' unparseable/missing in last rows")
                status = "BROKEN"
        last = rs[-20:]
        if o.get("keys") and last:
            bad = Counter(k for r in last for k in o["keys"] if empty(r.get(k)))
            if bad:
                worst = max(bad.values())
                issues.append("empty key fields in last %d rows: %s" % (len(last), ", ".join(f"{k} {n}/{len(last)}" for k, n in bad.items())))
                status = "BROKEN" if worst == len(last) else "WARN"
        elif o.get("keys") and not rs:
            issues.append("no matching rows in the file tail")
            status = "WARN"
    if on:
        gap = o["max_gap_h"]
        age = res["mtime_age_h"]
        row_age = res.get("last_row_age_h")
        if age > gap:
            issues.append(f"not written for {fmt_h(age)} (max {fmt_h(gap)})")
            status = "STALE" if SEV["STALE"] > SEV[status] else status
        elif row_age is not None and row_age > gap:
            issues.append(f"file touched {fmt_h(age)} ago but newest row is {fmt_h(row_age)} old (max {fmt_h(gap)})")
            status = "STALE" if SEV["STALE"] > SEV[status] else status
    else:
        issues.append(why)
        if status == "OK":
            status = "OFF-SEASON"
    res.update(status=status, msg="; ".join(issues))
    return res


def check_picks(pk: dict, now: datetime, seasons: dict, days: int = 14) -> dict:
    p = X(pk["path"])
    what = pk.get("what", "picks")
    if not p.exists():
        return {"status": "BROKEN", "msg": f"{what} file missing: {p}"}
    rs = [r for r in rows(tail_text(p, PICK_SCAN_BYTES)) if match(r, pk.get("where"))]
    per = Counter()
    first = None
    for r in rs:
        t = parse_ts(r.get(pk["ts"]))
        if t is None or t > now:
            continue
        d = t.astimezone(PT).date()
        per[d] += 1
        first = d if first is None or d < first else first
    today = now.astimezone(PT).date()
    hist, streak, counting = [], 0, True
    for i in range(1, days + 1):                     # complete days only: yesterday backwards
        d = today - timedelta(days=i)
        on, _ = active(pk.get("active"), datetime.combine(d, datetime.min.time(), PT).replace(hour=12), seasons)
        if first is None or d < first:
            break
        hist.append((d.isoformat(), per.get(d, 0), on))
        if not on:
            continue
        if counting and per.get(d, 0) == 0:
            streak += 1
        else:
            counting = False
    res = {"what": what, "today": per.get(today, 0), "per_day": hist[:7][::-1], "zero_streak": streak,
           "zero_days": pk["zero_days"]}
    bad_keys = ""
    if pk.get("keys") and rs:
        last = rs[-20:]
        bad = Counter(k for r in last for k in pk["keys"] if empty(r.get(k)))
        if bad:
            bad_keys = "; empty key fields in last pick rows: " + ", ".join(f"{k} {n}/{len(last)}" for k, n in bad.items())
            if max(bad.values()) == len(last):
                res["status"] = "BROKEN"
    days_txt = " ".join(f"{d[5:]}:{n}" + ("" if on else "(off)") for d, n, on in res["per_day"])
    if streak >= pk["zero_days"]:
        res["status"] = "BROKEN"
        res["msg"] = f"0 {what} on the last {streak} active day(s) (limit {pk['zero_days']}); per day {days_txt}" + bad_keys
    else:
        res.setdefault("status", "WARN" if bad_keys else "OK")
        res["msg"] = f"{what} per PT day {days_txt}; today so far {res['today']}" + bad_keys
    return res


def check_job(j: dict, now: datetime, seasons: dict) -> dict:
    expect = j.get("expect", "loaded")
    ld = launchd(j["label"])
    res = {"job": j["job"], "label": j["label"], "expect": expect, "launchd": ld, "checks": [], "status": "OK"}

    def add(status: str, kind: str, msg: str):
        res["checks"].append({"kind": kind, "status": status, "msg": msg})
        if SEV[status] > SEV.get(res["status"], 0):
            res["status"] = status

    if expect == "unloaded":
        if ld.get("loaded"):
            add("WARN", "launchd", f"expected unloaded ({j.get('note', '')}) but it is LOADED")
        else:
            res["status"] = "RETIRED"
            res["checks"].append({"kind": "launchd", "status": "RETIRED", "msg": j.get("note", "unloaded by design")})
        return res
    if not ld.get("loaded"):
        add("BROKEN", "launchd", "NOT LOADED in launchd" + ("" if ld.get("plist") else " and no plist"))
    elif ld.get("last_exit") not in ("0", "(never exited)", None):
        add("BROKEN", "launchd", f"last exit code {ld['last_exit']}")

    logs = [X(x) for x in j.get("logs", [])]
    errs = [X(x) for x in j.get("stderr", [])]
    if j.get("qs"):
        logs.append(S / "logs" / f"{j['qs']}.log")
        errs.append(S / "logs" / f"launchd-{j['qs']}.err")
    for lp in logs:
        e = log_errors(lp, now, 24, j.get("warn_re"), j.get("err_re"))
        if e.get("missing"):
            continue
        if e["err_blocks"]:
            frac = e["err_blocks"] / max(e["blocks"], 1)
            st = "BROKEN" if (e["blocks"] >= 3 and frac >= 0.5 and e["last_block_err"]) else "WARN"
            add(st, "log", f"{lp.name}: {e['err_blocks']}/{e['blocks']} runs with errors in 24h; last "
                           f"{e['last_err_t'] or ''} `{e['last_err']}`")
        if e.get("warn_blocks"):
            add("WARN", "log", f"{lp.name}: {e['warn_blocks']} run(s) matching /{j['warn_re']}/ in 24h; e.g. `{e.get('last_warn', '')}`")
    for ep in errs:
        if ep.exists() and ep.stat().st_size > 0 and now.timestamp() - ep.stat().st_mtime < 24 * 3600:
            last = (tail_text(ep, 2000) or [""])[-1][:200]
            add("WARN", "stderr", f"{ep.name} written {fmt_h(age_h(ep.stat().st_mtime, now))} ago: `{last}`")

    if expect == "idle":
        if res["status"] == "OK":
            res["status"] = "IDLE"
        res["checks"].append({"kind": "note", "status": "IDLE", "msg": j.get("note", "no output expected")})
        return res

    hb = j.get("heartbeat")
    if hb:
        p = X(hb["path"])
        if not p.exists():
            add("BROKEN", "heartbeat", f"{p.name} missing")
        else:
            a = age_h(p.stat().st_mtime, now)
            txt = p.read_text(errors="ignore").strip()[:80]
            if a > hb["max_gap_h"]:
                add("STALE", "heartbeat", f"{p.name} {fmt_h(a)} old (max {fmt_h(hb['max_gap_h'])}): {txt}")
            elif re.search(r"rc=[1-9]", txt):
                add("WARN", "heartbeat", f"last run non-zero: {txt}")
    outs = [check_output(o, now, seasons) for o in j.get("outputs", [])]
    for o in outs:
        msg = f"{o['path']}: written {fmt_h(o.get('mtime_age_h'))} ago" + (
            f", newest row {fmt_h(o['last_row_age_h'])}" if o.get("last_row_age_h") is not None else "") + (
            f" -- {o['msg']}" if o.get("msg") else "")
        add(o["status"], "output", msg)
    if j.get("picks"):
        pk = check_picks(j["picks"], now, seasons)
        on, why = active(j["picks"].get("active"), now, seasons)
        add(pk["status"] if on else "OFF-SEASON", "picks", pk["msg"] + ("" if on else f" ({why})"))
        res["picks"] = {k: pk.get(k) for k in ("what", "today", "per_day", "zero_streak", "zero_days")}
    if outs and all(o["status"] == "OFF-SEASON" for o in outs) and res["status"] in ("OK", "OFF-SEASON"):
        res["status"] = "OFF-SEASON"
    return res


# ------------------------------------------------------------------------------------------------ alerting + output
def notify(title: str, msg: str) -> None:
    """macOS notification; never raises (same pattern as quant-kalshi scripts/kalshi_exec.py notify)."""
    safe = lambda s: str(s).replace("\\", "").replace('"', "'")[:200]  # noqa: E731
    try:
        subprocess.run(["osascript", "-e", f'display notification "{safe(msg)}" with title "{safe(title)}"'],
                       timeout=5, capture_output=True)
    except Exception:  # noqa: BLE001
        pass


def alert(results: list[dict], root: Path, today: str, send: bool) -> list[str]:
    """notify_once per job per PT day for STALE/BROKEN; one notification lists every newly alerting job."""
    sp = root / "state" / "health_notified.json"
    try:
        seen = json.loads(sp.read_text())
    except Exception:  # noqa: BLE001
        seen = {}
    seen = {k: v for k, v in seen.items() if k.endswith(today) or v >= (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()}
    new = []
    for r in results:
        if r["status"] in ("STALE", "BROKEN"):
            key = f"{r['label']}|{today}"
            if key not in seen:
                seen[key] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                new.append(f"{r['job'].split(' (')[0]} {r['status']}")
    if new and send:
        notify(f"Quant health: {len(new)} job(s) need attention", "; ".join(new) + f" -- digests/health-{today}.md")
    if send:
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(seen, indent=1))
    return new


def unconfigured(cfg_labels: set[str]) -> list[str]:
    la = HOME / "Library" / "LaunchAgents"
    return sorted(p.stem.replace("com.jonathanwang.", "") for p in la.glob("com.jonathanwang.quant-*.plist")
                  if p.stem.replace("com.jonathanwang.", "") not in cfg_labels)


def digest(results: list[dict], now: datetime, extra: list[str], new: list[str]) -> str:
    c = Counter(r["status"] for r in results)
    order = sorted(results, key=lambda r: (-SEV.get(r["status"], 0), r["job"]))
    out = [f"# Quant job health -- {now.astimezone(PT):%Y-%m-%d %H:%M %Z}", "",
           "Generated hourly by `~/.claude/quant-sports/scripts/health_monitor.py` (source: sports/scripts/health_monitor.py, "
           "config scripts/health_jobs.json). Read-only. " + ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda x: -SEV.get(x[0], 0))), ""]
    if new:
        out += ["Notified this run: " + "; ".join(new), ""]
    bad = [r for r in order if SEV.get(r["status"], 0) >= SEV["WARN"]]
    out.append("## Needs attention")
    for r in bad:
        out.append(f"### {r['status']} -- {r['job']} (`{r['label']}`)")
        out += [f"- [{c['status']}] {c['kind']}: {c['msg']}" for c in r["checks"] if SEV.get(c["status"], 0) >= SEV["WARN"]]
        out.append("")
    if not bad:
        out += ["- nothing", ""]
    if extra:
        out += ["## Not in the monitor config"] + [f"- {x}" for x in extra] + [""]
    out += ["## All jobs", "| Job | Status | launchd (last exit / runs) | Summary |", "|---|---|---|---|"]
    for r in order:
        ld = r["launchd"]
        lds = "not loaded" if not ld.get("loaded") else f"{ld.get('last_exit')} / {ld.get('runs')}"
        worst = max(r["checks"], key=lambda c: SEV.get(c["status"], 0), default={"msg": ""})
        summ = worst["msg"] if r["checks"] else ""
        pk = next((c["msg"] for c in r["checks"] if c["kind"] == "picks"), "")
        if pk and pk != summ:
            summ += " // " + pk
        out.append(f"| {r['job']} | {r['status']} | {lds} | {summ.replace('|', '/')[:300]} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(here / "health_jobs.json"))
    ap.add_argument("--root", default=str(S), help="where digests/, state/, data/health/ go")
    ap.add_argument("--no-notify", action="store_true")
    ap.add_argument("--now", help="evaluate as of this ISO time (replay; log/pick checks only are time-aware)")
    ap.add_argument("--print", action="store_true", help="print the digest to stdout")
    a = ap.parse_args(argv)
    now = parse_ts(a.now) if a.now else datetime.now(timezone.utc)
    cfg = json.loads(Path(a.config).read_text())
    root = Path(a.root)
    results = []
    for j in cfg["jobs"]:
        try:
            results.append(check_job(j, now, cfg.get("seasons", {})))
        except Exception as e:  # noqa: BLE001 -- one bad config entry must not hide the others
            results.append({"job": j.get("job"), "label": j.get("label"), "status": "WARN", "launchd": {},
                            "checks": [{"kind": "monitor", "status": "WARN", "msg": f"monitor error {type(e).__name__}: {e}"}]})
    extra = unconfigured({j["label"] for j in cfg["jobs"]})
    today = now.astimezone(PT).date().isoformat()
    replay = bool(a.now)
    new = alert(results, root, today, send=not (a.no_notify or replay))
    md = digest(results, now, extra, new)
    if not replay:
        (root / "digests").mkdir(parents=True, exist_ok=True)
        (root / "digests" / f"health-{today}.md").write_text(md)
        (root / "state").mkdir(parents=True, exist_ok=True)
        (root / "state" / "health_status.json").write_text(json.dumps(
            {"generated": now.isoformat(timespec="seconds"), "digest": str(root / "digests" / f"health-{today}.md"),
             "unconfigured": extra, "jobs": results}, indent=1, default=str))
        (root / "data" / "health").mkdir(parents=True, exist_ok=True)
        with open(root / "data" / "health" / "history.jsonl", "a") as f:
            for r in results:
                f.write(json.dumps({"ts": now.isoformat(timespec="seconds"), "job": r["label"], "status": r["status"],
                                    "issues": [c["msg"][:160] for c in r["checks"] if SEV.get(c["status"], 0) >= SEV["WARN"]]}) + "\n")
    if a.print or replay:
        print(md)
    c = Counter(r["status"] for r in results)
    print(f"health {now.isoformat(timespec='minutes')}: " + ", ".join(f"{k} {v}" for k, v in c.items()) +
          (f"; notified: {'; '.join(new)}" if new else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
