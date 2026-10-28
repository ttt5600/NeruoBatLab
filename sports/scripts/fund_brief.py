"""DAILY PM BRIEF for the research fund (notes/FUND_ARCHITECTURE.md role 10, Reporter). READ-ONLY: places no orders.

    ../.venv_quant/bin/python scripts/fund_brief.py            # from the repo (sports/)
    ~/.claude/quant-kalshi/.venv/bin/python scripts/fund_brief.py   # from the deployed copy (launchd; see installer)
    options: --date YYYY-MM-DD (file name only), --stdout (print, do not write), --timeout SECONDS per reporter

Pulls, all read-only:
  1. executor status  (kalshi_exec.py status): live record + P&L, open exposure vs caps, recent errors. Balance and
     bankroll lines are REDACTED -- the brief never shows account values.
  2. every forward paper logger's report: nba_open_totals, open_lines (nhl, nfl), kalshi_wx_monthly_logger,
     nba_midgame_scan.
  3. manual_bets.py orders (user-authorized manual Kalshi bets).
  4. fund_stats report (the statistician gate over lab/ledger.jsonl + lab/lockbox.jsonl).
  5. qualified daily volume per edge vs the doubling target.
Writes <root>/notes/briefs/brief_YYYY-MM-DD.md where <root> is the copy this script runs from.

The reporters always run inside the DEPLOYED copy (~/.claude/quant-kalshi, env FUND_DEPLOY): that is where the
launchd loggers keep their data and the executor keeps its key. launchd cannot read ~/Desktop (TCC), so the deployed
brief reads the ledger snapshot the installer copied into <deploy>/lab/ (its age is printed).

Doubling target: $1-2k profit in 12 months = $2.74-5.48/day; at a 2% net edge that is ~$140-280/day of qualified
volume. QUALIFIED = the edge passed sealed (gate verdict PASSED or SEALED-ASK) AND is switched on in the executor.
Planning ROIs come from notes/fund_economics.md (about 2/3 of sealed), not from the sealed point estimates.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = Path(os.environ.get("FUND_DEPLOY", Path.home() / ".claude" / "quant-kalshi"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("FUND_DATA", str(DEPLOY / "data"))
import fund_stats as FS  # noqa: E402

TARGET_PROFIT_DAY = (1000 / 365, 2000 / 365)
TARGET_EDGE = 0.02
GOAL_BANKROLL = (1000.0, 2000.0)
NIGHT_FRAC = 0.10  # scripts/kalshi_exec.py: day and open caps = min(env cap, NIGHT_FRAC * bankroll)

# Planning assumptions (notes/fund_economics.md 2026-10-27). Each: planning net ROI per $ and how capacity is measured.
PLAN = {
    "weather_taker": {"roi": 0.006, "src": "fund_economics: taker plan 0.6% (sealed 1.07%)"},
    "weather_maker": {"roi": 0.015, "src": "fund_economics: maker plan 1.5% (sealed 2.22%); cap ~$75/pick, fill ~80%"},
    "maker_cap_per_pick": 75.0, "maker_fill": 0.80, "taker_depth_frac": 0.5,
}

REDACT = [(re.compile(r"balance \$?-?[\d,]+(\.\d+)?"), "balance [redacted]"),
          (re.compile(r"bankroll: \$?[\d,]+(\.\d+)?"), "bankroll: [redacted]"),
          (re.compile(r"(?i)(portfolio[_ ]value|account[_ ]value)\D*[\d,.]+"), r"\1 [redacted]")]


def redact(text: str) -> str:
    for rx, rep in REDACT:
        text = rx.sub(rep, text)
    return text


def python_for_deploy() -> str:
    p = DEPLOY / ".venv" / "bin" / "python"
    return str(p) if p.exists() else sys.executable


def run(args: list[str], timeout: int) -> tuple[int, str]:
    """Run a deployed reporter read-only; never raise."""
    script = DEPLOY / "scripts" / args[0]
    if not script.exists():
        return 127, f"(missing: {script})"
    try:
        r = subprocess.run([python_for_deploy(), str(script), *args[1:]], cwd=DEPLOY, capture_output=True, text=True,
                           timeout=timeout, env={**os.environ, "PYTHONWARNINGS": "ignore"})
        out = (r.stdout + ("\n" + r.stderr if r.returncode else "")).strip()
        out = "\n".join(ln for ln in out.splitlines() if "Warning" not in ln and "warnings.warn" not in ln)
        return r.returncode, redact(out)
    except subprocess.TimeoutExpired:
        return 124, f"(timed out after {timeout}s)"
    except Exception as ex:  # noqa: BLE001
        return 1, f"(failed: {type(ex).__name__}: {str(ex)[:200]})"


def read_jsonl(p: Path) -> list[dict]:
    return FS.read_jsonl(p)


# ------------------------------------------------------------------------------------------------ executor
def parse_exec(text: str) -> dict:
    out = {"mode": None, "style": None, "kill": None, "live": None, "errors": []}
    m = re.search(r"launchd mode: (\w+).*?KILL: ([^;]+);.*exec style: (.+)", text)
    if m:
        out["mode"], out["kill"], out["style"] = m.group(1), m.group(2).strip(), m.group(3).strip()
    m = re.search(r"\[live\] (\d+) decisions, (\d+) with contracts, (\d+) settled \((\d+) wins\)\s+P&L \$([+-][\d,.]+)"
                  r".*?\n\s+today \$([\d,.]+) / cap \$([\d,.]+)\s+open \$([\d,.]+) / cap \$([\d,.]+)\s+24h net "
                  r"\$([+-][\d,.]+)\s+7d net \$([+-][\d,.]+)", text, re.S)
    if m:
        f = [float(x.replace(",", "")) for x in m.groups()]
        out["live"] = dict(zip(["decisions", "with_contracts", "settled", "wins", "pnl", "today", "day_cap", "open",
                                "open_cap", "net24", "net7d"], f))
        out["live"]["open_count"] = len(re.findall(r"^\s+open KX", text, re.M))
    out["errors"] = [ln.strip() for ln in text.splitlines() if ln.startswith("recent ")]
    return out


# ------------------------------------------------------------------------------------------- volume model
def weather_volume(days: int = 14) -> dict:
    """Picks/day and taker/maker capacity $/day from the deployed paper log (every qualifying pick, both cohorts)."""
    rows = read_jsonl(DEPLOY / "data" / "kalshi_paper.jsonl")
    cut = datetime.now(timezone.utc) - timedelta(days=days)
    by_day: dict[str, list[dict]] = {}
    for r in rows:
        try:
            t = datetime.fromisoformat(str(r.get("logged_at")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if t >= cut:
            by_day.setdefault(t.date().isoformat(), []).append(r)
    if not by_day:
        return {"days": 0}
    n_days = max(len(by_day), 1)
    picks = sum(len(v) for v in by_day.values()) / n_days
    dollars = sorted(float(r.get("depth_1c") or 0) * float(r.get("no_price") or 0) for v in by_day.values() for r in v)
    med = dollars[len(dollars) // 2] if dollars else 0.0  # median: a few very deep books dominate the mean
    taker = picks * PLAN["taker_depth_frac"] * med
    maker = picks * PLAN["maker_fill"] * PLAN["maker_cap_per_pick"]
    return {"days": n_days, "picks_day": picks, "median_depth_usd": med, "taker_cap_day": taker,
            "maker_cap_day": maker}


def volume_table(ex: dict, gate_leads: dict) -> tuple[list[str], dict]:
    wv = weather_volume()
    live = ex.get("live") or {}
    deployed = live.get("day_cap") or 0.0
    style = (ex.get("style") or "taker").lower()
    roi_now = PLAN["weather_maker"]["roi"] if style.startswith("maker") else PLAN["weather_taker"]["roi"]
    rows = []
    q_ok = gate_leads.get("E1") in ("PASSED", "SEALED-ASK") and gate_leads.get("E2") in ("PASSED", "SEALED-ASK")
    rows.append(("E1+E2 weather (live, %s)" % style, gate_leads.get("E1"), deployed if q_ok else 0.0, roi_now,
                 f"executor day cap (binding); {PLAN['weather_taker' if roi_now < .01 else 'weather_maker']['src']}"))
    if wv.get("days"):
        rows.append(("  book capacity, taker", "-", wv["taker_cap_day"], PLAN["weather_taker"]["roi"],
                     f"{wv['picks_day']:.1f} picks/day x 0.5 x median ${wv['median_depth_usd']:,.0f} within 1c "
                     f"({wv['days']} days of paper)"))
        rows.append(("  book capacity, E4 maker ($75/pick)", gate_leads.get("E4"), wv["maker_cap_day"],
                     PLAN["weather_maker"]["roi"], "fill ~80%; needs KALSHI_EXEC_STYLE=maker (PM)"))
    for b in GOAL_BANKROLL:  # the executor stakes at most NIGHT_FRAC of bankroll per night; picks settle next day
        rows.append((f"  at goal bankroll ${b:,.0f}, maker", gate_leads.get("E4"),
                     min(NIGHT_FRAC * b, wv.get("maker_cap_day") or NIGHT_FRAC * b), PLAN["weather_maker"]["roi"],
                     f"executor NIGHT_FRAC {NIGHT_FRAC:.0%} of bankroll per night caps volume"))
    rows.append(("L32/L34/L33 totals vs opener", "/".join(str(gate_leads.get(k)) for k in ("L32", "L34", "L33")), 0.0, 0.0, "not qualified (no sealed pass); Kalshi taker "
                 "planning ROI ~0%"))
    rows.append(("L39 monthly rain", gate_leads.get("L39"), 0.0, 0.01, "not qualified; capacity ~$25/day-equivalent, "
                 "last 7 days of month only"))
    qualified = rows[0][2]
    exp_profit = qualified * roi_now
    lo_need, hi_need = TARGET_PROFIT_DAY
    L = ["| edge | gate | volume $/day | planning ROI | expected $/day | basis |", "|---|---|---:|---:|---:|---|"]
    for name, v, vol, roi, basis in rows:
        L.append(f"| {name} | {v} | {vol:,.0f} | {roi:.1%} | {vol * roi:,.2f} | {basis} |")
    L.append("")
    L.append(f"**Qualified volume today: ${qualified:,.0f}/day -> expected ${exp_profit:,.2f}/day vs the doubling line "
             f"${lo_need:.2f}-{hi_need:.2f}/day ({exp_profit / lo_need:.0%}-{exp_profit / hi_need:.0%} of target).** "
             f"At the current planning ROI {roi_now:.1%}, doubling needs ${lo_need / roi_now:,.0f}-"
             f"{hi_need / roi_now:,.0f}/day of qualified volume (at a 2% edge: ${lo_need / TARGET_EDGE:,.0f}-"
             f"{hi_need / TARGET_EDGE:,.0f}/day).")
    return L, {"qualified": qualified, "exp_profit": exp_profit, "roi": roi_now, "wv": wv}


# ------------------------------------------------------------------------------------------------- health
def logger_health() -> list[str]:
    now = datetime.now().timestamp()
    out = []
    for name in ("run", "nba_open_totals", "open_lines", "kalshi_wx_monthly", "nba_midgame", "nba_bigleads",
                 "gap_logger", "live_signals"):
        log, err = DEPLOY / f"{name}.log", DEPLOY / f"{name}.err"
        hb = DEPLOY / f"{name}.heartbeat"
        ages = [(p.name, (now - p.stat().st_mtime) / 3600) for p in (log, hb) if p.exists()]
        last = min(ages, key=lambda a: a[1]) if ages else None
        err_tail = ""
        if err.exists() and err.stat().st_size and (now - err.stat().st_mtime) < 86400:
            lines = [ln for ln in err.read_text(errors="replace").splitlines()
                     if ln.strip() and "Warning" not in ln and "warnings.warn" not in ln]
            if lines:
                err_tail = redact(lines[-1][:160])
        out.append(f"| {name} | {'-' if not last else f'{last[1]:.1f} h ago ({last[0]})'} | {err_tail or '-'} |")
    return ["| logger | last write | stderr in last 24h (last line) |", "|---|---|---|"] + out


def code(text: str, max_lines: int = 30) -> list[str]:
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    return ["```"] + lines + ["```"]


# --------------------------------------------------------------------------------- added sections (2026-10-28)
def _jload(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return default


def _f(v, fm: str = "+.1f") -> str:
    return "-" if v is None else format(v, fm)


def q004_lines() -> list[str]:
    """Q004 pick_tight paper stats per open-line logger (deployed tight files)."""
    L = ["| logger | events tracked | first tight (uncensored) | pick_tight decided | picks | settled | W-L | "
         "ROI @ask+fee | mean CLV |", "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    pool = 0
    for name, f in (("NBA totals", "open_totals_tight.json"), ("NHL", "open_lines_nhl_tight.json"),
                    ("NFL", "open_lines_nfl_tight.json")):
        T = _jload(DEPLOY / "data" / f, None)
        if T is None:
            L.append(f"| {name} | (missing {f}) | | | | | | | |")
            continue
        mk = list(T.get("markets", {}).values())
        ft = sum(1 for m in mk if m.get("first_tight") and not m["first_tight"].get("censored"))
        picks = list(T.get("picks", {}).values())
        made = [p for p in picks if p.get("pick")]
        st = [p for p in picks if "result" in p]
        pool += len(st)
        w = sum(p["result"] == "win" for p in st)
        roi = [p["roi"] for p in st if p.get("roi") is not None]
        clv = [p["clv"] for p in st if p.get("clv") is not None]
        L.append(f"| {name} | {len(mk)} | {ft} | {len(picks)} | {len(made)} | {len(st)} | {w}-{len(st) - w} | "
                 f"{_f(sum(roi) / len(roi) if roi else None, '+.1%')} | {_f(sum(clv) / len(clv) if clv else None, '+.3f')} |")
    L += ["", f"Pooled settled pick_tight: {pool} of the 100 the Q004 judge needs (mean CLV > 0 AND ROI @ask+fee > 0 "
          "-> write a sealed ask)."]
    return L


def q003_lines() -> list[str]:
    q = DEPLOY / "QUESTIONS.md"
    q = q if q.exists() else ROOT / "QUESTIONS.md"
    txt = q.read_text() if q.exists() else ""
    m = re.search(r"^## Q003 .*?(?=^## Q\d|\Z)", txt, re.S | re.M)
    if not m:
        return ["- Q003: QUESTIONS.md not found (re-run the installer)."]
    blk = m.group(0)
    fm = re.search(r"^finding:\s*(.*?)(?=^\w+:|\Z)", blk, re.S | re.M)
    finding = " ".join(fm.group(1).split()) if fm else "-"
    lock = ROOT / "lab" / "lockbox.jsonl"
    sealed = [r for r in read_jsonl(lock) if "q003" in json.dumps(r).lower()] if lock.exists() else []
    status = (f"SEALED RESULT in lockbox ({len(sealed)} rows)" if sealed
              else "SEALED ASK PENDING -- awaiting PM yes; nothing scored on the sealed window yet")
    return [f"- **{blk.splitlines()[0].replace('## ', '')}**", f"- status: {status}",
            f"- dev finding: {finding[:700]}"]


def picks_today_lines(day: str) -> list[str]:
    """Today's NBA totals (main + bias-corrected) and NHL logger decisions from the deployed state files."""
    L = ["**NBA totals (nba_open_totals.py; gap BC = model - recent level bias vs Pinnacle close - open)**", "",
         "| game | open | model | gap | bias (n) | gap BC | pick | pick BC |", "|---|---:|---:|---:|---:|---:|---|---|"]
    S = _jload(DEPLOY / "data" / "open_totals_state.json", {})
    n = 0
    for k, g in sorted(S.items()):
        if not k.startswith(day):
            continue
        n += 1
        d = g.get("decision") or {}
        if not d:
            L.append(f"| {k.split('|')[1]} | {(g.get('pin_open') or {}).get('line', '-')} | (undecided) | | | | | |")
            continue
        b, nb, mo, op = d.get("bias"), d.get("n_bias"), d.get("model"), d.get("open_line")
        gbc = None if None in (b, mo, op) else mo - b - op
        L.append(f"| {k.split('|')[1]} | {_f(op, '.1f')} | {_f(mo, '.1f')} | {_f(d.get('gap'))} | {_f(b, '+.2f')} ({nb}) "
                 f"| {_f(gbc)} | {d.get('pick') or '-'} | {d.get('pick_bc') or '-'} |")
    if not n:
        L.append("| (no NBA games today in the logger) | | | | | | | |")
    L += ["", "BC picks need >= 30 earlier decided games for the bias estimate; below that pick_bc stays empty and "
          "gap BC is informational.", "",
          "**NHL (open_lines_logger.py nhl)**", "", "| game | market | open | model | gap | pick |", "|---|---|---|---:|---:|---|"]
    H = (_jload(DEPLOY / "data" / "open_lines_nhl_state.json", {}) or {}).get("games", {})
    m = 0
    for k, g in sorted(H.items()):
        if not k.startswith(day):
            continue
        for mkt, d in (g.get("decision") or {}).items():
            m += 1
            op = d.get("open_line") if d.get("open_line") is not None else d.get("open_novig")
            L.append(f"| {k.split('|')[1]} | {mkt} | {op} | {d.get('model')} | {d.get('gap')} | {d.get('pick') or '-'} |")
    if not m:
        L.append("| (no NHL decisions today) | | | | | |")
    return L


# -------------------------------------------------------------------------------------------------- brief
def build(timeout: int = 120) -> tuple[str, str]:
    now = datetime.now().astimezone()
    rc, exec_txt = run(["kalshi_exec.py", "status"], timeout)
    ex = parse_exec(exec_txt)
    reports = {
        "NBA open totals (L32 forward)": run(["nba_open_totals.py", "report"], timeout),
        "NHL open lines (L34 forward)": run(["open_lines_logger.py", "report", "--league", "nhl"], timeout),
        "NFL open lines (L33 forward)": run(["open_lines_logger.py", "report", "--league", "nfl"], timeout),
        "Monthly rain/snow (L39 forward)": run(["kalshi_wx_monthly_logger.py", "report"], timeout),
        "NBA mid-game (L37 recorder)": run(["nba_midgame_scan.py", "report"], timeout),
        "E5 ForecastEx FX1-fwd paper (forecastex_paper.py)": run(["forecastex_paper.py", "report"], timeout),
    }
    mrc, manual = run(["manual_bets.py", "orders"], timeout)
    manual = "\n".join(re.sub(r"\s[0-9a-f]{8}-[0-9a-f-]{27,}$", "", ln) for ln in manual.splitlines())

    ledger, lockbox = ROOT / "lab" / "ledger.jsonl", ROOT / "lab" / "lockbox.jsonl"
    gate = FS.Gate.load(ledger, lockbox, DEPLOY / "data")
    leads = FS.lead_rows(gate)
    gv = {e["id"]: e["verdict"] for e in leads}
    asks = [e for e in leads if e["verdict"] == "SEALED-ASK"]
    ledger_age = (now.timestamp() - ledger.stat().st_mtime) / 86400 if ledger.exists() else None
    vol_lines, vol = volume_table(ex, gv)

    live = ex.get("live") or {}
    err7 = [e for e in ex["errors"] if _recent(e, 7)]
    head = []
    if live:
        head.append(f"live weather {int(live['wins'])}/{int(live['settled'])} settled wins, P&L ${live['pnl']:+.2f} "
                    f"(7d ${live['net7d']:+.2f}); open ${live['open']:.2f} of ${live['open_cap']:.2f} cap")
    else:
        head.append("executor status unavailable")
    head.append(f"qualified volume ${vol['qualified']:,.0f}/day -> ~${vol['exp_profit']:.2f}/day expected vs "
                f"${TARGET_PROFIT_DAY[0]:.2f}-{TARGET_PROFIT_DAY[1]:.2f} needed")
    head.append(f"{len(asks)} SEALED-ASK" + (f" ({', '.join(a['id'] for a in asks)})" if asks else ""))
    head.append(f"{len(err7)} executor error(s) in 7d")
    headline = "; ".join(head)

    L = [f"# Fund daily brief -- {now:%Y-%m-%d %H:%M %Z}", "",
         f"**Headline:** {headline}.", "",
         f"Read-only snapshot. Source copy: `{ROOT}`; reporters run in `{DEPLOY}`. Account balances are redacted by "
         "design. Goal: double a $1-2k unlevered bankroll in 12 months, proven live.", ""]

    L += ["## 1. Executor (live money)", ""]
    if live:
        L += [f"- mode {ex['mode']}, style **{ex['style']}**, KILL {ex['kill']}",
              f"- record: {int(live['decisions'])} decisions, {int(live['with_contracts'])} with contracts, "
              f"{int(live['settled'])} settled, {int(live['wins'])} wins; P&L ${live['pnl']:+.2f} "
              f"(24h ${live['net24']:+.2f}, 7d ${live['net7d']:+.2f})",
              f"- today spent ${live['today']:.2f} of ${live['day_cap']:.2f} day cap; open exposure "
              f"${live['open']:.2f} of ${live['open_cap']:.2f} cap ({live['open_count']} open weather positions)"]
    L += [f"- recent errors: {len(ex['errors'])} listed, {len(err7)} in the last 7 days"] + \
         [f"  - `{e[:200]}`" for e in ex["errors"]]
    L += ["", "<details><summary>raw status (redacted)</summary>", ""] + code(exec_txt, 60) + ["", "</details>", ""]

    L += ["## 2. Statistician gate (registered leads)", "",
          f"{gate.n} scored trials, ledger Bonferroni bar p < {gate.bonf:.2e}; ledger file "
          f"{'age %.1f days' % ledger_age if ledger_age is not None else 'MISSING'}"
          + (" (deployed snapshot: refresh by re-running the installer)" if ROOT == DEPLOY else ""), "",
          "| id | status | verdict | dev p | BH q | bar | halves | forward n | forward mean | why |",
          "|---|---|---|---:|---:|---|---|---:|---:|---|"]
    for e in leads:
        fw = e["forward"] or {}
        L.append(f"| {e['id']} {e['name']} | {e['status']} | **{e['verdict']}** | {FS._f(e['p'], '.1e')} | "
                 f"{FS._f(e['q'], '.3f')} | {e['tier']} | {({True: 'ok', False: 'flip', None: '-'})[e['sign_ok']]} | "
                 f"{FS._f(fw.get('n'), 'd')} | {FS._f(fw.get('mean'), '+.4f')} | {'; '.join(e['reasons'])} |")
    flags = gate.missing_halves()
    L += ["", f"Gate flags: {len(flags)} trial(s) with q < .10 lack a both-halves check: "
          + ", ".join(r["strategy"] for r in flags[:8]) + ("" if len(flags) <= 8 else ", ..."), ""]

    L += ["## 3. Qualified daily volume vs the doubling target", ""] + vol_lines + [""]

    L += ["## 4. Forward paper loggers", ""]
    for title, (rc_, txt) in reports.items():
        L += [f"### {title}" + ("" if rc_ == 0 else f" (exit {rc_})"), ""] + code(txt, 18) + [""]

    L += ["## 4b. Q004 pick_tight (first tight Kalshi quote, paper)", ""] + q004_lines() + [""]
    L += ["## 4c. Q003 Kalshi Mentions maker NO", ""] + q003_lines() + [""]
    L += [f"## 4d. Today's NBA / NHL logger picks ({now:%Y-%m-%d})", ""] + picks_today_lines(f"{now:%Y-%m-%d}") + [""]

    L += ["## 5. Manual bets (user-authorized, scripts/manual_bets.py orders)", ""] + code(manual, 25) + [""]

    L += ["## 6. Logger health", ""] + logger_health() + [""]

    L += ["## 7. Asks for the PM", ""]
    pm = []
    if asks:
        pm += [f"- SEALED-ASK {a['id']} {a['name']}: dev {a['tier']} bar + forward evidence. Say \"yes {a['id']}\"."
               for a in asks]
    if gv.get("E4") == "PASSED" and not str(ex.get("style") or "").lower().startswith("maker"):
        pm.append("- E4 maker passed sealed but the executor runs **taker**: set KALSHI_EXEC_STYLE=maker (or KALSHI_AB=1 "
                  "for A/B) in ~/.claude/quant-kalshi/.env to roughly double per-$ edge (QUESTIONS Q001).")
    if vol["exp_profit"] < TARGET_PROFIT_DAY[0]:
        pm.append(f"- Volume gap: expected ${vol['exp_profit']:.2f}/day vs ${TARGET_PROFIT_DAY[0]:.2f}/day minimum. "
                  "No current edge closes it at today's caps; the nightly cycle works QUESTIONS.md Q001-Q006.")
    L += pm or ["- none"]
    return "\n".join(L) + "\n", headline


def _recent(line: str, days: int) -> bool:
    m = re.search(r"(\d{4}-\d{2}-\d{2}T[\d:]+(?:\+00:00)?)", line)
    if not m:
        return False
    try:
        t = datetime.fromisoformat(m.group(1))
    except ValueError:
        return False
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - t < timedelta(days=days)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--stdout", action="store_true")
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args(argv)
    text, headline = build(a.timeout)
    if a.stdout:
        print(text)
        return 0
    out = ROOT / "notes" / "briefs" / f"brief_{a.date}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}\nheadline: {headline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
