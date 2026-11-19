"""Silent-skip alarm for forward paper loggers (added 2026-11-18 after Q013 logged ZERO evaluated RT picks for 15 days
because RT renamed criticsScore.score -> scorePercent and every decision row said "no score").

A logger "writes 0 picks" on a day when it logged decision rows but none of them was evaluated by the model (no p_yes).
An ELIGIBLE day is a UTC day with >= 1 decision row (a decision window opened). ALERT when the 3 most recent eligible
days all have 0 evaluated rows. Also a field-null check: a source field that used to be present goes null.

    python scripts/picks_health.py            # print table; exit 1 + notification + ALERTS file on any alert
    python scripts/picks_health.py --quiet    # print only alerts
Data root: $FUND_DATA or <root>/data (deployed copy: ~/.claude/quant-kalshi/data). READ-ONLY on logger data.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("FUND_DATA", ROOT / "data"))
ALERT_FILE = ROOT / "ALERTS_picks_health.txt"
K_DAYS = 3

# logger -> picks file (relative to DATA) and the field that marks a model-evaluated row
LOGGERS = {
    "q013": {"picks": "q013/picks.jsonl", "evaluated": "p_yes"},
    "q014": {"picks": "q014/picks.jsonl", "evaluated": "p_yes"},
}


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


def day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def zero_pick_streak(rows: list[dict], field: str, k: int = K_DAYS) -> dict:
    """Per UTC day: decision rows and evaluated rows. Alert iff the last k eligible days all have 0 evaluated."""
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("ts") is None:
            continue
        d = by[day(r["ts"])]
        d[0] += 1
        d[1] += r.get(field) is not None
    days = sorted(by)
    last = days[-k:]
    alert = len(last) == k and all(by[d][1] == 0 for d in last)
    reasons = defaultdict(int)
    for r in rows:
        if r.get("ts") is not None and day(r["ts"]) in last and r.get(field) is None:
            reasons[r.get("reason") or "?"] += 1
    return {"eligible_days": len(days), "last": {d: {"rows": by[d][0], "evaluated": by[d][1]} for d in last},
            "alert": alert, "skip_reasons": dict(reasons)}


def q013_field_null(data: Path, hours: float = 24, n_min: int = 10) -> dict:
    """RT rows with n >= n_min but score None in the last `hours` (the exact 2026-11-03 failure mode)."""
    p = data / "q013" / "scores.jsonl"
    if not p.exists():
        return {"rows": 0, "null": 0, "alert": False}
    cut = time.time() - hours * 3600
    rows = []
    with p.open("rb") as f:   # file is large: read only the tail
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 2_000_000))
        for ln in f.read().decode(errors="replace").splitlines()[1:]:
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("ts", 0) >= cut and r.get("source") == "rt" and (r.get("n") or 0) >= n_min:
                rows.append(r)
    latest = {}
    for r in rows:   # judge each film on its LATEST fetch, so the alarm clears on the first good tick after a fix
        if r["event"] not in latest or r["ts"] >= latest[r["event"]]["ts"]:
            latest[r["event"]] = r
    nul = sum(1 for r in latest.values() if r.get("score") is None)
    return {"rows": len(latest), "null": nul, "alert": nul > 0}


def check(data: Path = DATA) -> tuple[list[str], list[str]]:
    lines, alerts = ["| logger | eligible days | last 3 eligible days (evaluated/rows) | status |", "|---|---|---|---|"], []
    for name, cfg in LOGGERS.items():
        rows = read_jsonl(data / cfg["picks"])
        s = zero_pick_streak(rows, cfg["evaluated"])
        last = ", ".join(f"{d} {v['evaluated']}/{v['rows']}" for d, v in s["last"].items()) or "-"
        st = "ALERT" if s["alert"] else "ok"
        lines.append(f"| {name} | {s['eligible_days']} | {last} | {st} |")
        if s["alert"]:
            alerts.append(f"{name}: 0 evaluated picks on the last {K_DAYS} eligible days ({last}); "
                          f"skip reasons {s['skip_reasons']}")
    fn = q013_field_null(data)
    lines.append(f"| q013 RT score field (latest fetch per film, 24 h, n>=10) | - | {fn['null']}/{fn['rows']} films with score=None | "
                 f"{'ALERT' if fn['alert'] else 'ok'} |")
    if fn["alert"]:
        alerts.append(f"q013: {fn['null']}/{fn['rows']} films' latest RT fetch has n>=10 but score=None (page JSON changed?)")
    return lines, alerts


def notify(alerts: list[str]) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with ALERT_FILE.open("a") as f:
        for a in alerts:
            f.write(f"{stamp} {a}\n")
    try:
        msg = (alerts[0][:180] + (f" (+{len(alerts) - 1} more)" if len(alerts) > 1 else "")).replace('"', "'")
        subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "Paper logger health ALERT"'],
                       timeout=10, check=False, capture_output=True)
    except Exception:  # noqa: BLE001
        pass


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    lines, alerts = check()
    if "--quiet" not in argv:
        print("\n".join(lines))
    for a in alerts:
        print("ALERT " + a)
    if alerts and "--no-notify" not in argv:
        notify(alerts)
    return 1 if alerts else 0


if __name__ == "__main__":
    sys.exit(main())
