"""L32b FORWARD PAPER logger: de-biased NBA opening totals. Paper only -- never places orders.

    ../.venv_quant/bin/python scripts/l32b_logger.py tick      # one pass (every 10 min on game days)
    ../.venv_quant/bin/python scripts/l32b_logger.py report    # W-L, ROI (no-vig open / -110), CLV, over/under mix

Pre-registration: notes/l32b_debiased_totals.md (dev PASS 2026-11-01: 967 bets, no-vig +8.1% [+1.9, +14.4], p .006).
Inputs are READ-ONLY from the frozen L32 logger's state (~/.claude/quant-kalshi/data/open_totals_state.json):
  OPEN  = L32's first Pinnacle main total for the game (pin_open line + fair_over),
  MODEL = L32's decision-tick model (decision.model = nba_simple_live.predict(date)["model_total"]),
  CLOSE = L32's last pre-tip Pinnacle total (pin_last).
  c(g)  = mean(MODEL - OPEN) over the last 100 L32-decided games (with model + open) on ET dates strictly before g's;
          >= 30 required, else no decision.
  PICK  = OVER if MODEL - c - OPEN >= 3.0, UNDER if <= -3.0; one per game, flat 1u, graded vs OPEN (push on integer).
A decision made at a tick >= 30 min before tip is type "live" (the only kind judged). Games first decided later than
that (e.g. the first run's backlog) are type "backfill": computed from the same frozen pre-tip records, reported
separately, never judged. Finals: ESPN scoreboard. JUDGE: after 200 settled live picks, CLV > 0 (one-sided p < .05);
kill if mean CLV <= 0 after 100.
Files: data/l32b_log.jsonl (decisions + settlements), data/l32b_state.json (per-game decision cache).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
L32_STATE = Path.home() / ".claude" / "quant-kalshi" / "data" / "open_totals_state.json"
LOG = ROOT / "data" / "l32b_log.jsonl"
STATE = ROOT / "data" / "l32b_state.json"
N_WIN, N_MIN, GAP = 100, 30, 3.0
MIN_LEAD = pd.Timedelta(minutes=30)
TEAMS = {"atlanta hawks": "hawks", "boston celtics": "celtics", "brooklyn nets": "nets", "charlotte hornets": "hornets",
         "chicago bulls": "bulls", "cleveland cavaliers": "cavaliers", "dallas mavericks": "mavericks",
         "denver nuggets": "nuggets", "detroit pistons": "pistons", "golden state warriors": "warriors",
         "houston rockets": "rockets", "indiana pacers": "pacers", "la clippers": "clippers",
         "los angeles clippers": "clippers", "los angeles lakers": "lakers", "memphis grizzlies": "grizzlies",
         "miami heat": "heat", "milwaukee bucks": "bucks", "minnesota timberwolves": "timberwolves",
         "new orleans pelicans": "pelicans", "new york knicks": "knicks", "oklahoma city thunder": "thunder",
         "orlando magic": "magic", "philadelphia 76ers": "76ers", "phoenix suns": "suns",
         "portland trail blazers": "trail blazers", "sacramento kings": "kings", "san antonio spurs": "spurs",
         "toronto raptors": "raptors", "utah jazz": "jazz", "washington wizards": "wizards"}


def now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def write(row: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")


def finals(date_et: str) -> dict[str, int]:
    url = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates="
           + date_et.replace("-", ""))
    with urllib.request.urlopen(url, timeout=20) as r:
        d = json.loads(r.read())
    out = {}
    for e in d.get("events", []):
        if not e["status"]["type"].get("completed"):
            continue
        c = {t["homeAway"]: t for t in e["competitions"][0]["competitors"]}
        h = TEAMS.get(c["home"]["team"]["displayName"].lower())
        a = TEAMS.get(c["away"]["team"]["displayName"].lower())
        if h and a:
            out[f"{date_et}|{a}@{h}"] = int(c["home"]["score"]) + int(c["away"]["score"])
    return out


def level(l32: dict, date_et: str) -> tuple[float | None, int]:
    xs = sorted((pd.Timestamp(g["start"]), g["decision"]["model"] - g["pin_open"]["line"]) for k, g in l32.items()
                if (g.get("decision") or {}).get("model") is not None and g.get("pin_open") and k[:10] < date_et)
    xs = xs[-N_WIN:]
    return (float(np.mean([x for _, x in xs])), len(xs)) if len(xs) >= N_MIN else (None, len(xs))


def cmd_tick() -> int:
    t = now()
    l32 = json.loads(L32_STATE.read_text())
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    new = 0
    for k, g in sorted(l32.items(), key=lambda kv: kv[1]["start"]):
        dec = g.get("decision") or {}
        if k in st or dec.get("model") is None or not g.get("pin_open"):
            continue
        c, n = level(l32, k[:10])
        if c is None:
            continue                                    # not enough history yet; retried next tick
        start = pd.Timestamp(g["start"])
        o = g["pin_open"]
        adj = dec["model"] - c - o["line"]
        side = "over" if adj >= GAP else ("under" if adj <= -GAP else None)
        row = {"type": "decision", "kind": "live" if start - t >= MIN_LEAD else "backfill", "key": k,
               "t": t.isoformat(), "start": g["start"], "model": dec["model"], "l32_decision_t": dec.get("t"),
               "c": round(c, 3), "n_c": n, "open_line": o["line"], "open_t": o["t"], "adj_gap": round(adj, 2),
               "pick": side, "nv_prob": None if side is None else
               round(o["fair_over"] if side == "over" else 1 - o["fair_over"], 4),
               "l32_pick": dec.get("pick")}
        st[k] = row
        write(row)
        new += 1
    # settle
    due = [k for k, r in st.items() if r["pick"] and "result" not in r and t - pd.Timestamp(r["start"]) > pd.Timedelta(hours=3)]
    errs = []
    for d in sorted({k[:10] for k in due}):
        try:
            fin = finals(d)
        except Exception as e:  # noqa: BLE001
            errs.append(f"espn {d} {type(e).__name__}")
            continue
        for k in [k for k in due if k[:10] == d and k in fin]:
            r = st[k]
            diff = fin[k] - r["open_line"]
            res = "push" if diff == 0 else ("win" if (diff > 0) == (r["pick"] == "over") else "loss")
            close = (l32[k].get("pin_last") or {}).get("line")
            clv = None if close is None else (close - r["open_line"]) * (1 if r["pick"] == "over" else -1)
            pnl_nv = 0.0 if res == "push" else (1 / r["nv_prob"] - 1 if res == "win" else -1.0)
            pnl_110 = 0.0 if res == "push" else (100 / 110 if res == "win" else -1.0)
            r.update(result=res, final_total=fin[k], close_line=close, clv_pts=clv, pnl_novig=round(pnl_nv, 4),
                     pnl_110=round(pnl_110, 4))
            write({"type": "settle", "key": k, "kind": r["kind"], "t": t.isoformat(), "final_total": fin[k],
                   "result": res, "close_line": close, "clv_pts": clv, "pnl_novig": r["pnl_novig"], "pnl_110": r["pnl_110"]})
    STATE.write_text(json.dumps(st, indent=1))
    if new or errs:
        print(f"{t:%F %T}Z l32b: {new} new decisions" + (f"; errors {errs}" if errs else ""))
    return 0


def cmd_report() -> int:
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    if not st:
        print("no decisions yet")
        return 0
    d = pd.DataFrame(st.values())
    print(f"decisions {len(d)} (live {(d.kind == 'live').sum()}, backfill {(d.kind == 'backfill').sum()}); "
          f"latest c {d.sort_values('start').c.iloc[-1]:+.2f} (n {d.sort_values('start').n_c.iloc[-1]})")
    for kind in ("live", "backfill"):
        p = d[(d.kind == kind) & d.pick.notna()]
        s = p[p.get("result", pd.Series(index=p.index, dtype=object)).notna()] if "result" in p else p.iloc[:0]
        line = f"[{kind}] picks {len(p)} (over {(p.pick == 'over').sum()}, under {(p.pick == 'under').sum()}); settled {len(s)}"
        if len(s):
            w, l_, pu = (s.result == "win").sum(), (s.result == "loss").sum(), (s.result == "push").sum()
            clv = s.clv_pts.dropna()
            line += (f": {w}-{l_}-{pu}  ROI no-vig {s.pnl_novig.mean():+.1%}  ROI -110 {s.pnl_110.mean():+.1%}  "
                     f"CLV {clv.mean():+.2f} pts (sd {clv.std():.2f}, n {len(clv)})")
        print(line)
    cols = [c for c in ["key", "kind", "pick", "open_line", "model", "c", "adj_gap", "l32_pick", "close_line",
                        "final_total", "result", "clv_pts"] if c in d]
    print(d[d.pick.notna()].sort_values("start")[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("tick", "report"):
        print(__doc__)
        sys.exit(1)
    sys.exit(cmd_tick() if cmd == "tick" else cmd_report())
