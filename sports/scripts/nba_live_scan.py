"""Live NBA scan: each side's Kalshi ask vs the historical win rate in the same (period, clock, margin) state.

    ~/.claude/quant-kalshi/.venv/bin/python scripts/nba_live_scan.py          # print live games
    ~/.claude/quant-kalshi/.venv/bin/python scripts/nba_live_scan.py --log    # also append rows to data/nba_live_scan.jsonl

Built 2026-10-23 from the user's in-game checks (DET up 15 after Q1 at 80c = fair; DAL up 31 mid-Q3 at 96c = cheap,
history 99.4%). DESCRIPTIVE / PAPER ONLY: no orders. A spot that looks cheap here is a judgment call until a frozen rule
(e.g. "leader up >= 25 in Q3, ask <= 0.96") has a forward paper record -- see notes/HANDOFF_2026-10-25.md.

History: data/nba_big_lead_plays.parquet + _meta (ESPN pbp 2015-16..2025-26, regular + post season; built by
scripts/nba_big_lead.py). State match: same period (OT -> Q4), first play within 90 s at/after the live clock, margin
within +-1 (exact if |margin| < 10). Break-even = ask + Kalshi taker fee 0.07*p*(1-p) (unrounded; prod showed ~exact fees).
Caveat: unconditional on team strength; the Kalshi price includes it, history does not (see nba_big_lead.md).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / ".claude" / "quant-kalshi"))
from sports import live  # noqa: E402  (deployed copy: Kalshi market pull)

DATA = ROOT / "data"
LOG = DATA / "nba_live_scan.jsonl"
# ESPN abbreviation -> Kalshi team code where they differ
ESPN2K = {"SA": "SAS", "GS": "GSW", "NY": "NYK", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHX": "PHX", "BKN": "BKN"}


def history():
    p = pd.read_parquet(DATA / "nba_big_lead_plays.parquet", columns=["eid", "period", "clock_s", "hs", "as"])
    m = pd.read_parquet(DATA / "nba_big_lead_meta.parquet")
    return p, m[m.game_type != "preseason"][["eid", "home_won"]]


def hist_home_win(p, m, period, left, mar):
    q = p[(p.period == min(period, 4)) & (p.clock_s <= left) & (p.clock_s >= left - 90)]
    q = q.sort_values(["eid", "clock_s"], ascending=[True, False]).groupby("eid").head(1).merge(m, on="eid")
    x = q[((q.hs - q["as"]) - mar).abs() <= (1 if abs(mar) >= 10 else 0)]
    return (float(x.home_won.mean()) if len(x) else np.nan), len(x)


def main() -> int:
    p, m = history()
    j = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard", timeout=20).json()
    k = live.kalshi("nba")
    rows = []
    for e in j["events"]:
        c = e["competitions"][0]
        st = c["status"]
        if st["type"]["state"] != "in":
            continue
        per, clk = st.get("period"), st.get("displayClock", "0:00")
        mm, ss = clk.split(":") if ":" in clk else ("0", clk)
        left = int(float(mm) * 60 + float(ss))
        t = {x["homeAway"]: (ESPN2K.get(x["team"]["abbreviation"], x["team"]["abbreviation"]), int(x.get("score") or 0))
             for x in c["competitors"]}
        mar = t["home"][1] - t["away"][1]
        ph, n = hist_home_win(p, m, per, left, mar)
        ev = k[k.event.str.contains(t["away"][0]) & k.event.str.contains(t["home"][0])]
        for side in ("home", "away"):
            team = t[side][0]
            r = ev[ev.ticker.str.endswith("-" + team)]
            ask = float(r.yes_ask.iloc[0]) if len(r) else np.nan
            hist = ph if side == "home" else 1 - ph
            be = ask + 0.07 * ask * (1 - ask)
            rows.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "game": e["shortName"],
                         "state": f"Q{per} {clk}", "margin_home": mar, "team": team, "ask": ask,
                         "breakeven": round(be, 4) if be == be else None, "hist_win": round(hist, 4) if hist == hist else None,
                         "n_hist": n, "edge_pts": round((hist - be) * 100, 1) if (hist == hist and be == be) else None})
    if not rows:
        print("no live NBA games right now")
        return 0
    out = pd.DataFrame(rows)
    print(out.drop(columns="ts").to_string(index=False))
    flag = out[(out.edge_pts >= 2) & (out.n_hist >= 100)]
    print(f"\nflags (edge >= 2 pts, n >= 100): {len(flag)}" + ("" if flag.empty else "\n" + flag[["game", "team", "ask", "hist_win", "edge_pts", "n_hist"]].to_string(index=False)))
    if "--log" in sys.argv:
        with LOG.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
