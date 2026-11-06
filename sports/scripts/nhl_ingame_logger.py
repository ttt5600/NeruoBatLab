"""NHL in-game FORWARD paper logger (no orders, no launchd): Poisson + empty-net model vs live KXNHLGAME quotes.

    python scripts/nhl_ingame_logger.py tick          # one pass: log every live / pre-game NHL game
    python scripts/nhl_ingame_logger.py loop [--sec 60]

Why forward-only: Kalshi NHL markets start 2025-04; NHL 2025-26 is SEALED by the frozen MANIFEST
(notes/nhl_simple_model.md), so the only non-sealed history is 83 games (Apr-Jun 2025) -- too thin for a trial.
Model = scripts/kalshi_ingame_lowscore.py nhl_probs (2023-24 pbp prior: base goal rate + last-3-min empty-net
multipliers by margin; OT win prob = pregame strength share). Pregame anchor = the home market's last two-sided mid
logged before puck drop (state file). Writes data/live/nhl_ingame_paper.jsonl: one row per game per tick with state,
model p(home), home-market bid/ask, and the paper signal (model - price - taker fee >= 0.05 on either side).
A trial on this log must be registered in edgelab BEFORE its first evaluation (forward split only).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from sports.http import fetch
from kalshi_ingame_lowscore import PRIOR_NHL, nhl_probs

API = "https://api.elections.kalshi.com/trade-api/v2"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
OUT = ROOT / "data" / "live" / "nhl_ingame_paper.jsonl"
STATE = ROOT / "data" / "live" / "nhl_ingame_paper_state.json"
ALIAS = {"TB": "TBL", "NJ": "NJD", "LA": "LAK", "SJ": "SJS", "UTAH": "UTA"}
EDGE, TAKER = 0.05, 0.07


def codes(ab):
    return {ab, ALIAS.get(ab, ab)} | {k for k, v in ALIAS.items() if v == ab}


def tick() -> int:
    prior = json.loads(PRIOR_NHL.read_text())
    tot = 2 * prior["base_per_team_min"]
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    sb = fetch(ESPN, cache=False)
    mk = fetch(f"{API}/markets", {"series_ticker": "KXNHLGAME", "status": "open", "limit": 1000}, cache=False)
    mk = mk.get("markets", [])
    now = datetime.now(timezone.utc)
    rows = []
    for e in sb.get("events", []):
        c = e["competitions"][0]
        s = c["status"]
        tm = {x["homeAway"]: x for x in c["competitors"]}
        h, a = tm["home"]["team"]["abbreviation"], tm["away"]["team"]["abbreviation"]
        hm = [m for m in mk if m["ticker"].split("-")[-1] in codes(h) and any(
            m["event_ticker"].endswith(x + y) for x in codes(a) for y in codes(h))]
        if len(hm) != 1:
            continue
        m = hm[0]
        bid, ask = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
        key = e["id"]
        state = s["type"]["state"]          # pre / in / post
        if state == "pre":
            if 0 < bid < ask < 1 and ask - bid <= 0.05:
                st[key] = {"pm": (bid + ask) / 2, "as_of": now.isoformat()}
            continue
        if state != "in" or key not in st:
            continue
        per = int(s.get("period", 1))
        mm, ss = (s.get("displayClock", "20:00").split(":") + ["0"])[:2]
        left = (3 - per) * 20 + int(mm) + float(ss) / 60 if per <= 3 else 0.0
        if per > 3:
            continue
        pm = st[key]["pm"]
        lo_, hi_ = -3.0, 3.0
        for _ in range(40):
            mid_ = (lo_ + hi_) / 2
            r = 1 / (1 + np.exp(-mid_))
            if nhl_probs(tot * r, tot * (1 - r), 0, 60.0, prior, r) > pm:
                hi_ = mid_
            else:
                lo_ = mid_
        r = 1 / (1 + np.exp(-(lo_ + hi_) / 2))
        margin = int(tm["home"].get("score") or 0) - int(tm["away"].get("score") or 0)
        p = nhl_probs(tot * r, tot * (1 - r), margin, left, prior, r)
        sig = None
        if 0 < bid < ask < 1:
            ey = p - ask - TAKER * ask * (1 - ask)
            en = (1 - p) - (1 - bid) - TAKER * (1 - bid) * bid
            if max(ey, en) >= EDGE:
                sig = {"side": "yes" if ey >= en else "no", "price": ask if ey >= en else 1 - bid, "edge": max(ey, en)}
        rows.append({"ts": now.isoformat(), "eid": key, "ticker": m["ticker"], "period": per, "minutes_left": left,
                     "margin": margin, "p_pre": pm, "p_model": p, "bid": bid, "ask": ask, "signal": sig})
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st))
    with OUT.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{now:%H:%M:%S} logged {len(rows)} live games, {sum(r['signal'] is not None for r in rows)} paper signals")
    return 0


def main() -> int:
    a = sys.argv[1:]
    if a[:1] == ["loop"]:
        sec = int(a[a.index("--sec") + 1]) if "--sec" in a else 60
        while True:
            try:
                tick()
            except Exception as ex:  # noqa: BLE001
                print("tick error", ex)
            time.sleep(sec)
    return tick()


if __name__ == "__main__":
    sys.exit(main())
