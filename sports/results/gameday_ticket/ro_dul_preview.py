"""READ-ONLY DUL preview: reuse the deployed lead_exec functions (no journal writes, no orders).
For each 2026 week>=10 division game today, call dul_games() at kickoff-60min, then price the strike nearest 50c NOW.
Also: per-game cap checks (cash reserve) using the live shard-0 balance."""
import sys, os
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.expanduser("~/.claude/quant-kalshi/scripts"))
import kalshi_exec as kx
import lead_exec as le

e = kx.env(); cfg = kx.config(e); c = kx.Kalshi(e, cfg["env"])
cash = le.shard0_cash(c)
print(f"shard-0 cash ${cash:.2f}; reserve ${le.CASH_RESERVE:.0f}; cash cap left = ${cash - le.CASH_RESERVE:.2f}")
state = le.fold(le.journal(), "live")
print(f"lead open ${le.open_exposure(state):.2f}/{le.GLOBAL_OPEN_CAP}")
kicks = ["2026-11-15T18:00:00+00:00", "2026-11-15T21:05:00+00:00", "2026-11-16T01:20:00+00:00"]
seen = set()
for k in kicks:
    now = datetime.fromisoformat(k) - timedelta(minutes=60)
    for g in le.dul_games(now):
        if g["game_id"] in seen:
            continue
        seen.add(g["game_id"])
        ev = le.nfl_event_ticker(g)
        ms = le.nfl_markets(c, ev)
        m = le.pick_strike(ms)
        if not m:
            print(g["game_id"], ev, f"n_markets={len(ms)} -> NO two-sided strike found"); continue
        no_ask = round(1 - float(m["yes_bid_dollars"]), 2)
        ok = no_ask <= STR if (STR := le.STRATS["DUL"]["max_no_ask"]) else None
        sig = {"strategy": "DUL", "ts": now.isoformat()}
        left, why = le.budget_left(sig, state, cash)
        print(f"{g['game_id']:<18} kick {g['kickoff']} {ev}  strike {m.get('floor_strike')} yes_mid {m['_mid']} "
              f"NO ask now {no_ask} (<= .56: {ok})  n_strikes {len(ms)}  budget_left ${left:.2f} binding={why}")
print("division games seen:", sorted(seen))
