"""Read-only probe of candidate series: open-market spreads (quoted rungs) + settled events in the last 90 days.
Public GETs only via kx_throttle. Output: candidate_probe.json"""
import sys, json, time, statistics as st
sys.path.insert(0, "/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports/scripts")
from kx_throttle import kget

C = sys.argv[1].split(",")
NOW = int(time.time())
out = {}
for s in C:
    d = kget("/markets", {"series_ticker": s, "status": "open", "limit": 1000}) or {}
    ms = d.get("markets") or []
    spreads, mids = [], []
    for m in ms:
        a, b = float(m.get("yes_ask_dollars") or 0), float(m.get("yes_bid_dollars") or 0)
        if 0 < b < a < 1:
            spreads.append(a - b); mids.append((a + b) / 2)
    tight = [x for x, mm in zip(spreads, mids) if 0.15 <= mm <= 0.85]
    # settled events in last 90 days
    ev, cur, n = set(), None, 0
    while n < 20:
        prm = {"series_ticker": s, "status": "settled", "min_close_ts": NOW - 90 * 86400, "limit": 1000}
        if cur: prm["cursor"] = cur
        dd = kget("/markets", prm) or {}
        for m in dd.get("markets") or []:
            ev.add(m.get("event_ticker"))
        cur = dd.get("cursor"); n += 1
        if not cur or not dd.get("markets"): break
    out[s] = dict(open_mkts=len(ms), two_sided=len(spreads),
                  med_spread=round(st.median(spreads), 3) if spreads else None,
                  med_spread_mid15_85=round(st.median(tight), 3) if tight else None,
                  settled_events_90d=len(ev), events_per_month=round(len(ev) / 3, 1))
    print(s, out[s], flush=True)
json.dump(out, open("/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports/results/strategy_synthesis/candidate_probe.json", "w"), indent=1)
