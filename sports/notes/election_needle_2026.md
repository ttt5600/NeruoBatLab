# Election-night NYT-needle bot (2026-11-03..06) -- what happened and what it teaches

Live, real money, outside edgelab (user-directed judgment trading). Code: scripts/nyt_needle_house.py, bet.py;
run log + handoff: ~/.claude/quant-kalshi/data/NEEDLE_STATE.md, nyt_needle_house_log.jsonl.

## Result (Kalshi fills, all election tickers)
- Net in $731 ($1,000 deposits - $269 withdrawn 11-04). Account ~$633 on 11-06 with NC-03 R (25c, pays $228) and
  near-locked favorites open -> roughly -$96 marked; settles ~$565 or ~$785 depending on NC-03.
- Fees ~$78 on the night = most of the loss came from churn, not from being wrong.
- Worst: 1-9c longshots Kelly-sized off a stale needle (PA-1, CA-22, MI-10, TX-9, WA-3) ~ -$150.
- Best: buying where NYT's count was FRESH and the market lagged (AZ-08 R 47c -> sold 94-95c, +$25; AZ-05; CO-3 Hurd
  was NYT-right too but sold early on a market-vs-NYT rule, ~$96 of value lost); count-locked favorites (TX-32, TX-2,
  VA-5, FL-13, SC-2, OH-9) all won small.

## Mechanism (the part that generalises)
The edge is an **information-latency gap**: a public source updates a quantity that determines resolution, and the
exchange reprices slower than the source. It existed only when ALL of these held:
1. **Freshness**: the source's own number was current (NYT `votes_counted == AP total_votes`). A frozen source looks
   like an edge and is the opposite: stale NYT needles lost every time (markets knew about late D ballots).
2. **The source is the resolution path** (vote count -> winner), not a model opinion about it.
3. **The market is thin/retail-paced** (Kalshi KXHOUSERACE districts, Polymarket followed Kalshi within minutes).
4. **Cross-venue check**: when Polymarket and Kalshi both disagreed with the source, the source was usually the stale
   one; when only Kalshi lagged, the edge was real.

## Lessons (also in FUND_BOOK s6)
- Gate on source FRESHNESS, not on market agreement; a frozen feed is the most dangerous input.
- Price-cap stakes (never Kelly a 1-9c contract off a model); per-race cap of value at max(mark, cost).
- Fees from re-trading the spread were bigger than the edge; hold to settlement once in.
- Bank must be measured from the exchange (cash + portfolio_value); partial marks + a withdrawal made the
  concentration rule dump six favorites at the bid.
- Ops: sub-cent price rounding killed fills; $1.00 sells are invalid; 85-series polling at 20s draws 429s; launchd/
  python lose ~/Desktop access (TCC) and the Mac sleeps with the lid closed -> run long-lived bots from outside Desktop.
- Not an edgelab strategy: judgment trading with user sizing. Any repeat (2028, specials, primaries) should be
  pre-registered as a forward paper test on the freshness-gated rule first.
