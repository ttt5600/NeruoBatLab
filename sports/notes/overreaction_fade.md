# Overreaction fade on Polymarket (user hypothesis 2026-11-06) -- DEAD

User: "things swing so wildly for things that don't really affect the game or election". Test: do large short-window
moves that look immaterial revert, net of what it costs to fade them? Code `scripts/pm_jumps.py`; outputs
`data/pm_jumps/` (gitignored: describe_dev.csv, maker_dev.csv, maker_devlow.csv, jumps_*.parquet, tape/).

## Panel (dev only; vault `pm_jumps` split fixed before any tape/outcome read)
- 92 sampled days 2025-05..2026-04 (smallmarket seed), binary markets, lifetime volume >= $10k, up to 6 markets/event:
  5,777 markets (politics_world 2,011, sports 1,426 pre-game only, culture 966, finance 777, mentions 540, other 57).
- Public taker tape, data-api /v2/trades, [S-14d, S]: 3.64M prints (0.05% of markets truncated at 40 pages).
  API trap: the `cursor` only pages when `condition` is ALSO passed; cursor alone returns [] after page 1.
  `scripts/pm_maker_no.py:trades_after` has that bug (other agent's file, not edited) -> its tape stops at 1000 prints.
- Jumps: |YES move| >= 5c within 15 min (W15) -> 14,004; >= 10c within 60 min (W60) -> 7,168; one per market per 6 h.
  Decision = jump print + 60 s. Placebo = 4 random prints/market, random side (15,531).

## Descriptive (event-clustered bootstrap, 95%)
- Marked price does revert: +1.7c (1h), +1.9c (24h) W15; +2.2c / +3.4c W60 -- about the bid/ask bounce.
- TAKER fade (buy the fade side at the first print after the decision that paid that ask, <= 15 min; hold to res):
  W15 -0.8% [-6.9,+6.0] n 4,395; W60 -3.9% [-13.4,+7.7] n 2,695. Placebo taker -11.8% [-15.7,-7.7].
  "Immaterial" thin & >= 48h out: W15 -1.3% [-9.4,+7.6]; W60 -13.9% [-25.9,-1.9].
  "Material" heavy $ >= 10k: W60 -35.9% [-56.5,-17.3] (fading real news loses, as expected).
  Only 31-38% of jumps had a fade-side print within 15 min; jumps WITHOUT one drift more (3.6c/5.8c to res vs 1.0/0.8c).
- MAKER fade (rest at the jump print level, fill only by a print strictly through, no cancel; hold to res), W15:
  up-jumps (sell YES) -3.6c/contract [-6.0,-1.2]; down-jumps (buy YES) +2.5c [0.0,+4.9]; placebo maker -1.2c.

## Trial (family overreaction_fade, bar p < .05, 1 trial)
`ovr_t1_down_maker-492cf435`: down-jump W15 maker YES bid, politics/finance/sports/other (sealed-eligible cats).
- DEV edgelab verbatim: verdict needs_audit, n 3780, net_roi 0.2655, ci [0.0706, 0.4984], p 0.0022, half1 0.2034,
  half2 0.3276, concentration share 0.149, clv mean 0.0349, sharpe_ann 2.87, reasons [too_good_guard].
- AUDIT: profit sits in <= 10c YES (ROI +64..+104%). The frame is sampled on EX-POST lifetime volume. The complement it
  excluded (politics+finance, lifetime $1k-10k, 1,800 markets): the same down-fade is -2.75c [-6.5,+1.0],
  ROI -10.0% [-36.9,+21.2] (n 2,006 fills); the up-fade flips to +4.5c [-0.2,+9.5]. A drop followed by a comeback
  attracts volume and lands in the >= $10k frame, so the "edge" is a selection effect. RETIRED (dead); sealed
  2026-05..10 never pulled or unsealed.

## Lesson
Never sample a price-path study on lifetime volume. It is measured after the outcome and correlates with the path.
Any rerun must use an ex-ante universe (all markets, liquidity = tape $ before the decision).
Not done: Kalshi panel, cross-venue mirror proxy.
