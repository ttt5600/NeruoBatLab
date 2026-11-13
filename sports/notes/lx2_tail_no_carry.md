# LX2 -- long-dated tail-NO carry on Kalshi (agent lx2_tail_no_carry, started 2026-11-12)

Question: buying NO on Kalshi contracts whose YES ask is <= 10c, 90 or 180 days before scheduled resolution, at the
real NO ask + taker fee and holding to settlement -- does it earn a positive return, and does Kalshi's interest on open
positions change the answer?
Prior: Burgi et al. (longshots lose > 60%); Maresca arXiv 2602.21091 (time value removes ~83% of the longshot return).
Related, not redone: G1/G4 (Kalshi sports FLB 1 h pre-start), notes/pm_timing.md (Polymarket fav return at mid real,
not capturable as maker), H2 (Polymarket longshot-NO maker, dead).

## Interest (checked 2026-11-12)
help.kalshi.com/en/articles/13823847-apy-on-kalshi: "Rate: 3.50% APY. This rate is variable"; interest on the "net
value of your portfolio at 12am ET", positions valued "based on last traded prices"; needs >= $250 average monthly
balance. Launch post (news.kalshi.com/p/interest-cash-open-positions) cited 4.05% from Oct 2024 -- so before ~Oct 2024
positions earned nothing on Kalshi; applying 3.5% to 2022-24 trades is a counterfactual (flagged in results).
Model: interest = cost * 0.035 * days_held / 365 (cost basis ~ mark for a 90c+ NO; simple, not compounded).
NB: because Kalshi pays the SAME rate on idle cash, interest is not an edge vs. keeping cash at Kalshi; it only
matters vs. holding T-bills elsewhere. So the honest edge test is the NO-interest ROI.

## PRE-REGISTRATION (written 2026-11-12, before any price or outcome was loaded)
Universe: Kalshi series (all categories) except frequency hourly / fifteen_min / daily / weekly; markets binary,
result yes/no, from /historical/markets (settled before the API's historical cutoff 2026-09-13).
Anchor S = expected_expiration_time (the scheduled resolution known at listing; checked KXNBA-26-WAS: expected
2026-06-30 though it closed early 2026-04-09). Markets without it are dropped (close_time can be the actual early
close = lookahead). Entry t = S - 90 d (trial A) and S - 180 d (trial B).
Eligible at t: open_time <= t, close_time > t (still trading). Quote = last DAILY candle (period 1440, /historical
candlesticks) ending in (t - 7 d, t]; need yes_bid >= 0.01 and yes_ask <= 0.10 (and yes_ask > yes_bid).
Trade: buy 1 NO at NO ask = 1 - yes_bid, fee = series fee_multiplier * 0.07 * p * (1-p) (quadratic series;
fee_type != quadratic -> 0.07 flat formula, conservative), exact (unrounded) as primary; 1-ct ceil-to-cent as
sensitivity. Hold to settlement; days_held = settlement_ts - t.
P&L per contract = (result==no) - price - fee. ROI = sum P&L / sum cost. With interest: + cost*0.035*days/365.
Annualized = sum P&L / sum(cost*days/365).
Split (fixed now): DEV = settlement_ts < 2025-10-01 UTC. SEALED = settlement_ts >= 2025-10-01 (not priced, not scored).
Categories reported separately: politics (Politics, Elections, World), economics (Economics, Financials, Companies,
Commodities, Crypto), climate (Climate and Weather), culture (Entertainment, Culture, Social, Mentions, Science and
Technology, Health, Transportation, other), sports (Sports = futures by construction, >= 90 d life).
Inference: 95% CI by bootstrap over EVENTS (event_ticker), 10,000 resamples; also series-clustered as sensitivity.
SUCCESS (per trial, all categories pooled): dev ROI incl. interest CI lower > 0 AND >= 150 distinct events AND no
single series (= event cluster, e.g. KXNBA, PRES) contributes > 30% of total positive-or-net P&L
(share = cluster P&L / total P&L when total > 0). Stricter edge bar (mine, reported alongside): ROI WITHOUT interest
CI lower > 0 -- else the "edge" is just the interest Kalshi also pays on cash.
KILL: either trial failing success -> LX2 dead for that horizon; a category-only pass is a lead (post hoc), not a pass.
Two trials (A T-90, B T-180) registered in edgelab family kalshi_longdated_tail.

## Log
- 2026-11-12: registered in edgelab before any price was loaded: lx2_a_t90_tail_no-c89029c7, lx2_b_t180_tail_no-0e30b9b9
  (family kalshi_longdated_tail, 2 trials -> bar p < .025). Dataset kalshi_longdated_tail (vault.yaml, loader in
  edgelab/loaders.py, rule edgelab/rules/longdated_tail.py). Code: scripts/lx2_tail_no_carry.py (markets/prices/dev).
  Series census: 16,260 series, 15,386 kept after dropping hourly/15min/daily/weekly.
- WRONG TURN (disk/OOM): first `markets` run cached every /historical/markets page; Kalshi "custom" multivariate
  combo series (KXMVECROSSCATEGORY, KXMVE*SINGLEGAME/MULTIGAME...) list millions of short-lived parlay markets ->
  20.7M rows, OOM kill, and ~61 GB of page cache in data/cache (deleted the KXMVE* pages again, re-fetchable).
  Fix: no new caching, keep only binary markets with open_time <= expected_expiration - 90 d at ingest, stop a series
  after 5 consecutive pages with none kept (metadata-only rule). Lesson: census the series' market counts before
  paging all of Kalshi; never disk-cache bulk list pages.
- Metadata (no prices/outcomes): 128,820 long-lived binary markets (data/lx2_tail_no_carry/markets.parquet).
  DEV eligible (open & trading at t, settled < 2025-10-01): 6,908 market-trials; A 741 events, B 528 events
  (results/lx2_tail_no_carry/describe_meta.py). A by category: economics 297 events (KXFED, KXU3, CPI ladders
  dominate), culture 136, politics 161, climate 92 (KXHURCAT), sports 55. Settlement years A: 2022 269 / 2023 1,170 /
  2024 829 / 2025 1,988 market rows.
- Quotes (results/lx2_tail_no_carry/describe_quotes.py, no outcomes): 5,222 / 6,908 market-trials have a daily candle
  in (t-7d, t]; 21% have yes_bid 0 (one-sided, excluded); quote age median 11 h, p90 82 h. Qualifying: 1,062 rows,
  A 280 events, B 150 events (B sits exactly on the 150-event floor). Median spread 2c; 360 of 1,062 have yes_bid 1c
  (NO at 99c).

## RESULT (DEV, settled < 2025-10-01) -- both trials FAIL -> LX2 dead
Source: `scripts/lx2_tail_no_carry.py dev` -> results/lx2_tail_no_carry/dev_summary.json, bets_dev.csv; edgelab
evaluate reproduces the no-interest ROI exactly (independent code path).
| trial | n / events | NO win / mean price | ROI no interest [95% event CI] | ROI + 3.5% APY | 1-ct fee rounding | annualized no-int / +int |
|---|---|---|---|---|---|---|
| A T-90 | 768 / 280 | 97.5% / 97.2c | +0.15% [-1.01, +1.18] | +0.94% [-0.22, +1.98] | -0.69% [-1.85, +0.35] | +0.65% / +4.15% |
| B T-180 | 294 / 150 | 94.9% / 96.5c | -1.90% [-4.63, +0.48] | -0.61% [-3.38, +1.78] | -2.70% [-5.42, -0.32] | -5.2% / -1.7% |
edgelab: lx2_a_t90_tail_no-c89029c7 FAIL n 768 net ROI +0.15% [-0.99, +1.18] p .39 (bar .025), halves +0.18/+0.12% ->
dead; lx2_b_t180_tail_no-0e30b9b9 FAIL n 294 -1.90% [-4.63, +0.66] p .93 -> dead.
Both fail the task's own bar (incl. interest CI lower > 0) -- "not distinguishable" from zero. A's annualized
+4.15% with interest is just Kalshi's cash rate + 0.65%: no carry beyond what idle cash earns there. Mean hold 83 d (A),
135 d (B). Concentration fine for A (top series KXINXY 10% of P&L). Series-clustered CIs ~identical.
By category (A, event CI, no interest): climate +3.0% [+2.6, +3.4] (36 events, 70 bets, 100% won); culture +0.1%,
economics -0.2%, politics -0.2%, sports -0.2% (31 events) -- all cross 0. B: climate +4.6% (9 events), others <= +0.4%.
Climate is POST HOC and is effectively 3 hurricane seasons (2022-24) of city-landfall markets (KXHURNO/NYC/ORL/MIA...);
zero losses means the event bootstrap cannot see the tail (one landfall = -97c). Lead at most; not a pass under the
prereg (category-only pass = lead). Interest caveat: Kalshi paid nothing on positions before ~Oct 2024, so for
2022-24 rows the +interest column is counterfactual.

## Proposal (no action taken)
LX2 dead as a broad strategy. If the lead wants the hurricane slice: pre-register KXHUR* NO at T-90 on the SEALED
2025 season (settled >= 2025-10-01, user's yes) and price the tail (seasons, not markets, are the unit).

## Learnings
### Kalshi long-dated tail NO earns ~0 at the ask
LX2 A (T-90, YES ask <= 10c, NO at 1 - bid + fee): +0.15% [-1.01, +1.18] over 768 bets / 280 events, dev 2022-25
(results/lx2_tail_no_carry/dev_summary.json). The longshot bias at 90-180 days does not survive the spread and fee.
### Kalshi interest only matches cash, it is not an edge
With 3.5% APY on positions, A annualizes to +4.15% vs +0.65% without: the whole gain is the rate Kalshi also pays on
idle cash. Report long-hold trades excess of the cash rate.
### Longer horizon is worse, not better
B (T-180) -1.90% [-4.63, +0.48]; NO won 94.9% at 96.5c. Longshots six months out are not overpriced on Kalshi.
### 1-contract fee rounding flips thin edges
Ceil-to-cent on 1 ct moves A from +0.15% to -0.69%: at 97c prices the 1c minimum fee is ~a third of the edge budget.
### Never page all Kalshi markets with a disk cache
KXMVE* combo series hold millions of markets; caching every /historical/markets page wrote ~61 GB and OOM-killed at
20.7M rows. Filter at ingest, stop a series after empty pages, census counts first.
### Hurricane landfall NO is a post-hoc lead, not evidence
A climate +3.0% with 100% wins over 36 events = 3 hurricane seasons; a no-loss sample hides the tail.
