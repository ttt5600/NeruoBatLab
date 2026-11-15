# state_gas -- Kalshi KXAAAGASCD* state daily gas-change brackets (agent state_gas, 2026-11-15)

Lead: series watcher digest `notes/series_watch/2026-11-15.md` (50 new state AAA gas series). Prior: Q016
(`notes/q016_commodities.md`): national weekly AAA ladder priced within ~1c of a futures pass-through model.

## Verdict (2026-11-15): TOO NEW TO TEST. Lead parked; forward plan + pre-registration below.

## Census (read-only GETs; `results/state_gas/census.py` -> `results/state_gas/census_raw.json`)
- Contract (rules of KXAAAGASCDIL-26NOV16-*; series contract_url AAAGASNEW.pdf; fee_type quadratic, mult 1):
  the **one-day change** D-1 -> D of AAA's state average regular price, in cents, "first published price for each date
  at full precision, no rounding". NOT a level ladder. 5 buckets per event: LD <= -1.5, SD (-1.5, -0.5],
  MC (-0.5, 0.5), SI [0.5, 1.5), LI >= 1.5 (bucket edges read from the SI rule text "[0.5, 1.5)" and the LD rule
  "<= -1.5"; SD/MC exact inclusivity inferred, confirm before coding a scorer).
- Timing: event D opens 13:00 UTC on D-1 (08:00 ET, after AAA's D-1 print is public); closes early when AAA's D print
  appears, ~08:15-08:38 UTC on D (~03:15-03:40 ET) (close_time lists, census). So the market is a ~19 h forecast of
  one day's change with yesterday's change already known.
- History: 140 events over 6 dates. 26NOV11-13: only 5 states (CA, FL, IL, NJ, OH); 26NOV14: 25 states; 26NOV15: all
  50; 26NOV16: 50 active. Settled = 89 state-days on 5 distinct dates (1 IL event 'closed', not finalized).
  `expiration_value` is numeric (cents) on 79 state-days, a bucket label ("Large decrease") on the rest.
- Regime so far: every settled date is a falling market (winners: LD 62, SD 24, MC 2, SI 1; census script output).
  Day-to-day changes are persistent but decay (e.g. OH -5.91, -5.76, -6.67, -3.41; CA -3.89, -3.27, -1.92, -0.91).
- Volume: 26NOV14 44.2k contracts, 26NOV15 78.2k, 26NOV16 45.1k so far (census).
- Book snapshot 2026-11-15 ~16:30 UTC (`results/state_gas/book_snapshot.py` -> `book_snapshot_2026-11-15.json`):
  every state shows a market maker quoting ~700-900 contracts per level on both sides of the live buckets; total
  resting YES depth 92k, NO depth 340k contracts. Favourite bucket LD in 26 states, SD 18, MC 6; sum of YES asks
  median 1.08 (range 0.97-1.59). Far buckets (SI/LI in a falling market) sit at YES 0.00/0.02.

## What this means for the brief's angles
- (c) E4 consensus-reject on far brackets: **nothing to buy**. A far bucket with YES bid 0.00 has NO ask 1.00; with bid
  0.01 the NO costs 0.99 for at most 1c gross. The tails are already priced as impossible -- the opposite of the
  national ladder's "far rungs lacking bids" being cheap NO. Dead as an E4 analogue on today's book.
- (a)+(b) interior mispricing (LD vs SD vs MC): bucket width is 1c while daily changes run 0-7c, so interior buckets are
  genuinely uncertain and the 2-4 adjacent buckets carry the money. Edge needs a state model better than a maker who
  already sees yesterday's change. Wholesale lag (RBOB 1-3 days back, asymmetric) and day-of-week effects are the
  candidate extra information. Cannot be tested on 5 dates.
- Historical state-level DAILY AAA data: not obtainable here. `curl gasprices.aaa.com/?state=IL` returns a Cloudflare
  "Attention Required" block page (5.7 kB, /tmp/aaa_il.html). AAA publishes no archive; EIA state data is weekly and
  only ~9 states. So no pre-Kalshi backtest is possible; the fit data must be the series' own settlements.

## Forward plan (proposal; nothing installed, no launchd change)
Retroactive sources make a dedicated logger mostly unnecessary: Kalshi keeps finalized markets' `expiration_value`
(the daily change itself) and hourly candles (Q016 method), and RB=F daily closes come from Yahoo. Only order-book
depth and AAA levels are not retroactive. Proposal: (1) no new launchd job now; (2) optionally extend the existing
read-only kx_forward job with one 50-event snapshot/day at 14:00 UTC (top-of-book + depth, 250 GETs, ~90 s through
kx_throttle) so a maker variant can be judged filled-vs-unfilled later; lead/user call.

## Book logger LIVE (2026-11-15, ops_state_gas_book; notes/ops_state_gas_book.md)
launchd `com.jonathanwang.quant-sports-state-gas-books` (qs_job.sh `state-gas-books`, scripts/state_gas_books.py, read-only,
~100 throttled GETs/snapshot) snapshots every open KXAAAGASCD* book (5 buckets, full depth) twice a day: slot `am` = first run
in [14:00,18:00) UTC, slot `close` = [02:15,03:05) ET -> ~/.claude/quant-sports/data/kx_forward/state_gas_books.jsonl.gz
(pull: scripts/qs_sync.sh pull). First real snapshot 2026-11-15T16:32Z: 50/50 states, 250 markets, event 26NOV16.
30-date FIT freeze: if all 50 states settle every day from 26NOV15, the 30th settlement date is **26DEC14** (~2026-12-14
08:30 UTC); books cover 29 of those 30 FIT dates (26NOV16 on). The maker variant becomes judgeable on logged books from then.

## PRE-REGISTRATION (written 2026-11-15 before any Kalshi price was joined to any outcome)
Question: does a state lag model beat the KXAAAGASCD* book on the daily change buckets after fees?
- Data: all KXAAAGASCD* events. FIT window = first 30 distinct settlement dates on which >= 40 states settled
  (from 26NOV15 on; expected to end ~mid-Dec 2026). TEST = every later date (forward). No outcome-vs-price join on the
  FIT window except to fit the model; the TEST window is never looked at before the model + rule are frozen in edgelab.
- Model (frozen form, coefficients fit on FIT only, pooled across states with state random intercept optional):
  c_D = a + b1 c_{D-1} + b2 c_{D-2} + g_up max(dRB,0) + g_dn min(dRB,0) + w_dow(D) + e, where c = daily change (cents,
  Kalshi expiration_value; label-only days -> bucket midpoint, -2.5 / +2.5 for LD/LI), dRB = change in RB=F daily close
  over the 2 sessions ending before D-1 08:00 ET, w_dow = Sunday/Monday dummies. e ~ Student-t(4), scale = pooled FIT
  residual sd per state group (fallback pooled). Bucket p = model mass in each bucket interval.
- Rule GS1 (taker): at decision time 14:00 UTC on D-1, last hourly candle <= t; buy YES or NO on a bucket when
  p_model - ask - fee >= 0.05, ask in [0.03, 0.97], spread <= 0.05; one bet per event (largest edge); $1 cost per bet;
  fee 0.07 p(1-p) exact. Cluster = settlement date (all 50 states move together on wholesale; dates, not state-days,
  are the independent unit).
- Rule GS1-NO (E4 analogue): buy NO on buckets with p_model < 0.03 at NO ask + fee. Registered only if the FIT-window
  book shows tail YES bids >= 0.04 on >= 20% of state-days; otherwise not registered (today it would be empty).
- Maker variant: deferred until a depth logger exists (needs fills, not candles).
- Success: TEST ROI CI (date-cluster bootstrap) > 0 after fees with >= 40 dates and >= 300 bets, AND edgelab family
  bar (kalshi_data_ladders, ~1e-3). Kill: ROI CI below 0 at 40 dates, or FIT-window model log-loss not better than the
  book's normalised mid by a date-cluster CI excluding 0 (then never registered as a trial).
- edgelab: registration happens at FIT freeze (a trial cannot be registered before its model coefficients exist).
  Count GS1 (+ GS1-NO if eligible) as 1-2 trials of family kalshi_data_ladders at that point.

## Learnings
### Kalshi state AAA gas series are one-day CHANGE brackets, not level ladders
KXAAAGASCD<ST> pays on AAA's D-1 -> D change in 5 one-cent-wide buckets, opens 08:00 ET D-1 and closes ~03:30 ET D
(census_raw.json). Yesterday's change is public when it opens, so the naive lag predictor is already in the price.
### State gas far brackets are priced at YES 0.00/0.02 with maker depth, so no E4-style cheap NO exists
Book snapshot 2026-11-15: a market maker quotes ~700-900 per level in all 50 states; far buckets have YES bid 0-1c, NO
ask 0.99-1.00 (book_snapshot_2026-11-15.json). The consensus-reject edge needs stale tail bids; these are not stale.
### AAA state pages are Cloudflare-blocked to curl; use Kalshi expiration_value as the AAA daily-change record
gasprices.aaa.com returned a Cloudflare block page; Kalshi's finalized expiration_value carries the first-published
change in cents (79/89 numeric, rest labels). No free daily state archive exists, so a backtest must wait for ~30 dates.
