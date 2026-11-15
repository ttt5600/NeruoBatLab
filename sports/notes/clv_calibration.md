# clv_calibration -- experiment B of bar_review.md: is the Kalshi close calibrated enough for net-CLV-primary? (agent, 2026-11-15)

Read-only. No trial registered (validates the instrument, not an edge). Scratch/code: results/clv_calibration/.

## Question
bar_calibration.md s4: a +0.5c close bias in our favour gives 17-71% false passes for the proposed net-CLV rule.
Is the last pre-start Kalshi mid (and last trade) an unbiased estimate of P(YES) on Kalshi sports (NBA/NFL/CFB/CBB/NHL
totals, plus spreads where cached)? And does realized ROI track net CLV on our strategy rows (slope ~1)?

## Pre-registration (written before any outcome was computed)
Statistic: bias = mean(outcome - close_mid) per market, close_mid = last candle mid with end <= game start, markets with
close spread <= 4c and close mid in [.03,.97]. CIs: cluster bootstrap by game date (ET), 2,000 reps. Reported overall,
by league x market type, price bin, and YES side (over / team-covers).
- ACCEPT (net CLV usable for league L): two-sided 95% CI of the bias for L lies inside (-0.25c, +0.25c).
- REJECT for L: CI excludes 0 (close biased) -> net CLV not usable without a bias correction.
- otherwise UNRESOLVED (CI contains 0 but wider than +-0.25c) -> net CLV not certified for L; keep ROI primary.
- Secondary (bar_review exp B): OLS slope of per-bet P&L on net CLV (close_prob - cost) on strategy rows, date-cluster
  bootstrap; success = slope CI contains 1 and excludes 0 AND intercept CI within [-1%,+1%]; kill = slope CI excludes 1
  or intercept CI below -1%.
Power, stated up front: per-market outcome SD ~ sqrt(p(1-p)) <= .5; ladder strikes in one game share one final score,
so effective n ~ games, not markets. A 95% half-width of .25c needs SE ~.0013 -> on the order of 10^5 independent games
per league. Our caches hold hundreds per league => the outcome test is EXPECTED to be UNRESOLVED everywhere
(half-widths ~1-4c). Pre-registered fallback that can reach .25c precision: bias of the Kalshi close mid vs the
de-vigged sportsbook close at the same line (no outcome noise); ACCEPT-vs-book if its 95% CI lies inside +-.25c. That
shows Kalshi close ~= book close; it does NOT prove the book close is calibrated (that rests on outside evidence).

## Data
results/kalshi_openers_xleague/{nfl,nhl,cfb,cbb}_{kmarkets,candles,games}.parquet; results/k50_strike/nba_*;
results/nba_l32b_exec/q2_*; strategy rows: results/k50_strike/rows_*.csv, q2_k1_rows_dev.csv, cbb_opener rows.

## Log
- Data actually used: xleague nfl/nhl/cfb/cbb caches (2026-10-01..11-11; hourly yes bid/ask candles, open -> start+3h)
  + NBA totals fetched read-only here (results/clv_calibration/nba_fetch.py reusing q1_fetch.py; 188 games 10-20..11-14,
  KXNBATOTAL hourly candles + ESPN/DK book close). K50/l32b NBA caches only cover T-60..T around decisions, not start.
  Close = last candle with end <= ESPN start (age median 0 min, <= 90 kept), spread <= 4c, mid in [.03,.97].
  Code: results/clv_calibration/calib.py, reg.py. Outputs calib.csv, book.csv, reg.csv, selection_bias.csv, report.txt.
- Wrong turn: first NBA fetch crashed on mixed ISO open_time formats (with/without fractional secs); normalised in wrapper.
- Last-trade close NOT measured: these caches hold bid/ask only (no trade prices). Mid only.
- Lead (mid-task, after critic_bar_calibration): keep the pre-reg; also report (a) achievable bound, (b) slope,
  (c) bias under our selection. Done below.

## Results (seed 20261115, date-cluster bootstrap 2,000; cents of probability; [95% CI])
### 1. Outcome-based bias (outcome - close mid), all markets -- pre-registered test: UNRESOLVED in every league
| cut | markets | games | dates | bias | 95% CI | half-width |
|---|---|---|---|---|---|---|
| all | 11,485 | 1,068 | 45 | +0.12 | [-1.30, +1.39] | 1.34 |
| NBA totals | 869 | 188 | 25 | -2.92 | [-7.58, +1.99] | 4.78 |
| NFL totals | 498 | 78 | 16 | +5.26 | [-0.90, +12.07] | 6.48 |
| NFL spreads | 1,140 | 78 | 16 | +0.92 | [-2.43, +4.34] | 3.38 |
| CFB totals | 1,911 | 284 | 28 | -0.73 | [-4.53, +2.50] | 3.52 |
| CFB spreads | 3,910 | 284 | 28 | -0.05 | [-1.91, +2.29] | 2.10 |
| CBB totals | 795 | 227 | 9 | +1.76 | [-6.28, +10.66] | 8.47 |
| CBB spreads | 912 | 228 | 9 | -1.32 | [-3.83, +1.73] | 2.78 |
| NHL totals | 1,450 | 290 | 42 | +1.10 | [-2.32, +4.24] | 3.28 |
(YES = over for totals, = named team covers for spreads, so these rows are the by-side cut.) Price-bin and league x bin
cuts in calib.csv: 3 of ~45 small cells "REJECT" (e.g. cfb totals 15-35c -10.2c [-18.8,-1.6], n188) and near-50c NFL
totals +5.7c [+0.6,+11.6] (78 games) -- consistent with multiple-testing noise at ~45 cuts; not acted on.
(a) Achievable bound: 95% half-width 3.3-8.5c per league-market (pooled all leagues 1.34c) vs the 0.25c target, i.e.
13-34x too wide. Scaling n by (hw/.25)^2: ~50-70k games per league (NHL ~50k, CFB totals ~56k, NBA ~69k) -> cannot be
validated from outcomes at our scale in any league (a full NBA season is ~1,230 games).
### 2. Fallback: Kalshi close mid minus de-vigged DK book close at the same line (totals, strike == book total)
pooled +0.01c [-0.07, +0.08] (n711 games, 45 dates, per-game SD 1.24c); CFB -0.01 [-0.21,+0.15] ACCEPT; NHL +0.10
[-0.03,+0.23] ACCEPT; NBA +0.25 [+0.03,+0.48] REJECT (n80); NFL -0.46 [-0.98,-0.06] REJECT (n55); CBB -0.17
[-0.36,-0.01] REJECT (n84, 9 dates). => Kalshi's close tracks the book close to ~0.1-0.5c; at the 0.25c standard it
matches for CFB/NHL totals and differs by 0.2-0.5c for NBA/NFL/CBB. This says nothing about whether EITHER is
calibrated (outcome - book close: pooled +0.87c [-1.69, +3.64], same power wall). Only ~50c strikes covered.
### 3. (b) ROI on net CLV, our strategy rows (K50 cfb/nfl/nba bets dedup by ticker+side, CBB opener Kalshi bets)
pooled n398, 32 dates: slope 1.25 [0.28, 2.31], intercept -2.4% [-18.9%, +13.6%]. Slope CI contains 1 and excludes 0,
but intercept CI is far wider than +-1% -> pre-reg success NOT met, kill NOT met -> UNRESOLVED. Mean close-implied EV
(net CLV / cost) -3.9%/$ vs mean realized ROI -7.2%: our existing rows have NEGATIVE net CLV vs Kalshi's close.
Per source (reg.csv): k50_cfb 1.62 [0.28,3.33] n217; k50_nba 0.83 [-0.43,2.18] n72; k50_nfl 1.19 [-2.2,15.1] n48;
cbb_opener 1.34 [-0.74,2.53] n61.
### 4. (c) Bias under OUR selection (outcome of our side - its close prob; selection_bias.csv)
pooled -1.27c [-8.78, +5.64] n398 -> not distinguishable from 0 or from the all-market bias; CI ~28x the 0.25c
target, so selection-induced bias is undetectable at our scale. cbb_opener +12.45c [+0.49, +26.78] (61 bets, 9 dates:
few clusters, one of 5 cuts) -> weak, not acted on.

## Verdict vs pre-registration
UNRESOLVED for NBA, NFL, CFB, CBB, NHL totals (and spreads): the close's bias cannot be bounded within +-0.25c from
outcomes at our scale (achievable +-3.3 to +-8.5c). Net CLV is NOT certified as a primary statistic in any league.
The book comparison shows Kalshi close ~= DK close (CFB/NHL within .25c; NBA/NFL/CBB off by .2-.5c), so net CLV on
Kalshi is about as trustworthy as book CLV -- usable only if the user accepts outside evidence that sharp closes are
calibrated (an assumption, not a test), and then with a per-league offset (NBA -0.25c, NFL +0.46c, CBB +0.17c) or a
book-close cross-check. Proposal for lead: keep ROI primary; net CLV as a secondary/diagnostic. Not critic-reviewed.

## Learnings
### The Kalshi close's calibration cannot be validated from outcomes at our scale
95% half-width of mean(outcome - close mid) is 3.3-8.5c per league (results/clv_calibration/calib.csv, ~80-290 games
per league); a 0.25c bound needs ~50-70k games per league. A rule that relies on an unbiased close must import that
assumption, not test it.
### Kalshi's pre-start close tracks the de-vigged book close to ~0.1-0.5c
Kalshi minus DK at the book line (book.csv): pooled +0.01c [-0.07,+0.08], n711; CFB/NHL within .25c; NBA +0.25c,
NFL -0.46c, CBB -0.17c. Comparing to a reference price removes outcome noise (SD 1.2c per game vs ~50c).
### Our existing sports rows have negative net CLV against Kalshi's close
reg.csv: mean close-implied EV -3.9%/$ on 398 K50/CBB-opener bets vs realized ROI -7.2%; slope 1.25 [0.28,2.31].
Gross book-points CLV > 0 hid this (the K50 lesson, repeated on the traded instrument).
### Hourly candle caches carry bid/ask only, not trade prices
xleague candles have no last-trade field; a last-trade close needs a re-fetch with price fields.
