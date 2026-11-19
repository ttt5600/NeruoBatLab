# mortgage_rates -- Kalshi weekly 30Y mortgage-rate ladders (KX30YMORTW) vs a daily-index PMMS nowcast  (agent mortgage_rates, started 2026-11-18)

## Question
Kalshi KX30YMORTW ("30-year mortgage rate this week") settles on Freddie Mac PMMS 30Y FRM, FIRST published value,
released Thursday 12:00 ET (17:00Z). Daily public data (Optimal Blue OBMMI 30Y conforming, FRED OBMMIC30YF; 10Y
Treasury DGS10) cover the PMMS measurement window before the release. Are rungs far from a nowcast mispriced (E4 / TSA
pattern: stale quotes on the near-certain side)?

## Census (2026-11-18/19, results/mortgage_rates/census.py -> series_hits.json; events.py -> markets.json)
16,702 Kalshi series scanned; mortgage-rate related:
- **KX30YMORTW** weekly: 10 events (26SEP17 .. 26NOV19; 26OCT01 was a 1-day event), 15-35 `greater` rungs per event,
  strikes 1-2 bp apart (e.g. 7.10, 7.11, 7.12 ...), 227 markets (192 finalized, 35 active = 26NOV19). volume_fp per
  event 31k-161k contracts. Opens ~Thu/Fri after the previous release, close 15:59Z (EDT) / 16:59Z (EST) on release
  Thursday, early close on release. fee: see API (quadratic assumed; confirm in logger).
- FRM / KXFRM (legacy "Mortgage 30-yr rate this week"): 0 markets returned by /markets (archived).
- KXFM30YMTG (3 events, year-end levels, active, 0 settled), KXMORTGAGERATE (2 events, Dec-26 / Dec-27),
  KXFRMMAX/KXFRMMIN (annual): long-dated, not the information-timing mechanism -> out of scope (long-dated tail = cash
  rate, FUND_BOOK Learnings).
- KXMBAAPPS (MBA weekly applications, 6 events) -- different underlying, noted only.
- Closest home-related: *HOME Zillow series (notes/home_values.md, forward only).
Rules (KX30YMORTW-26NOV19-T7.32, rules_secondary): underlying = PMMS 30Y FRM national average for the date; weekly
measurement period = Thursday 00:00 ET .. Wednesday 23:59 ET before the release; first published value; later
revisions ignored. "Above K" = strictly greater (rules_primary "is above").

## Disclosed before pre-registration (seen while exploring; nothing else looked at)
- FRED current tail: MORTGAGE30US 10-29 7.24, 11-05 7.18, 11-12 7.11; OBMMIC30YF 11-13 7.255, 11-16 7.263,
  11-17 7.286; DGS10 11-17 5.10.
- markets.json printed the three LOWEST rungs' results per event (all `yes` except 26OCT29 T7.24 = `no`). No quotes,
  candles, trades or nowcast errors examined.
- ALFRED vintage CSV works without a key: alfred.stlouisfed.org/graph/alfredgraph.csv?id=X&vintage_date=YYYY-MM-DD
  (one vintage per call). Vintage 2026-10-15 of OBMMIC30YF already contains obs 2026-10-15 (same-day posting).

## Pre-registration (written 2026-11-19 ~05:00Z, BEFORE any nowcast-error, quote, tape or outcome join)
Event = release Thursday D. Decision time T = D 14:00Z (09:00 ET EST / 10:00 EDT; >= 1 h before close).
Inputs strictly available before T: ALFRED vintage dated D-1 (Wednesday) of OBMMIC30YF and of MORTGAGE30US.
Nowcast N1 (frozen):
  A = mean OBMMI obs dated D-7 .. D-1 (Thu..Wed window) in vintage D-1; B = mean OBMMI obs dated D-14 .. D-8 in the
  SAME vintage; P0 = PMMS for D-7 in vintage D-1 (= first print; checked against later vintages for revisions).
  mu = P0 + beta * (A - B). beta = OLS slope (no intercept) of (PMMS_D - P0) on (A - B) over calibration weeks.
  Residual e = PMMS_D(first print) - mu. Margin m = max |e| over calibration weeks, rounded UP to the bp, + 0.02.
Calibration weeks (no Kalshi data exists for them): D in 2023-01-05 .. 2026-09-10 (post Nov-2022 PMMS methodology
  change to LPA application data). Vintages: one ALFRED Wednesday vintage per week (results/mortgage_rates/vintages/).
Far rungs: "greater K" with K >= mu + m -> bet NO; K < mu - m -> bet YES (YES needs value > K; K + 0.01 <= mu - m).
Trial MR1 taker: at T, buy the far side at its ask (last hourly candle ending <= T; side ask = yes_ask for YES,
  1 - yes_bid for NO) + exact taker fee (0.07 p(1-p), ceil to the cent per 100 ct) iff side ask in [0.50, 0.98].
  One contract per rung (all qualifying rungs in an event), held to settlement.
Trial MR2 maker: same rungs; post q = min(floor(side mid), side ask - 0.01), q in [0.50, 0.98]; filled iff a later tape
  trade before close prints strictly THROUGH q on that side; maker fee 0. Report unfilled-rung outcomes alongside
  (filled-vs-unfilled adverse-selection check, FUND_BOOK learning).
Cluster = event (all rungs of one week share one PMMS print).
Splits by D: dev 2026-09-17 .. 2026-10-29 (7 events), sealed 2026-11-05 .. 2026-11-12 (2 events; opened only on the
  user's yes), forward 2026-11-19 ..
Honest power: a loss-free 96-98c strategy on 7 events cannot pass the bar (null p = price^n; 7 clusters). Dev is a
  sanity/capacity check only. The decision is FORWARD:
  Success (forward, each trial separately): n >= 40 bets over >= 10 events AND net ROI 90% event-cluster bootstrap CI
  > 0 AND the edgelab family bar. Kill: losses_in_first n=20 max_losses=2 (a rung outside max residual + 2 bp losing
  3 of 20 = nowcast broken), or ROI CI upper < 0 after 40.
Ex-ante doubt: OBMMI (rate locks) and PMMS (LPA applications) are both public and the Kalshi ladder is narrow (strikes
  within ~+-15 bp of the last print), so far rungs may already be 99/1 (no edge at <= 98c) -- expected outcome: few or
  zero qualifying rungs.

## Plan
1. vintages.py: ALFRED Wednesday vintages 2023-01 .. 2026-11 for OBMMIC30YF + MORTGAGE30US.
2. nowcast.py: beta, residuals, m (calibration only); then N1 for the 10 Kalshi events.
3. dev.py: hourly candles at T + tape for dev events -> MR1/MR2 picks and outcomes (dev only; sealed untouched).
4. edgelab: dataset kalshi_mortgage30 (vault splits above), MR1/MR2 specs, forward rule unwired until an adapter exists.

## Step 1-2 results: calibration of N1 (results/mortgage_rates/nowcast.py -> calib.json, calib.csv; 2026-11-19)
408 ALFRED Wednesday vintages (vintages.sh via curl; python-requests is tarpitted by alfred.stlouisfed.org, read
timeouts -- use curl). Calibration weeks n 180 (2023-01-05 .. 2026-09-10):
- beta 0.981 (OBMMI weekly-mean change maps ~1:1 to PMMS change).
- residual |e|: sd 0.0181, median 0.0088, q95 0.0370, q99 0.0528, max 0.0853 (2023-06-01). Naive "no change" sd 0.0931,
  max 0.28. So OBMMI explains ~96% of weekly PMMS variance: the PMMS print is ~known by Wednesday night.
- margin m = ceil(max) + 0.02 = 0.11 (frozen rule). A_last = Wednesday in 173/180 vintages (OBMMI posts same day).
- PMMS first print vs current value: 0 of 180 revised (rev_max 0.00) -> FRED current vintage == first print here.

## Step 3 result: dev (7 events 26SEP17..26OCT29; sealed 26NOV05/26NOV12 untouched) -- DEAD
results/mortgage_rates/dev.py quotes dev -> quotes_dev.csv (151 rungs, 1 without a candle at T); score dev -> score_dev.csv.
- Far rungs selected by N1 (|K - mu| beyond m = 0.11): 59 with a quote. All 59 won on the far side.
- Side ask at T: 0.99 (3 rungs, 26SEP17 6.81/6.82/6.83) or 1.00 (56). MR1 taker: 0 bets (ask cap 0.98).
  MR2 maker: 3 postable at q = 0.98, 0 filled (0-2 tape trades per rung after T). => no bet either way.
- Why: the market's two-sided rungs (yes bid >= 3c, ask <= 97c) at T sit within -0.049 .. +0.043 of mu (per-event
  ranges, dev.py descriptive), so the market's band on Thursday 09:00 ET is about as tight as the nowcast's own error
  (sd 0.018). The ladder spans ~+-15-25 bp, so every rung beyond m is pinned at 99/1. Same shape as the TSA-taker zero.
- Wrong turn: the first picks() let a rung with NO candle at T default to mid 0.50 (26OCT01 T7.09 "postable at .50");
  fixed to "no quote = no bet" before scoring.
- edgelab: dataset kalshi_mortgage30 (vault.yaml), mr1_mortgage_far_taker-ee4a3843, mr2_mortgage_far_maker-e55a22ba
  registered (family kalshi_data_ladders -> 65 trials, bar p < 7.69e-04) and retired (dead, dev 0 bets); verify ok.
- Not run (would be new trials needing a new prereg): an earlier decision time (Mon-Wed: nowcast has fewer OBMMI days,
  market band presumably wider) and a probabilistic interior-rung rule (N1 normal sd ~0.018 vs mid). With 10 events of
  history neither can pass the bar before ~40+ forward weeks. No logger proposed: the far-rung rule has nothing to log.
- Capacity (for any future version): event volume_fp 31k-161k; far rungs show resting 1c YES asks ~2.6k contracts
  (26NOV19 T7.32 yes_ask_size_fp 2581): NO at 99c exists in size, but max return ~1%/week with ruin on a miss.

## Learnings

### Weekly PMMS is ~96% known by Wednesday night from OBMMI
mu = last PMMS + 0.981 x (OBMMI Thu..Wed mean change) leaves residual sd 1.8 bp vs 9.3 bp for "no change" (n 180 weeks
2023-01..2026-09, max 8.5 bp; results/mortgage_rates/calib.json). PMMS first prints were never revised (0/180).
Useful input for any rates/housing market that settles on PMMS.

### Kalshi KX30YMORTW far rungs are already 99/1 by release morning -- no E4-style edge
All 59 dev far rungs (beyond max residual + 2 bp) had side ask 0.99-1.00 at release-day 14:00Z; 0 taker bets, 0 maker
fills (results/mortgage_rates/score_dev.csv). The ladder is 1-2 bp spaced and only +-15-25 bp wide; the public daily
index is priced in. Dead for the release-day decision time (do-not-redo).

### ALFRED vintage CSVs need curl, not python-requests
alfred.stlouisfed.org/graph/alfredgraph.csv?id=X&vintage_date=D works without an API key, one vintage per call, but
python-requests reads time out (tarpit); curl returns in ~0.6 s (results/mortgage_rates/vintages.sh).
