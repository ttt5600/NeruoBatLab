# crypto_barrier -- barrier model for Kalshi monthly crypto max/min (one-touch) ladders

Agent: crypto_barrier (research/ops subagent, 2026-11-16). No orders, no git, no .env/kill changes. Kalshi GETs via
scripts/kx_throttle.py; Deribit + Coinbase public market data only. Scratch: results/crypto_barrier/.
Origin: notes/ot1_extension.md proposal (2): KXBTCMAXMON etc. are one-touch barriers, 27.3M contracts over 11 events,
~13 DEV months -> cannot be powered on DEV -> forward-only trial.

## Contract mechanics (read 2026-11-16, before any candle / outcome)
Sources: live GET /series + /events (results/crypto_barrier/rules_probe.json, rules_probe.py); contract terms
https://assets.kalshi.com/contract_terms/CRYPTOMINMAX.pdf (monthly series) and BTCMINMAX.pdf (KXBTCMAXY/MINY, KXBTCMAXM).
- Monthly (KXBTCMAXMON/MINMON, KXETHMAXMON/MINMON, KXSOLMAXMON/MINMON; terms CRYPTOMINMAX): YES iff "after issuance and
  through 11:59 PM ET on <last day of month>" the index is ever strictly above (MAX) / below (MIN) the strike. Index =
  CF BRTI / ETHUSD_RTI / SOLUSD_RTI. Terms: per-MINUTE value = simple average of that minute's (per-second) index values
  ignoring the top and bottom 20% "for each minute"; the market text adds "if the trimmed mean associated with any single
  minute ... is above the threshold -> Yes". The market rules_secondary wording ("top/bottom 20% of the cumulative
  dataset") is ambiguous (could read as a running trimmed mean since issuance); the TERMS say per minute. Working
  interpretation (pre-registered): monitored quantity = per-minute 20/20-trimmed mean of the index ~ the minute's central
  price; monitoring = every minute; a touch closes the market early (e.g. KXBTCMAXMON-BTC-26NOV30-8000000 close_time
  2026-11-04T15:34:39Z, seen in the open-event listing -- a FORWARD-period market, not DEV). Strike in ticker = $ x 100.
  Fee: series fee_type quadratic, multiplier 1 (taker 0.07 p(1-p); maker 0 -- maker is dead per lead brief).
- Yearly (KXBTCMAXY/MINY, terms BTCMINMAX; KXETHMAXY/MINY): rules_secondary describes a trimmed mean over "all BRTI values
  for each minute from market issuance" AND "if the BRTI crosses the threshold at any point ... the market immediately
  resolves". Contradictory -> yearly is DIAGNOSTIC ONLY (logged, never a bet).
- Mechanics check (pre-registered, uses outcomes only for MECHANICS, not for any model/threshold choice): for DEV markets
  that closed early, compare Kalshi close_time with the first minute where the Coinbase 1-min proxy (O+H+L+C)/4 crosses
  the strike. Per-minute interpretation predicts close within ~hours after the first proxy crossing; a running trimmed
  mean since issuance would predict far later / rarer touches. If the check contradicts the per-minute reading, the
  forward trial is RE-REGISTERED as a new trial before its first bet (never edited).

## PRE-REGISTRATION (written 2026-11-16 ~10:10Z before any Kalshi candle, quote history or settled result is read;
## the only quotes seen: none printed; one forward-period early close noted above)

### Models (frozen; code = edgelab/rules/crypto_barrier.py once written, same math in logger and descriptive check)
Notation: decision time T; S = spot; B = strike; tau = (market close_time - T) in years (365 d); sigma = Deribit DVOL/100
(BTC or ETH 30-day constant-maturity implied ATM vol, the last hourly DVOL candle CLOSE with candle end <= T; flat term
structure -- one input used identically on DEV and forward). Distance d = ln(B/S) (MAX) or ln(S/B) (MIN); d <= 0 ->
already through the strike on the spot proxy -> row skipped (`touched_proxy`). Discrete per-minute monitoring:
Broadie-Glasserman-Kou shift d' = d + 0.5826 sigma sqrt(1 min).
- M0 (diagnostic) Gaussian reflection with risk-neutral drift (r = 0): log-price drift m = -sigma^2/2 (MAX), +sigma^2/2
  for the mirrored MIN process; P = Phi((-d'+m tau)/(s)) + exp(2 m d'/sigma^2) Phi((-d'-m tau)/s), s = sigma sqrt(tau).
- M1 (PRIMARY barrier model, fat tail): driftless reflection on a unit-variance Student-t(4) terminal:
  P1 = min(1, 2 * (1 - F_t4( d' / (s * sqrt(1/2)) ))). (Same variance as M0; heavier tails -> more far-strike touches.)
- M2 (empirical bootstrap, vol-agnostic): Coinbase <ASSET>-USD hourly candles over the 730 days strictly before T (hour
  end <= T). Each hour -> (r = ln C_h/C_{h-1}, u = ln H_h/C_{h-1}, l = ln L_h/C_{h-1}). Paths = i.i.d. draws of whole
  UTC days (24-h blocks) until tau is covered (partial last day truncated to the hour); path max = max over hours of
  (cum r before the hour + u) (min with l for MIN). P2 = share of 20,000 paths (seed = int(T epoch) mod 2^32) with
  max >= d (no BGK shift: hourly H/L are tick extremes, already >= a trimmed per-minute mean -> M2 leans to MORE touches).
- S = Coinbase spot: DEV = close of the hourly candle ending at T; forward = Coinbase /ticker price GET before the Kalshi
  GET. (BRTI is a Coinbase/Kraken/Bitstamp/... composite; Coinbase basis is a few bp, disclosed, not corrected.)

### Descriptive check on DEV (counts only; NO trade-rule scoring, to avoid tuning)
Events closing <= 2026-04-10 (vault DEV cut used by ot1_extension; sealed 2026-04-11..2026-10-31 NOT fetched):
KXBTCMAXM (2024-03..2025-12), KXETHMAXM, KXBTCMAXMON/MINMON, KXETHMAXMON/MINMON (Jan-Mar 2026). Rows: each open market
x each day at T = 16:00 UTC with an hourly Kalshi candle ending <= T carrying yes_bid and yes_ask (mid = average).
Report: calibration table (bins of P on YES; counts, mean M0/M1/M2, mean mid, realised YES), Brier of M0/M1/M2/mid,
the same by event-month (clusters ~13) -- rows within a market are heavily autocorrelated; reported as descriptive.
NOT reported: any ROI of the forward rule on DEV (it is forward-only by design).

### Forward trial CB1 (edgelab, family kalshi_data_ladders, forward-only; registered before its first decision)
- Universe: open markets of KXBTCMAXMON, KXBTCMINMON, KXETHMAXMON, KXETHMINMON. (SOL: no DVOL -> M2-only diagnostic
  log; yearly: diagnostic log; KXBTCMAXM legacy series: included if it lists an open event.)
- Decision: once a day, first logger run at/after 16:00 UTC (bet only if the Kalshi GET starts in [16:00, 16:30) UTC;
  later -> logged `off_time`, not a bet). Order of reads: DVOL, Coinbase hourly history, Coinbase spot (all < Kalshi
  GET start) -> one Kalshi GET /markets per event (live yes_bid/yes_ask).
- Rule (taker only): for side X in {YES, NO}: ask_X = yes_ask (YES) or 1 - yes_bid (NO); P_X from M1 and M2 (NO = 1 - P);
  P_cons = min(P1_X, P2_X); fire iff P_cons - ask_X - fee(ask_X) >= 0.05, fee = 0.07 a (1-a) exact; ask_X in
  [0.05, 0.95]; yes_ask - yes_bid <= 0.05; tau >= 2 days; spot not already through the strike. At most ONE bet per
  market over its life (the first day it fires). 1 contract, stake = ask; pnl = (1 - ask - fee) or (-ask - fee).
- Cluster = ET calendar month of close (BTC + ETH, MAX + MIN of one month share a cluster: outcomes are one path).
- Outcome = Kalshi market `result` (yes/no) after settlement (expiration <= a week after month end).
- Judge: edgelab forward split; no verdict before n >= 60 bets AND >= 12 month-clusters (so >= 12 months, realistically
  more). Bar = family kalshi_data_ladders alpha .05 / trials. Kill (edgelab spec): roi_ci_below_zero after 60. Too-good
  guard: ROI > +15% on >= 60 bets = bug until rows inspected.
- Power, stated now: one month = one cluster, so this cannot pass in under a year; the logger is mainly a forward
  calibration record of M1/M2 vs the Kalshi book (every rung's quote + P logged daily for that purpose).

## Log
- 11-16 ~10:00Z started; rules + terms read (above); feasibility (results/crypto_barrier/feas_probe.py): DVOL hourly
  history BTC/ETH yes (back to 2024-01 at least), SOL none; Coinbase 1-min/1-h candles OK; Kalshi /historical/markets
  lists DEV events (e.g. KXBTCMAXMON-BTC-26JAN31: 8 rungs 95,000..112,500).
- 11-16 Frozen code: scripts/crypto_barrier_model.py sha256 6093f148...99ba (models M0/M1/M2 + decide), unchanged from
  before the DEV frame was built. DEV pipeline scripts/crypto_barrier.py (markets | candles | ext | frame | mech | calib).
  Wrong turn fixed BEFORE any outcome read: historical candles carry dollar STRINGS ("0.1500") under yes_bid.close; first
  parse divided by 100 (all quotes <= 1c) -> fixed in num(). tau runs to the window end parsed from rules_primary
  (window_end), never close_time (which moves earlier on a touch = outcome leak).
- 11-16 DEV markets: 22 events / 147 markets (results/crypto_barrier/markets.parquet); legacy KXBTCMAXM/BTCMAXM 2024-03..
  2025-11 + KXETHMAXM-25DEC01 + MON series Jan-Mar 2026. KXBTCMAXM-25 (1-day event) has no candles -> no rows.
- 11-16 Frame (results/crypto_barrier/frame.parquet): 3,106 rows (market x day at 16:00 UTC), 139 markets, 21 events,
  12 calendar months; median quote age 0 h; 1 row touched_proxy (dropped).

## Results
### Mechanics check (results/crypto_barrier/mech.csv; 33 early-closed YES markets)
Kalshi close_time minus the first Coinbase minute whose (O+H+L+C)/4 crossed the strike: median 0.66 h, 15/33 < 0.5 h,
20/33 < 1 h; monthly MON series 18/25 < 1 h (typically 7-12 min). 12/33 > 3 h (max 72 h) -- legacy rules, a Coinbase
proxy that pokes through while BRTI's trimmed minute does not, or admin lag; not resolved. A cumulative trimmed mean
since issuance could not close within minutes of a fresh spot crossing -> PER-MINUTE interpretation supported; CB1 stands
as registered (no re-registration).

### Descriptive calibration on DEV (results/crypto_barrier/calibration.txt; counts only, NO rule ROI computed)
Rows are autocorrelated within market and month; 12 months ~ 12 independent draws of the BTC/ETH path.
| Brier | M0 Gaussian | M1 t4 | M2 bootstrap | Kalshi mid |
|---|---|---|---|---|
| all 3,105 rows | .0634 | .0605 | .0653 | .0634 |
| spread <= 5c (2,651) | .0496 | .0465 | .0508 | .0478 |
Per event (21): M1 Brier < mid in 13 (binomial two-sided p .38), M0 14 (p .19), M2 9; per month (12): M1 < mid in 8.
-> NOT distinguishable: no model beats the Kalshi mid on DEV. M1 sits BELOW the mid in 20/21 events (mean p1 - mid < 0).
By barrier side (spread <= 5c): MAX rows 1,962: p1 .083 / p2 .110 / mid .093 / realised .045 (Kalshi MAX YES rich, both
models closer); MIN rows 689: p1 .080 / p2 .096 / mid .124 / realised .123 (mid right, BOTH models too LOW on downside
touches -- consistent with put skew that a flat ATM DVOL and a symmetric t ignore; only ~5 MIN events / 3 months, incl.
the Jan 2026 BTC slide where all 8 KXBTCMINMON-26JAN31 rungs hit -- rows inspected, not a bug).
Low tail: p1 < .02 bin mid .0205 vs realised .0065 (YES longshot rich by ~1.4c) -- untradeable: NO ask there is
>= .98 (> the .95 cap) and the fee/spread eat it.
Would-fire COUNT (no outcome): 18 DEV markets in 8 events would have fired on some day, 13 of them MIN-NO -- the rule's
bets concentrate exactly where DEV shows the models under-predicting (MIN). Stated as a risk, not scored.

### Forward trial (registered 11-16 10:18Z, before the first 16:00 UTC decision)
- edgelab cb1_crypto_barrier_taker-4b3dd64d (family kalshi_data_ladders 56 trials -> bar 8.93e-4); spec
  edgelab/specs/cb1_crypto_barrier_taker.yaml (rule forward:unwired, min_n 60, kill roi_ci_below_zero after 60);
  vault.yaml dataset kalshi_crypto_barrier (forward 2026-11-16+, adapter edgelab/adapters/crypto_barrier_fwd.py
  sha256 ab820521...). Sandbox evaluate --split forward (results/crypto_barrier/registry_sandbox.jsonl): runs, 0 bets.
  Synthetic adapter check (results/crypto_barrier/adapter_check.py): only status bet + settled -> bet; PIT 0 violations.
- Logger scripts/crypto_barrier_forward.py (tick | settle | status); deploy scripts/install_crypto_barrier_agent.sh ->
  ~/.claude/quant-sports/scripts/{qs_job.sh, kx_throttle.py, crypto_barrier_model.py, crypto_barrier.py,
  crypto_barrier_forward.py}; launchd com.jonathanwang.quant-sports-crypto-barrier every 300 s (qs_job.sh case
  crypto-barrier). NOTE: the deployed qs_job.sh now also carries the repo's `e4p` case (another agent's edit, inert until
  its own launchd job is installed).
- Verified 11-16 10:18Z: kickstart -> idle tick (before 16:00 UTC) rc 0, settle ran, heartbeat written, launchd err
  empty. Forced manual run under the DEPLOYED venv (writes only *_manual.jsonl): 11 series, 71 quotes, 57 scored, 0
  fires (e.g. KXBTCMINMON-26NOV30-60000 bid/ask .42/.43 vs M1 .304 / M2 .462 -> models disagree, P_cons blocks).
  ~35-60 s per decision (Coinbase 730-d hourly history x3 assets). First real decision: 2026-11-16 16:00 UTC.
- scorecards.py card CB1 (test render results/crypto_barrier/scorecards_test.md -> too-early).

## Verdict
No DEV edge claim is possible or made. On 12 DEV months the barrier models are not distinguishable from the Kalshi mid
(M1 Brier .0605 vs mid .0634, 13/21 events). Models look better on MAX (Kalshi MAX YES rich) and worse on MIN (downside
touches under-predicted by both). CB1 runs forward as a paper trial + daily calibration log; expect no verdict for
>= 12 months. PROPOSAL (lead/user): a NEW skew-aware trial (e.g. 2 x P_RN(S_T beyond K) from the Deribit monthly smile,
OT1's smile code) logged side by side, registered before it sees forward outcomes. Nothing done.

## Learnings
### Kalshi monthly crypto max/min ladders settle on a per-minute trimmed BRTI value and close minutes after a touch
CRYPTOMINMAX terms: 20/20-trimmed average "for each minute"; DEV early closes follow the first Coinbase-minute crossing
by a median 0.66 h, 18/25 monthly-series markets < 1 h (results/crypto_barrier/mech.csv). Model them as per-minute
one-touch barriers on spot; take tau from the rules' window end, never close_time (it moves on a touch = leak).

### On DEV, reflection/t4 and bootstrap barrier models do not beat the Kalshi mid on monthly crypto max/min
Brier M1 .0605 / M0 .0634 / M2 .0653 / mid .0634 over 3,105 market-days, 12 months; M1 wins 13/21 events (p .38).
Source: results/crypto_barrier/calibration.txt. Not distinguishable -- 12 months is ~12 draws of one path.

### Kalshi crypto MAX ladders ran YES-rich on DEV; on MIN the mid was fair and the models under-predicted touches
Spread <= 5c rows: MAX mid .093 vs realised .045 (models .083/.110); MIN mid .124 vs realised .123 (models .080/.096).
A flat ATM vol plus a symmetric tail misses put skew; the frozen rule's DEV would-fire markets are 13/18 MIN-NO --
exactly where it is weakest.

### Kalshi historical candles return dollar strings, not cents
/historical/markets/{t}/candlesticks yes_bid.close = "0.1500" (string dollars); legacy fields are integer cents.
Parse strings as dollars. options_implied_tails._px treats any number <= 1 as dollars (an int 1-cent quote reads $1).
