# ufc_tsa -- UFC method/round of victory (KXUFCMOV, KXUFCVICROUND) + Kalshi daily TSA (KXTSAD)

Agent ufc_tsa, started 2026-11-19 (lead brief: STRATEGY_v2 top-15 #5, #11, #12). Scratch: results/ufc_tsa/.
Rules: no git, no orders, read-only GETs; sealed untouched; prereg written before any return is computed.

## Plan
1. Census of both families (settled events, volume, spread at decision time, fee type) -- descriptive, no outcome x price.
2. Per-rung calibration of mid and ask vs outcome (descriptive).
3. Pick the better candidate, write prereg here, register in edgelab, run DEV only.

## Log
### 2026-11-19 census (results/ufc_tsa/census_fetch.py -> census_raw.json; census.py output)
All four series fee_type quadratic, multiplier 1 (taker 0.07 p(1-p) per contract, rounded up per order; maker 0).
| series | settled events | span (open_time) | rungs/event | event volume p25/p50/p75 (contracts) | zero-vol markets |
|---|---|---|---|---|---|
| KXUFCMOV | 551 | 2026-01-21 .. 11-12 | 7 (A/B x KO-TKO-DQ / SUB / DEC + DRAW) | 139k / 303k / 548k | 0.4% |
| KXUFCVICROUND | 456 | 2026-01-22 .. 11-12 (3-6/mo until Apr) | 7 (A1-3,B1-3,OTHER) or 11 (5-rd) | 15k / 33k / 71k | 1.7% |
| KXUFCFIGHT | 824 | 2025-05-08 .. 11-11 | 2 | 277k / 839k / 1.65M | 0.4% |
| KXTSAD | 41 | 2026-10-09 .. 11-18 | 15 (one 18) | 314 / 1,340 / 2,596 | 69.6% |
Contract details: MOV "wins by Submission / KO,TKO,DQ / Decision", plus Draw rung; 7 of 551 events had no YES (NC/cancel).
VICROUND: "wins by KO/TKO/Sub/DQ in Round r"; OTHER = decision/draw/NC (technical decision counts as decision).
Market close_time = fight end (closes early); occurrence_datetime is a scheduled placeholder, NOT fight start.
KXTSAD: "At least K passengers on <date>", greater_or_equal, 100k spacing; opens 14:00 UTC ON the measured day, closes
11:59 UTC next day -> trading happens while the day is in progress; TSA publishes next morning.

### SPLITS FIXED NOW (before any candle/price is joined to an outcome)
- UFC (MOV + VICROUND): DEV = card date (ticker date code) < 2026-08-01; SEALED = >= 2026-08-01 (~35% of events).
  Calibration and every dev number use DEV only; SEALED candles are not fetched until a DEV pass + user yes.
- KXTSAD: DEV = event dates 2026-10-09 .. 2026-11-05 (first ~28 of 41); SEALED = 2026-11-06 onward.
- UFC decision time T = (earliest KXUFCFIGHT close_time on that card date) - 60 min, i.e. before the first fight of the
  card starts (first fight ends <= ~25 min after it starts). No quote at or after T is used.
- Fetches running: results/ufc_tsa/fetch_candles.py (DEV only, 3,916 markets: FIGHT 560, MOV 1,981, VICROUND 1,375;
  hourly candles [T-6h, T]); results/ufc_tsa/fetch_tsa_candles.py (DEV KXTSAD, hourly open..close).
  Excluded before fetch: markets that closed before T (MOV 28 / VICROUND 28 / FIGHT 78 markets, cancellations) and
  markets opened after T (MOV 84, VICROUND 88) -- printed by ufc_frame.py.

### UFC fighter history (outcome-only, no Kalshi) -- results/ufc_tsa/ufc_hist.py
Source: github.com/Greco1899/scrape_ufc_stats CSVs (scraped from ufcstats.com, updated through UFC 334 2026-11-14;
ufcstats.com/robots.txt returns 404, i.e. no robots file). Saved in results/ufc_tsa/dl/. 9,028 bouts.
Method classes mapped to Kalshi: KO = KO/TKO + doctor stoppage + DQ; SUB; DEC = any decision; overturned/CNC/NC = other.
Model M (P(method | fighter wins)): multinomial logistic on PIT shrunk shares (winner's KO/SUB/DEC share of past wins,
opponent's finished-by-KO/SUB/DEC per fight, alpha 3 toward weight-class base), weight-class base rates, 5-round, women,
log experience. Out-of-time check (fit 2012-2023, test 2024-2025, n 1,026): log loss 0.9224 vs weight-class base
rate 0.9743 (ufc_hist.py output). So fighter history carries method information beyond the class base rate; whether
Kalshi already prices it is the question.

### KXTSAD timing (tsa_peek.py)
Kalshi settles day d at ~13:45-16:15 UTC on d+1 (weekdays); Fri/Sat/Sun settle Monday. So at T = open + 2 h
(15:00/16:00 UTC on day d) Y(d-1) is known on Tue-Fri only (verifiable: settlement_ts of the d-1 event <= T).

### KXTSAD nowcast (results/ufc_tsa/tsa_model.py; history = data/e1x/tsa_daily_*.csv + KXTSAD expiration_value)
CSV vs KXTSAD first-published values: 21 overlapping days, max abs diff 0. Known-days rule above (d-1 only if published).
Pre-dev residuals ln(Y/F), 998 days 2024-01-15..2026-10-08:
- W (same weekday last week x 7-day WoW ratio): MAE 0.0641, sd 0.1008.
- A (same day 364 d ago x 14-day YoY ratio): MAE 0.0373, sd 0.0672, q05 -0.081, q95 +0.070; last 15 months MAE 0.0334.
-> A chosen (fixed before any KXTSAD price is looked at). Rung spacing 100k ~ 4% of volume, so the forecast is ~1 rung wide.

### KXTSAD DEV descriptive at T = open + 2 h (results/ufc_tsa/tsa_calib.py -> tsa_dev_quotes.parquet)
28 DEV events, 420 rungs; 321 have a candle <= T. Lifetime volume > 0 on only 25.7% of rungs.
Book at T: two-sided 14.3%, ask-only 30.0%, bid-only 27.1%, no quote 28.6%; median two-sided spread 0.10.
Calibration (two-sided rungs, n 60 over 28 events -- tiny): mid bins .05-.15 n12 yes 0/12; .15-.35 n9 mid .213 yes .111;
.35-.65 n12 mid .504 yes .667; .65-.85 n10 mid .762 yes 1.000; .85-.95 n15 mid .923 yes .933.
One-sided far rungs: ask-only (no YES bid) n 126, mean YES ask .15, YES rate 1.6%; bid-only (no YES ask) n 114, mean YES bid
.818, YES rate 98.2%. So the far brackets ARE left mispriced, exactly as in E4/E1X-T -- but the cheap side is not
buyable (NO ask = 1 - YES bid = 1.00 on ask-only rungs; YES ask absent on bid-only rungs). A taker rule has nothing to buy
on the far rungs; the middle rungs cost a 10c spread. Only a maker variant (post inside on the far side, filled by a rare
YES taker) could harvest it -- the same mechanism as the live TSA weekly maker lead, with 28 dev / 13 sealed events.
Capacity (outcome-free; results/ufc_tsa/fetch_tsa_trades.py -> tsa_trades_dev.json, 108 traded markets, 1,791 trades;
tsa_capacity.py): a 1c-inside maker on the one-sided far rungs at T would have been hit on 10 of 126 ask-only rungs
(22 events) and 4 of 114 bid-only rungs (20 events) in all of DEV. ~14 fills / 28 days; SEALED (13 events) would hold ~5.
-> Not testable with power. TSA daily is NOT pre-registered as a dev trial; recommendation: add KXTSAD far rungs to the
existing TSA maker paper logger (scripts/e1x_tsa_logger.py) as a forward-only arm with the T2 rule, and judge on forward
fills. UFC is the better candidate -> prereg below.

## PRE-REGISTRATION U1 -- UFC method of victory (KXUFCMOV), written 2026-11-19 BEFORE any MOV/VICROUND price is joined to
## an outcome (only the TSA descriptive above has been looked at; no UFC calibration yet). Edgelab id ufc_mov_m1_t60.
Question: does translating the Kalshi winner price (KXUFCFIGHT) x a fighter-history method model buy mispriced MOV rungs?
- Data: KXUFCMOV finalized events matched to a UFCStats bout (ufc_join.py; card date +-1 d, names); events with no YES
  (cancel/NC) dropped. VICROUND is descriptive only (not part of U1).
- T = earliest KXUFCFIGHT close on the card - 60 min (before the first fight starts). Quote = last hourly candle with
  end_period_ts <= T (no post-start quotes). FIGHT pair needs both YES bids > 0 and asks < 1.
- Model: pA = midA / (midA + midB) x (1 - 0.0175) [draw .0075 + NC .01 pre-2026 base]; q(A,m) = pA x M(m | A wins),
  M = multinomial logistic fit on UFCStats 2012-2025 only (ufc_hist.py / ufc_probs.py, frozen; columns m_KO/m_SUB/m_DEC).
  Draw rung never bet.
- Rule: for each of the 6 fighter x method rungs: YES cost = ask, NO cost = 1 - bid; require cost in [.03, .97] and the
  side's book to exist (ask < 1 for YES, bid > 0 for NO). fee per contract = ceil(100 x 0.07 x 10 x c(1-c)) / 100 / 10
  (10-lot, rounded up to the cent per order). edge = q_side - cost - fee. Per fight bet only the max-edge (rung, side) if
  edge >= 0.04; 1 contract, hold to settlement. ROI = sum pnl / sum (cost + fee).
- Split: DEV cards 2026-01-21..2026-07-31; SEALED 2026-08-01..2026-11-15 (not fetched; opened once only after a DEV pass +
  user yes); forward from 2026-11-20.
- CI: card-clustered bootstrap (resample cards), 10,000 draws, seed 20261119; one-sided p = share of ROI <= 0.
- Placebo P0 (base-rate shape only): same rule with M replaced by the multinomial on weight-class base rates + 5-round
  + women only (columns b_*). Placebo PS (shuffled model): fighter-specific M rows permuted across DEV fights (keeping
  each fight's pA), 200 permutations, rule ROI distribution.
- PASS (DEV) = all of: ROI CI lower > 0; edgelab family bar p (kalshi_sports_pregame, currently 0.05/23); paired card
  bootstrap ROI(U1) - ROI(P0) 95% CI lower > 0; U1 ROI > 95th percentile of PS; >= 60 bets; both chronological DEV
  halves > 0. Too-good guard: ROI > +15% on > 100 bets -> inspect rows before reporting.
- KILL: U1 ROI <= 0, or U1 not above P0 (difference CI includes 0 AND point <= 0) -> dead. ROI > 0 but any PASS item
  missed -> "not distinguishable"; LEAD only if U1 beats both placebos on the point estimate (forward paper only).
- Secondary, descriptive (no bar): Brier of q vs Kalshi mid over all 6 rungs, card-clustered CI; same for P0.

### UFC DEV descriptive calibration (results/ufc_tsa/ufc_calib.py; 23 DEV cards; card-clustered bootstrap 2,000)
Books at T: valid two-sided FIGHT 98.2%, MOV 93.6%; median spread FIGHT .01, MOV .02; quote age median 31 min.
MOV overround: sum of 7 mids per fight median 1.05 (IQR 1.015-1.08, n 277).
| rung | n | mean mid | mean ask | YES rate | y - mid [95% CI] |
|---|---|---|---|---|---|
| DEC | 549 | .214 | .235 | .235 | +.021 [-.014, +.056] |
| KO/TKO/DQ | 545 | .203 | .222 | .174 | -.029 [-.057, +.000] |
| SUB | 540 | .105 | .122 | .091 | -.014 [-.041, +.012] |
| DRAW | 208 | .022 | .032 | .010 | -.012 [-.018, +.000] |
By mid bin, only 0-.05 is outside zero (n 500, -.012 [-.021, -.002]); vs the ASK every bin but .20-.35 is negative.
Reading: the overround sits mostly on KO and the longshots; decisions are, if anything, cheap. Nothing is outside the cost.
VICROUND: DEV candle fetch stopped after FIGHT+MOV (shared throttle ~0.3 req/s; spread .05-.09 per STRATEGY_v2 probe = R5 0);
no VICROUND calibration done.

### U1 DEV RESULT (results/ufc_tsa/u1_dev.py -> u1_bets_dev.csv, p0_bets_dev.csv; scored once)
DEV: 1,554 rungs, 259 fights, 23 cards.
- U1: 205 bets, 116 wins, ROI +4.38%, card CI [-7.50%, +15.91%], p .226 (family bar 2.17e-3). Halves +12.7% / -3.3%.
- P0 (base-rate shape): 244 bets, ROI +4.05% [-6.12%, +14.53%], p .230.
- U1 - P0 paired: +0.33% [-8.68%, +9.87%].
- PS shuffled fighter model x200: mean +2.66%, p95 +8.35%; U1 sits at the 63.5th percentile.
- Brier(q) - Brier(mid): U1 +0.00175 [-0.00193, +0.00531]; P0 +0.00514 [-0.00071, +0.01078] (positive = model worse), n 1,526.
- Bets by side: NO KO 86 (+9.0%), NO SUB 54 (-7.8%), NO DEC 15, YES DEC 29 (+94.8% on $4.62 stake -- a few longshot hits).
VERDICT: not distinguishable. Fails the ROI CI, the family bar, the P0 difference, the shuffle percentile and the halves.
By the letter of the prereg (point above both placebos) it is lead-eligible, but the margin over P0 is +0.3 pts and over
the shuffle mean +1.7 pts: whatever return exists is the base-rate shape (selling KO / longshots into a 5% overround),
not fighter information. Kalshi's MOV mid already out-predicts the fighter-history model (Brier). Recommendation: DEAD
for agent time; no forward logger. Sealed untouched.
Row check (u1_bets_dev.csv): costs .04-.93 (median .61), sides/fees/pnl correct on inspected rows; mean claimed edge
+11.0% vs realized +4.4% -> the model's disagreements with the market are mostly the model's error (STRATEGY_v2 "rule PA").
Edgelab: registered ufc_mov_m1_t60-844fbeb5 (vault dataset kalshi_ufc_mov added to edgelab/vault.yaml; family
kalshi_sports_pregame now 23 trials, bar 2.17e-3). Unwired loader -> lead should retire it as dev-dead
(`python -m edgelab retire ufc_mov_m1_t60 --reason ...`); I did not retire it myself.

## Learnings
### Kalshi UFC method-of-victory books are tight and already price fighter finish history
KXUFCMOV median spread .02 at T, 5% overround; mid Brier beats a UFCStats fighter-history model (+0.00175 [-0.0019,+0.0053])
that itself beats weight-class base rates out of time (log loss .9224 vs .9743). Source: results/ufc_tsa/u1_dev.py, ufc_hist.py.
### UFC MOV rule return equals the base-rate placebo
U1 +4.4% [-7.5,+15.9] vs base-rate-shape P0 +4.1%; diff +0.3 [-8.7,+9.9]; shuffled-model median-ish rank 63.5%.
Any MOV return is the overround shape (KO and longshots rich), not fighter information. Source: notes/ufc_tsa.md U1.
### KXTSAD far brackets are mispriced but not buyable
At open+2 h, ask-only rungs (n 126) offer YES at .15 that settles YES 1.6%; bid-only rungs (n 114) bid .818 settle 98.2%;
the cheap side has no quote. Maker fills on those rungs in 28 DEV days: 14. Source: tsa_calib.py, tsa_capacity.py.
### Only 23 cards in 7 months of UFC dev
Card-clustered CIs on UFC rest on ~3 cards/month; anything under +10% ROI is unresolvable in one dev window.
