# k50_strike -- bet the model vs Kalshi's OWN ~50c strike, not the book-opener strike (agent, 2026-11-13)

## Question
K1 (NBA L32b), K-CFB (L36) and K-NFL (L33) carry CLV onto Kalshi but ROI ~0: we buy the strike nearest the BOOK opener,
which Kalshi has often already moved away from, so the contract costs ~0.54-0.58 (notes/nba_l32b_exec.md,
notes/kalshi_openers_xleague.md). Fix: measure the model gap vs Kalshi's own anchor strike (mid nearest 50c) and buy
that strike; and a ladder-optimal variant that picks the strike/side with the largest model EV after fee.

## Disclosed before pre-registering
- Seen (from the notes): K1 dev +1.9% n27 CLV +1.37; K-CFB dev -2.1% n63 CLV +1.49, cost 0.577, POST HOC subset
  (bought strike <= 4c wide) +11.4% n43; K-NFL dev -4.7% n14 CLV +0.79. This rule is motivated by that post-hoc look
  (forking path) and uses the SAME dev games, so dev here is not independent of what motivated it.
- NOT yet computed: any outcome of the K50 or ladder-optimal rules.
- SD for the ladder-optimal variant fit BEFORE any outcome, on the prior season only (results/k50_strike/sigma.py ->
  sigma.json): sd(final - book open), regular season: NBA 2025-26 total 18.61 (n 1230); NFL 2025 total 13.09 /
  home margin 12.51 (n ~270); CFB 2025 total 15.39 (n 901).

## Pre-registration (written before any outcome of these rules was computed)
Common ladder at decision time T: per strike, the last candle ending <= T (1-min candles <= 60 min old for NBA;
hourly candles <= 6 h old for CFB/NFL, as q1_timing.py), two-sided only (bid > 0, ask < 1). p_over(x) = YES mid
(or 1 - mid for away-team spread markets, kx.side_x). Anchor S50 = strike with |p_over - .5| smallest; ties -> narrower
spread, then the strike nearer the model.
Model m: NBA = L32b model - c (de-biased, as L32b); CFB = frozen L36 pred_total (raw, K-CFB primary); NFL = L33
logger model cache (total model C, spread model B home margin; raw, per the K-NFL bias check).

**Rule K50 (primary, one trial per league):** g = m - S50. Over/home if g >= k, under/away if g <= -k. Buy S50 on that
side at the ask (YES or NO = 1 - yes_bid per side_x) + exact Kalshi taker fee (ceil per 100-ct order). k fixed =
the existing thresholds: NBA 3.0, CFB 3.5, NFL total 2.5 / spread 2.0. Descriptive grid k in {1.5, 2, 2.5, 3, 4, 5}.
**Rule LO (ladder-optimal, one trial per league):** P_side(x) from Normal(m, sigma) with the sigma above; for every
two-sided strike and both sides EV = P - (ask + fee); take the max; bet if EV >= 0.03 per contract.
Universe / decision time T:
- NBA: every L32b "live" decision (pick or not) with t < 2026-11-11 (69 games); T = L32b tick (>= 30 min pre-tip).
  Quotes from results/nba_l32b_exec/q2_candles.parquet (28 picked games) + a light new fetch for the other 41
  (1-min candles T-60 min..T, strikes within 5 pts of the Pinnacle open, batch endpoint, results/k50_strike/).
  Graded by the Kalshi market result of the bought strike; CLV = Pinnacle last pre-tip line - strike, signed.
- CFB: K-CFB universe (q1_cfb_rows.csv totals with a first-tight hour >= 30 min pre-kick, FBS, pred available);
  T = first-tight hour (hourly cfb_candles.parquet). Graded on ESPN final; CLV = DK close - strike, signed.
- NFL: games in the open_lines logger model cache weeks 8-10 (live L33 models), totals + spreads;
  T = later of the cache time and Kalshi's first-tight hour (q1_nfl_rows.csv), >= 30 min pre-kick; hourly
  nfl_candles.parquet. Graded on ESPN final; CLV = DK close - strike, signed (home-margin axis for spreads).
Stats: ROI per bet = (1-cost)/cost win, -1 loss. Mean with 95% CI from a bootstrap clustered by game date (ET), 4,000
resamples, seed 0; one-sided p = share of resampled means <= 0. Same for CLV. Report avg cost (ask+fee), n, W.
Dev verdict per league x rule (pre-registered):
- PASS dev (-> sealed/real-money ask) only if ROI p < the edgelab family bar AND CLV > 0.
- GO forward paper if ROI > 0 (point), CLV > 0 (point), n >= 15.
- KILL if ROI CI upper < 0, or CLV <= 0 with n >= 20.
- Otherwise NEEDS DATA (forward paper only if the lead wants it).
Mechanism check (descriptive): avg cost should fall toward ~0.52 vs K1 0.543 / K-CFB 0.577 / K-NFL 0.544.

## Results

### Registered before scoring (edgelab, rule forward:unwired)
k50_nba_k50_kalshi-a6120730, k50_nba_lo_kalshi-969c41df (nba_totals, 34 trials, bar p < 1.47e-03);
k50_cfb_k50_kalshi-5009a8ea, k50_cfb_lo_kalshi-ba159055 (cfb_lines, 18, bar p < 2.78e-03);
k50_nfl_k50_kalshi-d7b7da0f, k50_nfl_lo_kalshi-9dbac4d6 (nfl_lines, 132, bar p < 3.79e-04). Specs edgelab/specs/k50_*.yaml.

### Data
- NBA: 41 non-pick dev games fetched (results/k50_strike/nba_fetch.py, batch endpoint, ~90 GETs, log nba_fetch.log);
  2 games returned 0 candles (CLE@CHA, MEM@TOR 11-09) -> "no quote". 67 of 69 decisions have a ladder.
  Grading by Kalshi settlement; asserted equal to final_total vs strike on every picked game (no mismatch).
- CFB: 255 FBS events with a first-tight ladder (5 no pred / non-FBS).
- NFL: model cache weeks 8-9 settled (week 10 not final); 47 game x market ladders. Lost: 8 rows "no kalshi event"
  (IND@JAX, PHI@WAS, JAX@BAL, LA@WAS -- the q1 fetch never pulled JAX/WSH games, a code-mapping gap upstream),
  3 never tight. NFL bets fall on only 3 game dates -> the date-clustered CI is degenerate.

### Dev results (results/k50_strike/score.py -> rows_<lg>.csv, summary.json; CIs clustered by ET game date)
| rule | n | W | ROI @ ask+fee [95%] | p(<=0) | CLV vs bought strike, pts [95%] | avg ask / cost | pre-reg verdict |
|---|---|---|---|---|---|---|---|
| NBA K50 k3 | 25 | 13 | -2.2% [-29.1, +24.6] | .59 | +0.88 [+0.56, +1.31] | 0.512 / 0.530 | NEEDS DATA |
| NBA LO | 33 | 13 | -17.3% [-47.4, +22.7] | .82 | -1.73 [-2.37, -1.17] | 0.454 / 0.471 | KILL (CLV <= 0, n >= 20) |
| CFB K50 k3.5 | 71 | 35 | -7.3% [-29.5, +27.7] | .66 | +0.82 [+0.40, +1.35] | 0.513 / 0.530 | NEEDS DATA |
| CFB LO | 125 | 57 | -10.5% [-29.1, +11.4] | .84 | +0.02 [-0.52, +0.74] | 0.492 / 0.510 | NEEDS DATA (CLV ~0; de facto dead) |
| NFL K50 (tot 2.5 / spr 2.0) | 17 | 10 | +11.6% (3 date clusters, CI degenerate) | .25 | +0.24 | 0.509 / 0.527 | GO forward paper (underpowered) |
| NFL LO | 30 | 8 | -36.2% [-71.0, -9.1] | .98 | -2.68 [-3.32, -1.30] | 0.393 / 0.409 | KILL (ROI CI upper < 0) |
- NFL K50 by market: total n7 5-2 +37.0% but CLV -0.14; spread n10 5-5 -6.2%, CLV +0.50. The ROI is luck on totals.
- Rows inspected (primary bets, all leagues): anchors two-sided, spreads 1-3c, none > 4c; NBA quotes <= 33 min old;
  S50 within ~0.6 (NBA) / 1.35 (CFB) pts of the book open.
- Descriptive k grid (NOT judged, forking paths): NBA ROI -2% .. -16% for k 1.5-4, -53% at k5 (n12); CFB -14% (k1.5)
  .. +2.0% (k4, n59) .. -10% (k5). No monotone dose-response in ROI; CLV rises with k in CFB (+0.48 -> +1.56).

### Mechanism check: the fix worked on price, and exposed that the CLV was smaller than it looked
- Avg cost fell to 0.530 (NBA, CFB) / 0.527 (NFL) from 0.543 / 0.577 / 0.544 under K1 / K-CFB / K-NFL.
- But CLV measured against the strike we actually buy at ~50c is +0.88 (NBA) / +0.82 (CFB), not +1.37 / +1.49:
  part of the old CLV was Kalshi's own move off the book opener, which was already in the price.
- results/k50_strike/clv_value.py: 1 pt of CLV is worth 2.1c (NBA, sigma 18.6) / 2.6c (CFB, 15.4) at 50c. At cost
  0.530 break-even needs CLV +1.40 (NBA) / +1.16 (CFB) pts. CLV-implied ROI: NBA -2.1% (observed -2.2%), CFB -1.7%
  (observed -7.3%). So the observed NBA loss is exactly what the CLV buys; the models' edge vs Kalshi (~0.85 pts)
  is about 60-70% of what taker friction (fee ~1.75c + half-spread ~1c) needs.
- LO fails by design: Normal(m, sigma) treats the model as the truth, so EV is maximized on strikes far toward the
  model (cost 0.41-0.47); the market is right more often than the model there (CLV -1.7 / -2.7 pts).

## Verdict
No league passes dev (no ROI p near any family bar). K50 fixes the price (cost ~0.53) but the CLV vs Kalshi's own
50c strike is ~0.85 pts, below the ~1.2-1.4 pts taker break-even. LO dead (NBA, NFL killed; CFB CLV ~0).
NBA/CFB K50: NEEDS DATA -- forward paper is cheap but a pass would need the model's CLV vs Kalshi to grow ~60%.
NFL K50: GO forward paper per the pre-reg rule, but n17 on 3 dates and the win is on totals with negative CLV.

## Forward rule (paper, no orders) -- for the ops agent, K50-NFL (k50_nfl_k50_kalshi-d7b7da0f); same form for NBA/CFB
At T = later of the L33 model-cache time and Kalshi's first tight hour (anchor <= 4c), >= 30 min pre-kick, for each
NFL game x {total, spread}: read the KXNFLTOTAL / KXNFLSPREAD ladder (orderbook or last 1-min candle <= T), two-sided
rungs only; S50 = rung with YES mid nearest 50c on the over / home-margin axis (kx.side_x); g = model - S50 (total
model C; spread model B home margin). Bet if g >= +2.5 (total) / +2.0 (spread) -> buy "axis > S50" side, if g <= -k
-> the other side, at the ask + exact taker fee (100 ct). Grade on the final vs S50; CLV = DK/Pinnacle close - S50
signed. Score with results/k50_strike/score.py logic on forward games (needs a forward split; currently dev-only).
Judge: same as K-NFL (100 settled ROI > 0 one-sided p < .10 AND CLV > 0 -> sealed ask; kill ROI CI upper < 0 after 60
or CLV <= 0 after 100). Realistically the NFL season ends at ~60-80 bets: judge then, labelled underpowered.
NBA/CFB: same rule with k = 3.0 (L32b m - c) / 3.5 (L36 pred at Kalshi first tight).

## Learnings

### Measure model CLV against Kalshi's own 50c strike, not the book-opener strike
L32b/L36 CLV vs the strike nearest the book open was +1.37 / +1.49 pts; vs Kalshi's own ~50c rung it is +0.88
[+0.56, +1.31] (NBA n25) / +0.82 [+0.40, +1.35] (CFB n71). Part of the old CLV was Kalshi's move already priced in.
Source results/k50_strike/summary.json. Matters: the old CLV overstated the executable edge by ~40%.

### Kalshi taker needs ~1.2-1.4 pts of CLV at 50c
1 pt of totals CLV = 2.1c (NBA, sigma 18.6) / 2.6c (CFB, 15.4); at cost 0.530 break-even CLV = 1.40 / 1.16 pts.
NBA CLV-implied ROI -2.1% vs observed -2.2%. Source results/k50_strike/clv_value.json. Matters: convert CLV to cents
before calling it an edge; our opener models deliver ~0.85 pts vs Kalshi.

### Buying at Kalshi's anchor cuts cost to ~0.53 but not to profit
K50 avg cost 0.530 (vs 0.543 K1, 0.577 K-CFB); ROI NBA -2.2% [-29.1, +24.6] n25, CFB -7.3% [-29.5, +27.7] n71.
Source results/k50_strike/summary.json.

### Ladder-optimal EV with a model-as-truth normal is a trap
LO buys far strikes toward the model (cost 0.41-0.47) and loses: NBA -17% (CLV -1.73), NFL -36% [-71, -9] (CLV -2.68),
CFB -10% (CLV +0.02). The market is right more than the model away from 50c. Source rows_<lg>.csv rule == lo.
Matters: any ladder optimizer must shrink the model toward the market (or fit P on market-relative residuals).
