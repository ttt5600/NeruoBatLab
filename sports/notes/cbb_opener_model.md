# cbb_opener_model -- a de-biased CBB totals model vs the opener (DK) and vs Kalshi KXNCAAMBTOTAL (agent, 2026-11-13)

## Question
Kalshi CBB total ladders list ~23 h pre-tip and sit on the book opener (notes/kalshi_openers_xleague.md: D -1.07
[-1.33,-0.81], n 171). We have no CBB pregame model. Does a simple, de-biased CBB totals model lead the opener
(CLV) and make money at (a) the DK open price and (b) Kalshi's ~50c rung at ask + fee?

## Disclosed before pre-registering
- Known from other leagues: L32b NBA de-biased dev +8.1% no-vig (p .006), -110 not distinguishable; CFB simple model
  total CLV +1.02 pts, ROI -110 +2.6% [-1.3,+6.8] (vig eats it); L36 CFB sealed CLV holds, ROI ~0.
- Probed (schema only, no outcome): ESPN core odds for 1 game each of 2023-24 and 2026-27 return DraftKings (and ESPN BET)
  total open/close with prices. I have NOT computed any model, any model-vs-line gap, or any CBB Kalshi price.
- CBB in-game WP is dead (do-not-redo); this is pregame only.

## Data / split (fixed now)
- Results: ESPN site scoreboard, D-I group (groups=50), finals only. Seasons (ESPN labels): 2023 = 2022-23 (burn-in prior
  only), 2024 = 2023-24 and 2025 = 2024-25 (DEV, scored), 2026 = 2025-26 (SEALED for scoring: its lines are not read;
  its RESULTS are used only as the prior/training input for 2026-27 forward predictions), 2027 = 2026-27 (Kalshi test +
  forward).
- Opener: ESPN core odds, DraftKings total open/close + American prices (fallback ESPN BET if DK missing). Timing of the
  "open" is unknown (no timestamp). Kalshi: KXNCAAMBTOTAL hourly candles already pulled by kalshi_openers_xleague
  (results/kalshi_openers_xleague/cbb_*.parquet, 11-03..11-11) plus a refetch through the latest settled date.

## Model (fixed a priori; no tuning against any market number)
- Per game two rows: log(points scored by team t vs opponent u) = mu + o_t + d_u + h*site, site = +1 home / -1 away /
  0 neutral. Teams with < 8 D-I scoreboard appearances in the season pooled as one "NONDI" team.
- Walk-forward date by date: fit on this season's finals with ET date strictly < game date. Weighted ridge, recency weight
  0.5^(age_days/45). Ridge pulls o_t, d_t toward 0.6 x last season's final value (new teams -> 0) with penalty
  lambda = 6 (game-equivalents); mu, h unpenalized but mu/h get a weak prior at last season's value (lambda 6) so day 1
  is defined.
- Prediction: total = exp(mu + o_h + d_a + h*s) + exp(mu + o_a + d_h - h*s).
- One outcome-only sanity check allowed (MAE, bias of model vs final total per season) -- no market.

## De-biasing + bet rule (fixed a priori)
- c(g) = mean(MODEL - OPEN) over the most recent N = 200 games of the SAME season with MODEL and OPEN and ET date strictly
  earlier than g's; requires >= 50 such games, else no bet. No cross-season carry.
- adj = MODEL - c - OPEN. OVER if adj >= k, UNDER if adj <= -k, **k = 4.0**. Flat 1u, one bet per game, D-I vs D-I and
  D-I vs NONDI both allowed (whatever the scoreboard + DK lists). Graded vs the OPEN total (push = 0).
- Descriptive only (not judged): k = 3, 5, 6.

## Dev test (2023-24, 2024-25)
- PRIMARY (model edge): ROI at proportional no-vig DK open price. SECONDARY (executable as a book taker): ROI at the
  actual DK open American price. Bootstrap by ET date clusters, 4,000 resamples, seed 0; one-sided p = P(mean <= 0).
- CLV (pts) = signed (close - open) in the bet direction, same bootstrap.
- PASS (model edge) = no-vig ROI p < .05 AND both seasons > 0 AND CLV > 0. Executable pass = actual-price ROI p < .05.
- Too-good guard: ROI > +10% on > 300 bets -> inspect rows before reporting.
- Kill (no Kalshi forward): no-vig ROI CI upper < 0 OR CLV CI upper <= 0.

## Kalshi test (2026-27 so far; descriptive go/no-go)
- Decision candle = Kalshi first-tight hour (anchor two-sided spread <= 4c) that is >= 30 min before the ESPN tip.
- S = Kalshi's own anchor strike (YES mid nearest 50c at that candle; tie -> the strike favourable to our side).
  MODEL from the walk-forward fit with games strictly before the game's ET date; c from 2026-27 games (>= 50 prior, N 200).
- Bet if |MODEL - c - S| >= k (4.0): over -> buy YES at ask, under -> buy NO at 1 - bid; + exact Kalshi taker fee
  ceil(0.07*C*P*(1-P)) per contract at 1 contract (edgelab fees). Graded final total vs S. CLV = DK close - S signed.
- GO forward if n >= 30 and CLV > 0 (point estimate); KILL if CLV CI upper < 0. ROI reported with CI; no claim from it.
- 2025-26 Kalshi CBB totals (if they existed) are sealed: not read.

## Edgelab
New family `cbb_totals`; trial cbb_t1_kalshi_taker registered before the Kalshi pricing is run.

## Results
(appended after scoring)

### Edgelab (registered 2026-11-13 ~08:10Z, before kalshi.py was run)
cbb_t1_kalshi_taker-4dd8f3e6, family cbb_totals (1 trial -> bar p < .05), rule forward:unwired. Spec edgelab/specs/cbb_t1_kalshi_taker.yaml.
The dev book test (DK open) is a separate, non-edgelab trial set (k = 4 primary, k 3/5/6 descriptive).

### Model build (results/cbb_opener_model/fetch.py, model.py; outcome-only check)
ESPN D-I scoreboard finals: 2023 6,228 / 2024 6,243 / 2025 6,292 / 2026 6,300 / 2027 685 (11-02..11-12).
Walk-forward model vs final total: MAE 13.61 / 13.46 / 13.06 / 13.95 / 14.63; bias -2.22 / -1.77 / -1.38 / -1.09 / -3.15
(log-points model predicts a geometric mean and ignores OT -> under-predicts; the de-bias c removes this level).
First-2-weeks MAE 13.6-15.5. Amendment (made in code before any line was joined): the D-I set is fixed from the PREVIOUS
season (>= 8 appearances), not the current one, so early-season forward predictions do not need future appearance counts.
ESPN DK lines 2026-27: 490 of 685 games have DK open+close+prices (odds_2027.parquet).

### Kalshi test, 2026-11-03..11-12 (kalshi.py -> kalshi_rows.csv, kalshi.json)
Kalshi events with a total ladder: 241 (11-03..11-11 from kalshi_openers_xleague + 11-12 refetch tagged _cbbom);
220 had a first-tight hour >= 30 min pre-tip and a c; 21 never tight. c (2026-27) ranged -3.53 .. -1.69.
| rule | n | W-L | ROI @ ask+fee [95%], p | P&L/contract | CLV vs S, pts [95%] | mean cost | overs |
|---|---|---|---|---|---|---|---|
| **k = 4 (primary)** | 67 | 42-25 | **+16.3% [-6.0, +37.3]**, p .077 | +0.085 [-0.034, +0.198] | **+0.04 [-0.55, +0.66]** | 0.541 | 38 |
| k = 3 (descr.) | 92 | | +15.1% [-3.4, +33.3], p .050 | | | | |
| k = 5 (descr.) | 50 | | +26.2% [+1.0, +49.2], p .020 | | | | |
| k = 6 (descr.) | 28 | | +19.7% [-13.5, +52.4], p .127 | | | | |
- Pre-reg go/no-go: n 67 >= 30 and CLV point estimate > 0 (+0.04) -> GO forward, but only just: CLV is
  NOT distinguishable from 0. ROI not distinguishable from 0. A 62.7% win rate with zero CLV is what luck looks like on
  67 bets (sd of the ROI mean ~11%); the k=5 CI excluding 0 is one of 3 descriptive cuts (forking path) -- not a claim.
- Rows inspected: no duplicate events; anchor spread median 3c (1 bet had a 47c-wide anchor, NAUARIZ, lost);
  S is 0.81 pts from the DK open on average (Kalshi sits on the opener, as kalshi_openers_xleague found);
  residual mean(model - c - open) on priced games +0.54.

### Dev test vs DK opener, 2023-24 + 2024-25 (dev.py -> dev.json, dev_bets.csv)
11,173 games with a DK (fallback ESPN BET) open + close + open prices; 10,952 with c. Outcome MAE: model 13.12,
de-biased 13.10, DK open 12.90, DK close 12.78 (the scores-only model alone is worse than the opener, as expected).
c (dev): 2023-24 mean -0.69 (-2.37..+1.91), 2024-25 -0.24 (-1.25..+1.52). Date-cluster bootstrap 4,000, seed 0.
| rule | n | over/under | win | no-vig ROI [95%], p | ROI at DK open price [95%], p | CLV pts [95%] | seasons no-vig | seasons CLV |
|---|---|---|---|---|---|---|---|---|
| **k = 4 (primary)** | 2,213 | 1105/1108 | .522 | **+3.2% [-0.7, +7.0], p .054** | -1.4% [-5.1, +2.2], p .78 | **+0.83 [+0.69, +0.97]** | +3.4% / +3.1% | +0.95 / +0.72 |
| k = 3 (descr.) | 3,610 | 1856/1754 | .529 | +4.9% [+1.9, +8.0], p .0005 | +0.2% [-2.7, +3.1] | +0.65 [+0.54, +0.76] | +4.6% / +5.1% | |
| k = 5 (descr.) | 1,287 | | .521 | +2.4% [-2.6, +7.4], p .18 | -2.3% [-7.0, +2.6] | +0.97 [+0.76, +1.18] | | |
| k = 6 (descr.) | 750 | | .544 | +6.2% [-0.6, +12.7], p .038 | +1.4% [-5.1, +7.6] | +1.24 [+0.96, +1.52] | | |
- PRE-REG VERDICT: model-edge PASS needs p < .05 -> **FAIL (p .054), narrowly**. Executable at the book (DK open price):
  FAIL (-1.4%). Kill rule (no-vig CI upper < 0 or CLV CI upper <= 0) NOT triggered: CLV is large and clean in both seasons.
  k = 3's p .0005 is one of three descriptive cuts (forking path); not a pass.
- Same story as CFB/NBA: the model leads the opener (CLV +0.83 pts, both seasons, balanced over/under), the vig eats it.
- Rows: no duplicate eids; early (Nov-Dec) vs later: CLV +0.95 vs +0.71, no-vig +2.1% vs +4.2% (descriptive).

## Verdict
- CBB totals model: leads the DK opener (CLV +0.83 [+0.69,+0.97], n 2,213) but does not make money at the open price
  (dev no-vig +3.2% p .054 = pre-reg FAIL by a hair; at DK price -1.4%).
- Kalshi (2026-27, n 67): ROI +16.3% [-6.0,+37.3] but CLV +0.04 [-0.55,+0.66] -> luck-shaped; pre-reg says GO forward
  (CLV point > 0, n >= 30) only just. Structural problem: buying the 50c rung costs 0.541 (ask + fee), dearer than DK -110
  (0.524). At the dev win rate .522 the expected Kalshi taker ROI is about .522/.541 - 1 = -3.5%. Do not expect +16% to hold.
- 2026-27 data so far: 685 D-I finals (11-02..11-12), 490 with DK lines, 241 Kalshi total ladders (220 priceable).

## Forward (paper, no orders) + proposal
- cbb_t1_kalshi_taker stays forward:unwired (edgelab). Rerun after each day settles: extend SEASONS[2027] end in
  fetch.py -> `fetch.py games 2027` -> `fetch.py odds 2027` -> `model.py 2027`; in results/kalshi_openers_xleague
  `KX_LO=<d> KX_HI=<d> KX_TAG=_cbbom q1_fetch.py cbb; KX_TAG=_cbbom q1_timing.py cbb` (the _cbbom tag is overwritten per
  run: use a per-date tag and add it to kalshi.py's tag list) -> `kalshi.py`. ~7 bets/day -> 100 in ~2 weeks.
- PROPOSAL (fresh prereg needed, not run): MAKER at the 50c rung (rest a bid ~0.49-0.50 on our side when |gap| >= 4).
  At .522 win and no taker fee that is ~+5%, but E5/Q003 show maker fills are adversely selected -- test with a trade-based
  fill model first. Also possible: a box-score possessions x efficiency (KenPom-style) model, ~6k summary calls/season.

## Learnings

### CBB opener models lead the DK opener by ~0.8 pts but lose to the vig
De-biased scores-only log-points ridge, |adj| >= 4: CLV +0.83 [+0.69,+0.97] pts on 2,213 dev bets (2023-25), win .522,
no-vig ROI +3.2% p .054, at DK price -1.4%. Source results/cbb_opener_model/dev.json. Same pattern as NBA/CFB/NHL.

### Kalshi's 50c rung costs more than a -110 book bet
CBB totals at Kalshi first tight: mean ask + taker fee on the bought rung 0.541 (n 67) vs 0.524 implied by -110.
Source results/cbb_opener_model/kalshi.json. Matters: Kalshi sitting on the opener does not make it cheaper than the book.

### ESPN core API has multi-book CBB openers back to 2023-24
DraftKings (and ~12 other providers) total open/close + prices for ~5.3k of ~6.2k D-I games per season; 2026-27 names the
provider "Draft Kings" (with a space). Source results/cbb_opener_model/odds_2024.parquet, odds_2027.parquet.

### A log-points model under-predicts CBB totals by 1-3 pts
exp(mu+o+d) is a geometric-mean predictor and ignores OT; walk-forward bias vs final -1.1 to -3.2 pts per season (largest
early season). Source model.py output. The trailing de-bias c absorbs it; never bet the raw gap.
