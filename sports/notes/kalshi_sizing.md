# Confidence-based (Kelly) sizing for E1/E2, 2026-10-27

Script `scripts/kalshi_sizing.py` (build | calib | backtest | fit | sealed-descriptive). Data: the 1,713 DEV picks
(`data/kalshi_temp_maker_picks.parquet`, 70 losses, 2024-03..2026-02; E1 1,269 + X1 444), with features recomputed from
the frozen rule's own forecasts (`data/kalshi_sizing_dev.parquet`). Backtest table: `data/kalshi_sizing_backtest.json`.
Frozen params: `data/kalshi_sizing_params.json`. Nothing deployed, no orders, executor untouched.

## Verdict (read this first)

**Do not switch to confidence-based sizing. It does not help on dev, and the data cannot tell one pick's confidence
from another's beyond what the price already says.**

1. **Confidence does vary, but the price already prices it in.** Calibrated P(NO wins) runs from about 82% to 99.4%
   across picks. Almost all of that spread is the market price. Once the price is known, the other decision-time
   features add nothing measurable: distance of the nearest model to the bracket (logit coefficient -0.18, se 0.15),
   model spread, side, and cohort all land inside each other's noise. The one stable fact is that **YES wins about
   0.69x as often as the NO ask implies** (dev 70 losses vs 101.8 implied; the sealed check, frozen first and
   descriptive only, gives 0.665).
2. **Under that model Kelly gives every pick about the same stake.** Our edge is (1 - 0.69) x the market's implied risk,
   and the downside is also proportional to that risk, so the ratio (p - cost)/(1 - cost) comes out nearly constant.
   At the 80% lower bound it is 4.4% of bankroll per pick at every price from 80c to 99c (quarter Kelly).
   "99.9% vs 99%" does not show up as a difference between picks.
3. **The bins that look like 99.9% are noise.** Dev loss rates per NO-ask bin are 6.2 per 1,000 at 99c
   (95% Wilson [2.1, 18.2]; break-even 9.3) and 12.5 at 98c ([4.9, 31.8]; break-even 18.6). In the sealed set the
   99c bin came out at 2.9 per 1,000 and the 95-97c bin at about break-even, which is the opposite ordering from dev
   (dev 95-97c was the *best* bin). Bin-level confidence flips between samples, so we cannot size on it.
4. **Skipping did not find bad picks.** On the walk-forward, the primary rule skipped 20.5% of picks. Every skip fell in
   Jan-Feb 2025, when the 2024-only fit was too uncertain, and after that it bet every pick. The skipped picks earned
   +4.4% per unit against +2.0% for the ones kept. Flat sizing with the skip rule lost to plain flat (log-growth
   -0.11, paired night-bootstrap 95% [-0.19, -0.03]).
5. **Primary Kelly vs flat on dev: no better.** Bankroll multiple was 1.25x vs 1.32x (log-growth difference -0.056,
   95% [-0.18, +0.06]). The variants that did beat flat did it by **betting more on every pick**, not by betting
   smarter: the point-estimate fit at quarter Kelly made 1.55x, staking 8.2% of bankroll per night vs 4.5%, with max
   drawdown 7.6% vs 3.3%.

**What the analysis does say:** this is a "how big is the uniform unit" question. Even the conservative quarter-Kelly
number (about 4.4% per pick, which the 10%/night cap then limits) is far above today's 0.78% orig7 / 0.5% x1 tiers.
If you want more money working, raise the flat tier step by step and keep the night cap. Building a per-pick
confidence model is not the way to do it. The bigger lever is still the maker execution (E4, sealed pass), which
earns more on every pick.

## Kelly in plain words (for the learner)

A NO contract costs c (price + fee) and pays $1 if NO wins (probability p). Bet a fraction f of your bankroll:
win -> bankroll x (1 + f(1-c)/c), lose -> x (1 - f). The f that maximises long-run growth is **f\* = (p - c)/(1 - c)**:
your edge (p - c) divided by what you stand to lose per contract relative to the payout scale.

- **Why near-certain bets get huge Kelly numbers.** At c = 0.9907 (99c + fee), if p were really 0.999, f* = 0.0083/0.0093
  = 89% of bankroll. If p is 0.990, f* is *negative*: don't bet. The whole answer hangs on the third decimal of p.
- **Why you can't see the third decimal.** To tell 1 loss per 1,000 from 6 per 1,000 you need thousands of picks. Rule
  of three: with zero losses in n bets, the 95% upper bound on the loss rate is still about 3/n. So **40-0 live is
  consistent with p = 99%** (0.99^40 = 67%), and it takes about 3,000 straight wins to *certify* 99.9%.
- **So we shrink first, then fractionate (protocol rule 7).** Plug in a *lower* confidence bound for p (p_lo, 80%
  one-sided), then bet k = 1/4 of the Kelly number from that. Quarter Kelly keeps about 3/4 of the growth for 1/16 of the
  variance, and it protects you when p is overestimated. Overbetting with full Kelly on a p that is too high is the
  classic way to lose money while holding a real edge.
- **Correlation.** All cities on one night share the weather pattern, so a bad night can hit several picks at once.
  That is why the night total is capped (10%), separately from the per-pick cap (5%).

## Method

- Calibrators (form chosen after looking at dev loss-rate tables, so the walk-forward is time-honest in its *fits*,
  not in its *form choice*):
  - **ratio (PRIMARY):** P(loss) = theta x yes_bid. theta upper bound from the exact Poisson interval.
  - **logit:** loss ~ logit(yes_bid) + gap, with the delta method for p_lo.
  - **bins:** NO-ask bins with a Wilson lower bound.
- Walk-forward: each month is fit only on picks dated before (month start - 1 day). The eval window is 2025-01..2026-02,
  1,237 picks, 224 nights, and every strategy is scored on the same window.
- Sizing: f = k (p_lo - q_eff)/(1 - q_eff), set to 0 if p_lo <= q_eff, capped at 5% per pick. If a night's total goes
  over 10%, every pick that night is scaled down to fit. q_eff for taker = NO ask + 0.07a(1-a). For maker, q_eff = the
  M3 resting price with the fill tape; the maker calibrator is fit on FILLED picks (adverse selection: theta 0.86).
- Flat = the executor's TIER_FRAC (E1 1%, X1 0.5%) of current bankroll.
- Risk: 2,000 night-bootstrap paths. Paired comparisons: bootstrap of the nightly log-growth difference.

## Calibration in the 98-100% region (out-of-fold, ratio, eval 2025+)

| predicted P(NO) bin | n | losses | per 1,000 [95% Wilson] | predicted | taker break-even |
|---|---|---|---|---|---|
| 0.97-0.98 | 237 | 1 | 4.2 [0.7, 23.5] | 23.4 | 30.2 |
| 0.98-0.99 | 249 | 4 | 16.1 [6.3, 40.6] | 15.0 | 18.6 |
| 0.99-0.995 | 361 | 2 | 5.5 [1.5, 20.0] | 7.7 | 9.3 |

- No pick reaches a ratio-model p above 99.5%. The logit puts 30 picks at 99.5-99.9%; they had 0 losses, but the CI is
  [0, 113] per 1,000, so that tells us nothing.
- OOF log-loss: ratio 0.1054, logit 0.1050, bins 0.1112, market price alone 0.1098. The models beat the price only by
  the uniform theta shrink.

## Backtest (dev walk-forward, eval 2025-01..2026-02; `kalshi_sizing.py backtest`)

| strategy | bets (skip) | avg night staked | multiple | max DD | boot P(DD>20%) | vs flat dlog [95%] |
|---|---|---|---|---|---|---|
| flat taker (today's tiers) | 1237 (0%) | 4.5% | 1.32 | 3.3% | 0 | - |
| **PRIMARY kelly ratio L.80 k.25** | 983 (20.5%) | 5.3% | 1.25 | 2.9% | 0 | -0.056 [-0.18, +0.06] |
| skip-only (flat size, skip p_lo<=cost) | 983 | 3.4% | 1.18 | 1.4% | 0 | -0.110 [-0.19, -0.03] |
| kelly k .10 / .50 | 983 | 3.6% / 5.9% | 1.16 / 1.30 | 2.1% / 4.9% | 0 / .0005 | - |
| kelly L.50 (point theta) | 1136 (8%) | 8.2% | 1.55 | 7.6% | .0005 | +0.159 [+0.003, +0.31] |
| kelly L.90 / L.95 | 834 / 636 | 4.0% / 2.4% | 1.18 / 1.09 | 2.3% / 2.4% | 0 | - |
| kelly logit / bins | 561 / 233 | 3.4% / 3.4% | 1.26 / 1.59 | 3.4% / 7.1% | .0005 / .025 | -0.04 / +0.19 [-0.12, +0.49] |
| flat maker (M3, filled only) | 916 | 3.3% | 1.41 | 3.0% | 0 | - |
| kelly ratio maker | 819 | 7.7% | 1.65 | 6.2% | .0005 | +0.154 [-0.02, +0.31] vs flat maker |

- Read these as about 15 sensitivity rows, not trials. Nothing here is a pass/fail test. Bootstrap P(final < 1) is
  2% or less in every row.
- The kelly rows beat flat only where they stake more per night.
- The bins row is the only calibrator whose skips looked selective: kept picks +4.8% per unit, skipped +1.9%.
  But it bets just 233 of 1,237 picks, its interval crosses zero, and the bin ordering it relies on flipped in the
  sealed set. Do not select it.

## Sealed descriptive check (frozen params first, already-scored sealed picks, NOT a trial, not used for any choice)

The frozen ratio params were applied to the 4,308 sealed picks (106 losses):

- theta was 0.665 on sealed vs 0.687 on dev.
- Primary Kelly skipped none. It made 1.57x with 10.8% max drawdown, vs flat 1.37x with 4.2%.
- Per-bin loss rates did not keep the dev ordering (see verdict point 3).

## Proposed executor change (NOT applied; ~/.claude/quant-kalshi untouched). Recommendation: leave it OFF.

`.env` (all optional, default off):
```
KALSHI_SIZING=flat            # flat (default, today) | kelly
KALSHI_KELLY_K=0.25
KALSHI_KELLY_LEVEL=0.80
KALSHI_KELLY_PICK_CAP=0.05
KALSHI_SIZING_PARAMS=data/kalshi_sizing_params.json   # copy from the worktree; frozen 2026-10-27
```

With kelly on and B = $400, sizes are about $17.7 taker or about $20 maker per pick. Those only take effect if
KALSHI_MAX_ORDER_USD / MAX_DAILY / MAX_OPEN are also raised (today $3.10 / $30 / $40 bind first). NIGHT_FRAC 0.10
stays as the night cap.

`scripts/kalshi_exec.py` patch:
1. `config()` reads the keys above. `sizing` must be `flat` or `kelly`; anything else raises, like `_style`. If kelly
   is set and the params file is missing, refuse to trade.
2. `caps(cfg, cohort, pick=None, style="taker", maker_q=None)`: when `cfg["sizing"] == "kelly"` and a bankroll is set,
   `order = min(cfg["max_order"], kalshi_sizing.size_pick(pick, params[...], b, k=..., level=..., pick_cap=...,
   style=style, maker_q=maker_q)["frac"] * b)`. Use `calibrator_maker` for maker, `calibrator` for taker. Otherwise
   keep TIER_FRAC.
3. `decide()` / `_decide_maker()`: pass the pick and price into `caps`. If size_pick returns skip, record
   `status: "skip_kelly_edge"` along with p_hat, p_lo and q_eff on the journal row.
4. Log `sizing`, `p_lo` and `frac` on every order row, so live picks can be checked against the calibration later.
5. Tests: flat path unchanged (golden); kelly order size; skip path; maker uses calibrator_maker; a missing params file
   refuses to trade.

## Caveats
- Depth and local hour were not available in the dev data. The decision is always about 22:00 local, and depth was not
  logged historically, so neither could be tested. Live picks now log `depth_1c`, so depth can be tested forward.
- Mix shift: 66% of live paper picks are at 99c, vs 28% of dev. The uniform-theta assumption is least tested exactly
  there (dev: 3 losses in 480).
- Within-night correlation is handled only by the night cap. The bootstrap resamples whole nights, so it keeps
  within-night correlation but not streaks across nights.
