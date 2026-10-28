# NBA live totals vs Kalshi KXNBATOTAL in-game ladders (2026-10-27, DEV ONLY)

Script: `scripts/nba_live_totals.py` (build -> fit -> kalshi -> dev). The pre-registration is in its docstring. It was
written before any KXNBATOTAL price was pulled. Ledger dataset `nba_live_totals`: 3 trials, bar p < 0.0167.
Results are in `results/nba_live_totals_dev.json`. No orders were placed and nothing was committed.

## Verdict
**Kalshi's live NBA totals are efficient against this model. 0 of 3 trials pass, and the model is worse than the
market.**
- T1 (calibration): Brier score, Kalshi .1196 vs model .1240. The model is worse in both halves and at every
  checkpoint (p for "model better" = .999).
- In a regression of the outcome on both prices (logit(Kalshi mid) + logit(model), standard errors clustered by
  game), the model's coefficient is **0.02 ± 0.21** and Kalshi's is **1.03 ± 0.21**. The ladder already holds what
  the model knows.
- T2 (rule, edge >= 5c): 336 bets in 239 games, **-2.1% ROI** (p .64). The halves disagree: +2.7% then -4.6%.
- T3 (Q4 close games only): 46 bets, **-41.5% ROI** (p .98). On the bets it picked, the model said 38% and the price
  said 25%. Those bets won 15%. In close late games, where the model disagrees with the market, the model is the one
  that is wrong.
- **Nothing goes into `lb`.**

## Why: the game's own pace carries almost no information
Model, fitted per checkpoint on 2015-16..2023-24 (11.4k games; Amendment 1 drops ~12 games with typo lines in
OddsData):

  remaining = b1 * minutes_left * (pregame total / 48) + b2 * minutes_left * w * (pace - pregame rate)

The weight w = e/(e+24) was fixed in advance. The spread around the mean comes from actual residuals, grouped by
|margin|.

Fitted b2 (how much of the game's pace so far carries forward; 1 would mean all of it, 0 none):

| checkpoint | Q1 end | half | Q3 end | Q4 6:00 | Q4 3:00 |
|---|---|---|---|---|---|
| b2 | 0.05 | 0.03 | 0.09 | 0.06 | -0.11 |

So a hot first half barely predicts the second half. Scoring above the pregame line is mostly noise, and the rest of
the game reverts to the pregame per-minute rate.

On the 2024-25 holdout, log loss of P(final > L):

| checkpoint | full model | no pace | history (ignores pregame total) |
|---|---|---|---|
| Q1 end | .6101 | .6095 | .6845 |
| half | .5799 | .5797 | .6364 |
| Q3 end | .4977 | .4980 | .5113 |
| Q4 6:00 | .4265 | .4266 | .4332 |
| Q4 3:00 | .3530 | .3529 | .3711 |

- The useful inputs are the pregame total, time left and |margin|.
- Pace adds nothing.
- Margin works through the residuals. At Q4 3:00 the mean extra points are +3.0 when the margin is 0-3, and -2.8 in
  blowouts of 16+. This is fouling plus OT on one side and garbage time on the other.

Holdout calibration is good. Most reliability bins are within about 2-3 points, and the PIT deciles are roughly flat.
So the model is a sound statistical table, and Kalshi still prices it in.

The user's worry ("history ignored game pace") was reasonable to check, but pace does not matter. The part of
"history" that does matter is the pregame total: per-minute scoring is anchored to it.

## The POR@LAL example (one game; descriptive only)
State: tied 121-121, Q4 3:49, total 242, pregame 233.5.
- Using the Q4_3 residuals the model gives P(over 265.5 / 267.5 / 269.5) = .48 / .38 / .30. Using the Q4_6
  residuals it gives .30 / .21 / .15.
- Interpolating to 3:49 gives about .38 / .29 / .22. That is close to the user's history numbers and to Kalshi's
  .345 / .265 / .25.
- The prices were fair, with perhaps a slightly rich top rung. Those few cents are smaller than the model's own
  uncertainty between checkpoints.

## Data
- **Kalshi coverage.** 590 of 603 dev games matched a KXNBATOTAL event. There are 11 fixed strikes per game, set
  pregame. I pulled 6,490 markets (816k 1-min candles).
- **Snapshots.** 30,504 snapshot rows in total. Only **3,675 qualify** (two-sided quote, spread <= 4c), from 492
  games and 1,363 game-checkpoints. Most rungs of the in-game ladder are one-sided or wide.
- **Alignment.** Median |Kalshi mid - model| is .026, so the timing alignment is sound.
- **Settlement.** Kalshi's result matches "final total incl. OT > strike" in 99.95% of rows.
- **Pregame total.** For dev games: ESPN pickcenter overUnder (351 games), else OddsData OU. The two sources differ by
  a median of 1.0 point.
- **Sealed data.** Candles dated >= 2026-01-15 were never pulled. The market list was filtered before any candle
  call, and 2025-26 pbp was cut at 01-14 when loaded.

Kalshi calibration on the qualifying rows (descriptive; not a trial):

| mid bin | n | games | mid | realised | model |
|---|---|---|---|---|---|
| 0-5c | 704 | 188 | .031 | .026 | .032 |
| 5-15 | 544 | 175 | .088 | .066 | .104 |
| 15-30 | 245 | 133 | .222 | .171 | .245 |
| 30-45 | 375 | 223 | .390 | .347 | .394 |
| 45-55 | 443 | 265 | .500 | .438 | .516 |
| 55-70 | 334 | 203 | .618 | .617 | .625 |
| 70-85 | 199 | 123 | .783 | .749 | .773 |
| 85-95 | 392 | 171 | .914 | .934 | .905 |
| 95-100 | 439 | 194 | .973 | .961 | .974 |

Overs realised 2-6 points below the mid in the 5-55c bins. This is one dev window, rows within a game are
correlated, and it was not pre-registered. It is a lead to note, not a rule.

T2 broken down after the fact:
- By side: NO bets made +6.8u on 132 bets; YES bets lost -9.6u on 204.
- By checkpoint: Q1 end +5.5u on 29 bets; half -6.9u on 130.

These are noise-level splits. Do not mine them.

## Caveats
- **Price.** Each snapshot uses the close ask of the first candle 60-360 s after the checkpoint. Candles are sparse,
  and there is no queue or depth modelling.
- **Fees.** The fee is 0.07p(1-p), continuous and not rounded.
- **Off-checkpoint states.** States that fall between checkpoints use rescaled residuals from the checkpoint. The
  POR@LAL example shows that this rescaling is rough away from the checkpoint.
- **Not tested.** Sealed data (2026-01-15 onward) was not scored, and there is no reason to ask for it.
