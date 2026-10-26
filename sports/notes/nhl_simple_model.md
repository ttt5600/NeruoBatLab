# NHL simple model (team rating + starting goalie + rest + style) vs the OPENING line (2026-10-25, DEV ONLY)

- Script: `scripts/nhl_simple_model.py` (`build` / `backtest [--ledger]`). The docstring is the pre-registration. It was
  written before any model-vs-market join. One amendment (A1, below) was made after the accuracy-only `build` and
  before any market join.
- Ledger: dataset `nhl_simple_model`, 8 trials, logged 2026-10-25. Dataset bar 0.05/8 = 0.00625. Whole-ledger bar
  0.05/255 ≈ 0.0002.
- Data: DEV only. Features use 2021-22 (warm-up) through 2024-25. Trials are scored on 2023-24 and 2024-25: 2,556
  games, of which 2,522 have an ESPN BET open line.
- NHL 2025-26 is sealed. `nhl_games.parquet` is read with a pyarrow `date < 2025-08-01` filter, and the script asserts it.
- Outputs:
  - `results/nhl_simple_model.json`
  - `data/nhl_simple_model/wf_games.pkl` (walk-forward predictions)
  - `data/nhl_simple_model/stack_fits.json`

## Verdict (plain language)
1. **Same shape as NBA: the model clearly LEADS the opener, but here it does not pay.** No money trial passes.
2. **Moneyline:**
   - CLV is real. The line moves +1.04 probability points toward the model (T1). That is 1,360 bets, CI
     [+0.90, +1.18], p < 1/4000, and both seasons agree (+0.97 / +1.11).
   - It is not a blind drift. Dogs drift −0.24 pts blind, and in games the model does not bet the dog side drifts
     −0.65.
   - The stacked logit given the OPEN gives the model +0.46 [+0.04, +0.89]. Given the CLOSE it is +0.14 [−0.31, +0.60].
     **The close already contains the model.**
   - Money does not pass:
     - No-vig ROI +2.9% (p .18).
     - At the offered opening price −1.1% (T3).
     - Close-implied EV at the offered open is **−1.8%** (SE 0.16%, both seasons negative).
   - The cause: ESPN BET's ML opener has ~4.0% overround, and +1 pt of CLV does not clear it.
3. **Totals:**
   - CLV is real. The market's implied total moves +0.073 goals toward the model (T4). That is 910 bets, CI
     [+0.063, +0.082], p < 1/4000, both seasons.
   - No-vig ROI is **+7.1% [+0.8, +14.0], p .0125**. It fails the dataset bar (.00625). Both seasons are positive
     (+6.2% / +8.6%).
   - At the offered price it is +2.6% (p .20). Close-implied EV at the offered open is −1.9%.
   - **Caveat: much of T5 is a blind-under lean.** 76% of the bets are unders. Blind unders at the opening 6.5 make
     +5.8% no-vig (1,448 games). Model unders at 6.5 make +8.5%.
4. **Goalie-change games (T7/T8):**
   - CLV +1.23 pts (p < 1/4000) vs +0.98 in no-change games.
   - No-vig ROI +5.6%, p .075, which fails.
   - My pre-registered "change" definition (starter differs from the last-10 modal starter) fires in **66% of games**,
     because NHL teams rotate tandems. So T7/T8 is not a clean "late goalie news" test.
   - Descriptive check with a narrower definition: games where tonight's starter made ≤ 2 of the team's last 10
     starts (30% of games). CLV there is +1.28 vs +1.09 elsewhere, and no-vig ROI is +3.1% vs +3.6%.
   - **So the openers do not visibly mis-price backup starts more than other games.**
5. **No longshot trap in the CLV.**
   - ML bets are 72% dogs. Dog CLV is +0.77 and fav CLV +1.72, so both are positive.
   - The money is worse on dogs: −2.0% at the offered price vs favourites +1.5%.
6. **Bottom line:** DEAD as a bet at ESPN BET opening prices (−1.8% close-implied EV).
   - Like NBA spreads, NFL (L33) and O1, this is a "model predicts the line move" signal.
   - It would only become money at a book whose opener has much lower vig, or as maker/no-vig on Kalshi if Kalshi
     lists NHL early. That last point is UNVERIFIED.
   - Not proposing a sealed test. The only thing close (T5 totals no-vig) is half a blind under lean, and it fails at
     the offered price.

## Step 0: which openers exist (dev seasons)
| Season | Openers with prices in our data |
|---|---|
| 2021-22, 2022-23 | NONE. ESPN pickcenter `*_open` is empty for every book. |
| 2023-24 | ESPN BET 1,363 games ML / 1,370 total. Bet365 1,303. DraftKings 1,265 ML but only 127 totals. |
| 2024-25 | ESPN BET only, 1,440 games (ML, puck line and total, open and close, with prices). |

ESPN BET is the only book with openers in both seasons, so it is the only opener used. No external download was
needed. For older seasons, the sportsbookreviewsonline NHL archives have open and close; I did not fetch them.
Opening totals are 6.5 (60%) and 5.5 (40%), with juiced prices. The line moves open→close in only 14% of games, so
totals CLV is measured as a shift in the implied mean (below), not in line points.

## 1. Model (as fit; walk-forward by season)
Unit = team-game. Target = regulation goals (final score minus the OT/SO winner's goal; `nhl_games.overtime`).

| Feature | Coef fit 22-23 → scores 23-24 | Coef fit 22-24 → scores 24-25 |
|---|---|---|
| const | +1.17 (.48) | +1.41 (.34) |
| r_gf: ridge goals (att/def/home, half-life 90 d, λ 10) | +0.62 (.15) | +0.54 (.11) |
| g_opp: opposing starter GSAA/shot × 30, shrunk with 1,500 pseudo-shots, half-life 365 d | −0.75 (.34) | −0.48 (.25) |
| b2b team | −0.25 (.09) | −0.23 (.07) |
| b2b opponent | +0.23 (.09) | +0.20 (.07) |
| S1 shots for + opponent shots against (z) | **+0.15 (.04)** | **+0.16 (.02)** |
| S2 PIM drawn + opponent PIM taken (PP proxy, z) | −0.04 (.02) | −0.01 (.02) |
| S3 own shooting % (z) | −0.05 (.04) | −0.01 (.03) |

- λ feeds independent Poisson. P(home) = P(H > A) + ½P(H = A).
- Total = H + A + 1{tie}, because OT/SO adds one goal to the final.
- Residual SD is 1.67-1.69 goals per team.
- **What matters:**
  - The team rating.
  - Back-to-back, about −0.24 goals: the tired team scores less, and its opponent scores about 0.2 more.
  - Shot volume beyond the ridge.
  - The goalie, about −0.5 to −0.75 goals per goal-per-30-shots of GSAA.
- The PP proxy and shooting-% luck are ≈ 0.
- Shooting-% regression does NOT replicate NBA's 3P% effect.
- Starter matching was 100%: ESPN goalie names matched to `nhl_games` starter names by last name within the team.

Accuracy (2,447 games with all inputs):

| Source | Log loss |
|---|---|
| Model STRICT | 0.6617 |
| Model ACT (actual starter) | 0.6621 |
| ESPN BET open | 0.6594 |
| ESPN BET close | 0.6566 |

The model is worse than the opener on its own, yet adds information to it (stack +0.46).

## 2. Trials (bar p < .00625; one-sided game bootstrap, 4,000 resamples; ESPN BET opener)
| Trial | Bets | Effect per bet | 95% CI | p | Per season (24 / 25) | Pass |
|---|---|---|---|---|---|---|
| T1 ML STRICT CLV (prob pts), gap ≥ .03 | 1,360 | +1.04 pts | [+0.90, +1.18] | <.00025 | +0.97 / +1.11 | **yes** |
| T2 same, no-vig ROI | 1,360 | +2.9% (46.3% win) | [−3.5, +9.1] | .18 | +2.6 / +3.2 | no |
| T3 same, offered open price | 1,360 | −1.1% | [−7.2, +5.0] | .62 | −1.3 / −0.7 | no |
| T4 totals STRICT CLV (implied goals), gap ≥ .04 | 910 | +0.073 g | [+0.063, +0.082] | <.00025 | +0.060 / +0.092 | **yes** |
| T5 same, no-vig ROI | 910 | +7.1% (52.7% win) | [+0.8, +14.0] | .0125 | +6.2 / +8.6 | no (dataset bar) |
| T6 same, offered open price | 910 | +2.6% | [−3.4, +9.2] | .20 | +1.7 / +4.1 | no |
| T7 ML ACT, goalie-change games, CLV | 938 | +1.23 pts | [+1.05, +1.41] | <.00025 | +1.07 / +1.41 | **yes** (lookahead) |
| T8 same, no-vig ROI | 938 | +5.6% | [−2.1, +13.3] | .075 | +2.6 / +8.9 | no |

- The CLV passes have z ≈ 14 (T1) and ≈ 15 (T4), so they also clear the whole-ledger bar in substance. 1/4000 is
  only the bootstrap's floor.
- Fav/dog splits (T1 / T2 / T3):

  | Side | Bets | CLV | No-vig | Offered |
  |---|---|---|---|---|
  | Favourites | 376 | +1.72 pts | +5.6% | +1.5% |
  | Dogs | 984 | +0.77 pts | +1.9% | −2.0% |

- Over/under splits (T5):

  | Side | Bets | No-vig |
  |---|---|---|
  | Overs | 218 | +4.4% |
  | Unders | 692 | +8.0% |

## 3. Descriptive (not trials)
- **Dose:**
  - ML CLV rises with the gap: +0.92 (≥ .02), +1.04 (≥ .03), +1.27 (≥ .05), +1.78 pts (≥ .07).
  - Offered-price ROI goes −2.2%, −1.1%, +2.3%, +3.0% over the same gaps, with the dog share rising to 81%.
  - Totals CLV also rises: +0.058 → +0.104 goals from gap ≥ .02 to ≥ .08. No-vig ROI is not monotone (+5.6%, +7.1%,
    +4.6%, +15.8% on 224 bets).
  - Do not cherry-pick a threshold: these are post-hoc.
- **Close-implied EV at the offered open** (fair close probability × open decimal − 1): ML −1.8%, totals −1.9%, both
  seasons negative. This is the cleanest money read. The opener is beaten on direction but not by enough to pay a
  ~4% overround.
- **Blind controls:**
  - Home side drift −0.32 pts.
  - Dog side −0.24.
  - Under +0.008 goals.
  - The model's CLV is not a blind drift.
- **Blind under at the opener:** at 6.5, +5.8% no-vig (1,448). At 5.5, +0.3% (972). Opening 6.5 overs look slightly
  overpriced on dev. That is a separate, post-hoc lead, not tested.
- **ACT vs STRICT:** knowing the actual starter adds almost nothing. ML CLV +1.15 vs +1.04; log loss is no better.

## 4. Caveats
- One opener book (ESPN BET). Its open timestamp is undocumented, and ESPN BET ended in 2025-12. So a forward version
  would need a different book's first number.
- **Amendment A1** (after `build`, before any market join): the first build under-predicted 2024-25 totals by about
  0.4 goals, because league shots per game fell (31.3 → 28.4) and the style z-scores used fixed training means. The fix
  expresses each style feature relative to the league's decayed mean. Nothing else changed.
- A descriptive "rare starter" flag was added after build 1 and before the market join, because the pre-registered
  goalie-change definition fires in 66% of games.
- 65 of the 2,522 games with openers lack model inputs and are excluded.
- GSAA here is save% vs league average. There is no shot quality, because we have no xG or shot-location data. A real
  GSAx might do better, but the goalie term is already small relative to the market (ACT ≈ STRICT).
- The first `backtest --ledger` run crashed in the descriptive section, after the trials printed and before any ledger
  write. The rerun gave identical trial numbers, and the ledger has exactly 8 rows.
- Nothing committed. No orders.
