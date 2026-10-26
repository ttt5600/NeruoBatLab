# NBA simple model: team rating + players + playing style, vs the OPEN and CLOSE (2026-10-25)

User ask (10-25): "Kalshi follows Pinnacle. Find the edge with a SIMPLE model that uses the betting rates and stats on
each team, does simple analysis to work out playing style, and looks at the players and their performances. Simpler
might be better. Top teams use xgboost, which might not be that useful."

- Script: `scripts/nba_simple_model.py` (`build` / `backtest [--ledger]` / `today [--date] [--refresh] [--out "A,B"]`).
  The rules were pre-registered in its docstring before any market join. One amendment came after the accuracy-only
  `build` and before any market join (see section 5).
- Ledger dataset: `nba_simple_model`, 8 trials, all logged 2026-10-25. The dataset bar is p < 0.00625. The
  whole-ledger bar is 0.05/213 ≈ 0.00024.
- Data: DEV only. Players and teams use 2021-22 (warm-up) through 2024-25. The scored games are 2023-24 and 2024-25:
  2,295 regular-season games, 2,286 of them with an opener.
- The 2025-26 season and `nba_ingame.parquet` were not read by build or backtest.
- Outputs:
  - `results/nba_simple_model.json`: every trial and descriptive result.
  - `results/nba_simple_model_frozen.json`: the frozen coefficients that `today` uses.
  - `data/nba_simple_model/wf.pkl`: the walk-forward predictions.
- No xgboost, nothing opaque. The model is one small linear regression for player value, two ridge team ratings, and
  two OLS "stacks" with 9 coefficients each.

## Verdict (plain language)
1. **Simple wins again, but only against the OPENER. Nothing beats the CLOSE.**
   - Given the closing line, every version of the model gets a stacking weight of about 0 (−0.11 to +0.08, all CIs
     straddle 0).
   - Bet against the close, it loses on spreads (T6: −1.2%, 49.4% ATS) and does not pass on totals (T7: +3.2%,
     p .14).
   - So the user's premise holds: the closing price (Pinnacle, and Kalshi following it) already contains everything
     this model knows.
2. **Spreads: the model leads the opener, but it is not money even at no-vig.**
   - The line moves **+0.79 points toward the model** (T1, 1,262 bets, p < 1/4000, positive in both seasons).
   - The bets still cover only 52.1%: +2.3% at no-vig (T2, p .21) and −0.6% at −110 (T3).
   - This is O1 again. It is the same "the model predicts where the line goes" signal as player_sim. On spreads it is
     too small to beat the result noise.
3. **Totals are the surprise, and the best number in this study.**
   - Pre-registered T5 bets the total when the model is at least 3 points off the opening total.
   - **55.5% O/U on 1,190 bets: +8.7% at the no-vig opening price, CI [+3.1%, +14.3%], p .001. Both seasons are
     positive (+8.5% and +9.0%), and both opener books are positive (DraftKings +10.3% on 620, ESPN BET +7.0% on 570).**
   - At −110 the same bets make +5.9% (post-hoc, not a trial).
   - The total moves +2.0 points toward the model from open to close (T4, p < 1/4000).
   - It **passes the dataset bar but NOT the whole-ledger bar** (.001 > .00024).
   - Graded against the CLOSING total, the same bets win only 52.4%. The edge is in the opening number being stale,
     not in out-predicting the market.
   - Read it as a strong lead, not a proven edge (see the traps).
4. **Players and availability help, and openers do err on absences** (this matches the literature).
   - "Actual availability" means who really played, a lookahead proxy for the injury report at tip.
   - Its spread CLV vs the opener is **+1.75 points in games with absence news** (a top-5-minutes player who played
     the previous game sits tonight). That is T8, 564 bets, both seasons +1.6 / +1.9. In no-news games it is +1.03.
   - The pure no-lookahead model (who played last game) gets +0.73 in news games and +0.84 without. So the extra
     CLV comes only from knowing tonight's absences.
   - This is an **upper bound**: much of that news breaks after the open, and you would only see the moved price.
     The bettable version needs real-time injury news faster than the book.
5. **Playing style adds a little, and it is the same two things the comebacks study found.**
   - Paint defence on sides (L30), and 3P% regressing in totals (L31). Those are the only style coefficients clearly
     away from 0 (table below).
   - The style part of the model correlates +0.05 with the result vs the open, and +0.03 to +0.04 vs the close.
   - Most of the totals lift comes from these style terms. T1 (no style) makes +2.6% no-vig; T2 (with style)
     makes +8.7%.
   - The other three style matchups, rebounding, turnovers and threes, are noise.
6. **Longshot trap: avoided by construction.** Spreads and totals are priced near −110, so the gap rule cannot drift
   into +240 underdogs the way O1's moneyline rule did. Favourite vs dog splits:
   - T1 CLV: fav +0.84, dog +0.77.
   - T5 P&L: over +6.9% (932 bets), under +15.3% (258 bets).
   - T2 P&L: fav +9.3%, dog −0.8%. This split is noisy.

## 1. The model (each piece in one line)
| Piece | What it is | Frozen setting |
|---|---|---|
| M0 team rating | O1's ridge: home margin = home edge + team rating diff + b2b/rest/travel | half-life 60 d, λ 5 |
| T0 team totals | ridge: team points = league + offence[team] + defence-allowed[opp] + home | same |
| BPM-lite player value | on-court +/- per 48 regressed on per-48 box stats; prior seasons only | ≥300-min player-seasons, minutes-weighted |
| Player form | decayed box rates (half-life 180 d, so last season is the prior), shrunk to replacement with 500 pseudo-minutes | |
| Team strength tonight | Σ projected minutes × player value / 240; minutes = EWMA(0.3), rescaled to 240, cap 42 | |
| Availability | STRICT = played the team's last game (no lookahead); ACTUAL = played tonight (lookahead proxy) | |
| Style | decayed (60 d) pace, 3PA rate, FT rate, ORB%/DRB%, TOV%, 3P%, paint share, each own and allowed | |
| Stack | OLS: M1 = M0 + player strength + absence; M2 = M1 + 5 matchups. Totals T1/T2 likewise | walk-forward by season |

### Player value (BPM-lite, all dev; points of on-court margin per 48 for each extra unit per 48)
| Stat | Coef (se) | Reading |
|---|---|---|
| points | +1.20 (.09) | read together with FGA: a made 2 costs 1 FGA, so it nets ~+1.2, and a miss costs −1.23. **Efficiency, not volume** |
| FGA | −1.23 (.10) | |
| FTA | −0.45 (.12) | |
| 3PM | +0.77 (.16) | on top of its 3 points: spacing value |
| ORB | +0.09 (.15) | ~0 |
| DRB | +0.34 (.07) | |
| assists | +0.79 (.08) | creation |
| turnovers | **−2.54** (.23) | the costliest event |
| steals | **+1.81** (.26) | a proxy for defensive activity |
| blocks | +0.34 (.22) | |
| fouls | +0.04 (.14) | ~0 |

R² is about 0.26 on player-seasons. Box score explains about a quarter of plus-minus, which is typical for BPM-type
models.

### Spread stack M2 (STRICT; fit on 2023-24 + 2024-25 for live use; style terms are points per SD)
| Term | Coef (se) | Reading |
|---|---|---|
| const | +0.38 (.24) | |
| mu0 (team rating) | +0.75 (.05) | the old ridge, shrunk by 25% |
| dPfull (player-strength diff, full roster) | +1.81 (.33) | per point of strength diff; players add information beyond team results |
| dAbs (absence deficit) | +1.20 (.21) | each point of strength lost to absences ≈ 1.2 pts of margin (ACTUAL variant: +1.94) |
| S1 paint defence (home − away share of points allowed in the paint) | **−1.12 (.21)** | teams that leak the paint underperform (L30) |
| S2 3PA rate × opponent's 3PA allowed | +0.31 (.21) | weak |
| S3 offensive rebounding vs opponent's defensive rebounding | +0.05 (.21) | 0 |
| S4 forced turnovers × opponent's carelessness | −0.07 (.21) | 0 |
| S5 FT rate × opponent's fouling | +0.49 (.21) | drawing fouls against foul-prone teams, weak positive |

Residual SD is 13.7, so P(home) = Φ(margin/13.7).

### Totals stack T2 (STRICT)
| Term | Coef (se) | Reading |
|---|---|---|
| tot0 | +0.93 (.08) | team-points ridge |
| Ofull (players' expected points) | +1.48 (.38) | per SD |
| oAbs (points lost to absences) | +1.05 (.28) | per SD; ACTUAL +1.74 |
| U1 pace sum | +0.77 (.52) | weak; already in tot0 |
| U2 sum of the teams' own 3P% | **−1.88 (.39)** | hot shooting regresses, and the market under-regresses (L31) |
| U3 sum of opponents' 3P% allowed | **−2.34 (.33)** | same, on defence |
| U4 3PA-rate sum | +0.13 (.28) | 0 |
| U5 back-to-back count | −0.13 (.28) | 0 |

**What "playing style" means here:** 3 of 10 style terms matter. Paint defence (sides), and 3-point percentage luck on
both ends (totals). Pace, rebounding, turnovers and 3-point volume are already priced, or they are noise.

## 2. Nested comparison (out of sample, 2023-24 and 2024-25)
| Model | Margin MAE 24 / 25 | Stack weight given OPEN | Given CLOSE | Spread CLV vs open (gap ≥ 2) | No-vig ATS |
|---|---|---|---|---|---|
| M0 team rating (old ridge) | 11.11 / 10.99 | +0.32 [+.16, +.49] | +0.02 | +0.72 (n 1,285) | +1.0% |
| M1 + players (strict) | 10.97 / 10.95 | +0.39 [+.20, +.58] | −0.01 | +0.80 (n 1,106) | −0.8% |
| M2 + style (strict) | 10.96 / 10.96 | +0.41 [+.24, +.57] | +0.08 | +0.79 (n 1,262) | +2.3% |
| M2 ACTUAL (lookahead) | 10.96 / 10.89 | +0.48 [+.31, +.64] | +0.02 | +1.03 / +1.75 (no-news / news) | +3.3% / +8.3% |
| (opening line) | MAE 10.82 | | | | |
| (closing line) | MAE 10.55 | | | | |

| Totals model | Total MAE | Given OPEN | Given CLOSE | No-vig O/U (gap ≥ 3) |
|---|---|---|---|---|
| T0 ridge | 14.75 | +0.71 | −0.08 | +4.9% (n 1,078, p .05) |
| T1 + players | 14.85 | +0.70 | −0.04 | +2.6% (n 1,151) |
| T2 + style | 14.73 | +0.79 | +0.05 | **+8.7%** (n 1,190, p .001) |
| T2 ACTUAL | 14.67 | +0.82 | +0.06 | +12.8% (lookahead) |
| (open / close) | 14.73 / 14.17 | | | |

The opening total is notably worse than the close: MAE 14.73 vs 14.17, and the line moves 3.0-3.4 points on average.
The opening spread moves 1.75 points on average. **The opening total is the softer number.**

## 3. Trials (bar p < .00625; one-sided game bootstrap, 4,000 resamples)
| Trial | Bets | Effect per bet | 95% CI | p | Seasons + | Pass |
|---|---|---|---|---|---|---|
| T1 spread STRICT vs open, CLV (pts) | 1,262 | +0.79 | [+0.63, +0.95] | <.00025 | 2/2 | **yes** |
| T2 same, ATS no-vig | 1,262 | +2.3% (52.1%) | [−3.2, +7.7] | .21 | 2/2 | no |
| T3 same, ATS −110 | 1,262 | −0.6% | [−5.8, +4.6] | .58 | 1/2 | no |
| T4 totals STRICT vs open, CLV (pts) | 1,190 | +2.00 | [+1.65, +2.45] | <.00025 | 2/2 | **yes** |
| T5 same, O/U no-vig | 1,190 | **+8.7% (55.5%)** | [+3.1, +14.3] | .001 | 2/2 | **yes** (dataset bar only) |
| T6 spread ACTUAL vs close, no-vig | 1,231 | −1.2% (49.4%) | [−6.6, +4.3] | .68 | 0/2 | no |
| T7 totals ACTUAL vs close, no-vig | 1,177 | +3.2% (51.6%) | [−2.5, +9.1] | .14 | 2/2 | no |
| T8 spread ACTUAL vs open, CLV, news games | 564 | +1.75 | [+1.52, +1.99] | <.00025 | 2/2 | **yes** (lookahead upper bound) |

Post-hoc dose, not trials. T5's totals rule at other gaps:

| Gap | Bets | No-vig | At −110 |
|---|---|---|---|
| ≥2 | 1,511 | +9.9% | +6.6% |
| ≥4 | 913 | +8.9% | |
| ≥6 | 522 | +7.9% | |
| ≥8 | 298 | +16.4% | |

The effect is flat to rising with the gap, not a cliff. Spread CLV rises with the gap: +0.63 at ≥1, +1.44 at ≥4,
+3.10 at ≥6. The spread win rate does not rise reliably.

## 4. Traps and caveats (read before acting)
- **Whole-ledger multiplicity.**
  - T5 p .001 fails 0.05/213. The same "model leads the opener" family already produced O1's dev pass and then its
    sealed FAIL. In O1, CLV halved on sealed data, DraftKings openers were sharper, and the money test failed.
  - Expect shrinkage. A true edge of 3-4% is more plausible than 8.7%.
- **The opener is not documented.** ESPN's "open" timestamp is unknown, and opener limits are low. The totals edge
  disappears against the close (52.4% graded at the close), so it is only real if you can bet at or near the
  first-posted number.
  - ESPN BET ended 2025-12 (O1 note).
  - Whether Kalshi lists NBA totals early enough to be "stale" is UNVERIFIED.
- **The ACTUAL-availability results (T6-T8, the "act" rows) use who really played.** That is lookahead relative to
  the open. Only the STRICT rows (T1-T5) are no-lookahead.
- **The amendment (section 5) was made after looking at out-of-sample model accuracy**, though before any market join.
- **Live caveats for `today`:**
  - stats.nba.com timed out from this Mac, so the paint term (S1) is 0 live. It is the largest style coefficient on
    sides.
  - Early in the season, "full roster" (the last 10 team games) reaches back into last April. Tanking-era G-leaguers
    then show up as "top-5 players not in last game". Use `--out` for real injury news.
  - P(home) assumes normal errors with SD 13.7.
- Under-power: 1,190 bets at sd ~1 unit gives an SE of ~2.9%. A sealed season (~600 bets) at a true +4% has only
  ~25% power at one-sided .05. That would be inconclusive, not a fail.

## 5. Amendment log
- 2026-10-25, after `build` and before any market join: 2021-22 is a cold-start season (no prior player history).
  The 2023 totals stack trained on it was biased by +4 points (T1 MAE 15.2 vs T0 14.65).
- The fix: 2021-22 became warm-up only. Stacks are fit from 2022-23 onward, and trials are scored on 2023-24 and
  2024-25.
- No feature, threshold or hyperparameter changed.

## 6. Proposed next step (needs the user's yes; NOT run)
**Sealed test O2-TOT** on NBA 2025-26:
- Model T2_STRICT, frozen exactly as in `results/nba_simple_model_frozen.json`. Refit nothing: the BPM-lite weights and the stack
  coefficients stay as fit on dev 2022-23..2024-25. Only the ratings and state update day by day, as here.
- Rule: bet over/under when |T2 − opening total| ≥ 3.0. Opener = ESPN BET where present, else DraftKings (O1's
  amended order, because ESPN BET ended 2025-12).
- PRIMARY: realised O/U P&L at the no-vig opening price, one-sided game bootstrap, pass p < .05 and > 0.
- Secondary: CLV in points, P&L at −110, and the close-graded win rate.
- First, a schema-only check that `lines_2026` has `total_open` and the open odds (needs the user's OK).

**Forward (bettable) version:**
- Log the first-posted total at DraftKings, FanDuel and Pinnacle (if reachable) and Kalshi's first NBA total price,
  with timestamps (the opener logger already proposed for O1).
- Then run `today` each morning and paper-bet |gap| ≥ 3 at the logged first price, at maker/no-vig.
- Grade after ~300 bets. Spreads stay paper-only (no money signal).
- The "absence news" angle needs a real-time injury feed beating the book. That is not something we have.
