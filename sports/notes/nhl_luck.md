# NHL PDO luck vs the market + blind unders by opening total (2026-10-27, DEV ONLY)

- Script: `scripts/nhl_luck.py` (`build` / `backtest [--ledger]`). Its docstring is the pre-registration, written
  before any market was joined to luck features or outcomes.
- Ledger: dataset `nhl_luck`, 17 trials, logged 2026-10-27. Dataset bar 0.05/17 = 0.0029. BH q=.05 also computed.
- Output: `results/nhl_luck.json`. The joined game table is `data/nhl_luck/games.pkl`, and the join and price checks
  are in `data/nhl_luck/build_log.json`.
- Nothing committed. No orders.

## Verdict (plain language)
1. **DEAD. 0/17 trials pass, and 0/17 pass BH.** The best p is T2c, +0.5 pts with p .23.
2. **The "blind under at an opening 6.5" lead does NOT replicate on older seasons.**
   - SBR 2007-08 .. 2022-23: 1,160 games. The under at the open price returns **−4.3%** [−10.1, +1.7]. Hit rate is
     50.0% against a 52.2% break-even, and 5/11 seasons are positive. No-vig ≈ +0.2%.
   - The under at the close returns −3.8%.
   - Under at 5.5 is −1.6% at the open and −2.7% at the close. Under at 6.0 is −4.0% / −4.3%.
   - Unders lost in most recent seasons: 6/7 seasons from 2016-17 onward are negative at 5.5.
   - The ESPN BET 2023-25 discovery sample under 6.5 makes +1.2% at the offered open price (n 1,539). Its +5.8% was
     no-vig, and that lead was post-hoc. **It looks like a 2-season fluctuation, not a structural lean.**
3. **The market already prices PDO luck.**
   - Betting against the luck (totals and ML, extreme quintiles, ≥ 16 games played) loses about the vig: −2.6% to −4.6%.
   - Regression of (final − closing total) on total luck LT gives a slope of −0.08 (SE 0.05). A luck-blind market would
     give ≈ −1. **The close prices ~90%+ of luck regression.**
   - ML: the slope of (home win − p_close) on LM is −0.008 (SE 0.009), which is ≈ 0.
4. **Early season (≤ 15 games):** every trial is within ±0.4%, with p ≥ .44.
5. **Does the market move on luck? (T4)** Yes, but the "wrong" way for totals.
   - From open to close, totals move *away* from the regression side, by −0.031 implied goals per bet
     (z ≈ −11; 13/15 seasons).
   - So money follows the hot-shooting / hot-goalie teams after the open. This is why the luck bets do better at the
     close (−3.3%) than at the open (−4.6%).
   - Even so, the close is not beatable on luck.
   - ML: no movement on luck (−0.05 pts).

## Data
| Source | What | Coverage |
|---|---|---|
| sportsbookreviewsonline NHL archive → `data/sbr_nhl/` (16 html pages) | Open/close ML, open/close total + price, puck line from 2014-15 | 2007-08 .. 2022-23. 2022-23 is partial (327 games, Oct-Dec). There is no 2020-21 file gap: "2021" = the 2020-21 season |
| NHL stats REST API (`api.nhle.com/stats/rest/en/team/summary?isGame=true`, keyless) → `data/nhl_api_teamgames/` | Per team-game goals and shots for/against, shootout wins | 2007-08 .. 2022-23 regular season, complete |

- **Join:** SBR is joined to the API's regular-season games on date + teams. That gives 17,904 games.
  - After the shootout fix, 4 joined games had a final-score mismatch and were dropped. The other 17,904 all match.
  - Playoffs and unjoined SBR rows are dropped.
  - One pair listed home-first in 2019-20 was dropped.
- **Shootouts:** API `goalsFor` EXCLUDES the shootout goal. SBR finals include it.
  - Luck uses the non-shootout goals.
  - Grading uses the SBR final, where a shootout counts as +1, as books grade.
  - My first build stripped the shootout goal twice. The score check caught it and it was fixed before any market
    join.
- **Totals price convention:** SBR row 1 = OVER and row 2 = UNDER. I verified this against ESPN's last-snapshot
  consensus (market vs market):
  - 2021-22: corr +0.79, n 1,310.
  - 2022-23: corr +0.85, n 327.
  - Mean absolute difference in de-vigged P(over) is 0.5 pts.
  - **Older seasons are assumed to follow the same layout. This is unverified.**
- **GAP: SBR does not name the book or the time of "Open" (or of "Close").** Treat "open" as an unnamed early
  number.
- **Lines:** opening totals are mostly 5.5 (9,656 games), 5.0 (3,875), 6.0 (3,147) and 6.5 (1,160). 6.5 is rare
  before 2016: only 48 games in the first half, so T1a/T1b's "first half" is tiny.
- **Median overround:** totals 4.5%, ML open 3.9%, ML close 3.5%.
- **What luck cannot see:** there is no empty-net, xG or strength-state adjustment. Luck = all-situation shooting% and
  save% excess over a K = 2,500 pseudo-shot regression to the league's season-to-date mean.

## Trials (one-sided cluster bootstrap over game-dates, 4,000 draws; effect = ROI per unit unless noted)
| Trial | n | Effect | 95% CI | p | Hit / break-even | Halves (≤14-15 / ≥15-16) | Seasons + |
|---|---|---|---|---|---|---|---|
| T1a under, open 6.5, at open | 1,160 | −4.3% | [−10.1, +1.7] | .92 | .500 / .522 | −23.2% (48) / −3.4% | 5/11 |
| T1b under, open 6.5, at close | 1,160 | −3.8% | [−9.6, +2.2] | .89 | .496 / .517 | −22.2 / −2.9 | 5/11 |
| T1c under, open 5.5, at open | 9,656 | −1.6% | [−3.4, +0.3] | .96 | .526 / .534 | −0.1 / −4.0 | 5/16 |
| T1d under, open 5.5, at close | 9,656 | −2.7% | [−4.6, −0.9] | 1.00 | .522 / .536 | −1.5 / −4.8 | 4/16 |
| T1e under, open 6.0, at open | 3,147 | −4.0% | [−7.1, −0.9] | .99 | .500 / .523 | +1.5 / −5.7 | 6/16 |
| T1f under, open 6.0, at close | 3,146 | −4.3% | [−7.5, −1.1] | 1.00 | .497 / .522 | +0.6 / −5.8 | 4/16 |
| T2a totals luck, full season, at close | 5,053 | −3.3% | [−5.9, −0.7] | .99 | .498 / .518 | −1.7 / −5.1 | 4/15 |
| T2b same, at open | 5,053 | −4.6% | [−7.3, −2.1] | 1.00 | .494 / .520 | −2.6 / −6.8 | 4/15 |
| T2c ML luck, residual vs de-vigged close (prob pts) | 5,653 | +0.47 | [−0.75, +1.79] | .23 | — | +1.83 / −0.74 | 8/15 |
| T2d ML luck, at close | 5,653 | −2.6% | [−5.4, +0.5] | .95 | .432 / .442 | +0.2 / −5.0 | 5/15 |
| T2e ML luck, at open | 5,652 | −3.2% | [−6.0, −0.1] | .98 | .432 / .444 | −0.4 / −5.6 | 5/15 |
| T3a totals luck, early (1-15 games), at close | 1,221 | +0.3% | [−4.4, +5.2] | .45 | .520 / .518 | −1.6 / +2.1 | 9/15 |
| T3b same, at open | 1,221 | +0.2% | [−4.6, +5.1] | .46 | .523 / .521 | −2.4 / +2.5 | 9/15 |
| T3c ML luck, early, at close | 1,443 | +0.3% | [−5.3, +6.1] | .46 | .490 / .487 | +2.5 / −1.2 | 8/15 |
| T3d same, at open | 1,443 | −0.1% | [−5.6, +5.6] | .51 | .490 / .488 | +1.9 / −1.5 | 9/15 |
| T4a totals: CLV toward luck side (implied goals) | 5,053 | −0.031 | [−0.037, −0.025] | 1.00 | — | −0.011 / −0.052 | 2/15 |
| T4b ML: CLV toward luck side (prob pts) | 5,652 | −0.05 | [−0.13, +0.03] | .90 | — | −0.21 / +0.09 | 6/15 |

- Over/under split, T2a: overs −3.7% (2,548), unders −2.9% (2,505).
- Home/away split, T2d: away −3.5%, home −1.6%. Neither side works.

## Descriptive (not trials; scored 2008-09 .. 2022-23)
- **(final − close total) ~ LT:**
  - Full season: slope −0.080 (SE 0.051), n 12,878, sd(LT) 0.40 goals.
  - Early: −0.026 (SE 0.026).
  - The sign is right and the size is about 1/10 of a luck-blind market.
- **(μ_close − μ_open) ~ LT:** full season +0.050 (SE 0.0045), early +0.011 (SE 0.0025). Lines drift *toward* the
  luck-inflated total after the open.
- **ML:**
  - (home win − p_close) ~ LM: −0.008 (SE 0.009).
  - (p_close − p_open) ~ LM: +0.0004 (SE 0.0006).
- **No-vig under at the open:** 5.5 +2.8%, 6.0 −0.0%, 6.5 +0.2%.
  - The 5.5 "edge" is just the under being juiced: median −114, fair P(under) 0.511.
  - The offered price removes it.
- **Not checked:** the ESPN BET T1 numbers here use all ESPN regular-season finals with an opener (n 1,539 at 6.5).
  The earlier note used n 1,448, restricted to games with model inputs, and reported no-vig rather than the offered
  price. So the two are not the same metric.

## Caveats
- Before writing the docstring I looked at the table layouts and about 5 rows each. No luck or outcome aggregates.
- One run with `--ledger` gave 17 rows. There was no earlier scored run.
- Fixes before any market join:
  - The shootout double-strip.
  - The H-first pair parse.
  - Rewriting the price-convention check to use ESPN's last snapshot, because ESPN's `*_close` is empty for 2021-22.
- One unnamed "open" source. Its timing is unknown, so the T4 "move" is from an unknown time to the close.
