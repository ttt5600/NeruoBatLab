# NFL recency bias: does the line overreact to last week's result? (2026-10-27, DEV 2007-2019 only)

Script: `scripts/nfl_recency.py` (`features` builds features without touching this week's lines or outcome; `run [--ledger]`
does the join, about 30 s). The pre-registration is the script's docstring. I wrote it and ran `features` before the first join.
JSON: `data/lab/nfl_recency_results.json`. Ledger: 12 rows, dataset `nfl_recency`, with a Bonferroni bar of p < .05/12 = .0042 and BH
correction at q .05 across all 12. Data: target games 2007-2019 from week 2 on (3,259 with SBR open and close). Last week's lines
come from the nflverse schedule close. `nfl_games.parquet` starts in 2015 and is the same nflverse close. EPA comes from nflverse
pbp 2006-19. I did not read anything from 2020 on.

## Verdict

**There is no recency-bias edge. Neither the opener nor the close overreacts to last week in a way you can bet.**
Of 12 counted trials, 11 fail. The one that "passes" (T2b) describes how the line moves. It is not a bet.

1. **Fading last week's result loses at both the open and the close.** Fading a 14+ point cover or non-cover hits 51.7% at the
   close (738/1428) and 50.2% at the open. Fading last week's outright dog winners hits 50.2%, and the luck-fade rule hits 48.7%.
   Breakeven at -110 is 52.4%, and every rule is below it. In the slope trials, the residual against the close and against the open
   shows no overreaction to last week's cover margin: +0.10 and -0.17 pts per SD (SD 18.8 pts), with both CIs about +-0.5. The halves
   disagree in sign.
2. **What does move:** after the open, the line drifts further toward last week's coverer, by +0.275 pts per SD of dCover
   [+0.20, +0.35]. The p-value is 3e-13, both halves are positive and 12 of 13 seasons are positive (T2b). The decomposition shows the move is
   **all EPA, none of it luck**. The open-to-close move is +0.37/SD on last week's EPA-implied cover and +0.01/SD on its luck part.
   The outcome does not reward the move. Neither the open nor the close is significantly mis-set (T2a, T1a), so there is no CLV to
   collect: a CLV bet here would pay only if the outcome agreed. The pattern looks like sharp or model money arriving on EPA after the
   open, which matches the simple-model result that EPA leads the opener. It does not look like public money chasing scores.
3. **Luck versus skill (T3): the market does not price luck as skill in any way you can bet against.** The luck coefficient on the
   ATS residual is -0.19/SD vs the close and -0.20 vs the open. Both have the wrong sign for a fade, and p is about .8. Not counted:
   the EPA ("skill") coefficient is -0.30/SD vs the close (SE 0.25) and +0.07 vs the open. That means the close slightly *over*-moves
   on last week's EPA. It is not significant.
4. **Totals (T4) follow the same pattern.** After a team's last game goes over, the total rises from open to close by +0.40/SD of
   the summed last-game total residual (p .0001, 12/13 seasons; not counted). Against the close that is a weak overshoot: +0.30/SD
   in the fade direction, p .09, halves +0.60/+0.09. The open is fine (-0.10). The totals-fade rule at the close hits 53.3%
   (796/1494), +1.7% ROI, CI [-3.2%, +6.5%], p .25.

**Bottom line:** the NFL market does not overreact to blowouts, upsets, cover margin or turnover luck. The only systematic effect
is that the line keeps moving toward last week's EPA, and on totals toward last week's over, after the open, and it ends at about
the right number. The two weak overshoots at the close (EPA spread coefficient -0.30, totals +0.30/SD) point the same way. Both are
non-significant, and they cannot be re-tested on 2020-25, which was spent on the refit. That leaves forward data only, and the
effect is too small to justify a logger.

## Counted trials (bar p < .0042; slopes are pts/SD sign-flipped so + = supports fade, T2b two-sided raw sign; rules = ROI at -110)

| trial | n | effect | 95% CI | p | BH q | 2007-12 / 2013-19 | seasons + | hit | pass |
|---|---|---|---|---|---|---|---|---|---|
| T1a slope (margin - close) ~ dCover | 3259 | +0.10 | [-0.39, +0.60] | .34 | .97 | +0.27 / -0.06 | 8/13 | | no |
| T1b blowout-fade at close | 1470 | -1.3% | [-6.1, +3.7] | .70 | .97 | +2.5 / -4.9% | 6/13 | 51.7% | no |
| T1c upset-fade at close | 931 | -4.0% | [-10.4, +2.5] | .89 | .97 | -7.0 / -1.3% | 2/13 | 50.2% | no |
| T2a slope (margin - open) ~ dCover | 3259 | -0.17 | [-0.67, +0.32] | .75 | .97 | +0.12 / -0.45 | 5/13 | | no |
| T2b move (close - open) ~ dCover | 3259 | **+0.275** | [+0.20, +0.35] | <1e-10 | <.001 | +0.15 / +0.39 | 12/13 | | yes (descriptive, not a bet) |
| T2c blowout-fade at open | 1470 | -4.0% | [-8.9, +1.0] | .94 | .97 | +1.6 / -9.4% | 5/13 | 50.2% | no |
| T3a slope (margin - close) ~ dLuck given dSkill | 3259 | -0.19 | [-0.66, +0.28] | .79 | .97 | +0.17 / -0.53 | 4/13 | | no |
| T3b slope (margin - open) ~ dLuck given dSkill | 3259 | -0.20 | [-0.67, +0.27] | .80 | .97 | +0.21 / -0.59 | 5/13 | | no |
| T3c luck-fade at close | 739 | -6.7% | [-13.7, +0.1] | .97 | .97 | -3.4 / -10.0% | 2/13 | 48.7% | no |
| T4a slope (points - close) ~ sTot | 3259 | +0.30 | [-0.14, +0.74] | .09 | .55 | +0.60 / +0.09 | 9/13 | | no |
| T4b slope (points - open) ~ sTot | 3259 | -0.10 | [-0.58, +0.38] | .66 | .97 | +0.12 / -0.26 | 7/13 | | no |
| T4c totals-fade at close | 1514 | +1.7% | [-3.2, +6.5] | .25 | .97 | -0.8 / +3.7% | 7/13 | 53.3% | no |

Feature SDs: dCover 18.8, dLuck 12.9, dSkill 14.9, sTot 19.0 pts. Raw T1a slope: -0.006 pts of residual per point of
last-week cover difference (SE 0.013).

## Reported, not counted
- Rules at the other line: upset-fade at the open 49.1%, luck-fade at the open 47.9% (-8.2%), totals-fade at the open 52.1%.
- Fade only a big coverer whose opponent is not one: 51.1% (800 decided). Back only a big non-coverer: 51.1% (748).
- Clipping the residual at +-14: T1a +0.07, T3a -0.13. Still null.
- Line structure (pts per SD; coefficients are joint fits on dLuck and dSkill):

| dependent | dLuck | dSkill |
|---|---|---|
| raw this-week margin | +1.26 (0.26) | +0.59 (0.27) |
| open spread | +1.06 (0.10) | +0.53 (0.10) |
| close spread | +1.07 (0.11) | +0.90 (0.11) |
| close - open | +0.01 (0.04) | **+0.37 (0.04)** |
| margin - close | +0.19 (0.24) | -0.30 (0.25) |

  Read: on luck, the opener and the close load the same amount, and that amount matches the outcome. On EPA, the opener loads about
  as much as the outcome does, the close loads more (0.90 vs 0.59), and the residual vs the close is -0.30 (not significant).

## Caveats
- **The "luck" proxy is not pure luck.** "Luck" here means margin minus the EPA-implied margin, where the EPA-implied margin uses
  EPA/play with interceptions and lost fumbles removed (52.4 pts per EPA/play, home +2.6, R2 0.62). It predicts next week's raw
  margin (+1.26/SD), so it also carries persistent qualities: turnover avoidance, takeaways, special teams. Last week's line is not
  controlled in that row either. The decisive tests (T3, residuals against the line) do not depend on the proxy being pure luck.
  The market loads it at the right weight.
- The EPA-to-points mapping was fit once on all of 2006-19 (box scores only, no market data). It is a fixed scale, not a forecast.
- SBR opener: the book and the timestamp are not documented (see `nfl_simple_model.md`). Spreads are assumed to be -110 at the open.
- Population: weeks 2+ with REG and POST combined. The previous game is the team's last game in the same season, skipping byes.
- This study was not pre-filtered by any earlier finding. It overlaps with `nfl_comebacks_matchups.md` finding 5 ("point margin
  and turnover margin net of EPA are flat vs the close"), which it agrees with.
