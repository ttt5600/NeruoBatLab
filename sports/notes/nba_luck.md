# NBA 3-point luck: does the market overreact to small-sample shooting? (DEV, 2026-10-27)

## Verdict
**No bettable edge. Kill the "both teams cold early → over" angle.**

1. **The effect is real, but it is tiny.** Games whose teams have been shooting hot from 3 do go a bit under the close.
   - Slope: −0.31 pts of (final − closing total) per point of game luck.
   - It passes the study's own bar: T1a p .002, both halves negative, 9/10 seasons.
   - It does **not** pass the whole-ledger bar (0.05/~308 rows = .00016).
   - Size: the gap between the q20 and q80 luck games is about 1.5 luck points, so about 0.5 pts of total. Against a
     ~18-pt residual SD, that does not move a bet. The pre-registered quintile bet **loses −1.1% at −110**
     (4,719 bets, 51.8%, 2/10 seasons positive).
2. **The early-season version is not better.**
   - Slope −0.28 (p .02). The early quintile bet is +3.7% (572 bets, 54.3%, p .18, 7/10 seasons).
   - That is noise-sized, and it fails the bar.
3. **The +7.0 "both cold early" seed was a 3-season fluke.**
   - On the same rule (both teams ≤ 5 games, luck ≤ −4) I reproduce it in 2022-23..2024-25: +6.5, +5.4, +8.8 (n 58).
   - In the 7 earlier seasons it averages about −1.3 (n 126; 2021-22 alone is −8.7).
   - Over all 10 seasons: +1.3, 5/10 seasons positive.
   - It is not dose-responsive: luck ≤ −6 gives −0.2, and luck ≤ −8 gives −1.1.
4. **The market does not move on luck, and openers are not softer on it.**
   - Open→close move ~ luck: slope +0.02 (p .81).
   - Residual vs the opener: slope −0.20 (p .19).
   - Opener quintile bet: −0.3%.
   - This only covers 2023-24 and 2024-25.
5. **Sides: same story.**
   - The lucky team does underperform the closing spread: −0.09 pts ATS per point of margin luck, p .011, 10/10
     seasons negative.
   - Fading it loses: −2.6% (4,725 bets, 51.0%).
6. **It is already in our model.**
   - Once L31's raw season 3P% for/against terms are in the regression, the luck coefficient goes to about zero or
     positive (+0.32, se .18).
   - Those terms keep their weight (A −1.01, B −0.89).
   - So whatever the market under-regresses is the 3P% level that L31 / nba_simple_model U2-U3 already use, not a
     separate "luck" signal.
   - Partial effect on the 2023-25 model sample: luck given the model gap −0.37 (se .37). That is unresolved, and the
     model gap gets 0 weight given the close (known).
7. **Sheet guidance.** Keep the luck column as context. Do not tilt overs on cold-shooting games.

## Design (pre-registered in the `scripts/nba_luck.py` docstring before any market join)
- **Data:**
  - DEV only: 2015-16..2024-25 regular season, 11,814 games with both teams ≥ 1 game played.
  - 3P makes and attempts come from `data/nba_comebacks/teamgames.parquet`.
  - Closing lines come from `data/nba_comebacks/games.parquet`.
  - Openers are DraftKings, else ESPN BET, from ESPN lines for 2023-24 and 2024-25 (2,424 games).
  - The model is `wf.pkl` T2_strict.
  - SEALED 2025-26 and `nba_ingame.parquet` were not read.
- **Feature:** the same as `nba_tonight_sheet.py`.
  - 3P% is shrunk with 750 attempts of prior: the team's own previous season for offence, the previous-season
    league rate for defence.
  - 2015-16 has no previous season in the data, so its prior is the league rate to date.
  - Luck pts/g = (3PM − 3PA·shrunk)/GP·3.
  - Game luck = 0.5 × (both offences + both defences).
  - Margin luck D = (off − def)_home − (off − def)_away.
  - Everything is season-to-date, strictly before the game.
- **Buckets:** EARLY = both teams have played 1-10 games (1,425 games). REST = all other games.
- **Quintile cuts:** taken within each bucket, from the feature alone (no outcomes).
- **Statistics:** slopes are OLS with SEs clustered by date. ROI is at −110, with a one-sided game bootstrap.
  Bonferroni bar = 0.05/14 = .0036, plus BH.

## Trials (14, all in lab/ledger.jsonl, dataset `nba_luck`)
| Trial | What | Effect | 95% CI | p | Halves (2015-19 / 2020-25) | Seasons | Pass |
|---|---|---|---|---|---|---|---|
| T1a | total res vs close ~ game luck | slope −0.305 | [−0.51, −0.10] | .0021 | −0.47 / −0.23 | 9/10 | **yes** (study bar; fails ledger bar) |
| T1b | quintile bet vs close (cold→over, hot→under) | −1.1% (51.8%, n 4,719) | [−3.8, +1.6] | .78 | 0.0 / −1.7% | 2/10 | no |
| T2a | total res vs opener ~ luck | −0.199 | [−0.65, +0.25] | .19 | 23-24 −0.11 / 24-25 −0.25 | 2/2 | no |
| T2b | quintile bet vs opener | −0.3% (52.2%, n 855) | [−6.5, +6.0] | .52 | −1.4 / +1.1% | 1/2 | no |
| T2c | open→close move ~ luck (two-sided) | +0.019 | [−0.13, +0.17] | .81 | +0.17 / −0.13 | — | no (market does not move on luck) |
| T3a | home ATS res vs close ~ D | −0.089 | [−0.17, −0.01] | .011 | −0.12 / −0.07 | 10/10 | no (BH no, Bonferroni no) |
| T3b | fade the lucky team ATS | −2.6% (51.0%, n 4,725) | [−5.3, +0.1] | .97 | −0.4 / −3.9% | 3/10 | no |
| T4-1a | EARLY T1a | −0.275 | [−0.54, −0.01] | .020 | −0.39 / −0.22 | 9/10 | no |
| T4-1b | EARLY T1b | +3.7% (54.3%, n 572) | [−4.0, +11.4] | .18 | +8.9 / +0.7% | 7/10 | no |
| T4-2a | EARLY vs opener | −0.204 | [−0.76, +0.36] | .24 | +0.10 / −0.47 | 1/2 | no |
| T4-2b | EARLY opener bet | +1.6% (n 94) | [−16.7, +19.8] | .46 | −6.4 / +11.0% | 1/2 | no |
| T4-2c | EARLY move ~ luck | −0.016 | [−0.20, +0.17] | .87 | | — | no |
| T4-3a | EARLY sides slope | −0.089 | [−0.18, −0.00] | .023 | −0.05 / −0.11 | 9/10 | no |
| T4-3b | EARLY fade lucky team | +2.2% (53.5%, n 572) | [−5.7, +9.9] | .29 | +2.2 / +2.2% | 6/10 | no |

With season fixed effects, the T1a slope is −0.32 (p .002), so the result is not a season-level artefact.

## Descriptive
- **REST bucket:**
  - Totals slope −0.61 (p < .001). That is steeper per luck point, but luck itself is about 5× smaller after 10
    games.
  - The quintile bet makes −0.2%. Sides: −0.09 (p .13), and the bet makes −3.3%.
- **2015-22 only** (before the seed seasons):
  - ALL: slope −0.31 (p .007), but the bet makes −1.5%.
  - EARLY: slope −0.27, and the bet makes +2.4% (n 401).
- **Totals deciles (ALL):** residuals run from about +0.8..+2.0 in the cold deciles down to −1.1 in the hottest. The
  shape is monotone-ish, but every decile is within ±2 pts.
- **Totals deciles (EARLY):** the hottest decile is −4.2 (n 143). Apart from that the pattern is noisy, with no clean
  dose-response.

## Caveats
- **Closing lines:** OddsData totals are assumed to be the close (unverified, see `nba_comebacks_matchups.md`).
  ROI assumes −110 because there are no historical prices.
- **Opener trials** cover 2 seasons only, so they are under-powered.
- **Quintile cuts** were pooled over all DEV seasons. They are feature-only, but era drift in 3PA volume shifts them.
- **The L31 overlap regression** z-scores the raw 3P% sums within each season. That is a mild feature lookahead,
  acceptable because it is descriptive only.

Files: `scripts/nba_luck.py`, `results/nba_luck.json`, 14 rows in `lab/ledger.jsonl`.
