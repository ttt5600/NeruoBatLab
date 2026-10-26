# L4: NFL player props by category x game script (DEV only, 2026-10-25)

## Verdict
**Dead as specified: 0 of 8 pre-registered cells pass.** None of the cells clears Bonferroni (p < 0.00625) or BH (lowest q = 0.31).
The two best cells are rushing-yard unders (C7: low total; C3: underdog non-QB). Both are positive in both halves, but they have n = 155 and n = 99 and p = 0.04 and 0.08, which is noise after 8 trials.
The tests can only detect large effects: one book, one season, 40-80 games per cell. A real effect of 2-3pp would go undetected here. No sealed 2025 ask: nothing earned one.

## Data: what's actually there
- **2022 and 2023:** no yardage, reception or attempt props on ESPN. The only markets are TD-scorer "yes" markets, QB interceptions O/U and tackles O/U. **The brief's halves (2022 vs 2023-24) can't be tested.**
- **2024:** one book (ESPN BET), all the yardage markets, `line_open` and `dec_open` on 100% of rows on both sides. After de-duplication this gives 19,376 graded player-game-market pairs over 280 games, including the postseason.
- `line`/`dec` timestamps mostly fall after kickoff. We used **openers only**. As a sanity check, Brier on the "current" prices was 0.245 vs 0.240 on the openers. So "current" is not leaking in-game information. It just isn't used.
- Game script uses the same book's opening home spread and total. Every player's team came from the box score (0% missing).
- 2025 was never opened. The script asserts this.

## Method (frozen in the `scripts/nfl_props_script.py` docstring before any outcome join)
- **Observations:** one per (event, athlete, market). Where there are several open pairs, we kept the main pair (de-vigged probability closest to 0.5). Pairs with overround outside [1.00, 1.20] were dropped, and so were pushes.
- **Metric:** residual = win minus the proportional de-vigged open probability. P-values are cluster-robust by game.
- **Secondary:** ROI at `dec_open`.
- **Halves:** H1 = weeks 1-9, H2 = weeks 10+ and the postseason.

| cell | side | n | hit | de-vig p | resid | p | BH q | ROI | H1 / H2 resid |
|---|---|---|---|---|---|---|---|---|---|
| C1 fav <= -6.5 QB pass yds | U | 79 | .418 | .482 | -6.5pp | .24 | .50 | -18.6% | -.178 / +.001 |
| C2 fav QB pass att | U | 73 | .507 | .507 | 0.0 | 1.00 | 1.00 | -7.0% | +.011 / -.006 |
| C3 dog >= +6.5 non-QB rush yds | U | 99 | .596 | .501 | +9.5pp | .078 | .31 | +12.1% | +.130 / +.071 |
| C4 dog receptions | O | 377 | .483 | .488 | -0.6pp | .82 | 1.00 | -4.6% | -.007 / -.005 |
| C5 total <= 40.5 pass yds | U | 77 | .455 | .499 | -4.5pp | .46 | .73 | -15.3% | -.011 / -.093 |
| C6 low total rec yds | U | 306 | .467 | .501 | -3.4pp | .25 | .50 | -12.2% | -.037 / -.028 |
| C7 low total rush yds | U | 155 | .581 | .503 | +7.7pp | .039 | .31 | +11.1% | +.020 / +.178 |
| C8 RB-proxy rec yds (PFF) | U | 345 | .504 | .504 | 0.0 | .99 | 1.00 | -6.2% | -.033 / +.043 |

The hypotheses with a story behind them (favourite QBs under, trailing receivers over, low-total passing and receiving under) all came out flat or negative.

## Blind-under baseline per market (all scripts, 2024 openers)
The de-vigged under probability already includes the skew. Against the de-vigged line, the under residual is about 0 everywhere:
- Receiving yards: -1.5pp
- Receptions: +0.3pp
- Rushing yards: +2.2pp (p .16)
- Pass yards: -1.2pp
- Carries: +1.2pp
- Tackles+assists: +1.1pp

The one strong residual is **passing TD unders at -7.8pp (p .001)**: overs were the better side in 2024. This was descriptive and not a trial. ROI on blind unders runs from -1% (rushing yards) to -13% because the opener overround is about 6.4%. To profit, a cell needs a residual of about +3.2pp or more.

## Files
- `scripts/nfl_props_script.py` (pre-registration, `inventory`, `dev --ledger`)
- `data/lab/nfl_props_script_results.json`
- `lab/ledger.jsonl`: 8 rows, dataset `nfl_props_script`, bar 0.00625

**Possible follow-up (not a trial):** rushing-yard unders keep coming out positive (C3, C7, and the blind +2.2pp). Any forward-paper test would need its own pre-registered single cell. Even then, one book's 6.4% vig is a high bar.
