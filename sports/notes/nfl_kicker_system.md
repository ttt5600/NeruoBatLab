# NFL kicker OVER 3.5 system (@linemakersports) -- DEV test, 2026-10-29

**Verdict: no edge. The system's 84.7% hit rate is the base rate (83.0%) plus the team total. It "beats" -150 to -200 only because no book prices an 85% event at -200. The fair price is about -550 (empirical) to -810 (market-implied).**

Pre-registered in `scripts/nfl_kicker_system.py` (docstring). One trial, logged to `lab/ledger.jsonl` (dataset `nfl_kicker_system`). DEV 2018-2024 regular season only; 2025-26 not scored. No orders were placed.

## Rules as tested
- (a) Opponent's season-to-date kicker points allowed per game >= 7.
- (b) Own team's red-zone trips per game >= 3. "Red-zone attempts" is undefined in the claim, so I read it as drives with a scrimmage or FG play at the opponent's 20 or closer.
- (c) Kicker's season-to-date FG% >= 90%.
- All features use prior games only. Kicker, team and opponent each need >= 4 prior games.
- Hit = 3*FGM + XPM >= 4.

## 1. Hit rates (Wilson 95%)
| set | n | hit rate | CI | mean pts |
|---|---|---|---|---|
| qualifying | 354 | 84.7% | 80.6-88.1 | 7.6 |
| all eligible | 2574 | 83.0% | 81.5-84.4 | 7.1 |
| non-qualifying | 2220 | 82.7% | 81.1-84.3 | 7.0 |

- **Matched baseline.** I fit a logistic on implied team total (nflverse spread and total). It predicts 85.5% for the qualifying games. Residual: -0.7pp, one-sided p = .65.
- **By season.** Raw rate beat the base in 5 of 7 seasons. The residual vs team total is positive in only 2 of 7 (2023 +1.9pp, 2024 +2.8pp). n per season is 24-74.
- **The "wins 90%" claim.** 90% is outside the CI (upper bound 88.1%).

## 2. Break-even price
- 84.7% hit rate supports **-555** at best (all eligible games: -489).
- At -150 to -200 the ROI would be +27% to +41%. That figure is meaningless unless those prices actually exist.

## 3. Market (ESPN BET 2024, `Total Kicking Points`)
- ESPN core odds does carry this market. The main line opens at **4.5-9.5 (mostly 6.5/7.5), never 3.5**, so we have no over-3.5 prices.
- Matched 324 priced kicker-games to pbp. The two sources agree on actual points 98% of the time.
- **Market-implied P(>= 4).** I fit a logistic on the de-vigged opener P(over) plus the line, across all priced games. The qualifying games (n = 52) have implied P = 89.0% vs actual 88.5%: residual -0.6pp, p = .55. Fair price is about **-810**.
- Betting OVER the main line at openers returned +0.6% ROI on qualifying games (n = 52) vs +1.8% on all priced games. No signal.

## Price assumption
- "Kicker over 3.5 at -150 to -200" is the brief's assumed range. I could not verify it from any source.
- Main lines sit at 6.5-7.5, so a 3.5 alt is roughly 3 points below the median. A book would price it near the -500 to -800 implied above.
- If any book really did hang -200 on over 3.5, every kicker would be a bet, and the filter would add nothing.
- The system's conditions select high-total games, and the market already prices those.
