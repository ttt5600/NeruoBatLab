# Public NBA game models vs the closing line (2026-10-22)

User ask (10-21): "No way the ESPN model is the best source for modeling NBA games. See what RotoBot says and whether
there are other analytical models."
Script: `scripts/nba_public_models.py`. The rules were pre-registered in its docstring before any join.
Ledger dataset: `nba_public_models`, 7 trials, all logged, so the bar is p < 0.0071. All data is DEV only: NBA 2026 and
the nba_sbr sealed window (2019-20 onward in the sqlite) were not opened.

## Verdict
- **ESPN's model is not the best public model, but no public model beats the market either.**
  - On the same 2,443 games (2021-22 and 2022-23), ESPN's pregame WP is the closest to the closing line. RAPTOR is next,
    and 538 Elo is clearly worst.
  - **All of them lose to the de-vigged close on log loss, in every period tested.**
- **None of the 7 pre-registered betting rules passed.** Six lost money. The one positive rule, Elo at a 10-point gap,
  made +0.92% with p .36, which is noise.
- **Combining a model with the market adds nothing.** A walk-forward stack of market plus Elo ties the market exactly
  (log-loss difference -0.0000, CI -0.0006..+0.0005). This is the "combine decisions" lesson from the encoder work: it
  only pays when the second forecaster carries information the first lacks. The close already contains Elo/RAPTOR/BPI-type
  team strength, plus injuries and lineups those models miss.
- **RotoBot:** a paid AI research assistant ($14.99/wk to $179.99/yr). It has no published or audited pick record and
  makes no accuracy claims in any public material found. It is not a model we can backtest (see notes/rotobot_tracker.md).

## 1. Survey (sources = URLs fetched by the research agent 10-22; UNVERIFIED where marked)
| Model | Predicts | Free archived per-game pregame predictions? | Cost | Published record vs close |
|---|---|---|---|---|
| **538 Elo / CARM-Elo / RAPTOR** | Game win prob plus Elo/RAPTOR ratings | **YES.** The live link projects.fivethirtyeight.com/nba-model/nba_elo.csv now redirects to ABC News. The full file is on the Wayback Machine (snapshot 2023-10-24, md5 b11901fb…), and datahub/Kaggle mirrors have Elo and CARM-Elo. Coverage: Elo 1947-2023, CARM-Elo 2015-16..2018-19, RAPTOR 2018-19..2022-23 | Free | CARM-Elo vs Yahoo spreads 2016-17: 559-560-17 (49.96%), no significant band ([Stanford SA](https://stanfordsportsanalytics.wordpress.com/2017/06/18/in-search-of-a-winning-strategy-comparing-fivethirtyeight-coms-carm-elo-predictions-to-las-vegas-point-spreads/)). RAPTOR blend vs SBR close 2021-22: 137-137-4 ATS ([Walker Harrison](https://www.walker-harrison.com/posts/2022-05-27-evaluating-the-point-spreads-produced-by-538-s-raptor-metric/)). Forecasts stopped before 2023-24; 538 was shut down 2025-03 |
| **ESPN BPI / game WP** | Pregame and in-game WP | **Effectively yes.** The first value of ESPN summary `winprobability[]` gives pregame WP for every past game (we cache 2021-22..2025-26 in data/nba_late_states.parquet). Whether that value is exactly BPI is UNVERIFIED | Free | None published |
| **DARKO** (darko.app) | Daily player DPM, fantasy projections; no game predictions on the homepage | No game archive. A "Time Machine" with player-rating history back to 1996-97 and CSV export exists; whether it gives daily bulk snapshots is UNVERIFIED | Free site | None. The [Power Rank podcast](https://thepowerrank.libsyn.com/kostya-medvedovsky-on-the-darko-nba-player-projections) discusses betting with no numbers |
| **EPM** (dunksandthrees.com) | Player and team EPM, game win probs, box-score and live WP | Paid only. "Team ratings for every date" plus CSV and API. Whether past *predictions* are stored is UNVERIFIED | $24.95, $49.95, or $250/yr with API ([subscribe](https://dunksandthrees.com/subscribe)) | None |
| **inpredictable** | Market-derived team ratings from spreads; in-game WP using margin, time, possession and the Vegas spread | Per-game WP box scores can be browsed 1996-97..2025-26. There is no bulk download, so it was not scraped. The rankings page shows the current snapshot only | Free | None. Its WP model uses the spread as an input, so it is market-anchored by design |
| **Massey / Sagarin / B-Ref SRS** | Rating systems based on score, venue and date | Massey returned 403 (UNVERIFIED). Sagarin's USA Today run ended in 2023, with no daily NBA archive found. B-Ref has only season-level preseason odds | Free | None found |
| **Cleaning the Glass** | Filtered stats, on/off; no predictions | n/a | Paid (price UNVERIFIED) | n/a |
| **numberFire** (FanDuel Research) | DFS and prop projections | No archive found | Free | UNVERIFIED |
| **Action Network projections** | Blend of staff experts' lines, with a "B grade / +3.5% edge" threshold | No | PRO, paid | None found |
| PropGPT / Outlier / PropFinder / BettingPros | Prop projections and hit rates | No | Paid apps | No audited records |
| **RotoBot AI** | Chat, prop analyzer with L10 hit cards, "insights" picks feed, parlays | No | $14.99/wk, $39.99/mo, $179.99/yr | None, and no accuracy claims |
| TeamRankings / AccuScore | Picks and projections | ESPN's odds feed lists both as "providers". **Their rows are relayed market lines, not forecasts:** de-vig correlation with the book median is 0.999 (AccuScore) and 0.995 (TeamRankings), with log loss equal to the market's | Paid | TeamRankings itself says its preseason projections were less accurate than the market ([betIQ](https://betiq.teamrankings.com/articles/preseason-rankings-ratings-explain/)) |
| Academic in-game WP | Stern 1994 Brownian motion; Song et al. 2020 gamma process pinned to the pregame line; Shirley Markov possession model | n/a | n/a | No model-vs-live-sportsbook study found. Our own work (notes/late_game_calibration.md) found Kalshi calibrated late in games, and our probit beat ESPN WP but added nothing to Kalshi |

**Free per-game pregame archives that exist:** 538 (Elo, CARM-Elo, RAPTOR) and ESPN's pregame WP. Everything else is
paid, current-only, or would need a rebuild from rating snapshots. EPM's per-date ratings are the next candidate, at $25-250.

## 2. Backtest (`scripts/nba_public_models.py`, output 2026-10-22)
Markets:
- **A** = SBR sqlite, 2007-08..2018-19. It has an ML only, and its timing is not stated, so it is *probably* the close
  (UNVERIFIED). Joined to 538 on date, home margin and total: 14,393 of 15,507 games; with team1 taken as away, only 62.
- **B** = ESPN lines, 2021-22..2024-25. Real books only, median de-vig. `home_ml` equals `home_ml_close` in 98-100% of rows
  that have both.

Both are regular season with proportional de-vig.

**Accuracy vs the de-vigged market on the same games.** dLL is the model's log loss minus the market's; dLL > 0 means the
model is worse. The CI comes from a paired bootstrap over games.
| Model | Games | Model LL / Brier | Market LL / Brier | dLL [95% CI] |
|---|---|---|---|---|
| 538 Elo, A 2007-08..2018-19 | 14,393 | 0.5997 / 0.2068 | 0.5813 / 0.1990 | +0.0184 [+0.0151, +0.0217] |
| 538 CARM-Elo, A 2015-16..2018-19 | 4,879 | 0.6113 / 0.2115 | 0.5923 / 0.2037 | +0.0190 [+0.0133, +0.0248] |
| 538 RAPTOR, A 2018-19 | 1,225 | 0.6139 / 0.2124 | 0.5934 / 0.2038 | +0.0204 [+0.0079, +0.0331] |
| 538 Elo, B same games (2021-23) | 2,443 | 0.6523 / 0.2288 | 0.6144 / 0.2129 | +0.0379 [+0.0275, +0.0486] |
| 538 RAPTOR, B same games | 2,443 | 0.6297 / 0.2188 | 0.6144 / 0.2129 | +0.0153 [+0.0076, +0.0236] |
| **ESPN pregame WP, B same games** | 2,443 | 0.6268 / 0.2178 | 0.6144 / 0.2129 | **+0.0124 [+0.0053, +0.0197]** |
| mean(RAPTOR, ESPN), B same games | 2,443 | 0.6227 / 0.2162 | 0.6144 / 0.2129 | +0.0083 [+0.0024, +0.0144] |
| ESPN pregame WP, B 2021-22..2024-25 | 4,911 | 0.6108 / 0.2113 | 0.5976 / 0.2058 | +0.0131 [+0.0085, +0.0179] |

ESPN WP loses to the market in each of the 4 seasons by itself: +0.0125, +0.0119, +0.0149 and +0.0133, and every CI
excludes 0.

**Does a model add information to the market?** We fit the stack y ~ logit(market) + logit(model) in-sample and read the
model's coefficient:
- Elo: +0.09 [0.00, +0.18] over all of A. That splits into +0.16 [+0.03, +0.30] in 2007-13 and +0.02 [-0.11, +0.15] in
  2013-19, so the information that was there has decayed.
- CARM-Elo: -0.01. RAPTOR: +0.14 [-0.11, +0.40]. ESPN WP: +0.04 to +0.08, with all CIs spanning 0.

Walk-forward, the stack ties the market on A (dLL -0.0000) and is slightly worse on B (+0.0015 [+0.0004, +0.0027]).

**Pre-registered betting trials.** Each rule bets 1u flat at the market ML, at most one bet per game. The bar is p < 0.0071.
| Trial | Bets | ROI | Units [95% CI] | p | Seasons positive |
|---|---|---|---|---|---|
| M1 Elo gap ≥ .05 (A) | 7,636 | -0.86% | -65.6 [-313.9, +179.3] | .71 | 6/12 |
| M2 Elo gap ≥ .10 (A) | 3,473 | +0.92% | +32.0 [-137.5, +205.2] | .36 | 7/12 |
| M3 CARM-Elo gap ≥ .05 (A) | 2,582 | -2.05% | -53.0 [-183.2, +75.2] | .78 | 1/4 |
| M4 RAPTOR gap ≥ .05 (B, 2021-23) | 1,374 | -2.11% | -29.0 [-103.6, +46.7] | .77 | 1/2 |
| M5 ESPN WP gap ≥ .05 (B, 2021-25) | 2,374 | -3.59% | -85.3 [-210.0, +39.4] | .91 | 1/4 |
| M6 walk-forward stack(market, Elo) gap ≥ .03 (A) | 536 | -3.51% | -18.8 [-58.1, +20.1] | .83 | 2/11 |
| M7 walk-forward stack(market, ESPN WP) gap ≥ .03 (B) | 376 | -1.46% | -5.5 [-41.9, +29.2] | .62 | 0/1 (all bets fell in one season) |

## 3. What it would take
- **The market is the strongest public "model".** Every free model's disagreement with the close is mostly error. Betting
  that gap (M1-M5) returns about the vig. That matches all ~45 earlier NBA closing-line angles.
- **A model only adds value if it has information the close lacks.** Candidates:
  - a (a) timing edge: beat the close by betting earlier lines. The test is CLV versus the open, not the close; ESPN lines
    have `*_open` for 2024+.
  - (b) a faster read on injury and lineup news than the market. The report schedule is in research_prop_models.md.
  - (c) a thin market where the line is soft: props, alt lines, Kalshi early prices.
- Stacking public ratings onto the close is a fair test of (c)-type information, and on game moneylines it is empty.
- **Next cheap test, if wanted:** compare the same models to the **opening** line rather than the close. ESPN 2024-25 has
  `home_ml_open`. A model that beats the open and agrees with where the close went is a CLV source. This needs a new
  pre-registration.
- **Caveats:**
  - RAPTOR and CARM-Elo rows may be 538's re-computations rather than the numbers as published. Re-computation would
    flatter them, and they still lose.
  - The timing of the SBR ML is not stated in the file.
  - ESPN WP p0 is the first in-game value. Whether it equals BPI is UNVERIFIED.
