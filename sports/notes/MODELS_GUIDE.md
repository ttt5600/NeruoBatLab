# MODELS GUIDE: what our models do, what they eat, how they compete (2026-10-26)

Written for the question "what is our model actually doing?". Every number is copied from a file (cited next to it)
or comes from a fresh **dev-only** computation saved in `results/models_guide_diag.json` (scripts:
`/Users/jonathanwang/.claude/jobs/588c8b26/tmp/nba_guide_diag.py`, `nba_guide_diag2.py`; they read only
`data/nba_simple_model/wf.pkl` = 2023-24 + 2024-25 and dev ESPN lines). No sealed data was read for this guide; the
one sealed number quoted (L32) comes from the already-written `results/nba_l32_sealed.json`. Arithmetic I did myself
is marked "(arithmetic)". Paths are relative to `sports/`.

## 0. The whole project on one page

- **Every pregame model we have is the same shape**: a slowly-updating team rating, plus one or two
  "who is actually playing" terms (NBA players, NFL quarterback, NHL goalie), plus a handful of style terms, combined
  by a small linear regression. Nothing opaque, every coefficient printed below.
- **Every one of them beats the opening line and loses to the closing line.**

| Model | Leads the opener? (CLV) | Beats the close? | Money at normal vig? | Status |
|---|---|---|---|---|
| NBA spreads | +0.79 pts | no (stack weight ≈ 0) | no (−0.6% at −110) | paper |
| NBA totals | +2.00 pts; +8.7% no-vig dev | no | sealed L32 +6.4% p .054: FAIL, inconclusive | forward logger |
| NFL spreads / totals | +0.76 / +0.81 pts | no | no (totals +6.6% at −110, p .014, fails the .0036 bar) | forward paper (L33) |
| NHL moneyline / totals | +1.04 prob-pts / +0.073 goals | no | no (ESPN BET ~4% vig) | dead as a bet (L34) |
| CFB totals | +1.02 pts; +7.4% no-vig | no | no at −110 (+2.6%) | lead (L36) |
| NBA in-game (big leads, late-game, comebacks) | n/a | Kalshi ≥ our model | no | L28 + L37 forward paper |
| **Kalshi weather consensus-reject** | n/a | n/a | **yes: sealed +1.45% / +0.78%; maker +2.22%** | **LIVE** |

- The lesson in one sentence: **our sports models know roughly what the market will know at tip-off, a few hours early.**
  That is worth money only if we can bet the *first* price at *low* vig. The only rule that is already making money
  (weather) is not a prediction model at all but a filter: "five forecasters all agree this bracket is impossible".

### Glossary (used throughout)
| Term | Meaning |
|---|---|
| Opener / open | the first line a book posts (ESPN's stored "open"; exact timestamp undocumented) |
| Close | the line at tip-off: the market's final, best-informed price |
| CLV (closing line value) | how far the line moved toward our side between our bet and the close. Positive CLV = we were early and right about where the market went |
| No-vig price | the book's two prices rescaled to sum to 100% (`sports/market.py devig_two_way`). Our proxy for "a resting maker order at fair" |
| STRICT / ACTUAL / avail | three rules for "who plays tonight" (section 1.5). Only STRICT is lookahead-free in backtests |
| Stack | the small final OLS that weights the model pieces |
| Walk-forward | season S is predicted by a model fit only on seasons before S |
| Dev / sealed | dev = the seasons we may look at; sealed = held-out seasons scored once by a pre-registered script |
| Trial / bar | each pre-registered test is one trial; the pass bar is 0.05 / (number of trials in that dataset) |

## 1. NBA simple model (pregame spread, win probability, total)

Files: `scripts/nba_simple_model.py` (pre-registration in the docstring; `build` / `backtest` / `today`),
`scripts/nba_simple_live.py` (live input plumbing), `results/nba_simple_model_frozen.json` (every frozen coefficient),
`results/nba_simple_model.json` (dev trials), `notes/nba_simple_model.md`, `results/nba_l32_sealed.json`,
fresh diagnostics in `results/models_guide_diag.json`.

### 1.1 What it predicts
For every game, three numbers: the **home margin** (home points minus away points), **P(home wins)**, and the
**total points**. It never looks at a betting line when it makes them. The line only enters afterwards, when we decide
whether the model disagrees with the market by enough to bet.

### 1.2 The pipeline in one picture
```
 past box scores ─┬─> TEAM RATING ridge (M0 margin, T0 total) ─────────────┐
                  ├─> PLAYER VALUE "BPM-lite" ─> team strength tonight ────┼─> STACK (small OLS) ─> margin, total
                  │       (who is expected to play: STRICT / ACTUAL / avail)│        │
                  └─> STYLE stats ─> 5 spread matchups + 5 totals terms ────┘        └─> P(home) = Φ(margin / σ)
```
Every arrow uses only games dated before tonight (state updates in Eastern-date batches).

### 1.3 Piece 1: team rating (M0 for the margin, T0 for the total)
- **M0** (`ridge_preds` in `scripts/nba_player_sim.py`): one regression over all past games,
  `home margin = home edge + rating[home] − rating[away] + b2b_home + b2b_away + rest difference + travel_home + travel_away`.
  - **Half-life 60 days**: a game two months ago counts half as much as last night's.
  - **Ridge λ = 5** on the 30 team ratings only (pulls ratings toward 0 = average, so a team with a few lucky blowouts
    is not over-rated). The home/rest/travel terms are effectively unpenalised.
  - Rest is capped at 4 days (3 in the difference); travel is great-circle km/1000 from the previous game's city.
- **T0**: the totals analogue. Each team-game gives a row `points scored = league + offence[team] + defence_allowed[opp] + home`.
  T0 = predicted home points + predicted away points. Same half-life and λ.
- What this means: M0/T0 are "how good has this team been lately, adjusted for opponents", with no player information.

### 1.4 Piece 2: player value ("BPM-lite")
A minutes-weighted linear regression over player-seasons with ≥ 300 minutes: on-court plus-minus per 48 minutes
regressed on 11 per-48 box stats. Live/sealed use the all-dev fit (n 1,689 player-seasons, R² 0.28;
`results/nba_simple_model_frozen.json` → `bpm`). Points of on-court margin per 48 for one extra unit per 48:

| Stat | Coef (se) | Reading |
|---|---|---|
| points | +1.20 (.09) | read with FGA: a made 2 is +2×1.20 − 1.23 ≈ +1.2; a miss is −1.23. **Efficiency, not volume** |
| FGA | −1.23 (.10) | every shot attempt costs a possession |
| FTA | −0.45 (.12) | |
| 3PM | +0.77 (.16) | on top of its 3 points: spacing value |
| ORB | +0.09 (.15) | ≈ 0 |
| DRB | +0.34 (.07) | |
| assists | +0.79 (.08) | creation |
| turnovers | **−2.54** (.23) | the most expensive event |
| steals | **+1.81** (.26) | a proxy for defensive activity |
| blocks | +0.34 (.22) | |
| fouls | +0.04 (.14) | ≈ 0 |
| intercept | −8.37 | |

- **Form and shrinkage.** A player's per-minute rates are decayed sums over all his past games (**half-life 180 days**,
  so last season acts as the prior for this one), then shrunk toward **replacement-level rates with 500 pseudo-minutes**:
  `rate = (his decayed stat sum + 500 × replacement rate) / (his decayed minutes + 500)`. A player with 100 recent
  minutes is mostly replacement; one with 2,000 is mostly himself.
- **Replacement level** = minutes-weighted average rates of 100-500-minute player-seasons. Plugged into the formula it
  rates **−2.70 per 48** (computed from the frozen file), and scores 0.36 points per minute (`repl_ppm`). A brand-new
  player starts here.
- R² 0.28 means box score explains about a quarter of plus-minus. That is normal for BPM-type models and is why the
  stack below shrinks this term rather than trusting it 1:1.

### 1.5 Piece 3: team strength tonight, and the three availability rules
- `strength = Σ projected minutes × player rating / 240`. Projected minutes = EWMA(0.3) of his past minutes, rescaled
  so the set sums to 240, with a 42-minute cap. `O` = Σ minutes × points-per-minute = expected points from that set.
- **FULL roster** = players whose latest team is this team and who appeared in any of its last 10 games.
- The model feeds two features per side: `dPfull` (full-roster strength difference, home − away) and `dAbs` (how much
  strength each side loses from the full roster to tonight's set, home loss minus away loss). Totals use `Ofull`
  (sum) and `oAbs` (expected points lost).
- **Who is "tonight's set"?** This is the single most important modelling choice:

| Rule | Tonight's set | Lookahead? | Used in |
|---|---|---|---|
| **STRICT** | players who played the team's previous game | none (pure) | pre-registered model, sealed L32, `model_*` live (plus injury-feed OUT removals) |
| **ACTUAL** | players who really played tonight | yes: proxy for the injury report at tip | backtest upper bound (T6-T8); fits `M2_act`/`T2_act` |
| **avail** (live only) | full roster minus ESPN-feed Out/Doubtful | uses the live injury report | `model_*_avail` in `nba_simple_live.py`; NOT the sealed model |

- STRICT's blind spot: a star who played last night but rests tonight (back-to-backs, end of season) is assumed to
  play. The live path patches some of this with the ESPN injury feed (`scripts/nba_simple_live.py` docstring).

### 1.6 Piece 4: playing style
Team style stats are decayed sums over the team's past games (**half-life 60 days**): pace, 3PA rate (own/allowed),
FT rate (own/allowed), ORB%/DRB%, TOV% (own/forced), 3P% (own/allowed), and paint-points share (own/allowed). They
are combined into 10 pre-chosen matchup terms, each standardised on the training fold:

| Term | Formula (home vs away) | Idea |
|---|---|---|
| S1 paint_def | home's opp. paint share − away's opp. paint share | teams that leak the paint (L30) |
| S2 three | z(h 3PAr)·z(a opp 3PAr) − z(a 3PAr)·z(h opp 3PAr) | volume shooters vs defences that allow threes |
| S3 reb | z(a ORB)·z(h DRB) − z(h ORB)·z(a DRB) | offensive vs defensive rebounding |
| S4 tov | z(h forced TOV)·z(a TOV) − mirror | ball-hawks vs careless teams |
| S5 ft | z(h FTr)·z(a opp FTr) − mirror | foul-drawers vs foul-prone defences |
| U1 pace | z(h pace) + z(a pace) | fast games score more |
| U2 3P% | h 3P% + a 3P% | does hot shooting persist? (L31) |
| U3 opp 3P% | h opp 3P% + a opp 3P% | does "good 3-pt defence" persist? |
| U4 3PA rate | z(h 3PAr) + z(a 3PAr) | three-point volume |
| U5 b2b | number of teams on a back-to-back | fatigue |

### 1.7 Piece 5: the stacks (how the pieces are weighted)
Small OLS regressions fit walk-forward by season. Frozen fits below are on all dev scored games 2022-23..2024-25
(n 3,445; `results/nba_simple_model_frozen.json`). Style and player terms are **points per training SD**; the last
column converts to raw units (computed from the frozen `sc` scales).

**Spread stack M2 (STRICT), residual σ = 13.60**

| Term | Coef / SD (se) | Raw meaning |
|---|---|---|
| const | +0.41 (.27) | |
| mu0 (team rating) | +0.73 (.06) | the ridge margin, shrunk by ~27% |
| dPfull | +1.92 (.36) | +1.52 pts of margin per point of full-roster strength edge |
| dAbs | +1.35 (.24) | **+2.62 pts per point of strength lost to absences** |
| S1 paint_def | **−1.13 (.24)** | −0.40 pts per percentage point of extra paint share allowed |
| S2 three | +0.24 (.23) | noise |
| S3 reb | +0.03 (.23) | noise |
| S4 tov | −0.07 (.23) | noise |
| S5 ft | +0.61 (.23) | weak positive |

M1 = the same without S1-S5 (const +0.24, mu0 +0.81, dPfull +1.53, dAbs +1.37; σ 13.65). M0 = mu0 alone.
ACTUAL variant M2_act: dAbs +2.06 (.23), σ 13.51 (absences matter more when you know who really sat).

**Totals stack T2 (STRICT), residual σ = 18.35**

| Term | Coef / SD (se) | Raw meaning |
|---|---|---|
| const | +54.1 (20.0) | |
| tot0 (team totals ridge) | +0.77 (.09) | |
| Ofull | +1.89 (.39) | +0.28 per expected point from the full roster |
| oAbs | +0.84 (.31) | +0.13 per expected point lost |
| U1 pace | +2.00 (.59) | +1.42 per unit of summed pace z |
| U2 3P% (own, sum) | **−1.17 (.42)** | −0.45 pts per percentage point of summed 3P%: hot shooting regresses |
| U3 opp 3P% (sum) | **−1.41 (.36)** | −0.67 pts per percentage point: "good 3-pt defence" regresses too |
| U4 3PA rate | +0.33 (.32) | noise |
| U5 b2b | −0.23 (.31) | noise |

T1 (no style): const +23.4, tot0 +0.90, Ofull +1.03, oAbs +0.85; σ 18.51.

Note: `notes/nba_simple_model.md` section 1 prints an older T2/M2 table (e.g. tot0 +0.93, U1 +0.77) labelled "fit on
2023-24 + 2024-25". The frozen file the live and sealed code actually use (fit 2022-23..2024-25, n 3,445) is the one
above; the sealed run checked it matches to 1e-6 (`results/nba_l32_sealed.json` → `frozen_match: true`).

### 1.8 From numbers to prices
- **Margin** = stack output. **P(home)** = Φ(margin / 13.60) (normal errors; `model_home_wp` in the live script).
- **Total** = T2 output. P(over line L) is not used for decisions; the bet rule is a point gap.
- **Bet rules (pre-registered)**: spread side when |model margin − market margin| ≥ 2.0; total side when
  |model total − line| ≥ 3.0. Flat 1 unit.

### 1.9 Behaviour on dev (fresh diagnostics, 2023-24 + 2024-25 walk-forward, n 2,286)

**(a) Ablation: what each layer buys** (`results/models_guide_diag.json` → `ablation_spread`, `ablation_total`; CLV
and no-vig columns from `results/nba_simple_model.json` → `descriptive.splits`). Win-prob log loss uses Φ(margin/σ)
with each fold's training σ; the market rows use the same σ on the spread, plus the de-vigged median closing moneyline.

| Spread model | MAE vs result | Mean \|model − close\| | Log loss | Brier | CLV vs open (gap ≥ 2) | No-vig ATS |
|---|---|---|---|---|---|---|
| M0 team rating | 11.05 | 3.00 | 0.605 | 0.209 | +0.72 (n 1,285) | +1.0% |
| M1 + players (STRICT) | 10.96 | 2.62 | 0.599 | 0.207 | +0.80 (n 1,106) | −0.7% |
| M2 + style (STRICT) | 10.96 | 2.86 | 0.599 | 0.207 | +0.79 (n 1,262) | +2.3% |
| M1 ACTUAL (lookahead) | 10.91 | 2.28 | 0.592 | 0.203 | | |
| M2 ACTUAL (lookahead) | 10.92 | 2.56 | 0.592 | 0.203 | +1.03 / +1.75 (no-news / news) | +3.3% / +8.3% |
| opening spread | 10.82 | | 0.596 | 0.204 | | |
| closing spread | 10.56 | | 0.579 | 0.198 | | |
| closing moneyline (de-vigged) | | | 0.578 | 0.197 | | |

| Totals model | MAE vs result | Mean \|model − close\| | Mean model − close | No-vig O/U vs open (gap ≥ 3) |
|---|---|---|---|---|
| T0 team ridge | 14.75 | 3.78 | +0.41 | +4.9% (n 1,078) |
| T1 + players | 14.85 | 4.15 | +1.96 | +2.6% (n 1,151) |
| T2 + style | 14.73 | 4.12 | +2.28 | **+8.7%** (n 1,190) |
| T2 ACTUAL | 14.67 | 3.93 | +2.45 | +12.8% (lookahead) |
| opening total | 14.73 | | | |
| closing total | 14.19 | | | |

What this means:
- Players help the spread a little (MAE 11.05 → 10.96, log loss .605 → .599); style adds nothing to accuracy on
  spreads. Knowing who actually plays helps more than any style term (10.92, .592).
- **Every version is worse than the opening spread on accuracy, yet it still predicts where the opener moves.**
  That is not a contradiction: the model and the opener make *different* errors, so a blend beats either; the stack
  weight given the opener is +0.32 to +0.48, given the close ≈ 0 (`results/nba_simple_model.json` → `b_model`).
- On totals T2 roughly ties the opener on MAE (14.73 vs 14.73) and loses clearly to the close (14.19).
- **Level bias:** T1/T2 sit **+2.0 to +2.3 points above the closing total on average**. The player-points term
  (Ofull) is what introduces it (T0 is only +0.41). Section (c) shows where it comes from.

**(b) Calibration of P(home), M2 STRICT, by decile** (`calibration_M2_strict`)

| Decile | n | Model P(home) | Actual home win rate | Closing ML P(home) |
|---|---|---|---|---|
| 1 | 230 | .225 | .170 | .217 |
| 2 | 229 | .338 | .301 | .328 |
| 3 | 230 | .416 | .430 | .416 |
| 4 | 229 | .483 | .459 | .481 |
| 5 | 230 | .538 | .543 | .546 |
| 6 | 229 | .591 | .563 | .592 |
| 7 | 229 | .641 | .655 | .651 |
| 8 | 230 | .697 | .687 | .694 |
| 9 | 229 | .763 | .777 | .773 |
| 10 | 230 | .855 | .870 | .847 |

Logistic recalibration slope 1.11 (se 0.06), intercept −0.07. What this means: the probabilities are honest to within
a few points in every decile; if anything slightly *under*-confident at the extremes (slope > 1), mostly in the
bottom decile (model 22.5% vs 17.0% actual). Calibration is not where the model loses to the market; sharpness is
(log loss .599 vs .578).

**(c) Error by month** (`by_month`; dev scoring starts once both teams have 5 games, so October is excluded and
"early season" here means November)

| Month | n | Spread MAE model / close / open | \|model − close\| | Total MAE model / close / open | Mean T2 − close |
|---|---|---|---|---|---|
| Nov | 405 | 10.36 / 10.00 / 10.50 | 2.75 | 14.97 / 14.47 / 15.07 | +1.96 |
| Dec | 396 | 10.40 / 10.21 / 10.43 | 2.74 | 14.09 / 13.92 / 15.01 | +0.90 |
| Jan | 455 | 11.81 / 11.31 / 11.58 | 2.66 | 15.21 / 14.62 / 15.26 | +1.10 |
| Feb | 350 | 10.96 / 10.73 / 10.80 | 2.75 | 14.52 / 13.78 / 14.20 | +0.82 |
| Mar | 468 | 10.90 / 10.60 / 10.69 | 2.83 | 14.69 / 13.88 / 14.04 | **+4.80** |
| Apr | 220 | 11.41 / 10.22 / 10.84 | **3.94** | 14.55 / 14.27 / 14.85 | **+4.73** |

League scoring by month vs the model (`total_level_by_season_month`), the key rows:

| Season-month | Actual mean total | Opening total | T2 | T0 |
|---|---|---|---|---|
| 2023-24 Feb | 227.9 | 231.7 | 233.3 | 231.4 |
| 2023-24 Mar | 222.6 | 223.2 | 230.5 | 228.8 |
| 2023-24 Apr | 224.3 | 220.9 | 228.7 | 226.7 |
| 2024-25 Mar | 229.8 | 228.4 | 229.6 | 227.9 |

What this means:
- **The early-season problem is not visible in dev spreads** (November is the model's best month relative to the
  close). The dev scoring already skips each team's first 5 games, and in dev the "full roster" is built from real
  box scores. The early-season problem is a *live* plumbing issue: in October the STRICT roster reaches back to last
  April (tanking-era call-ups) and new signings have no history (`notes/nba_simple_model.md` section 4).
- **The real seasonal failure is April spreads** (|model − close| 3.94 vs ~2.75 elsewhere; MAE 11.41 vs close 10.22):
  end-of-season rest that STRICT cannot see (see (d)).
- **March-April totals**: in 2023-24 league scoring fell ~5 points from February to March and the market re-priced
  within weeks (opener 223.2); the model's 60-day half-life kept T2 at 230.5, ~8 points high. A slow-decay model lags
  a league-wide level shift. In 2024-25 there was no shift and no bias (229.6 vs 229.8).
- November totals CLV is large (+4.06 pts) but ROI ≈ 0 (`T5_by_month`): the line moves our way but the results did not
  follow in that month; noise at n 233.

**Is the totals edge just "bet overs"?** (post-hoc, descriptive, not trials)
- Betting every over at the opener: +0.9% no-vig (n 2,286); every under −1.0% (`naive_all_overs/unders`).
- T5 bets were 78% overs (932 of 1,190). Subtracting the model's own trailing-30-day mean gap to the opener and
  re-applying the ≥ 3 rule gives 988 bets, **50% overs, +7.2% no-vig, CLV +2.21 pts** (`T5_debiased_posthoc`).
- What this means: the level bias is real and costs accuracy, but the totals signal is mostly *relative* (which
  games), not "the league will score more". A bias-corrected T2 is the obvious v2 (needs a new pre-registration).

**(d) The 10 biggest spread disagreements with the close** (`top10_spread_misses_vs_close`; gap = model home margin
− market home margin)

| Date | Game | Model | Close | Actual | Gap | Gap with ACTUAL availability | Who sat (top-5 minutes, played previous game) |
|---|---|---|---|---|---|---|---|
| 2025-04-09 | BOS @ ORL | ORL −10.1 | ORL +7.5 | ORL +20 | −17.6 | −12.3 | BOS: Tatum, White, Holiday, Porzingis |
| 2025-04-13 | IND @ CLE | CLE +10.1 | CLE −6.5 | CLE −8 | +16.6 | +7.7 | CLE: Mobley, Garland, Strus (Mitchell, Hunter missed previous game); IND: Haliburton, Siakam, Turner, Nembhard missed previous |
| 2025-03-07 | POR @ OKC | OKC +16.6 | OKC +1.5 | OKC +18 | +15.1 | +11.2 | OKC: Gilgeous-Alexander, J. Williams, Dort, Wallace |
| 2025-04-13 | DET @ MIL | MIL +4.0 | MIL −9.5 | MIL +7 | +13.5 | +2.9 | MIL: Antetokounmpo, Portis, Lopez, Porter Jr.; DET: Cunningham, Schroder |
| 2025-04-13 | CHA @ BOS | BOS +22.7 | BOS +10.5 | BOS +7 | +12.2 | +10.3 | BOS: Tatum, White, Holiday, Horford |
| 2025-04-13 | LAL @ POR | POR −6.6 | POR +5.5 | POR +28 | −12.1 | +0.2 | LAL: Reaves, Doncic, Hachimura, James, Finney-Smith |
| 2025-03-17 | DEN @ GS | GS +0.8 | GS +12.5 | GS −9 | −11.7 | −1.2 | DEN: Jokic, Murray, Braun |
| 2023-11-30 | MIL @ CHI | CHI +1.1 | CHI −10.5 | CHI +7 | +11.6 | +8.7 | CHI: DeRozan, LaVine |
| 2025-03-17 | IND @ MIN | MIN −1.0 | MIN +10.5 | MIN −2 | −11.5 | −1.8 | IND: Haliburton, Siakam, Nesmith, Turner |
| 2024-12-07 | OKC @ NO | NO −18.8 | NO −7.5 | NO −10 | −11.3 | −11.0 | none |

What caused them:
- **9 of 10 are rest/absence news that STRICT cannot see**: 5 are the last week of the 2024-25 regular season
  (April 9-13, seeding locked, starters rested), 3 are mid-March mass rests. Knowing who actually sat shrinks the
  average gap in the 50 biggest disagreements from 10.4 to 6.6 points (`gap_summary`).
- When the model *does* know (ACTUAL), some gaps stay large (OKC −4 starters still +11.2; BOS +10.3): for whole-lineup
  rests the per-player absence term is too small. Across all 134 games with a big availability change, though, the
  model's STRICT→ACTUAL move tracks the market's move one-for-one (regression slope 1.01; `absence_sizing`), so
  ordinary single-player absences are sized right; it is the "rest four starters" games that break the linear model.
- **No trade case shows up in the top 10.** The one non-availability miss (OKC @ NO, Dec 2024) is a pure rating
  disagreement: the model had OKC 18.8 better, the market 7.5. Pelicans injuries before that date would show up as
  "missed previous game", and none did, so this one is unexplained by our data.
- The top-10 **totals** misses (`top10_total_misses_vs_close`) are all March-April 2023-24 and 8 of 10 involve
  Atlanta (Trae Young injured from late February 2024): model ~19-26 points above a ~215-222 close, and both
  STRICT and ACTUAL agree. That is the stale scoring-level effect from (c) plus Atlanta's high-scoring early-season
  defence ratings decaying too slowly.

### 1.10 How it competes, and the scoreboard
| Test | Result | Source |
|---|---|---|
| Spread CLV vs opener (T1) | **+0.79 pts**, 1,262 bets, p < 1/4000, both seasons | `results/nba_simple_model.json` |
| Spread ATS at no-vig (T2) / −110 (T3) | +2.3% p .21 / −0.6% | same |
| Totals CLV vs opener (T4) | **+2.00 pts**, 1,190 bets | same |
| Totals O/U at no-vig opener (T5) | **+8.7%** (55.5%), p .001: dataset bar yes, whole-ledger bar no | same |
| Vs the close, spreads (T6) / totals (T7) | −1.2% / +3.2% p .14 | same |
| Absence-news games, ACTUAL vs opener (T8) | CLV +1.75 (lookahead upper bound) | same |
| **L32 SEALED 2025-26** (T2 STRICT vs DK/ESPN BET opener, ≥ 3 pts) | 575 bets, **+6.4% no-vig, CI [−1.6%, +14.4%], p .054 → FAIL (inconclusive at ~25% power)**; −110 +1.6%; CLV +1.20 pts; 48.2% vs the close; 480 overs / 95 unders | `results/nba_l32_sealed.json` |

What this means: the NBA model is an **opener-chaser**. It sees roughly what the market will know by tip-off and gets
there early, so its value is entirely in betting the first-posted number at low vig. Against the closing line it adds
nothing. The sealed test moved in the right direction (+6.4%, CLV +1.2) but was too small to prove it; the forward
logger `scripts/nba_open_totals.py` is now collecting the bettable version.

## 2. NFL simple model (lead L33)

### What it predicts
Two numbers per game, each in points: the **home margin** (the spread) and the **total points** (the total)
(`scripts/nfl_simple_model.py` docstring). It comes in three nested versions. **A** uses team EPA only, **B** adds a
quarterback term, and **C** adds style terms. The live picks are model B for spreads and model C for totals
(`notes/nfl_simple_model.md`).
**The model does not produce a win probability, and it has no sigma or normal-CDF step.** The script outputs only
margin and total (no `prob`, `sigma` or `cdf` appears in `scripts/nfl_simple_model.py`). A moneyline price is not
documented.

### Inputs (stats and variables)
The source is nflverse play-by-play. It uses REG and POST plays that have an EPA value, and drops kneels and spikes
(`scripts/nfl_simple_model.py`). Every feature uses only earlier games.
- **Team EPA (model A).** Four rates: offence EPA per dropback, offence EPA per designed run, defence EPA allowed per
  dropback, and defence EPA allowed per designed run (`scripts/nfl_simple_model.py`, `RATES`).
- **QB (model B).** A QB rating is the shrunk EPA per dropback plus 0.02 x the shrunk CPOE. `qb_adj` = today's
  starter's rating minus the dropback-weighted rating of the QBs whose plays built the team's season pass EPA
  (`scripts/nfl_simple_model.py` docstring).
  - In dev, the starter comes from the nflverse schedule.
  - Live, it comes from the nflverse depth chart, and drops to the next QB if the ESPN injury report lists QB1 as
    Out, IR, Doubtful or Suspended (`notes/nfl_simple_model.md`).
- **Style (model C).** Five features (`scripts/nfl_simple_model.py` docstring):
  - pass rate over expected (PROE), measured in neutral situations (win prob .2-.8, downs 1-3)
  - pace (offensive plays per game)
  - explosive-play rate (passes of 20+ yards, runs of 10+), made and allowed
  - pressure (sack or QB hit per dropback), generated and allowed
  - the dev fit also had six pre-chosen matchup interactions (I1-I6). The live refit drops them, because all had
    p > .27 in dev (`notes/nfl_simple_model.md`).
- **League level.** `lg_pts` is last season's mean total points (`scripts/nfl_simple_model.py`, `build_all`).
- **Home field.** A constant, plus an `hfa_off` term that is -1 at neutral sites (`fit_predict`).
- **Not built:** key non-QB players (WR1, LT, edge rusher) (`notes/nfl_simple_model.md`).

### How it turns inputs into a price
1. **Shrink each team rate toward a prior.** The rate is `(this season's sum + K x prior) / (this season's count + K)`.
   The prior is `league + 0.5 x (team last season - league)`. So a team starts each season halfway back to the
   league average, and its own games take over as they pile up (`scripts/nfl_simple_model.py`).
   - K is the number of "pretend" plays at the prior value: 250 dropbacks for pass EPA, 200 rushes for run EPA,
     200 neutral plays for PROE, 250 plays for explosive rate, 250 dropbacks for pressure, and 4 games for pace.
2. **Rate the QB the same way, with memory across seasons.** Each QB's totals are multiplied by 0.6 at every new
   season. They are shrunk with K = 150 dropbacks toward a "replacement" prior of league EPA/dropback - 0.10. So an
   unknown backup is assumed to be clearly below average (`scripts/nfl_simple_model.py`: `QB_K, QB_DECAY, QB_REPL,
   CPOE_W = 150, 0.6, -0.10, 0.02`).
3. **Build matchup features.** Each spread feature is a home-minus-away difference (for example
   `off_pass = 10 x (home off pass EPA - away off pass EPA)`). Each total feature is a home-plus-away sum. The x10
   scaling means a coefficient reads as points per 0.1 EPA (`_derive`).
4. **Combine with linear regression.**
   - **Legacy fit.** Plain OLS, walk-forward: test season s is fit on 2007..s-1 (`scripts/nfl_simple_model.py`).
   - **Live "modern" fit.** Weighted least squares on 2020-2025. Each season gets weight `0.5^((2025 - season)/2)`, a
     2-season half-life, so 2025 = 1, 2023 = .5, 2021 = .25 and 2020 = .18 (`scripts/nfl_simple_model.py`,
     `HALF_LIFE`).
5. **Make a bet decision against the opener.** Bet the model's side when |model - open| >= 2.0 points (spread) or
   >= 2.5 points (total) (`scripts/nfl_simple_model.py`).
These constants were fixed before the market join and never tuned (`notes/nfl_simple_model.md`).

### Coefficient table
Each coefficient is in points per unit of the feature, with its standard error. "Per SD" is the effect, in points,
of a typical one-standard-deviation difference. Spread model B uses the legacy fit on dev 2007-19
(`notes/nfl_simple_model_tables.md`) and the live fit on 2020-25 (`notes/nfl_simple_model_refit_tables.md`).

| term | legacy 2007-19 coef (se) | per SD | live 2020-25 coef (se) | per SD | plain reading |
|---|---|---|---|---|---|
| off_pass | +3.173 (0.194) | +4.04 | +2.811 (0.280) | +3.43 | Passing offence is the biggest driver of the margin. It is a bit smaller in the modern era |
| def_pass | +1.979 (0.251) | +1.93 | +1.728 (0.369) | +1.64 | Pass defence counts about half as much as pass offence |
| off_run | +2.616 (0.381) | +1.69 | +2.356 (0.498) | +1.61 | Rushing efficiency matters, but less than passing |
| def_run | +1.889 (0.382) | +1.21 | +2.510 (0.542) | +1.57 | Run defence matters more in the modern fit |
| qb_adj_diff | +1.562 (0.241) | +1.50 | +1.807 (0.316) | +1.77 | About +1.6 points per 0.1 EPA/dropback of starter-vs-usual QB. A backup starting typically moves 3-6 points (`notes/nfl_simple_model.md`) |
| hfa_off (neutral site) | +5.080 (1.984) | +0.59 | +0.241 (2.068) | +0.03 | Removes home field at neutral sites. Not identified in 2020-25 |
| const (home field) | +2.313 | | +2.075 | | Home-field advantage in points |

Totals, model C:
| term | legacy coef (se) | per SD | live coef (se) | per SD | plain reading |
|---|---|---|---|---|---|
| off_pass_sum | +1.272 (0.227) | +1.71 | +1.690 (0.327) | +2.27 | Good passing offences mean more points |
| def_pass_sum | +0.584 (0.259) | +0.61 | +0.861 (0.383) | +0.92 | Leaky pass defences mean more points |
| qb_adj_sum | +1.057 (0.242) | +1.01 | +1.263 (0.299) | +1.33 | Better-than-usual QBs raise the total |
| lg_pts | -0.601 (0.190) | -0.87 | -0.545 (0.236) | -1.02 | Mean-reverting: a high-scoring last season pulls the total *down*. Identified from few season values, so fragile (`notes/nfl_simple_model.md`) |
| pace_sum | +0.190 (0.065) | +0.80 | +0.148 (0.092) | +0.56 | Faster teams mean more points (weak in the modern fit) |
| proe_sum | +0.339 (0.056) | +1.65 | +0.106 (0.082) | +0.54 | Pass-heavy teams play higher totals. Strong in dev (p 1.49e-9), faded in 2020-25 (p .194) |
| expl_sum | +0.492 (0.115) | +1.19 | +0.267 (0.184) | +0.56 | Explosive-play teams mean more points (weak in the modern fit) |
| I4-I6 interactions | all \|per SD\| <= 0.16 | | dropped | | Matchups add nothing (`notes/nfl_simple_model_tables.md`) |
| const | +28.026 | | +42.729 | | |

Other terms: off_run_sum and def_run_sum are -0.09 and +0.02 per SD in the legacy fit, and +0.75 and +0.41 per SD
live (same tables).

### How well it does
**Accuracy (dev 2010-19, RMSE vs the actual result)** (`notes/nfl_simple_model_tables.md`). Lower is better:

| | margin RMSE | total RMSE |
|---|---|---|
| opener | 13.39 | 13.45 |
| close | 13.27 | 13.35 |
| model B | 13.49 | 13.64 |
| model C | 13.53 | 13.55 |

**Distance from the closing line, 2020-25.** This is mean |model - close| in points, an accuracy diagnostic and not
a bet result (`notes/nfl_simple_model_refit_tables.md`):

| fit | spread B | spread C | total B | total C |
|---|---|---|---|---|
| modern, leave-one-season-out (n 1693) | 2.37 (bias +0.56) | 2.36 | 2.61 (+0.13) | 2.36 (+0.47) |
| legacy 2007-19 fit | 2.35 (+0.72) | 2.34 | 2.21 (-0.16) | 2.74 (-1.68) |
| 2026 wk1-7, modern (n 105) | 2.29 (+0.65) | 2.20 | 2.68 (+1.04) | 2.33 (+0.17) |
| 2026 wk1-7, legacy | 2.34 (+0.63) | 2.35 | 2.52 (+0.44) | 3.53 (-3.21) |

**Pre-registered trials against the opener.** These are 14 trials with a pass bar of p < .00357, on dev 2010-19
(`notes/nfl_simple_model_tables.md`). CLV (closing line value) is how far the line moved toward the model's side
between open and close:

| trial | n | mean | 95% CI | p | pass |
|---|---|---|---|---|---|
| T02 spread CLV, model B (pts) | 1243 | +0.755 | [+0.641, +0.874] | 0.0000 | YES |
| T09 total CLV, model C (pts) | 1017 | +0.812 | [+0.683, +0.941] | 0.0000 | YES |
| T05 spread ROI at the open, -110, B | 1243 | +1.1% | [-3.6%, +5.9%] | 0.3206 | no |
| T12 total ROI at the open, -110, C | 1017 | +6.6% | [+0.7%, +12.5%] | 0.0138 | no |
| T14 spread CLV, B, QB-change games | 210 | +1.155 | [+0.771, +1.550] | 0.0000 | YES |
| total C ROI with no vig (not counted) | 1017 | +11.6% | [+5.5%, +17.7%] | | |

Seven of the 14 trials pass, and all seven are CLV trials (`notes/nfl_simple_model.md`).

**Is the CLV just a stale opener?** A know-nothing control model gets CLV of +0.14 on spreads and +0.22 on totals
(`notes/nfl_simple_model.md`). Stacking the outcome on (model - close) gives total C +0.172 [-0.018, +0.362]. That
interval includes zero, so the close already holds what the model knows (`notes/nfl_simple_model_tables.md`).

**Power.** To show the totals C ROI at the bar would take about 2,513 bets, against 1,017 available
(`notes/nfl_simple_model_tables.md`).

### What this means
- The model reliably knows where the opener will move (+0.76 to +0.81 points), and QB changes are where that edge is
  largest. It knows nothing the close does not.
- At -110 none of the money trials pass. The edge would pay only at prices near no-vig.
- The live coefficients were fit on 2020-25, so the sealed test is **forfeited**. L33 is forward-paper only, with
  live picks = modern B spread and modern C total (`notes/nfl_simple_model.md`).

---

## 3. NHL simple model (lead L34)

### What it predicts
The model gives an expected regulation-goals rate (lambda) for each team. From those it gets **P(home wins)**,
including OT and shootouts, an **expected final total**, and **P(over a line)** (`scripts/nhl_simple_live.py`
docstring). These are scored against the ESPN BET opening moneyline and total (`scripts/nhl_simple_model.py`).

### Inputs (stats and variables)
The unit is one team-game, so there are two rows per game. The target is regulation goals: the final score minus
the OT/SO winner's goal (`scripts/nhl_simple_model.py` docstring).
- **r_gf: team goal rating.** A ridge regression: goals = league + attack[team] - defence[opponent] + home. Games are
  weighted with a 90-day half-life, ridge lambda is 10, and it is refit every date.
- **g_opp: opposing starting goalie.** A shrunk goals-saved-above-average (GSAA) per shot, x 30 shots:
  - rating = sum_w(saves - shots against x league save%) / (sum_w(shots against) + 1,500)
  - the decay half-life is 365 days, and all appearances count.
  - "GSAA" is save% vs the league average. It has no shot-quality adjustment (no xG data) (`notes/nhl_simple_model.md`).
- **b2b_t and b2b_o.** Whether the team, or its opponent, played the previous day.
- **S1_shots.** z(team shots for per game) + z(opponent shots against per game), with a 60-day decay.
- **S2_pen.** A power-play proxy: z(opponent PIM taken) + z(team PIM drawn), with a 60-day decay.
- **S3_sh.** z(team shooting %), with a 60-day decay. It is meant to catch shooting luck that should regress.
- **Amendment A1.** Each style feature is first expressed relative to the league's decayed mean, because league
  shots per game fell from 31.3 to 28.4 (`scripts/nhl_simple_model.py`).
- **Which goalie.** STRICT = the projected starter, meaning most starts in the team's last 10 games. ACT = the actual
  starter, which is lookahead relative to the open. The live path uses STRICT only, because no probable-starter feed
  is used (`scripts/nhl_simple_live.py`).

### How it turns inputs into a price
1. **Expected goals.** OLS: goals ~ r_gf + g_opp + b2b_t + b2b_o + S1 + S2 + S3, then lambda = max(prediction, 0.5).
   - Walk-forward: scored season s is fit on 2023..s-1. The ESPN season label 2023 means 2022-23 (`scripts/nhl_simple_model.py`).
2. **Score distribution.** Home and away goals are treated as **independent Poisson** variables with those lambdas,
   over 0-15 goals.
   - **Moneyline:** P(home) = P(H > A) + 0.5 x P(H = A). A regulation tie is a coin flip.
   - **Total:** final total T = H + A + 1 if tied, because OT or a shootout adds exactly one goal. P(over L) is read
     from that distribution, and an integer line uses P(over | no push) (`probs()`).
3. **Totals CLV in goals.** The opening and closing prices are de-vigged (proportional) and turned into an
   "implied mean total" mu by bisection. CLV = side x (mu_close - mu_open). This handles both 5.5<->6.5 line moves
   and juice-only moves (`implied_mu()`).
4. **Bet rule.** Bet when |p_model - p_open_fair| >= 0.03 (ML) or >= 0.04 (P(over)) (`scripts/nhl_simple_model.py`).
5. **Sigma.** The regression residual SD is reported (1.70 goals per team in the live fit, `results/nhl_simple_live_fit.json`)
   but is **not used for pricing**. Pricing goes through Poisson.

### Coefficient table
Goals per unit of the feature (se). The walk-forward folds are from `data/nhl_simple_model/stack_fits.json` and
`notes/nhl_simple_model.md`; the live fit, on seasons 2023-2025 (n 7736), is from `results/nhl_simple_live_fit.json`.

| term | fit 22-23 -> score 23-24 | fit 22-24 -> score 24-25 | live fit | plain reading |
|---|---|---|---|---|
| const | +1.17 (.48) | +1.41 (.34) | +1.495 (.269) | baseline goals |
| r_gf | +0.62 (.15) | +0.54 (.11) | +0.506 (.087) | The ridge rating is trusted at about half to two-thirds of face value |
| g_opp | -0.75 (.34) | -0.48 (.25) | -0.624 (.200) | Each goal per 30 shots of opposing-goalie GSAA removes ~0.5-0.75 goals |
| b2b_t | -0.25 (.09) | -0.23 (.07) | -0.237 (.056) | A tired team scores about a quarter-goal less |
| b2b_o | +0.23 (.09) | +0.20 (.07) | +0.197 (.056) | ...and its opponent scores about 0.2 more |
| S1_shots | +0.15 (.04) | +0.16 (.02) | +0.143 (.019) | Shot volume adds signal beyond the rating |
| S2_pen | -0.04 (.02) | -0.01 (.02) | -0.019 (.014) | Power-play proxy is about 0 |
| S3_sh | -0.05 (.04) | -0.01 (.03) | +0.001 (.026) | No shooting-luck regression (unlike NBA 3P%) |

The residual SD is 1.67 and 1.69 goals per team in the two folds (`data/nhl_simple_model/stack_fits.json`).

### How well it does
**Log loss on the moneyline** (2,447 games; lower is better) (`results/nhl_simple_model.json`):
- model STRICT 0.6617
- model ACT 0.6621
- ESPN BET open 0.6594
- ESPN BET close 0.6566

Stacked logit, model weight (`results/nhl_simple_model.json`):
- given the open: +0.46 [+0.04, +0.89]
- given the close: +0.14 [-0.31, +0.59]

**Trials** (8 trials, pass bar p < .00625; ESPN BET opener; seasons 2023-24 and 2024-25)
(`results/nhl_simple_model.json`):

| trial | bets | effect | 95% CI | p | pass |
|---|---|---|---|---|---|
| T1 ML CLV (probability points) | 1360 | +1.04 | [+0.90, +1.18] | 0.0 | yes |
| T2 ML ROI, no vig | 1360 | +2.9% | [-3.5%, +9.1%] | .1785 | no |
| T3 ML ROI, offered open price | 1360 | -1.0% | [-7.2%, +5.0%] | .624 | no |
| T4 totals CLV (implied goals) | 910 | +0.073 | [+0.063, +0.082] | 0.0 | yes |
| T5 totals ROI, no vig | 910 | +7.1% | [+0.8%, +14.0%] | .0125 | no |
| T6 totals ROI, offered open price | 910 | +2.6% | [-3.4%, +9.2%] | .2005 | no |
| T7 ML CLV, goalie-change games (ACT) | 938 | +1.23 | [+1.05, +1.41] | 0.0 | yes |
| T8 same, ROI no vig | 938 | +5.6% | [-2.1%, +13.3%] | .0745 | no |

Notes on these trials (`notes/nhl_simple_model.md`):
- Close-implied EV at the offered open: ML **-1.8%**, totals -1.9%.
- ESPN BET's ML opener carries about 4.0% overround.
- 76% of the totals bets are unders. Blind unders at an opening 6.5 make +5.8% with no vig.

### What this means
- As in the NFL, the model predicts where the opener will move, and the close already contains it.
- The edge (+1 probability point) is smaller than ESPN BET's ~4% vig, so it is **DEAD as a bet** at those prices.
- About half of the totals no-vig result is a blind lean toward the under (`notes/nhl_simple_model.md`).

---

## 4. CFB simple model (lead L36), brief

### What it predicts
Home margin and total points for FBS-vs-FBS games, scored against SBR's opening line for 2008-2022
(`scripts/cfb_simple_model.py`).

### Inputs (stats and variables)
- **Final scores only** (no play-by-play, no EPA), from the sportsbookreviewsonline NCAAF archive for 2007-2022.
- **Team ratings** come from those scores. All non-FBS teams are pooled into one "FCS" pseudo-team.
- **Not used:** QB availability and pace are skipped (`scripts/cfb_simple_model.py`).

### How it turns inputs into a price
- **Margin.** `HFA x (not neutral) + r_home - r_away`. The ratings are refit before each game date by weighted ridge
  regression on this season's games:
  - each game's margin is **capped at +/-28**
  - ridge lambda is 4 game-equivalents, pulling each rating toward a prior of **0.6 x last season's rating**
  - new FBS teams start at -10
  - recency weight is `0.5^(days_ago/42)`, a 6-week half-life
  - HFA is fixed per season at the mean of prior seasons' estimates.
- **Total.** `2mu + o_home + d_away + o_away + d_home`:
  - offence and defence ratings use the same ridge, shrunk toward 0.6 x last season's values (lambda 4)
  - mu is shrunk toward last season's mean (lambda 50)
  - points per team are capped at 70.
- **Bet rule.** |model - open| >= 3.0 (spread) or >= 3.5 (total) (`scripts/cfb_simple_model.py`).

### Coefficient table
There is no regression stack. The ridge ratings are the model, and fitted rating values are not documented.

| term | value | plain reading |
|---|---|---|
| margin cap | +/-28 | Blowouts count as 28 |
| ridge lambda (margin, o/d) | 4 | 4 games' worth of pull toward the prior |
| ridge lambda (mu) | 50 | League scoring level moves slowly |
| prior | 0.6 x last season | 40% regression to zero each off-season |
| recency half-life | 42 days | |

### How well it does
14 trials, pass bar p < .00357 (`data/lab/cfb_simple_model_results.json`):

| trial | n | mean | 95% CI | p | pass |
|---|---|---|---|---|---|
| T02 total CLV (points) | 3014 | +1.02 | [+0.87, +1.17] | 0.0 | yes |
| T04 total ROI at -110 | 3014 | +2.6% | [-1.3%, +6.8%] | .1006 | no |
| T06 total ROI, no vig | 3014 | +7.4% | [+3.3%, +11.8%] | .0004 | yes |
| T14 total CLV, weeks 1-4 | 814 | +1.03 | [+0.74, +1.31] | 0.0 | yes |
| T01 spread CLV | 6334 | -0.12 | [-0.19, -0.06] | 1.0 | no |
| T03 spread ROI at -110 | 6334 | -7.8% | [-10.1%, -5.6%] | 1.0 | no |

Mean absolute error against the result (`data/lab/cfb_simple_model_results.json`):
- spread: model 13.58, open 12.43, close 12.30
- total: model 13.37, open 13.25, close 13.01

Stacking the total outcome on (model - close) gives b -0.004 [-0.086, +0.079], so the close holds everything the
model knows (same file).

**Why spreads fail (the mechanism).** College spreads can reach 50 points (`notes/cfb_simple_model.md`), and two
pre-registered choices both squash the model's margins toward zero:
1. **The +/-28 cap.** A 56-0 win enters the fit as +28. The ratings never learn how big the gaps between the best and
   worst teams really are.
2. **Ridge shrinkage.** Lambda 4 plus a prior of 0.6 x last season pulls every rating further toward zero.
So the model's predicted margins are compressed (`notes/cfb_simple_model.md`):
- the model's slope vs the open is 0.58
- the outcome's slope vs the model is 1.59.
Against a big favourite, the model therefore almost always "sees value" on the underdog. The 3-point rule fired on
6,334 bets, 6,094 of them on the dog (`data/lab/cfb_simple_model_results.json`), and lost 7.8% at -110. This is a
calibration flaw in the model and says nothing about the market. A fix (walk-forward slope recalibration, or a weaker
cap) needs a fresh pre-registration and was not run (`notes/cfb_simple_model.md`).

### What this means
- On totals, CFB repeats the pattern: the model leads the opener by about 1 point, and it pays only at no-vig
  prices. Nothing is bet.
- The spread failure is a lesson in how a cap plus shrinkage can make a model systematically timid.

## 5. In-game NBA models

### A1. Big-lead tables (how often does a team up 15/20/25 win?) and the B1-B3 Kalshi test

**What it predicts.** The chance that the team leading by a lot goes on to win. It also asks whether Kalshi's price
for that leader is too low.

**Inputs.** ESPN play-by-play for 14,170 games, 2015-16 .. 2025-26 (`notes/nba_big_lead.md`). Score margin at fixed
checkpoints (end Q1, halftime, end Q3, Q4 6:00). Kalshi KXNBAGAME 1-minute candles for the 86 games of the 2025
play-in + playoffs (`notes/nba_big_lead.md`).

**How it turns inputs into a decision.** There is no fitted model. It is a counting table: one row per game per
checkpoint, win rate, Wilson 95% CI. The pre-registered bets (in the `scripts/nba_big_lead.py` docstring, per
`notes/nba_big_lead.md`): buy the leader's YES at the close ask of the first 1-min candle that starts >= 60 s after
the event. Fee = ceil_to_cent(0.07 x 100 x P x (1-P)) on 100 contracts. Hold to settlement.
- B1 = first lead >= 20 in Q2-Q3
- B2 = first lead >= 20 in Q4 with > 6:00 left
- B3 = lead >= 15 at halftime
Bar: 0.05/3 (3 trials, p < 0.0167). DEV = 2024-25 (on Kalshi, only the 86 play-in + playoff games). SEALED = 2025-26+,
not scored.

**Tables** (`notes/nba_big_lead.md`, section 1; leader win rate)
| checkpoint | lead >= 15 | lead >= 20 | lead >= 25 |
|---|---|---|---|
| end Q1 | .874 (n 1,177) | .926 (n 283) | .983 (n 57) |
| halftime | .921 (n 2,828) | .968 (n 1,211) | .985 (n 470) |
| end Q3 | .983 (n 4,161) | .997 (n 2,282) | .998 (n 1,149) |
| Q4, 6:00 left | .996 (n 4,563) | .9996 (n 2,650, 1 loss) | 1.000 (n 1,424) |

Pooled blown-lead rate (team led by >= 20 at any point and lost): **4.9% [4.4, 5.5]** (298/6,107)
(`notes/nba_big_lead.md` section 2). Pregame strength matters a lot: up 15+ at half, the leader wins .974 if it was the
pregame favourite by 5+, .894 near pick'em, .750 if it was the underdog by 5+ (`notes/nba_big_lead.md`).

**Results (DEV)** (`notes/nba_big_lead.md`, section 3)
| trial | bets | won | mean ask | break-even | units | ROI | boot p | exact p | ask $1 = no bet |
|---|---|---|---|---|---|---|---|---|---|
| B1 lead20 Q2-Q3 | 37 | .838 | .914 | .919 | -3.02 | -8.9% | .930 | .978 | 2 |
| B2 lead20 Q4 > 6 min | 15 | 1.000 | .976 | .978 | +0.34 | +2.3% | .000* | .710 | 14 |
| B3 lead15 half | 23 | .913 | .951 | .954 | -0.94 | -4.3% | .720 | .916 | 2 |
*B2's bootstrap p = 0 is an artefact of zero losses; the exact binomial p is .710. No trial passes.

Kalshi calibration of big favourites, DEV (`data/nba_big_lead_calib.json`, same numbers in the note):
| bucket | games | mean mid | mean ask | break-even | realised [95% CI] | losses |
|---|---|---|---|---|---|---|
| 90-93c | 75 | .9115 | .9229 | .928 | .8533 [.7733, .9333] | 11 |
| 93-96c | 75 | .9404 | .9499 | .9532 | .92 [.8533, .9733] | 6 |
| 96-99c | 74 | .9654 | .9751 | .9769 | .9054 [.8378, .973] | 7 |
ESPN's public WP is overconfident on big favourites: 90-93 bucket ESPN .910 vs realised .875 (n 10,959)
(`notes/nba_big_lead.md`).

**What this means.** Big leads really do hold up, but Kalshi already charges about that much. At a 95c price one loss
wipes out ~19 wins, so the leader must win more than ~95.3% just to break even (`notes/nba_big_lead.md`). Late in
games the ask is often $1.00, so there is nothing to buy.

### A2. L28 big-lead forward paper logger (`scripts/nba_bigleads_logger.py`)

**What it watches.** Live NBA games via the ESPN live scoreboard, every 120 s under launchd (deployed 10-26 as
`com.jonathanwang.quant-nba-bigleads`, `STATE.md`). Paper only, no orders.

**Rule (pre-registered 2026-10-25, script docstring).** A game qualifies when the leader is
(a) up >= 25 in Q3 (any clock), or (b) up >= 20 in Q4 with clock > 6:00 (strictly > 360 s). OT never qualifies.
Entry: buy the leader's KXNBAGAME YES at the Kalshi YES ask at that tick, only if 0 < ask <= 0.96 (`MAX_ASK = 0.96`)
and the market is active. One entry per game (first qualifying tick), 1 contract, hold to settlement.
Fee (primary): `0.07 * p * (1 - p)` unrounded; sensitivity: `ceil(100 * 0.07 p (1-p)) / 100`.
P&L/contract = (1 if leader won else 0) - ask - fee. It also logs every qualifying tick where the ask is > 0.96 or
missing, to measure how often the price is too high to buy.

**Comparison numbers.** The report compares to a historical rate `HIST = (0.993, 0.999)` (code; docstring says
"99.3-99.9%, notes/nba_big_lead.md"). **Results:** none documented yet. The docstring says no significance claim is
possible until n is in the hundreds ("a 1-pt edge at 95c needs ~3,700 bets").

**What this means.** It is a cheap check of whether buyable prices (<= 96c) even exist in these states, and of the P&L
sign. It is not a pass/fail test.

### A3. Comeback lookup and matchup study (`scripts/nba_comebacks.py`, `results/nba_comebacks.json`)

**What it predicts.** P(trailing team wins) from deficit, clock and pregame spread; plus whether team traits, matchups,
coaches, rest or refs add anything beyond the closing line.

**Inputs.** Per-minute game states (12,803 games, 2015-16 .. 2024-25), closing spread from the trailer's side
(`notes/nba_comebacks_matchups.md`). 2025-26 was not read.

**How it works.** Logistic regression on deficit/sqrt(time left), spread x sqrt(time left), deficit, spread, time,
deficit x time, plus 3 quadratic terms, fit leave-one-season-out (`notes/nba_comebacks_matchups.md`, Q1 detail).
Coefficients of this logistic model are not stored in `results/nba_comebacks.json` (keys found: calibration,
comeback_types, lookup, window_*, tests, desc_3pa) -- not documented.

**Lookup excerpt** (P(trailer wins), model; `notes/nba_comebacks_matchups.md`; full grid in
`results/nba_comebacks.json` -> q1.lookup)
| clock | down | trailer dog by 8 | dog by 4 | pick | fav by 4 | fav by 8 |
|---|---|---|---|---|---|---|
| Q2 9:00 | 15 | .082 | .129 | .197 | .291 | .406 |
| halftime | 15 | .063 | .095 | .143 | .209 | .295 |
| halftime | 20 | .031 | .049 | .076 | .118 | .177 |
| end Q3 | 15 | .029 | .042 | .059 | .084 | .117 |
| Q4 6:00 | 10 | .042 | .055 | .072 | .094 | .122 |
(Spot check: json q1.lookup Q2 9:00 / 15 / spread -8 = 0.0819, matches .082.)

**Results.** Calibration vs ESPN (`results/nba_comebacks.json` q1.window_*):
| window | n | realised | model baseline | ESPN (subset) |
|---|---|---|---|---|
| halftime | 4,805 | .1294 | .1275 | .0920 vs realised .1315 (n 3,862) |
| end Q3 | 5,745 | .0484 | .0519 | .0331 vs .0494 |
| Q4 6:00 | 5,997 | .0195 | .0172 | .0118 vs .0197 |
Trait tests: 0/69 cells survive BH. Two small possibly-unpriced effects: paint defense ATS -0.54 pts/SD (p 2.5e-5) and
3P% regression in totals. TOT_3PT walk-forward at |pred| >= 1.5: 1,417 bets, win .536 [.510, .562], ROI +2.3% at -110
(`results/nba_comebacks.json` rules.TOT_3PT).

**What this means.** Comeback odds depend on deficit, time and pregame strength, and nothing else we could measure.
ESPN's app under-rates trailers by 30-50%; our table does not. That is a warning about ESPN, not a Kalshi edge.

### A4. Late-game probit (`scripts/nba_late_wp.py`, `data/nba_late_wp_coef.json`)

**What it predicts.** P(home wins) in Q4/OT.

**Inputs** (script docstring + `notes/late_game_calibration.md`):
- m = home score - away score
- tau = (seconds left in current period + 4) / 2880
- z0 = Phi^-1(ESPN pregame home win probability) -- pregame strength comes from ESPN's pregame WP, not a spread
- poss = +1 / -1 / 0 (home / away / unknown possession), from a play-type heuristic (agreed with the next shot 98.9%
  of the time, 7,069 shots)

**How it works.** P(home wins) = Phi(b0 + b1 m/sqrt(tau) + b2 z0 sqrt(tau) + b3 poss/sqrt(tau) + b4 m). This is a
"Brownian motion" shape: the margin matters more as time runs out (divide by sqrt(tau)); pregame strength matters less
(multiply by sqrt(tau)). Fit on all Q4/OT plays of games before 2025-04-15 (2021-22 .. 2024-25, 5,191 games).

**Coefficients** (`data/nba_late_wp_coef.json`, `coef`, train_end 2025-04-15):
| term | b0 | m/sqrt(tau) | z0 sqrt(tau) | poss/sqrt(tau) | m |
|---|---|---|---|---|---|
| full fit | -0.02086 | 0.04479 | 1.06179 | 0.01556 | 0.01763 |
| validation fit (`val_coef`) | -0.01319 | 0.04513 | 0.86289 | 0.01488 | 0.01796 |

**Results.** OOS vs ESPN (fit 2021-23, scored 2023-25, `notes/late_game_calibration.md`): Q4 <= 6:00 or OT and
|margin| <= 8: model log loss 0.3939 vs ESPN 0.4006. Kalshi calibration, DEV 2025-04-15 .. 2026-01-14
(`data/nba_late_calib.json`, trailing team, first snapshot per game per bin):
| Kalshi mid bin | games | mean mid | realised [95% CI] | our model | ESPN |
|---|---|---|---|---|---|
| 0-5c | 278 | .034 | .043 [.022, .068] | .094 | .110 |
| 5-10c | 267 | .077 | .071 [.041, .101] | .132 | .147 |
| 10-20c | 292 | .153 | .178 [.137, .226] | .190 | .185 |
| 20-30c | 253 | .250 | .241 [.190, .300] | .258 | .247 |
| 30-40c | 224 | .348 | .335 [.277, .397] | .315 | .313 |
| 40-50c | 174 | .446 | .460 [.385, .534] | .370 | .360 |
| 50-60c | 122 | .543 | .557 [.475, .648] | .394 | .380 |
Information test: Kalshi coefficient 0.99 [0.75, 1.22], model -0.06 [-0.25, +0.17]. Trials: T1 trail_buy 311 bets,
ROI -1.47%, p .552; T2 any_side 331 bets, ROI -5.43%, p .852 (`notes/late_game_calibration.md`).

**What this means.** The model beats ESPN, but Kalshi beats the model. Bets chosen where model and market disagree were
the bets where the model was wrong. With zero latency the model only ties Kalshi.

### A5. NEW: mid-game comeback model, trial L37 (`scripts/nba_midgame_scan.py`, `results/nba_midgame_fit.json`)

**Status.** New: committed in 2c29645 ("L37 NBA mid-game comeback scanner + forward paper logger (frozen fit
2015-24, launchd 90 s)") while this guide was being written. The docstring pre-registers L37 as a forward paper trial
(frozen 2026-10-26 ~01:00 PT), paper only. There are no results yet beyond the history fit.

**What it predicts.** P(trailing team wins) in the gap the earlier work left open: Q2 .. Q4 6:00 with deficits 6-25.

**Inputs.** d = deficit; s = trailer's pregame expected margin (+ = trailer favoured; from the ESPN pickcenter spread, or
backed out of the last pregame Kalshi mid via pregame_sigma); tau = regulation time left / 2880. History =
`data/nba_comebacks/states.parquet`, seasons 2015-16 .. 2024-25.

**How it works.** Probit: P(trailer wins) = Phi(b . [1, d/sqrt(tau), s sqrt(tau), d, s, tau, (d/sqrt(tau))^2]), fit
on elapsed 600..2640 s, d 1..35 (`results/nba_midgame_fit.json` "features"). Leader probability = 1 - trailer.
Trial rule (docstring): window Q2 (incl. halftime), Q3, Q4 with clock > 360 s; deficit 6..25; for each side
edge = p_model - ask - 0.07 ask (1-ask); enter the first tick where edge >= 0.05 AND the empirical neighbour estimate
(games within ±120 s, deficit ±2, spread ±3, n >= 30) also beats ask + fee. One contract, hold. Descriptive until
n >= 100 entries.

**Coefficients** (`results/nba_midgame_fit.json`; n_states 426,961, n_games 12,737; pregame_sigma 12.187)
| term | 1 | d/sqrt(tau) | s sqrt(tau) | d | s | tau | (d/sqrt(tau))^2 |
|---|---|---|---|---|---|---|---|
| coef (all dev) | -0.04234 | -0.05150 | 0.06926 | 0.006927 | 0.01334 | 0.06377 | -6.27e-05 |
| coef_le2023 | -0.03849 | -0.05249 | 0.06943 | 0.008033 | 0.01324 | 0.05558 | -6.25e-05 |

**Results (out-of-sample: fit <= 2023-24, test 2024-25, 24,539 states / 1,303 games)** (`results/nba_midgame_fit.json` oos)
| metric | model | no-spread model |
|---|---|---|
| log loss | 0.3753 | 0.4164 |
| Brier | 0.1181 | 0.1316 |
Mean model .1634 vs mean realised .1723. Calibration bins (state-level): (0.1, 0.15] model .1239 vs realised .1329;
(0.3, 0.4] .3486 vs .3814; (0.6, 1.0] .6658 vs .7664 (n 304 states / 92 games). Empirical-neighbour estimate vs model
(n 3,530): log loss .43885 vs .43942 -- a tie.

**What this means.** Pregame strength clearly matters out of sample (log loss .375 vs .416 without it). The model runs
slightly low (about 1 point on average). Whether Kalshi misprices this region is exactly what the forward log will
test. Nothing has been measured against Kalshi yet.

## 6. Kalshi weather consensus-reject (the only live-money edge)

**What it predicts.** Which Kalshi daily-high temperature brackets for tomorrow are almost certainly NOT going to hit.

**Inputs: 5 forecast models** (`scripts/kalshi_paper.py`, `MODELS`), from Open-Meteo, hourly temperature_2m in F:
GFS (`gfs_seamless`), ECMWF (`ecmwf_ifs025`), ICON (`icon_seamless`), GEM (`gem_seamless`), NBM (`ncep_nbm_conus`).
Each model's forecast = max over hours <= 18:00 local tomorrow (needs >= 15 hourly values; >= 3 models must be
available). Cities: 7 original KXHIGH series (NY/KNYC, CHI/KMDW, MIA, AUS, DEN, LAX, PHIL; `sports/wx.py` CITIES) +
17 X1 KXHIGHT* series (`X1_CITIES`), 24 in total.

**How it decides** (`scripts/kalshi_paper.py`, `rejected()`, `MARGIN = 3.0`, `DECISION_HOUR = 22`):
- Runs 22:00-23:59 city-local, the night before.
- A bracket is "rejected" when every model misses it by more than 3F:
  - "greater" bracket: `max(forecasts) + 3 < floor + 1`
  - "less" bracket: `min(forecasts) - 3 > cap - 1`
  - between bracket: `max(forecasts) + 3 < floor` or `min(forecasts) - 3 > cap`
- Quote filter: YES bid > 0, YES ask < 1, ask - bid <= 10c, ask > 1c.
- Bet: **buy NO** at 1 - best YES bid (= the NO ask). It never buys YES.

**Execution, taker (default, live now)** (`scripts/kalshi_exec.py`, `notes/go_live.md`): re-check the rule; pick age
<= 120 min; market active and >= 60 min to close; NO ask within +1c / -3c of the pick price; size = min(caps, 50% of
depth within 1c, KALSHI_MAX_CONTRACTS); one immediate-or-cancel limit order at the ask. Sizing tiers: <= 1% of
bankroll per orig7 pick, <= 0.5% per X1 pick (`TIER_FRAC`).

**Execution, maker (E4; deployed but OFF until `KALSHI_EXEC_STYLE=maker`)** (`scripts/kalshi_exec.py`
`_decide_maker`): resting post-only NO bid at
`q = round(min(cent_floor((ask + no_bid) / 2), ask - 0.01), 4)`, i.e. the NO mid floored to the cent, at most ask - 1c;
skip if ask - q > 5c (`MAKER_MAX_IMPROVE = 0.05`); good-till-cancelled with expiration = market close; no taker
fallback. Fee assumed 0 when the series `fee_type == "quadratic"`, else the taker formula.

**Results** (dev from `lab/ledger.jsonl`; sealed from `lab/lockbox.jsonl` / `notes/EDGES_AND_AGENTS.md`)
| edge | period | bets | ROI | units [95% CI] | p |
|---|---|---|---|---|---|
| E1 dev (kalshi_reject_safe_check, orig7) | 2024Q1-2025Q2 | 1,269 | +1.97% | +25.05 [+7.90, +42.39] | .002 |
| E1 sealed | 2025Q3-2026Q4 | 1,844 | +1.45% | +26.82 [+10.86, +42.61] | <.0001 (logged 0.0) |
| E2 = X1 dev (17 new cities) | 2026Q1 | 444 | +1.22% | +5.41 [-1.35, +11.80] | .057 |
| E2 = X1 sealed | 2026-03..10 | 2,464 | +0.78% | +19.22 [+4.74, +33.03] | .0047 |
| E4 maker dev, M3_N10 (`notes/kalshi_temp_maker.md`) | E1+X1 dev picks | 1,239 of 1,713 picks | +3.22% | delta vs taker +9.5 [+4.2, +14.4] | .0003 |
| E4 maker sealed, M3_N10 | 2025-07..2026-10 | 3,456 fills of 4,308 picks (fill 80.2%) | +2.22% (taker +1.07%) | delta +30.61 [+26.21, +34.85] | <.001 |
E4 sealed loss rate: filled 3.07% vs all-picks 2.46%; unfilled 0.00% (every loser filled). Delta with a 0.0175 maker
fee: +27.42 units (`lab/lockbox.jsonl`). E3 (soccer soft books vs Pinnacle, not a Kalshi edge): sealed CLV +3.63%,
ROI +8.71%, 2,004 bets, p .010, not tradeable from CA (`notes/EDGES_AND_AGENTS.md`).

**Live and paper to date** (`notes/HANDOFF_2026-10-25.md`, as of 2026-10-25 ~09:20Z; nothing newer is documented)
| | fills / picks | settled | wins | P&L |
|---|---|---|---|---|
| LIVE real money ($50 bankroll, $1 units, since 10-22) | 29 fills | 14 | 14 | +$0.23 |
| Paper (24 cities) | 90 picks | 69 | 68 | +0.53u |
Most live picks are NO @ $0.99 (win 1c); caps (daily $15 / open $20) caused 6 skip_cap (`notes/HANDOFF_2026-10-25.md`).

**What this means.** When five independent weather models all say "not even close", the bracket rarely hits, and the
market still charges a little too much for those YES longshots. The edge is about 1-1.5% of money risked, so at $1
units it earns pennies. Resting a bid (maker) instead of crossing the spread roughly doubles the ROI despite adverse
selection (losers always fill).

## 7. How our models compete

### 7.1 Two different opponents: the opener and the close
- **The close is the market after everyone has spoken**: sharp bettors, injury reports, lineup news, limits raised.
  On dev NBA the closing spread has MAE 10.56 vs our best no-lookahead model's 10.96, and log loss .579 vs .599
  (`results/models_guide_diag.json`). Given the close, every model we built gets a stacking weight of ≈ 0 (NBA:
  −0.11 to +0.08, `results/nba_simple_model.json`; NFL total C +0.172 [−0.018, +0.362]; NHL +0.14 [−0.31, +0.59];
  CFB −0.004). **We do not beat the close anywhere.**
- **The opener is a first guess made with less information, at low limits.** It is softer, and softest on totals:
  NBA opening total MAE 14.73 vs close 14.19; the opening total moves ~3 points before tip, the opening spread ~1.75
  (`notes/nba_simple_model.md`). Our models carry information the opener lacks, so the opener drifts toward us.
- What this means: we are not out-predicting the market; **we are front-running its own later self.** That is a real
  but fragile edge: it exists only at the first-posted price, which is exactly where limits are lowest and where our
  data's "open" timestamp is least certain.

### 7.2 Why CLV is the leading indicator
- A bet's result is ±1 unit of noise; its CLV is a much quieter number. NBA T1: CLV +0.79 pts was significant at
  p < 1/4000 on 1,262 bets, while the same bets' ATS profit (+2.3% no-vig) had p .21 (`results/nba_simple_model.json`).
- Because the close is the best estimate we have of the truth, consistently getting a better number than the close
  should pay *on average*, and CLV shows up in far fewer bets than profit does. NBA totals show both together:
  T4 CLV +2.0 pts and T5 +8.7% no-vig, positive in both seasons.
- The trap: CLV can be "real" yet unbankable. O1 had sealed CLV +0.68 pts (p .005) and still lost −2.55% EV
  (`notes/HANDOFF_2026-10-25.md`), because the vig at the opener was bigger than the CLV was worth.
  **CLV tells you that you are early; vig decides whether being early pays.**

### 7.3 Vig and fee arithmetic (what we must clear)
| Venue / price | Cost | Break-even win rate |
|---|---|---|
| Sportsbook spread/total at −110/−110 | overround 4.76% (arithmetic, `devig_two_way`) | **52.4%** (110/210) |
| ESPN BET NHL moneyline opener | ~4.0% overround (`notes/nhl_simple_model.md`) | |
| Kalshi taker | `fee = ceil(0.07 × n × p × (1−p))` cents (`scripts/kalshi_exec.py`) | at 50c: 50% + 1.75c = **51.75%** (arithmetic) |
| Kalshi taker, 95c favourite | 0.07 × .95 × .05 = 0.33c | ~95.3% (`notes/nba_big_lead.md`) |
| Kalshi maker on `fee_type: quadratic` series | 0 (`scripts/kalshi_exec.py maker_fee_unit`; KXHIGH checked 10-22) | = price; whether NBA/NFL series are also quadratic is **UNVERIFIED** |
| No-vig (our backtest proxy) | 0 | 50% at even odds |

What this means: a 55% totals picker (NBA T5's 55.5%) makes ~+5.9% at −110 and ~+8.7% at no-vig; a 52% picker
(NBA spreads) loses at −110, breaks even on a Kalshi taker, and only pays as a maker. **Execution venue is worth as
much as the model.**

### 7.4 Where the information edge actually lives
1. **Stale openers**, mostly on totals (NBA, NFL, NHL, CFB all show it). The opening total moves toward the model by
   +2.0 (NBA), +0.81 (NFL), +0.073 goals (NHL), +1.02 (CFB).
2. **Player absences.** In NBA games where a top-5-minutes player who played last game sits tonight, a model that
   knows who sat leads the opener by +1.75 pts vs +1.03 elsewhere (T8 vs no-news; lookahead upper bound). NFL QB-change
   games: +1.16 pts (T14). NHL goalie-change games: +1.23 prob-pts (T7). The news is the edge, and only if we see it
   before the book reprices.
3. **Two style facts the opener under-weights**: NBA paint defence (S1) on sides and 3-point-percentage regression
   (U2/U3) on totals. These are where T2's +8.7% comes from (T1 without style: +2.6%).
4. **Weather consensus** (not sports): five independent forecasts agreeing a bracket is out of reach.

### 7.5 What the models ignore (on purpose or not)
- **NBA**: lineup combinations, minutes restrictions, motivation/tanking/seeding (except via who sits), coaching,
  referees (tested dead, `notes/nba_referees.md`), altitude, in-season trades beyond "latest team" bookkeeping, the
  market itself (no line is an input).
- **NFL**: non-QB personnel (WR1, LT, edge rusher not built), weather (separate rain study), coaching.
- **NHL**: shot quality (no xG; GSAA is plain save%), confirmed starting goalies (live uses projected starter only).
- **CFB**: everything except final scores (no QB, no pace, no play-by-play).

### 7.6 Known weaknesses
| Weakness | Evidence | Effect |
|---|---|---|
| STRICT availability rule ("played last game = plays tonight") | 9 of the 10 biggest NBA dev disagreements with the close are rested stars; ACTUAL shrinks the top-50 gap from 10.4 to 6.6 pts (`models_guide_diag.json`) | spread bets in April / rest games are mostly the model not knowing |
| Whole-lineup rests are under-sized | OKC 2025-03-07 (4 starters out): model still 11.2 pts off even with ACTUAL availability; single absences are sized right (slope 1.01 over 134 games) | linear absence term breaks at the extremes |
| Totals level bias | T2 averages +2.3 pts above the close; +4.8 in March, +4.7 in April; 78% of T5 bets are overs (`models_guide_diag.json`) | slow 60-day decay lags league-wide scoring shifts (2023-24 March drop ~5 pts) |
| Early-season roster churn (live) | in October STRICT's "previous game" is last April; new signings start at replacement (`notes/nba_simple_model.md` §4) | October predictions are the least trustworthy; dev never scored them (5-game minimum) |
| Opener timestamp unknown | ESPN's "open" time is undocumented (`notes/nba_simple_model.md` §4) | the bettable opener may be sharper than the backtest's |
| Under-power | sealed L32 had ~25% power at a true +4% (`notes/nba_simple_model.md` §7) | a sealed miss is inconclusive, not a refutation |
| CFB spread compression | ±28 cap + shrinkage: 6,094 of 6,334 bets on the dog, −7.8% at −110 | model is systematically timid on big spreads |
| Normal-error win prob | P(home) = Φ(margin/13.6) with one σ for every game | calibration is fine on dev (slope 1.11), but σ should differ for rest/blowout games |

### 7.7 Improvement ideas, ranked by expected value
1. **Switch the weather edge to MAKER** (one line: `KALSHI_EXEC_STYLE=maker` in `~/.claude/quant-kalshi/.env`).
   Already passed sealed: ROI +1.07% → +2.22% on 4,308 picks (`notes/EDGES_AND_AGENTS.md` E4). Zero research cost,
   the only idea here with a sealed pass behind it.
2. **Execute the NBA (and CFB/NHL) totals signal at the first posted number, at low vig.** The forward logger
   `scripts/nba_open_totals.py` already records first-seen Pinnacle totals and Kalshi KXNBATOTAL ladders. If Kalshi
   totals ladders are maker-fee-free and post early, the dev +8.7% no-vig / sealed +6.4% becomes bankable; at −110 it
   shrinks to +1.6% sealed. First step: confirm NBA series `fee_type` and how early the ladder posts (NFL Kalshi totals
   were *not* early, `notes/kalshi_early_nfl_totals.md`).
3. **De-bias T2's level** (anchor to a short-half-life league scoring mean, or subtract its trailing gap to the
   opener). Post-hoc dev: 988 bets, 50% overs, +7.2% no-vig, CLV +2.21 (`models_guide_diag.json`), i.e. the same edge
   without the over-lean. Needs a new pre-registration; would also have avoided the March 2024 misses.
4. **Use "fair = blend of opener and model", not the raw model.** Given the opener, the OLS weights are ~0.70 opener
   + ~0.41 model (NBA M2 STRICT, `results/nba_simple_model.json`). Quoting a blended fair price would cut false
   positives where the model alone is far off (e.g. rest games) and give honest bet sizes.
5. **Log the injury-report ("avail") variant forward next to STRICT.** The T8 upper bound (+1.75 vs +0.73 CLV) says
   that knowing absences early is worth more than any style term. The live script already computes
   `model_*_avail`; record it with timestamps vs the line so we learn whether we see news before the book.
6. **Rest-game guard for spreads**: skip (or flag) games in the last ~10 days of the regular season and second nights
   of back-to-backs unless the injury report is posted. Dev April |model − close| is 3.94 vs ~2.75 other months.
7. **Nonlinear absences**: replace the linear dAbs with a term that grows faster when 3+ rotation players sit
   (whole-lineup rests are where the model is furthest off). Medium effort, modest EV (these games are rare and the
   market prices them anyway).
8. **CFB spread recalibration** (walk-forward slope fix or a looser cap) before any CFB spread work; totals are the
   only CFB lead now.

### 7.8 Things I found that disagree between files (flag, not fixed)
- `notes/nba_simple_model.md` §1 prints M2/T2 coefficient tables (e.g. T2 tot0 +0.93, U1 +0.77) "fit on 2023-24 +
  2024-25"; the frozen file used live and sealed (`results/nba_simple_model_frozen.json`, fit 2022-23..2024-25,
  n 3,445) has tot0 +0.77, U1 +2.00. This guide uses the frozen file.
- `notes/EDGES_AND_AGENTS.md` L33 still reads as if a 2020-25 NFL sealed test were pending; the live refit on 2020-25
  forfeited it (`notes/nfl_simple_model.md`). `nfl_simple_model.md` caveats still contain pre-refit text.
- The NBA mid-game model (L37: `scripts/nba_midgame_scan.py`, `results/nba_midgame_fit.json`) was not yet in
  STATE.md or the registry when this guide was drafted (since committed in 2c29645; check the registry).
- `notes/nba_big_lead.md` tables pool play-by-play through 2025-26 (the sealed NBA season) as descriptive counts.
- L28 logger's `HIST = (0.993, 0.999)` lower bound is not a number that appears in `notes/nba_big_lead.md`.
- The comeback logistic model's coefficients are not saved anywhere (`results/nba_comebacks.json` has the lookup grid
  only).
- E2 (X1 cities) did not pass its dev test (p .057) before going to sealed, where it passed (p .0047).

## Appendix. Fee and vig arithmetic as implemented in code

**Break-even at -110.** A -110 bet wins 100/110 per unit risked (`scripts/angles.py`: `SPREAD_PAYOUT = 100 / 110`).
Break-even p solves p x 100/110 = (1 - p), so p = 110/210 = 0.5238 (arithmetic); the notes use 52.4%
(`notes/nba_comebacks_matchups.md`).

**American odds -> probability** (`sports/market.py`, `american_to_prob`): negative ml: -ml/(-ml + 100); positive:
100/(ml + 100). Docstring examples: -150 -> 0.600, +130 -> 0.435.

**Proportional de-vig** (`sports/market.py`):
```python
def devig_two_way(ml_a, ml_b):
    qa, qb = american_to_prob(ml_a), american_to_prob(ml_b)
    s = qa + qb
    return qa / s, qb / s, s - 1.0
```
Each side's implied probability is divided by their sum; the overround is sum - 1. Example (arithmetic): -110/-110 gives
0.5238 + 0.5238 = 1.0476, so each side de-vigs to 0.500 and the overround is 4.76%. The docstring warns proportional
de-vig "slightly overstates longshots"; the registry calls this the "proportional de-vig trap" (MEMORY index) and E3
uses power de-vig (`notes/EDGES_AND_AGENTS.md`).

**Kalshi taker fee as implemented.**
- Per order, rounded up to the cent (sizing, `scripts/kalshi_exec.py`):
  ```python
  FEE = 0.07
  def fee_total(n, p):
      return math.ceil(round(FEE * n * p * (1 - p) * 100, 6)) / 100 if n > 0 else 0.0
  ```
- Per contract, unrounded (paper trader, backtests, NBA loggers): `0.07 * p * (1 - p)` (`sports/wx.py FEE_RATE = 0.07`,
  `scripts/nba_bigleads_logger.py fee()`). Prod fills showed fees ~exact, not cent-rounded (`notes/HANDOFF_2026-10-25.md`).
- Example (arithmetic): NO at 0.99 -> 0.07 x 0.99 x 0.01 = $0.00069 per contract; at 0.95 -> $0.0033.

**Kalshi maker fee as implemented** (`scripts/kalshi_exec.py`):
```python
def maker_fee_unit(q, fee_type):
    return 0.0 if fee_type in ("quadratic",) else FEE * q * (1 - q)
```
KXHIGH series report `fee_type: quadratic` (checked 10-22) = no maker fee (`notes/kalshi_temp_maker.md`). Sensitivity
in the backtest: maker fee 0.0175 x q x (1-q) (`scripts/kalshi_temp_maker.py MAKER_FEE_SENS`).

**What this means.** Fees and vig, not prediction, decide most of these questions: at -110 you must win 52.4% to break
even; at a 95c Kalshi price you must be right more than ~95.3% of the time (`notes/nba_big_lead.md`).
