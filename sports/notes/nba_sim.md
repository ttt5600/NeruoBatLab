# NBA player-level Monte Carlo simulator, v1 (`scripts/nba_sim.py`), built 2026-10-29

**What it's for:** a fast pricing aid. It gives a fair win probability, spread and total for a game, the change from
removing a player, and player points/rebounds/assists distributions. It is NOT a betting signal. In the 2024-25
backtest it does not beat the close: it adds no information over the close, and its Brier score is worse.

```
../.venv_quant/bin/python scripts/nba_sim.py game  CLE ATL [--date 2026-10-29] [--out "A,B"] [--in "C"] [--minutes "P:30"]
                                                   [--n 20000] [--seed 7] [--spread -1.5] [--total 237.5] [--props f.csv] [--no-refresh]
../.venv_quant/bin/python scripts/nba_sim.py delta CLE ATL --out "Jalen Johnson"        # change in cents / points
../.venv_quant/bin/python scripts/nba_sim.py validate --season 2024-25
../.venv_quant/bin/python scripts/nba_sim.py fit                                         # refit live params (dev 2022-23..2024-25)
```
Arguments are AWAY then HOME. The props CSV has the columns `player,stat,line`, where stat is points/rebounds/assists.
The game and delta commands refresh ESPN games and box scores first, as team_rating.py does, unless you pass
`--no-refresh`. They read ESPN's roster and injury feeds as of now.

## v2.1 PRE-REGISTRATION (written 2026-10-31 16:15 PT, BEFORE any v2.1 tuning run)

Question: do (A) shrunk BPM-lite betas, (B) a fast-recency component for per-minute rates, (C) a player x arena
points effect make the sim a better estimate of who wins / of the close?
- Tuning: every knob (ridge lambda, TOV/FGA usage factor, impact A/K/G, KSHRINK, KROLE, HP, recency weight/half-life,
  arena K) is picked on dev **2023-24 ONLY** (team stack fit on 2022-23). 2025-26 is never scored (state only for 26-27).
- Confirm out of sample on **2024-25** (stack fit 2022-23..2023-24) and **2026-27 to date** (frozen dev stack,
  walk-forward, DK close). ACT set (as in the v2 table). Baseline = v2 (A .5, K 1500, G .2, OLS betas).
- **Keep rule:** a component is ON in the default only if, on BOTH 24-25 and 26-27, (spread MAE vs close: paired
  game-bootstrap 95% CI of (new - v2) excludes 0 below zero in at least one season and its point estimate is not
  worse (> 0) in the other) OR (the same for Brier vs outcomes). Otherwise it stays implemented behind a flag, OFF.
- Components that cannot move the margin (C arena; the pts-only part of B) are additionally reported on player
  points MAE vs actual and vs prop open (graded_2024/2025, ESPN-listed books); the SAME paired-CI rule applied to
  player points MAE vs actual is the keep criterion for a player-projection-only component (declared here, before
  any run, because the game-level rule cannot be met by construction for such a component).
- Kill / too-good guard: if any variant beats the CLOSE on MAE-vs-actual or Brier, check lookahead before reporting.
- Expected: A moves usage creators up (Brown from -1.42 toward ~0+); B small; C ~0 (venue reliability r .024).

## v2 (2026-10-31): impact rating fix -- now the default; `--legacy` reproduces v1 exactly

**Why.** Two live complaints: (1) favourites priced 6-17c below Kalshi on 10-30/10-31; (2) `--in "Jaylen Brown"` made
PHI weaker. Diagnostics (tmp scripts, numbers below are from their printed output):
- **Bug 2 root cause = the BPM-lite rating, not the trade.** Brown's rating was -3.18/48. The live box fit
  (seasons 2023-24..2025-26) has TOV -3.04/48 and FGA -1.24/48; his 4.0 TOV/48 alone costs -12.3. Adding any player
  rated below his team's minutes-weighted average lowers the team, and removing him raises it. Legacy
  `delta NY PHI --out "Jaylen Brown" --date 2026-10-30`: P(PHI) **+3.5c** when Brown sits. Other usage creators were
  also rated below average: Cunningham -1.5, Banchero -2.4, Morant -2.0, Sengun -1.4, and Edwards only +0.8. The TOV
  beta is unstable by season (-2.1 for 2021-24, -2.8/-3.3/-3.0 for single seasons 2024/2025/2026).
  The trade itself adds one thing: an acquired player kept his full old-team minutes (Rupert, 36.4 at MEM, came
  out at 39 min for PHI). On dev 2022-25, in his first game for a new team a 15+ mpg player plays about 0.77 of his
  old EWMA (n=460).
- **Bug 1: the mean margin is NOT systematically compressed.** Slope of close on sim (ACT): 1.09 / 1.01 on dev
  2023-24 / 2024-25. It is 0.95-1.11 in the first 5 games of a season, so early-season prior shrinkage is not the
  cause. On 2026-27 to date (81 games, DK close) it is 1.03. The "favourite gap" comes from selection: if you
  condition on the MARKET being a big favourite (|close| >= 6), a noisier but calibrated sim must on average sit
  closer to 0. That gap is -1.5 pts on dev 2024-25 and -2.6 on 2026-27, while the slope stays near 1. The fix is
  a less noisy sim, not a stretch factor. The live 10-31 misses were concentrated in games with star absences:
  MIN without Edwards (+0.8 rated), OKC without Holmgren/J. Williams/Caruso/Wallace, POR with Morant out rated as
  a *gain*. These were absences the BPM-lite under-valued.
- **Probability flattening is small.** sigma_m 13.55 (v1) / 13.47 (v2), against a close-implied SD of 12.9-13.6 on dev.
  Logit slope of p_close on p_sim is 1.03 (v1) / 1.09 (v2) on 2026-27. At a 12.5-pt spread this is ~1c. Not changed.

**What changed (v2 = `IMPACT` in nba_sim.py):** r = BPM-lite + 0.5 x on/off48 x n_eff/(n_eff+1500) + 0.2 x (mpg - 24).
- on/off48 = 48 x (his on-court +/- per minute - the team's margin per minute while he sat), from decayed sums
  kept in State.
- mpg is shrunk with 5 pseudo-games at 12.
- A player who has not yet played for his team gets 0.77 x his EWMA minutes (15+ mpg) and participation 1.
- A=0.5, K=1500, G=0.2 were picked on dev 2023-24 alone (grid; G=0.2 1.955 vs G=0 2.133 MAE vs close). Then
  they were confirmed on 2024-25 and on 2026-27.
- New stack (`fit`, dev 2022-23..2024-25): margin = 0.48 + 0.694 mu_ridge + 1.40 dP_full + 2.39 dP_absence,
  sigma 13.47. The absence coefficient fell from 3.96 to 2.39: the ratings need less stretching.
- Files: results/nba_sim_params_v2.json; results/nba_sim/validate_{2024,2025}_v2*. Legacy files are untouched.
  nba_injury_teammate_props.py is pinned to legacy.

| (ACT set unless noted; DK/ESPN BET close) | 2023-24 v1 | 2023-24 v2 | 2024-25 v1 | 2024-25 v2 | 2026-27 v1 | 2026-27 v2 |
|---|---|---|---|---|---|---|
| spread MAE vs close | 2.23 [2.13, 2.33] | **1.94** [1.85, 2.03] | 2.30 [2.19, 2.41] | **1.97** [1.88, 2.06] | 2.74 | **2.39** |
| STRICT MAE vs close | 2.57 | 2.38 | 2.73 | 2.55 | 2.88 | 2.65 |
| slope close ~ sim | 1.09 | 1.12 | 1.01 | 1.03 | 1.03 | 1.10 |
| Brier sim (close) | .2048 (.1975) | .2029 | .2026 (.1971) | .2010 | .2208 (.1945) | .2208 |
| Brier sim - close | +.0074 [+.0035, +.0110] | +.0054 [+.0022, +.0086] | +.0054 [+.0014, +.0096] | +.0038 [+.0003, +.0074] | | |
| within 1.5 pts of close | 53.5% | 60.0% | 51.9% | 60.8% | | |
| injury news: corr / sign agree | .600 / 75.5% | .642 / 78.6% | .459 / 69.4% | .499 / 74.0% | | |

- 2026-27 (scripts/nba_sim_check27.py) is 81 home games through 10-30. The player state is point-in-time; the stack
  is frozen dev params; the full rotation is last-10-games (there is no roster-feed history). The paired MAE
  difference v2 - v1 is **-0.35 [-0.59, -0.12]** (ACT) and -0.23 [-0.43, -0.04] (STRICT). The 2025-26 season
  enters only as state, never as a fit.
- The sim still does NOT beat the close on any season, so there is no lookahead flag. The improvement does not
  change the verdict: the sim is a repricer, not a signal.
- Totals and player distributions are unchanged.
- The 1.5-pt bar is still not met (1.94-1.97).
- **Brown after the fix:** he is rated -1.42 (from -3.18). `--out "Jaylen Brown"` is now -0.4c for PHI (it was +3.5c).
  The sign is right but the size is still far below a market star value. The BPM TOV penalty remains the
  limitation (next step: shrink the box betas toward a stable pooled fit, or add a public impact metric).

**10-31 slate, 15:20 PT, descriptive only (fav = Kalshi favourite; Kalshi mid of yes bid/ask):**

| game | fav | Kalshi | v1 | v2 | v2 - Kalshi |
|---|---|---|---|---|---|
| POR@UTA | POR | no 2-sided quote | 57.8 | 57.9 | - |
| PHI@CHA | PHI | 60.5 | 61.8 | 62.5 | +2.0c |
| ATL@NOP | ATL | 65.0 | 58.8 | 60.8 | -4.2c |
| SAC@MEM | MEM | 75.5 | 75.4 | 72.8 | -2.7c |
| CLE@MIL | CLE | 76.5 | 63.9 | 66.3 | -10.2c |
| MIN@SAS | SAS | 82.5 | 66.4 | 72.4 | -10.1c |
| OKC@HOU | HOU | 56.5 | 44.2 | 52.7 | -3.8c |

The mean favourite gap went from -7.7c to -5.3c. CLE@MIL and MIN@SA remain about 10c short. CLE@MIL is not explained
(possibly MIL roster turnover that the team ridge still carries -- unverified); in MIN@SA the sim still under-values
the absent Edwards and Gobert.

## Design (and what was used for each piece)

| piece | v1 choice |
|---|---|
| Data | ESPN `data/espn/nba/{games,box}_{S}`. A live game loads S-3..S, so it reads 2025-26 for the player state, for prediction only. Validate never loads 2025-26 (sealed). |
| Player impact | **BPM-lite**, the same as nba_simple_model: on-court plus-minus/48 regressed (WLS by minutes) on per-48 box rates, over player-seasons of seasons < S with >= 300 min. Player rates are decayed (half-life 180 d) and shrunk to replacement with 500 pseudo-minutes. A ridge on-court (RAPM) model was **not feasible**: ESPN box has per-game player +/- only, with no stints or lineups. |
| Shrink to role prior | pts/reb/ast per minute shrunk to the player's ROLE prior with 150 pseudo-minutes. Role (G/W/B) comes from his assist:rebound mix, because the box data has no positions. |
| Minutes | Full rotation = players on the team in its last 10 games, plus ESPN roster adds/drops when live. Expected minutes = EWMA(0.3) of minutes when he played; for non-regulars (EWMA < 24) this is multiplied by his participation share of those games. OUT players' minutes go to teammates in proportion to m(48-m), x2 for the same role. Then an additive rescale (weights m(48-m)) to exactly 240, with a 42 cap. `--minutes` pins a player. |
| Team mean | margin = c0 + c1*mu_ridge + c2*dP_full + c3*dP_absence; total = k0 + k1*tot_ridge + k2*O_full + k3*O_absence. mu_ridge = the frozen nba_player_sim ridge (H=60, lam=5), which carries home court, b2b, rest diff and travel. tot_ridge = the decayed off/def points ridge, i.e. recency-weighted pace x efficiency. P = minutes-weighted impact /240; O = expected points from those minutes. The coefficients are OLS on scored dev games. |
| Game noise | Bivariate normal (margin, total) using the training residual SDs and correlation. Scores are integers; ties go to a 5-minute OT. |
| Player stats | Per sim: minutes jittered lognormally (SD fitted on the actual vs projected log ratio) and renormalised to 240. Team points come from the game sim. Team reb/ast = expectation x lognormal (CV fitted); assists also scale with team points. Points are split by Gamma/Dirichlet shares, with the concentration kappa chosen on the training season so the 10-90 interval covers about 80%. |
| **Deliberately excluded** | Arena / opponent / revenge / body-clock effects. All were tested dead: notes/nba_player_opp_venue.md (0/5) and notes/nba_clock_calendar_arena.md (0/6). |

Live params (`results/nba_sim_params.json`, fit on 2022-23..2024-25, n=3445 games):
margin = 0.30 + 0.777 mu_ridge + 1.42 dP_full + 3.96 dP_absence, sigma 13.55;
total = -12.55 + 0.895 tot_ridge + 0.177 O_full + 0.255 O_absence, sigma 18.43, rho -0.004.
The absence coefficient is large even in the STRICT (no-lookahead) fit: 3.36 in the 2024-25 validation.
So the minutes-weighted BPM-lite differences are compressed, and the regression stretches them back out.

## Validation, 2024-25 (`validate --season 2024-25`, results/nba_sim/validate_2025.json)

Point-in-time: the state for every game uses only games dated before it. The team stack is fit on 2022-23 and
2023-24 (n=2297). Player noise is calibrated on 2023-24. There are 1147 scored games with ESPN BET open and close.
The figures in brackets are 95% game-bootstrap intervals for the mean.

| | ACT (who played = injury-report proxy) | STRICT (previous game's players) | ESPN BET close |
|---|---|---|---|
| spread MAE vs close | **2.30** [2.20, 2.41] | 2.74 [2.60, 2.87] | 0 |
| share of games within 1.5 pts of the close | 51.4% | 47.8% | |
| spread MAE vs actual margin | 10.82 [10.31, 11.33] | 10.93 | 10.50 [10.02, 10.98] |
| total MAE vs close | 3.10 [2.96, 3.24] (bias +0.43) | 3.50 | 0 |
| total MAE vs actual | 14.40 [13.76, 15.01] | 14.52 | 14.05 [13.45, 14.65] |
| Brier P(home) | 0.2026 | 0.2061 | 0.1971 (de-vig) |
| Brier sim - close | +0.0054 [+0.0014, +0.0096] | +0.0090 [+0.0040, +0.0141] | |
| actual margin ~ close + sim | close 1.154, sim -0.110 | close 1.016, sim 0.053 | |

Reference points (same games): the team ridge alone is 3.08 MAE vs the close (totals 3.48). ESPN BET's own open is
1.87 from its close (totals 3.39). The sim (ACT) vs the OPEN is 2.74 (totals 3.21).

**Verdict against the stated bar** (mean |sim spread - close| <= ~1.5): **NOT met, at 2.30.** The player layer
improves on the team ridge (3.08 -> 2.30) and on the totals ridge (3.48 -> 3.10). But the sim sits further from the
close than the book's own opener does. Calibration is reasonable in the middle bins. The <0.2 bin is overconfident
toward the favourite (sim 0.16, won 0.05, n=38). Use it as a quick first-pass repricer, never as a bettable fair.

**Injury delta check** (a >= 28 mpg rotation player did not play; delta in home margin; move = ESPN BET open->close):
- News-type absences (he played the previous game): n=505, mean|sim delta| 2.02 vs mean|move| 2.25, corr 0.458,
  slope move~delta 0.52 [0.42, 0.62], sign agreement 69.4%.
- All such absences, sim delta vs the full rotation: n=929, corr 0.395, slope 0.49 [0.42, 0.57], sign 67.6%.
- What this means: the direction is mostly right. But the line moved only about half the sim delta from open to
  close. Two explanations are mixed together. (a) The open already priced part of the news. (b) The sim overstates
  single-player deltas on average, which fits an absence coefficient of about 4. Treat a sim delta as roughly 2x the
  market move you should expect after the open. It is an upper bound, not a fair price.

**Player distributions** (800 random 2024-25 team-games, players projected >= 15 min who played, ACT minutes):
10-90 coverage was 79.3% for points (bias +0.11, MAE 5.04), 77.4% for rebounds (+0.07, MAE 2.00) and 80.3% for
assists (-0.04, MAE 1.49). This check conditions on the player playing. It uses team points ~ N(exp, sigma_t/sqrt2)
as a stand-in for the joint game sim.

Minutes check (ACT sets, 2023-24 + 2024-25): projected vs actual by EWMA tier is 8.8 vs 10.7 (EWMA < 15), 19.4 vs
19.6, 27.4 vs 27.1, 32.5 vs 31.7, and 36.5 vs 34.8 (EWMA 34+). Projection MAE is at or below raw-EWMA MAE in every
tier. Stars are still about +1.7 min high, partly because EWMA itself runs +1.3 high for them (regression to the mean).

**Honesty note:** 2024-25 is a DEV season and I looked at it while building. I changed the minutes model three times
after seeing outputs: participation weighting, an additive rescale, and known-play sets. Each change was driven by
the minutes-bias diagnostic above, not by the market MAE. The headline moved 2.36 -> 2.34 -> 2.38 -> 2.30, with every
interval overlapping. So 2.30 is a dev number, not a holdout number. A clean read needs the forward 2026-27 season.

## Tonight (2026-10-29, run at about 17:40 PT; ESPN/Draft Kings line at the same time for reference)

- **CLE @ ATL**: P(ATL) 48.7c, fair spread ATL +1.0, total 238.0. Market: ATL -1.5, total 237.5 (open 239.5).
  The sim is 2.5 pts off on the spread.
- **delta CLE ATL --out "Jalen Johnson"**: P(ATL) 48.7c -> 47.0c (**-1.7c**). Mean margin -0.57, total -0.90. His
  32 min go mostly to Daniels, McCollum and Alexander-Walker (about +4.7 to +4.8 each). **This understates him**: his
  BPM-lite rating is only +0.15/48, because the live box regression (seasons 2023-24..2025-26, R² 0.277, n=1288) penalises FGA (-1.24/48) and
  turnovers (-3.04/48), so
  high-usage creators rate as average. On a star, a market move several times larger than the sim delta is the
  expected failure mode.
- **MEM @ POR**: P(POR) 63.4c, fair spread POR -5.0, total 231.0. Market: POR -7.5, total 232.5.

## Limitations / next steps
1. BPM-lite impact is crude (R² 0.277 in the live fit) and undervalues usage creators. A next step is to add an on-court
   +/- term from play-by-play stints (needs PBP), or to borrow a public impact metric.
2. There is no historical injury feed. Validation's ACT set is a lookahead proxy for the injury report (it includes
   late scratches). STRICT shows the cost of knowing nothing. Live mode uses ESPN's feed as of now.
   Questionable / day-to-day players are counted IN.
3. Early season: rosters are reshuffled. The ESPN roster add/drop handles trades, but acquired players keep their old
   team's rate context. Rookies start at replacement.
4. Not modelled: blowout minutes, foul trouble, minutes restrictions (use `--minutes`), matchups, and the
   correlation between player shares and the game margin.
5. Players' minutes are jittered, so a `--minutes` pin is the projection mean, not a hard value.
6. Fair spread = the median of integer simulated margins, so it is coarse in 1-pt steps. The model mean is printed
   next to it.

Files: scripts/nba_sim.py, tests/test_nba_sim.py (25 tests), results/nba_sim_params.json,
results/nba_sim/validate_2025{.json,_games.csv,_injury.csv}. There is a feature cache in data/nba_sim/ (gitignored).
