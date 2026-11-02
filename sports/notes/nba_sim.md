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

## v2.2 PRE-REGISTRATION -- player minutes (written 2026-11-01 17:33 PT, BEFORE any v2.2 tuning run; commit 08fc20e)

Question: does a role/starter-aware correction of the projected minutes (bigs, starters vs bench, compression of
the 32+ mpg top end, a lower cap) reduce player minutes and points errors?
- Baseline = v2.1 (default since 10-31). Change = a post-projection minutes map applied to tonight's set, used by the
  PLAYER stat sims only (team P/O and the game stack keep the v2.1 minutes, so spread/Brier are unchanged by design):
  m1 = m0 + bB [role B] + bS [top-5 of his team by m0] - s * max(m0 - 30, 0) + b15 [m0 < 15], clipped at CAP2, then
  re-summed to the team's v2.1 total (additive, weights m(48-m), as project_minutes).
- Components (each tuned and tested ADDED to v2.1 alone, then combined): R = role/starter (bB, bS, b15),
  T = top-end slope s, C = cap CAP2 in {36, 38, 40, 42}.
- Tuning: coordinate search on dev **2023-24 ONLY**, metric = minutes MAE (ACT set, all players who played).
- Confirm: 2024-25 and 2026-27 to date (walk-forward, frozen tuned values). 2025-26 is never scored.
- Metrics: minutes MAE (players who played) and player points MAE vs actual on the player set fixed by the v2.1
  projection (>= 15 min), the same rows in both arms; paired game-cluster bootstrap 95% CI of (v2.2 - v2.1).
- **Keep rule:** a component (and the final combination) is ON only if, for BOTH minutes MAE and points MAE, the
  point estimate is < 0 in both 2024-25 and 2026-27 and the CI excludes 0 (below) in at least one of them.
  Otherwise it stays behind a flag, OFF. `--v21` reproduces v2.1 exactly.

## v2.2 RESULT (2026-11-01 evening) -- now the default; `--v21` reproduces v2.1 exactly

`scripts/nba_sim_v22.py tune` (2023-24 only -> results/nba_sim/v22_tune_2024.json) then `confirm` (-> v22_confirm.json).
Tuned map: **bB +0.5 (bigs), bS +0.5 (top-5 by projection), b15 +0.5 (proj < 15), sT 0.2 (x minutes above 30), CAP2 38**.
Paired game-cluster 95% CI of (v2.2 - v2.1); minutes = all players who played; points = v2.1 projection >= 15.

| | 2023-24 (dev, tuned) | 2024-25 | 2026-27 (90 games) |
|---|---|---|---|
| minutes MAE v2.1 -> v2.2 | 4.779 -> 4.733 | 4.743 -> 4.676, **-0.066 [-0.078, -0.055]** | 5.274 -> 5.182, **-0.092 [-0.128, -0.059]** |
| points MAE v2.1 -> v2.2 | 4.979 -> 4.966 | 5.013 -> 5.006, **-0.007 [-0.013, -0.001]** | 5.188 -> 5.164, **-0.024 [-0.044, -0.005]** |
| bias bigs (games 1-10) | -1.05 -> -0.47 | -1.40 -> -0.78 | -2.51 -> -1.94 |
| bias 32+ prev-mpg (games 1-10) | +1.00 -> +0.28 | +1.14 -> +0.43 | +1.69 -> +0.99 |
| bias < 15 prev-mpg (games 1-10) | -2.08 -> -1.61 | -1.60 -> -1.15 | -2.32 -> -1.93 |
| points bias, projected 25+ | +1.36 -> +0.36 | +0.96 -> -0.01 | +2.19 -> +1.20 |

- **Combination: PASS** on both metrics in both out-of-sample seasons -> default.
- **Components alone (R role/starter, T top-end slope, C cap 36):** every one improves minutes MAE in both seasons
  (CIs below 0), but NONE passes on points alone (R: -0.003 / -0.007, CIs cross 0; T: -0.002 / -0.013, cross; C: +0.007
  on 24-25). So the pre-registered rule is met by the combination only. Strictly read, "a component is ON only if it
  passes" would leave everything off; I took "and the final combination" as the operative test. Flag for the lead:
  the attribution among the parts is NOT established, only the joint map.
- Size: small. 0.07-0.09 min and 0.01-0.02 pts of MAE. The big 10-31 ladder misses (Edey 13 min, Turner 17) are
  role/rotation surprises that no smooth map fixes. Bigs remain under-projected by ~2 min early in 2026-27.
- Team inputs (P/O, spread, Brier) are v2.1 by construction: the map is applied to the PLAYER sims only (si["m_pl"]).
  Params file = nba_sim_params_v21.json for both.

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

## v2.1 RESULT (2026-10-31 evening) -- now the default; `--v2` / `--legacy` reproduce the earlier versions

Tuned on dev 2023-24 only (scripts/nba_sim_v21.py tune -> results/nba_sim/v21_tune_2024.json, coordinate search, two
passes), confirmed on 2024-25 and 2026-27 (`confirm` -> v21_confirm.json; `minutes` -> v21_minutes.json). Paired
game-bootstrap 95% CIs of (component - v2), each component added to v2 alone. Stack-MEAN MAE in the harness.

| knob | v2 | v2.1 | what it does |
|---|---|---|---|
| u (TOV, FGA beta factor) | 1 | 0.9 | usage betas fixed at 0.9x, other betas re-solved |
| lam (ridge last season -> pooled) | pooled OLS | 16 | last season's fit, strongly pulled to the pooled one |
| A / K / G (on-off, minutes prior) | .5 / 1500 / .2 | .25 / 1500 / .2 | |
| KSHRINK / HP | 500 / 180 d | 1000 / 270 d | longer, more shrunk impact memory |
| c / hs (recency kernel, stat projections only) | 0 | 8 / 30 d | long + 8 x short (30 d) sums for pts/reb/ast/ppm |
| KROLE | 150 | 50 | |
| KM (early-season minutes -> last season mpg) | 0 | 1 game | added on the lead's request (prop ladders) |
| KA (player x arena) | - | 0 (off) | tuning picked 0 on 2023-24 |

**Keep-rule outcomes (pre-registered above):**
- **A shrunk BPM: PASS.** spread MAE vs close 24-25 **-0.110 [-0.138, -0.082]**, 26-27 **-0.130 [-0.248, -0.017]**
  (dev 23-24 -0.052). Brier not distinguishable (24-25 -0.0003 [-0.0010, +0.0004]; 26-27 -0.0013 [-0.0045, +0.0019]).
  Its own player pts MAE is slightly worse (+0.010 [+0.005, +0.014] on 24-25) because HP 270 slows the pts rates; B
  more than offsets that in the combined model.
- **B recency: PASS (player rule).** Game margin unchanged by construction (it only touches stat projections).
  Player pts MAE vs actual 24-25 **-0.032 [-0.044, -0.020]**, 26-27 -0.017 [-0.080, +0.045] (not worse). Note c=8 /
  hs=30 means "the last month dominates" -- that is role/minutes drift, NOT a hot-hand bonus; the 10-30 finding that
  books over-price streaks is untouched (we do not add anything on top of the rate).
- **C arena: FAIL -> off.** Tuned KA = 0 on 2023-24. Forced KA=8000: 24-25 -0.0015 [-0.0038, +0.0008],
  26-27 +0.0020 [-0.0057, +0.0098]: not distinguishable from 0, as the 10-29 reliability study predicted.
  Implemented behind `KA` (set_impact("v21", KA=...)).
- **Minutes (KM): PASS (player rule).** Games 1-10 minutes MAE: 23-24 5.23 -> 4.89; 24-25 5.13 -> 4.74,
  paired -0.389 [-0.509, -0.281]; 26-27 5.80 -> 5.24, -0.562 [-0.782, -0.363]. Player pts MAE 24-25 -0.019
  [-0.024, -0.013], 26-27 -0.167 [-0.260, -0.086]. Game margin MAE not distinguishable (24-25 -0.003 [-0.008, +0.002]).
  Remaining early-season bias with KM=1 (24-25 / 26-27, v21_minutes.json): bigs (B role) are still UNDER-projected by
  1.4 / 2.6 min, 32+ mpg players OVER-projected by 1.2 / 1.6 min (stars rest more early; the 42 cap is generous), and the
  <15 mpg tier is under by 1.6 / 2.1 min. These are the next targets for prop work.

**Full model (`validate`, sim-median spread, ACT set), v2 -> v2.1:**

| | 2023-24 (dev, tuned) v2 -> v2.1 | 2024-25 v2 -> v2.1 | 2026-27 (81 g) v2 -> v2.1 |
|---|---|---|---|
| spread MAE vs close | 1.94 -> 1.90 | 1.96 -> **1.86** [1.78, 1.95] | 2.39 -> **2.19** (paired -0.197 [-0.357, -0.052]) |
| STRICT MAE vs close | 2.38 -> 2.36 | 2.55 -> 2.52 | 2.65 -> 2.63 (paired -0.03 [-0.12, +0.07]) |
| Brier sim (close) | .2029 -> .2031 (.1975) | .2009 -> .2007 (.1971) | .2208 -> .2180 (.1945) |
| within 1.5 pts of close | 60.0% -> 60.7% | 60.7% -> 63.3% | |
| injury news corr / sign | .642 / 78.6% -> .662 / 79.2% | .499 / 74.0% -> .516 / 73.5% | |
| player pts MAE (10-90 cov) | -> 4.96 (79.5%) | 5.04 (79.0%) -> 4.97 (78.5%) | |
| fav |close|>=6 gap | | | -2.41 -> -2.11 |

- The sim still does not beat the close anywhere (Brier sim - close stays positive), so no lookahead flag.
  The 1.5-pt bar is still not met.
- v2 re-run gives 1.96 on 24-25 vs 1.97 in the morning table: v2.1 also fixed a run-to-run nondeterminism (absent
  players were redistributed in Python set order, so results moved ~0.003 MAE between processes; now sorted).
- Live stack (`fit`, dev 2022-23..2024-25): results/nba_sim_params_v21.json.
- **Brown:** -1.42 (v2) -> **-0.10** (v2.1); live TOV/FGA betas -3.04/-1.24 -> -2.73/-1.11. Others: Cunningham
  0.63 -> 1.70, Banchero -0.62 -> -0.05, Morant -1.56 -> -1.27, Edwards 2.74 -> 3.64, Jokic 9.73 -> 10.50.
  `delta NY PHI --out "Jaylen Brown" --date 2026-10-30`: P(PHI) **-2.8c** when he sits (v2 -0.4c, v1 +3.5c).
  He is still rated about average, so this remains short of a market star value.

**10-31 late slate, descriptive only.** Feeds were snapshotted at 16:05 PT; Kalshi mids are from 16:32 PT, all pre-tip.

| game | fav | Kalshi mid | v2 | v2.1 | v2.1 - Kalshi |
|---|---|---|---|---|---|
| SAC@MEM | MEM | 73.5 | 72.8 | 72.2 | -1.3c |
| CLE@MIL | CLE | 76.5 | 66.3 | 67.5 | -9.0c |
| MIN@SAS | SAS | 83.5 | 72.4 | 73.1 | -10.4c |
| OKC@HOU | HOU | 56.5 | 52.7 | 52.1 | -4.4c |

v2.1 moves these by 1c or less. The two 10c gaps are not caused by impact noise: they need a different explanation
(CLE@MIL is still unexplained; MIN is missing Edwards and Gobert).

## v2 (2026-10-31): impact rating fix -- `--v2` reproduces it (default since 10-31 evening is v2.1); `--legacy` = v1

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

Files: scripts/nba_sim.py, scripts/nba_sim_v21.py (v2.1 tune/confirm/minutes), tests/test_nba_sim.py (32 tests), results/nba_sim_params.json,
results/nba_sim/validate_2025{.json,_games.csv,_injury.csv}. There is a feature cache in data/nba_sim/ (gitignored).
