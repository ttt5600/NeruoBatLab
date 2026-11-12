# L30 P-PAINT -- reproduce, audit, and bet-test "teams allowing a high paint share fail to cover"

Agent topic `nba_l30_paint` (research subagent, 2026-11-11). Script `scripts/nba_l30_paint.py`. Scratch/results
`results/nba_l30_paint/`. Source claim: notes/nba_comebacks_matchups.md Q2 ("ATS diff opp paint share (H-A) -0.54
pts/SD [-0.79, -0.29] p 2.5e-5, 10/10 seasons"), scripts/nba_comebacks.py `build`/`q2`, results/nba_comebacks.json.

## How P-PAINT differs from what already died (read before anything was computed)
- **L2 team-rating sides (dead, -13.9% / 789 sealed)**: a margin-based Bayesian rating vs the closing ML. It asks "is the
  team better than the price says". It carries no shot-location information. P-PAINT is a *specific defensive style trait*
  (share of opponents' points scored in the paint), not team quality. The Q2 joint model already controlled for net
  rating (net coef 0.04) and paint stayed -0.74, so P-PAINT is not L2 re-run.
- **NBA style matchups (dead, notes/nba_style_matchups.md)**: tested the *interaction* (offense zone mix x defense zone
  allowed), a product of deviations with SD ~0.25 pts -- too small by construction. P-PAINT is a *marginal* defensive
  trait. BUT that study also reported, descriptively, "plain zone marginal (offense + defense, no interaction) vs close
  margin residual +0.000 pts/SD [-0.34, +0.39]" on 2021-22..2024-25 -- i.e. a pbp zone-marginal expected-points term was
  fully priced. That term is points-per-shot weighted, not "share of points allowed in the paint", so it is not the same
  variable, but it is the closest prior evidence and it points AGAINST P-PAINT. Prior: low-moderate.

## PRE-REGISTRATION (written 2026-11-11 BEFORE the reproduction was run and before any bet was scored)
Looked at before writing: column names/coverage of data/nba_comebacks/teamlogs.parquet and data/espn/nba/lines_*.parquet
(no outcomes); the original Q2 code and its published numbers.

### Step 1 -- reproduction + leak audit (no new trials; descriptive)
- Independent recomputation from raw `teamlogs.parquet` (stats.nba.com team advanced logs), NOT reusing games.parquet
  features: opp_paint_share(team, game) = OPP_PTS_PAINT / opponent PTS. Season-to-date value for a game on ET date d =
  mean over the team's same-season regular-season games with date < d (strict). Need >= 10 prior games for both teams.
  x = home value - away value; z within season (as the original) -> OLS ats_res ~ z, HC1. Lines = games.parquet `spread`
  (the original's main line) for the reproduction only.
- Audit checks (printed): (a) OPP_PTS_PAINT of team A == PTS_PAINT of team B in the same game; (b) no team has two
  games on one date; (c) the feature on game g equals the hand-computed mean of strictly earlier games (spot rows);
  (d) shifting the feature to INCLUDE the current game (deliberate leak) changes the slope a lot -> shows the test is
  sensitive; (e) z with walk-forward scale (SD from earlier seasons) vs within-season scale; (f) per-line-source
  (ESPN pickcenter vs OddsData) and vs DraftKings / ESPN BET CLOSE from ESPN core odds (2021-22..2024-25).
- traps.py: date handling goes through ~/agent-harness/lib/traps.py checks where applicable (all dates here are
  calendar dates from strings; no epoch conversion).

### Step 2 -- walk-forward betting trials (k = 2, edgelab family nba_sides; bar = edgelab's family bar, ~2.6e-4)
Signal and threshold fixed now. Sign fixed a priori (fade the side whose defense concedes the higher paint share).
- z_g = x_g / SD_prev(s), SD_prev = SD of x over all qualifying regular-season games of dev seasons < s (season 2015-16
  is burn-in only: scored seasons 2016-17..2024-25).
- **PP1 (-110, main line)**: if z >= +1.0 bet AWAY against the spread; if z <= -1.0 bet HOME; at the main closing line
  of games.parquet (ESPN pickcenter 2016-17..2022-23, OddsData 2023-24..2024-25), price -110 (1/1.909). Pushes = 0 pnl,
  kept. Seasons 2016-17..2024-25.
- **PP2 (real price)**: same selection, but line and price = ESPN core odds DraftKings plain (closing) spread and its
  spread odds; ESPN BET when DK is missing. Seasons 2021-22..2024-25 (where ESPN core odds exist).
- Metric: observed ROI; 95% CI and one-sided p from a DATE-clustered bootstrap (10,000, seed 2026). Also edgelab's own
  evaluate (its gates: p < family bar, both halves > 0, concentration, n >= min_n).
- PASS = edgelab dev pass (p < family bar ~2.6e-4 AND halves > 0). KILL as a standalone bet = PP1 p >= bar.
- Descriptive only (not trials): |z| >= 1.5 / 2.0 slices; win rate vs 52.38%.
- **Power, written now:** -0.54 pts/SD; bets at |z| >= 1 have mean |z| ~1.5 -> ~0.8 pts edge -> ~+2.5 pp cover prob on
  an ATS SD of ~12.9 -> ~52.5% ~= break-even at -110. Expected ROI ~0%; the bar is unreachable even if the slope is
  exactly true. A dev "pass" would therefore be a bug until rows are inspected.

### Step 3 -- market learning (descriptive)
Per-season slope (pts per within-season SD) with 95% CI, n; OLS of slope on season index (weighted by 1/se^2) with CI;
early (2015-16..2019-20) vs late (2020-21..2024-25) and the most recent two seasons separately.

### Verdict rule
- dead: PP1/PP2 fail and the per-season slope is not distinguishable from 0 in recent seasons, OR a leak is found.
- lead (feature only): reproduces leak-free, robust to the closing line, but too small to bet alone -> feature for a game
  model / CLV logging.
- ready for sealed ask: only if PP1 or PP2 passes the edgelab dev gate (expected: no).

## Results
(filled in below as they come; nothing above is edited after a run)

### Step 1 -- reproduction + audit (results/nba_l30_paint/repro.json; `nba_l30_paint.py repro`)
- **Reproduces exactly:** -0.537 pts/SD (se 0.128, p 2.5e-5, n 10,403) from raw teamlogs. My independently computed
  point-in-time feature equals the original `s_opp_paint_share` to 2e-16 (max abs diff).
- **No lookahead found.** Feature = mean of the team's same-season regular-season games strictly before the date
  (cumsum minus current / n_prior). One game per team per date (asserted). OPP_PTS_PAINT == the opponent's own PTS_PAINT
  in 0 mismatching rows. Deliberate leak (include the current game) moves the slope to -0.36, so the test is sensitive to
  the window. Within-season z (mild scale lookahead) vs walk-forward z (SD from earlier seasons): -0.58 vs -0.55 on the
  same 9,330 rows, so the scale does not matter.
- 5 neutral-site 2024-25 games (both rows "@") dropped. In the original, `h.join(a)` would have duplicated them. The
  effect is negligible (n 10,403 in both).
- **Line source:** ESPN pickcenter rows -0.47 (p .0007, n 8,267); OddsData rows (2023-24, 2024-25) -0.77 (p .012, n
  2,131). Against the ESPN core odds **DraftKings/ESPN BET CLOSE** (2021-22..2024-25, n 4,276; sign check corr .997 with
  the main line) it is -0.72 (se .21, p .0006). DK-only: -0.71 (p .003). So the effect is not a stale-line artifact.
- Mechanism: per SD the trait moves the raw margin by -0.75 pts (p 1e-7), but it moves the spread by only -0.22.
  The market moves about 30% of the way.

### Step 3 -- market learning (results/nba_l30_paint/trend.json)
| season | pts/SD | 95% CI | n |
|---|---|---|---|
| 2015-16 | -0.17 | [-0.89, +0.55] | 1,073 |
| 2016-17 | -0.40 | [-1.15, +0.35] | 1,074 |
| 2017-18 | -0.32 | [-1.05, +0.40] | 1,075 |
| 2018-19 | -0.61 | [-1.36, +0.14] | 1,074 |
| 2019-20 | -0.89 | [-1.72, -0.07] | 906 |
| 2020-21 | -0.04 | [-0.87, +0.80] | 924 |
| 2021-22 | -0.87 | [-1.72, -0.02] | 1,073 |
| 2022-23 | -0.52 | [-1.25, +0.22] | 1,073 |
| 2023-24 | -0.76 | [-1.61, +0.09] | 1,060 |
| 2024-25 | -0.78 | [-1.64, +0.07] | 1,071 |
The effect is **not shrinking**. WLS trend is -0.054 pts/SD per season [-0.118, +0.009] (p .085), so if anything it is
getting slightly bigger. Early 2015-19: -0.47 (p .007). Late 2020-24: -0.61 (p .001). Last two seasons: -0.77 (p .012).
Each single season's CI is ±0.75-0.85, so a season-by-season story is not resolvable.

### Step 2 -- pre-registered bet trials (results/nba_l30_paint/bets.json, bets_PP1.csv, bets_PP2.csv; edgelab dev)
| trial | bets | win | ROI (date-cluster 95% CI) | p | halves | edgelab |
|---|---|---|---|---|---|---|
| PP1 main line, -110, 2016-24 | 2,972 | 52.38% | -0.01% [-3.56, +3.43] | .50 | -1.6% / +1.5% | dead (bar 2.5e-4) |
| PP2 DK/ESPN BET close, real odds, 2021-24 | 1,160 | 52.70% | +0.66% [-5.09, +6.58] | .41 | -0.3% / +1.7% | dead |
Edgelab: `l30_paint_pp1-756c5fbc`, `l30_paint_pp2-6958e251`, family nba_sides now 197 trials. Its evaluate reproduced
these ROIs ([-3.43, +3.35] / [-5.12, +6.40]). PIT violations: none. PP2 covers 1,160 of 1,161 eligible signal games
(DK 947, ESPN BET 213). Per-season PP1 ROI runs from -8.6% (2022-23) to +8.1% (2024-25).
Descriptive only (not trials, post hoc): |z| >= 1.5 PP1 1,328 bets +3.9%; |z| >= 2 501 bets +1.6% (not monotone).
**The bets carry the points but not the covers.** PP1 bets beat the closing spread by +0.86 pts on average (se 0.24,
roughly [+0.39, +1.33]), which matches the regression. But the win rate is exactly 52.38%, i.e. break-even at -110.
This is what the pre-registered power note predicted (~0.8 pts ≈ +2.5 pp of cover probability ≈ break-even). The
residual's mean shifts partly through blowouts, which do not change how often the bet covers.

### Verdict: **LEAD (feature only), DEAD as a standalone bet. Not a sealed ask.**
The effect is real, leak-free, holds against the DK close and is not decaying. But it is about 0.5-0.7 pts per SD, and
at -110 that buys no ROI (2,972 bets, -0.01%). Best use: one feature in a game-level spread model, alongside other
sub-point signals (L31 P-TOT3 3P% regression, ALT altitude +1.0 ATS), where several small signals stack into a usable
edge. It could also be logged as CLV against the opener. A sealed 2025-26 shot of PP1 would have about 0 expected ROI,
so it would waste the sealed season.

## Learnings
### A significant ATS slope can still be worth 0% ROI
L30 P-PAINT: -0.54 pts/SD, p 2.5e-5, reproduced exactly and leak-free. Fading |z| >= 1 at -110: 2,972 bets, 52.38%,
ROI -0.01% [-3.6, +3.4] (results/nba_l30_paint/bets.json). The bets beat the close by +0.86 pts (se .24) but cover no
more often than break-even. Rule of thumb: on the NBA ATS SD (~13), 1 pt ≈ 3 pp of cover probability at best. You
need ≥ ~1.5 pts per bet before -110 pays. Do the power arithmetic before running a bet trial on a "significant" slope.

### Paint share allowed is under-priced by ~70% but stable, not decaying
Per SD of the opp-paint-share difference: raw margin -0.75, closing spread only -0.22. The ATS slope is -0.47 in
2015-19 and -0.61 in 2020-24, with a trend of -0.054/season [-0.118, +0.009] (results/nba_l30_paint/trend.json,
repro.json). Against the DraftKings close in 2021-24 it is -0.71 (p .003). Treat it as a durable model feature, not a
standalone bet.

### stats.nba.com team logs: neutral-site games have no home row
Five 2024-25 regular-season games (Paris, Mexico City, Cup) show "@" for both teams. Any `is_home` split followed by a
join on GAME_ID duplicates them. Filter to games with exactly one home row (scripts/nba_l30_paint.py game_table).
