# NBA matchup graph: WHO GUARDS WHOM (offensive player x defender)

User idea 2026-10-30: "look at positional relationships, graph those relationships out as edges and simulate them".
Bipartite graph: nodes = players, edge (offensive player i, defender j) weighted by partial possessions, with
outcomes (points, FGA, AST, TOV per partial possession). Fed into nba_sim only if a component is real.

Not a redo: notes/nba_player_opp_venue.md tested player x opponent TEAM (reliability r .017, ~0.36 pt, DEAD);
notes/nba_style_matchups.md tested team style (0/4, DEAD). Neither used individual defender assignment.

Code: scripts/nba_matchup_fetch.py (fetch), scripts/nba_matchup_graph.py (parse / reliability [/ trials]).
Data: data/nba_matchups/ (gitignored). Results: results/nba_matchup_graph/. Ledger dataset `nba_matchup_graph`.

## Data (Step 0, checked 2026-10-30)
- stats.nba.com `boxscorematchupsv3?GameID=...` answers with browser headers (UA, Referer/Origin nba.com,
  x-nba-stats-origin/token, Sec-Fetch-*); plain curl hangs. Per-game data exist at least for 2019-20 .. 2026-27
  (probed game 0001 of each season: 165-232 pairs per game).
- Orientation (checked on 0022300061 DEN-LAL vs boxscoretraditionalv3): `players` = OFFENSIVE player, `matchups` =
  defenders guarding him. Sums of matchup FGM/TOV per offensive player match the box score for most players (Murray
  8 FGM both, KCP 9 vs 8, TOV equal for 5/6 DEN starters), but NOT for all: Jokic matchup FGM/FGA 17/32 and
  playerPoints 40 vs box 12/22 and 29; team sums of playerPoints run ~10% above team points (131 vs 119, 118 vs 107).
  So matchup "points" are NBA tracking attributions with some double counting; they are used as-is (per partial
  possession), and this quirk is a reason any simulator must re-scale to box-score points, not add matchup points.
- Fetched: regular season 2022-23, 2023-24, 2024-25 (DEV), 2025-26 and 2026-27-to-date (descriptive / point-in-time
  state for the worked example only; 2025-26 props remain SEALED and are not loaded).

## PRE-REGISTRATION -- Step 1 reliability gate (written 2026-10-30 BEFORE any reliability number was computed;
## only the one game above was looked at, for orientation)

Unit: pair-season (or pair-half) aggregate over games: n_ij = partial possessions, y_ij = 100 * playerPoints / n_ij.
Seasons used: 2022-23, 2023-24, 2024-25 regular season only (2025-26 is not used for any reliability number).

Model, fit separately in every split cell (each season; each half of each season):
  y_ij = mu + a_i + d_j + e_ij,  weighted least squares with weights n_ij, ridge penalty on a and d with
  lambda = 300 partial possessions each (a priori; i.e. a player with 300 possessions is shrunk ~halfway).
  Raw interaction r_ij = y_ij - (mu + a_i + d_j) for pairs with n_ij > 0.
  Empirical Bayes for m_ij: Var(r_ij) = tau^2 + sigma^2 / n_ij. sigma^2 = per-partial-possession variance of points
  (x100^2), estimated from game-level pair rows in that cell as the weighted within-pair variance; tau^2 by method
  of moments over pairs with n_ij >= 20 in the cell, floored at 0. Shrunk m_ij = r_ij * tau^2 / (tau^2 + sigma^2/n_ij).
  Defender main effect d_j = the ridge estimate (it is already shrunk by lambda).

Splits:
  - Season-to-season (PRIMARY): pairs (2022-23 -> 2023-24) and (2023-24 -> 2024-25), stacked.
  - Split-half: within each season, games sorted by (date, game id), alternate games -> half A / half B; stacked
    over the three seasons.
Eligibility (fixed a priori): m_ij: pairs with n_ij >= X = 50 partial possessions in BOTH cells.
  d_j: defenders with >= 500 partial possessions defended in BOTH cells.

Reliability r = Pearson correlation across eligible units of the shrunk estimate in cell 1 vs cell 2 (primary);
raw-residual correlation also reported. 95% CI by bootstrap 2000, resampling OFFENSIVE players (m_ij) or defenders
(d_j) with replacement (clusters); Fisher-z CI printed alongside. If tau^2_hat = 0 in a cell, m_ij is DEAD
(no persistent interaction variance) regardless of r.

Persistent SD: persistent variance (per 100) = cov(raw estimate cell 1, raw estimate cell 2) across eligible
units (noise is independent across cells, so the covariance estimates the persistent variance), SD = sqrt(max(cov,0)).
For m_ij the raw residuals r_ij are used. For d_j the ridge (lambda=300) estimates are used in both cells (an
unpenalized fit is unstable); shrinkage makes this covariance UNDERSTATE the persistent SD somewhat, so a narrow
d_j failure is flagged as such, with the de-shrunk value sqrt(cov)/mean shrink factor printed alongside.
Conversion to points per game for a typical starter: an offensive player-game has defender exposures n_ij(g);
if pair effects are independent across defenders, SD_game = SD_per100 * sqrt(sum_j n_ij(g)^2) / 100.
"Typical starter" = median of sqrt(sum_j n_ij(g)^2) over offensive player-games with total partial possessions
>= 55 (2023-24). Printed also: the defender-total version SD_per100 * 70/100 (all possessions a defender faces).

GATE (per component, m_ij and d_j, on points): season-to-season r >= 0.15 with 95% bootstrap CI excluding 0 AND
persistent SD_game >= 1.0 point for a typical starter. Otherwise that component is DEAD, printed as
"~x pts persistent". Secondary outcomes (FGA, AST, TOV per 100 partial possessions) are DESCRIPTIVE only (r printed,
not gated). If both m_ij and d_j fail: stop; no Step 2, no prop trials, no ledger trials (a ledger row records the
gate failure).

Step 2 (graph simulator + trials G1/G2/G3 at the open) is pre-registered here ONLY if a component passes, in a
separate commit before any outcome is joined to a line.

### AMENDMENT A1 (2026-10-30, BEFORE any reliability number; reason = fetch throttling)
The per-game fetch stalled after 599 games of 2023-24 (2023-10-24 .. ~2024-01-06): from 15:03 every request
timed out (stats.nba.com soft throttle at ~1 req/s). Per-game history for >= 2 seasons is therefore NOT in hand.
`leagueseasonmatchups` (PerMode=Totals) returns ALL pairs of a season in one request (2023-24: 137,763 pairs with
GP, PARTIAL_POSS, PLAYER_PTS, MATCHUP_FGA/AST/TOV) -- the same pair-season sums the per-game rows would give.
Changes (everything else as pre-registered):
- Season-to-season (PRIMARY gate): pair-season and defender-season aggregates come from leagueseasonmatchups
  totals for 2022-23, 2023-24, 2024-25.
- sigma2 (per-possession variance, needed for EB shrink of m_ij) is estimated once from the 599 per-game 2023-24
  games and used for every cell.
- Split-half: only within the 599 per-game 2023-24 games (alternate games), secondary.
- "Typical starter" exposure from those 599 games.
- Season totals contain the whole season = LOOKAHEAD for any trial; they are used for the reliability gate only.
  If a component passes, Step 2 needs per-game data (re-fetch at a slower pace) before any trial.

## STEP 1 RESULT (reliability gate, run 2026-10-30, results/nba_matchup_graph/reliability.json)
Points per 100 partial possessions, season-to-season stacked (22-23->23-24, 23-24->24-25):
- m_ij (pair interaction): 933 pairs >= 50 poss both seasons. tau2_hat = 0 in a cell (EB shrinks every pair to
  0); raw-residual r 0.069 [-0.001, 0.136]; persistent SD 2.2/100 -> **DEAD (~0.70 pts persistent)** for a typical
  starter (sqrt(sum n^2) = 31.6 poss). Split-half (599 games) has only 7 eligible pairs: uninformative.
- d_j (defender main effect): 688 defenders >= 500 poss both seasons. r 0.857 [0.825, 0.879]; persistent SD
  4.6/100 -> 1.46 pts/game for a typical starter (de-shrunk 1.65) -> **PASSES the gate** as written.
  Split-half 0.783 [0.728, 0.830].
- CAVEAT (looked at after the gate, before Step 2): d_j is mostly POSITION / role. 2023-24 ridge d by position:
  C +6.4, F -4.3, G -3.7 per 100; the "worst" defenders are rim protectors (Brook Lopez +12.9, Lively, Zubac,
  Gafford), the "best" are wings (KCP -11.4, Jalen Williams, Jaylen Brown). Centers guard bigs whose matchup points
  per partial possession are high; it is not "bad defense". corr(d_j, poss-weighted a_i of his assignments) = -0.49.
  Persistence of position is trivially high, so the gate pass says little; what matters for a bet is the
  opponent-specific DEVIATION from a player's usual defender mix, which is what Step 2 measures.
- Secondary: FGA d_j r .909, AST .805, TOV .622; m_ij FGA r_raw .142 [.072, .209] (0.65 pts-equivalent), AST/TOV ~0.

## PRE-REGISTRATION -- Step 2 graph signal + trials (written 2026-10-30 BEFORE any signal was joined to a line,
## prop price or outcome; no signal distribution looked at yet)
Per-game history is not available (A1), so point-in-time state = the PREVIOUS SEASON's totals only (strictly
before every game of the trial season; no lookahead). Season s game uses season s-1 totals:
2022-23 <- 2021-22, 2023-24 <- 2022-23, 2024-25 <- 2023-24. No within-season updating (stale by design).

Signal for offensive player i (NBA personId) in ESPN game g vs opponent O:
- D_O(g) = players with minutes > 0 for O in O's most recent ESPN game of the same season strictly before g
  (point-in-time roster/rotation; O's first game of the season -> no signal).
- n_ij = prior-season partial possessions of i vs j (any team). Require sum_{j in D_O} n_ij >= 30, else no
  signal (the a-priori FALLBACK is "no signal"; position-similar imputation is not attempted).
- share_j = n_ij / sum_{k in D_O} n_ik.  d_j = prior-season ridge defender effect (lambda 300, Step 1 model).
- usual_i = poss-weighted mean of d_j over ALL of i's prior-season pairs (his normal defender mix).
- P_i = i's prior-season partial possessions per game = (sum_j n_ij) / max_j GP_ij.
- graph_delta = (sum_j share_j d_j - usual_i) * P_i / 100   [points; matchup points are used unscaled].
  m_ij is DEAD and not used.
- ID join NBA -> ESPN: normalized name (accents stripped, lowercase letters only, suffix jr/sr/ii/iii/iv dropped,
  nba_injury_teammate_props.NAME_ALIAS applied) within the season's ESPN box; ambiguous/unmatched dropped,
  match rate reported.
Prop data: as nba_injury_teammate_props: DEV ESPN graded 2023/2024/2025 (DraftKings/DraftKings/ESPN BET),
POINTS market only, OPENING line and price, one quote per player-game (no-vig p closest to .5), pushes out.
Trials (3; Bonferroni bar p < 0.05/3 = 0.0167, date-clustered bootstrap 4000, ROI > 0 in both date halves;
observed ROI with 2.5/97.5):
- G1 OVER at the open when graph_delta >= +1.5.
- G2 UNDER at the open when graph_delta <= -1.5.
- G3 OLS slope of (actual - line_open) on graph_delta, all quotes with a signal; pass = slope > 0 at the bar,
  both halves > 0.
Too-good guard: ROI > +10% on > 200 bets -> inspect rows before reporting. SEALED 2025-26 (Oct-Nov ESPN BET)
only after a dev pass, once (state would be 2024-25 totals).
Descriptive (not trials): how much of graph_delta the opening line already has: slope of (line_open - player's
prior-10-game points mean) on graph_delta.

### AMENDMENT A2 (2026-10-30, bug fix BEFORE any outcome/line join)
GP in leagueseasonmatchups counts the games a PAIR met, not the player's games, so P_i = sum n / max GP was ~500
"possessions per game" and graph_delta had SD ~15 pts. Fixed: P_i = prior-season matchup possessions / prior-season
ESPN games played (minutes > 0), name-matched the same way. Signal after the fix (no outcomes): SD ~0.95-1.01 pts
per season, 11.1% of player-games beyond +-1.5; 52,549 player-games with a signal; name match 99.9%+ of minutes.
