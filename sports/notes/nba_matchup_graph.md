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
