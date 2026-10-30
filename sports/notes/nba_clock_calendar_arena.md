# NBA clock / calendar / arena factors (body clock, early tips, day of week, nightlife, arena shooting, altitude props)

PRE-REGISTERED 2026-10-29, written BEFORE any outcome was joined to any signal, price or prop line.
What had been looked at before writing (structure only): ESPN games/lines/box/graded schemas; per-season line coverage
(DK 2021-22..2023-24, ESPN BET 2024-25, DK 974 + ESPN BET 257 in 2025-26; spreads/totals/odds present on ~1,230 games a
season); the ET tip-hour distribution (all tips on :00/:30, 12:00-22:30 ET, plus a few 23:00 in 2025-26); the venue city/state list; prop
market counts per book. No ROI, hit rate, margin or coefficient had been computed for any signal below.
SITUATIONAL_FACTORS.md says rest/b2b/3-in-4/travel km/trip game #/altitude home ATS are tested (priced or lead); none of
the six signals below is on the dead list (rows: "Early tip -> under" UNTESTED, "Nightlife" UNTESTED, "Arena sightlines"
LIT, "Visitor props at altitude" UNTESTED, "Westward/eastward jet lag" LIT; body-clock tip for NBA never tested).

Script: `scripts/nba_clock_calendar_arena.py` (one run; scores DEV, then SEALED only for trials that pass DEV).
Output: `results/nba_clock_calendar_arena/`. Ledger: `lab/ledger.jsonl`, dataset `nba_clock_calendar_arena`, one row per trial.

## Data
- Games: `data/espn/nba/games_{2022..2026}.parquet` (ESPN season label = year the season ends), regular season,
  STATUS_FINAL. Tip time = `start_utc` (tz-aware UTC; ESPN scheduled tip).
- Game lines: `lines_{season}.parquet`, ONE book per event: DraftKings if it has both MLs else ESPN BET (same rule as
  nba_early_season_bias.md, where plain fields = close 98-100% of the time). Fields: `spread` (home line),
  `home_spread_odds`/`away_spread_odds`, `total`, `over_odds`/`under_odds`, `home_ml`/`away_ml`. Paid at those actual
  close prices; rows with a missing price for the bet leave that trial. Pushes = 0 profit, kept.
- Props: `graded_{2023,2024,2025}.parquet` at the OPENING line/price (`line_open`, `dec_open`) -- ESPN "current" is partly
  in-game (nba_player_opp_venue.md). Book: DK 2022-23 & 2023-24, ESPN BET 2024-25 (sealed 2025-26 ESPN BET, Oct-Nov only).
  One quote per player-game-market: the over/under pair at the open line whose proportional no-vig p_over is closest to .5.
  Pushes leave. Markets: points, rebounds, 3-point FG made. Join on ESPN (event_id, athlete_id) to box scores;
  player's team from the box.
- Local time: venue time zone from venue city/state (AZ -> America/Phoenix, no DST; CO/UT -> Denver; CA/OR/WA/NV ->
  Los_Angeles; TX/OK/TN/LA/MN/WI/IL -> Chicago; ON and the eastern states -> New_York; Mexico City -> America/Mexico_City;
  Paris/Berlin/London -> their zones). Team body-clock zone = the zone of the team's home arena (fixed dict by team).
  zoneinfo handles DST. Check: home-team local tip hour distribution per team must be ~19:00-20:00 for weeknights.
- Schedule features (previous game date, previous venue city, b2b, road-trip game #) come from the full regular-season
  schedule and use only EARLIER games of that team -- all known before tip. No outcome enters any signal.

## Splits
DEV = 2021-22, 2022-23, 2023-24, 2024-25 (props: 2022-23..2024-25). Halves for game lines BY SEASON: H1 = 2021-22 +
2022-23, H2 = 2023-24 + 2024-25. Props halves: split at the median bet date.
SEALED = 2025-26 (ESPN 2026), scored once, in the same run, only for trials that pass DEV (user authorised sealed tests
10-29). 2025-26 is not virgin for the project (L2, O1, L32, early-season used it), but none of these six signals was
evaluated on it.

## Trials (k = 6, Bonferroni one-sided bar p < 0.05/6 = 0.00833; flat 1u)
**B1 Body clock (game lines, ATS at close).** Body-clock tip = tip time in the team's home zone. A team is
circadian-disadvantaged if its body-clock tip is >= 22:00 (late; e.g. an Eastern team at a 7 PM Pacific tip) or
<= 10:00 (early; e.g. a Pacific team at a 1 PM ET matinee). Bet the OPPONENT ATS at the close price when exactly one
team is disadvantaged. One pooled rule. Reported, not trials: late and early components separately; the same rule at
the ML close.
**B2 Early tips -> under.** Local tip (venue zone) <= 15:30 -> bet UNDER the close total at the close price.
**B3 Day of week (one combined rule).** Games with local date = Friday (any tip), or Saturday/Sunday with local tip
<= 15:30 ("night out before"): bet HOME ATS (against the road team) AND the UNDER, two bets per game, game-clustered.
Components reported, not trials. (Overlaps B2 on weekend matinees; accepted.)
**B4 Nightlife.** Road team (listed away, venue not neutral) whose PREVIOUS regular-season game was in a party metro
(LA = Los Angeles/Inglewood; NY = New York/Brooklyn; Miami; Las Vegas) on ET date d-2 exactly (one off day, so no b2b;
the off-day night is spent there or after a late flight), excluding teams whose own home metro is that metro
(LAL, LAC, NY, BKN, MIA). Bet HOME ATS at close. Travel/b2b control: by construction (one off day, all such teams travel);
descriptive control = home ATS ROI for road teams with exactly one off day whose previous game was NOT in a party metro.
**B5 Arena visitor shooting.** Step 1 (gate, box scores only, ESPN 2019..2025 = 2018-19..2024-25): team-game visitor
3P% residual = visitor 3PM/3PA minus that visitor's season 3P% from its OTHER games (leave-one-out). For each home
team at its main venue_id (most home games in the half), arena effect E = mean residual of visitors at that arena
minus mean residual of that team's opponents when it plays on the ROAD (removes the home team's defence).
Halves = even vs odd ESPN seasons; arenas with >= 60 home games in each half. Pearson r(E_even, E_odd) across arenas,
95% CI by arena bootstrap (10,000). Same for FG%. Step 2 only if a stat's r CI lower bound > 0: for prop season s,
E is recomputed from box seasons < s; arenas in the bottom quintile of E (most visitor-suppressing) -> visitor
player UNDER at OPEN: 3PM props if the 3P% gate passed, points props if the FG% gate passed (pooled if both).
If neither gate passes, B5 is DEAD at step 1 (it still counts in k = 6).
**B6 Altitude visitor props.** Games at DEN or UTAH home venues: visitor players' points AND rebounds UNDER at OPEN,
pooled. Reported, not trials: subgroup visitor on a b2b or road-trip game >= 2; the same unders at all other venues
(baseline, since books shade overs everywhere).

## Metric, bar
ROI = observed mean profit per 1u at the price above. 95% CI = percentile of a clustered bootstrap (10,000, seed 2026):
game lines clustered by GAME (event_id); props clustered by GAME DATE. One-sided p = share of resampled ROIs <= 0.
DEV PASS: p < 0.00833 AND ROI > 0 in both dev halves. SEALED PASS (passed trials only): p < 0.05 / (number taken to
sealed) and ROI > 0. A DEV pass is a bug until the bet rows are inspected (tz/DST, joins, spread sign, lookahead).
Break-even at -110 = 52.38% (so a 1 pt ATS edge ~ 3 pp is needed to clear the vig); props hold ~7% at the open.
Prior: these are mostly public angles; NFL and NBA rest/travel/b2b were all priced. Expected: 0/6 pass.
KILL: no DEV pass -> each factor DEAD in SITUATIONAL_FACTORS (rows listed by the coordinator).

## Result
(filled in below after the run; nothing above edited after scoring)

### Scored 2026-10-29 -- 0/6 pass, all DEAD. SEALED 2025-26 not scored (nothing passed DEV).
One run of `scripts/nba_clock_calendar_arena.py --ledger` (sha 011b833c5d94); output `results/nba_clock_calendar_arena/`
(results.json, bets_dev_B1..B4,B6.csv, b5_arena_split_half.csv); ledger rows ncca_B1..B6 (dataset nba_clock_calendar_arena).
Checks: 0 games without a venue zone; home weeknight local tips inside 17:00-21:00 for 92-100% of each team's games (NY lowest .921);
PHX modal local tip 19:00 in both DST and standard months (DST handled); spread sign agrees with ML favourite .993.
Props: 139,667 opening quotes, 130,502 matched to a regular-season box row; actual == box stat 100%. Unmatched = 8,325
post-season + 811 play-in + 13 regular-season quotes (the games file is regular season only, as pre-registered).
Base rate, all matched prop UNDERS at the open: ROI -3.18%, hit - no-vig p +1.8 pp (books shade overs).

| trial | n | ROI (observed) | 95% CI (clustered) | p one-sided | H1 / H2 | pass |
|---|---|---|---|---|---|---|
| B1 body clock, opponent ATS | 237 | -1.75% | [-14.18, +10.24] | .617 | -2.83% / -0.69% | no |
| B2 local tip <= 15:30 -> under | 316 | -0.78% | [-11.43, +9.54] | .562 | -7.99% / +5.99% | no |
| B3 Fri + wkend matinee, home ATS + under | 2,330 | -6.76% | [-10.68, -2.98] | 1.000 | -8.77% / -4.69% | no |
| B4 nightlife (prev game LA/NY/MIA/LV, 1 off day), home ATS | 291 | -9.30% | [-19.95, +1.72] | .952 | -7.05% / -11.63% | no |
| B5 arena visitor props | -- | gate failed | 3P% r +.103 [-.366, +.457]; FG% r +.288 [-.029, +.528] (30 arenas) | -- | -- | DEAD at step 1 |
| B6 DEN/UTA visitor pts+reb UNDER at open | 2,996 | -1.90% | [-5.46, +1.79] | .851 | +3.53% / -7.47% | no |

Descriptives (not trials):
- B1: every flagged team was the VISITOR (232 late = Eastern visitors at Western tips, body clock >= 22:00; only 5 early).
  Late ATS -2.93% (232); early +52.8% on 5 bets (meaningless n). Same rule at the ML close +0.50% [-11.48, +12.62].
- B3 components: home ATS -7.40% (1,165), under -6.11% (1,165); Friday ATS -6.16% (890), weekend-matinee ATS -11.42% (275).
  If anything the direction is reversed (road teams cover), but the CIs include the ~-4.5% vig.
- B4 by metro: LA -6.39% (113), NY -2.53% (107), Miami -24.71% (69), Las Vegas 2 bets. Control (road teams on 1 off day,
  previous game elsewhere) home ATS -5.13% (2,566) = about the vig. The post-hoc Miami cell is selection, not a lead.
- B6: hit - no-vig +2.4 pp (vs +2.1 pp for visitor unders at all other arenas, ROI -2.65% on 41,235); points -0.87% (1,489),
  rebounds -2.91% (1,507). b2b or road-trip game >= 2: -4.01% (1,810); rested +1.33% (1,186). The halves flip sign.
- B5: the arena effect SD across arenas is ~1.2 pp (3P%) and ~0.7-0.8 pp (FG%) per half, but 3P% does not persist between even and
  odd seasons; FG% is suggestive (r +.29) but its CI includes 0, so the pre-registered prop step was not run.

## Why it fails (the lesson)
Clock, calendar and nightlife are public schedule facts known weeks ahead; the close prices them, so every rule pays
roughly the vig. The NBA body-clock exposure is almost entirely Eastern visitors at 7 PM Pacific tips (232/237), routine
travel that is in every book's model. Arena "sightlines" are not a stable trait for 3P% (split-half r .10). Altitude
visitor unders look like the generic over-shading (+2.4 pp vs +2.1 pp elsewhere), not an altitude effect.
Not critic-reviewed (nulls).
