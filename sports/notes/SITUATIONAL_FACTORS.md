# SITUATIONAL FACTORS -- every data/situational factor we tested or that bettors commonly claim (2026-10-29)

**Purpose.** One row per factor ("the kicker guy" style claims: stadium, weather at the stadium, back-to-backs, refs...),
with what we found **against the market price**, not against 50%. Check here before proposing a factor: if it is
DEAD, do not redo it without a new mechanism (protocol.md); if it is UNTESTED, the row says what test it needs.
Every number is copied from the file in the "file" column. "not recorded" = no number exists in the repo.

**How to read a row.**
- *tested?* YES = a pre-registered trial or a counted test exists. **NO = UNTESTED -- candidate research question.**
  LIT = only outside literature (notes/research_situational_vars.md, research_prop_models.md); we never ran it.
- *result vs MARKET*: effect, n, p, and whether the price already contains it. ROI is at the price the trial used
  (usually -110 or the closing line). Slopes are "points beyond the closing line per SD of the factor".
- *usable how*: **model** (keep as a feature inside a game model; adds information but is not a bet alone),
  **filter** (a guard/skip rule), **lead** (alive, below the bar), **DEAD**, **candidate** (untested).
- Ledger = `lab/ledger.jsonl` (dataset:strategy). Scan = `lab/reports/nfl_scan.csv` / `mlb_scan.csv` (one-pass
  feature scans, BH q-values; screening only, not trials).

**Counts.** 117 factor rows: **96 tested by us** (YES), **21 not tested by us** (6 LIT-only + 15 UNTESTED -- candidate).
Of the 96 tested: no situational factor has passed a sealed test as a sports bet (the passes are weather-as-market
and soft-vs-sharp soccer). Leads still alive: WCN, DUL, paint-D, 3P% totals, QB-change opener CLV, NBA altitude,
RYU game-script props. Kept as model features: NHL b2b, goalie GSAA, absences, pace/style, offence EPA.

**The lesson.** Nearly every public situational factor is **priced at the close**: rest, back-to-backs, travel,
altitude, refs, coaches, recency, luck, home/away and big-favourite angles all lose about the vig
(nba_angles, nfl_spread, nfl_recency, nba_comebacks). The market knows these factors; a real effect on the *score*
(e.g. NHL b2b -0.25 goals, crews move FTA p .002) is not an effect on the *price*. Where a factor can still pay:
1. **Openers** (before sharp money): QB changes, absences, model gaps -> CLV vs opener is real (L32/L33/L34/L36),
   but it must clear the opener's overround (O1 sealed FAIL).
2. **Thin markets / props**: books shade overs; info that lands after props open (refs at ~9am ET, injury news).
3. **Forecast-vs-price on thin venues** (weather is the market itself: E1/E2/E5 are our only passes).
4. **Observed-after-kickoff weather** is unpriced but not bettable pre-game (oracle +21.4% vs causal +8.0%).

---------------------------------------------------------------------------------------------------------------------
## A. Venue

| factor | sport | tested? | result vs MARKET | file | usable how |
|---|---|---|---|---|---|
| Home field / court (overall) | NBA, NFL | YES | NBA home ATS -0.03 pts p .80; NFL home ML control -3.25%/2,922 (= the vig) | nba_comebacks_matchups.md Q5; ledger nfl_moneyline:control_home_ml | DEAD (priced) |
| Home edge shrank post-2020 (2.74 -> 1.75 pts) | NBA | LIT | STATISTICAL; whether books adjusted slowly in 2020-23 is untested | research_situational_vars.md s2 | model: fit home edge per season |
| Home dog (ML / ATS) | NBA, NFL | YES | NBA home dog ML -3.87%/4,766 p .97; NFL scan home_dog ATS -0.74%/1,264 p .43 | ledger nba_sbr:nba_home_dog_ml; nfl_scan.csv | DEAD |
| Late-season home dog (wk >= 15) | NFL | YES | -7.8%/207 p .91 | ledger nfl_spread:late_home_dog | DEAD |
| Altitude home (Denver / Utah) | NBA | YES | +0.98 pts ATS [-0.02, 1.99] p .055, 6/10 seasons | nba_comebacks_matchups.md Q5; FUND_BOOK 3c ALT | lead (tier 3) |
| Altitude home (Denver) | NFL | YES | -7.37%/94 p .76 | ledger nfl_spread:den_altitude_home | DEAD |
| Visitor props / 2nd-half splits at altitude | NBA | NO | **UNTESTED -- candidate**: visitor FT% 74.9 vs 75.8% (LIT); needs prop lines + pbp by half | research_situational_vars.md #11 | candidate |
| Dome team playing outdoors in cold | NFL | YES | +3.06%/33 p .43 | ledger nfl_spread:dome_team_cold | DEAD (n tiny) |
| Roof / dome vs outdoor (scoring level) | NFL | NO | **UNTESTED as its own factor** (used only as a filter for wind/rain trials: `roof == outdoors`) | scripts/nfl_scan.py | filter |
| Playing surface (turf vs grass) | NFL | NO | **UNTESTED -- candidate**; not recorded anywhere; needs nflverse `surface` + totals/props | -- | candidate (low prior) |
| Park factors / wind at a specific park (Wrigley) | MLB | YES | Wrigley wind -5.27%/63 p .67 | ledger mlb_total:park_wind_wrigley | DEAD |
| Arena sightlines / rims | NBA | LIT | "No evidence"; proposed test = arena fixed effects on visitor 3P%/FT% | research_situational_vars.md s2 | candidate (low prior) |
| Crowd / attendance | NBA | LIT | bubble: FTA rate +2.6%, D rating ~5.5/100 worse; normal seasons barely vary | research_situational_vars.md s2 | low value |

## B. Weather

| factor | sport | tested? | result vs MARKET | file | usable how |
|---|---|---|---|---|---|
| Wind >= 15 mph at outdoor kickoff -> under | NFL | YES | +7.34%/256 p .11 (wind field is observed at kickoff) | ledger nfl_total:wind_unders; nfl_scan wind15 slope -2.36 pts p .0019 q .068 | lead-below-bar / priced mostly |
| Rain 3 h pre-kickoff (causal) -> under | NFL | YES | corrected +8.0%/219 p .11, 11/14 seasons (buggy first run +12.4%) | ledger nfl_total:pre_rain_unders (2nd row); results.md 10-16 | F2 forward paper (live_signals.py); gate DEAD |
| Rain + wind pre-kickoff -> under | NFL | YES | corrected -10.2%/67 p .80 (buggy +16.0%) | ledger nfl_total:pre_rain_wind_unders | DEAD |
| Observed in-game rain+wind ("oracle") | NFL | YES | +21.4%/70 p .033; rain -1.9 pts beyond close (se 0.86), wind -0.24 pts/mph (se 0.07) | results.md 2026-10-16 bug note | not causal -> not bettable pre-game; forecast-error is the angle |
| Forecast vs observed (forecast error at bet time) | NFL | NO | **UNTESTED -- candidate**: needs archived kickoff *forecasts* (not reanalysis) vs the close | research_situational_vars.md s5 | candidate (#3 below) |
| Cold (<= 20F) -> over (market over-adjusts) | NFL | YES/LIT | nfl_scan outdoor_cold: totals +4.8%/248 p .21, spread +4.8%/248 p .13; LIT 58% overs since 2009 | nfl_scan.csv; research_situational_vars.md s5 | DEAD at our n |
| Snow | NFL | NO | **UNTESTED separately** (not recorded; ERA5 precip in pre_rain trials does not split snow) | -- | candidate (rare games) |
| Humidity | NFL, MLB | LIT | Paul: humidity strategies "reject efficiency" (old data); never run here | research_situational_vars.md s5 | candidate (low prior) |
| Weather GLM on totals (temp + wind) | MLB | YES | -5.43%/371 p .85 | ledger mlb_total:mlb_weather_glm | DEAD |
| Temperature tails (nonlinear) | MLB | YES | -12.0%/34 p .75 | ledger mlb_total:temp_tail_nonlinear | DEAD |
| Precip / cloud cover | MLB | YES | +3.18%/208 p .32 | ledger mlb_total:condition_precip_cloud | DEAD |
| Day vs night x temperature | MLB | YES | -3.5%/326 p .75 | ledger mlb_total:day_night_temp_interaction | DEAD |
| Weather as the market (forecast consensus vs bracket price) | Kalshi / ForecastEx temps | YES | E1 sealed +1.45%/1,844; E2 +0.78%/2,464; E5 +6.9%/219 | lab/lockbox.jsonl | **LIVE** (the only passes) |
| Daily rain all-models-dry -> NO | Kalshi rain | YES | dev +3.42%/361; sealed FAIL +0.49%/274 p .28 | lockbox kalshi_rain_R1_dry_no | DEAD |
| Travel disruption (flight delays) | NBA | NO | **UNTESTED -- candidate**; needs BTS on-time data; "speculative" | research_situational_vars.md s5 | candidate (low prior) |

## C. Schedule, rest and travel

| factor | sport | tested? | result vs MARKET | file | usable how |
|---|---|---|---|---|---|
| Rest-days advantage ATS | NBA | YES | -1.46%/4,361 p .85; slope +0.14 pts/SD p .26 | ledger nba_sbr:nba_rest_edge_ats; nba_comebacks_matchups.md Q3 | DEAD (priced); model term |
| Rest-days difference | NFL | YES | nfl_scan diff_rest ATS +0.27%/1,260 p .48 | nfl_scan.csv | DEAD |
| Back-to-back -> unders | NBA | YES | -3.73%/5,469 p .998 | ledger nba_sbr:nba_b2b_unders | DEAD |
| Back-to-back home / away ATS | NBA | YES | home b2b -0.33 pts p .37; away b2b +0.58 [-0.01, 1.17] p .054 | nba_comebacks_matchups.md Q3 | DEAD; in M0 model |
| Back-to-back (goals) | NHL | YES | real on the score: team -0.25 goals (.09), opponent +0.23 (.09); priced at close (model CLV only) | nhl_simple_model.md | **model** feature |
| 3-in-4 nights | NBA | YES | -0.54 pts [-0.97, -0.12] p .011, fails BH | nba_comebacks_matchups.md Q3 | DEAD |
| 4-in-6 / road-trip game number | NBA | YES (trip #) | trip game # +0.11/SD p .39; 4-in-6 not tested | nba_comebacks_matchups.md Q3 | DEAD |
| Travel distance (km) | NBA | YES | -0.01/SD p .97 | nba_comebacks_matchups.md Q3 | DEAD; in M0 model |
| West-coast team at night vs East (body clock) | NFL | YES | +21.5%/107 p .0068; gate LEAD q .053; mirror test lost | ledger nfl_spread:west_coast_night; FUND_BOOK WCN | lead (lockbox pending, not recommended) |
| West-coast team early ET kickoff (mirror) | NFL | YES | -10.6%/110 p .86 | ledger nfl_spread:west_early_east | DEAD (weakens WCN) |
| Westward / eastward jet lag (time zones) | NBA | LIT | papers disagree on direction; Leota rebounds -1.29 p<.0001 (home, eastward) | research_situational_vars.md s1 | candidate for props (#5 below) |
| Early tip (before 5pm ET) -> under | NBA | NO | **UNTESTED -- candidate**: tout stat 53.4% unders, no method | research_situational_vars.md #7 | candidate |
| Bye-week team | NFL | YES | bye_fade -2.37%/326 p .67; scan diff_off_bye p .44 | ledger nfl_spread:bye_fade; nfl_scan.csv | DEAD |
| Short week / Thursday night | NFL | YES | TNF unders +0.76%/156 p .47; scan sum_short_week -1.4%/180 | ledger nfl_total:tnf_unders | DEAD |
| Starter rest (days) | MLB | YES | scan sum_sp_rest p .55, diff_sp_rest p .66; 0/21 BH | mlb_scan.csv; results.md 10-09 | DEAD |
| Starter short rest -> K under | MLB | YES | n 2 (untestable), -15.5% | ledger mlb_k_props:k_short_rest_under | DEAD (no sample) |
| Starter workload (prev pitches) | MLB | YES | K workload under -4.1%/486 p .82; scan diff_sp_prev_pitches p .086 | ledger mlb_k_props:k_workload_under; mlb_scan.csv | DEAD |
| Bullpen fatigue (b2b arms, pitches 1-3 d) | MLB | YES | best sum_bp_b2b_tired totals p .067, rule -2.96%/9,784; 0/21 BH | mlb_scan.csv | DEAD |
| Day game after night game | MLB | YES | sum_day_after_night totals slope -0.11 p .052, rule -1.28%/2,065, q .47 | mlb_scan.csv | DEAD |
| Early season (small-sample stats) | NBA, NHL | YES | NBA luck early quintile +3.66%/572 p .18; NHL <= 15 games all within +-0.4%, p >= .44 | ledger nba_luck:T4-1b; nhl_luck.md | DEAD |
| Early-season openers posted weeks ahead | NFL | YES | CLV holds with week 1/playoffs removed (B spread +0.75) | nfl_simple_model.md | part of L33 opener edge |
| Late season / rest games (seeding locked) | NBA | YES | dev April abs(model - close) 3.94 vs ~2.75 other months; 9/10 biggest gaps = rested stars | MODELS_GUIDE.md (rest-game guard) | **filter** (skip/flag) |
| Favourites in March/April | NBA | YES | -0.23 pts p .33 | nba_comebacks_matchups.md Q5 | DEAD |
| Divisional game -> under (all season) | NFL | YES | +5.98%/1,055 p .036; scan slope -1.65 p .0003 q .022 | ledger nfl_total:div_unders; nfl_scan.csv | below dataset bar |
| Divisional under, week >= 10 (post hoc) | NFL | YES | +9.81%/623 p .0098; gate LEAD q .070 | ledger nfl_total:div_unders_late; FUND_BOOK DUL | lead (lockbox pending) |
| Week of season (trend) | NFL | YES | scan week totals p .66, spread p .89 | nfl_scan.csv | DEAD |
| Playoff games | NFL, NBA | YES | NFL scan playoff p .63/.70; NBA playoff zig-zag +0.89 p .105 (n 639) | nfl_scan.csv; nba_comebacks_matchups.md Q5 | DEAD |
| Series extension (trailing team G4-6) | NBA | YES | won 42.4% vs market 43.5%, n 125, p .85 | nba_referees.md R3 | DEAD |
| Primetime -> under (public overs) | NFL | YES | +1.43%/542 p .37 | ledger nfl_total:primetime_unders | DEAD |
| Nightlife / "party city" the night before | NBA | NO (LIT) | **UNTESTED -- candidate**: preprint ATS -0.557 (SE .149) on LA/NY within 24 h, 2010-17, predicts the OVER; Harden strip-club claim = anecdote | research_situational_vars.md s2; FUND_BOOK L7 | candidate (#6 below) |

## D. Personnel and coaching

| factor | sport | tested? | result vs MARKET | file | usable how |
|---|---|---|---|---|---|
| QB change week (vs OPENER) | NFL | YES | model B CLV +1.15 pts/210 [+0.77, +1.55]; opener excess error +0.08 [-0.17, +0.35] (not significant at the close) | nfl_simple_model.md; FUND_BOOK QBC | **model** + lead (tier 3) |
| QB change / backup QB (vs CLOSE) | NFL | YES | scan diff_qb_change totals slope -1.22 p .019, rule +0.49%/676; backup QB p .05, rule -0.38% | nfl_scan.csv | DEAD at close |
| Absences / injury news (vs OPENER) | NBA | YES | ACTUAL availability CLV +1.75 pts in news games (564 bets) vs +0.73 no-lookahead -> upper bound | nba_simple_model.md finding 4 | **model** (`avail` variant); needs news faster than the book |
| Whole-lineup rests (3+ starters) | NBA | YES | linear absence term under-sized: OKC 2025-03-07 still 11.2 pts off | MODELS_GUIDE.md | filter / model fix |
| Key non-QB players (WR1 / LT / edge) | NFL | NO | **UNTESTED -- candidate**; injury + snap data not local | nfl_simple_model.md caveats | candidate |
| Starting goalie (GSAA) | NHL | YES | real: -0.48 to -0.75 goals per goal/30 shots GSAA; goalie-change games not a clean late-news test | nhl_simple_model.md | **model** feature |
| Minutes volatility / restriction / injury return | NBA props | NO | **UNTESTED -- candidate**; "likely the largest real driver of prop errors" (LIT, no line test) | research_situational_vars.md #3 | candidate (#2 below) |
| Teammate return / usage shift | NBA props | NO | **UNTESTED -- candidate**; props "slowly rising" after news (anecdote) | research_prop_models.md | candidate (#1 below) |
| Healthy roster + preseason ratings (single games) | NBA | YES (1 game) | DEN@OKC 10-22: no bet by pre-committed rule | FUND_BOOK L2 | lead (log ~200 games) |
| New head coach | NFL | YES | scan sum_new_coach totals p .071, diff_new_coach ATS -1.97%/1,281 p .35 | nfl_scan.csv | DEAD |
| Coach ATS persistence | NBA, NFL | YES | NBA +0.15 pts/SD p .47; NFL odd/even r +.21 (46 coaches) p .17 | nba_comebacks_matchups.md Q3; nfl_comebacks_matchups.md | DEAD |
| Coach late-game / 4th-down aggressiveness | NBA, NFL | YES | NBA close-game residual -0.67 pp/SD p .61; NFL 4th-down trait real (r .07) but nothing vs ATS/totals | same | DEAD |
| Timeouts after a run | NBA | YES | fair design -0.03 pts (SE 0.11), n 8,836 runs (naive -0.92 was selection) | nba_comebacks_matchups.md Q3 | DEAD |
| Contract year | NBA | LIT | mixed theses, sign unstable | research_situational_vars.md s4 | DEAD-ish (absorbed by form) |
| Revenge / homecoming game | NBA | NO (LIT) | **UNTESTED -- candidate**: HSAC "minimal"; contrarian under idea | research_situational_vars.md #10 | candidate (low prior) |

## E. Officiating

| factor | sport | tested? | result vs MARKET | file | usable how |
|---|---|---|---|---|---|
| Referee crew -> total vs close | NBA | YES | excess R^2 .0054, crew SD ~1.3 pts, p .017 (bar .00714); bet-by-crew -1.9%/2,222 p .83 | nba_referees.md R1, B1; ledger nba_referees | DEAD (priced) |
| Crew -> home result | NBA | YES | p .27 (R2); crew home-ATS rating +0.05/SD p .82 | nba_referees.md; nba_comebacks_matchups.md Q4 | DEAD |
| Crew whistle x team style (FT drawers, foul-prone, stars, favourite) | NBA | YES | 4 cells all p >= .27, halves flip 4/6 | nba_comebacks_matchups.md Q4 | DEAD |
| Crew FT tendency -> player points props (crew known ~9am, after open) | NBA props | YES | overs at open -8.1%/2,174 p 1.0; unders +1.89%/347 p .37; total slope p .099 | ledger nba_ref_ft_props; notes/nba_ref_ft_props.md | DEAD (lines move toward the crew) |
| L2M errors favour team trailing | NBA | YES | 58.2% of 1,979 errors, p < .0001 (bias real) | nba_referees.md R4c | no betting use |
| L2M errors favour home / favourite | NBA | YES | 50.6% p .33; 49.4% p .37 | nba_referees.md R4a/b | DEAD |
| Head referee flags / DPI / holding -> totals | NFL | YES | -0.36 to -0.40 pts/SD p .08-.11 (opposite sign to "more flags, more points"); scan ref_total_res p .37 | nfl_comebacks_matchups.md; nfl_scan.csv | DEAD |
| Head referee home bias | NFL | YES | home ATS +0.03 p .91; scan ref_home_ats -2.26%/1,354 p .10 | same | DEAD |
| Home-plate umpire -> totals | MLB | YES | -1.2%/1,867 p .70 | ledger mlb_total:umpire_total_tendency | DEAD |
| Fixing screen (Donaghy-style line moves by referee) | NBA | YES | max referee z 2.70 vs null p95 3.70, perm p .38 | nba_referees.md R5 | nothing found |

## F. Style, matchup and luck

| factor | sport | tested? | result vs MARKET | file | usable how |
|---|---|---|---|---|---|
| Team quality (net rating / EPA margin) | NBA, NFL | YES | NBA -0.03 pts ATS/SD (n 10,403); NFL +0.3 pts/SD p .25 | nba_comebacks_matchups.md; nfl_comebacks_matchups.md | priced |
| Offence EPA under-weighted | NFL | YES | slope +0.98 pts/SD p .0003 vs open; bet rule -1.6%/994 at -110 | nfl_leads_open.md L09/L07 | **model** (already in model B) |
| Bad D x bad D -> under | NFL | YES | vs opener CLV -0.16, -7.7%/214 | nfl_leads_open.md BD | DEAD |
| Paint points allowed share | NBA | YES | -0.54 pts ATS/SD [-0.79, -0.29] p 2.5e-5, 10/10 seasons; rule +5.6%/179 | nba_comebacks_matchups.md Q2; FUND_BOOK L30 | **model** feature (L2) |
| Season 3P% (own + allowed) -> under | NBA | YES | -0.68 / -0.59 pts/SD; TOT_3PT walk-forward 53.6%, +2.3%/1,417 (post hoc) | nba_comebacks_matchups.md Q2; FUND_BOOK L31 | lead (forward paper) |
| 3P% shooting luck (per game) | NBA | YES | slope -0.31 p .002; quintile bet -1.1%/4,719; absorbed by 3P% level terms | nba_luck.md; ledger nba_luck | DEAD (inside model) |
| Abs(spread) size -> over (blowout scoring) | NBA | YES | +0.58 pts/SD p .0012 | nba_comebacks_matchups.md Q2 | part of L31 model |
| Blowout risk -> star prop unders | NBA props | NO | **UNTESTED -- candidate** (practitioner consensus only) | research_situational_vars.md #2 | candidate (#2 below) |
| Pace (pregame) | NBA, NFL | YES | NBA U1 +0.77 (.52) already in totals; NFL pace +0.6 p .11 modern fit | nba_simple_model.md; nfl_simple_model.md | **model** feature |
| In-game pace | NBA live totals | YES | adds ~nothing; Kalshi Brier .1196 < model .1240; rule -2.1%/336 | nba_live_totals.md | DEAD |
| Pass rate over expected (PROE) | NFL | YES | totals dev +1.65 p 1e-9 faded to +0.5 p .19 modern; nothing for comebacks | nfl_simple_model.md; nfl_comebacks_matchups.md | weak model term |
| Style matchups (3PA, FT rate, ORB, TOV, transition) | NBA | YES | best 3PA x 3PA allowed +0.23 p .07; none survive | nba_comebacks_matchups.md | DEAD |
| Schemes (pressure, blitz, man/zone) | NFL | YES | 14 interactions + 9-cell grid, none survive; zone/blitz -> FGs r = 0.00 | nfl_comebacks_matchups.md; results.md 10-12 | DEAD |
| PDO / goalie luck | NHL | YES | 0/17 trials; total-luck slope -0.08 vs -1 if blind (~90% priced) | nhl_luck.md | DEAD |
| Turnover margin / one-score luck | NFL | YES | flat net of EPA | nfl_comebacks_matchups.md | DEAD |
| Game script by category (props) | NFL props | YES | 0/8 cells; C7 low-total rush-yd unders +11.1%/155 p .039 (FUND_BOOK quotes +7.7pp); C3 +12.1%/99 p .078 | ledger nfl_props_script | RYU lead (tier 3) |
| Comeback traits (3PA, bench, star usage...) | NBA | YES | 0/69 cells survive BH | nba_comebacks_matchups.md Q1 | DEAD |
| Comeback = deficit x time x spread (state) | NBA, NFL | YES | Kalshi calibrated 7/7 bins; ESPN WP too low on trailers | late_game_calibration.md; nba_comebacks_matchups.md | lookup only |
| Kicker system (opp kicker pts allowed, RZ trips, FG%) | NFL props | YES | 84.7%/354 vs base 83.0%; residual vs team total -0.7pp p .65 | nfl_kicker_system.md | DEAD (base rate) |

## G. Psychology, public money and the line itself

| factor | sport | tested? | result vs MARKET | file | usable how |
|---|---|---|---|---|---|
| Recency: fade last week's blowout / upset / cover | NFL | YES | blowout fade 51.7% at close, upset fade 50.2%, luck fade 48.7%; 11/12 trials fail | nfl_recency.md | DEAD |
| Fade hot form (10-game margin gap >= 14) | NFL | YES | +4.73%/538 p .145 | ledger nfl_spread:fade_hot_form | DEAD |
| Last-game ATS overreaction / bounce-back | NBA | YES | -0.23/SD p .067; bounce-back after 20+ loss +0.21 p .51 | nba_comebacks_matchups.md Q5 | DEAD |
| Rolling player form (props) | NBA props | YES | -3.29%/130,000; overs -4..-7% every market | results.md 10-21 (P1) | DEAD (books shade overs) |
| Big favourite fade / heavy fav ML / big-spread dog | NFL, NBA | YES | NFL big fav fade -2.67%/383; NFL heavy fav ML -1.1%/833; NBA heavy fav ML -2.7%/4,239; NBA big dog +0.56%/1,218 | ledger nfl_spread, nfl_moneyline, nba_sbr | DEAD |
| Favourite-longshot bias (prediction markets) | Kalshi sports | YES | fav >= 85% 1 h pre-start +2.0%/383 p .062; dogs 20-50c +1.4%/4,429 | FUND_BOOK G1/G4 | lead, not recommended |
| Public betting % / reverse line moves / steam | all | NO | **UNTESTED -- candidate**: ESPN feeds carry no public % (flags not computed) | tonight_nba_2026-10-26.md | candidate (needs a paid splits feed) |
| Line move: follow / fade open->close | MLB | YES | follow -3.0%/3,308; fade -2.3%/3,308; momentum totals -1.26%/253 | ledger mlb_ml_sbr, mlb_total:line_move_momentum | DEAD |
| Line drift toward last week's EPA | NFL | YES | +0.275 pts/SD p < 1e-10 (descriptive); close ends about right | nfl_recency.md T2b | not a bet (sharp money) |
| Totals move toward luck after open | NHL | YES | moves the "wrong" way, z ~ -11; close still not beatable | nhl_luck.md T4 | DEAD |
| Stale opener (model gap vs first-posted line) | NBA, NFL, NHL, CFB | YES | NBA L32 sealed +6.4% p .054 (FAIL, ~25% power), CLV +1.2; NFL CLV +0.76; CFB no-vig +7.4% p .0004; O1 sealed FAIL | FUND_BOOK 3b/3c; lockbox | **the main live sports lead** (paper) |
| Kalshi first listing stale | NFL | YES | CLV +3.3c p < .0002 (109 games); taker first ask -4.5% | FUND_BOOK L35 | folded into Q004 forward |
| Soft books lag the sharp book | soccer | YES | E3 sealed CLV +3.63%, ROI +8.71%/2,004 | lockbox soccer_best3_pow3 | PASSED (not tradeable from CA) |
| Motivation / lookahead / letdown / sandwich spots | NFL, NBA, CFB | NO | **UNTESTED -- candidate**; never defined in the repo; tanking only via who sits | MODELS_GUIDE.md s(not modelled) | candidate (low prior, define first) |
| News/rumor spikes | Kalshi | YES | 4/4 rules lose -7..-13%; price moves before news (median 4 h) | FUND_BOOK 3d (rumor_signals.md) | DEAD |
| Big plays in-game (overshoot) | NFL Kalshi | YES | reprices ~87% within 1 min, reverts 1.8 pts over 7 min | results.md 10-20; FUND_BOOK F1 | F1 maker fade forward paper |

---------------------------------------------------------------------------------------------------------------------
## Best untested candidates (ranked; each needs a /prereg, dev-only, scored vs the market)

1. **Injury/teammate news vs stale props** (NBA): event study of prop moves after the timestamped NBA injury report
   (5 pm day-before / game-day). Mechanism = information timing in thin markets; T8 upper bound +1.75 CLV says absences
   are what openers miss. Needs Odds API snapshots (paid) + report PDFs.
2. **Minutes-driven prop unders**: blowout risk (|spread| >= 10-12 x 4th-quarter minutes share) and minutes volatility /
   returning-from-injury. Books build props off projected minutes; test whether they scale enough. ESPN graded props +
   nba_api pbp (free).
3. **NFL forecast error at bet time**: archived kickoff forecasts (e.g. Open-Meteo historical forecast API) vs the
   close; the oracle says observed weather is unpriced (+21.4%/70) but pre-kickoff rain only reaches +8.0%. F2 logs it
   forward already.
4. **Altitude visitor props / 2nd-half splits** (DEN/UTA): team ATS lead is +0.98 p .055; props never tested.
5. **Jet lag x rebounds** (NBA props): strongest literature coefficient (Leota -1.29 reb); R_REB failed sealed, so
   it must be a NEW mechanism-specific prereg, not a re-mine.
6. **Nightlife / LA-NY within 24 h -> game OVER** (pre-registered exposure, travel-direction controls). Expect null;
   cheap with free data.
7. **Early tip times -> under** (NBA): one pass to log the null.
8. **Public betting % / reverse line move**: only if a splits feed is bought (PM's call on spend).

Deprioritised (low prior, from research_prop_models.md): blind b2b/bye fades (decayed), post-COVID home-court angles,
NBA totals bands, surface, humidity, revenge, contract year.
