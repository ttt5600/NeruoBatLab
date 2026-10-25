# NBA comebacks, matchups, coaching, refs: what predicts winning beyond the line? (2026-10-25)

User ask (after DEN came back from 23 down, 53-30 at Q2 7:00, to beat NOP 141-135 in OT): "find team passing rates,
offense, defense, all the metrics that correlate to which one wins, what types of comebacks are possible, what
teams' schemes play well against what other team defenses -- coaching, reffing, all factors."

Script: `scripts/nba_comebacks.py` (`pull`, `timeouts`, `build`, `q1`..`q5`, `rules`, `all`).
Results: `results/nba_comebacks.json`. Data (gitignored): `data/nba_comebacks/` (stats.nba.com team + player game logs,
head coaches, ESPN timeouts, per-minute game states).
Seasons: **2015-16 .. 2024-25 only.** 2025-26 is sealed (lab MANIFEST, lead L28), so `data/nba_ingame.parquet`,
ESPN box_2026/2027 and any 2025-26 stats were not read.

## Verdict (plain language)

1. **"Good teams win" is fully priced.** A 1-SD better season-to-date net rating is worth +5.7 points of actual margin.
   Against the closing spread it is worth **-0.03 points** (CI -0.28..+0.21, n 10,403). The line already contains
   everything in a box score.
2. **Comebacks depend on the deficit, the time left and pregame strength. That is it.** We tested 69 cells of team
   traits: the trailing team's 3PA rate, pace, offense, bench, star usage, ball movement; the leader's defense,
   turnovers, FT shooting, bench drop-off, and habit of pulling starters. We tested each at halftime, end of Q3 and
   Q4 6:00. **None survives** multiple-testing correction (0/69 BH). The "high-3PA teams come back more" idea is
   not supported: +0.5 pp per SD at halftime, CI -0.5..+1.6, and zero by the end of Q3.
3. **The DEN comeback was not a miracle, and ESPN's number would have misled you.** When a team favoured by about
   4+ points pregame trails by 21-25 between Q2 9:00 and 5:00, it won **20 of 96 (21%)**. Our model says 21.5% and
   ESPN's WP said **8.2%**. ESPN under-rates trailing teams everywhere: at halftime when down 10+, 13.2% won vs ESPN
   9.2%; at the end of Q3, 4.9% vs 3.3%; at Q4 6:00, 2.0% vs 1.2%. Kalshi was already shown to be calibrated late
   (late_game_calibration.md), so this is a warning about the ESPN app, not a Kalshi edge.
4. **Matchups / schemes: 61 cells, 5 survive BH and both halves. 4 survive BH over all 152 tests.** Two are
   real-looking and small:
   - **Paint defense (ATS).** A team whose defense gives up a big share of opponents' points in the paint covers
     **0.54 pts less per SD** (CI 0.29-0.79, p 2.5e-5, 10/10 seasons same sign). It holds against a multi-book
     line in 2021-24 (-0.73, p .0005) and after controlling for 3P% luck and 3PA allowed (-0.74). The market slightly
     overrates teams that leak the rim/paint.
   - **3-point regression (TOTALS, not sides).** When both teams have shot well from 3 so far, or allowed opponents
     to shoot well, the game goes **under** the closing total (-0.68 and -0.59 pts per SD). Bigger spreads go
     **over** (+0.58 per SD). This is the textbook "3P% is mostly luck and the market regresses it too little".
   Neither is big enough to beat -110 alone. In walk-forward, the combined totals model at |pred| >= 1.5 went
   **1,417 bets 53.6% [51.0, 56.2], ROI +2.3%** (break-even 52.4%). The paint factor only reaches a 1-point
   prediction on ~25 games a season.
5. **Coaching: nothing beyond the line.** Coach ATS does not persist (prior seasons -> this season +0.15 pts/SD,
   p .47; split-half r within a coach-season 0.03). Coaches show no "closes games well" skill vs the baseline
   (p .61). Rest, back-to-backs, 3-in-4 and travel are all priced (best: 3-in-4 -0.54 pts, p .011, not surviving).
   **Timeouts after a run do nothing.** A naive comparison says a timeout after an 8-0 run is followed by 0.92 pts
   *worse* play. That was selection: "no timeout" included teams that scored right away. With a fair design (same
   30-s window, score unchanged) the effect is **-0.03 pts (SE 0.11)**.
6. **Refs: nothing on sides.** Six crew tests vs ATS (walk-forward crew ratings, whistle-heavy crew x FT-drawing /
   foul-prone / star-heavy teams, crew x favourite), 2021-22..2024-25: all p >= .27.
7. **Other: nothing survives.** Altitude home (DEN/UTA) +0.98 pts ATS (p .055, 6/10 seasons). Overreaction to the last
   game's ATS result -0.23/SD (p .067). Playoff zig-zag +0.89 (p .105, n 639). All are under-powered leads at best.

**Bottom line for betting:** sides are efficient against every team trait, matchup, coach, rest and ref factor we
could build from public data. The best leftovers are two small regressions: 3P% luck in **totals** and paint defense
in **sides**. They are worth logging forward, not betting yet. For live comeback bets, compare Kalshi to the lookup
below, never to ESPN's WP.

## GAME-DAY: comeback probability lookup (compare to a live Kalshi price)

P(trailing team wins). Rows = game clock and deficit. Columns = the **trailer's pregame expected margin** (closing
spread from the trailer's side; +8 = trailer was an 8-point favourite, -8 = 8-point underdog). The number is the
model (logistic on deficit/sqrt(time left), spread*sqrt(time left), quadratic terms, fit on 12,803 games
2015-16..2024-25). The brackets give the empirical rate / games in a band (deficit ±2, clock ±2 min, spread ±3; shown
when n >= 30). The band is biased upward where the curve bends sharply (late Q4). For Q4 inside 6:00 with a close
score, use the possession-aware table in notes/late_game_calibration.md.

| clock | down | trailer dog by 8 | dog by 4 | pick | fav by 4 | fav by 8 |
|---|---|---|---|---|---|---|
| Q2 9:00 | 15 | .082 (.09/1069) | .129 (.14/1222) | .197 (.22/782) | .291 (.31/770) | .406 (.41/425) |
| Q2 9:00 | 20 | .047 (.06/447) | .078 (.10/455) | .125 (.12/286) | .195 (.16/244) | .293 (.28/122) |
| Q2 9:00 | 25 | .026 (.02/131) | .044 (.06/125) | .073 (.10/70) | .121 (.13/61) | .193 |
| Q2 6:00 | 15 | .076 | .118 | .180 | .264 | .371 (.39/468) |
| Q2 6:00 | 20 | .042 (.05/552) | .068 (.08/578) | .108 (.12/362) | .169 (.19/307) | .254 (.27/157) |
| Q2 6:00 | 25 | .022 (.02/218) | .036 (.05/201) | .060 (.07/122) | .099 (.09/96) | .158 (.15/41) |
| Q2 6:00 | 30 | .011 (.03/60) | .018 | .031 | .053 | .090 |
| halftime | 10 | .117 | .170 | .241 | .329 | .433 |
| halftime | 15 | .063 (.07/1378) | .095 (.12/1528) | .143 (.18/1002) | .209 (.22/969) | .295 (.29/494) |
| halftime | 20 | .031 (.03/750) | .049 (.06/774) | .076 (.10/471) | .118 (.15/433) | .177 (.20/189) |
| halftime | 25 | .014 (.01/351) | .023 (.02/337) | .037 (.03/184) | .060 (.07/138) | .095 (.10/61) |
| Q3 6:00 | 15 | .047 (.06/1477) | .070 (.09/1636) | .103 (.13/1037) | .148 (.16/1015) | .210 (.21/487) |
| Q3 6:00 | 20 | .020 (.02/929) | .030 (.04/954) | .046 (.06/559) | .070 (.09/523) | .104 (.12/240) |
| Q3 6:00 | 25 | .007 (.00/519) | .012 (.02/484) | .018 (.04/268) | .029 (.06/213) | .045 (.08/93) |
| end Q3 | 10 | .080 | .110 | .149 | .199 | .261 |
| end Q3 | 15 | .029 (.04/1470) | .042 (.05/1656) | .059 (.07/1024) | .084 (.09/977) | .117 (.13/505) |
| end Q3 | 20 | .009 (.01/1061) | .013 (.02/1074) | .020 (.02/626) | .029 (.03/528) | .042 (.06/223) |
| Q4 9:00 | 10 | .064 | .086 | .115 | .152 | .198 |
| Q4 9:00 | 15 | .019 | .026 | .036 | .051 | .070 |
| Q4 6:00 | 10 | .042 | .055 | .072 | .094 | .122 |
| Q4 6:00 | 15 | .008 | .011 | .015 | .021 | .028 |
Full grid (Q2 9:00 .. Q4 3:00, down 10/15/20/25/30): `results/nba_comebacks.json` -> q1.lookup.
Rule of thumb: with this much time left a pregame favourite is worth about **5-7 points of deficit**. A favourite by 8
down 20 at halftime (.18) is about as likely to come back as a pick'em team down 15 (.14-.18).

**Last night's game (descriptive only; 2026-27 is not a dev season).** Down 23 at Q2 7:00, the model gives .08 (pick'em),
.13 (trailer favoured by 4), .17 (by 6), .21 (by 8) and .25 (by 10). I did not look up the actual DEN-NOP line, so the
exact number is not stated here. Historical band (trailer favoured by 4+, down 21-25, Q2 9:00-5:00): **20/96 = .208**.
ESPN said .082 for those same states.

**How to use it live:** if Kalshi's trailer ask is well **below** the table, check first for news (injury, ejection,
starters sitting). Kalshi was calibrated in every late-game bin tested; a gap usually means you are missing
information. A sustained gap would need a pre-registered rule (L28-style) before any money goes in.

### Comeback types: which are realistic? (first time a team trails by >= D, regular + playoffs, Wilson 95% CI)
| first trailing by | when | all | trailer was dog >3.5 | pick ±3.5 | trailer was fav >3.5 |
|---|---|---|---|---|---|
| 15 | Q1 | .183 (1,338) | .095 | .202 | **.460** (226) |
| 15 | Q2 | .153 (2,910) | .074 | .189 | **.356** (492) |
| 15 | Q3 | .079 (2,418) | .048 | .092 | .164 (402) |
| 15 | Q4 | .018 (1,525) | .011 | .015 | .047 (253) |
| 20 | Q1 | .151 (265) | .067 | .180 | .439 (41) |
| 20 | Q2 | .066 (1,414) [.055-.081] | .032 | .092 | **.193** (171) [.141-.259] |
| 20 | Q3 | .035 (1,788) [.027-.044] | .018 | .045 | .100 (230) |
| 20 | Q4 | .002 (1,405): 3 games in 10 seasons | .002 | 0/364 | .006 |
| 25 | Q2 | .032 (554) | .019 | .030 | .130 (54) |
| 25 | Q3 | .013 (1,027) | .006 | .022 | .044 (91) |
| 25 | Q4 | 0/993 | | | |

Realistic: a **better team** down 15-20 in the first half (19-46%). Rare but real: anyone down 20 in Q3 (3.5%; 10% if
the trailer was the better team). Essentially never: 20+ down once the 4th quarter starts (3 of 1,405), 25+ in Q4 (0
of 993). ESPN's WP is below the realised rate in every one of these rows.

## PRICED vs possibly UNPRICED

PRICED (no residual vs the closing line / pregame-adjusted baseline; CIs exclude anything bettable):
- Team quality: net / offensive / defensive rating (ATS -0.03 pts/SD).
- In-game comebacks by trailer/leader traits: 3PA rate, pace, ORtg, bench net, star usage, ball movement (AST ratio,
  assisted share), ORB%, TOV%, leader DRtg, FT%, bench drop-off, starter-pull habit (0/69 cells).
- Style matchups for sides: 3PA vs 3PA allowed (+0.23, p .07), FT-rate vs FT allowed (+0.23, p .08), ORB vs DRB,
  TOV vs forced TOV, transition vs transition D, pace mismatch.
- Rest / back-to-back / 3-in-4 / travel / road-trip length. Coach identity and coach late-game "skill". Timeout habits.
- Referee crews on sides (6 tests) and on totals (earlier R1/B1). The FT effect of crews is real but in the line.
- Home court overall (-0.03 pts ATS).

POSSIBLY UNPRICED (survived BH + both halves; small; NOT yet bettable):
- **Paint defense**: teams that concede a high share of points in the paint cover ~0.5-0.7 pts less per SD.
- **3P% luck in totals**: high season-to-date 3P% (own or opponents') -> unders. About 0.6-0.8 pts per SD each.
- **Spread size -> overs**: +0.58 pts per SD of |spread| (blowout pace / garbage time scoring?).
Leads only (nominal p < .1, not surviving): altitude home +1.0 pt ATS; last-game ATS overreaction -0.23/SD;
leader's pace lowers halftime comebacks (-1.1 pp/SD, p .026); trailer star-usage share raises them (+1.2 pp/SD, p .015);
playoff zig-zag +0.9 pts. With 152 tests, 7.6 nominal p<.05 hits are expected by chance; we got 15.

## Q1 detail: comeback heterogeneity

Baseline: per-minute game states (47 per game, 12,803 games), trailer outcome ~ logistic(deficit/sqrt(tau),
trailer spread*sqrt(tau), deficit, spread, tau, deficit*tau, + 3 quadratic terms), fit leave-one-season-out.
Quadratic terms were added after the first calibration look: the linear version ran 1-5 pts high for 20+ deficits.
LOSO log loss went from .45139 to .45124. The choice was made on calibration only, before the trait tests were rerun.
Calibration (first state per game per cell, LOSO), realised vs model vs ESPN:
Q2 down 15-19 .145 / .140 / .092; Q2 20-24 .066 / .074 / .041; Q3 15-19 .084 / .078 / .049; Q3 20-24 .033 / .032 / .016;
Q4 10-14 .065 / .067 / .045. The model is within about 1 pt everywhere with n >= 1,500. ESPN is 30-50% too low on
trailers.

Trait tests (regular season, deficit >= 10, one state per game per window; residual = outcome - LOSO baseline; traits
season-to-date, >= 10 prior games, z within season; effect in pp of comeback probability per SD):
| window | n | comeback rate | baseline | ESPN (subset) |
|---|---|---|---|---|
| halftime | 4,805 | .129 | .127 | .092 vs realised .132 (n 3,862) |
| end Q3 | 5,745 | .048 | .052 | .033 vs .049 |
| Q4 6:00 | 5,997 | .0195 | .0172 | .0118 vs .0197 |
Smallest p of 69: halftime trailer star-usage share +1.23 pp/SD [0.24, 2.23] p .015 (9/10 seasons); halftime leader
pace -1.10 [-2.06, -0.13] p .026; end-Q3 leader FT rate +0.66 p .030. BH needs p < ~.0015 at the first rank: **0
survivors**. Trailer 3PA rate: halftime +0.54 [-0.52, 1.60], end Q3 +0.10, Q4 6:00 +0.00. Trailer 3PA x leader
3PA-allowed: -0.12, -0.19, +0.29 (signs flip). Leader starter-pull habit (core minutes share in prior 15+ wins): +0.19,
+0.09, +0.24, all p > .2. Descriptive, down 15+ at halftime: high-3PA-tercile trailers came back 9.7% vs baseline 7.7%
(n 689 per tercile, CI about ±2.2 pts). At the end of Q3, 2.2% vs 2.3%.
**Power:** at halftime n ≈ 4,250, the SE is ~0.5 pp per SD. Effects under ~1 pp per SD cannot be resolved, and a 1 pp
shift on a 13% event is not tradeable through a 1-2c spread anyway.

## Q2 detail: four factors and style matchups vs the closing line (regular season, 10,403 games, 61 cells)

Targets: ATS residual = home margin - closing home spread (SD 12.9); total residual = points - closing total (SD 18.1).
Features are season-to-date (>= 10 prior games), z within season. "Matchup" = H offense x A defense-allowed minus the
reverse (sum for totals).
| cell | pts per SD [95% CI] | p | early 15-19 / late 20-24 | seasons same sign | BH (Q2) | BH (all 152) |
|---|---|---|---|---|---|---|
| ATS diff opp paint share (H-A) | **-0.54** [-0.79, -0.29] | 2.5e-5 | -0.47 / -0.61 | 10/10 | yes | yes (Bonferroni too) |
| TOT sum 3P% | **-0.68** [-1.03, -0.33] | 1.5e-4 | -0.78 / -0.57 | 9/10 | yes | yes (Bonferroni too) |
| TOT sum opponents' 3P% | **-0.59** [-0.93, -0.24] | 8e-4 | -0.66 / -0.51 | 9/10 | yes | yes |
| TOT abs(spread) | **+0.58** [+0.23, +0.93] | .0012 | +0.33 / +0.83 | 9/10 | yes | yes |
| TOT matchup 3P% x opp 3P% | +0.51 [+0.17, +0.85] | .0036 | +0.44 / +0.58 | 8/10 | yes | no |
| TOT sum ORtg | -0.44 | .014 | | 8/10 | no | no |
| ATS diff opp 3P% | +0.28 | .032 | +0.52 / +0.03 | 7/10 | no | no |
| ATS diff last-5 ATS record | -0.25 | .048 | | 8/10 | no | no |
Robustness, against the multi-book median line in refs_dev (2021-22..2024-25, n 4,277): paint -0.73 (p .0005);
sum opp 3P% -0.77 (p .005); abs(spread) +0.86 (p .002); 3P% matchup +0.63 (p .02); sum 3P% -0.45 (p .10). Same-season
estimates on the main line are near-identical, so the line source is not driving these.
Paint is not 3P luck in disguise. Joint ATS model: opp paint share -0.74 (p 1e-5), opp 3P% +0.07 (p .64), opp 3PA
allowed -0.31 (p .056), own paint share +0.30 (p .025), net 0.04. Reading: **paint/rim battles are slightly
under-weighted**. Teams that score inside do a bit better than the line and teams that concede inside do a bit worse.
Caveat: "paint points" is a proxy, not rim attempts or rim protection.

Walk-forward (fit on seasons < s, bet season s, -110, |pred| >= threshold):
| rule | thr | bets | win [95% CI] | ROI | 2018-20 / 2021-24 |
|---|---|---|---|---|---|
| TOT_3PT (sum 3P%, sum opp 3P%, abs spread) | 1.0 | 2,780 | .533 [.515, .552] | +1.8% | .521 / .541 |
| TOT_3PT | 1.5 | 1,417 | .536 [.510, .562] | +2.3% | .525 / .541 |
| TOT_3PT | 2.0 | 651 | .542 [.504, .581] | +3.5% | .527 / .550 |
| ATS_PAINT | 1.0 | 179 | .553 [.480, .626] | +5.6% | .455 / .575 |
TOT_3PT is post-hoc: the combination was chosen after seeing the survivors. Every CI includes break-even (.524).

## Q3 detail: coaching and rest (9 cells, 0 survivors)
| test | effect [95% CI] | p | n |
|---|---|---|---|
| C1 coach's prior-seasons ATS mean (H-A) | +0.15 pts/SD [-0.26, 0.56] | .47 | 4,248 |
| C2 coach's prior close-game residual (Q4 5:00, within 5) | -0.67 pp/SD [-3.2, 1.9] | .61 | 1,182 |
| R1 rest-days diff | +0.14 pts/SD | .26 | 10,403 |
| R2 home on back-to-back | -0.33 pts | .37 | |
| R3 away on back-to-back | +0.58 pts [-0.01, 1.17] | .054 | |
| R4 3-in-4 nights diff | -0.54 pts [-0.97, -0.12] | .011 | |
| R5 travel km diff | -0.01/SD | .97 | |
| R6 away road-trip game # | +0.11/SD | .39 | |
| T1 coach prior-season timeout-after-run propensity | -0.06/SD | .85 | 1,950 (2022-24 only; halves not possible) |
Timeouts (ESPN plays 2021-22..2024-25): after an opponent 8-0 run, coaches call time within 30 s in half the
fair-comparison cases. Propensity differs by coach-season (SD 7 pp, year-to-year r .36), so it is a real habit. It
does not change the next 4 minutes (-0.03 pts, SE 0.11, n 8,836 runs) and does not predict ATS.
Lesson: the naive version said -0.92 pts (p 1e-28). The "no timeout" group included teams that answered at once. A
huge p-value from a bad comparison is still a bad comparison.

## Q4 detail: referees on sides (2021-22..2024-25 regular season, 3,771 games, 6 cells, 0 survivors)
Walk-forward ref ratings (>= 30 prior games per ref, crew = mean). F1 crew home-ATS rating +0.05/SD (p .82);
F2 crew home-FTA-edge -0.22/SD (p .30); F3 whistle-heavy crew x FT-rate drawers -0.22 (p .29); F4 x foul-prone
defenses +0.24 (p .27); F5 x star usage -0.18 (p .38); F6 x favourite +0.14 (p .52). Halves (2021-22 vs 2023-24)
flip sign in 4/6. The known findings stand (trailing-team leniency in L2M, crew FTA effect on totals is priced). Crews
add nothing for sides.

## Q5 detail: other (7 cells, 0 survivors)
Altitude home +0.98 pts [-0.02, 1.99] p .055 (6/10 seasons). Previous-game margin diff -0.17/SD p .17. Previous-game
ATS residual diff -0.23/SD p .067 (7/10). Bounce-back after a 20+ loss +0.21 p .51. Favourites in March/April -0.23
p .33. Home ATS overall -0.03 p .80. Playoff zig-zag +0.89 [-0.19, 1.97] p .105 (n 639; early -0.12 / late +2.03).

## Proposed PRE-REGISTERED forward-paper rules (paper only; nothing is bet until a forward or sealed pass)

Frozen 2026-10-25, before any 2025-26 or 2026-27 outcome is looked at. Coefficients come from `nba_comebacks.py rules`
(fit on all of 2015-16..2024-25).

**P-TOT3 (totals, 3P regression).** For each regular-season game where both teams have >= 10 games played this season:
- Features: A = home season-to-date 3P% + away season-to-date 3P% (FG3M/FG3A summed over prior games). B = the same
  for 3P% allowed. C = abs(closing spread).
- Each feature is z-scored against the current season's games-to-date (same filter).
- pred = -0.785 z(A) - 0.713 z(B) + 0.596 z(C) (points vs the closing total).
- Bet OVER if pred >= +1.5, UNDER if pred <= -1.5. Take the closing total (or the Kalshi/Pinnacle total at a fixed time,
  logged before tip). 1 unit, hold.
- Grade: ROI at the actual price and win rate vs 52.4%. Evaluate after >= 600 bets (about 3 seasons at ~200/season).
  Pass = one-sided p < .05 vs break-even AND ROI > 0 in each half.
- Honest power note: dev says ~+2% ROI. Detecting that takes ~14k bets (protocol.md 7), so this mainly guards against
  the effect being zero. Expected outcome: inconclusive after one season.

**P-PAINT (sides, paint defense).** pred = -0.537 z(home opp-paint-share - away opp-paint-share), with
opp-paint-share = opponents' paint points / opponents' points, season-to-date. Bet the favoured side vs the closing
spread when |pred| >= 1.0. Expect only ~20-30 bets a season. **Better use:** add it as a feature to the frozen L2
game model (nba_matchup.py / nba_fourfactors.py) and log CLV, not as a standalone bet.

**Sealed option:** 2025-26 is the NBA sealed season. P-TOT3 and P-PAINT can each be scored there ONCE (that needs
2025-26 team logs and closing lines) on the user's explicit "yes". Do not pull 2025-26 stats before that.

## Caveats
- Lines: spreads = ESPN pickcenter (2015-16..2022-23; the provider varies, effectively the close) and OddsData
  sqlite (2023-24..2024-25). OddsData spreads are UNSIGNED in older seasons (signed here by the ML favourite), and
  2019-20 rows disagree with ESPN (corr .58), so ESPN is preferred. Totals = OddsData (170-270 kept). The source is
  probably a closing line, but that is unverified. Survivors were re-checked against the multi-book median for
  2021-24: same answers.
- Coaches = stats.nba.com head coach for the team-season. Mid-season firings mislabel some games.
- "Starters" = top 5 by minutes in that game. "Paint" is not "rim". Passing = AST ratio / assisted share. Player-tracking
  passes were not pulled.
- Trait tests in Q1 are regular season only. Comeback tables include playoffs.
- Empirical lookup bands (±2 pts, ±2 min, ±3 spread) bias the bracketed rate upward in steep late-game cells.
- Nothing here touches Kalshi prices. Any live comeback rule needs its own pre-registration (see L28).

Reproduce: `../.venv_quant/bin/python scripts/nba_comebacks.py pull && ... timeouts && ... build && ... all`
(stats.nba.com reachable from this Mac; pull takes ~10 min).
