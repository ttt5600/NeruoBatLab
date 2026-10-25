# NFL: comebacks, style matchups, coaching, referees -- all measured against the market (2026-10-25)

Script: `scripts/nfl_comebacks.py` (about 5 s). Full generated tables, including every test row:
`notes/nfl_comebacks_matchups_tables.md`. JSON: `data/lab/nfl_comebacks_results.json`.
Data: nflverse play-by-play 2006-2019 (660,349 plays, 3,738 games), participation 2016-19 (pressure and pass rushers;
man/zone exists only for 2018-19), and the nflverse schedule (closing spread and total, head referee, coaches).
**Only dev seasons 2006-2019 were used.** 2020-2025 is the lab NFL lockbox and 2026 is forward. The one exception is a
Kalshi pricing check on the in-game DEV split that `scripts/nfl_ingame.py` pre-registered (2025 REG weeks 1-12, already
used for H1-H3). That trial's sealed set (2025 wk13+/POST and 2026) was not read.

## Verdict

1. **Comebacks follow the market's win probability.** Across 24,044 game states, nflfastR's spread-aware
   `vegas_wp` is calibrated to within about 1 pp in every bin from 0-40% (table D). The one exception is the 40-50% bin:
   trailing teams won 49.3% against 44.7% predicted, +4.6 pp, with a game-clustered bootstrap CI of +1.8 to +7.4. Those
   states are mostly trailing pregame favourites.
2. **Team traits help only through "the better team comes back more", and Kalshi already prices that.** The trailing
   team's season-to-date EPA margin minus the leader's adds **+2.6 pp per SD** of comeback probability over vegas_wp
   (CI +0.9 to +4.2, p .0027, positive in both 2006-12 and 2013-19). It still **fails correction**: the Bonferroni bar is
   p < .00071 over 70 tests, and nothing passes BH. Most of it is the pregame spread: vegas_wp lets the spread fade too
   fast. With the spread as a control the EPA slope falls to p .12. On the Kalshi dev games, Kalshi prices trailing
   pregame favourites **+3.9 pp above vegas_wp** (35 games). That is at least as much as history says vegas_wp misses
   (+2.4 pp), so the effect is already in the price.
3. **Style traits add nothing beyond vegas_wp**: pass rate over expected, pace/no-huddle, urgency, late-game offence or
   defence, coach 4th-down aggressiveness, and the pregame total all have |effect| < 1 pp/SD with p > .5.
4. **Scheme-vs-scheme matchups predict nothing against the closing line.** I ran 14 interaction tests (pass/run tendency
   and efficiency against the opposing defence's weakness, QB pressure or blitz sensitivity against defence pressure or
   blitz rate, QB man-vs-zone split against defence man rate) plus a 9-cell style grid. None survives. The best is
   p .0055 in one cell of the grid, with no pattern across neighbouring cells. Pressure, blitz and man/zone are
   under-powered (2-4 seasons, 380-750 games).
5. **"Good teams win" is priced.** Season-to-date EPA margin predicts the raw score margin strongly (+5.3 pts/SD,
   p 1e-76), but predicts only +0.3 pts/SD against the spread (p .25). Offence EPA is +0.56 pts/SD ATS (p .04) and pass
   offence +0.48 (p .08), both positive in both halves. That is a weak "the market under-weights season-to-date EPA" hint,
   nowhere near the bar. Turnover margin, one-score "luck" and point margin, each net of EPA, are flat. The market does
   not overreact to them here.
6. **Coaching shows no persistent skill against the market.** Coach ATS in odd vs even seasons gives r = +.21 (46
   coaches, p .17). Coach residual when trailing at half gives r = +.06 (35 coaches, p .72). 4th-down aggressiveness
   does predict a coach's own later decisions (per-game r = .07, so it is a real trait), but it predicts nothing against
   ATS, totals or comebacks.
7. **Referees move flags but not points against the line.** The positive control works: a referee's past flags/game
   predicts flags in this game, +0.69/SD, p 2e-23. But past flags, DPI and holding predict totals residuals of -0.36 to
   -0.40 pts/SD (p .08-.11). That direction is opposite to the "more flags, more points" idea, and the flags test flips
   sign between halves. A referee's home penalty bias gives home ATS +0.03 (p .91). The NBA result repeats (L6): crews
   are real, points are priced. The data has only the head referee, not full crews.

**Bottom line: nothing here is an edge.** Every factor the user listed is either priced or noise at the available n.
The only market observation with a plausible mechanism is that Kalshi charges more for trailing teams than vegas_wp
does, +2.6 pp on average, and trailing underdogs (who come back at about the vegas_wp rate) are priced +1.8 pp
(SE 0.4) above it. That is a weak fade-the-trailing-dog lead (CB1 below), not a finding.

## What to use on game day

### Comeback lookup (trailing team's actual win rate, dev 2006-2019; ties count as half)
Rows are game clock, columns are the deficit. "Has ball" splits come from table C; the full grids with n and
vegas_wp are in the tables file (A, B, C).

| time left | down 1-3 | down 4-7 | down 8-10 | down 11-14 | down 15-17 | down 18-21 |
|---|---|---|---|---|---|---|
| halftime | 42% | 30% | 17% | 12% | 7% | 8% |
| mid Q3 (22:30) | 42% | 31% | 18% | 11% | 6% | 4% |
| end Q3, has ball / not | 46 / 31% | 31 / 23% | 20 / 12% | 12 / 4% | 3 / 1% | 1% |
| 10:00 Q4, has ball / not | 50 / 28% | 29 / 16% | 20 / 9% | 7 / 3% | 1 / 0% | 1% |
| 5:00 Q4, has ball / not | 52 / 24% | 29 / 12% | 12 / 3% | 5 / 1% | 2 / 0% | 0% |
| 2:00 Q4, has ball / not | 48 / 10% | 28 / 3% | 4 / 2% | 1 / 0% | 0% | 0% |

Adjust for the pregame spread with table B in the tables file. At halftime, down 4-7: a trailing favourite by >3
wins 47%, within 3 points 29%, an underdog by 3.5-7 22%, an underdog by >7 14%.

**How big a comeback is possible:** over 14 seasons the eventual winner trailed by at least 10 points in 13.4% of games,
by 14 in 5.5% (206 games), by 17 in 2.0% (74), by 21 in 0.7% (26), by 24 in 0.2% (8), and by 28 once.

**How to use it against a Kalshi price.** In 2025 dev games Kalshi priced the trailing team above vegas_wp by about
+2 pp on average (+3.9 pp for trailing favourites, +1.8 pp for trailing underdogs). History says vegas_wp is about right,
except that it under-rates trailing favourites by about 2 pp. So:
- When Kalshi's trailing-team price is about the table value plus 2-4 pp, that is normal. Do nothing.
- Buying a trailing team on Kalshi is not supported by anything here. The "stronger team will come back" premium is
  already in the price.
- A trailing underdog priced 3+ pp above the table at the same time, deficit and possession is the only direction with
  a hint (CB1). Paper only.

### PRICED (do not bet on these)
Team quality (EPA margin, offence and defence EPA, QB EPA+CPOE, point margin), pass rate over expected, pace/no-huddle,
late-game offence and defence splits, 4th-down aggressiveness, turnover margin and one-score luck, the pregame total in
comeback states, all scheme interactions (pass/run, pressure, blitz, man/zone), referee flag, DPI and holding rates and
home penalty bias, coach identity (ATS and halftime comebacks), and possession and timeouts (already inside nflfastR WP
and the market).

### Looks UNPRICED: nothing passes. Weak leads, each needing a new pre-registered trial:
- **CB1 fade trailing underdogs on Kalshi in-game.** Rule, written now: at the 2H checkpoints (30:00, 22:30, 15:00,
  10:00, 5:00 left), take the first state per game where the trailing team was a pregame underdog by >3, trails by 4-14,
  and its Kalshi mid minus vegas_wp is ≥ 0.02. Buy the LEADER YES at the ask (quote spread ≤ 5c), pay the 0.07·p·(1-p)
  fee, and hold. Score ROI with a one-sided bootstrap over games, bar p < .05.
  Basis: dev trailing underdogs won 10.9% against Kalshi 11.4% and vegas_wp 9.6% (80 games), so the edge is about
  1-2 pp on 88-90c contracts. That is roughly a 1% ROI after the fee, which needs thousands of games to detect. The
  related K-FAV favourites idea was dead in non-sports. **Recommend: do not spend a sealed shot on CB1**; log it as
  forward paper only if the NFL live scanner is already running.
- **L-EPA (season-to-date offence/pass EPA against the spread):** +0.5 pts/SD, p .04-.08, positive in both halves. That
  is about 1-2% ATS at best. The 2020-25 lockbox is the only fair test and needs the user's yes. Low priority.

## Tests and caveats
- 70 tests are counted (positive controls excluded). Bonferroni bar p < .00071. BH (q .05) passes 0, Bonferroni passes 0.
  Smallest p values: EPA gap vs vegas_wp .0027; both defences' EPA against the total .0049 (-0.79 pts/SD: games between two
  bad defences go UNDER the closing total, meaning the market over-adjusts totals for bad defence; negative in both
  halves but fails correction, so it is a lead for a new trial at most); one grid cell .0055; blitz interaction .013 (2016-19 only).
- Traits are season-to-date from earlier games only, with at least 4 prior games, so weeks 1-4 are excluded. Coach and
  referee traits use all earlier dev games, at least 8. 4th-down "expected go" is a same-season league logistic model
  (league norm only; the team side uses prior games). That is a proxy for nflfastR's go-boost, which is not in the pbp.
- vegas_wp is probably partly in-sample for 2006-19 (nflfastR's WP model was fit on roughly this era; I did not verify
  the exact seasons), which flatters its calibration. Kalshi beat nflfastR in-game in the earlier H1-H3 study, so a
  vegas_wp residual is an upper bound on any market residual.
- Comeback tests pool up to 7 checkpoints per game with game-clustered SEs. The "half" columns for the family-1 rows are
  logit slopes; read their signs.
- Clock and timeout behaviour was only tested as no-huddle urgency when trailing (null). Timeouts left are already a
  vegas_wp input. Second-half lines were not available, so "coach ATS when trailing at half" uses the halftime vegas_wp
  residual instead.
- The Kalshi dev check prices states at the 1-minute candle ending ≥ 60 s after the snap (same convention as
  nfl_ingame.py), with spread ≤ 5c. It covers 177 games and 904 states; slope of Kalshi minus vegas_wp on the trailing
  pregame spread is +0.74 pp/SD, p .024.
