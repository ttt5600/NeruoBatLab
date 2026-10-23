# NBA referees: bias, crew effects and "is the league fixing games?" (2026-10-22)

User ask (10-22): "Biases exist; if they're fixing matches there should be trends and patterns. I think we can
statistically prove inherent bias and perhaps match fixing by the NBA."
Script: `scripts/nba_referees.py` (pre-registration in the docstring, written before any result).
Results: `results/nba_referees.json`. Ledger: `lab/ledger.jsonl`, dataset `nba_referees`, 7 trials, bar p < 0.00714.
DEV = 2021-22 .. 2024-25 (ESPN labels 2022-2025). SEALED 2025-26 was never loaded.

## Verdict in plain language

- **Bias: yes, one kind, and it fits a known pattern.** In Last Two Minute (L2M) reports, the NBA's own graded
  calls, **58.2% of officiating errors benefit the team that is behind at that moment** (1,979 errors, 1,079 games,
  2021-22..2024-25, p < .0001; passes Bonferroni). This matches Price, Remer & Stone (2012), who found referees
  favour teams trailing in a game. **See the mechanism check below.** Part of the 58% is opportunity:
  trailing teams commit more infractions. But within the same call type, the trailing team's infractions still
  go uncalled ~5.7 pp more often (post-hoc, CI +3.7..+7.7). So this is a real leniency toward whoever is behind.
  That is a bias that keeps games close, not fixing.
- **No home, favourite or series-extension bias** survives. Errors favouring home: 50.6% (p .33, 6,097 errors 2015-2025).
  Errors favouring the favourite: 49.4% (p .37). A team trailing in a playoff series in games 4-6 won 42.4% vs a
  market 43.5% (n 125, p .85). There is no sign of "the league stretches series".
- **Crews differ, and the market prices it.** Crews clearly change how many fouls and free throws there are
  (positive control, p .002). Crew identity explains a sliver of points relative to the closing total
  (R1 p .017, which does not pass the bar). That sliver is about 1.3 points of SD, does not persist well
  (split-half r 0.22), and is not bettable: walk-forward "bet by crew" lost -1.9% on 2,222 bets (p .83).
- **Fixing: no evidence, and these data could not show it if it were happening on a small scale.** The
  Donaghy-style screen (late total moves clustering on particular referees) is within chance: the most extreme
  referee is at z 2.7 vs a 95% null maximum of 3.7, perm p .38.

## What each test can and cannot show

| Test | What a positive would mean | What it can never mean |
|---|---|---|
| R1/R2 crew vs closing line | some crews push totals/home results in a way the market misses | that a crew *intends* it. Foul-calling styles differ honestly (C0 shows they do) |
| R3 series extension | trailing teams outperform the market in games 4-6 | league orders. Effort, desperation and adjustments are alternatives |
| R4 L2M error direction | errors are not symmetric between teams | intent. L2M is self-graded by the NBA and covers only close games' last 2 minutes |
| R5 Donaghy screen | a referee's games attract unusual line moves | anything about that referee. Moves follow injury news, and several dozen referees guarantee some "outliers" by chance |
| B1 betting | the bias is exploitable after vig | anything about honesty |

**What would count as evidence of fixing:** information that is *not* in box scores. That means betting-account
or sportsbook integrity alerts (how Jontay Porter in 2024 and the Rozier case in 2023 were flagged), money flows,
communications (the FBI investigation that caught Donaghy), or a referee/player whose games show a large, pre-registered,
*out-of-sample* anomaly tied to betting volume. **What would not count:** any aggregate bias like the ones above
(crowds, compensation and stars produce them with no intent), a single referee's bad season (with ~90 referees,
the worst one is always ~2.5-3 SD out by chance), or post-hoc cherry-picked games.

The base-rate problem: a referee who fixes a handful of games a season out of the dozens he works,
in 5,000+ games is invisible to any crew-level test here unless it is pre-targeted. Statistics did not catch
Donaghy; analyses of his games came only *after* the fact, and even then the analyses disagreed (see literature).

## Results (DEV, all pre-registered, Bonferroni bar 0.00714)

| Trial | Result | p | Pass |
|---|---|---|---|
| R1 crew -> total vs close | excess R^2 0.0054 (obs 0.0219 vs null 0.0164), crew SD ~1.3 pts of 17.8 | 0.017 | no |
| R2 crew -> home vs de-vigged ML | excess R^2 0.0014, crew SD 1.7 pp | 0.27 | no |
| R3 series trailing team, G4-6 | win 42.4% vs market 43.5%, n 125 (81 elimination games, excess 0.000); ATS 52% | 0.85 | no |
| R4a L2M errors favour home | 50.6% of 6,097 (IC 52.4%, INC 50.4%) | 0.33 | no |
| R4b L2M errors favour favourite | 49.4% of 5,693 | 0.37 | no |
| R4c L2M errors favour team trailing at the time | 58.2% of 1,979 (IC 59.3%, INC 58.0%) | <0.0001 | **yes** |
| B1 bet over/under by walk-forward crew rating | -42.0u / 2,222 bets, ROI -1.9%, win 51.6%; 2023 -5.2%, 2024 +0.9%, 2025 -1.6% | 0.83 | no |

Descriptive (no p in the bar):
- C0 positive control: crew identity explains total FTA (excess R^2 0.020, ~1.5 FTA SD) and total fouls
  (0.035, ~1.3 fouls SD), both p .002 (the floor at 500 perms). So the R1 method has power. Crews really call games
  differently. That just barely shows up in points and is mostly already in the closing total.
- League-level sanity: totals beat the close by +0.48 pts (SD 17.8); home teams won 0.7 pp less than the de-vigged ML.
- R5 (2023-24, 2024-25, 2,618 games with open and close): cut = 9-point total move, 158 flagged games. The result
  went with the move 55.1% of the time. Over 70 referees with >= 40 games, the max over-representation was
  z +2.70 vs a 95% null max of 3.70 (perm p .38). The max "with-move" rate among those with >= 5 flagged games was
  0.90 vs a null 95% max of 1.00. **Nothing anomalous.** Referees are reported as anonymous ranks only, on purpose.

## R4c mechanism check (POST-HOC, not a trial)

The pre-registered null ("errors are equally likely to help either team") ignores **opportunity**. In the last two
minutes the trailing team commits more infractions: 57.5% of real ones (CC + INC). Many are deliberate fouls.
So I compared error **rates per opportunity** (POST-HOC: scripts/nba_referees_posthoc_r4c.py writes /tmp/l2m_lc.parquet,
then scripts/nba_referees_posthoc_r4c_withintype.py). Not in the ledger.

| Cut (2021-22..2024-25 L2M, matched to ESPN play-by-play) | missed-call rate INC/(INC+CC), leading team commits | trailing team commits | wrong-whistle rate IC/(IC+CC), leading team whistled | trailing team whistled |
|---|---|---|---|---|
| all graded plays | 18.1% (of 3,906) | 18.5% (of 5,290) | 5.1% (of 3,372) | 2.8% (of 4,436) |
| Q4 with > 60 s left | 20.6% | 29.4% | 5.3% (1,178) | 4.1% (861) |
| excluding "Personal" fouls (where deliberate fouls sit) | 20.7% | 30.9% | 5.7% (2,520) | 4.7% (1,858) |

- Raw, the trailing team's infractions are missed at the same rate (diff -1.3..+2.0 pp, game-cluster 95% CI). But
  that is because deliberate fouls by the trailing team are nearly always called correctly, and they pad its
  denominator. Strip them out and the trailing team's infractions are missed ~10 pp more often.
- **Within call type** (17 types with >= 15 plays per side, excluding Personal fouls), the trailing team's
  infractions are missed **+5.7 pp** more often (game-cluster bootstrap 95% CI +3.7..+7.7). The direction holds in
  every large type: shooting fouls 22.1% vs 15.9%, loose ball 39.0% vs 27.4%, offensive 50.0% vs 35.6%,
  travelling 78.2% vs 70.5%.
- Wrong whistles also land more on the leading team: 5.1% vs 2.8% raw (lead-minus-trail CI +1.5..+3.2 pp). The gap
  is 5.7% vs 4.7% once deliberate fouls are excluded.

**Reading.** Part of the 58% headline is opportunity (trailing teams foul and gamble more). After adjusting for
it, a real residual remains: **late in close games, officials are more lenient toward the team that is behind.**
The same "trailing-team" bias appeared in Price-Remer-Stone (2012) for 2002-08 with a different method. That is
consistent with an unconscious "let them play / keep it close" tendency, and is also partly unresolvable: the same
call type is not the same play when one team is scrambling on defence. **It is not evidence of fixing.** It is
small, symmetric across teams, visible only in the league's own self-published grading, and it helps whoever is
behind, not a particular team, market or betting side. A pre-registered replication would be the next step:
within-type miss-rate difference, 2025-26 L2M, when that season's compilation exists.

## Literature (Part A; full notes from the 10-22 research agent; EVIDENCE = peer-reviewed/court/official)

- **Donaghy (2007).** He pleaded guilty Aug 2007 to wire-fraud conspiracy and transmitting betting info. He bet on
  games he refereed in 2003-07 and sold picks. He was caught through an FBI organized-crime investigation, **not
  statistics** (EVIDENCE for the plea; the detection details are ANECDOTE). The Pedowitz Report (Oct 2008, 200+
  interviews) found no other referee bet and no proof his calls changed outcomes. Critics noted the NBA reviewed its
  own disputed games. Statistics after the fact: R.J. Bell (Pregame) reported overs 57% in his games vs 44% before,
  and line-move side 15/15 in 2006-07 (ANECDOTE, not peer-reviewed). BYU's Shane Reese found he "was not an outlier"
  on over/unders (ANECDOTE). I found no Zitzewitz paper specifically on Donaghy. His "Forensic Economics" (JEL 2012)
  exists, but its Donaghy content is unverified.
- **Price & Wolfers (2010, QJE 125(4))**: opposite-race crews call about 4% more fouls (0.18 per 48 min). Assignment
  is effectively random (EVIDENCE). **Pope, Price & Wolfers (2018, Mgmt Sci)**: the bias was 0.23 before the 2007
  publicity and ~0 after (the drop is significant only at 10%) (EVIDENCE). Lesson: biases are real, measurable, and
  respond to scrutiny. That is the opposite of a fix.
- **Price, Remer & Stone (2012, JEMS 21(1))**: discretionary turnovers favour home teams, teams trailing in the game,
  and teams trailing in a playoff series. They read this as demand-driven bias (EVIDENCE; magnitudes only via 2009
  press coverage, a 5-10% edge; one reviewer called the series effect insignificant). Our R4c agrees on in-game.
  Our R3 finds nothing on series (2022-25, n 125).
- **Pelechrinis (2023, Sci Rep 13:4664)**: L2M 2015-22, home whistle edge 1.6% overall. It is 2.2% in 2015-19 and
  ~0 in 2020-22 (p .49) (EVIDENCE). Our R4a (2015-2025, 50.6%, n.s.) is consistent with a small, faded home edge.
- **Home bias / bubble**: Higgs & Stavness (2021, Sci Rep): bubble home advantage fell below the pre-COVID mean
  with P = 0.999 (EVIDENCE). Moskowitz & Wertheim *Scorecasting* (2011) argued crowd-driven referee bias drives home
  advantage (trade book). Their NBA numbers are unverified.
- **Series extension**: Zimmer & Kuethe (2009, JQAS) found closer early-series margins and a large-market tilt in
  2003-08, "suggests bias, source unidentified" (EVIDENCE, weak).
- **Jontay Porter (2024)**: sportsbooks flagged unders bets of $10-20k against typical $1-2k limits. NBA lifetime ban
  Apr 2024; federal guilty plea Jul 2024. **Detected by betting-pattern alerts.**
- **Oct 2025 EDNY indictments**: Rozier and Damon Jones (insider availability info, prop bets; Rozier's Mar 23 2023
  game had props pulled before tip). Billups is in a separate mafia-linked rigged-poker case. Jones pleaded guilty
  ~Apr 2026, and a superseding Rozier indictment followed May 2026. Rozier and Billups pleaded not guilty
  (as of mid-2026 reporting). **No referee was implicated.** These were player-information schemes, detected
  through betting alerts and investigation, not officiating statistics.
- **Crew totals**: no peer-reviewed study found that crews move totals relative to the line. Rodenberg (JQAS 2011;
  JPM 2013) found nulls on specific crew-team bias claims. "Crew X adds 3 points" claims on betting sites are
  unsourced (ANECDOTE). Our R1/B1 agree: small and priced.

## Data

- Crews: `gameInfo.officials` in cached ESPN summaries, 5,240 of 5,289 DEV games. Playoff and play-in games list
  4 officials, alternate included; all 4 enter the crew. 88 referees. `data/espn/nba/refs_dev.parquet`
  (`build` rebuilds it from cache in ~45 s).
- L2M: public compilation github.com/atlhawksfanatic/L2M (`data/l2m/L2M_stats_nba.csv`, 2015-03..2025-05,
  pulled 10-22). official.nba.com robots.txt disallows the `?gameId=` report pages, so there was no fresh scrape.
  The sealed 2025-26 L2M is therefore not available here anyway.
- Favourites for R4b: SBR closing moneylines (`data/nba_odds/OddsData.sqlite`, 91% of L2M games matched).
- Trailing team for R4c: ESPN play-by-play score just before the error clock.

## Caveats

- ESPN "current" totals for 2021-22/2022-23 are approximately closing, not a verified close.
- L2M covers only close games (league criterion: within about 5 points in the final two minutes; exact criterion history not verified here), and the league grades itself.
- R3 has only 125 games: it can rule out a large effect (~>10 pp) but not a small one.
- Sealed test available on a "yes": R4c is a bias test with no betting use, so it has no sealed P&L. R1 failed the
  bar and B1 lost, so there is nothing to promote.
