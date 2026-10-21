# Situational variables vs NBA/NFL betting lines: literature and data review

Written 2026-10-20 by a research subagent. Web research only; no backtests were run for this note.

## Evidence labels

- **MARKET-TESTED**: tested against a betting line (ATS, totals or odds-conditioned win probability).
- **STATISTICAL**: the variable changes performance, but nobody has shown that the line misprices it.
- **ANECDOTE**: blogs, Reddit, or tout claims with no reproducible method.

## The verdict up front

A few situational variables have been tested against the line, and almost all of those tests are old (pre-2017 data), so the market has had time to learn them. The academic evidence on fatigue and circadian effects is real but small. Effects run about 1 to 4.5 points of margin, with most studies landing near 1 to 1.5. These variables are widely known and are almost certainly priced into NBA sides and totals at close. Our own results already agree: rest edge, back-to-back unders, West-coast-early and similar angles lost about the vig.

The most plausible place left for an edge is the thinner **player prop** market. The best candidates there change *minutes*, not efficiency: blowout risk, minutes restrictions and role changes. A second group is information released on game day that a prop opened before: the referee crew at ~9am ET, and late injury news. For props, I found **no peer-reviewed study of situational effects measured against the line.** That gap is an opportunity, and it also means every prop prior below is borrowed from team-level or performance-only research.

---

## 1. Schedule and fatigue

**Back-to-backs and rest differential.** This is the one schedule effect with a peer-reviewed test against the spread. Ashman, Bowman & Lambrinos (2010, *J. Sports Economics*) used 19 seasons. Home teams on the second night of a back-to-back did badly against the spread when the visitor had 1-2 days of rest, and worse still after travelling 1-2 time zones east ([RePEc](https://ideas.repec.org/a/sae/jospec/v11y2010i6p602-613.html)). **MARKET-TESTED, but on data that ends around 2008.** Only the abstract was available; I could not check effect sizes.

Later, mostly student or observational work suggests rest matters less than people think:
- "Tired of Misattribution" (arXiv 2112.14649) concludes that the impact of cumulative fatigue on game outcomes is minimal ([arXiv](https://ar5iv.labs.arxiv.org/html/2112.14649)).
- A 2023 UNC study finds performance peaks after one rest day. Both too little and too much rest hurt ([UNC](https://our.unc.edu/abstract/abolins-rested-or-rusty)).
- A 2025 CMU capstone found that rest days had little effect on player plus/minus once minutes were controlled for ([CMU](https://www.stat.cmu.edu/capstoneresearch/460files_s25/team15.pdf)).

All of this is **STATISTICAL**. Practitioner consensus is that blindly fading back-to-back teams is priced in ([Action Network](https://www.actionnetwork.com/nba/moore-dont-be-fooled-by-blindly-fading-teams-on-back-to-backs)). That matches our own null result.

**3-in-4, 4-in-6 and road-trip length.** I found no line-tested study of these specific compressions. A Wharton paper notes that teams play about 11.5 home back-to-backs but about 27 on the road ([Wharton PDF](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Nba.pdf)). That makes "back-to-back" and "road" heavily confounded. **Evidence is thin.**

**Time zones and circadian effects.**
- **Roy & Forest (2018, *J. Sleep Research*):** five seasons of NBA, NHL and NFL games. Visiting teams travelling *west* for evening games did worse: NBA F(2,5908)=16.12, p<.0001; NHL p=.011; NFL a trend at p=.058 ([AASM summary](https://aasm.org/nfl-nba-and-nhl-teams-have-a-disadvantage-when-traveling-west), [Sleep Review](https://sleepreviewmag.com/sleep-health/parameters/circadian-rhythms/nfl-nba-nhl-teams-disadvantage-traveling-west/)). **STATISTICAL (win %, not ATS).**
- **McHill & Chinoy (2020, *Sci. Reports*):** before COVID, home teams won 63.8% and away teams 50.8%. Away teams lost more often after crossing 1 or 2 zones, with the effect driven by westward travel. Two zones cost about 1.5 points of eFG%. Teams allowed at least 2.8 more points per 100 possessions at every zone count. Defensive rating also got worse in the bubble, which suggests the measured effect is not purely circadian ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7732833/)). **STATISTICAL.**
- **Leota et al. (2022, *Front. Physiol.*):** 11,481 games, 2011-2021. Eastward jet lag hurt *home* teams: -1.29 points of margin (p=.015), -1.29 rebounds (p<.0001), -1.2 points of eFG%. The effect reached -4.53 points after 2 hours of eastward lag. Away teams showed no effect ([Frontiers](https://public-pages-files-2025.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.892681/text)). **STATISTICAL.**

The papers disagree on direction: Roy/Forest and McHill find a westward penalty, while Leota finds eastward hurts home teams. That disagreement is itself a warning that the effect is small and depends on specification. One useful hint: Leota's **rebound** effect is the strongest coefficient in the paper, which is relevant to our post-hoc rebound-unders lead.

**Altitude (Denver, Utah).** Lopez, Matthews & Baumer rank Denver's home edge as the largest in the four major US leagues, about a 67% win chance against an equal team ([MSU Denver](https://red.msudenver.edu/2018/advantage-altitude-exploring-the-largest-home-advantage-in-sports/)). Visiting free-throw percentage at Denver was 74.9% vs 75.8% elsewhere ([HSAC](https://harvardsportsanalysis.org/2011/05/does-altitude-affect-free-throw-percentage/)). The NBA now schedules to reduce the effect ([CBS](https://www.cbssports.com/nba/news/nba-trying-to-weaken-nuggets-home-court-advantage/)). This is **STATISTICAL, and fully priced** as a team-level home edge. Two splits remain untested as far as I can find: second-half or fourth-quarter production of visitors at altitude, and visitor props in Denver.

**Early and late tip times.** Action Network reports that games starting noon to 5pm ET have gone under 53.4% of the time since 2005. On Christmas Day, noon-4pm games went under 21-11 (65.6%), a tiny sample ([Action Network](https://www.actionnetwork.com/nba/nba-on-christmas-day-betting-trends-stats-notes-action-network-betting-primer)). This is **MARKET-TESTED by a tout outlet**, with no method and no correction for selecting the cut. A Westgate oddsmaker explicitly warns against betting all these unders blind. The idea is plausible given circadian research (West-coast teams at a noon ET tip are playing at 9am body time). Early games are also few, and our "west-coast early" NFL angle already lost.

## 2. Venue and location

**Home-court advantage after 2020.** Sagarin-based home edge was 2.74 points in 2015-19 and 1.75 in 2020-22 ([arXiv 2210.13687](https://arxiv.org/pdf/2210.13687)). Home teams won 54.3% in 2023-24, the lowest in league history ([KSAT/AP](https://www.ksat.com/sports/2024/04/14/inside-the-nba-numbers-lots-of-comebacks-no-home-court-edge-3s-went-up-scoring-went-down/)), and stayed near 54% afterwards except for 2022-23 at 58% ([Boston25](https://www.boston25news.com/news/national/fact-or-fiction-is/NVE32FOQOEVRYBXQSAJNA6OTJA/)). **STATISTICAL.** The lesson for us is to fit home edge per season, not pool it, or any pre-2020 travel feature will be miscalibrated. Books adjusted their home-edge assumptions too; whether they did so slowly is testable on 2020-23 closing lines.

**Arena effects (shooting backgrounds, rims).** I found nothing rigorous. The obvious test is to fit arena fixed effects on visitor 3P% and FT% with shrinkage. Expect them to be mostly noise; the Denver free-throw result is the only documented case and it may be altitude, not sightlines. **No evidence.**

**Nightlife and "party city" effects.** This is the most rigorous source for the idea the user raised:
- **Santucci & Lax, "Causal Hangover Effects" (arXiv 2412.21181, Dec 2024, preprint).** Their NBA sample is 2010-11 to 2016-17, 7,829 games, restricted to games played within 24 hours of the previous one.
  - They define "party city" two ways: a discrete indicator (the team was in LA or NY the night before, chosen from player interviews), and a continuous measure (log count of music groups, recording studios and publishers in the previous metro, from BLS data).
  - **NBA against the spread:** the discrete indicator gives -0.557 (SE 0.149, p<.01). The continuous measure gives -0.158 (SE 0.095, p<.1).
  - **Points allowed:** +2.97 (SE 0.84) after an LA/NY night. Points scored show no effect. In other words, the paper finds a defensive effect, which points toward the **over** on the following game, not the under.
  - **Placebo:** LA/NY with more than 24 hours of rest shows nothing (0.053, SE 0.084).
  - **Betting returns:** they report profitable betting for MLB only. They report **no NBA return** ([arXiv](https://arxiv.org/html/2412.21181v1)).

  **MARKET-TESTED (ATS), but thin.** I see four problems:
  1. The LA/NY indicator is confounded with back-to-back travel geometry. Leaving LA means flying east, and a team may play both LA teams on the same trip. This is my own concern; I have not tested it.
  2. Only two cities were chosen, based on interviews. That is a researcher degree of freedom.
  3. The continuous measure only reaches p<.1.
  4. The data ends in 2017, and the paper is not peer-reviewed.
- **Harvard Sports Analysis (2014)** tested away teams in Sunday-afternoon games after a Saturday night in "party cities" ([HSAC](https://harvardsportsanalysis.org/2014/11/champagne-and-campaign-do-nba-teams-play-worse-on-the-road-in-party-cities/)). The page returned HTTP 500 and I could not read the results. **Unverified.**
- **The Harden strip-club claim: ANECDOTE.** A 2019 Reddit post compared four seasons of Harden's road games with average Google review ratings of each city's strip clubs. It claimed significance, but the comparison has no controls, and both the cities and the metric were chosen after the fact ([Fox News](https://www.foxnews.com/sports/james-harden-performance-strip-club-reddit), [Sports Betting Dime via search](https://www.sportsbettingdime.com/?p=182760)). Fox noted that Harden's road/home split (23.9 vs 24.9 ppg) is a normal home-floor effect. A sportsbook later posted novelty props on the idea. A separate Fox piece reports Sapphire NYC foot traffic rising 29% or more when visiting teams were in town. That data came from the club, has no sample size, and has **no link to performance** ([Fox News](https://www.foxnews.com/sports/new-york-city-strip-club-uptick-foot-traffic-team-plays-the-knicks)). Treat this as entertainment, not a prior.

**How to test the nightlife idea honestly.** Define the exposure *before* looking at any data. Use the previous night's city, and a nightlife index from OpenStreetMap bar and nightclub counts per capita, or BLS QCEW counts of drinking places (NAICS 7224). Interact it with "next game within 24 hours" and "Saturday night". Control for travel direction, time zones crossed and rest. The prediction is pre-specified: **visitor defense gets worse, so the over wins on the next game, and prop overs for the *opponent's* scorers win.** Expect a null result. The interaction cell is small, perhaps a few hundred games a season.

**Crowd and attendance.** McHill's bubble comparison found a 2.6% higher free-throw attempt rate and about 5.5 points per 100 worse defensive rating in the bubble ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7732833/)). Attendance in normal seasons barely varies. **STATISTICAL; low value.**

## 3. Officials

- **Assignment timing.** The NBA posts referee crews at about 9:00am ET on game day ([official.nba.com](https://official.nba.com/referee-assignments/)). Openers on props and totals are often posted the night before. The crew is therefore **true new information that arrives after the open**. That is structurally the most promising variable in this note.
- **Effects on fouls.** Price & Wolfers showed that crew composition measurably moves player foul counts (up to 4%) and points (up to 2.5%). They also established that referee assignment is effectively random, which is the key condition for clean causal estimates ([NBER w13206](https://www.nber.org/papers/w13206.pdf)). **STATISTICAL.**
- **Totals.** I found no peer-reviewed crew-vs-total study. Practitioner claims (for example, "crew X adds 4.2 points") are unsourced. The one market-level data point: after a mid-season free-throw crackdown, unders went from 50.1% to 52.2% according to BetMGM data ([Legal Sports Report](https://www.legalsportsreport.com/179419/how-an-officiating-change-altered-the-nba-betting-landscape/)). That is a league-wide officiating shift, a related but different idea. Rating: **ANECDOTE to weak MARKET-TESTED.**
- **Noise.** Basketball-Reference publishes per-referee FTA, PF and points relative to league average for each season since 1988-89. It cautions that with three officials per game the data is noisy, and that PF counts include fouls called by all three ([B-Ref register](https://www.basketball-reference.com/referees/2025_register.html), [blog](https://www.sports-reference.com/blog/2015/03/page/2/)).
- **Model.** Use crew-level random effects, built from the sum of three shrunk referee effects, to predict free-throw attempts. Test against **free-throw-sensitive props**: points unders for high-FTA guards, and FT-made props. Our prop data stores opening and best opening prices, so we can check whether the 9am crew moves the closing line, and whether bets placed at the open *before* the crew is known capture that move. **This is the most direct "information timing" test we can run.**

## 4. Player-level variables

- **Blowout risk.** Large spreads cut fourth-quarter minutes for starters. The practitioner guidance is to be wary of overs with spreads of about 3 possessions or more ([FantasyLabs](https://www.fantasylabs.com/?p=5245), [Action Network](https://www.actionnetwork.com/nba/how-blowouts-hurting-prop-bettors-nba-postseason)). One newsletter puts blowouts at about 17% of 2020 games, method unverified ([The Box Score](https://theboxscore.beehiiv.com/p/examining-the-trend-of-the-nba-blowout)). I found **no quantified test against the line**. Mechanically this is the most direct route to unders: minutes multiplied by rate. It is probably partly priced, because books build lines from projected minutes. The question is whether they scale minutes enough for spreads of 12 points or more.
- **Minutes restrictions and injury returns.** These are documented case by case, for example Jokić with 31 points in 25 restricted minutes ([GoSkagit/AP](https://www.goskagit.com/sports/ap/nuggets-nikola-jokic-returns-to-lineup-against-clippers-in-1st-game-since-knee-injury-on/article_bf931b5a-9b86-5bca-b236-5864341f95ae.html)). One preview argues that books already post conservative lines for returning players ([TheLines](https://www.thelines.com/best-nba-player-props-for-today-friday-night-edition/)). **ANECDOTE.** The sign is ambiguous: restricted minutes push toward the under, while a soft line plus preserved usage pushes toward the over.
- **Minutes volatility and role changes.** No literature exists against the line. It is still a sound feature: the coefficient of variation of a player's last ~10 games' minutes, and the change in teammates available (an absent co-star raises usage). It is likely the largest real driver of prop errors. Books also update fastest on teammate injuries.
- **Contract year.** Results are mixed and mostly come from theses. One finds about +2.8% PER only after dropping stars ([Sewanee](https://dspace.sewanee.edu/items/94de5213-9a1d-4ed7-aaa2-a03fe62c6042)). Another uses an instrumental-variable design and finds a +3-5 percentile effect ([Harvard thesis](https://dash.harvard.edu/entities/publication/73120378-eb27-6bd4-e053-0100007fdf3b)). Others find declines ([Claremont](https://scholarship.claremont.edu/cmc_theses/768), [BC](https://boots.bc.edu/islandora/object/bc-ir:109737)). **STATISTICAL, weak.** It is a season-long effect that game-level lines absorb through recent form, so it is useless for game-level props.
- **Revenge games.** HSAC studied 486 players and found only "minimal" GameScore increases in games against a former team ([HSAC](https://harvardsportsanalysis.org/2018/02/do-nba-players-perform-better-in-revenge-games/); the page itself returned 500, so this relies on the search summary). It is a media narrative that likely inflates *public* interest in the over. **ANECDOTE.** If anything it is a contrarian **under** candidate. Test it only with a pre-registered definition, such as the first meeting after a trade.
- **Matchup and defense-vs-position.** Raw defense-vs-position numbers mix opponent quality, frequency and role, and are widely considered misleading without adjustment ([Rotowire](https://www.rotowire.com/basketball/article/dfs-basketball-101-defense-vs-position-strategies-31305), [Ringer on rebound noise](https://www.theringer.com/nba/2021/5/11/22423517/nba-defense-analytics-nikola-jokic)). **STATISTICAL and heavily used by books and DFS, so likely priced.** Opponent pace is mechanically real (more possessions mean more counting stats) and is surely in the line.

## 5. Weather and environment (NFL)

- **Market tests.** Borghesi (2008, *Applied Financial Economics*) studied weather biases in the NFL totals market ([RePEc listing via search](https://ideas.repec.org/a/taf/applec/v39y2007i15p1889-1903.html); I could not read the abstract). Rodney Paul finds that humidity, wind and combined strategies reject efficiency ([FIT](https://fitpublishing.com/node/3503)). **MARKET-TESTED, but on old data.**
- **Cold.** Cold barely moves scoring: games below 20°F averaged 42.8 points vs 42.1 overall, n=3,935 outdoor games ([4for4](https://www.4for4.com/fantasy-football/how-does-cold-weather-impact-nfl-scoring)). Overs hit about 58% in games at 20°F or colder since 2009, which suggests the market *over*-adjusts ([Sharp Football](https://www.sharpfootballanalysis.com/betting/cold-weather-impact-sports-betting-nfl-games-over-under/)). If anything, **cold is an over angle**, on a small sample.
- **High totals.** Kelly & Chen's under on totals of 47.5 or more (59.7% in 2001-09) disappeared in 2010-18 ([search summary of the SEF paper](https://www.doi.org/10.1108/SEF-11-2018-0354)). This is a textbook case of an edge decaying after publication.
- **Wind.** The claim that moderate 10-15 mph wind is underpriced has no method behind it ([startuphub](https://www.startuphub.ai/news/nfl-betting-does-moderate-wind-create-unders-market-inefficiencies)). **ANECDOTE.** Our own finding fits the general picture: in-game observed rain and wind are partly unpriced, but much less is visible before kickoff. The edge is in **forecast error**: comparing the forecast at bet time with observed conditions.
- **NBA.** There is no plausible weather channel other than travel disruption, such as delayed flights causing a late arrival. That would need flight-delay data (BTS on-time database) and is speculative.

## 6. Methods: testing many variables without fooling ourselves

1. **The multiple-testing hurdle.** Harvey, Liu & Zhu argue that a new factor needs t > 3.0, not 2.0, given how many factors have been mined ([NBER w20592](https://www.nber.org/papers/w20592)). Apply the same bar here. With 50+ candidate variables, report Benjamini-Hochberg q-values and Bonferroni across the full list we tried, including the failures.
2. **Backtest overfitting.** Use combinatorially symmetric cross-validation and the "probability of backtest overfitting" (Bailey, Borwein, López de Prado, Zhu, *J. Comp. Finance*; an R `pbo` package exists) ([SSRN](https://papers.ssrn.com/abstract=2326253), [CRAN README](https://packages.oit.ncsu.edu/cran/web/packages/pbo/readme/README.html)). Run it on the grid of strategy variants (thresholds, cutoffs), not just one rule.
3. **Model the residual, not the outcome.** The target is `actual - line` (or a de-vigged over probability), with the line included as an offset. Then every feature is asked only "what does the market miss?" Use regularized logistic regression (L1/elastic net) as the primary model and a shallow gradient-boosted model (GBM) as a check. If the GBM finds signal the linear model does not, suspect leakage first.
4. **Walk-forward evaluation by season.** Fit on seasons ≤ t and bet season t+1. Keep one sealed lockbox season, as the project already does.
5. **Select models on calibration, not accuracy.** Walsh & Joshi (2024) report that choosing NBA models by calibration gave about +35% ROI while choosing by accuracy gave about -35% ([Bath](https://researchportal.bath.ac.uk/en/publications/machine-learning-for-sports-betting-should-model-selection-be-bas/)). Treat the magnitude skeptically (one season, one paper), but the direction is standard.
6. **Graph and network models.**
   - A graph-attention-plus-temporal-convolution model over teammate/opponent graphs claims to support profitable betting strategies ([arXiv 2303.16741](https://arxiv.org/pdf/2303.16741)). I could not verify the returns.
   - A tennis GNN reports +3.26% ROI over 1,903 bets on "intransitive" matchups against Pinnacle ([arXiv 2510.20454](https://arxiv.org/pdf/2510.20454)).

   Both are preprints with in-sample-designed subsets and no independent replication. **There is no demonstrated out-of-sample betting value for NBA props.** A cheaper proxy for "graph" information is teammate on/off usage features, such as usage when player X sits.
7. **Two traps specific to this project.**
   - Our prop data showed that overs and unders carry very different holds. Always evaluate a variable *within side* (unders only) against the side-specific baseline loss. Otherwise "variable X helps unders" just means "unders lose less than overs".
   - Use only features known at bet time. Referee crews are known only after 9am ET, injury news has timestamps, and ESPN's `last_updated` field is post-kickoff.

## 7. Free data sources

| Variable family | Source |
|---|---|
| Schedules, box scores, game officials | [nba_api](https://github.com/swar/nba_api) (stats endpoints; `BoxScoreSummaryV2` includes an officials table per [nba-db docs](https://nba-db.readthedocs.io/user_guide/endpoints.html)); Basketball-Reference box scores |
| Referee season stats (1988-89 on) | [B-Ref referee register](https://www.basketball-reference.com/referees/2025_register.html) |
| Referee assignment time (live) | [official.nba.com/referee-assignments](https://official.nba.com/referee-assignments/), scraped daily; historical crews come from box scores, so the *timing* has to be assumed to be 9am ET |
| Travel distance, time zones | Computed: arena coordinates plus haversine distance plus IANA time zones from the schedule (the method in [Fox Sports](https://amp.foxsports.com/stories/nba/2016-17-nba-schedule-team-dashboards)) |
| Altitude | Arena elevation (static; Denver ~5,280 ft, Utah ~4,300 ft) |
| Tip time | nba_api schedule / ESPN API game date-time |
| Lines and props | ESPN core API (already used by `collect.py`) |
| NFL weather | nflverse schedules (`roof`, `temp`, `wind`, outdoor/open only, from PFR) ([dictionary](https://nflverse.r-universe.dev/nflreadr/data/dictionary_schedules/ndjson)); [Open-Meteo ERA5 archive](https://open-meteo.com/en/docs/historical-weather-api) (hourly, 1940 on, free, 25 km grid) |
| Nightlife proxy | OpenStreetMap Overpass (`amenity=bar|nightclub|pub` counts per metro); BLS QCEW establishment counts (NAICS 7224 drinking places), the same family of data Santucci & Lax used (music-industry counts) |
| Injuries and minutes restrictions | NBA official injury report PDFs (game-day, timestamped); news, hard to collect historically |
| Contract status | Spotrac/B-Ref contracts pages (scrape) |

## 8. Ranked: 15 variables to test against NBA prop UNDERS and game totals

"Already-priced risk" is my judgment of how likely the closing line already reflects the variable. "Evidence" uses the labels defined at the top.

| # | Variable | Expected direction | Evidence strength | Already-priced risk | Data source |
|---|---|---|---|---|---|
| 1 | Referee crew FTA tendency, released after the prop opened (bet at open vs crew-adjusted fair value) | Low-FTA crew: points/FT unders and game under | STATISTICAL (Price-Wolfers random assignment); line-tested only as anecdote | Medium at open, high at close (books adjust once the crew posts) | nba_api officials + B-Ref register; scrape 9am posting going forward |
| 2 | Blowout risk: absolute spread ≥ 10-12, interacted with the player's 4th-quarter minutes share | Starter unders (pts/reb/ast) | Practitioner consensus only; mechanism is minutes | Medium | Spreads (ESPN), nba_api play-by-play for 4th-quarter minutes |
| 3 | Minutes volatility (CV of last-10 minutes) and returning from injury | Unders when the line is anchored to full-minute averages | ANECDOTE | Medium | nba_api game logs, injury report |
| 4 | Rebound props × eastward jet lag for home team (Leota: -1.29 reb diff) | Rebound unders for jet-lagged home team | STATISTICAL (strongest single coefficient found) | Medium-high | Schedule-derived time zones; our post-hoc rebound-under lead |
| 5 | Previous game in LA/NY or high nightlife index, within 24h (pre-registered) | Visitor defense worse: game OVER, opponent scorer overs; visitor's own props unclear | MARKET-TESTED (ATS, 2010-17 preprint, p<.01 discrete / p<.1 continuous) | Low-medium (niche), but thin and confounded | Schedule + OSM/QCEW |
| 6 | Westward multi-zone travel for evening game (Roy-Forest, McHill) | Away team eFG and ORB down: away player unders, game under (offset by worse defense) | STATISTICAL, replicated in 2 papers; direction disputed by a third | High | Schedule-derived |
| 7 | Early tip (before 5pm ET), esp. for teams from West/Central zones | Under | Tout MARKET-TESTED (53.4% since 2005, no method) | Medium | Schedule tip times |
| 8 | Second night of back-to-back for player ≥ 34 min/game | Minutes/pts unders | STATISTICAL weak; team-level ATS test old | High (our own test lost) | Schedule |
| 9 | Teammate return (high-usage teammate back after absence) | Usage drop: unders for the fill-in player | ANECDOTE/mechanical | Medium-high (books update fast on injuries) | Injury report, nba_api lineups |
| 10 | Revenge/homecoming game (first meeting after trade) | Contrarian under (public over bias, minimal real effect) | ANECDOTE (HSAC: "minimal" bump) | Low-medium | Transactions (B-Ref) |
| 11 | Visitor at altitude (Denver/Utah), 2nd half/4th quarter splits | Visitor unders, notably FT% | STATISTICAL (FT 74.9 vs 75.8%) | High at team level; untested for props | Arena elevation, play-by-play |
| 12 | Season-specific home edge (post-2020 drop to ~1.75 pts) | Road player props less depressed than old models assume | STATISTICAL | High (books refit) | B-Ref results |
| 13 | Long road trip (game ≥ 4 of trip) / 3-in-4, 4-in-6 | Unders | No direct study; heavily confounded with back-to-backs | Medium-high | Schedule |
| 14 | Opponent pace and position-adjusted defense | Slow/elite D: unders | STATISTICAL; core book input | Very high | nba_api team stats |
| 15 | Contract year | Small over (if any) | Mixed theses, sign unstable | High (absorbed into form) | Spotrac/B-Ref |

**Honest priors.** Items 1-3 are the only ones where I would expect any surviving edge. They work through *minutes* or *information timing*, which is exactly where thin prop markets are slowest. Items 4-8 have real but small effects in the published research and have probably been priced since well before our data starts. Items 10-15 are worth one pre-registered pass each, mainly to log the null result. The nightlife hypothesis is testable with free data. The one serious study points to worse *defense* (the over), not worse individual output. The Harden strip-club version has no evidential weight.
