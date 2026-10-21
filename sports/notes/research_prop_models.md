# Research: player-prop models, prop biases, and NBA/NFL market inefficiencies

Written 2026-10-20 from web research. Every factual claim has a URL. Each claim carries a tag:

- **[MT] market-tested**: shown to beat the vig or the closing line, preferably out of sample.
- **[IS] in-sample / historical record only**: a hit-rate or ROI table, no held-out test.
- **[A] anecdote / practitioner claim**: unsourced, or from a vendor that sells picks.

**Summary.** No peer-reviewed paper tests whether NBA or NFL *single-game player props* are efficient. I searched SSRN, arXiv, JSE/JQAS-style queries and practitioner sites. Everything specific to props is practitioner tracking ([IS]/[A]). The game-line literature is large. Its strongest results are either old (pre-2010) or closed after publication. That fits our own result: about 45 closing-line angles all lose the vig.

---

## 0. News today that changes the problem (2026-10-20)

- The NBA told its *authorized sportsbook partners* (on 2026-10-16) to restrict several markets. Restricted: parlays combining several unders on the same player; unders on players averaging or expected to play fewer than 12 minutes; certain alternate under lines with larger payouts; first-basket and similar "first" markets; and "player misses shot" markets. This adds to earlier bans on two-way/10-day players, turnovers, fouls and missed free throws. The rules do **not** cover Kalshi or Polymarket. "They do not constitute a blanket ban on player props or unders." ([SBR, 2026-10-20](https://www.sportsbookreview.com/news/nba-restricts-player-props-and-parlays-following-betting-scandals-oct-20-2026/))
- **What this means for us:** (a) our historical "unders on bench players" and alt-under results cannot be traded at US books from 2026-27. Filter them out of any forward test. (b) Licensed books now treat unders as an integrity risk, which may tighten under limits. (c) Kalshi sports prop markets fall outside the restrictions.
- NBA injury reporting (2025-26 rules): the report is due 5 p.m. local the day before the game, or 1 p.m. on the day for the second game of a back-to-back. Since December 2025 there is also a game-day report between 11 a.m. and 1 p.m. local. Reports update continually. ([official.nba.com](https://official.nba.com/nba-injury-report-2025-26/)) These fixed timestamps give an exogenous, timestamped news clock for event studies (see #3 below).

---

## 1. How serious bettors and academics build prop models

### 1a. Projection layer
- **Originators build one model per prop and bet where they disagree with the market.** Rufus Peabody "has created computer models for each of the hundreds and hundreds of different props" and bets "the ones that differ from his". Limits force him to split action across a network of partners, and individuals "can't even bet $10,000". ([The Ringer](https://www.theringer.com/2022/9/21/23363621/gamblers-super-bowl-props-rufus-peabody-nfl)) [A: profitable by reputation, no audited record]
- **Peabody's later approach:** he waits for the market to form, then targets mispriced props, alt lines and game-script scenarios. ([VSiN](https://vsin.com/the-vsin-daily/inside-the-mind-of-sharpest-of-the-super-bowl-prop-sharps/)) [A]
- **Daily-updating Bayesian/Kalman projections (NBA):** DARKO ("Daily Adjusted and Regressed Kalman Optimized") combines exponential decay and Kalman filtering, adjusts for aging, seasonality and garbage time, and blends two systems with ML. ([Power Rank podcast](https://thepowerrank.libsyn.com/kostya-medvedovsky-on-the-darko-nba-player-projections)) It is the best-known public example of "shrink per-minute rates toward a prior, update daily". I found no published test of DARKO against prop lines. [untested vs market]
- **The standard decomposition is minutes × per-minute rate, with adjustments.** Minutes come from rotation, injuries, blowout risk and rest. The rate gets usage, pace and opponent adjustments. Every practitioner source assumes this structure (e.g. [Rotogrinders situational projections](https://rotogrinders.com/articles/nba-situational-player-projections-dfs-betting-props-analysis-3-9-22-3768105)), but I found **no market-tested write-up**. Our rolling-10 form model (−3.3% on 130k props) fits the general pattern: a projection that does not beat the market's own projection is just paying vig.
- **Two academic results on what to optimise:**
  - *Calibration beats accuracy.* Walsh & Joshi (2024, *Machine Learning with Applications*) chose NBA models by calibration instead of accuracy. Reported ROI was +34.69% vs −35.17%, and +36.93% vs +5.56% in the best case, on one test season. ([Bath portal](https://researchportal.bath.ac.uk/en/publications/machine-learning-for-sports-betting-should-model-selection-be-bas/)) [IS: one season, no CLV, implausibly large; treat as a method lesson, not an edge]
  - *Decorrelate from the bookmaker.* Hubáček, Šourek & Železný (2019, *IJF*) penalise the model's correlation with the bookmaker's odds. They report positive cumulative profit on NBA moneylines 2007-2014. ([IDEAS](https://ideas.repec.org/a/eee/intfor/v35y2019i2p783-796.html), [CTU](https://ida.fel.cvut.cz/papers/hubacek2019exploiting.html)) [IS/historical backtest, old seasons] **Lesson for us:** a model that reproduces the line cannot win. Train on the residual against the line, or with the market line as an input and a decorrelation penalty.

### 1b. Distribution layer: turning a projection into P(over)
- **Counts are overdispersed.** Negative binomial beats Poisson for NBA team scoring in a Bayesian hierarchical model ([SciTePress 2023](https://scitepress.org/publishedPapers/2023/121591/pdf/index.html)). Poisson fits points per possession poorly ([arXiv 2406.09895](https://arxiv.org/pdf/2406.09895)). General overdispersion argument: ([MetricGate](https://metricgate.com/blogs/overdispersed-sports-scores-negbin/)). For player-level stats I found no validation study. Fit dispersion per stat × player-role and check calibration on held-out games.
- **Median vs mean is the key mechanism for unders.** Books are said to set lines near the *median*, while projections report the *mean*. For right-skewed stats (3PM, yardage, TDs) the mean sits above the median. Example: a shooter with mean 2.9 and median 2 can go under 2.5 more than half the time ([Stokastic](https://www.stokastic.com/articles/dfs-strategy/player-props-for-dfs-players)). OddsShopper example: mean 12.1 receiving yards against a 13.5 line, yet P(≤13) ≈ 70%. PFF calls football stats right-skewed because they are bounded at zero ([PFF](https://www.pff.com/news/nfl-betting-2022-midseason-player-prop-performance-review)). **Implication:** a model that compares a *mean* projection to the line will be biased toward overs. That alone could explain part of our −3.3%.
- **Parametric vs simulation.** Parametric fits (Poisson/NB/normal/lognormal) are fast. Monte Carlo game simulation captures correlation and game script ([OpticOdds](https://opticodds.com/blog/probability-paths-in-player-prop-modeling)). For alt-line ladders, de-vig each rung and fit a curve to the market's implied distribution; that market curve is the baseline to beat ([BetScope](https://betscope.substack.com/p/how-to-spot-weak-markets)). [A: methods, untested]

### 1c. Comparing to the market
- **De-vig method matters at the tails.** Proportional normalisation overstates longshots. Shin and power methods take more margin off longshots. Štrumbelj (2014, *IJF* 30(4):934-943) found Shin probabilities more accurate than basic normalisation (RPS) ([shin package, citing it](https://pypi.org/project/shin/); [implied vignette](https://cran.uni-muenster.de/web/packages/implied/vignettes/introduction.html)). For two-way props at −110/−110 the method barely matters. For −140/+110 props and alt lines it does. Recommendation: report results under both proportional and Shin, and pre-register one.
- **Which book is "sharp" for props?** No single book is sharp across the whole NFL prop board. Caesars and FanDuel are the most efficient US prop books. DraftKings, Caesars and MGM share a data source. Many offshore books use one line-setting tool. Caesars allows sharp accounts ≥$500 even on openers and moves fast, sometimes overshooting ([Establish The Run](https://establishtherun.com/understanding-the-current-ecosystem-of-nfl-player-props/)). FanDuel is used as the prop reference book, but its one-way markets carry large holds ([BetSmart](https://betsmart.beehiiv.com/p/efficient-prop-markets)). [A]
- **CLV as the judge.** CLV is industry consensus. The best quantitative check I found is a 14-season soccer study: Pinnacle pre-close prices as ground truth gave a profitable strategy. From 2023/24 onward realised ROI was 1.9% against an expected 4.2%, a warning that the benchmark itself can drift ([Networked substack](https://networked.substack.com/p/a-view-from-the-pinnacle)). [IS] For props there is no Pinnacle-equivalent closing benchmark. Use a de-vigged multi-book consensus of FanDuel, Caesars and Pinnacle where it is offered.

---

## 2. Evidence on specific prop strategies

| Claim | Best evidence | Tag |
|---|---|---|
| NFL single-game prop unders win more often | PFF: unders hit ~55% 2017-2019 regular seasons ([PFF via search summary](https://www.pff.com/news/nfl-betting-2022-midseason-player-prop-performance-review)); since 2020 blind RB receiving-yard unders +108 u, passing-yard unders +14 u. Receiving-yard overs are "really difficult" ([PFF 2022 midseason](https://www.pff.com/news/nfl-betting-2022-midseason-player-prop-performance-review)) | IS (vendor, units not ROI, no sample sizes in text) |
| NFL *season-long* prop unders | 4for4: blind unders 60% (2021) and 63% (2022), 604 bets, closing lines ([4for4](https://www.4for4.com/2023/preseason/key-winning-season-long-player-props)). Action Network 2017: 250-101-10 (71.2%) on 361 props ([Action](https://www.actionnetwork.com/nfl/nfl-betting-player-props-over-unders-season-long-percentages)) | IS; season-long is a different market (overround, injury risk skews to under) |
| NBA points-prop unders | Action Network: blind unders profitable at all four tracked books in one (unspecified) season; blind overs down 7.8-24 u at each ([Action](https://www.actionnetwork.com/nba/2024-25-nba-points-per-game-player-prop-bets-odds-picks-predictions)). Same source says rebounds/assists/3PM were "not as clearly tilted" | IS, season-long PPG props, not single-game |
| Public bets overs; books respond | Peabody: public favours overs; books nudge lines up near game time to attract under action. So "bet overs early, unders as late as possible" ([The Ringer ep. 3](https://www.theringer.com/2022/9/28/23375387/gamblers-super-bowl-episode-3-rufus-peabody), via search summary) | A, but from a credible originator. Directly testable on our data (#4) |
| Books *shade* props toward overs: by how much? | No measured figure for NBA/NFL props. The only academic test is a 2010 Syracuse study of CFL/Arena football: favourites and overs get >50% of dollars, with "slight evidence" of shading and no strong support for profit-maximising pricing ([Syracuse](https://surface.syr.edu/researchcenter/4)). PFF says the line is "constantly shaded toward the over" without data | A / weak |
| Prop markets got more efficient | BetSmart: props went from "one of the easiest markets to beat" to efficient ([BetSmart](https://betsmart.beehiiv.com/p/efficient-prop-markets)). 4for4 says 2022 was sharper. Peabody says Vegas books now copy each other ([VSiN](https://vsin.com/the-vsin-daily/inside-the-mind-of-sharpest-of-the-super-bowl-prop-sharps/)) | A |
| Injury news → stale teammate props | Articles describe props "slowly rising throughout the morning" after a teammate is ruled out, books pulling and reposting, and bettors front-running news ([Action](https://www.actionnetwork.com/nba/nba-player-prop-bets-picks-value-on-deandre-ayton-and-scottie-barnes-sunday-november-21)). No measurement of book update latency | A |
| Same-game-parlay correlation mispricing | Mechanism is real (books must price correlation; engines disagree on sign across books). But "majority of SGPs don't have positive EV" and books add extra margin ([OpticOdds](https://opticodds.com/blog/correlation-in-same-game-parlays), [JuiceReel](https://juicereel.beehiiv.com/p/the-dangers-of-same-game-parlays)) | A |
| Prop hold is higher than sides | Prop hold estimates conflict: 12-18% vs 6-8% ([Track360](https://track360.io/blog/prop-betting-affiliate-strategy-operator-margin-guide-2026), [nfelo](https://www.nfeloapp.com/tools/sportsbook-hold-calculator/)). SGP-driven Super Bowl LX hold was 31% in NY/NJ/PA vs 7.4% in Nevada ([H2GC](https://h2gc.com/news/general/super-bowl-wagering-declines-across-major-states-sportsbooks-record-elevated-hold)) | IS (realised, outcome-dependent) |

**Does any prop strategy have an out-of-sample or CLV test? No, none that I could find in public.** The under tendency shows up in every source and in our own data. Our data is the stronger evidence: 14.5k rebound-unders bets plus a consistent over/under asymmetry across five markets. Its size after vig is the open question.

---

## 3. Game-line inefficiency literature (NBA/NFL)

- **Levitt (2004), *Economic Journal* 114:223-246.** NFL 2001-02 bookmaker contest, 19,770 bets. Visiting favourites drew 68.2% of bets and won 47.8% ATS. Home underdogs won **57.7%** (n = 2,286). Over 1980-2001 (~5,000 games) visiting favourites covered 46.7%, and "a naive strategy of always betting against visiting favourites would actually have yielded positive profits". Bookmakers set prices so bettors win 49.45%, raising gross margin from 5.0% to 6.16% (+23%). ([PDF](https://www.stat.berkeley.edu/users/aldous/157/Papers/Levitt_Gambling_2004.pdf)) [MT historically, but the Kelly & Chen and Dmochowski items below show the shading is not universal or permanent]
- **Shading needs both a lean and a shade.** Dmochowski (2026) shows "a lean is thus worthless without shading". In 1,139 MLB games the public leans heavily to favourites, but **no price shading** was detected and realised margin ≈ hold ([arXiv 2609.06739](https://arxiv.org/abs/2609.06739)). [IS] This is the right frame for our over/under asymmetry: overs lose more than unders only if books actually shade.
- **NFL totals.** Paul & Weinbach: overs on high totals draw significantly more bets. Fading the public is profitable on spreads but **not** totals ([bepress](https://works.bepress.com/rodney_paul/12)). Kelly & Chen: unders at totals ≥47.5 won 59.7% in 2001-2009 and the edge **disappeared 2010-2018**; they attribute it to a change in book pricing ([AABRI](https://mobile.aabri.com/manuscripts/193138.pdf)). [MT then decayed]
- **NFL weather.** Borghesi (2008), 1984-2004: the totals market under-adjusts for adverse weather, with an *out-of-sample* win rate significantly above 52.38% ([IDEAS](https://ideas.repec.org/a/taf/apfiec/v18y2008i12p947-953.html)). [MT, old] Humidity and wind strategies rejected efficiency in a later study ([FIT](https://fitpublishing.com/node/3503)). Bet Labs data: 10+ mph games went under ~56% since 2003, but the edge goes to ~50/50 when the total already fell ≥1.5 points open→close ([Action Network](https://www.actionnetwork.com/how-to-bet-on-sports/nfl/nfl-over-under-betting-weather-trends-wind-hot-cold-temperatures)). [IS] So the market absorbs weather by the close, consistent with our corrected rain result (+8%, p = .11).
- **NFL rest.** Sung (2020, *IJSF*), 2002-2015: post-bye bias "remained profitable across the studied years"; the Thursday edge was not profitable ([SAGE](https://journals.sagepub.com/doi/pdf/10.32731/IJSF.152.052020.02)). [IS] A 2024 *Frontiers* paper finds the bye advantage (+2.2 pts) mostly vanished after the 2011 CBA; a bye-fade strategy hit 52% since 2011, below the vig ([Frontiers](https://www.frontiersin.org/articles/10.3389/frbhe.2024.1479832/full), [Footballguys](https://www.footballguys.com/article/2024-odds-and-ends-week07)). [decayed]
- **NFL other.** Shank: divisional games cover less often for the home team and go over less often ([Emerald](https://www.doi.org/10.1108/SEF-11-2018-0354)). Bettors overreact to recent results ([SSRN 3861231](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3861231)). Gray & Gray (1997): probit strategies profitable in-sample, but inefficiencies "dissipate over time".
- **NBA rest.** Ashman, Bowman & Lambrinos (2010, *JSE*), 19 seasons: home teams on the second night of a back-to-back against a rested visitor underperform ATS, more so after eastward travel ([IDEAS](https://ideas.repec.org/a/sae/jospec/v11y2010i6p602-613.html)). [IS] Inpredictable (2003-2010): 0-rest costs ~1.25 pts and spreads do adjust for it; no extra adjustment for 3-in-4 was found ([inpredictable](https://www.inpredictable.com/2012/02/nba-home-court-advantage-and-rest.html)). Action Network argues blind B2B fades no longer work ([Action](https://www.actionnetwork.com/nba/moore-dont-be-fooled-by-blindly-fading-teams-on-back-to-backs)). [A]
- **NBA totals.** 2000-2008, 10,325 games: overs 49.87%. No total band was significant after vig: "betting the over ... is a fair bet" ([Sport Journal](https://thesportjournal.org/article/nba-gambling-inefficiencies-a-second-look/)). [efficient]
- **Referees.** NBA referees favour home, trailing and series-trailing teams (Price, Remer & Stone 2012, *JEMS*). The abstract is about demand, not betting ([IDEAS](https://ideas.repec.org/a/bla/jemstr/v21y2012i1p271-300.html)). The only betting number is the Donaghy-era over rate (57% vs 44%), which a statistician called "a fairly naive calculation" ([Deseret](https://www.deseret.com/2007/8/16/20035625/could-a-statistical-model-detect-cheating-nba-ref)). Per-crew O/U tables regress to the mean ([Action](https://www.actionnetwork.com/nba/nba-finals-referees-betting-trends-warriors-raptors-officials-2019)). [A] **Do not prioritise.**
- **Home advantage after COVID.** Bundesliga books failed to reprice ghost games at first ([arXiv 2008.05417](https://arxiv.org/pdf/2008.05417)). Later work finds fast adjustment ([CEPR DP17273](https://cepr.org/publications/dp17273)). In the NFL 2020-21, spreads "adjusted quickly" ([Bryn Mawr thesis](https://scholarship.tricolib.brynmawr.edu/handle/10066/23566?show=full)). European basketball books anticipated the drop ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9517988/)). [MT then closed within weeks]
- **Momentum.** Moskowitz (2021, *JF*): open→close line moves show momentum, which the outcome fully reverses (overreaction). The effects are too small to beat costs ([NBER](https://conference.nber.org/conf_papers/f92072/f92072.pdf)). [MT: real but not exploitable] Relevant to props: we should check whether prop line moves also overreact (#4).
- **Over all of this:** a 14-season football study finds single-season inefficiencies are not persistent and are "likely to be observed even under full market efficiency" ([Bielefeld/JSE](https://pub.uni-bielefeld.de/record/2983980)). That justifies our held-out season and multiple-testing correction.
- **Prediction markets.** Kalshi, 300k+ contracts: a clear favourite-longshot bias. Takers lose ~32% on average, makers ~10%, and prices sharpen toward the close ([Bürgi, Deng & Whelan](https://www.ifo.de/en/cesifo/publications/2026/working-paper/makers-and-takers-economics-kalshi-prediction-market), [CEPR column](https://cepr.org/voxeu/columns/economics-kalshi-prediction-market)). [MT, all categories] On Polymarket (588M trades) the two-sided FLB pattern is "surprisingly absent in Sports" ([arXiv 2609.12878](https://arxiv.org/abs/2609.12878)). So FLB is not a free sports edge.
- **Soft vs sharp consensus.** Kaunitz, Zhong & Kreiner: betting outliers against the mean of 32 books (~0.5M soccer matches) was profitable in paper and real trading, and the accounts were "severely" limited within months ([MIT Tech Review](https://www.technologyreview.com/2017/10/19/67760/the-secret-betting-strategy-that-beats-online-bookmakers/)). [MT, with the limiting problem]

---

## 4. Data sources for historical props with timestamps

| Source | Coverage | Cost | Notes |
|---|---|---|---|
| The Odds API historical | Player props from **2023-05-03**, 5-min snapshots | 10 credits / region / market / event ([docs](https://the-odds-api.com/historical-odds-data/)) | Best timestamped source. Gives true open/close per book. ~3 NBA seasons + 3 NFL seasons |
| ESPN core API (already in use) | 2022+ multi-book props | free | `last_updated` is post-kickoff (our note), so it cannot timestamp the close |
| DunkAPI | NBA props open/close by book from May 2023 | 500 free credits, then $19/mo ([mcpservers](https://mcpservers.org/servers/noahowsh/dunkapi-mcp)) | Unverified vendor; cross-check against Odds API |
| SportsGameOdds | per-book open/close | Pro plan ([site](https://sportsgameodds.com/use-cases/historical-odds-data-api)) | paid |
| SportsDataIO / BALLDONTLIE | historical props | paid ([SportsDataIO](https://sportsdata.io/historical-odds)) | — |
| NBA official injury reports | timestamped PDFs, 5 pm / game-day deadlines | free ([NBA](https://official.nba.com/nba-injury-report-2025-26/)) | gives the news clock for event studies |
| Princeton Historical Sports Odds DB | game-level open/close spreads & totals | academic ([DSS](https://dss.princeton.edu/catalog/resource6741)) | probably no props |

No free bulk Kaggle dataset of historical props turned up.

**Avoiding look-ahead bias:**
1. Decide on features known at the bet timestamp. Box scores must come only from games finished before the bet. For injury status, use the official report's timestamp, not the "inactive" field from the final box score.
2. Use the line at the timestamp you could actually have bet. ESPN's post-kickoff `last_updated` must never be treated as a pre-game price.
3. Survivorship: props for players who were scratched get voided or pulled. Opening-line samples therefore over-represent players who played. Keep voids as voids.
4. Exclude markets restricted from 2026-27 (§0) from forward tests.
5. Keep the held-out season sealed, and correct for multiple testing. The Bielefeld paper explains why single-season wins appear under efficiency.

---

## 5. Prioritised top 10 to backtest next

Ordered by (evidence × feasibility with data we have) ÷ testing cost. "Other side" is who loses to us if the edge is real.

1. **Confirm rebound unders out of sample, then test the skew mechanism.**
   - *Hypothesis:* the under edge is largest where mean − median is largest: low lines, high dispersion, bench-to-starter bigs, 3PM, blocks/steals.
   - *Other side:* recreational over bettors, plus books that set lines near the mean or shade up.
   - *Data:* have it (opening, 2022+).
   - *Test:* sealed-season replication of rebound unders. Then a pre-registered model: ROI ~ (fitted NB mean − median)/line.
   - *Evidence:* **moderate.** Our own 14.5k bets plus PFF/4for4/Stokastic mechanism (§1b, §2). Not market-tested. Exclude players under 12 minutes (now restricted).
2. **Distribution-correct projection bet against the line's residual.**
   - Replace "rolling-10 mean vs line" with NB/empirical-quantile P(under) at the line, shrunk toward the de-vigged market probability. Bet only when |model − market| > threshold. Select by calibration, not accuracy (Walsh & Joshi), with a decorrelation penalty (Hubáček et al.).
   - *Other side:* same as #1.
   - *Data:* have.
   - *Evidence:* **weak** (method papers on moneylines, not props). Most likely reason our form model lost: it compared means to medians.
3. **Injury-news event study (NBA).**
   - *Hypothesis:* when a starter is ruled OUT on an official report, teammates' props reprice with a lag at some books.
   - *Other side:* slow books; bettors who bet before the news.
   - *Data:* Odds API 5-min snapshots from 2023-05 plus timestamped NBA injury reports.
   - *Test:* line and price at T+5/15/30/60 min vs close. Measure CLV, not just ROI.
   - *Evidence:* **anecdote**, but the mechanism is the most plausible source of information edge, and it is measurable as CLV quickly.
4. **Prop line-movement timing ("overs early, unders late").**
   - *Hypothesis (Peabody):* books push lines up near game time to attract under money, so under prices improve toward the close.
   - *Test:* open→close drift by side. Check whether the open→close move has Moskowitz-style overreaction, i.e. whether the close is worse calibrated than the open on the moved side.
   - *Data:* Odds API.
   - *Evidence:* **anecdote** (credible source). Cheap to check, and it decides whether our under results should be bet at open or close.
5. **Soft-book vs sharp-consensus props (port the soccer finding).**
   - De-vig a FanDuel/Caesars(/Pinnacle) consensus and bet DK/MGM/offshore outliers beyond a threshold.
   - *Other side:* books with slower or shared feeds.
   - *Data:* Odds API (multi-book).
   - *Evidence:* **strong for soccer** (Kaunitz et al.; our CLV +2.9%), **untested for props**. Limits will bind fast.
6. **Minutes-shock model (NBA).**
   - Project minutes from rotation news, the spread (blowout risk), back-to-back and foul-prone matchups. Bet points/PRA unders when projected minutes fall below the line's implied minutes (line ÷ per-minute rate).
   - *Other side:* bettors anchored on season averages.
   - *Data:* box scores + spreads + injury reports.
   - *Evidence:* **weak** (DARKO-style practice; spreads already adjust for 0-rest per inpredictable).
7. **NFL prop unders by category × game script.**
   - Pre-register PFF's in-sample winners on ESPN 2022+ props, with a sealed season: RB receiving-yard unders; passing-yard unders; rushing unders when the team is favoured; receiving/passing unders when the team is an underdog.
   - *Other side:* fantasy-driven over bettors.
   - *Evidence:* **weak** (vendor units, no samples given).
8. **NFL weather → player props lag.**
   - *Hypothesis:* the game total absorbs wind/rain (Action Network: the edge goes to ~50/50 once the total drops ≥1.5), but QB/WR yardage props reprice less.
   - *Test:* condition on the total having moved down for weather; bet passing/receiving unders.
   - *Data:* ESPN props + weather + total movement.
   - *Evidence:* **moderate for totals historically** (Borghesi out of sample, pre-2005), **none for props**.
9. **Alt-line ladder tails.**
   - Fit the market-implied distribution from main lines (Shin/power de-vig), then price alt rungs. Look for favourite-longshot-type overpricing of large alt overs (sell side) and of alt unders.
   - *Other side:* longshot-loving recreational bettors.
   - *Data:* Odds API alt markets.
   - *Evidence:* **moderate in general** (FLB on Kalshi/takers), **weak for props**. Large-payout alt *unders* are now restricted at NBA partner books.
10. **Kalshi player-prop / game markets as maker.**
    - Post resting orders on the favourite side of sports contracts using a sportsbook-derived fair value.
    - *Other side:* takers, who lose ~32% vs makers ~10% across Kalshi.
    - *Data:* Kalshi API history (already in use).
    - *Evidence:* **strong that the FLB/taker penalty exists on Kalshi overall**, but **weak for sports specifically** (Polymarket: pattern absent in sports). Not covered by the NBA restrictions.

**Deprioritise:** referee crews (anecdote only), blind B2B or bye fades (decayed after publication), NBA totals bands (efficient), and post-COVID home-court angles (closed within weeks).

**Overall:** the public evidence does not show that anyone has a market-tested, post-vig prop edge we can copy. The most defensible paths: (a) the mechanism behind the under asymmetry, measured on our own data with a sealed season; (b) news latency measured as CLV; (c) cross-book consensus, which already worked for us in soccer.
