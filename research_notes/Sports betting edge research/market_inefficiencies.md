# Sports-betting market inefficiencies beyond the closing line (NFL, NBA, MLB, NHL, college football): game lines and totals

Scope note: game sides/totals/moneylines. Props belong to another researcher; prop-relevant items are flagged inline. Several primary PDFs were read directly (Levitt 2004, Moskowitz JF 2021, Lopez & Bliss 2024). Items marked [snippet] come from search-result summaries or abstracts only, so treat their exact numbers as unverified until someone reads the paper.

## 1. Favourite-longshot bias (FLB) in moneyline markets, by league

### Takeaway
The direction depends on the league. Pro MLB and NHL moneylines historically showed a *reverse* FLB (underdogs underpriced), and in MLB the bias shows up in opening odds but disappears by the close. College basketball and football moneylines show a classic FLB (longshots overpriced). NFL, NBA and NHL moneylines now look close to calibrated. I found no evidence that an FLB on closing lines is profitable after vig in any US pro league in recent data.

### Cited Findings
- Woodland & Woodland (1994, *Journal of Finance*) found that the classic racetrack FLB "exists in reverse" in MLB moneylines: underdogs are underbet. This is old, pre-internet data. — [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1994.tb04429.x)
- Woodland & Woodland (2001, *Southern Economic Journal*), NHL 1990–1996: the market was "somewhat inefficient", and simple longshot strategies were profitable. That is a 7-season, pre-2000 sample. [snippet] — [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/j.2325-8012.2001.tb00385.x)
- In MLB, the FLB "is clearly present in opening odds but disappears in closing odds", and it is stronger for home games (Harvard thesis, "Swing and a Miss"). [snippet; I couldn't fetch the full text] This matches the Moskowitz finding (section 2) that open-to-close moves correct mispricing. — [Harvard DASH](https://dash.harvard.edu/bitstreams/81e73570-084b-4e39-84d3-ccb4649c61ab/download)
- In college basketball and college football fixed-odds moneyline markets, a classic FLB "persists", unlike the reverse FLB in pro MLB and NHL (*Quarterly Review of Economics and Finance*, vol. 63, 2017, pp. 233–239). I couldn't access effect size or ROI (403). [snippet] — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1062976916000041); [RePEc](https://ideas.repec.org/a/eee/quaeco/v63y2017icp233-239.html)
- Weak-form efficiency tests (Robbins, *American Journal of Management* 2023) found NFL, NBA and NHL moneylines "mostly accurate with little evidence of unreliability of individual prices". [snippet] — [ECU PDF](https://myweb.ecu.edu/robbinst/PDFs/Weak%20Form%20Efficiency%20in%20Sports%20Betting%20Markets.pdf)
- A survey of FLB explanations (risk-loving bettors, misperception of probabilities, bookmaker margin structure, insider trading): — [ResearchGate](https://www.researchgate.net/publication/228884358_The_Favorite-Longshot_Bias_An_Overview_of_the_Main_Explanations); Cain et al. 2003 on how the FLB varies with bookmaker margins and insider trading across markets — [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-8586.00174)

### Inferences
- For a modern closing-line strategy, an FLB is most plausible in college moneylines, which have thinner markets and more retail volume. In MLB, any FLB seems to live in the open and get corrected by the close. That makes it a timing/CLV play, not a closing-line feature.
- The "reverse FLB" in MLB and NHL may be partly a by-product of how margins are split across favourite and underdog prices, so how you remove the vig matters (multiplicative vs additive vs Shin).

### Gaps
- I found no post-2015 peer-reviewed FLB estimate by league with ROI and n at Pinnacle/sharp closing prices.
- I couldn't extract the effect size for the college FLB paper (paywalled).

## 2. Recency / hot-hand / momentum overreaction (Moskowitz; Paul & Weinbach; Levitt)

### Takeaway
This is the strongest rigorous evidence. Moskowitz (JF 2021), using more than 100k contracts, found that momentum signals push prices from open to close and the game outcome **fully reverses** those moves. The pattern is statistically strong but economically small: it does **not** survive transaction costs. Older NBA and NFL "hot hand" overreaction findings (Paul & Weinbach; Woodland & Woodland) are consistent with this, but are mostly in-sample and pre-2010.

### Cited Findings
- Moskowitz, "Asset Pricing and Sports Betting" (*Journal of Finance* 76(6), 2021):
  - Data:
    - Covers/SportsDirect spreads: NFL 1985–2013, NBA 1999–2013, NHL 1995–2013, MLB 2004–2013.
    - SportsInsights opening and closing lines for spread, ML and O/U, 2005 to May 2013, with bet counts from Pinnacle, 5Dimes and BetCRIS.
  - Results: "strong evidence of momentum and weaker evidence of value effects that move prices from the open to the close of betting, which are then completely reversed by the game outcome." Size has no predictive power.
  - Magnitude: the returns "are not large enough to overcome transactions costs"; "the returns to momentum and value are easily wiped out by trading costs." (Read directly from the PDF.) — [NYU Stern PDF](https://w4.stern.nyu.edu/finance/docs/pdfs/Seminars/moskowitz_seminar_paper.pdf); [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2635517)
  - Reported magnitudes: the momentum Q5−Q1 strategy earns 4.73%/yr (Sharpe 0.73, t = 5.32). A strategy capturing both the momentum move and its reversal earns 13.07%/yr (t = 3.66, Sharpe 0.47). These figures come from search summaries of the paper; I didn't verify the table numbers in the PDF. — [JF version PDF](https://spinup-000d1a-wp-offload-media.s3.amazonaws.com/faculty/wp-content/uploads/sites/3/2021/08/AssetPricingandSportsBetting_JF.pdf)
  - Implication: the **closing** line has already absorbed the overreaction and the result reverses it. So "fade the move" is roughly breakeven before vig, and "bet the open in the direction of momentum" is the only side that captures anything, which is the CLV logic.
- Paul & Weinbach found a "hot hand" bias in the NBA: bettors back teams that recently beat the spread. [snippet] — [Bettor Misperceptions in the NBA, JSE 2005](https://journals.sagepub.com/doi/10.1177/1527002504266861)
- Shank (2018, *Journal of Economics and Finance*), "Is the NFL betting market still inefficient?":
  - Sportsbooks undervalue home teams, "particularly when they are substantial underdogs or have failed to cover spreads in recent games".
  - Totals show a quadratic (extreme-line) bias, and home teams that previously went over tend to go over again.
  - Both markets are "statistically inefficient". This is statistical, not after-vig profitability, as far as the abstract states. — [RePEc](https://ideas.repec.org/a/spr/jecfin/v42y2018i4d10.1007_s12197-018-9431-4.html)
- "Behavioral biases in the NFL gambling market: Overreaction to news and the recency bias" (*Journal of Behavioral and Experimental Finance* 2021): I couldn't access it (403). Title only. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2214635021000666)
- "Week 2 bias" / prior-playoff-team bias: teams that made the playoffs the previous season are over-favoured in Week 1. Median line error is about 6 points, and a strategy over 2004–2011 returned more than 25% per game. The error is not significant from Week 2 onward. Small sample (one week per season); in-sample. [snippet] — [ResearchGate](https://www.researchgate.net/publication/279194164_Exploiting_Week_2_Bias_in_the_NFL_Betting_Markets); related: "holdover bias" — [ResearchGate](https://www.researchgate.net/publication/263080729_Inefficient_pricing_from_holdover_bias_in_NFL_point_spread_markets)
- Counter-evidence that momentum *bettors* lose. "Gambling on Momentum" (arXiv 2211.06052) studied Bundesliga in-play betting: 612 matches from 2017/18–2018/19, with the focus on 212 matches tied 1-1 before the 85th minute. Bettors put about 40% more stake on teams that had just equalised, but there was "no evidence that such perceived momentum matters" for outcomes, and books didn't shade odds. Momentum-following ROI ranged from −7.4% to −23.3%. Soccer, not US; included as mechanism evidence. — [arXiv](https://arxiv.org/html/2211.06052.pdf)
- Time-series momentum in sports betting (*Journal of Behavioral and Experimental Finance*, 2023): title only, not read. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S221480432300040X)

### Inferences
- If bettors overreact, the closing line already contains the overreaction and the outcome reverses it. A model feature that measures recent ATS performance or streaks therefore has a theoretically signed effect (fade). Moskowitz shows its size is below the vig in liquid markets. Expect at most a small contribution as one of several features, not a standalone edge.

### Gaps
- I found no post-2013 replication of Moskowitz on modern (post-PASPA, post-2018) closing lines.
- I couldn't extract per-sport splits of the momentum effect from the PDF text.

## 3. Weather (wind, rain, temperature) and totals

### Takeaway
Borghesi (*Applied Financial Economics*, 2008), using NFL 1984–2004, found the market under-adjusts totals for heat, wind and rain. A weather-aware strategy beat 52.38% out of sample. That is old data. Practitioner claims that the market today still under-adjusts for moderate (15–20 mph) wind, and that Wrigley wind is mispriced, are unverified trend counts. Books now delay posting Wrigley totals until game-day forecasts are in, which suggests at least partial pricing.

### Cited Findings
- Borghesi, "Weather biases in the NFL totals market" (*Applied Financial Economics* 18(12), 2008), NFL 1984–2004: "heat, wind and rain reduce point production"; "bettors underestimate this effect"; the weather strategy has an "out-of-sample win rate significantly above the 52.38% profitability threshold". I couldn't access the exact win rate or n (403). — [T&F](https://www.tandfonline.com/doi/full/10.1080/09603100701335432); [ResearchGate](https://www.researchgate.net/publication/24071150_Weather_Biases_in_the_NFL_Totals_Market)
- The same author attributes the NFL **home-underdog** effect to weather and late-season bias, i.e. cold-weather home teams in late season. — ["Weather Biases in the NFL Betting Market: Explaining the Home Underdog Effect"](https://www.researchgate.net/publication/267995251_Weather_Biases_in_the_NFL_Betting_Market_Explaining_the_Home_Underdog_Effect_Weather_Biases_in_the_NFL_Betting_Market_Explaining_the_Home_Underdog_Effect); ["The late-season bias", *Applied Economics* 39(15)](https://doi.org/10.1080/00036840600690314)
- Practitioner claims (pick-seller/media; unverified, no out-of-sample test):
  - Outdoor NFL games with wind under 8 mph went over 52.3% across 2,345 games, vs a 48.6% baseline for all outdoor games.
  - The market "systematically under-adjusts for the 15–20 mph wind range".
  - Source: [Sharp Football Analysis](https://www.sharpfootballanalysis.com/betting/nfl-weather-betting/), a pick-selling site. Flag as anecdotal.
- Wrigley (MLB), Action Network / SportsInsights:
  - Wind blowing out vs in is worth +4.04 runs per game and +2.06 HR per game (season-adjusted).
  - Overs with wind out went 77-44-4 (63.6%); unders with wind in went 99-69-12 (58.9%).
  - Books post Wrigley totals late, on game day, to see the forecast.
  - These are in-sample trend counts from a pick-tool vendor. No peer review or out-of-sample test was found, and the sample periods are unclear. — [Action Network MLB weather primer](https://www.actionnetwork.com/mlb/mlb-weather-betting-over-unders-wind-heat-trends-data); [Action Network Wrigley](https://www.actionnetwork.com/mlb/brewers-cubs-sharp-betting-over-under-weather-wrigley-wind-friday-may-10)
- The Pinnacle article on NFL totals and weather wasn't extractable through fetch. — [Pinnacle](https://www.pinnacle.com/betting-resources/en/football/nfl-points-totals-and-the-effect-of-the-weather/5zn2av8e4w2g9sf9)

### Inferences
- The robust part is that weather lowers scoring, which is not in dispute. Whether the close prices it today is untested in the peer-reviewed literature after 2004. The best remaining edge is likely in **forecast-update timing** (betting before the line reacts to a forecast change) and in in-game rain that isn't in the pregame forecast, not in closing-line weather.
- Prop note: wind and rain affect passing yards, FG and kicking props more directly than game totals. Prop markets may adjust more slowly, but I didn't find evidence on this.

### Gaps
- I found no peer-reviewed study of weather pricing in college-football totals, or a post-2010 NFL replication with closing lines.
- There is no rigorous MLB wind/temperature totals study with a holdout; only vendor trend counts.

## 4. Rest, travel, time zones, circadian, bye weeks

### Takeaway
Lopez & Bliss (2024) is the best modern evidence. The NFL bye advantage collapsed after the 2011 CBA, from +2.2 to +0.3 points, but the market *raised* its bye valuation (+0.39 to +0.97). Fading teams with a bye edge went about 52% ATS since 2011, which is **not profitable after vig**. The famous West-Coast-night-game circadian effect (66% ATS, 1970–2011) is a single-study, in-sample result published in *Sleep*. NBA and NHL back-to-back "under-adjustment" claims are practitioner-only.

### Cited Findings
- Lopez & Bliss (with co-author), "Bye-bye, bye advantage" (*Frontiers in Behavioral Economics*, 2024; arXiv 2408.10867). Read from the PDF.
  - Sample: 5,679 regular-season NFL games, 2002–2023.
  - Bye advantage: +2.21 pts per game pre-2011 (95% CrI 0.61–3.80) vs +0.31 post-2011 (CrI −1.01 to 1.64).
  - Market-implied bye value: +0.39 pre-CBA (CrI 0.00–0.78) vs +0.97 post-CBA (CrI 0.65–…).
  - Probability that the market's bye value rose after 2011 = 0.988.
  - ATS: home teams with a bye edge covered 55.8% and road teams 56.9% in 2002–2010. In 2011–2023, home teams with a bye edge covered only 44.6% (road 52.7%).
  - MNF rest edge: the market prices about +0.37 pts while the estimated true effect is not significant.
  - Mechanism: extra *practice* days, not days off, drive the advantage.
  - — [arXiv](https://arxiv.org/pdf/2408.10867); [Frontiers](https://www.frontiersin.org/journals/behavioral-economics/articles/10.3389/frbhe.2024.1479832/full)
  - A summary of the paper says that backing the bye-disadvantaged team won about 52% ATS since 2011, "not profitable after accounting for vig". [summary snippet; consistent with the cover rates above] — [Frontiers](https://www.frontiersin.org/journals/behavioral-economics/articles/10.3389/frbhe.2024.1479832/full)
- Circadian effect: Smith et al. (*Sleep*, Dec 2013), NFL 1970–2011, games starting after 8 pm ET, West Coast vs East Coast teams. West Coast teams covered the Vegas spread in **66%** of games, by an average of 5.26 points. There was no effect in day games. In-sample; small n (prime-time East–West matchups only). I found no post-publication ATS replication. — [ScienceDaily](https://www.sciencedaily.com/releases/2013/11/131127115357.htm); [AASM](https://aasm.org/nfl-teams-on-west-coast-may-have-circadian-edge-in-night-games/); earlier version: Smith et al. 1997 *Sleep* — [OUP](https://academic.oup.com/sleep/article-abstract/20/5/362/2732132)
- NBA back-to-backs: practitioner claims that books shade 1–2 pts but the true drop is "slightly larger", especially with travel. No rigorous ATS test was found; source quality is low. — [Wager Theorem](https://thewagertheorem.com/nba-back-to-back-betting-analysis/)
- NHL back-to-back goalies: there is conflicting practitioner evidence. One analysis of 1,409 starts found 2.79 vs 2.77 goals against (no fatigue tax, because coaches filter starts). — [DataStreak](https://thedatastreak.com/insights/nhl-goalies-back-to-back). Another, unsourced, claims a save-percentage drop of 1.8–3.2 points. — [myhockeybet](https://www.myhockeybet.com/articles/how-back-to-back-road-trips-alter-goalie-save-percentages-and-shift-nhl-totals-markets/). Lopez & Bliss cite NHL work linking worse rest differential to fewer standings points, partly through goalie save percentage on zero days' rest. — [arXiv](https://arxiv.org/pdf/2408.10867)

### Inferences
- Rest is a case of the market *over*-learning. Once a well-known effect is priced, it can be overpriced after the underlying effect shrinks, as with the NFL bye after 2011. A model that estimates rest effects from recent data, and doesn't rely on folklore, can capture the gap. The size, though, is about 0.6 pts, roughly 2% cover probability, so it is below the vig on its own.
- Circadian or time-zone effects are worth testing as a feature on modern data. The 66% figure almost certainly includes small-sample luck plus a real but smaller effect.

### Gaps
- I found no peer-reviewed NBA/NHL back-to-back or travel ATS study with closing lines and post-2015 data.
- I found no post-2013 replication of the circadian ATS result.

## 5. Referee / umpire tendencies on totals

### Takeaway
I found no peer-reviewed study showing that umpire or referee assignment predicts totals beyond the closing line. All the evidence found is practitioner trend tables (Covers O/U by umpire) with no holdout. Umpire identity is public before game time, so it is likely priced at sharp books.

### Cited Findings
- Practitioner claim: zone size moves scoring by about 0.5 runs per game or more, and called-strike-rate spreads across umpires are about 8–10%. No out-of-sample betting test. — [Core Sports Betting](https://www.coresportsbetting.com/how-mlb-umpire-tendencies-affect-over-under-bets/); [NBC Sports](https://www.nbcsports.com/betting/mlb/news/article-odds-and-ends-how-handicap-mlb-umpire-betting-stats)
- Covers publishes O/U records by home-plate umpire. These are raw records that don't adjust for the posted total or for regression to the mean. — [Covers](https://www.covers.com/mlb/over-under-betting-tips)
- Prop note: umpire zone is used for **strikeout props** (pitch mix × zone). This is a pick-site claim and unverified. — [tonyspicks](https://www.tonyspicks.com/2026/05/14/mlb-strikeout-props-reading-pitch-mix-and-umpire-strike-zones-together/)

### Inferences
- Umpire O/U win-loss records are noisy (about 30 games per umpire-season) and are the classic setting for survivorship and multiple-comparisons artefacts. Treat umpire effects as a feature to estimate with shrinkage, not as a trend.

### Gaps
- I didn't search for NBA/NFL referee totals studies with closing lines, and none came up incidentally.

## 6. Injury/lineup news, line movement, steam, reverse line movement, public-betting % contrarian

### Takeaway
- Line movement contains information. Moskowitz shows open-to-close moves are partly noise that the outcome reverses, and in low-attention NFL games lines move more (Krieger & Davis 2024).
- Public-betting-% contrarian evidence comes mostly from Paul & Weinbach, using Sportsbook.com data from the 2000s.
- Levitt (2004) is the canonical evidence that books don't balance action and instead shade against bettor biases.
- Modern public "betting %" feeds are proprietary and noisy. No post-publication out-of-sample test was found.

### Cited Findings
- Levitt (*Economic Journal*, 2004). Read from the PDF.
  - Data: 285 entrants in an online-sportsbook NFL handicapping contest, 2001–02 season, $250 entry, 5 ATS picks a week × 17 weeks. About 20k picks, but only 236 games.
  - Visiting favourites drew 68.2% of picks; home favourites 56.1%.
  - Pick win rates: visiting favourites 47.8%, home favourites 49.1%, **home underdogs 57.7%** (n = 2,286 picks).
  - In the median game about 58% of picks went to the home team. In about 10% of games more than 80% of bets were on one side.
  - Books shade prices against bettor biases instead of balancing, which raises their gross margins by 20–30%.
  - Levitt cautions the result rests on "only 236 games of a single season".
  - — [Levitt PDF](https://pricetheory.uchicago.edu/levitt/Papers/LevittWhyAreGamblingMarkets2004.pdf)
- Paul & Weinbach, using Sportsbook.com betting-dollar shares:
  - Bet shares are highest on large favourites, road favourites and high totals.
  - Betting the underdog when more than 70% of bettors are on the favourite was profitable.
  - No profitable contrarian strategy was found in totals by betting share.
  - [Summary; exact n, years and win rates not verified] — [Paul & Weinbach, Tests of the Levitt model, IJSF 2008](https://kylewoodward.com/blog-data/pdfs/references/paul+weinbach-international-journal-of-sport-finance-2008A.pdf); [NFL Levitt hypothesis tests](https://ideas.repec.org/a/taf/apeclt/v18y2011i2p193-197.html); [NBA price setting](https://www.researchgate.net/publication/23534378_Price_Setting_in_the_NBA_Gambling_Market_Tests_of_the_Levitt_Model_of_Sportsbook_Behavior)
- Informed-vs-uninformed money ("Information asymmetry in the NFL gambling market: inside information versus informed bettors", *Journal of Behavioral and Experimental Finance* 2022):
  - When more than 60% of money is on the visitor and the book does *not* move the spread, the visitor is more likely to cover. This is the closest academic analogue to "reverse line movement" / a book holding against public money.
  - When books move the line toward the majority, the market is efficient.
  - [snippet; abstract not read in full] — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2214635022000806)
- Krieger & Davis (*Journal of Economics and Finance* 48(2), 2024), 3,756 NFL games 2007–2021: line moves are larger and more frequent in less-visible games (lower TV audience, shared kickoff, smaller fanbase). They interpret this as limited oddsmaker attention, so openers in low-attention games are less efficient. No betting-ROI test was reported in the abstract. — [RePEc](https://ideas.repec.org/a/spr/jecfin/v48y2024i2d10.1007_s12197-023-09656-5.html)
- Paul & Weinbach, "An Analysis of the Last Hour of Betting in the NFL" (2011): title only, not read. — [SAGE](https://doi.org/10.1177/155862351100600403)
- Opening vs closing accuracy: one study finds the NFL opening line has a lower MSE than the close, but the difference is not significant. A Mani (2018) claim that openers are more accurate is cited secondhand. — [arXiv 1211.4000](https://arxiv.org/pdf/1211.4000)
- A Claremont thesis (CMC), NFL moneylines 2020–2024 with high-frequency odds: odds shifts are predictable, and the market "overvalued home-field advantage in close games", making an away-team system profitable. This is a student thesis; I couldn't access out-of-sample status or n (403). Treat as unreplicated. — [Claremont](https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=5145&context=cmc_theses)
- Practitioner claim that CLV-positive bettors earn 2–3x the ROI of others. No data source; marketing copy. — [nflbettingstrategies](https://nflbettingstrategies.com/articles/nfl-line-movement-analysis/)

### Inferences
- Line *movement* is the market correcting itself, so it is information about the opener, not the close. Moskowitz implies that following momentum-driven movement after the close has negative expectation, while beating the move (CLV) is the right test of a model.
- The "book doesn't move despite lopsided public money" signal (2022 paper, Levitt) is the one contrarian construct with academic support. Retail betting-% data quality is the binding constraint.

### Gaps
- I found no rigorous public test of "steam" or reverse line movement on modern multi-book data.
- I found no study of injury-news timing and line efficiency for US leagues in this search.

## 7. Home underdogs, divisional games, primetime unders, high totals, key numbers

### Takeaway
The NFL home-underdog effect is the textbook example of **post-publication decay**: 58.1% (1973–79), 52.5% (1981–96), and about 49–51% in the 2000s–2010s. The "under on high totals" effect (Paul & Weinbach 2002: 58.7% for totals ≥ 47.5, 1979–2000) is old. Totals have risen a lot since, and I found no modern replication. Primetime-over bias in bettors is documented in college football but only for nationally televised games.

### Cited Findings
- NFL home underdogs ATS by era:
  - 58.1% in 1973–1979.
  - 52.5% in 1981–1996.
  - 53.5% in 2002–2011.
  - No consistently profitable variant in 2009–2018 (UNI thesis).
  - [Secondary summary of literature] — [UNI ScholarWorks](https://scholarworks.uni.edu/cgi/viewcontent.cgi?article=1538&context=hpt)
- 2015 home dogs went 45-42 ATS (51.7%). Over "the last 12 years", the home-underdog ATS rate was 48.9%. This is a pick-site article and the window is unclear. — [Doc's Sports](https://www.docsports.com/2016/nfl-betting-home-underdogs-516.html)
- Levitt 2001 contest picks: home underdogs won 57.7% (single season, 236 games). — [Levitt PDF](https://pricetheory.uchicago.edu/levitt/Papers/LevittWhyAreGamblingMarkets2004.pdf)
- High-total unders (Paul & Weinbach 2002), NFL 1979–2000: unders won 51% overall and 58.7% when the total was ≥ 47.5. Bettors prefer overs. [secondary summary] — [The Sport Journal](https://thesportjournal.org/article/factors-affecting-scoring-in-nfl-games-and-beating-the-overunder-line/); [AABRI, bettor biases in NFL totals](http://www.aabri.com/manuscripts/193138.pdf)
- College-football totals: the bias toward betting the over is significant only for nationally televised games on major networks (Paul & Weinbach). This is about bet volume, not necessarily returns. — [Sports Insights academic summary](https://www.sportsinsights.com/academic-research-on-football/); [Paul & Weinbach on TV ratings and the betting market](https://www.researchgate.net/publication/282698703_The_Betting_Market_as_a_Forecast_of_Television_Ratings_for_Primetime_NFL_Football)
- Shank 2018: there is a quadratic extreme-totals relationship (higher cover rates at extreme closing totals), and home underdogs are undervalued. — [RePEc](https://ideas.repec.org/a/spr/jecfin/v42y2018i4d10.1007_s12197-018-9431-4.html); Shank 2019, "NFL Betting Biases, Profitable Strategies, and the Wisdom of the Crowd" (*International Journal of Sport Finance*) — [SAGE](https://journals.sagepub.com/doi/10.32731/IJSF.141.022019.01)
- Divisional games and key numbers: I found no rigorous source in this search.

### Inferences
- These are the most-published anomalies, and they are the ones that have decayed. Don't build features around them unless a fresh holdout supports it.

### Gaps
- I found no rigorous source on divisional-game ATS or on key-number pricing (the value of half-points through 3/7) with modern data.

## 8. Overall: is anything persistent and profitable after vig, post-publication?

### Takeaway
I found **no peer-reviewed study showing a persistent, post-publication, after-vig profitable strategy on US closing game lines.**
- The best-identified effects (momentum overreaction, NFL rest mispricing) are real but smaller than the vig.
- Published anomalies decayed:
  - Home underdogs.
  - The pre-2011 NFL bye advantage, which reversed into over-pricing.
  - Week-1 prior-playoff bias, which corrects by Week 2.
- Claimed profitable strategies are old (1979–2004), small-n, in-sample, from student theses, or from pick sellers.

### Cited Findings
- Moskowitz: mispricing exists but is "not large enough to overcome transactions costs". — [NYU PDF](https://w4.stern.nyu.edu/finance/docs/pdfs/Seminars/moskowitz_seminar_paper.pdf)
- Lopez & Bliss: about 52% ATS fading the bye edge since 2011, below the 52.38% break-even. — [Frontiers](https://www.frontiersin.org/journals/behavioral-economics/articles/10.3389/frbhe.2024.1479832/full)
- NFL spreads 1992–2012 are "efficient in the aggregate sense", an unbiased minimum-variance estimator, though some strategies were profitable in-sample. [snippet] — [UNI Fohey](https://scholarworks.uni.edu/hpt/93/)
- Home-underdog decay from 58.1% to about 49–51%. — [UNI ScholarWorks](https://scholarworks.uni.edu/cgi/viewcontent.cgi?article=1538&context=hpt)
- Borghesi's weather strategy is the one with an explicit out-of-sample win above break-even, but only on data through 2004. — [T&F](https://www.tandfonline.com/doi/full/10.1080/09603100701335432)
- A 2026 arXiv paper on bookmaker profit as "the public's prediction error" (profit-bias identity) is potentially relevant, but I didn't read it. — [arXiv 2609.06739](https://arxiv.org/pdf/2609.06739)

### Inferences
- The surviving opportunity is structural, not a single feature:
  - Timing: bet openers or less-visible games before correction.
  - Information the closing line can't hold: in-game or late weather, late lineup news.
  - Thin markets: college moneylines, the MLB open.
  - Each needs CLV-based validation and a sealed holdout.
- Any one-feature backtest above 52.4% on a few hundred games should be presumed to be noise plus multiple testing. The published record is a series of in-sample wins followed by decay.

### Gaps
- I found no systematic post-publication decay study across sports-betting anomalies, analogous to McLean & Pontiff in equities.
- No rigorous NBA/NHL/college totals anomaly literature was surfaced in this pass.
- Unabated/Pinnacle practitioner data analyses weren't retrievable (fetch failures).
