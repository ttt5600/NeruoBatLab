# Historical Sports-Betting Odds Data Sources (game lines + player/team props), as of Oct 2026

Status legend: **[V]** verified directly this session (page fetched / file listing checked / API metadata read, 2026-10-12); **[S]** from a search snippet or third-party/competitor page, not confirmed on the vendor's own page; **[U]** unverified / not found.

## Q1. Which FREE sources give 4+ (ideally 10+) seasons of game lines, and what exactly is in them?

### Takeaway
Free game-line history of 10+ seasons is easy to get (SBRO 2007-08 to roughly 2021-23 for six leagues, nflverse 1999+, aussportsbetting 2006+, spreadspoke NFL 1979+). Free **prop** history of 10+ seasons does not exist anywhere I could find. The longest free prop archives start in 2022-2023 (ESPN, which we already have) or in 2026 (PropLine on Kaggle).

### Source table (free)

| Source | Leagues | Years | Markets | Open/close/snapshots | Books | Format | Terms | Status |
|---|---|---|---|---|---|---|---|---|
| SportsbookReviewsOnline (SBRO) | NFL, NBA, NHL, MLB, NCAAF, NCAAB | NFL 2007-08 to 2021-22 (+ preseason); NBA 2007-08 to 2022-23; NHL 2007-08 to 2022-23; NCAAF 2007-08 to 2022-23; MLB 2010-2021; NCAAB 2007-08 to 2020-21 | Spread/total/ML, 2H lines (NBA), quarter scores. No props, no team totals | Open + Close only | Unnamed single line (not per book) | MLB and NCAAB as .xlsx; the NFL/NBA/NHL/NCAAF pages are now **HTML tables** (no xlsx links found) | No explicit license found [U] | **[V] Online. No seasons added after 2022-23, so the archive looks frozen** |
| nflverse (`load_schedules`) | NFL | 1999+ | Spread, total, ML (closing) | Close only | Consensus | CSV/parquet | Open (nflverse) | Already in use |
| spreadspoke (Kaggle `tobycrabtree/nfl-scores-and-betting-data`) | NFL | Scores 1966+, lines 1979+, updated 2026-02-11 | Spread (favorite), total | Close-ish | Mixed sources (repole.com 1978-2013; sportsline/aussportsbetting 2014-2024) | CSV | CC BY-NC-SA 4.0 | [V] |
| aussportsbetting.com /data | NFL (2006+), plus NBA/NHL/others | NFL 2006+ [S]. NBA/NHL ranges [U] (site is behind a Cloudflare JS wall, so I could not read it) | H2H decimal odds, line, total | Open/close/min/max columns reported in past versions [U] | Pinnacle-heavy per the community, but sources are listed in the workbook's Notes sheet [S] | Excel | Free download; check the Notes sheet for terms [U] | [S] |
| Kaggle `cviaxmiwnptr/nba-betting-data-...` | NBA | Oct 2007 to Jun 2026 | Spread, total, ML, 2H spread/total, quarter scores | Single line | SBRO through Jan 2023, other sources after | CSV | CC0 | [V] metadata |
| Kaggle `christophertreasure/nba-odds-data` | NBA | 2008-2023 (2023 partial) | ML, spread, total, 2H total | Single | SBRO repackaged | CSV | "Other" | [V] |
| Kaggle `erichqiu/nba-odds-and-scores` | NBA | "most of past decade" as of 2020 | Game lines | ? | Scraped, mixed | CSV | CC0 | [V] (stale, 2020) |
| Kaggle `ehallmar/nba-historical-stats-and-betting-data` | NBA | Up to 2018 | ML/spread/total (appear to be per-book) | ? | ? | CSV, 250 MB | Unknown | [V] metadata only |
| Kaggle `caseydurfee/mgm-grand-nba-betting-data` / `...-nfl-...` | NBA 2021-22 to Feb 2026; NFL 2021-2025 | 4-5 seasons | Spread/total/ML **plus wager % and stake %** | Close only | BetMGM via Yahoo | CSV, code on GitHub | CC BY-SA 4.0 | [V] |
| Kaggle `minmaxer/*-closing-lines-ats-*` | NBA 2024-25, 2025-26; NFL 2025 | 1 season each | Game lines | Consensus open/close | — | CSV | ? | [V] listing |
| Kaggle `andyt338/graded-player-props-mlb-nfl-23-sportsbooks` (PropLine) | MLB (K, hits, TB), NFL (pass/rush/rec yds, receptions) | **Last 14 days only** (~612K rows). Full PropLine archive starts **April 2026** | Player props, graded against box scores | Opening + closing price/point with timestamps | 23 books + DFS apps | CSV | CC BY 4.0 | [V]. Too short for our purposes |
| Odds Shark database (now Covers) | NFL, NBA, MLB, NHL, NCAAF, NCAAB, CFL, WNBA | "30+ years" [S] | Spread/total, ATS results | Close | Consensus | Web query tool, no bulk download [S] | Scraping not addressed [U] | [S] |
| SportsOddsHistory (now on covers.com) | NFL/NBA/MLB/NHL | Futures back decades. Season win totals and **season-long player props** from about 2015-16 [S]; NFL game odds | Futures, win totals, season props, NFL closing lines | Close/preseason | — | HTML tables | Free archive [S] | [S] |
| Princeton DSS "Historical Sports Odds Database" | MLB 2009-2023, NBA 2006-2023, NFL 2009-2023, DFS 2017-2024 | ~15 yrs | Open/close ML and O/U (MLB); open/close spread and O/U (NBA) | Open + Close | ? | Princeton-affiliated users only | Licensed | [S]. Content matches the Odds Warehouse product (inference) |
| OddsPortal | Many incl. US leagues | Several years. Exact US start [U] | 1X2/ML, spread, total | Opening + closing per book; movement history on hover | ~80 books | HTML, scraped via third-party tools (OddsHarvester on GitHub, Apify actors) | ToS on scraping not verified [U]. Treat it as prohibited unless confirmed | [S] |
| Flashscore via Apify actor `sports-odds-lab/nba-nfl-mlb-nhl-odds` | NBA/NFL/MLB/NHL | **2009+** | ML, spread, total only (**no props/team totals**) | Open + close | 6 books (NBA 2009-13), rising to 17-28 now | JSON via Apify | Third-party scrape of Flashscore. Licensing risk is yours | [V] actor page. $5 per 1,000 games on the free plan |
| BettingIsCool / Pinnacle data API | Pinnacle markets, all sports | **2021+** ("5+ years") | Whatever Pinnacle posted, incl. props where offered [S] | Line history | Pinnacle only | REST (api.bettingiscool.com) | Pricing [U] | [S] |
| Pinnacle Odds Dropper archive | Pinnacle | Only since **18 Jun 2025** | All Pinnacle | Full line history | Pinnacle | Web, Gold plan | Paid | [S] |
| R package `pinnacle.data` | MLB 2016 + 2016 US election | 1 season | Pinnacle market odds | Time series | Pinnacle | R/CRAN | Open | [S] (curiosity only) |
| GitHub/Apify PrizePicks/Underdog scrapers | Many | Only what each scraper owner logged | DFS pick'em lines | Snapshots | DFS apps | — | ToS risk | [S]. I found no public multi-year archive |

### Cited Findings
- SBRO archive pages are live. NFL links run 2007-08 to 2021-22, NBA and NHL to 2022-23, NCAAF 2007-08 to 2022-23, MLB xlsx 2010-2021, NCAAB xlsx 2007-08 to 2020-21. The NBA 2022-23 page has columns Date, Rot, VH, Team, quarters, Final, Open, Close, ML, 2H. — [SBRO archive index](https://www.sportsbookreviewsonline.com/scoresoddsarchives/) (checked with curl on 2026-10-12; WebFetch gets a 404, so the site likely blocks bots)
- The NFL Kaggle set has lines since 1979: repole.com for 1978-2013, then sportsline.com and aussportsbetting for 2014-2024, under CC BY-NC-SA 4.0. — [Kaggle spreadspoke](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data)
- The aussportsbetting NFL file goes back to 2006 with home/away decimal odds, line and O/U, and its sources are listed in the Notes sheet. — [aussportsbetting NFL](https://www.aussportsbetting.com/data/historical-nfl-results-and-odds-data/)
- PropLine Kaggle sample: 612K graded props from 23 books with opening and closing lines, covering the last 14 days. The full archive starts April 2026. — [Kaggle PropLine](https://www.kaggle.com/datasets/andyt338/graded-player-props-mlb-nfl-23-sportsbooks)
- BetMGM closing lines plus wager/stake % for NBA 2021-22 to Feb 2026, scraped from Yahoo. — [Kaggle caseydurfee NBA](https://www.kaggle.com/datasets/caseydurfee/mgm-grand-nba-betting-data)
- Flashscore-derived odds go back to 2009 with open and close, 6 books in early NBA rising to 17-28, no props, priced at $5 per 1,000 games. — [Apify sports-odds-lab](https://apify.com/sports-odds-lab/nba-nfl-mlb-nhl-odds)
- Princeton DSS holds MLB 2009-2023 and NBA 2006-2023 open/close lines. — [Princeton DSS](https://dss.princeton.edu/catalog/resource6741)
- Odds Shark (now part of Covers) advertises 30+ years of NFL/MLB/NHL/NBA betting data. — [Odds Shark databases](https://www.oddsshark.com/sports-betting/sports-betting-databases)
- SportsOddsHistory archives futures and season-long props and is now hosted on Covers. — [SportsOddsHistory about](https://www.covers.com/sportsoddshistory/about-us/)
- BettingIsCool says its Pinnacle API holds 5+ years of history since 2021 (2.4B+ odds records). — [BettingIsCool](https://bettingiscool.com/)
- The Pinnacle Odds Dropper archive starts 18 Jun 2025. — [POD changelog](https://www.pinnacleoddsdropper.com/changelog)
- OddsHarvester is a public GitHub scraper for oddsportal.com. — [GitHub OddsHarvester](https://github.com/jordantete/OddsHarvester)

### Inferences
- SBRO is the free backbone for open/close game lines across all six leagues 2007-2022. It is a single unnamed line, not per-book, and not Pinnacle. The NFL/NBA/NHL/NCAAF seasons now have to be parsed from HTML (our `sports/sports/sbr.py` handles the MLB xlsx).
- For **free** NBA/NHL/NCAAF game lines after 2022-23, the realistic options are the Flashscore scrape (cheap: about 1,300 NBA games per season comes to about $6.50 a season), the BetMGM Kaggle sets, or our own ESPN collector.
- There is no free 10-season player-prop archive. The best free prop history is ESPN (2022+), which we already have.

### Gaps
- aussportsbetting NBA/NHL/MLB year ranges and current terms (blocked by Cloudflare).
- Whether SBRO allows redistribution. I found no license page.
- OddsPortal ToS on scraping, and its exact first season for US leagues.
- BettingData.com: not researched.
- I did not check Covers' own matchup archives separately from Odds Shark.
- The existence of a "Player Prop Database" (playerpropdatabase.com, with 2018 NFL QB props per a search snippet) could not be confirmed because the domain does not resolve.

## Q2. Paid sources: coverage, start dates, prices

### Takeaway
Every API that has props starts in **2019-2023**, because each vendor only began recording when it started up and none backfills. The cheapest credible route to multi-book, time-stamped props is **The Odds API** (props from 3 May 2023 at 5-minute snapshots, $30-$119/mo) or **OddsPapi** (free tier includes history, archive start undisclosed). The deepest prop history found is **SportsDataIO** (props from 2020, all six leagues), but its price is quote-only.

### Source table (paid)

| Source | Leagues | Game-line history | Prop history | Snapshots | Books / Pinnacle | Price | Format / terms | Status |
|---|---|---|---|---|---|---|---|---|
| **The Odds API** | All major US + NCAA + many intl | Featured markets (h2h/spread/total) from **6 Jun 2020**: 10-min snapshots, then 5-min from Sep 2022 | Player props and additional markets from **3 May 2023**, 5-min | Time-stamped snapshots (any timestamp) | US: DK, FD, MGM, Caesars, Bovada, MyBookie etc. Pinnacle is listed under the EU region | Free (500 credits, no history); $30/20K; $59/100K; $119/5M; $249/15M credits per month. Historical cost is 10 credits per region per market (per event for props) | REST JSON. Historical requires a paid plan | [V] |
| **SportsDataIO** (Vault) | NFL, MLB, NBA, NHL, CFB, CBB | **2019+** | Props and futures **2020+** (game props, team props, player props, futures). Recorded live, never backfilled | Line movement captured (granularity [U]) | ~9 named US books [S] | Free trial tier. Historical pricing is quote-only [V] | API, S3 bulk, custom | [V] |
| **OpticOdds** | 25+ sports | "Several years" for major leagues | Yes | Every line move, millisecond timestamps | 200+ books | ~$5,000+/mo per sport [S] | REST and bulk export | [V] page; price [S] |
| **OddsJam** API | US majors | Historical incl. opening/closing/live moves | Yes; start date [U] | — | 100+ | Not published; "$299/mo for historical" per a competitor [S] | Enterprise contract | [S] |
| **OddsPapi** | Many | Archive starts at their ingestion date (undisclosed) | Props included | Every line move | 350+ incl. Pinnacle, Singbet | Free 250 req/mo **includes history** (3 books per call); Pro price [U] | REST/WebSocket | [S], vendor's own blog |
| **SportsGameOdds** | 67+ leagues | **From ~Feb 2024**, deeper for NFL/NBA [S] | Props on Pro+ | — | 85+ | From $99/mo (yearly) to $149/mo [S] | REST | [S] |
| **EndZone API** | NFL only | — | NFL player props **2023-2025**, 2026 from week 5 | Open + close with capture timestamp | 7 books incl. **Pinnacle**, DK, FD, MGM, ESPN BET, Bovada, BetOnline | 10 credits per game. 500 free credits/mo; Starter $19/mo covers one full season (2,850 credits) | REST + MCP. GSIS player IDs join to nflverse | [V] |
| **Unabated** API | NFL, NBA, MLB, NHL, WNBA (Concierge) | Includes historical [S] | Yes | — | — | $799/mo Concierge (self-serve key, 5 leagues); $3,000+/mo enterprise [S] | — | [S] |
| **Sportradar** | All | Historical sold separately via licensing | Props for NBA/MLB/NHL/NFL | — | Aggregated US books | $10,000+/mo, contracts [S] | Enterprise | [S] |
| **BigDataBall** | NBA (also NFL/MLB datasets) | NBA odds datasets **2020-21 to 2025-26** only | "player props" mentioned; fields unspecified | [U] | [U] | Historical odds per season $10-30; bundles 5-15% off; in-season pass $148 | Excel/CSV | [V] page |
| **Odds Warehouse** | MLB, NBA, NFL, NHL (+ soccer, tennis) | 10+ yrs (likely the Princeton set) | No props | Open/close | [U] | $199 list ($39/season, $99 for 10+ yrs on a flash sale) | CSV | [V] page. No props |
| **Sports Insights Bet Labs** (Action Network) | NFL, MLB, NBA, NHL, WNBA, CFB, CBB | **2003+** (some pages say 2005) | No prop history found | Open + close spread/ML/total **plus public betting %** | Real-time feed covers 50+ books; the archive is mostly a consensus line | Subscription price [U] | Web filters + downloadable DBs | [V] page; price [U] |
| **KillerSports / SDQL** | NFL, NBA, MLB, NHL, NCAAF, NCAAB, WNBA, CFL | Long (decades) | No props found | Close lines (open [U]) | — | Free community tier; paid tiers [U] | Query language, not bulk | [S] |
| **BettingPros** | NFL, NBA, MLB, NHL | — | Prop analyzer shows historical hit rates and lines per player; depth and export [U] | — | Multiple books | Subscription | Web only, no bulk/API found | [U] |
| **Action Network PRO** | — | — | Not researched | | | [U] | | [U] |

### Cited Findings
- The Odds API: featured markets from June 6, 2020 (10-min snapshots, 5-min from Sep 2022), player props and additional markets from May 3, 2023 at 5-min snapshots, 10 credits per region per market (per event for props), history on paid plans only. — [The Odds API historical](https://the-odds-api.com/historical-odds-data/)
- The Odds API plans: $30/20K, $59/100K, $119/5M, $249/15M. Pinnacle is listed as an EU bookmaker. — [The Odds API](https://the-odds-api.com/)
- SportsDataIO: "historical betting lines for all major sports from 2019 onwards, with props and futures from 2020", delivered via API or S3. — [SportsDataIO historical odds](https://sportsdata.io/historical-odds)
- SportsDataIO does not collect retroactively. It covers Big Six game props, team props, player props and futures for NFL/MLB/NBA/NHL/CFB/CBB. — [SportsDataIO Historical Data Guide](https://support.sportsdata.io/hc/en-us/articles/4405005816215-Historical-Data-Guide)
- EndZone API: NFL props 2023-2025 plus 2026 from week 5, 7 books including Pinnacle, open and close, $19 Starter. — [EndZone player props](https://endzoneapi.com/data/player-props)
- OpticOdds: 200+ books with millisecond timestamps; history depends on when each book was first collected. — [OpticOdds historical](https://opticodds.com/historical-odds)
- Pricing comparison (Jan 2026, updated Aug 2026): SportsGameOdds $149/mo ($99 yearly), OpticOdds $5,000+/mo per sport, Unabated $3,000/mo. — [SportsGameOdds comparison](https://sportsgameodds.com/blog/comparing-odds-api-providers) (vendor-authored, possibly biased)
- Unabated: $799/mo Concierge for 5 leagues, $3,000+/mo enterprise. Sportradar: $10,000+/mo. — [search summary of unabated.com / sharpapi.io](https://unabated.com/get-unabated-api)
- OddsPapi: the free tier includes history and props in `/historical-odds`; archive start not disclosed. — [OddsPapi CSV guide](https://oddspapi.io/blog/historical-odds-csv-excel-backtesting/)
- OddsJam does not publish API pricing. — [OddsPapi on OddsJam](https://oddspapi.io/blog/oddsjam-api-alternative/) (competitor source)
- Bet Labs: open/close spread/ML/total plus betting % for 7 leagues from 2003. — [Sports Insights historical odds](https://www.sportsinsights.com/sports-betting-systems/historical-betting-odds-sports-data/), [Bet Labs DB blog](https://www.sportsinsights.com/blog/historical-betting-database/)
- BigDataBall NBA odds datasets only start in 2020-21. — [BigDataBall NBA](https://www.bigdataball.com/datasets/nba-data/)
- Odds Warehouse sells CSVs for 4 US leagues; list price $199. — [Odds Warehouse](https://www.oddswarehouse.com/)
- Most third-party Pinnacle history begins around 2018, because Pinnacle's public API closed in 2020. — [sportsapis.dev](https://sportsapis.dev/historical-odds) (aggregator, low confidence)

### Inferences
- **Cheapest path to the longest prop history:**
  1. The Odds API $119/mo for 1-2 months to bulk-pull props from May 2023 onward. This gives about 3.5 seasons NFL/NBA/NHL/MLB, multi-book, time-stamped.
  2. Get a SportsDataIO quote for 2020-2022 props, which would add about 3 more seasons. That is the only vendor found claiming props before 2023 across all six leagues.
  3. EndZone ($19) for an NFL Pinnacle prop close-line check from 2023.

  Rough credit math for The Odds API (my estimate, not verified): one NBA game at one snapshot with one region and 5 prop markets costs 50 credits. 1,300 games × 2 snapshots (open/close) is about 130K credits per season per league. 5M credits covers many league-seasons, so one month at $119 is probably enough.
- **10+ seasons of props cannot be bought anywhere**, as far as this search shows. The structural ceiling is ~2019-2020 (SportsDataIO) for multi-book data. Offshore books (Pinnacle, Bookmaker) did post NFL/NBA props before 2018, but I found no archive of them.
- For game lines with 10+ seasons and open/close, the paid options (Bet Labs 2003+, Odds Warehouse, KillerSports) add little beyond free SBRO + nflverse. The exception is Bet Labs' public betting %, which nothing free has for pre-2021 seasons.

### Gaps
- Exact prices for SportsDataIO historical, OpticOdds, OddsJam, Bet Labs, KillerSports, Action Network PRO, and BettingIsCool. All are quote-only or unverified.
- OddsPapi archive start date.
- BettingPros prop-history depth.
- Whether BigDataBall odds files contain actual player prop lines, and from which books.
- OddsJam historical start date.

## Q3. Realistic earliest prop-history year per league

### Takeaway
Multi-book player-prop history realistically starts at **2020 (SportsDataIO)** or **May 2023 (The Odds API)** for NFL/NBA/MLB/NHL. College props are thinner still. The only pre-2020 "prop" history that is easy to find is **season-long** props and win totals (SportsOddsHistory, about 2015+), not per-game props.

### Cited Findings
- SportsDataIO props start in 2020 for NFL/MLB/NBA/NHL/CFB/CBB. — [SportsDataIO](https://sportsdata.io/historical-odds)
- The Odds API props start 3 May 2023. — [The Odds API](https://the-odds-api.com/historical-odds-data/)
- EndZone NFL props start in 2023. — [EndZone](https://endzoneapi.com/data/player-props)
- SportsGameOdds history starts around Feb 2024. — [search result referencing sportsgameodds.com FAQ](https://sportsgameodds.com/docs/faq) [S]
- The PropLine archive starts April 2026. — [Kaggle PropLine](https://www.kaggle.com/datasets/andyt338/graded-player-props-mlb-nfl-23-sportsbooks)
- SportsOddsHistory carries season props and win totals from about 2015-16. — [SportsOddsHistory MLB](https://www.sportsoddshistory.com/mlb-odds/) [S]

### Inferences
- Earliest realistic per-game prop history by league:

  | League | Earliest per-game prop history | Source |
  |---|---|---|
  | NFL | 2020 | SportsDataIO, paid |
  | NFL | 2022-2024 | ESPN, free, which we have |
  | NFL | 2023 | Odds API / EndZone |
  | NBA | 2020 / 2023 | SportsDataIO / Odds API |
  | MLB | 2020 / 2023 | SportsDataIO / Odds API |
  | NHL | 2020 / 2023 | SportsDataIO / Odds API |
  | NCAAF/NCAAB | 2020 (thin) / 2023 | SportsDataIO / Odds API |

- With at most about 6 seasons of props even after paying, a 10-season prop backtest is not feasible. Prop studies should be designed around 3-6 seasons with walk-forward splits, consistent with our existing ESPN-based setup.

### Gaps
- No archive of pre-2019 offshore per-game props was found. It may exist privately, for example with syndicates or the Don Best/Sportradar enterprise feed. Unverified.
