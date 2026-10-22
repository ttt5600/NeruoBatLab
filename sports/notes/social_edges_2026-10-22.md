# Social / practitioner edge sweep, 2026-10-22

Research agent report (no backtests run, no sealed data touched, nothing committed).

**Brief:** go through what practitioners say online about edges in prediction markets (Kalshi, Polymarket US,
ForecastEx), sports betting and props, and retail stock/options strategies. For each claimed edge, record the claim,
the source, the evidence quality, the mechanism, whether it is legal and automatable for a CA/CO user (Kalshi API is
the automated venue; sportsbooks are manual only; no offshore), and a testable spec on free data. Rank the top 10 and
flag overlap with E1/E2 (Kalshi weather consensus-reject).

**To re-launch:** start a general-purpose agent with the brief above, and point it at
notes/EDGES_AND_AGENTS.md sections 1-5 first.

## 0. Coverage and limits (read first)

- **Reddit was not reachable.** The WebSearch tool refuses `reddit.com` ("domain not accessible to our user agent").
  I did not try to get around that. Reddit opinion only appears here when another site quotes it.
- **X/Twitter is barely indexed** by web search, and we have no API key. No X thread was recovered verbatim.
- **What worked:** Hacker News, Substack, GitHub (via `gh`), academic preprints (arXiv/NBER/CEPR/SSRN-linked), vendor
  blogs, and Kalshi's own public API. I checked several claims directly against the API (marked VERIFIED-API).
- **Main finding of the sweep.** Practitioner talk is overwhelmingly vendor marketing with unverifiable P&L. The best
  evidence about *how* money is made on Kalshi/Polymarket comes from four trade-level studies (Becker; Bürgi-Deng-Whelan;
  Bartlett-O'Hara; Cardozo et al.), and all four say the same thing: **makers beat takers, NO beats YES, favorites beat
  longshots.** Our executor is a pure taker (`immediate_or_cancel` at the ask; scripts/kalshi_exec.py:520-521).

Labels used below: **VERIFIED-PAPER** (trade-level academic study I read), **VERIFIED-API** (I checked it on Kalshi's
public API today), **VENDOR** (self-reported by someone selling a product), **UNVERIFIED** (anecdote, no data).

---------------------------------------------------------------------------------------------------------------------
## 1. The evidence base (trade-level studies)

| Study | Data | Key numbers (exact, as reported) | Label |
|---|---|---|---|
| Jon Becker, "The microstructure of wealth transfer in prediction markets", 2026-01-18 (SSRN version 2026-08-01). https://www.jbecker.dev/research/prediction-market-microstructure ; HN thread https://news.ycombinator.com/item?id=46680515 | Kalshi 2021-2025, 72.1M trades, $18.26B | Takers -1.12% / makers +1.12% excess return (**gross of fees**). By category (taker/maker): Sports -1.11/+1.12; **Weather -1.29/+1.29**; Finance -0.08/+0.08; Crypto -1.34/+1.34; Entertainment -2.40/+2.40; Media -3.64/+3.64; Politics -0.51/+0.51; World Events -3.66/+3.66. YES at 1c EV -41%, NO at 1c +23%; NO beats YES at 69 of 99 price levels; dollar-weighted YES -1.02% vs NO +0.83%. Maker NO +1.25 pp vs maker YES +0.77 pp (small, Cohen's d 0.02-0.03). Time: through 2023 takers +2.0%; flipped 2024 Q2; post-election makers +2.5 pp. | VERIFIED-PAPER (blog + dataset) |
| Bürgi, Deng & Whelan, "Makers and Takers: The Economics of the Kalshi Prediction Market", UCD, Jan 2026. https://www.karlwhelan.com/Papers/Kalshi.pdf (also CESifo / CEPR DP20631) | Kalshi 2021 to Apr 2025; 46,282 contracts, 12,403 events, >300k prices (daily snapshots up to 10 days before close) | Sub-10c contracts lose >60%; small, statistically significant positive post-fee returns above 70c; overall average about -20% (**per contract, equal-weighted, not dollar-weighted**). Makers -9.64% vs takers -31.46% on average; makers buying contracts >=50c earn +2.6%. The bias holds in every category (Financials, Climate & Weather, Crypto, Politics, Entertainment, Economics); smaller in politics/entertainment; **2025 coefficient smaller and less significant (bias may be fading).** Results are robust to dropping sports. | VERIFIED-PAPER (read the PDF) |
| Bartlett & O'Hara, "Adverse Selection in Prediction Markets: Evidence from Kalshi", NBER draft, 2026-07-02. https://conference.nber.org/conf_papers/f243677.pdf | Kalshi Jul 2021 to 2026-03-11, 41.6M trades; 478,167 markets (single-name vs broad-based; weather/sports/politics excluded from the main split) | Single-name markets: takers buy YES 60.2% of the time but only 30.9% settle YES. Makers win 63.4% of contracts and make 1.91c/contract (vs 0.82c on broad-based). Single-name YES prices exceed realized YES rates "across the entire probability spectrum", not just at the tails. Order-flow toxicity (VPIN) predicts maker losses. | VERIFIED-PAPER (read the PDF) |
| Cardozo & Rivero-Wildemauwe, "The Favorite-Longshot Bias in Prediction Markets: Evidence from Polymarket", arXiv 2609.12878, 2026-09-11. https://arxiv.org/abs/2609.12878 | Polymarket, 588M trades, 2.48M accounts | Purchases below 10c lose 19.3c per $; at or above 90c earn +0.83c. Depends on weighting: equal-weighted contracts -6.3c, event-grouped +4.1c. **Robust in Crypto and Politics, absent in Sports.** | VERIFIED-PAPER (abstract) |
| CEPR DP21615 (Polymarket wallet study; page returned 403, details via search summary) https://cepr.org/publications/dp21615 | 588M trades, $67B | Top 1% of profitable users capture 76.5% of profits; winners mostly post limit orders (makers), losers mostly take. Persistence is modest and may be selection. | Secondary summary, UNVERIFIED detail |
| Telonex, Polymarket 15-min crypto markets, 2026-02-13. https://telonex.io/research/top-crypto-traders-polymarket-15m | 1 week, 15.3M fills, 46,945 wallets | Makers +$728,501, takers -$728,501 (pre-fee) | VENDOR, one week |
| Kalshi parlays: "The fish are the product", UFO Holdings Substack, 2025-10-07. https://ufoholdings.substack.com/p/the-fish-are-the-product | All Kalshi parlay fills, first ~7 days | Taker hold 11% on $806k handle; ~1% fees, ~7% to makers (the rest isn't explained) | Practitioner, 1 week |

VERIFIED-API today (2026-10-22): weather series KXHIGHNY/KXHIGHCHI/KXHIGHTDC have `fee_type=quadratic` (taker-only
fee; sports KXNFLGAME is `quadratic_with_maker_fees`). Kalshi's public `/incentive_programs?status=active` lists
**6,926 active liquidity-incentive programs**, including 12 per city on KXHIGH*/KXHIGHT*/KXLOWT* (e.g.
KXHIGHNY-26OCT23-B59.5: `period_reward` 200000, `target_size` 1000, `discount_factor_bps` 5000, 14:00Z to 05:00Z next
day). **Units of period_reward are not confirmed.** If they are centi-cents, that is $20 per bracket per day, but that
is a guess. Fee formulas from secondary sources (https://whirligigbear.substack.com/p/makertaker-math-on-kalshi ,
https://www.quantvps.com/blog/kalshi-fees-explained): taker 0.07·P·(1-P); maker 0.0175·P·(1-P) only on series that
list maker fees.

---------------------------------------------------------------------------------------------------------------------
## 2. TOP 10, ranked by evidence x feasibility x capacity (each scored 1-5)

| Rank | Idea | E | F | C | Score | Overlaps E1? |
|---|---|---|---|---|---|---|
| 1 | Run E1/E2 as a **maker** (resting post-only NO bids) instead of a taker IOC, plus LIP rewards | 4 | 5 | 3 | 60 | YES (execution of E1) |
| 2 | **YES-optimism fade**: NO side in single-name / Mentions / Entertainment / World-events markets | 4 | 4 | 3 | 48 | Partial (same bias family) |
| 3 | **Favorite-longshot, ex-weather**: buy >=70c sides (as maker) across non-weather Kalshi categories | 4 | 4 | 3 | 48 | Partial (E1 buys high-priced NO) |
| 4 | **Earnings straddle selling** (variance risk premium at earnings) = L11 from the short side | 3 | 3 | 3 | 27 | No |
| 5 | **Liquidity-incentive harvesting** on weather brackets (two-sided resting depth near E1 fair values) | 3 | 4 | 2 | 24 | YES |
| 6 | **Weather intraday lock-in**: late-afternoon observed max (ASOS) vs Kalshi bracket prices | 2 | 5 | 2 | 20 | YES (same markets, new mechanism) |
| 7 | **Kalshi combos as maker** (price parlays from Pinnacle legs + correlation) | 2 | 2 | 4 | 16 | No |
| 8 | **NFL wind unders** vs closing totals (Kalshi totals or CO books) | 2 | 3 | 2 | 12 | No (sibling of F2 rain) |
| 9 | **Kalshi Fed-meeting markets vs futures-implied (FedWatch)** far from the meeting | 2 | 3 | 2 | 12 | No |
| 10 | **Soft US books vs Pinnacle consensus** (Kaunitz method) in Colorado | 4 | 1 | 2 | 8 | No (= E3 / L9) |

### #1 Maker execution of E1/E2 (+ liquidity rewards) -- strongest, cheapest, overlaps E1
- **Claim:** makers earn what takers lose; on Kalshi *weather* specifically, takers -1.29% and makers +1.29% gross
  (Becker, above). Makers pay no fee on weather (series fee_type quadratic, VERIFIED-API). On top of that, Kalshi pays
  liquidity rewards for resting depth on exactly the KXHIGH brackets E1 trades (VERIFIED-API).
- **Who:** Becker (independent researcher, open dataset, 3.9k-star repo https://github.com/Jon-Becker/prediction-market-analysis);
  Bürgi/Deng/Whelan (UCD economists); Bartlett/O'Hara (Stanford/Cornell). Practitioner echo: MarketWatch 2025-05-22
  profiles of two student Kalshi market makers claiming ~$150k and >$165k (UNVERIFIED;
  https://www.itiger.com/hant/news/2537923561); the 4AM Club Substack 2025-11-13 lists market making as the main money
  maker (UNVERIFIED; https://4amclub.substack.com/p/how-to-make-money-trading-on-kalshi).
- **Mechanism:** E1 currently pays the spread plus the taker fee (0.07·p·(1-p) per contract, about 0.7% of stake at a
  NO price of .90). A resting NO bid a cent or two below the ask saves both. The risk is adverse selection: a resting
  bid fills more often when the bracket is about to become live, which Whirligig Bear and Bartlett-O'Hara (VPIN)
  both discuss. Becker's maker average, however, includes that adverse selection.
- **Legal/automatable:** yes; same Kalshi API, `post_only: true, time_in_force: good_till_canceled` (the executor
  already has this code path in its demo smoke test, kalshi_exec.py:841-842).
- **Testable spec (free data):** (a) *Historical fill simulation on E1 DEV picks only*: for each E1 pick at ~22:00
  local, post a hypothetical NO bid at q = pick NO price minus k cents (k in {0,1,2}, pre-registered). Using the public
  trade tape `GET /markets/trades?ticker=` (VERIFIED-API: it has `taker_side`, `yes_price_dollars`, `created_time`),
  the bid counts as filled if a YES-taker trade prints at yes_price >= 1-q before the market closes. Report the fill
  rate, ROI of filled vs unfilled picks (the adverse-selection test), and blended ROI against the taker baseline.
  Pass = blended maker ROI > taker ROI on the same picks, with fill rate >= 50%. (b) *Forward A/B*: alternate
  maker/taker by pick hash for 4 weeks on the live $1 units. **Do not run (a) on the E1 lockbox quarters** without
  the lead's say-so. The lockbox is already spent, but the protocol keeps it untouched.
- **Capacity:** saving 0.5-1% of stake on a +1.45%/+0.78% edge is a large relative gain. LIP rewards come on top and
  scale with resting size (target_size 1000 means $1 orders earn roughly nothing until the bankroll grows).

### #2 YES-optimism fade in single-name / Mentions / Entertainment / World markets
- **Claim:** takers over-buy YES across the probability range. In single-name markets takers buy YES 60.2% of the time
  vs 30.9% YES settlement (Bartlett-O'Hara). The maker-taker gap is largest in Media (7.28 pp), World Events (7.32 pp)
  and Entertainment (4.79 pp) (Becker). DeFi Rate's Trump-mention piece (Feb 2026) says retail "disproportionately buy
  Yes" on memorable phrases and sharps fade with NO. But it also reports 915/1,312 = 69.7% of Trump mention contracts
  settled YES, so **a high YES base rate is not the same as YES being overpriced: only price vs outcome matters**
  (https://defirate.com/news/trump-state-of-the-union-multi-million-dollar-mention-market-how-traders-are-playing-it/).
- **Mechanism:** a behavioural "optimism tax" (Becker's term), the excitement of YES, and lottery preferences.
- **Legal/automatable:** yes (Kalshi API). Notable: KXTRUMPMENTION (166), KXRT (236) and KXHEADLINE (75) are among
  the series with the most active LIP programs (VERIFIED-API), so a maker here also collects rewards.
- **Testable spec (free):** pull settled markets in Kalshi categories Mentions, Entertainment, World and Media via
  `GET /markets?status=settled&series_ticker=` plus hourly candlesticks
  (`/series/{s}/markets/{t}/candlesticks?period_interval=60`, VERIFIED-API). Rule: at T-24h (and T-2h, pre-registered
  as 2 arms), buy NO at the NO ask when the YES mid is in [0.15, 0.85]; net of taker fees. Dev = settled before
  2026-01-01; sealed = 2026. Primary metric = ROI per $ with an event-clustered bootstrap (the Polymarket paper shows
  grouping by event flips signs, so cluster by event, never by contract). Secondary = the same rule as a maker (fill
  if the tape crosses).
- **Risks:** insiders (the Bezos-Super-Bowl NO flow in Bartlett-O'Hara; Kalshi has sanctioned insiders), resolution
  wording, and thin books.

### #3 Favorite-longshot, non-weather categories
- **Claim:** sub-10c contracts lose >60%, and >70c contracts earn small positive post-fee returns (Whelan, all
  categories). A 5c contract wins 4.18% (Becker). On Polymarket, absent in Sports (Cardozo).
- **Mechanism:** overweighting small probabilities plus disagreement (Whelan's model).
- **Caveats:** returns are equal-weighted per contract; the 2025 bias is weaker; and E1 is already a version of this
  (buying high-priced NO on brackets nobody's model supports), so **test it with weather excluded** to see whether it
  is a separate source.
- **Testable spec (free):** same Kalshi pull as #2 but every category except Climate/Weather and Sports. Buy the side
  priced >= 0.80 at T-24h; net of fees. Report ROI by category with event clustering. Kill it if 2025-26 ROI <= 0
  (the decay check).

### #4 Earnings straddle selling (VRP at earnings) -- L11 from the short side
- **Claim:** ORATS (options-data vendor) says the actual earnings move averages about 85% of the implied move, and
  straddle buyers win about 40% of the time at slightly below break-even. Its 2026 post says this season realized moves
  beat implied (https://blog.orats.com/marketwatch-talks-to-orats-about-this-earnings-season ,
  https://orats.com/blog/earnings-straddles-strong-season-2026). Broader VRP: the Cboe PUT index returned 9.40% vs
  S&P 9.91% with standard deviation 10.26% vs 15.38% and max drawdown -32.7% vs -51.0%, Jul 1986 to Aug 2023
  (https://www.cboe.com/insights/posts/generating-income-and-managing-risk-cash-secured-put-writing-in-a-low-equity-return-environment).
  Option sellers profited after fees in 0DTE SPX (Beckmeyer-Branger-Gayda 2023, secondary summary:
  https://neudata.co/literature-reviews/zero-day-to-expiry-options-a-losing-bet-for-retail-traders).
- **Evidence:** VRP is academic-grade; the earnings-specific 85% ratio is VENDOR, with an undisclosed method.
- **Legal/automatable:** stocks/options via Robinhood (agentic account only for agent trades; the main account is
  tickets). Use defined-risk iron condors, never naked short straddles at $50-5k scale.
- **Testable spec:** Robinhood MCP daily option OHLC + IV (per EDGES_AND_AGENTS L11). For S&P 500 names with
  earnings, sell the ATM straddle at the close before the report and buy it back at the next close. Report mean P&L
  as a % of premium, and the worst 1% tail. Iron-condor variant with wings at 2x the implied move. Note L11's existing
  bias warning (implied about 1.33x past mean move, measured badly).

### #5 Liquidity-incentive harvesting (weather and others)
- **Claim:** Kalshi pays rewards for resting depth near the top of book, even unfilled (Kalshi help:
  https://help.kalshi.com/en/articles/16076644-liquidity-incentive-programs-where-to-find-them). 6,926 programs are
  active today (VERIFIED-API). The older "volume rewards" program was reportedly being ended early under CFTC scrutiny
  (https://defirate.com/news/kalshi-volume-rewards-cftc-scrutinizes-prediction-markets/). **Liquidity** programs are
  still live per the API.
- **Spec:** read the scoring formula (docs.kalshi.com/api-reference/incentive-programs). Snapshot KXHIGH order books
  hourly for 2 weeks (paper). Compute what a two-sided quote around E1's fair value would have earned in rewards minus
  adverse-selection losses (fills from the trade tape). Pass = positive after inventory losses. Natural add-on to #1.
- **Caveat:** target size 1000 contracts; at $1 units our share is negligible. This matters only at the ~$5k bankroll.

### #6 Weather intraday lock-in (late day)
- **Claim:** practitioner tools join each market with "temperature already observed today at the settling station"
  (Apify actors). One trading desk's internal doc says that once the temperature is 4F+ below the intraday peak
  after the usual peak hour, there is an "87-92%" chance the high is locked in
  (https://edge-assist.duckdns.org/data-status/docs/weather/inventory, UNVERIFIED; the page returned 502 when fetched).
  Settlement nuance: the CLI high is the max of 2-minute ASOS averages in whole F, while METAR is hourly near :52 in
  0.1C, so the METAR max can sit 1-2F below the CLI. IBKR worked example: 88F / CLI 87F / METAR 86F
  (https://www.interactivebrokers.com/campus/traders-insight/ibkr-climate-energy/nuances-in-the-calculations-of-daily-temperature-market-values/).
  During DST the climate day runs 01:00 to 00:59 local (Kalshi help).
- **Why it isn't N1 (dead):** N1 was 06:00 forecasts. This uses *observations* after the peak, at 16:00-19:00 local.
- **Spec (free):** IEM ASOS 1-minute/5-minute archive (mesonet.agron.iastate.edu, free) for the settlement stations,
  plus Kalshi hourly candlesticks. At 17:00 local, buy NO on every bracket whose upper edge is below the observed
  max-to-date (certain loser, after F/C rounding margin of 1F), and on brackets above when temp is >=4F below peak and
  the NWS hourly forecast shows no rebound. Metric = ROI after fees; most of the value is whether the NO ask is still
  < 0.99. Likely already arbitraged by bots, so cheap to kill.

### #7 Kalshi combos as maker
- Parlay takers lost 11% in week 1 (UFO Holdings). Combos use RFQ with **1-second** confirm/execute timers
  (https://docs.kalshi.com/getting_started/rfqs). Whether a retail account can quote RFQs is unclear: one third-party
  page says API RFQ quoting needs Premier/Market Maker tier (UNVERIFIED). Pricing needs leg correlations (same-game).
  Large capacity, low feasibility. **First step: check tier eligibility with Kalshi support before any modelling.**

### #8 NFL wind unders
- Raw hit rates: under 55.4% at wind >= 10 mph (Action Network 2018, no CLV) and 54% at > 20 mph since 2015 (FOX).
  "Multi-season study" with no numbers (startuphub, UNVERIFIED). Skeptics say the market adjusts below 15-20 mph.
  Sources: https://www.actionnetwork.com/nfl/nfl-over-under-betting-week-7-windy-games-2018 ,
  https://amp.foxsports.com/stories/nfl/nfl-odds-weather ,
  https://www.startuphub.ai/news/nfl-betting-does-moderate-wind-create-unders-market-inefficiencies
- **Spec (free):** nflverse schedules (`total_line` = close, `roof`). **Do NOT use nflverse `wind`**: it is the game-time
  observation, which is the same leakage class as the F2 join bug. Use the pre-kickoff forecast from the Open-Meteo
  historical-forecast API at the stadium lat/lon (T-3h). Bet the under at close when forecast sustained wind >= 15 mph
  and the venue is outdoor. Dev 2015-2022, sealed 2023-25. Venue: Kalshi NFL totals (automatable) or CO books (manual).

### #9 Fed markets vs futures
- Dune analysis (late-2025 meetings) via search summary: peak Kalshi-vs-FedWatch gaps of 5-15c (Jan), 8-14c (Mar),
  20+c (Apr) far from the meeting; all converge within 1-2c near it; Kalshi-Polymarket ~1c. Partly definitional
  (Kalshi: change at this meeting; FedWatch: rate level after). The blog URL returned 404 when fetched
  (https://dune.com/blog/pricing-the-fed-kalshi-polymarket-cme-fedwatch), so the numbers are UNVERIFIED-secondary.
  Becker: Finance maker-taker gap only 0.17 pp, so the venue is efficient. Low priority. Spec: daily Kalshi KXFED
  candles vs fed-funds-futures-implied (needs ZQ settlement history, which may not be free); bet toward futures when
  the gap is >= 8c and the meeting is >= 30 days out.

### #10 Soft US books vs sharp consensus (Colorado, manual)
- Kaunitz et al. 2017 (https://arxiv.org/pdf/1710.02824): 10-year simulation on 479,440 soccer matches +3.5%; real
  money 256 bets, $957.50 profit (+~8.5% on $50 stakes) over 5 months, then the authors were restricted. It is the
  same mechanism as our sealed E3 and L9, so nothing new except confirmation. Manual only; accounts get limited.

---------------------------------------------------------------------------------------------------------------------
## 3. Claims reviewed and rejected or deprioritised (with reason)

| Claim | Source | Why not |
|---|---|---|
| GFS-ensemble weather bots (edge > 8%) make money | https://github.com/suislanchez/polymarket-kalshi-weather-bot (791 stars), Vesper, Predict & Profit ($67), WeatherEdge ($25/mo, "67% win rate"), "Highest profits $1.8k" | VENDOR/no track record. **Matters to us as a CROWDING signal for E1**: many retail bots now trade KXHIGH with free GFS ensembles. Bot for Kalshi itself warns that model-market gaps are not edges (https://www.botforkalshi.com/blog/kalshi-weather-trading-strategy). |
| Polymarket weather wallets: "$1k -> $24k London", Maskache2 +$34,106 | https://blog.devgenius.io/found-the-weather-trading-bots-quietly-making-24-000-on-polymarket-and-built-one-myself-for-free-120bd34d6f09 ; https://dev.to/xniiinx/my-top-7-most-profitable-weather-market-traders-on-polymarket-gb7 | UNVERIFIED; international Polymarket is not legal for US users; our PMUS port failed (F4). |
| Momentum/flow bot on Polymarket | https://falsifylab.substack.com/p/strategy-deep-dive-polymarket (2026-06-27) | A useful *negative*: real money, 104 trades, -$239, win rate 32.7%, Sharpe -2.46; overfit on a thin-liquidity period, then a fee change. |
| BTC hourly / 15-min Kalshi vs Deribit/Binance | claudeskills page; CarlosIbCu arb bot | HFT-dominated per the same sources; Polymarket 15m makers are pros. Not retail-feasible. |
| Kalshi S&P range buckets vs SPX options density | Stevens Institute PoC, 2025-05-16 https://fsc.stevens.edu/event-contract-mispricing-via-options-implied-probabilities/ | Proof of concept only; Finance maker-taker gap is just 0.17 pp (Becker). Possible later via Robinhood option chains. |
| CPI / jobs nowcast vs Kalshi | botforkalshi, nexusfi guides | Our CPI was DEAD; Fed research finds Kalshi beats consensus on headline CPI. |
| Rotten Tomatoes scraping | MarketWatch 2025-05-22 | The profiled trader *lost money* on it before switching to market making. |
| Copy-trading top Polymarket wallets | CEPR DP21615, MEXC "six strategies" | Persistence is modest and may be selection; international Polymarket is not US-legal. |
| Overnight drift (buy close, sell open) | Elm Wealth 2025-03-03 https://elmwealth.com/night-shift/ ; Quantpedia GDX | Gross only; Quantpedia shows open-price artifacts; our 14+11 ETF rules are DEAD. |
| PEAD | Martineau, CFR 2022 https://nowpublishers.com/article/Details/CFR-0122 | Gone in large caps since ~2006 (2025 counter-papers disputed). |
| Pre-FOMC drift | Kurov et al.; NY Fed 2018 | Disputed / largely gone after 2015. |
| MLB umpire / strikeout props | Action Network, FTN, VSiN | Anecdote only, small samples; our strikeout backtest was break-even. |
| Player-prop unders | PFF 2017-19 (55% unders, trending to 50% in 2020) | Same as our dead/failed R_REB; books adapted. |
| Kalshi athlete availability ("will play") markets | https://legalsportsreport.com/274126/kalshi-launches-markets-on-whether-athletes-will-play | Injury markets pulled at CFTC request (Sept 2026); insider/medical-information concerns; not pursuing. |
| Kalshi sports lags sportsbooks | Citizens JMP via casino.org; Blockworks | Kalshi *retail prices* are worse (fees), but gaps are short-lived (Blockworks: corr 0.996, 2.75% cost threshold). Already logged by F3. |
| Sportsbook odds boosts / promos | no quantitative source found | Real +EV in principle (math only), small and one-off, manual; no evidence to cite. |

---------------------------------------------------------------------------------------------------------------------
## 4. Notes for the lead

- **Do #1 first.** It needs no new edge, only the fill simulation on E1 dev picks plus a forward A/B. Everything needed
  is the public trade tape (no auth). It could plausibly add 0.5-1% of stake to E1 (fee saved plus spread), less
  adverse selection. That is the unknown the test measures.
- #2 and #3 share one data pull (settled markets + hourly candles across categories). Build one puller and
  pre-register both rules before looking.
- Becker's full 36 GiB dataset would answer #2/#3 offline, but the disk is 97% full. Use the API per series instead.
- E1 crowding watch: open-source KXHIGH bots (31-member GFS ensemble, trade either side when |edge| > 8%) price
  the same brackets from the same free data. If they spread, E1's NO asks should rise toward fair value. Track E1's
  entry NO price vs the model probability over time as an early decay signal (my inference, not measured).
