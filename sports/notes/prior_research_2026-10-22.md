# Prior-research sweep 2026-10-22: prediction markets, sports betting, retail equity/options anomalies

Brief (research agent, 2026-10-22): a systematic sweep of academic and credible industry research from 2015-2026 on
(1) Kalshi/Polymarket/PredictIt inefficiencies, (2) sports-betting inefficiencies that survived publication, and
(3) retail-accessible equity/options anomalies that survive costs after 2015. For each one: citation, sample, net effect,
decay evidence, and a testable spec on free data. Then rank the top 10 by effect x evidence x feasibility for a US/CA
retail account. It does not repeat notes/research_prop_models.md (props, Levitt, Moskowitz momentum, NBA/NFL lines)
or notes/research_situational_vars.md (rest, travel, refs, weather).

**Evidence labels.** **[FULL]** = I read the full text and the numbers are quoted from it. **[ABS]** = the numbers come
from the abstract or the official landing page. **[UNVERIFIED]** = search snippet, press coverage or a secondary summary
only; do not quote these as fact until someone reads the paper. I made up no numbers. Where a source does not give a
figure, I say so.

**Cross-check against our own ledger** (notes/results.md, EDGES_AND_AGENTS.md). Several literature results are things
we already found ourselves: Kalshi longshots lose badly; sports fav >= 85% +2.0% p .062; the maker MK1 dog lead; the
soft-vs-Pinnacle soccer gap (E3); parlays compound negative edge; MLB open-vs-close shows no favourite-longshot bias.
The ideas below are the ones that are either NEW to us or that sharpen a lead we already have.

General decay anchor: McLean & Pontiff (2016, *JF* 71(1)), 97 published cross-sectional predictors. Returns are 26%
lower out of sample and 58% lower after publication [ABS]. Expect any published equity anomaly to keep at most half its
paper effect. https://www.gwern.net/doc/economics/2016-mclean.pdf

---------------------------------------------------------------------------------------------------------------------
## TOP 10, ranked (effect after costs x strength of evidence x feasibility for a US/California retail account)

| Rank | Idea | Literature effect (net) | Evidence | Feasibility | Data we have |
|---|---|---|---|---|---|
| 1 | **K-FAV: Kalshi favourite side (88-97c) as a MAKER, non-sports daily categories** | Makers buying >= 50c earn +2.6% avg; contracts > 70c have significant small positive post-fee returns; 95c wins ~98% (+3.1% pre-fee) | Strong [FULL]; every category rejects efficiency; bias weaker in 2025 | High: same executor, same API | Kalshi trades API with taker_side |
| 2 | **E1/E2 maker execution** (post our live weather NO orders as resting limits instead of taking) | Average maker return -9.64% vs taker -31.46%, and the gap is concentrated in cheap contracts [FULL] | Strong on the mechanism; our own ladder-maker test warns about adverse selection | High; a pure execution change | data/kalshi_maker_candles*.parquet plus trade prints |
| 3 | **PredictIt political longshot/NO-bundle scan for the 11-03 midterms** | Bid sums "often" > 110, i.e. arbitrage after fees; longshot overpricing larger than other markets (2016-22) | Medium [ABS], NBER 2026 | High: free public API, legal in the US; time-boxed to 11-03 | PredictIt public marketdata JSON (no key) |
| 4 | **Kalshi political underconfidence: buy political favourites** | Political prices compressed toward 50% on Kalshi and Polymarket; no return figure given | Medium-weak [ABS]: one arXiv paper; Bürgi et al. find the politics slope not significant | High; midterms give a forward sample now | Kalshi political market history 2022-26 |
| 5 | **Mention markets: buy NO** (earnings-call, FOMC, speech mention contracts) | "Yes" overpaid relative to "No"; no return figures published | Weak-medium [UNVERIFIED]: SSRN abstract via Quantpedia, plus a snippet-level Bartlett-O'Hara stat | High; registry lead L10 | Kalshi KXEARNINGSMENTION* / mention series history |
| 6 | **Crypto threshold contracts vs option-implied probability** (Kalshi BTC/ETH daily above/range vs Deribit) | Directional bets track options; path-dependent and tail bets deviate; mispricing largest at inception and near expiry | Medium [UNVERIFIED/ABS]: FC'26 workshop paper, SSRN 2026, RePEc 2026 | Medium: Deribit public data is read-only; risk-neutral is not physical probability | Kalshi KXBTC*, Deribit public API |
| 7 | **Kalshi combos/parlays priced above the product of their legs: buy NO on the combo** | Overpricing grows with leg count, even where the legs themselves are calibrated | Medium-weak [ABS]: one arXiv paper, no net-return figure | Medium: must check that NO on combos is tradable and what the combo fee multiplier is | Kalshi multivariate (combo) market history |
| 8 | **Earnings straddle bought 3 days before** (L11) | +3.34% avg 3-day ATM straddle (JFQA); net positive only for short-dated, tight-spread options; retail buyers holding through lose 5-9% | Medium [ABS]; no post-2015 replication found | Medium: Robinhood option history is daily OHLC with no bid/ask | Robinhood MCP option historicals (peer session) |
| 9 | **Soccer soft books vs Pinnacle (E3), US-legal execution** | Kaunitz et al. 3.5% simulated, 8.5% live, then accounts limited; 3 of 11 leagues inefficient at best odds | Strong overall, and our own sealed pass | Low in CA (no legal soft 1X2 books with these prices) | football-data.co.uk (already used) |
| 10 | **Turn-of-month / T-3 liquidity reversal (SPY)** | Reversal around the last settlement day before month-end; high returns around the turn of the month | Medium [UNVERIFIED on numbers]; RFS 2020 | High, near-zero cost, but small and not a new kind of edge | yfinance SPY 1993-2026 |

Why 1-2 rank above everything else. They are the only items where (a) a paper we read in full shows a net-of-fee gain,
(b) the venue is one we already trade live, and (c) our own data independently agrees: Kalshi sports fav >= 85% +2.0%
p .062; longshots <= 15% -45%; E1 itself is a "buy the ~90c NO" trade.

---------------------------------------------------------------------------------------------------------------------
## 1. Prediction markets (Kalshi / Polymarket / PredictIt)

### 1.1 Bürgi, Deng & Whelan, "Makers and Takers: The Economics of the Kalshi Prediction Market" [FULL]
- Versions: UCD WP2025_19 (Sept 2025); GWU CER WP 2026-001; CEPR DP20631; CESifo WP 12122. PDF:
  https://www.karlwhelan.com/Papers/Kalshi.pdf , https://cepr.org/publications/DP20631
- Data: Kalshi API transaction data from inception (2021) to April 2025. Markets with final volume >= $1,000 and final
  spread <= 20c; hourly markets excluded. 46,282 YES contracts in 12,403 events; daily snapshots from close back to
  10 days, giving 156,986 YES prices and 313,972 YES+NO prices. 2/3 of prices are < 10c or > 90c. Median contract
  volume is $8,982; median trade $35.
- Findings (quoted):
  - Contracts costing <= 10c lose "over 60%". Contracts above 70c show "statistically significant, though small,
    positive post-fee returns". A 95c contract that wins 98% returns +3.1% pre-fee.
  - Average pre-fee return -20%; after fees -22%.
  - Makers -9.64% vs Takers -31.46% on average. "Makers who buy contracts costing 50c and over earn a 2.6% rate of
    return." Makers' share of buys is 43.5% in the 1-10c bin and 56.5% in the 90-99c bin.
  - The bias holds in every volume quintile and every trade-size quintile; the largest-trade quintile has the largest
    bias. It holds at every horizon from 0 to 10 days before close. At the final-day close, makers lose on cheap
    contracts much like takers do.
  - Category slopes (Table 8): every category rejects efficiency. Climate & Weather 0.031, Financials 0.032,
    Economics 0.034, Crypto 0.058; Politics 0.022 and Entertainment 0.020 are not significant.
  - Decay (Table 9): slope 2021 .041, 2022 .023, 2023 .036, 2024 .048, 2025 .021*. 2025 is weaker, but that is
    Jan-Apr only.
  - Fees in their sample: takers paid 0.07·P·(1-P) per contract, rounded up; makers paid nothing until April 2025.
- Caveat: these are average returns over traded prices, with contracts appearing on up to 11 days, not a tradable
  rule with fill modelling. Our ladder-maker test (results.md 10-18) found resting orders do NOT earn the spread net
  of adverse selection, so fills must be modelled.

### 1.2 Kalshi fee structure (affects every Kalshi spec)
- Taker fee = ceil(0.07·C·P·(1-P)). From 2025-07-01 the maker fee changed from a flat 0.25c per contract to
  ceil(0.0175·C·P·(1-P)) in covered markets, with per-series multipliers (some series have taker multiplier 0; combo
  markets have maker multiplier 2) [UNVERIFIED: InGame + Allium docs, not Kalshi's own page].
  https://www.ingame.com/kalshis-change-may-increase-fee-sports-traders/ ,
  https://docs.allium.so/historical-data/predictions/kalshi/series-fee-changes
- **Implication for our $1 units:** fees round up to the cent per order, so a 1-contract order pays 1c whether maker
  or taker. The maker advantage at $1 size is only the price improvement. The fee advantage appears at >= ~10
  contracts per order.

### 1.3 Crosier (2026), "Prediction Markets Beat the Weather Forecast on Tomorrow's High Temperature" [ABS]
- arXiv 2609.23969, https://arxiv.org/abs/2609.23969 . Five years of Kalshi high-temperature markets, 7 cities.
  In 6 of 7 cities the market beats the National Blend of Models; RMSE about 10% lower after the first trading hour,
  and the advantage persists into the target day. NBM moves toward the market about 4x as far as the market moves
  toward NBM. No trading results.
- Relevance: our E1 does NOT rely on beating the market's centre. It buys NO on brackets that all 5 models reject, i.e.
  it harvests the longshot overpricing in 1.1. This paper says a forecast-vs-centre strategy (our N1/X2 failures) was
  the wrong shape. Keep E1-style tail trades; don't build "better point forecast" trades.

### 1.4 Polymarket evidence
- Cardozo & Rivero-Wildemauwe (2026), "The Favorite-Longshot Bias in Prediction Markets: Evidence from Polymarket",
  arXiv 2609.12878 [ABS]. 588M trades, 2.48M accounts. Trade-weighted: buys < 10c lose 19.3c per $; buys >= 90c earn
  +0.83c per $. Contract-weighted, longshots lose 6.3c; grouped by parent event, longshots GAIN 4.1c. The two-sided
  pattern holds in Crypto and Politics, **largely absent in Sports**. https://arxiv.org/abs/2609.12878
- Qin & Yang (2026), "Polymarket-v1 Database", arXiv 2606.04217 [ABS]: every on-chain trade from 2022-11 to 2026-04,
  1.20B trades, 1.30M markets, ground-truth aggressor side; tick rule is 49.8% accurate. Free on Hugging Face
  (TimeSeventeen/Polymarket-v1). A ready-made dataset for maker/taker tests on Polymarket. https://arxiv.org/abs/2606.04217
- Le (2026), "Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics in Prediction Markets", arXiv 2602.19520
  [ABS]: 353M trades, 429k contracts on Kalshi + Polymarket. Most robust pattern: **political markets are
  underconfident (prices compressed toward 50%)** on both venues. Large political trades on Kalshi compress further
  (slope gap ~0.5), which does not replicate on Polymarket. About half the raw slope variation is noise.
  https://arxiv.org/abs/2602.19520
- Akey (2026, SSRN), "Who Wins and Who Loses in Prediction Markets? Evidence from Polymarket" [UNVERIFIED]: BTC/ETH
  Polymarket prices track option-implied benchmarks, with larger gaps in tails, high volatility and after macro
  shocks; a summary says 68.8% of users lost money. Fabi et al. (FC'26 DeFi workshop), "Market Efficiency in
  Prediction Markets" [UNVERIFIED]: Above/Range bets match options; Reach/Dip (path-dependent) deviate; mispricing
  largest at inception and near expiry, and on weekends. https://fc26.ifca.ai/defi/papers/market-efficiency-prediction-markets.pdf
  Portnaya (2026), RePEc arXiv 2606.19517 [UNVERIFIED]: Polymarket BTC thresholds vs Binance options, mean gap 5.6 pp in
  the main Sept-2023 contract. https://ideas.repec.org/p/arx/papers/2606.19517.html
- Deleep, Lee, Bai, Suresh & Dhawan (2026), "How Wise is the Crowd? Bias and Edge in Prediction Markets", SSRN 6322678
  [UNVERIFIED, via Quantpedia]: in Mention Markets, YES is overpaid relative to NO. The apparent favourite-longshot
  bias is an artefact once contract lifecycle timing is controlled for. Whales lose to smaller traders. No return
  figures. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6322678 ,
  https://quantpedia.com/how-wise-is-the-crowd-in-prediction-markets/
- Bartlett (Stanford) & O'Hara (Cornell) [UNVERIFIED, one press snippet, paper not located]: 41.6M trades across
  478,167 Kalshi markets. In single-name markets traders buy YES ~61% of the time; YES wins ~32%.

### 1.5 Kalshi sports-specific
- Moshrefi (2026), "Prices, Probabilities, and Parlays: Systematic Bias in Sports Prediction Markets", arXiv 2607.14430
  [ABS]: ~23M Kalshi moneyline trades. Calibration is near-perfect mid-contract and departs sharply in the final 10
  minutes ("step-like", Prelec curvature > 1, attributed to insurance-like demand from losing positions). **Cross-game
  parlays are priced above the product of their legs, growing with leg count**, even when the legs are calibrated. No
  net-return figures in the abstract. https://arxiv.org/abs/2607.14430
- Blockworks Research (industry, 2026) [UNVERIFIED]: 3.1k games, $8.5B notional; Kalshi tracks sportsbooks
  (corr up to .996); ~2.75% cost threshold leaves few arbitrage windows.
  https://app.blockworksresearch.com/unlocked/from-betting-to-trading-how-kalshi-is-reshaping-sports-markets
- Agrees with our DEAD list: in-game lag closed within ~1 minute; late-game calibration fine; soccer 1X2 made off the
  Pinnacle line.

### 1.6 Macro/economic-data markets
- Diercks, Katz & Wright (2026), "Kalshi and the Rise of Macro Markets", FEDS 2026-010 [ABS + press]:
  Kalshi beats the Bloomberg consensus on headline CPI (significant), ties on core CPI and unemployment; the modal
  forecast called every FOMC decision since 2022 the day before [press, UNVERIFIED detail].
  https://www.federalreserve.gov/econres/feds/files/2026010pap.pdf
- Relevance: a consensus-based CPI model will not beat Kalshi on the centre, which agrees with our CPI DEAD result. Any
  edge left is the tail longshot bias (1.1, Economics slope .034, significant).

### 1.7 PredictIt and election markets
- Zitzewitz (2026), "Limits to Arbitrage and Prediction Market Efficiency: the Case of PredictIt", NBER w35845 [ABS]:
  2016-22, $850 per-contract cap. Bid prices "often persistently" summed to > 110, an arbitrage profit even after
  fees. Longshot overpricing larger than in other prediction markets. https://www.nber.org/papers/w35845
- PredictIt fees, 10% of profits + 5% withdrawal [UNVERIFIED: forum comment]. Whether the position cap has been raised
  since 2022 is also UNVERIFIED; check predictit.org before any spec.
- Sethi (2016) blog: NO bundles in multi-outcome markets routinely cost less than the guaranteed payout.
  https://rajivsethi.substack.com/p/fee-structure-distortions-in-prediction-16-04-10
- Clinton & Huang (Vanderbilt, 2025) [UNVERIFIED, press only]: final 5 weeks of the 2024 election; share of markets
  beating a chance benchmark was PredictIt 93%, Kalshi 78%, Polymarket 67%; inefficiency rose in the final two weeks.
  https://www.dlnews.com/articles/markets/polymarket-kalshi-prediction-markets-not-so-reliable-says-study

---------------------------------------------------------------------------------------------------------------------
## 2. Sports betting (only what isn't already in research_prop_models / research_situational_vars)

| Study | Sample | Finding | Post-publication | Label |
|---|---|---|---|---|
| Kaunitz, Zhong & Kreiner (2017), "Beating the bookies with their own numbers", arXiv 1710.02824 | 10 yrs closing odds, 6 months of minute odds, 5 months of real bets | Bet when a soft book's odds exceed the cross-book average-implied fair price: 3.5% simulated; 8.5% real money; **accounts limited within months** | Our S1 Kaunitz replication on football-data was -3.4%/873; our power de-vig vs Pinnacle variant (E3) passed sealed | numbers [UNVERIFIED: press]; abstract [ABS] https://arxiv.org/abs/1710.02824 |
| Angelini & De Angelis (2019), *IJF* 35(2):712-721 | 41 books, 11 European leagues, 11 yrs | With best odds, 8 leagues efficient and 3 inefficient (profitable thresholds ex ante) | which 3 leagues not confirmed | [ABS] https://ideas.repec.org/a/eee/intfor/v35y2019i2p712-721.html |
| Hegarty & Whelan (2023-24), CEPR DP17948/17949 and MPRA 117243 | >80k European soccer games | 1X2 shows a favourite-longshot bias; Asian handicap does NOT; competitive books shouldn't show one; inverse-odds tests are biased against finding it | n/a | [ABS] https://cepr.org/voxeu/columns/forecasting-soccer-matches-betting-odds-tale-two-markets |
| Deutscher, Winkelmann & Ötting (2021), *Applied Economics* 53(26) | 83 Bundesliga ghost games 2019/20 | Books didn't price the lost home advantage; ~15% ROI (preprint) | transient by construction | [UNVERIFIED number] https://arxiv.org/abs/2008.05417 |
| Hubáček, Šourek & Železný (2019), *IJF* 35(2):783-796 | NBA 2007-14 | Penalising correlation with the bookmaker gives positive cumulative profit | our NBA decorrelated test (results 10-17) failed: +0.6% p .40 | [ABS] https://ideas.repec.org/a/eee/intfor/v35y2019i2p783-796.html |
| Croxson & Reade (2014), *Economic Journal* | Betfair, goals just before half-time | Prices update swiftly and fully: in-play is efficient on goals | n/a | [ABS] https://centaur.reading.ac.uk/34884 |
| Angelini, De Angelis & Singleton (2022), *IJF* | Betfair in-play, first goal as news | Mispricing (favourite bias) both pre-match and in-play, larger after surprise late goals by longshots | n/a | [ABS] https://research.reading.ac.uk/football-research/new-journal-article-informational-efficiency-and-behaviour-within-in-play-prediction-markets/ |
| Krieger, Pace, Clarke & Girdner (2015), *Financial Services Review* | NFL/NBA teams that have clinched | Opening spreads profitable against clinched teams; closing lines partly correct; mostly gone after commission except some NFL spots | n/a | [UNVERIFIED] https://openjournals.libs.uga.edu/fsr/article/view/3242 |
| Abinzano, Muga & Santamaría (2016), *Applied Economics Letters* | Betfair tennis | Favourite-longshot bias stronger for low-ranked matches, later rounds, high-profile events; no net strategy reported | n/a | [UNVERIFIED] https://academica-e.unavarra.es/handle/2454/18748 |
| Woodland & Woodland (NHL, 2011 + Economics Bulletin follow-up) | NHL totals/moneylines, 1990s-2000s | Reverse favourite-longshot bias / totals inefficiency; a 2018 thesis finds moneyline favourite-longshot bias too small to profit | old data, no post-2015 work found | [UNVERIFIED] https://commons.emich.edu/accfin_facsch/91 |
| Wolfers (2006); Sinkey & Logan; Colquitt et al. (2001) | NCAA | Heavy NCAAB favourites win but fail to cover; NCAAF favourites overpriced; conference markets efficient given information | old; nothing post-2015 found | [UNVERIFIED] |
| Moskowitz (2021), *JF* 76(6) | >100k contracts, 4 US leagues | open->close momentum, reversed by outcome; **does not cover costs** | (already in research_prop_models) | [ABS] |

Take-away. Inefficiencies that survived publication are the ones arbitrage cannot reach: soft-book vs sharp-book gaps
(accounts get limited, which is exactly the Kaunitz experience), position-capped venues (PredictIt), and
transient regime shifts (ghost games). Classic "angles" in liquid US markets have decayed, which matches our 0-for-many
NBA/NFL/MLB angle tests. No post-2015 peer-reviewed evidence turned up for NHL, college or tennis edges after costs.

---------------------------------------------------------------------------------------------------------------------
## 3. Retail equity / options

| Anomaly | Citation | Sample | Effect | Post-pub / post-2015 | Verdict |
|---|---|---|---|---|---|
| Earnings straddles (long) | Gao, Xing & Zhang (2018), *JFQA* 53(6):2587-2617 | WP: 1996-2010 | ATM straddle held from 3 days before through the announcement: +3.34% (published); 1-day: +2.3% (WP). Net positive only for 4-10 DTE straddles at half-spread (1.64%/day) or tight-spread names at full spread (1.16%/day); turns to loss with longer expiry or full spread | No replication after 2015 found | LEAD (rank 8) [ABS + CXO summary of the WP] https://www.cxoadvisory.com/?p=22707 |
| Retail options around earnings | de Silva, Smith & So (2025), *Review of Finance*, "Losing is Optional" | 2019-21 era retail flow (period UNVERIFIED) | Retail buyers lose 5-9% on average, 10-14% for high-expected-volatility announcements: overpaying vs realised vol, spreads, slow exit | recent | Tells us how NOT to run rank 8: exit at the first post-announcement close; never hold weeks. [ABS] https://www.gsb.stanford.edu/faculty-research/publications/losing-optional-retail-option-trading-expected-announcement |
| Short earnings straddles / IV crush | practitioner backtests only (OptionStrat NVDA: -69% avg over 25 events, naked; Option Strategist; tastytrade) | various | mixed; naked short loses to jumps | no academic post-2015 study found | DEAD for retail [UNVERIFIED] |
| Post-earnings drift | Martineau (2022), *Critical Finance Review* 11:613-646 | US | large caps: no PEAD since 2006; microcaps only recently gone | 2025: two accepted papers say PEAD is alive (UCLA Anderson brief); Subrahmanyam says it's research design | DEAD for liquid names [ABS] https://cfr.ivo-welch.info/published/papers/martineau2021rest.pdf |
| Earnings announcement premium | Barber et al. (2013); Heitz, Narayan & Zekhnin, "The Disappearing Earnings Announcement Premium" | US + intl | gone in the US (shifted to 8-K periods after 2004) | - | DEAD in the US [UNVERIFIED] |
| Overnight drift (2-3am ET futures) | Boyarchenko, Larsen & Whelan (2023), *RFS* 36(9):3502-47 | 1998-2019 | ~3.6%/yr annualised from that one hour | **Authors' own 2026 Liberty Street post: flat Jan 2021-Dec 2025** (ES, NQ, YM); closing order imbalances halved | DEAD [UNVERIFIED snippet of the 2026 post] https://libertystreeteconomics.newyorkfed.org/2026/07/the-disappearing-overnight-drift/ |
| Pre-FOMC drift | Lucca & Moench (2015) *JF*; Kurov, Wolfe & Gilbert (2021), *FRL* 40 | 1994-2011; extended to 2019 | +49 bp in the 24h before FOMC | essentially gone after 2015 | DEAD [ABS] https://www.skidmore.edu/economics/documents/KurovWolfeGilbert-TheDisappearingPre-FOMC-Announce-Drift-200914.pdf |
| S&P 500 index additions | Greenwood & Sammon (2025), *JF* 80(2):657-698 | 1980s-2020 | 7.4% in the 1990s -> 0.3% in the last decade (published abstract) | gone | DEAD [ABS] https://papers.ssrn.com/abstract=4294297 |
| 0DTE options | Beckmeyer, Branger & Gayda (2023); Vilkov (2023-24) "0DTE Trading Rules" | SPX 2016-24 | intraday variance risk premium about 0.001%; no strategy consistently profitable at expiry; retail net ~0, losses mostly spreads | - | DEAD [UNVERIFIED: secondary summaries] https://cxoadvisory.com/equity-options/0dte-strategy-outcomes |
| Put-writing (index variance risk premium) | Bondarenko (2016, CBOE-sponsored) | 1986-2015 | PUT 10.1% vs S&P TR 9.8% gross, lower vol; 2006-15 lower raw return (6.6 vs 7.1%) | no post-2015 study found | risk premium, not alpha; not a target [UNVERIFIED] |
| Robinhood herding reversal | Barber, Huang, Odean & Schwarz (2022), *JF* | 2018-20 | top bought stocks -4.7% over 20 days; extreme herding -19.6% (July 2021 draft) | the Robintrack feed ended in 2020; needs shorting | not feasible [ABS/UNVERIFIED figures] |
| Month-end liquidity (T-3) reversal / turn of month | Etula, Rinne, Suominen & Vaittinen (2020), *RFS*, "Dash for Cash" | US + intl, to ~2014 | reversal around T-3; explains turn-of-month returns | none found | LEAD (rank 10), cheap [UNVERIFIED numbers] https://orbilu.uni.lu/bitstream/10993/21145/1/Dash4Cash.pdf |
| Leveraged-ETF end-of-day rebalancing | Barbon, Beckmeyer, Buraschi & Moerke (2021); Rodier, Haryanto, Shum & Hejazi (2012) | 2006-11 and later | gamma imbalance predicts end-of-day momentum/reversal; 0.60%/trade gross (third-party summary) | - | needs intraday data, not free; skip [UNVERIFIED] |

Take-away. For a retail account, the equity calendar and flow anomalies that were large enough to matter have mostly
been published and arbitraged away (overnight, pre-FOMC, index adds, PEAD, earnings premium). Our 25 failed ETF/stock
trials agree. Options anomalies that survive tend to be short-dated and spread-sensitive, which is the worst case for
daily OHLC data. Only the pre-earnings long straddle (rank 8) has a peer-reviewed net-positive result, and it is
pre-2015 and spread-dependent.

---------------------------------------------------------------------------------------------------------------------
## 4. Testable specs for the top 10 (pre-register each in the script docstring; ledger + Bonferroni per protocol.md)

**1. K-FAV (Kalshi favourite side as maker).**
- Universe: settled Kalshi markets 2023-01..2026-10, excluding sports (the Polymarket evidence says sports has no
  favourite-longshot bias; our sports G1 is already in the ledger) and hourly series. Volume >= $1,000; spread <= 20c
  at decision time. Categories: Financials (index ranges), Economics, Climate (non-E1 series), Crypto daily, Other.
- Decision at T-24h and at T-3h (2 trials). If one side's best bid is 88-97c, rest a buy at that bid for 1 contract.
  Fill = a later print at or through our price (trade-through, as in MK1) before close.
- Fees: maker 0.0175 formula after 2025-07-01, 0 before; also report the taker variant.
- Dev 2023-01..2025-06. Sealed 2025-07..2026-10, which is also the post-maker-fee regime.
- Pass: ROI > 0, p < bar, >= 4/6 quarters positive, AND worst-quarter loss < 3x the mean quarterly gain (a 90c buy
  loses ~9x what it wins).
- Report by category, because politics/entertainment slopes were not significant.

**2. E1/E2 maker execution.**
- Replay the sealed E1/E2 pick list. At the 22:00 decision, post NO at best NO bid +0c / +1c (2 variants) instead of
  lifting the ask.
- Fill = trade-through before the existing deadline. Unfilled picks may either fall back to taker at the deadline
  (variant A) or be skipped (variant B).
- Metrics: ROI on filled picks vs taker ROI on the SAME picks, fill rate, and the adverse-selection gap (ROI of filled
  vs unfilled picks).
- Data: data/kalshi_maker_candles*.parquet + trade prints.
- At 1-contract size, fees tie at 1c, so this measures price improvement only. Live deployment = an executor flag,
  only if the sealed replay passes.

**3. PredictIt midterms scan (forward, time-boxed to 2026-11-03..certification).**
- Poll https://www.predictit.org/api/marketdata/all/ every 5 minutes; log to data/predictit/.
- Signal A (NO-bundle): in each multi-contract market, compute the cost of buying NO on every contract at best NO ask
  vs the guaranteed payout (N-1 dollars), after 10% of each winning contract's profit and 5% on withdrawal. Log any
  bundle with a guaranteed net profit > 0.
- Signal B (longshot NO): YES <= 10c in multi-candidate markets; paper-buy NO.
- First confirm the current fee schedule and position cap on predictit.org (both UNVERIFIED here).
- Reuse scripts/xvenue_elections.py for Kalshi/Polymarket US prices on the same races.

**4. Kalshi political recalibration.**
- Settled Kalshi political markets 2022-01..2024-12 as dev (fit a logistic recalibration slope by time-to-close
  bucket). 2025-01..2026-10 as sealed. Forward paper: 2026 midterms.
- Rule: buy the side where recalibrated p - price >= 3c + fee, as a maker.
- Expect conflict: Bürgi et al. found the politics slope not significant, so this is a weak prior. Run it only as a
  single pre-registered trial.

**5. Mention markets: buy NO.**
- All settled Kalshi mention series (KXEARNINGSMENTION*, FOMC/press-conference mentions, speech mentions).
- Decision at T-1h before the event start; YES price in 15-60c (excludes the < 10c tail, where Deleep et al. say the
  favourite-longshot pattern is a timing artefact). Buy NO as taker; maker variant as trial 2.
- Group by event (several strikes per call are correlated) and cluster SEs by event.
- Dev = events before 2026-04-01; sealed = after. Pass on event-clustered p.

**6. Crypto thresholds vs options.**
- Kalshi KXBTC*/KXETH* daily "above X" and range markets.
- At T-6h and T-1h, interpolate Deribit option IV at (strike, expiry) from the public read-only API (market data
  only, no account; stop if any geo-block appears), and compute the risk-neutral P(S_T > K).
- Bet when the Kalshi price deviates from risk-neutral P by >= fee + 3c. Separate trials for the tails (P < .15 or
  > .85) and the centre.
- Caveat: risk-neutral probability includes a variance/skew premium, so expect a systematic offset. Estimate the
  offset on dev and treat it as part of the rule.
- History: check how far back Deribit's free history goes before committing.

**7. Combo overpricing.**
- First check via the API that NO on multivariate combos is buyable and what the fee multiplier is.
- Then on settled combos: markup = combo YES price / product of leg YES prices at the same minute.
- Rule: buy NO when markup >= m* (m* chosen on dev), by leg count.
- Our 10-16 parlay result (legs' edge compounds) predicts the YES side loses; this tests whether the NO side profits
  after fees.

**8. Pre-earnings straddle.**
- S&P 100 names, events from earnings.md / Robinhood earnings calendar.
- Buy the nearest-expiry ATM straddle expiring >= the announcement day + 1, at the close 3 trading days before the
  announcement.
- Two exits: (a) the close before the announcement (pure IV run-up); (b) the first close after the announcement
  (Gao, Xing & Zhang).
- Costs: daily OHLC has no bid/ask, so assume full spread taken from same-name current quote snapshots, and report the
  result at 0.5x / 1x / 1.5x that spread.
- First confirm via the connect-robinhood-mcp session how many years of get_option_historicals exist. Under ~3 years
  will not resolve a 2-3% edge; leave this as LEAD.

**9. E3 in the US.** Keep L9 (Colorado books) if the user's home state is CO. In CA it's untradable, so no new test.

**10. T-3 / turn of month.**
- yfinance SPY daily adjusted 1993-2026. Long SPY from the close of trading day T-4 to the close of T+3 around
  month-end, cash otherwise.
- Compare with the T-3 reversal variant (short-term long after the T-3 dip).
- Dev 1993-2014; sealed 2015-2026 (= post-publication). Pass = sealed excess return > 0 at p < bar after 1 bp per
  side.

---------------------------------------------------------------------------------------------------------------------
## 5. Ideas the literature says NOT to spend time on (add to DEAD-by-literature)
- Overnight / 2-3am futures drift (gone 2021-25 per the authors).
- Pre-FOMC drift (gone after 2015).
- S&P inclusion (0.3%).
- PEAD in liquid names.
- US earnings-announcement premium.
- 0DTE strategies.
- Naked short earnings straddles.
- Public-model or angle betting in liquid US sports (Moskowitz: does not cover costs; our ledger agrees).
- In-play goal reaction in soccer (efficient on average).
- Point-forecast-vs-market trades on Kalshi weather or CPI (the market already beats NBM and the Bloomberg consensus).

## 6. Open verification items (cheap, do before relying on the related numbers)
- Read in full: Moshrefi 2607.14430 (parlay markup size, any net returns); Le 2602.19520 (political slopes by
  horizon); Deleep et al. SSRN 6322678 (mention-market Yes-bias size); Zitzewitz w35845 (how often sums > 110 occurred
  after 2020).
- Locate the Bartlett & O'Hara Kalshi paper (title unknown).
- Confirm PredictIt's current fees and cap, and Kalshi's current fee page (maker 0.0175, combo multipliers).
