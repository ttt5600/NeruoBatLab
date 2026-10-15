# Evaluation methodology for betting models, and the economics of prop markets

Scope note: 19 tool calls. Peer-reviewed and primary sources are marked as such. Several prop-market claims are only available from affiliate, pick-seller, or tool-vendor sites; they are flagged **[vendor/anecdotal]**. Arithmetic marked **[derived]** was computed by me from stated payouts and odds, not taken from a source.

## (a1) Calibration vs accuracy for model selection (Walsh & Joshi 2024)

### Takeaway
On NBA moneylines, picking the model by calibration (Brier/ECE-style) instead of accuracy turned average ROI from -35.17% to +34.69%. The direction is credible. The magnitudes come from one league, one betting season and an old closing-moneyline source, so treat them as illustrative, not as an expected edge.

### Cited Findings
- Walsh, C. & Joshi, A. (2024), "Machine learning for sports betting: Should model selection be based on accuracy or calibration?", *Machine Learning with Applications* 16:100539, doi 10.1016/j.mlwa.2024.100539. Abstract: calibration-based selection gives "return on investment of +34.69% versus -35.17%" on average and "+36.93% versus +5.56%" in the best case — [arXiv 2303.06021](https://arxiv.org/abs/2303.06021); [Bath research portal](https://researchportal.bath.ac.uk/en/publications/machine-learning-for-sports-betting-should-model-selection-be-bas/)
- Setup, per a secondary summary: NBA seasons 2014–2019; logistic regression, random forest, SVM and MLP on pre-game box-score features; odds are Westgate closing moneylines; betting is done on a single held-out season. Calibration-driven system ROI 32.45% vs accuracy-driven 5.56% in one reported experiment — [arXiv 2410.21484 systematic review](https://arxiv.org/pdf/2410.21484). I could not confirm the stake-sizing rule or bet count from the abstract. Code: [GitHub conorwalsh99/ml-for-sports-betting](https://github.com/conorwalsh99/ml-for-sports-betting)
- A secondary source restates the result as "69.86% higher average returns". That is the difference of the two ROIs (34.69 − (−35.17) = 69.86 points), not a relative increase — [arXiv 2410.21484](https://arxiv.org/pdf/2410.21484)

### Inferences
- The mechanism: a betting rule acts on p_model − p_implied, so it rewards correct probabilities and does not care about getting the argmax right. Accuracy-selected models are often overconfident, and overconfidence creates fake edges that get bet heavily.
- ROI of ±35% on NBA moneylines is far too large to be a stable edge. It shows high variance from a single test season and probably aggressive sizing. Read the size of the gap as "calibration matters a lot", not as an achievable return.
- For a prop model this means: select on log-loss/Brier against outcomes, and check calibration *conditional on the bet filter* (the bets where model and book disagree). Global calibration can hide miscalibration in the tails, and the tails are where the bets are.

### Gaps
- Exact staking rule, number of bets and confidence intervals: not in the abstract, and I did not open the full paper.

## (a2) Closing line value (CLV) as the evaluation metric

### Takeaway
The best quantitative evidence is Buchdahl's analysis of 87,960 Pinnacle soccer odds pairs. Beating Pinnacle's close by x% predicted realised yield with a slope of about 1.00, less the margin. CLV against a sharp close is a lower-variance proxy for edge than P&L. It only works where a sharp, liquid close exists, and that is weaker for props.

### Cited Findings
- Joseph Buchdahl (football-data.co.uk): 4 seasons from 2012/13, 87,960 odds pairs (home/draw/away, major European football leagues). The ratio of the pre-closing price to Pinnacle's closing price, as expected yield, against actual level-stakes returns shows "an almost perfect 1:1 relationship… slope is almost exactly 1.00". Observed yields run a few percent below expected because Pinnacle's margin was not removed — [football-data.co.uk: Efficiency of the Pinnacle closing line](https://football-data.co.uk/blog/pinnacle_efficiency.php)
- Pinnacle (Head of Trading Marco Blume, as relayed) says beating the closing line is the reliable indicator of long-term +EV ("beat it by 10% and you can expect a 10% profit") — [Pinnacle Betting Resources: market efficiency](https://www.pinnacle.com/betting-resources/en/educational/how-to-solve-a-problem-like-efficiency-part-one/rn8j5rnqj2p8t68p); [Buchdahl on CLV (Pinnacle Odds Dropper interview)](https://www.pinnacleoddsdropper.com/blog/closing-line-value--clv-demystified-by-expert-joseph-buchdahl). Note: the bookmaker itself has an interest in this message.
- Counterpoint: an article questions whether Pinnacle closing prices tell the full story in tennis, i.e. whether bettors can win without beating the close — [Sports Trading Network](https://www.sportstradingnetwork.com/article/do-pinnacle-closing-prices-in-tennis-tell-the-full-story-can-you-win-in-the-long-run-without-beating-them/) (not peer-reviewed)
- Kaunitz, Zhong & Kreiner (2017) used the *consensus* of bookmaker closing odds as the true probability and bet outliers at soft books. It was profitable over a 10-year simulation, a 6-month paper period and a 5-month live period, and the authors' accounts were then restricted ("discriminatory practices"). This is direct academic evidence both that the closing consensus is the benchmark and that winners get limited — [arXiv 1710.02824](https://arxiv.org/abs/1710.02824)

### Inferences
- Why CLV works statistically: per-bet P&L at -110 has SD of about 0.95 units. Per-bet CLV (log ratio of your price to the no-vig close) has SD of a few percent. CLV can therefore detect an edge with roughly 1–2 orders of magnitude fewer bets, *provided the close is efficient*. That condition is the whole assumption.
- For player props the "close" is often thin, posted by soft books, or not recorded. ESPN prop `last_updated` timestamps are post-kickoff (finding in this repo's own commit 7a63074), so "close" means roughly the last posted line, not necessarily a sharp one. CLV against a soft prop close can reward you for agreeing with stale or recreational lines. Benchmark against a sharp source or a multi-book no-vig consensus where possible, and keep realised-outcome evaluation as the final arbiter.
- To validate on your own data: regress realised return on CLV across your history. Slope ≈ 1 means the close is efficient for that market. A slope far below 1 means CLV is not a trustworthy proxy there.

### Gaps
- I found no peer-reviewed test of closing-line efficiency *specifically for US player props*, and no published CLV-vs-profit slope for props.

## (a3) Decorrelating the model from the bookmaker (Hubáček et al. 2019)

### Takeaway
A model can be less accurate than the book and still profit if its errors are decorrelated from the book's. Hubáček, Šourek & Železný add a decorrelation term to the training loss and use portfolio-theoretic bet allocation. They report consistent positive cumulative profit on NBA 2007–2014.

### Cited Findings
- Hubáček, O., Šourek, G., Železný, F. (2019). "Exploiting sports-betting market using machine learning." *International Journal of Forecasting* 35(2):783–796. The model is trained to reduce correlation with bookmaker implied probabilities instead of only maximising accuracy, with CNN predictors and modern-portfolio-theory bet allocation that trades off expected profit against variance. NBA 2007–2014, positive cumulative profit "systematically, as opposed to alternative methods" — [IJF / ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S016920701930007X); [author PDF](http://ida.felk.cvut.cz/zelezny/pubs/ijf.2019.pdf)
- Follow-up: Hubáček & Šír, "Beating the market with a bad predictive model" (arXiv 2010.12508). It formalises how a model worse than the bookmaker can still be profitable when decorrelated — [arXiv 2010.12508](https://arxiv.org/pdf/2010.12508)

### Inferences
- Consequence for features: if the model's inputs are the same public box-score stats the book already prices, it will mostly reproduce the line, which means no edge after vig. The edge has to come from information or structure the book underweights (e.g. coverage-scheme matchups, game-script). The value of a feature should be judged by its *partial* contribution once the market line is included as a feature or offset.
- Practical diagnostic: report corr(p_model − p_book, outcome − p_book). An edge exists only if the residual against the book is predictive. Raw accuracy does not show this.

### Gaps
- No replication on props found.

## (a4) Backtest overfitting and multiple testing with hundreds of trials

### Takeaway
With hundreds of strategy variants, the best backtest is mostly selection noise. Correct for it: (1) record the number of trials N; (2) deflate the best Sharpe/ROI by the expected maximum under the null (Deflated Sharpe Ratio); (3) estimate PBO with CSCV; (4) use a data-snooping test (White RC / Hansen SPA) over the whole family; (5) require t ≳ 3 (Harvey-Liu-Zhu); (6) keep an untouched lockbox.

### Cited Findings
- Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest Overfitting", *Journal of Computational Finance* (2017). Proposes Combinatorially Symmetric Cross-Validation (CSCV): split the T×N returns matrix into S blocks, form all C(S, S/2) train/test combinations, and measure how often the in-sample best strategy ranks below the median out-of-sample. That frequency is the PBO — [PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality", *Journal of Portfolio Management*. Raises the significance threshold for a reported Sharpe according to the number of trials, skewness and kurtosis. The chance of selecting an overfit strategy grows rapidly with trials — [Bailey et al., backtest overfitting overview](https://www.davidhbailey.com/dhbpapers/overfit-tools-at.pdf); [portfoliooptimizationbook 8.3](https://portfoliooptimizationbook.com/book/8.3-dangers-backtesting.html)
- Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns", *Review of Financial Studies* 29(1):5–68. A new factor should clear t > 3.0. Across correlation specifications, t ≈ 3.9 controls FWER at 5% and t ≈ 3.0 controls FDR at 1% — [RFS](https://academic.oup.com/rfs/article-abstract/29/1/5/1843824); [Duke PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P118_and_the_cross.PDF)
- White's Reality Check (Econometrica 2000) and Hansen's Superior Predictive Ability test (JBES 2005). Both are bootstrap tests of whether the *best* of a family of rules beats a benchmark, accounting for the search. I did not fetch either source this session; the descriptions are from background knowledge and need citation confirmation.

### Inferences (application to a betting-strategy search)
- Benchmark per bet = 0 units (or −vig for the "bet everything" null). Strategy returns are per-bet unit P&L aligned on a common calendar (e.g. per week), giving a T-weeks × N-strategies matrix. That matrix is the input to CSCV/PBO and to RC/SPA.
- **Block by time, not by bet.** Bets in the same game or slate are correlated (same-game props, weather, game script), so i.i.d. bootstrap understates variance. Use a stationary/block bootstrap over weeks and CSCV blocks of whole weeks or seasons.
- Deflated threshold, back-of-envelope: the expected max of N independent standard normals is about sqrt(2 ln N) — ≈3.0 for N=100, ≈3.4 for N=300, ≈3.7 for N=1000 [derived]. So a t-stat of about 3 from the best of a few hundred trials is consistent with *zero* edge. Correlated trials lower the effective N, so count effective independent trials, not raw ones.
- Keep a ledger of every trial (this repo already has `prop_loop_ledger.jsonl`, which is the right instinct). The deflation needs N, and N is exactly what gets forgotten.
- A sealed lockbox season that is evaluated once is the cleanest remedy. No statistical correction fully replaces it.

### Gaps
- No published application of PBO/SPA to sports-betting strategy searches found in this session.

## (a5) Kelly and fractional Kelly under estimation error

### Takeaway
Full Kelly computed on an estimated edge overbets whenever the edge is overestimated, which is the usual case after selection. If the estimate is 2× the true edge, growth drops to zero. Half Kelly keeps about 75% of the growth with about half the volatility. With selection-inflated edges, ¼–½ Kelly or flat staking is the defensible default.

### Cited Findings
- MacLean, Thorp & Ziemba (eds.), *The Kelly Capital Growth Investment Criterion* (2011); MacLean, Thorp & Ziemba, "Long-term capital growth: the good and bad properties of the Kelly and fractional Kelly capital growth criteria" (*Quantitative Finance* 2010). Fractional Kelly trades growth for security, and overbetting is costly — [ResearchGate](https://www.researchgate.net/publication/227623956_Long-term_capital_growth_the_good_and_bad_properties_of_the_Kelly_and_fractional_Kelly_capital_growth_criteria); [Ziemba chapter PDF](https://webhomes.maths.ed.ac.uk/mckinnon/blackouts/StochOptFinanceAndEnergySpringer/Chap1_KellyZiemba.pdf); [book review, Quantitative Finance](https://www.tandfonline.com/doi/full/10.1080/14697688.2011.619561)
- Claims via secondary sources that I did not verify in the primary text: if the estimated drift is twice the true drift, full Kelly reduces true growth to zero; half Kelly gives about 75% of full-Kelly growth at about 50% of the volatility (MacLean, Ziemba & Blazenko 1992, *Management Science*); P(50% drawdown) under full Kelly is about 50%; errors in means vs variances vs covariances matter roughly 20:2:1 (Chopra-Ziemba) — [Medium summary](https://medium.com/@tmapendembe_28659/the-dangers-of-full-kelly-criterion-why-most-traders-should-use-fractional-kelly-criterion-instead-0338e3bcc705) **[secondary]**

### Inferences
- These numbers follow from growth g(f) ≈ f·μ − f²σ²/2. Betting a fraction c of the true Kelly fraction gives growth c(2 − c) of the maximum: c=0.5 → 75%, c=2 → 0 [derived]. When the estimated edge is inflated by k, "full Kelly" is really c = k.
- Selection bias inflates the estimated edge of the bets you choose (winner's curse), so k > 1 is the expected case. Shrink the edge toward 0 (empirical Bayes) *before* applying Kelly, then use a fraction.
- Simultaneous correlated props (same game) need joint Kelly over the correlated set, or a cap per game. Independent Kelly on each leg overbets the shared factor.

### Gaps
- No betting-specific empirical study of fractional-Kelly performance under estimated edges found.

## (a6) Minimum sample size to detect a 2–5% ROI edge at -110

### Takeaway
At -110 the per-bet SD is about 0.95 units. Detecting ROI = 2% with 80% power takes about 14k independent bets at one-sided α=0.05, or about 18k at z=1.96. For 3% it is about 6–8k, and for 5% about 2.3–2.8k. Correlated bets need more. This is why outcome-only evaluation of a prop model is under-powered for years. [derived]

### Cited Findings
- No source found that tabulates this. All numbers below are **[derived]** from the binomial at -110 (win +0.909, lose −1; breakeven p = 0.5238).

### Inferences
- ROI r ↔ win rate: p = (1 + r)/1.9091. r=2% → p=53.4%; r=3% → 54.0%; r=5% → 55.0%.
- Per-bet return variance = p(1 − p)·1.909² ≈ 0.91, so SD ≈ 0.95.
- n ≈ ((z_α + z_β)·σ / r)² with z_α=1.645 (one-sided 5%) and z_β=0.84 (80% power): r=2% → ~14,000; r=3% → ~6,200; r=5% → ~2,250. With z_α=1.96, a two-sided-equivalent bar: r=2% → ~17,800; 3% → ~7,900; 5% → ~2,850. Significance alone (50% power, z=1.96): 2% → ~8,700; 5% → ~1,400.
- A Harvey-style t ≥ 3 bar (for a strategy chosen from many) roughly doubles these: n ∝ t², and (3/1.96)² ≈ 2.3.
- Effective n is smaller than the bet count when bets are correlated (same game/slate). Divide by a design effect, estimated from a week-block bootstrap.
- This is consistent with this project's own prior estimate ("~4.5 seasons to confirm a 2% edge" in the strikeout-props backtest memory).
- Hence: use CLV (a2) as the primary fast metric and realised ROI as the slow confirmation, with intervals reported.

### Gaps
- None for the arithmetic. The correlation design effect for props is data-specific.

## (b1) Documented inefficiencies in player and team props

### Takeaway
I found **no peer-reviewed study** that directly measures US player-prop pricing bias (over/under, star shading, recency, alt lines). The "public bets overs, so unders are +EV" thesis is widely repeated, but only by affiliate and pick-seller sites. The adjacent academic evidence (less efficient low-volume markets, NBA early-season totals bias, sportsbook overreaction) makes it plausible but does not establish it. Treat it as a hypothesis to test on your own line data.

### Cited Findings
- "Inefficiencies are more prevalent in player prop and alternative lines markets which have less volume and visibility" — Claremont McKenna senior thesis on NFL moneyline efficiency. This is a thesis, not peer-reviewed, and its prop claim is a framing remark rather than a measured result — [CMC thesis](https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=5145&context=cmc_theses)
- NBA early-season totals: betting unders/against closing totals in the first weeks reportedly hit 56.72% over 20 years. This reached me via an affiliate site citing "academic research", and I did not locate the primary paper — [nbabettingsystem.com](https://nbabettingsystem.com/articles/nba-betting-market-efficiency/) **[vendor; primary not verified]**; related: [Sport Journal, "NBA Gambling Inefficiencies: A Second Look"](https://thesportjournal.org/article/nba-gambling-inefficiencies-a-second-look/)
- Price moves in NFL/NBA/NHL pre-game markets show significant negative autocorrelation, i.e. books overreact — "Inefficiencies in Moneyline Movement for Three Major Sports" (2025) — [SAGE](https://sage.cnpereading.com/doi/10.1177/15586235251394815). Season-win-total markets (MLB, NFL, NBA) show heuristic-based inefficiency — [Journal of Economics and Finance](https://link.springer.com/article/10.1007/s12197-015-9322-x)
- Over bias on props, i.e. "public bias for overs… pushes over-lines higher… models have higher EV on unders, especially role players" — [nbabettingsystem.com/strategy](https://nbabettingsystem.com/strategy/); [Wizard of Odds, player props math](https://wizardofodds.com/article/player-props-understanding-the-math-behind-the-lines/) **[vendor/anecdotal, no data shown]**
- Star props are said to be efficient (heavy sharp attention) while secondary and exotic props are not — repeated in search summaries without primary data **[anecdotal]**
- Same-game parlay correlation: books reprice correlated legs, and the identical SGP was quoted at +369 at one book vs +580 at another — [Wizard of Odds, SGP math](https://wizardofodds.com/article/same-game-parlays-the-mathematics-of-correlation/); [oddsindex](https://oddsindex.com/guides/same-game-parlays-guide) **[vendor]**. US patents on "sportsbook odds optimization and correlated proposition bet analysis" show books model the correlation explicitly — [USPTO 11250671](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11250671)
- Integrity-driven restrictions: MLB and sportsbooks agreed to a nationwide limit on pitch-level bets — [ESPN](https://www.espn.com/mlb/story/_/id/46919720/mlb-sportsbooks-agree-nationwide-limit-pitch-bets). The NBA and books reviewed the most vulnerable bets and wager limits, which bears on low-limit role-player props — [ESPN](https://www.espn.com/nba/story/_/id/46785871/nba-books-review-most-vulnerable-bets-wager-limits)

### Inferences
- A skewed count stat (yards, receptions, strikeouts) has median < mean. If books set the line near the *mean* and price -110 both ways, the under wins more than 50% even with no behavioural bias at all. "Under bias" findings must separate this distributional effect from public-money shading. Test: compare P(under) with the model's own P(X < line), not with 50%.
- Injury and lineup news reprices props faster than any scheduled model run. Pre-news edges partly reflect stale lines that most bettors cannot actually get, so timestamps matter (see ESPN `last_updated` caveat in a2).
- Alt lines are typically priced off the main line plus an assumed distribution. Errors in the distribution's tails (e.g. Poisson vs overdispersed) are a structurally plausible edge, but I found no study that measures it.

### Gaps
- No academic paper found that measures over/under bias, star shading or recency in NBA/NFL/MLB player props. This is a real literature gap (or one my searches missed). SSRN- and Google-Scholar-targeted searches returned nothing on point.
- No data on how fast prop lines incorporate injury news.

## (b2) Hold on props vs main lines

### Takeaway
Main lines at -110/-110 hold 4.55%. Props are commonly said to hold 6–9%. However, the common "props are -115/-105" example has the *same* hold as -110/-110, so the real cost depends on the price pair and the market. Parlays/SGPs hold far more (15–25%+), but that figure comes only from vendor sites.

### Cited Findings
- Spreads: 4.76% "hold" at -110/-110 per some sources, which is the overround (1.0476 − 1). True hold is 1 − 1/1.0476 = 4.55% [derived]. Props "6–8% on average", live 8–10% — [propsbot glossary](https://propsbot.ai/glossary/hold-percentage/) **[vendor]**; props "7–9% hold" — [same site](https://propsbot.ai/glossary/hold/) **[vendor]**; "props carry higher juice, especially at smaller books" — [Outlier](https://www.outlier.bet/sports-betting-strategy/positive-ev-betting/what-is-a-vig-a-must-know-concept-to-beat-the-books/) **[vendor]**
- Typical prop price pairs quoted: -115/-105, -120/-100, -130/+100 — [propsbot](https://propsbot.ai/glossary/hold-percentage/) **[vendor]**
- SGP/parlay hold of 15–25%. "New Jersey reported a parlay hold of 24.2% in September 2024 vs 4.4% on all other bet types" — [etparlays](https://www.etparlays.com/learn/sgp-tax-explained) **[vendor; NJ DGE primary report not checked]**
- Rebounds/assists props are said to carry less vig than points props (-110 vs -115) — [nbabettingsystem.com/strategy](https://nbabettingsystem.com/strategy/) **[vendor]**

### Inferences
- [derived] Hold for each pair: -110/-110 → 4.55%; -115/-105 → 4.50%; -120/+100 → 4.35%; -125/-105 → 6.3%; -130/+100 → 6.1%; -135/+105 → 6.1%. Skewed pricing with the *same* hold makes one side cheaper. A model that only bets the plus-money side of a -120/+100 pair pays less vig than the 4.35% headline. Compute the no-vig probability on each side; do not use a single "prop hold" number.
- Hold is per market and per book, so line shopping (best price across books) cuts effective hold a lot. Backtests should use the price you would actually have taken (best available at decision time), not the consensus.

### Gaps
- No regulator or academic dataset of player-prop-specific hold found. State reports (NJ DGE, NV GCB) break out "parlay" but, as far as I could verify, not straight player props.

## (b3) Book limits, non-limiting venues, and DFS pick'em breakevens

### Takeaway
US books admit to limiting about 1% of customers (BetMGM, Massachusetts) and cite "better model" as a reason. Academic bettors were restricted within months (Kaunitz et al.). Exchanges and prediction markets (Novig, ProphetX, Sporttrade, Kalshi, Polymarket) do not limit winners in the same way but have thinner books. DFS pick'em needs about 54–55% per leg (PrizePicks 5-flex = 54.25%).

### Cited Findings
- Massachusetts Gaming Commission roundtable (Sept 2024), with Bally's, BetMGM, Caesars, DraftKings, Fanatics, FanDuel and Penn present. BetMGM limits "approximately 1% of Massachusetts patrons"; FanDuel limits a smaller share. Reasons given: mistake lines, bonus abuse, a "better model" or "more information". Books said limits are "not based on winning". Nearly half of limited Fanatics customers were net losers when restricted. MGC received about 60 complaint emails. The article does not mention props specifically or how fast limits happen — [ESPN, Sept 2024](https://www.espn.com/sports-betting/story/_/id/41231266/espn-sports-betting-news-sportsbooks-defend-practice-limiting-sharp-customers)
- Kaunitz et al.: soft-book accounts were restricted during a 5-month live betting period while betting mispriced odds against the market consensus — [arXiv 1710.02824](https://arxiv.org/abs/1710.02824)
- Exchanges: Novig charges no maker/taker commission on pregame straight trades (the cost is the bid-ask spread) and markets itself as "no limits on winners" with props as a first-class market — [Novig blog](https://novig.com/blog/novig-vs-kalshi-vs-polymarket) **[self-description]**. ProphetX charges 2% of net winnings per market; Kalshi charges a per-contract fee that scales with price; Sporttrade uses spread-based commissions — [oddsshopper](https://www.oddsshopper.com/articles/prediction-markets/prophetx-vs-novig); [predictionmarketspicks](https://predictionmarketspicks.com/compare/kalshi-vs-novig) **[affiliate]**. I could not fetch Kalshi's fee schedule PDF (HTTP 429). From background knowledge, the taker fee is ceil(0.07 × C × P × (1 − P)), but this is unverified this session.
- Polymarket NBA markets have documented arbitrage between related markets — [arXiv 2605.00864](https://arxiv.org/html/2605.00864)
- PrizePicks 5-pick Flex pays 10x for 5/5, 2x for 4/5 and 0.4x for 3/5. The per-leg breakeven is 54.25%, equivalent to -118.6 — [GamedayMath](https://www.gamedaymath.com/blog/prizepicks-math); [sickfade calculator](https://sickfade.com/calculators/pickem). [derived check: 10p⁵ + 2·5p⁴(1−p) + 0.4·10p³(1−p)² = 1.001 at p=0.5425 ✓]. Underdog payout math — [GamedayMath](https://www.gamedaymath.com/blog/underdog-fantasy-payout-math). The claim that 2-pick slips are the worst structure is in [betting-forum](https://www.betting-forum.com/threads/prizepicks-betting-strategy-the-2-pick-trap-vs-5-pick-edge-break-even-math-explained.46905/) **[forum]**

### Inferences
- 54.25% per leg (-118.6) is worse than a -110 straight bet (52.38%) but comparable to a -115/-120 prop. Pick'em suits a model whose edge is in *picking sides*, and it assumes legs are independent. Correlated legs can help or hurt depending on structure. Payout tables change often, so re-verify them before relying on them.
- If you expect to win, limits at retail books make the *executable* edge much smaller than the backtested one. Backtests should model price × available limit, or assume execution only on non-limiting venues, at their spreads and fees.
- Exchange liquidity on player props is thin, so the quoted top-of-book price may not fill at size. Measure fill rates before trusting exchange-price backtests.

### Gaps
- No credible data on *how fast* (bets, dollars or weeks) the major US books limit winning prop bettors. Only anecdotes exist.
- Kalshi fee formula and per-venue prop liquidity were not verified from primary sources this session.
