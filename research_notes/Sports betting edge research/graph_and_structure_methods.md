# Graph, Structure-Learning and Dimensionality-Reduction Methods in Sports Prediction and Betting

Scope note: I made about 20 search/fetch calls. Several primary PDFs could not be fetched (TLS or connection errors on constantinou.info and ida.fel.cvut.cz; a 403 on SAGE), so a few findings rest on abstracts and search snippets. These are flagged where it matters. Publication years are given throughout. Throughout, "market-tested" means the paper evaluated against bookmaker odds (ROI, CLV, or a probability score compared with the odds) on held-out data.

## Q1. GNNs on player/team interaction graphs: were they tested against betting markets?

### Takeaway
Most sports-GNN papers report only accuracy or loss gains over other models, with no market comparison. The one market-tested GNN I found (tennis, 2025) LOSES to Pinnacle on Brier score and accuracy overall, and loses money unfiltered (-5.49% Kelly ROI). It shows a small positive ROI only inside a subset chosen by a threshold the authors tuned, which is a classic garden-of-forking-paths result.

### Cited Findings
- Xenopoulos and Silva (2021/2022), "Graph Neural Networks to Predict Sports Outcomes". They use a sport-agnostic graph of the game state and report test-loss reductions of 9% (American football) and 20% (Counter-Strike) versus prior state of the art. These are in-game win-probability tasks. Nothing I found shows a comparison with betting odds. — [arXiv 2207.14124](https://arxiv.org/pdf/2207.14124); [IEEE](https://ieeexplore.ieee.org/document/9671833)
- HIGFormer (2025), a player-team heterogeneous interaction graph transformer for soccer on the WyScout open dataset. It claims to "significantly outperform existing methods". The abstract mentions no bookmaker-odds or ROI comparison. — [arXiv 2507.10626](https://arxiv.org/abs/2507.10626)
- Other graph-representation sports papers also evaluate against model baselines, not markets: "Who You Play Affects How You Play" (2023) and an SFU thesis on spatio-temporal team-player graphs (2022). — [arXiv 2303.16741](https://arxiv.org/pdf/2303.16741); [SFU thesis](https://summit.sfu.ca/_flysystem/fedora/2022-08/input_data/22492/etd21919.pdf)
- Tennis MagNet GNN on a directed player-dominance graph (arXiv 2510.20454, Oct 2025):
  - Data: about 16.7k men's and 16.4k women's matches, 2014 to June 2025.
  - Design: walk-forward retraining each quarter. Validation ran Aug 2019 to Nov 2022; the test period was 2023 to June 2025.
  - Overall test results: MagNet 65.7% accuracy and Brier 0.215. Weighted Elo scored 66.5% and 0.212. Pinnacle scored 69.0% and 0.196. So the GNN is worse than a simple weighted Elo and much worse than the market.
  - Unfiltered betting lost money: -5.49% Kelly ROI.
  - Only within "high intransitivity" matches (threshold I* ≥ 2.55) did it report +3.26% Kelly ROI on 1,903 bets (bootstrap p<0.005) and +1.14% unit-stake ROI on 2,063 bets (p=0.022).
  - The authors concede that profitability "depended on selecting the appropriate intransitivity threshold."
  - [arXiv 2510.20454](https://arxiv.org/html/2510.20454v1)

### Inferences
- The best evidence for GNNs here is a single paper whose profitable result is conditional on a tuned subset filter. The fetched summary did not say whether I* was chosen on the validation window or the test window. If it was chosen on test, the p-values are invalid. Either way, a p of 0.022 for the unit-stake result does not survive a Harvey-Liu-Zhu-style t>3 hurdle (see Q3).
- The underlying idea is still interesting. Look for a structural feature the market misprices (here, intransitive matchups such as A beats B, B beats C, C beats A), and bet only where it is present. That transfers to a tabular setting: build one "intransitivity/structure" feature and test it conditional on the market line, rather than training a full GNN.
- For pre-game tabular data with 50-150 features and thousands of rows, there is no evidence that a GNN beats Elo or GBM. Even the tennis GNN lost to weighted Elo overall.

### Gaps
- I found no GNN paper for NFL, NBA or soccer pre-game outcomes that reports CLV or ROI against closing odds.
- I could not check whether the tennis paper's I* threshold was fixed before the test window.

## Q2. Bayesian networks and causal graphs in sports betting: profit claims and out-of-sample status

### Takeaway
Constantinou and Fenton's Bayesian networks (pi-football, 2012-2013) are the main profit claims. They are genuinely forward-published, but each covers about one EPL season and leans heavily on expert-elicited (subjective) nodes plus the pi-rating. The profit likely comes from the rating and expert inputs, not from learning the graph's structure. I found no study that applies data-driven causal discovery (PC, GES, FCI, NOTEARS, LiNGAM) to betting and reports ROI.

### Cited Findings
- pi-football (Constantinou, Fenton and Neil, Knowledge-Based Systems 36, 2012) is a Bayesian network for EPL match outcomes. "Forecasts were published online at www.pi-football.com prior to the start of each match", so the claim is genuinely ex-ante. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705112001967); [academia.edu](https://www.academia.edu/106197180/pi_football_A_Bayesian_network_model_for_forecasting_Association_Football_match_outcomes)
- "Profiting from an inefficient association football gambling market: prediction, risk and uncertainty using Bayesian networks" (Knowledge-Based Systems, 2013):
  - With a discrepancy threshold of ≥5% between model and market, the model reportedly profits against maximum, mean and common bookmaker odds, even with the margin included.
  - It is described as the first academic study to show profitability consistently against published market odds over a sufficiently large number of trials without removing the margin.
  - I could not fetch the PDF, so the number of bets and the season count are not verified. — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S095070511300169X); [author PDF (fetch failed)](http://constantinou.info/downloads/papers/pi-model12.pdf)
- pi-ratings (Constantinou and Fenton, 2013) are a dynamic rating built from goal difference. They are said to beat two Elo variants on profitability over five EPL seasons (2007/08-2011/12), net of margin. — [pi-ratings PDF](http://www.constantinou.info/downloads/papers/pi-ratings.pdf)
- Constantinou (2020 preprint; Journal of Sports Analytics 2022) applied ratings plus BNs to 13 EPL seasons of Asian Handicap markets. It found that "the AH market is found to share the inefficiencies of the traditional market"; the abstract gives no headline out-of-sample ROI. — [arXiv 2003.09384](https://arxiv.org/abs/2003.09384)
- Hubáček et al.'s experimental review of score-based soccer models covers Poisson models, Elo, pi-ratings, PageRank (a graph method) and a hybrid BN on rating features. The lowest RPS came from gradient-boosted trees on rating features, not from the graph or BN methods. This attribution comes from a secondary review snippet. — [MathSport 2019 PDF](http://ida.felk.cvut.cz/zelezny/pubs/MathSport2019a.pdf); [arXiv 2403.07669 review](https://arxiv.org/pdf/2403.07669)
- In the 2017 Soccer Prediction Challenge (Machine Learning journal special issue), the top entries used engineered rating features (pi-ratings, PageRank, learned ratings) fed to gradient boosting or k-NN. The structure of the model mattered less than the rating features. — [Berrar et al., Open International Soccer Database](https://link.springer.com/article/10.1007/s10994-018-5726-0); [guest editorial](https://link.springer.com/article/10.1007/s10994-018-5763-8)
- Causal discovery has known weaknesses that make it unreliable for this use:
  - NOTEARS lacks scale invariance; rescaling variables can change the learned DAG a lot. — [arXiv 2104.05441](https://arxiv.org/pdf/2104.05441)
  - Reisach, Seiler and Weichwald (NeurIPS 2021) showed that the benchmark success of continuous structure learners is largely explained by "varsortability". In simulated data, marginal variance rises along the causal order, and a trivial sort-by-variance baseline matches the learners. On standardized data the same algorithms fail to recover the true DAG or its equivalence class. — [arXiv 2102.13647](https://arxiv.org/pdf/2102.13647)

### Inferences
- The BN wins in sports betting came from expert-knowledge priors plus a good rating, evaluated on small forward samples of one or a few EPL seasons from 2010-2013. EPL markets have since tightened. I found no independent replication of the pi-football profit.
- Running PC/GES/NOTEARS on 50-150 pre-game features is unlikely to add predictive value:
  - Features standardized for modelling remove the variance signal NOTEARS-type methods actually exploit.
  - Many sports features are deterministic functions of each other (rolling means, differences), which violates faithfulness assumptions.
  - With few thousand rows and tiny effects, conditional-independence tests have little power, so the learned graph will be unstable.
- If a graph is wanted, the defensible use is narrow. Make the target's Markov blanket (the set of features that shield the target from all others in the graph) conditional on the market line, and check it with bootstrap stability. Do not try to read a full causal DAG.

### Gaps
- I found no paper applying PC, FCI, GES, LiNGAM or NOTEARS to pre-game betting features with an ROI or CLV evaluation. This absence is itself informative, but cannot be proven exhaustively.
- I could not verify the exact bet counts or ROI percentages in the 2013 pi-football profitability paper.

## Q3. Graphical lasso, partial correlation, MI and PCA for feature selection in low-signal settings: finance lessons

### Takeaway
The asset-pricing literature is the right analogue: hundreds of candidate predictors, tiny true signal, heavy data mining. It points to three habits:
- raise the significance bar (t > 3);
- use regularized or reduced models over unrestricted OLS;
- treat most "discovered" predictors as false.

Tree and NN gains in finance came with very large panels (decades of thousands of stocks), which a few-thousand-row betting dataset does not have.

### Cited Findings
- Harvey, Liu and Zhu, "...and the Cross-Section of Expected Returns" (RFS 2016). At least 316 factors had been tested. Because of multiple testing, the usual t > 2.0 cutoff is "a serious mistake", and a new factor should clear about t > 3.0. — [NBER w20592](https://www.nber.org/system/files/working_papers/w20592/w20592.pdf); [RFS](https://academic.oup.com/rfs/article/29/1/5/1843824)
- A counterpoint (Chen, 2022): "Most claimed statistical findings in cross-sectional return predictability are likely true." This means the false-discovery rate in finance is disputed; it is not settled that most findings are false. — [arXiv 2206.15365](https://arxiv.org/pdf/2206.15365)
- Gu, Kelly and Xiu, "Empirical Asset Pricing via Machine Learning" (RFS 2020):
  - They compare generalized linear models, dimension reduction (PCR, PLS), boosted trees, random forests and neural networks.
  - Trees and NNs win, and the gains are traced to nonlinear predictor interactions.
  - Monthly out-of-sample R² values are tiny. A search snippet gives benchmark 0.16% and trees/NN at 1.08%-1.80%. I could not verify which panel or portfolio those figures refer to, so do not quote them without checking the paper.
  - [NBER w25398](https://www.nber.org/system/files/working_papers/w25398/w25398.pdf)
- On false-discovery-controlled selection: stability selection refits the lasso on subsamples and keeps consistently chosen variables. Model-X knockoffs provide provable FDR control. Both are reported to "have low power in simulations and select few features in practice", especially at low SNR with correlated features. — [GRIP2 arXiv 2602.00218](https://arxiv.org/html/2602.00218); [Barber and Candès knockoffs](https://arxiv.org/pdf/1404.5609)

### Inferences
- In betting, the market line is the dominant "factor". The partial-correlation analogue of a factor-model alpha test is to regress the outcome (or the residual against the closing line) on each candidate feature, conditioning on the market-implied probability. A feature carries signal only if it predicts what the market got wrong.
- Graphical lasso on the feature matrix alone, without the target or the line, only finds feature-feature redundancy. That helps for de-duplication and building PCA-style composites, but it says nothing about edge.
- PCA/factor compression of correlated feature blocks is a sensible variance-reduction step before a linear model when rows are few. In finance, PCR/PLS beat unrestricted OLS mainly by curbing overfitting, not by finding hidden signal. This is a recollection of the GKX framing and should be verified.
- Low power of knockoffs and stability selection at low SNR is a feature, not a bug, here. If they select nothing beyond the line, that is probably the truth. The local memory note on strikeout props (market beats model; context adds nothing) is consistent with this.

### Gaps
- I found no published application of graphical lasso, partial-correlation networks or knockoffs to sports-betting feature selection with an out-of-sample ROI or CLV test.
- I could not resolve the conflicting Gu-Kelly-Xiu R² figures from the sources fetched.

## Q4. Evidence comparing model classes on betting ROI, calibration and CLV

### Takeaway
No model class reliably beats the market. Where profits are claimed, they come from:
- **Calibration:** Walsh and Joshi 2024, one NBA season.
- **Decorrelation from the bookmaker:** Hubáček et al. 2019, NBA 2007-2014.
- **Rating features:** pi-ratings.

None of them comes from graph or structure learning. Large comparisons (Wilkens 2021 tennis) find that ML adds nothing over the odds and that returns are mostly negative. Logistic regression is consistently competitive at small n.

### Cited Findings
- Hubáček, Šourek and Železný, "Exploiting sports-betting market using machine learning" (IJF 35(2), 2019):
  - Three ingredients: (1) a loss term that reduces correlation with the bookmaker's implied probabilities; (2) a CNN over player-level stats; (3) portfolio-theory bet sizing.
  - Combined, they gave "positive cumulative profits in experiments with NBA data from seasons 2007–2014 systematically, as opposed to alternative methods tested."
  - Exact ROI not retrieved. — [EconPapers abstract](https://econpapers.repec.org/article/eeeintfor/v_3a35_3ay_3a2019_3ai_3a2_3ap_3a783-796.htm); [ResearchGate](https://www.researchgate.net/publication/331218530_Exploiting_sports-betting_market_using_machine_learning)
- Follow-up: Hubáček and Šír, "Beating the market with a bad predictive model" (arXiv 2020; IJF 2023). It argues that "it is generally possible to make systematic profits with a completely inferior price-predicting model" by explicitly decorrelating from the market, and shows this for stocks and sports betting. — [arXiv 2010.12508](https://arxiv.org/abs/2010.12508)
- Walsh and Joshi, "Machine learning for sports betting: should model selection be based on accuracy or calibration?" (Machine Learning with Applications 16, 2024; arXiv 2303.06021):
  - Setup: trained on NBA 2014/15-2017/18 (validation 2016/17, test 2017/18 for model selection). The betting simulation is one season (2018/19) on public Vegas closing moneylines.
  - Candidates on test classwise-ECE: SVM 3.23%, MLP 3.59%, LR 3.61%, RF 4.39%.
  - Candidates on test accuracy: SVM 66.55%, LR 65.69%, MLP 65.69%, RF 65.34%. SVM won both branches.
  - ROI for the calibration-selected vs accuracy-selected SVM:
    - fixed stake: +32.45% vs +5.56%;
    - Kelly: +36.93% vs -75.9%;
    - average across strategies: +34.69% vs -35.17%.
  - Paper text and [arXiv PDF](https://arxiv.org/pdf/2303.06021); [code](https://github.com/conorwalsh99/ml-for-sports-betting).
- Wilkens, "Sports prediction and betting models in the machine learning age: the case of tennis" (J. Sports Analytics 7, 2021):
  - Average accuracy cannot be pushed beyond about 70%.
  - "Most of the relevant information is embedded in the betting markets", and adding match/player data gives no significant improvement.
  - Betting returns are "mainly negative over the longer term" across most models and staking strategies.
  - [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3506302); [IOS Press](https://content.iospress.com/articles/journal-of-sports-analytics/jsa200463)
- Bunker and Thabtah (2019) put team-sport accuracy at about 70% (roughly 60-80% by sport). Domain knowledge and feature design matter more than algorithm choice. — [framework paper](https://www.researchgate.net/publication/319937079_A_Machine_Learning_Framework_for_Sport_Result_Prediction)
- A 2024 systematic review of ML in sports betting exists and covers these themes. I did not extract specific numbers from it. — [arXiv 2410.21484](https://arxiv.org/pdf/2410.21484)
- Kaggle March Madness:
  - A first-place solution used Vegas point spreads and KenPom efficiency in a logistic regression, partly because the logistic maximum-likelihood objective matches the log-loss scoring.
  - A recent write-up reports logistic regression beating XGBoost with about 650 training rows of differential features.
  - Luck dominates single-tournament rankings.
  - [sfirke how-to](https://github.com/sfirke/predicting-march-madness/blob/master/march_madness_how_to.md); [Kaggle 2024 8th place](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/writeups/flaty-8th-place-solution); [2026 3rd place](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/3rd-place-solution-march-machine-learning-mania)

### Inferences
- **Walsh and Joshi is weak evidence for its headline.** It is a single season of about 1,230 games, with one model per branch. The ECE gap between candidates is only 0.4 points. The accuracy branch's -75.9% Kelly result is a single bankroll path, which is very path-dependent. Treat it as "use calibration and proper scoring for selection", not as a +35% ROI expectation. It also bets against closing lines, which are hard to beat; a positive ROI there would be remarkable and needs multi-season replication.
- **Hubáček's decorrelation idea is the most transferable insight for feature selection.** Value lies in signal orthogonal to the line. Operationally, this means:
  - modelling the residual against the market;
  - including the line as a feature or offset and asking which features carry incremental conditional information;
  - penalizing correlation with the market's implied probability.
- **Across sources, the hierarchy for small n is:** strong ratings (Elo, pi, Massey, market) + logistic/ridge ≥ GBM on rating features >> GNN/CNN on raw interaction data. I found no evidence that structure learning adds ROI on top.

### Gaps
- I found no head-to-head comparison of BN or causal-discovery models against logistic regression, Dixon-Coles or GBM on CLV.
- Hubáček 2019's exact ROI and number of bets were not retrieved (PDF inaccessible).
- Wilkens 2026, "Can simple models predict football — and beat the odds? Lessons from the German Bundesliga", looks directly relevant but returned 403. — [SAGE](https://journals.sagepub.com/doi/10.1177/22150218261416681)

## Q5. Practical recipe: least-overfit structure/feature-selection approach for 50-150 features and thousands of rows

### Takeaway
Treat the market line as the conditioning variable and every feature as a hypothesis to falsify. Use stable, low-variance selection (penalized logistic with the line as an offset, plus stability selection or knockoffs), and evaluate on walk-forward CLV/log-loss with a t ≈ 3 bar. Graphs and PCA belong to de-duplication and composite-building, not to edge discovery.

### Cited Findings
- Multiple-testing hurdle of about t > 3 for new predictors. — [Harvey, Liu and Zhu](https://www.nber.org/system/files/working_papers/w20592/w20592.pdf)
- Decorrelate from the bookmaker, because profit comes from where model and market disagree. — [Hubáček et al. 2019](https://econpapers.repec.org/article/eeeintfor/v_3a35_3ay_3a2019_3ai_3a2_3ap_3a783-796.htm); [Hubáček and Šír](https://arxiv.org/abs/2010.12508)
- Select models on calibration or proper scoring, not accuracy. — [Walsh and Joshi 2024](https://arxiv.org/pdf/2303.06021)
- Stability selection and knockoffs control false selections but have low power at low SNR with correlated features. — [arXiv 2602.00218](https://arxiv.org/html/2602.00218); [arXiv 1404.5609](https://arxiv.org/pdf/1404.5609)
- Causal-discovery output depends on variable scaling and simulation artefacts. — [arXiv 2102.13647](https://arxiv.org/pdf/2102.13647); [arXiv 2104.05441](https://arxiv.org/pdf/2104.05441)
- Logistic regression beats GBM at very small n on Kaggle March Madness. — [Kaggle 2026 3rd place](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/3rd-place-solution-march-machine-learning-mania)

### Inferences
This recipe is synthesized from the sources above and has not been tested as a whole in any paper.
1. **Baseline:** logit(p) = logit(p_market, de-vigged) + β·x, with the market term as a fixed offset or a free coefficient. Every feature is judged by whether it lowers out-of-sample log-loss or Brier relative to the market alone, with walk-forward splits by season or week.
2. **De-duplicate first:** cluster features with |ρ| > 0.8. Graphical lasso or hierarchical clustering on the correlation matrix both work. Replace each cluster with one representative or its first PC, so that stability selection is not split across near-duplicates.
3. **Screen conditional on the line:** compute the partial correlation of each feature with the market residual (outcome minus market probability, or cover margin minus spread). Apply a Benjamini-Hochberg FDR or a t > 3 bar. With thousands of rows, a true correlation of 0.03 has t ≈ 0.03·√n, which is about 1.7 at n=3,000 and about 3 at n=10,000. So small edges are not detectable below about 10k rows, and the bar should be accepted as such.
4. **Stability selection:** L1-logistic with the market offset on 100+ half-subsamples, blocked by season so that within-season autocorrelation does not inflate stability. Keep features selected in more than 60-80% of subsamples. Optionally cross-check with model-X knockoffs (Gaussian knockoffs are fine after de-duplication). If nothing survives, accept that no feature adds information beyond the line.
5. **Stability in time:** require the sign of each surviving feature's effect to replicate in both halves of the seasons (split-half), and report an across-season spread.
6. **Model class:** ridge or logistic on survivors. Try GBM only if n ≥ ~10k and there is an a-priori interaction hypothesis. Skip GNN/BN/causal discovery unless one specific structural hypothesis exists (such as intransitivity). In that case encode it as a single feature and test it through steps 3-5.
7. **Evaluation:**
   - CLV (beating the closing line) is the lower-variance proxy for ROI.
   - Report ROI with season-resampled CIs.
   - Count every feature/threshold tried, because the multiple-testing burden is cumulative across the search.

### Gaps
- I found no empirical sports-betting paper validating stability selection, knockoffs or line-conditional partial correlation end-to-end against CLV. The recipe is an inference from finance and statistics practice plus the decorrelation literature, not a tested sports result.
- The 60-80% stability threshold and the |ρ| > 0.8 de-duplication cutoff are conventional choices, not sourced from sports-specific evidence.
