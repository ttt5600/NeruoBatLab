# TRADING PRINCIPLES -- the standard betting/trading playbook, and how this fund applies it (2026-10-29)

The professional customs every bettor/trader is expected to know. Each entry: **what** it is, the **formula**, **how we
apply it** (the script, rule or pipeline stage that enforces it), and **our evidence** (with the file). Per-edge worked
numbers live in `notes/EDGE_MATH.md`; the catalog of situational factors (weather, rest, travel, refs...) lives in
`notes/SITUATIONAL_FACTORS.md`. Calculator for every formula below: `scripts/edge_math.py` (CLI: `bet`, `devig`,
`clv`, `power`, `ruin`). Numbers marked **[em]** were computed by edge_math.py with the inputs shown; every other
project number cites its file. REPO IS PUBLIC: no account data.

## Contents
A. Pricing: 1 EV & implied probability · 2 Vig / overround / de-vig · 3 Fees, spread, slippage · 4 Break-even win rate ·
5 CLV · 6 Line shopping
B. Markets: 7 Efficiency hierarchy · 8 Maker vs taker, adverse selection, winner's curse · 9 Base rates & "system" claims ·
10 Regression to the mean & shrinkage · 11 Calibration
C. Bankroll: 12 Kelly (full / fractional / on a bound) · 13 Log growth · 14 Risk of ruin & drawdown · 15 Correlation &
simultaneous bets · 16 Unit tiers · 17 Loss limits, no chasing · 18 Liquidity & capacity · 19 Diversification
D. Statistics: 20 Variance, sample size, power · 21 CIs & z-scores · 22 Multiple testing · 23 Forking paths &
pre-registration · 24 Holdout / lockbox · 25 Deflated Sharpe / PBO · 26 Overfitting · 27 Look-ahead & leakage ·
28 Survivorship bias
E. Operations: 29 Record keeping & honest P&L · 30 Hedging, arbitrage, middles · 31 Discipline / tilt
F. Checklist before any bet · G. Gaps (what we do not enforce yet)

---------------------------------------------------------------------------------------------------------------------
## A. Pricing

### 1. Expected value & implied probability
- **What.** A price is a probability. You have an edge only if your probability beats the price you actually pay.
- **Formula.** Contract: implied p = price. Decimal odds d: p = 1/d. American: +A -> 100/(A+100); -A -> A/(A+100).
  EV per $ = p/c - 1; EV per contract = p - c (c = all-in cost per $1 payout).
- **We apply it.** `edge_math.py bet` (rows `raw_implied_prob`, `ev_per_dollar`, `ev_per_contract`); `ev_check.py`
  refuses EV <= 0 after fees; protocol rule 5: score vs the de-vigged market, never vs 50%.
- **Evidence.** "Skewed stats make unders > 50% for free" (notes/protocol.md rule 5); P2 props unders +0.27%/60,826
  (FUND_BOOK 3d) -- a high hit rate is not EV.

### 2. Vig / overround and de-vig (proportional, power, Shin)
- **What.** A book's implied probs sum to > 1; the excess is its margin. To get a fair probability you must remove it,
  and *how* you remove it matters.
- **Formula.** Overround = sum(q_i) - 1. Proportional: p_i = q_i / S. Power (Clarke, Kovalchik & Ingram 2017): find
  k with sum(q_i^k) = 1. Shin (1993): p_i = (sqrt(z^2 + 4(1-z) q_i^2/S) - z) / (2(1-z)), z solved so sum = 1.
- **Why proportional misleads on longshots.** It spreads the margin in proportion to price, but books load most of the
  margin on longshots (favourite-longshot bias). So proportional leaves longshots too expensive-looking-fair, i.e.
  their "fair" prob too high, and anything that buys longshots shows fake value/CLV. Example [em]
  (`devig --odds 1.25 6.0 11.0`): longshot fair = 0.0860 proportional vs 0.0735 power vs 0.0756 Shin; the favourite
  0.7564 vs 0.7843 vs 0.7745. -110/-110 is symmetric: all three give 0.500, overround 4.76% [em].
- **We apply it.** `edge_math.devig` (all three); notes/data.md: "De-vig with POWER, not proportional"; E3 soccer uses
  power de-vigged Pinnacle (FUND_BOOK 3a). `sports/market.py` still de-vigs proportionally (notes/MODELS_GUIDE.md
  ~line 1002) -- fine for two-way near-even markets, wrong for 3-way/longshots.
- **Evidence.** F3 gap logger flags were "almost all soccer draws (3-way de-vig artefact suspected)" -> N4 "fix Shin
  de-vig then re-judge" (FUND_BOOK 3b); lesson 6 "proportional de-vig fakes longshot CLV".

### 3. Fees, spread and slippage are part of the price
- **What.** The price you can trade at is the ask (taker) plus fee, not the mid or the last trade.
- **Formula.** Kalshi taker fee 0.07*C*p(1-p), rounded up per order; maker 0.0175*C*p(1-p) only on
  `quadratic_with_maker_fees` series; Polymarket US taker 0.0695*C*p(1-p); ForecastEx $0.01/contract
  (`edge_math.fee_per_contract`; sources in its docstring). Spread cost of crossing = ask - mid.
- **We apply it.** Every backtest prices at ask + fee (or a modelled maker fill); `edge_math.all_in_cost`;
  `kalshi_exec.py` KILLs if a fill's cost differs from the ask by > 2c (convention check).
- **Evidence.** L32 on Kalshi: 0.51 ask + 0.0175 fee = 0.5275 eats the shrunk edge -> planning 0% (notes/fund_economics.md);
  Polymarket US moneylines: 0/116 quotes +EV after the 0.0695 fee (notes/results.md 10-22); display prices are not
  quotes (Polymarket negRisk "+10-14%" artefact, FUND_BOOK 3d); ForecastEx DEV priced from last trades, flagged as the
  main reason not to trust it (notes/forecastex_wx.md).

### 4. Break-even win rate
- **What.** The win rate at which EV = 0.
- **Formula.** Binary at all-in cost c: p_BE = c. Odds d: p_BE = 1/d. -110 -> 52.38%. Loss-rate form: l_BE = 1 - c.
- **We apply it.** `edge_math` row `break_even_prob`; FX1 sealed gate required the Wilson 95% UPPER bound of the loss
  rate to sit below 1 - mean cost (scripts/forecastex_wx_sealed.py docstring; notes/forecastex_wx.md).
- **Evidence.** FX1 sealed: loss rate 4.57% [2.50, 8.20] vs break-even 10.23% (lab/lockbox.jsonl forecastex_wx_FX1);
  kicker: 84.7% hit supports -555 at best (notes/nfl_kicker_system.md).

### 5. Closing line value (CLV) -- the skill signal
- **What.** If you consistently get a better price than the close (the most-informed price), you are probably beating
  the market; CLV converges in hundreds of bets where ROI needs thousands.
- **Formula.** CLV (prob points) = de-vigged close prob of your side - de-vigged prob you paid; or in points/goals for
  spreads/totals. Significance: mean / (sd/sqrt n) -> one-sided p (`edge_math.clv_test`; cluster duplicates first).
- **We apply it.** Opener-model leads are judged on forward CLV (L32/L34/L33/Q004 logger rules, QUESTIONS.md Q004);
  `fund_stats.py` treats CLV metrics separately (`CLV_METRICS`).
- **Caveat (ours, learned the hard way).** CLV is necessary, not sufficient: it must exceed the overround you pay.
- **Evidence.** O1 sealed: CLV +0.68 p .005 but ROI -10.2% (lab/lockbox.jsonl nba_O1_open_model); L32 sealed CLV +1.2 pts
  yet ROI p .054 (lockbox nba_L32_tot_open); E3 soccer sealed CLV +3.63% with ROI +8.71% (lockbox soccer_best3_pow3).

### 6. Line shopping / best price
- **What.** Take the best available price across venues; small price gains compound into most of a thin edge.
- **Formula.** Best price = max decimal odds (min cost) across books at the same time, same market, same resolution.
- **We apply it.** E3 takes the best of B365/WH/bwin (scripts/soccer_fd.py); F4 logs Polymarket US vs same-pick Kalshi
  price (scripts/pmus_weather_paper.py); E5 is the E1 rule on a second, cheaper venue.
- **Evidence.** E3 "best of 3" sealed +8.71% (lockbox); single-book B365-vs-Pinnacle dev -0.80%/3,092 vs WH +11.2%/1,201
  (lab/ledger.jsonl soccer_fd trials 3 and 6). F4 forward "= Kalshi price" so no shopping gain there (FUND_BOOK 3b).

## B. Markets

### 7. Market-efficiency hierarchy
- **What.** Prices are sharpest where most money and information arrive: close > open; sharp book (Pinnacle) > soft
  books; heavily traded exchange markets ~ sharp books; thin/new/retail markets least efficient.
- **We apply it.** Scout filter: hunt only at the inefficient end (openers, thin tails, second venues, maker side)
  (notes/FUND_ARCHITECTURE.md section 4); the DEAD list is the do-not-redo filter.
- **Evidence.** Every NFL/NBA/MLB closing-line angle loses ~the vig (FUND_BOOK lesson 1); models lead the opener with
  CLV p < 1/4000 but not the close (lesson 3); soft books lag Pinnacle (E3); Kalshi/Polymarket game winners within ~0.5-1c
  of each other and of Pinnacle (notes/results.md M2, pmus_vs_pin); in-game Kalshi beats our models (Brier .1284 vs
  .1319, L37 sealed FAIL, FUND_BOOK 3d).

### 8. Maker vs taker, adverse selection, winner's curse
- **What.** Takers pay the spread and fee; makers earn them -- but a resting order is filled most often exactly when it
  is wrong (adverse selection). Winner's curse: the bets you "win" the right to make (the cheap price, the fill) are
  selected for being mispriced against you.
- **Formula.** Maker P&L = sum over winners of min(order, fillable volume) * r_win - sum over losers of order (losers
  fill fully) (notes/fund_economics.md). Compare FILLED vs UNFILLED outcome rates.
- **We apply it.** Maker backtests must model fills from the trade tape and fit calibration on FILLED picks only
  (notes/kalshi_sizing.md method: maker theta 0.86); per-pick cap ~$75 (fund_economics).
- **Evidence.** E4 maker sealed ROI +1.07% -> +2.22% but filled loss rate 3.07% vs 2.46% all, unfilled 0%
  (lockbox kalshi_temp_maker_M3_N10); K-FAV filled win 90.5% vs 99.3% unfilled (FUND_BOOK 3d); L28 cheap leaders =
  market knows something (lesson 4); L35 maker at first bid adverse +16% (FUND_BOOK 3c).

### 9. Base rates and "system" claims
- **What.** A high hit rate is meaningless without (a) the base rate of the event and (b) the price actually offered.
  Most touted systems re-discover the base rate.
- **Formula.** Compare hit rate to the base rate AND to the market-implied prob of the same bets; fair American =
  -100 p/(1-p). Residual = hit - matched baseline.
- **We apply it.** Claim-check recipe in notes/protocol.md (Edge math rule); every social claim gets a prereg + one
  ledger trial (notes/social_edges_2026-10-22.md, FUND_BOOK 3d "Social-media / practitioner systems").
- **Evidence.** Kicker over 3.5 "90%": 84.7% (n 354) vs 83.0% base; residual vs team total -0.7pp p .65; fair about
  -555 empirical / -810 market (notes/nfl_kicker_system.md). "75% of companies beat" is already priced (Polymarket
  calibrated .732 vs 74.3%, notes/earnings.md).

### 10. Regression to the mean & Bayesian shrinkage
- **What.** Early-season or small-sample results are part signal, part noise; the best estimate pulls them toward a
  prior in proportion to the noise.
- **Formula.** Blend = w*season + (1-w)*prior, w = n / (n + (sigma/tau)^2); sigma = game-noise SD, tau = SD of true
  talent around the prior; (sigma/tau)^2 = games at which season and prior get equal weight.
- **We apply it.** `rating <team>` / scripts/team_rating.py. Fit (dev 2021-22..2024-25, printed 2026-10-29): game sigma
  13.6 pts, prior tau 3.6 pts, carry-over b 0.60 -> (sigma/tau)^2 = **14.5 games to half weight**; e.g. 4 games in ->
  22% weight on the season. Also shrink before Kelly (section 12) and planning ROIs at ~2/3 of sealed (fund_economics).
- **Evidence.** NBA/NHL shooting/PDO luck is real but priced (notes/nba_luck.md, notes/nhl_luck.md); dev edges shrink
  out of sample (M3 +4.3% -> +0.45% sealed; R_REB +2.7% -> -1.56%; FUND_BOOK lesson 7).

### 11. Calibration (Brier, log loss, reliability)
- **What.** A probability model is useful only if its 70% calls happen ~70% of the time and it beats the market's own
  probabilities on a proper score.
- **Formula.** Brier = mean (p - y)^2 (Brier 1950); log loss = -mean [y ln p + (1-y) ln(1-p)]; reliability = observed
  rate per predicted-prob bin with Wilson CIs.
- **We apply it.** Compare model vs market on the same events before any betting rule (lab trials report both);
  kalshi_sizing reports out-of-fold log loss and a calibration table.
- **Evidence.** Kalshi in-game Brier .171 < ESPN .180 (M4); market .199 < Elo .207 (NBA); earnings market .155 vs base
  .191 (notes/earnings.md); weather ratio model log loss .1054 vs price alone .1098 (notes/kalshi_sizing.md); Kalshi
  late-game calibrated 7/7 bins (notes/late_game_calibration.md); CFB model miscalibrated (notes/cfb_simple_model.md).

## C. Bankroll

### 12. Kelly: full, fractional, on a confidence bound
- **What.** The stake that maximises long-run log wealth (Kelly 1956; Thorp 2006).
- **Formula.** f* = (b p - q)/b = (p - c)/(1 - c), b = (1 - c)/c. Fraction k of Kelly keeps growth 2k - k^2 of full
  Kelly at k^2 of the variance (half Kelly: 75% of growth, 1/4 variance; quarter: ~44% of growth, 1/16 variance).
  Fund rule: f = k * f*(p_LCB), with p_LCB a Wilson lower bound (80-95%), k = 1/4 default, 1/2 max, then caps.
- **We apply it.** `edge_math.stake_fraction` / `bet` rows `kelly_*_at_lcb`; notes/risk.md; protocol rule 7;
  `ev_check.py` tiers; Kalshi executor uses flat TIER_FRAC (orig7 1%, x1 0.5%) because Kelly per-pick sizing was
  tested and rejected (notes/kalshi_sizing.md).
- **Evidence.** E5 derivation in FUND_BOOK section 1 (half Kelly on the LCB ~5%); weather quarter Kelly at the 80% LCB
  ~4.4% of B per pick, flat across prices (notes/kalshi_sizing.md). Note: notes/risk.md says quarter Kelly "gives up
  ~44% of growth" and kalshi_sizing.md says it "keeps about 3/4 of the growth" -- both mis-state it; by 2k - k^2 quarter
  Kelly KEEPS ~44% (half Kelly keeps 75%).

### 13. Log-growth rate
- **What.** What compounding actually maximises; arithmetic EV can be positive while log growth is negative
  (overbetting).
- **Formula.** g(f) = p ln(1 + f b) + q ln(1 - f); g < 0 beyond ~2 f*.
- **We apply it.** `edge_math.log_growth` (row `log_growth_per_bet`); kalshi_sizing compares strategies by paired
  night-bootstrap log growth, not ROI.
- **Evidence.** Flat-with-skip lost to plain flat on log growth -0.11 [-0.19, -0.03] (notes/kalshi_sizing.md).

### 14. Risk of ruin and expected drawdown
- **What.** Probability of ever losing a given share of bankroll, and the typical worst peak-to-trough.
- **Formula.** Diffusion approximation: P(ever reach level x of start) = x^(2g/v) (g, v = mean and variance of log
  growth per bet); at fraction k of Kelly ~ x^(2/k - 1) (Thorp 2006; MacLean, Thorp & Ziemba 2011). Max drawdown by
  simulation (`edge_math.simulate_drawdown`).
- **Illustration [em]** (`ruin --p 0.55 --price 0.5238 --bets 500`, a 55% bettor at -110): P(ever halve) full Kelly
  0.50, half 0.12, quarter 0.0075; expected max drawdown over 500 bets 66% / 41% / 22%.
- **We apply it.** `edge_math.py ruin`; kalshi_sizing's 2,000-path night bootstrap reports P(DD > 20%).

### 15. Correlation and simultaneous bets
- **What.** Bets resolved by the same event/weather/night are not independent; Kelly per bet overstakes the portfolio.
- **Formula.** Cap the sum of simultaneous stakes; for k correlated bets treat them as fewer effective bets
  (design effect deff = 1 + (m-1) rho).
- **We apply it.** `kalshi_exec.py` NIGHT_FRAC 0.10 (night cap) and TIER_FRAC per pick; bet.py TIER_A_NIGHT_U = 1.0u;
  bootstraps resample whole nights/station-days; protocol rule 4 clusters duplicates (same player-game-market = one
  observation).
- **Evidence.** Weather within-night design effect measured 1.00 (notes/fund_economics.md) -- the cap is kept anyway
  (notes/kalshi_sizing.md caveats); ForecastEx and Kalshi picks overlap 46 of 62 station-days (notes/forecastex_wx.md).

### 16. Unit sizing tiers
- **What.** Fixed fractions by evidence quality, so judgment can't override math.
- **We apply it.** 1u = $50 (2.5% of $2k); Tier A sealed pass <= 1u/night; B paper lead <= 0.5u/bet; C judgment <= 0.25u,
  monthly C loss cap 2u (scripts/bet.py constants `TIER_BET_CAP_U`, `TIER_A_NIGHT_U`, `C_MONTH_LOSS_U`; FUND_BOOK s1).

### 17. Loss limits; never chase losses
- **What.** Pre-set stops that halt betting after a bad stretch; never raise stakes to win back losses.
- **We apply it.** kalshi_exec.py DAY_LOSS_FRAC 0.05 / WEEK_LOSS_FRAC 0.10 (weekly writes KILL); bet.py loss stops on
  manual tiers (daily -3u, weekly -6u, FUND_BOOK s1); shared KILL file; notes/risk.md "never add money to chase losses";
  strategy off if live CLV/ROI after 200 bets is below the sealed CI.
- **Evidence.** First E1/E2 live loss 10-29 after 58 wins was expected, no rule change (FUND_BOOK 3a).

### 18. Liquidity and capacity
- **What.** An edge is worth edge x volume you can actually fill; size beyond the book moves the price or selects you.
- **We apply it.** `bet --liquidity` caps stake at available contracts; taker executor takes 0.5x depth within 1c
  (notes/fund_economics.md).
- **Evidence.** Weather maker tops out ~$144/month at plan; above ~$75/pick adverse selection eats the gain; L39
  capped ~$7.5/month; E5 median 29 contracts/pick (fund_economics; lockbox forecastex_wx_FX1 median_d1_pair_quantity 29).

### 19. Diversification across uncorrelated edges
- **What.** Several independent edges grow faster and draw down less than one edge sized up.
- **We apply it.** Capital gate to $10k requires a SECOND edge passing sealed + forward (FUND_BOOK s1).
- **Evidence.** E5 is only partly independent of E1: same forecasts, 46/62 shared station-days, 2 of 5 losing days
  shared, but different settlement (WU vs NWS) (notes/forecastex_wx.md).

## D. Statistics

### 20. Variance, sample size, power
- **Formula.** n = ((z_a + z_b) sigma / mu)^2, one-sided alpha .05, power .8 (`edge_math.n_required`,
  `fund_stats.py power`). Binary sigma per $ = sqrt(p(1-p))/c.
- **We apply it.** Prereg must state power; refuse tests that cannot detect the target (notes/FUND_ARCHITECTURE.md s5).
- **Evidence.** 2% ROI at -110 needs ~14,000 bets [em: `power --roi 0.02 --cost 0.5238` = 14,018; protocol rule 7 says
  ~14k]; L32 needs ~1,500 bets at a true +6.4%, ~6,900 at +3%; weather maker ~1,050 filled picks (fund_economics).
  Rule of three: 0 losses in n -> 95% upper bound ~3/n, so 40-0 live is consistent with p = 99% (notes/kalshi_sizing.md).

### 21. Confidence intervals and z-scores
- **Formula.** Wilson interval for rates (`edge_math.wilson`); bootstrap CIs clustered by night/event for P&L; z = mean/SE.
- **Evidence.** Kicker 84.7% [80.6, 88.1] excludes the claimed 90% (notes/nfl_kicker_system.md); weather 99c bin
  6.2/1,000 [2.1, 18.2] vs break-even 9.3 (notes/kalshi_sizing.md).

### 22. Multiple testing (Bonferroni, BH-FDR)
- **What.** Test 300 ideas at p < .05 and ~15 "work" by chance.
- **Formula.** Bonferroni: p < alpha/m. BH (Benjamini & Hochberg 1995): sort p, reject p_(i) <= (i/m) q.
- **We apply it.** `fund_stats.py`: ledger-wide Bonferroni (p < 1.42e-4 at 351 trials), per-dataset bar .05/trials,
  LEAD needs BH q < .10 AND both halves agree; agents can't override the verdict (FUND_BOOK s4).
- **Evidence.** ~300 trials give ~15 false p < .05 (FUND_BOOK lesson 7); pattern mine 2,280 cells -> sealed confirm 0/10
  (FUND_BOOK 3d PM10).

### 23. Garden of forking paths; pre-registration
- **What.** Choosing the cut, threshold or metric after seeing data inflates significance even with one "test".
- **We apply it.** `prereg` skill + QUESTIONS.md template; rule in the script docstring, sha256 recorded BEFORE the
  market join; after the hash the scorer is frozen and a fix = `void_bug` row + new hash (FUND_BOOK s4). Every trial,
  including failures, goes to lab/ledger.jsonl.
- **Evidence.** ForecastEx docstring frozen (sha256 8a6edbb5...) before any price pull (notes/forecastex_wx.md);
  WCN/DUL flagged "post hoc" and kept as leads, not edges (FUND_BOOK 3c).

### 24. Holdout / lockbox
- **What.** A final sample never looked at during development, scored once.
- **We apply it.** `scripts/freeze.py` + MANIFEST; `lab/run.py promote --confirm-lockbox`; sealed scored once on the
  PM's explicit "yes", never re-scored; one-shot guards abort if a result exists (protocol rules 1-3).
- **Evidence.** 15 sealed scores (lab/lockbox.jsonl): passes E1, X1, E4, E3, FX1; fails R1, O1, R_REB, L28, L37, M3, E2p,
  C1, L32 (inconclusive at ~25% power).

### 25. Deflated Sharpe / probability of backtest overfitting (PBO)
- **What.** Bailey & Lopez de Prado (2014) deflate a Sharpe ratio for the number of variants tried and non-normal
  returns; PBO (Bailey et al. 2017) estimates, by combinatorial cross-validation, how often the in-sample best variant
  is below median out of sample.
- **We apply it.** **NOT implemented.** notes/FUND_ARCHITECTURE.md s5 calls for "SPA / PBO" when selecting over a family
  and lists a "deflated metric" for the gate, but `scripts/fund_stats.py` has neither (no deflated/PBO code; checked
  2026-10-29). Our substitutes: Bonferroni/BH over the ledger, both-halves sign check, sealed lockbox.
- **How it would apply.** Variant families such as E4's M2/M3 x N1/N10 (lockbox `secondary`), the 14 kalshi_sizing
  rows, or the 12 wx_timing trials: run CSCV over months to get PBO before picking the winner.

### 26. Overfitting
- **What.** A model or rule that fits noise in dev and decays out of sample.
- **We apply it.** Few parameters, walk-forward fits, ONE change per question (prereg), sealed test.
- **Evidence.** M3 +4.3% dev -> +0.45% sealed at 10x volume; R1 rain dev +3.42% -> sealed +0.49%; E2p dev +11.4% ->
  sealed -6.5% (FUND_BOOK 3d).

### 27. Look-ahead and leakage
- **What.** Any input not knowable at decision time.
- **We apply it.** data-auditor stage before trusting a result; bug list in FUND_BOOK lesson 6 is the checklist.
- **Evidence (our own bugs).** datetime64[us] // 1e9 epoch bug froze in-game quotes -> fake +62%; NFL weather joined by
  row position (53% wrong) -> observed-weather oracle +32.7%/70 became +21.4% (notes/results.md 10-09), pre-kickoff
rain unders +8.0%/219 p .11 after the fix (FUND_BOOK 3b F2); observed in-game rain is not causal (pre-kickoff only); revised FRED
  vintages leak into econ tests; lifetime volume is look-ahead in "small markets are mispriced"
  (notes/smallmarket_calibration.md); ESPN prop last_updated is post-tip; Kalshi expected_expiration = start + 6h
  (FUND_BOOK lesson 6, notes/results.md).

### 28. Survivorship bias
- **What.** Testing only what still exists (tickers, markets, tipsters with public records).
- **Evidence.** Company graph: only 679 of 2,300 supplier CIKs have a ticker today, tests ran on survivors
  (notes/company_graph.md); RotoBot has no public record of losses (FUND_BOOK 3c L17); vendor P&L claims unverifiable
  (notes/social_edges_2026-10-22.md).

## E. Operations

### 29. Record keeping and honest P&L
- **What.** Every bet, fee, void and paper pick recorded; paper and live kept apart; P&L net of fees.
- **We apply it.** `bet.py` is the one router (API and phone via `log-external`), append-only journal
  `<deploy>/data/bets.jsonl`, units = dollars at risk incl. fee; executors log every order/fill; paper loggers report
  separately; the ledger keeps failures. Private balances never enter the repo.

### 30. Hedging, arbitrage, middles -- real vs fake
- **What.** A true arb locks profit across venues only if both legs resolve on the SAME event under the SAME rules and
  both fill at quoted size after fees. A middle profits if the result lands between two lines.
- **Formula.** Two-leg arb exists iff c_A(YES) + c_B(NO) < 1 after fees, with identical resolution.
- **We apply it.** Cross-venue scans list resolution text before calling an arb (scripts/xvenue_elections.py).
- **Evidence.** 6 flagged midterm "arbs" were 3-candidate races = false; LA Senate real gap ~.965 + fee but Kalshi rule
  text says "Kentucky" (resolution risk) (notes/results.md 10-20); Kalshi vs Polymarket weather 0/48 brackets under $1
  after fees; cheapest sports round trip $1.021 (FUND_BOOK 3d).

### 31. Discipline / tilt
- **What.** Emotion-driven deviation after wins or losses.
- **We apply it.** Deterministic executors with caps, KILL and loss stops; agents never place orders or change .env;
  PM decisions are explicit "yes <name>" asks; judgment bets limited to Tier C; no bet without a pre-committed rule
  (L2 DEN@OKC "no bet by pre-committed rule", FUND_BOOK 3c).

---------------------------------------------------------------------------------------------------------------------
## F. Checklist before any bet
1. Is the rule pre-registered, ledgered and past its gate (sealed pass / paper tier)? If not: paper only.
2. Price = executable ask (or modelled maker fill) + venue fee -> all-in cost c (`edge_math.py bet`).
3. Fair p from a de-vigged SHARP price (power/Shin) or a validated model -- never a hit rate alone, never vs 50%.
4. Base rate and matched baseline checked (is the "system" just the base rate?).
5. EV per $ > 0 at the point estimate AND Kelly > 0 at the lower confidence bound.
6. Stake = k x Kelly(p_LCB), k <= 1/2, then per-bet, per-night (correlation) and liquidity caps; tier cap in bet.py.
7. Same resolution source/rule text on every leg (cross-venue).
8. Route through `bet.py` (journal) or the executor; never chase; respect stops and KILL.
9. Afterwards: log CLV; review vs the sealed CI after 100-200 bets.

## G. Gaps -- principles the fund does not yet enforce
- Deflated Sharpe / PBO / SPA for variant selection: called for in FUND_ARCHITECTURE s5, absent from fund_stats.py.
- Proportional de-vig still in `sports/market.py`; Shin de-vig not yet applied in the gap logger (N4 open).
- Kalshi executor sizes flat (TIER_FRAC), not Kelly-on-LCB -- deliberate (notes/kalshi_sizing.md), but bet.py manual
  tiers also don't compute Kelly; the Kelly check is advisory (`ev_check.py` / `edge_math.py`), not a gate.
- No automated CLV logging on live E1/E2/E5 fills (CLV is tracked only for the opener loggers).
- No cross-strategy correlation cap (e.g. Kalshi E1 + ForecastEx E5 on the same city-night) -- caps are per executor.
- Growth-fraction wording errors in notes/risk.md and notes/kalshi_sizing.md (section 12) -- not corrected here.
