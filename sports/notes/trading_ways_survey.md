# trading_ways_survey -- other ways to make money trading on a $1-2k risk budget (started 2026-11-18)

Agent: trading_ways_survey (research subagent of the quant-trader lead; protocol sports/notes/AGENT_PROTOCOL.md).
Scratch: sports/results/trading_ways_survey/.

## Question
User (CA resident; Kalshi, Robinhood (MCP), Schwab, Fidelity, Public, IBKR being emptied; $1-2k risk budget) asks for
"all other ways to make money through trading" beyond prediction markets, ideally something that could double the
bankroll in a month. Deliverable: ranked table with mechanism, best evidence, realistic monthly return after costs/taxes,
capacity, ruin risk, time cost, CA legality, free-data backtestability. Pre-register + run the top 1-2 backtestable ones.

## Do-not-redo (read 2026-11-18)
- FUND_BOOK 3d "Stocks / thesis / earnings": 14 ETF timing rules, pairs, twins (47 trials), company-graph momentum,
  Polymarket earnings, L11 earnings straddle T-5 (-3.95% [-5.34,-2.53]), Q009 SPY short-vol spreads (n.d.).
- Literature-dead (notes/prior_research_2026-10-22.md s3/s5): overnight drift (gone 2021-25), pre-FOMC drift,
  S&P index adds (0.3%/decade), PEAD in liquid names (Martineau 2022), US earnings-announcement premium, 0DTE, naked
  short earnings straddles. NOT re-searched here except for 2024-26 updates.
- RSA + brokerage/bank bonuses live in the life project (~/.claude/learnings/LEARNINGS.md keys rsa.*, brokerage.*,
  tastytrade.sweetbonus, sofi.transfer_bonus). Key facts read: realistic RSA ~$15-30 per good split per slot
  (`rsa.realistic_value`, unverified estimate); no round-up credited yet as of 11-17 (`rsa.status.outcomes_2026-11-17`);
  Robinhood rounds up once per PERSON (`robinhood.rsa.roundup_scope`); Fidelity blocks split tickers
  (`fidelity.rsa.error_020255`); participant-level wording rising 0 (2023) -> 27 (2026-07..11) (`rsa.participant_level_trend`).
- CA: no legal online sportsbooks; pick'em DFS called illegal by AG opinion 23-1001; dual-currency sweepstakes ended
  2026-01-01 (AB 831) (notes/soft_venues.md s3).

## Plan
1. Two researcher subagents in parallel: (A) equity event/special-situation + options evidence (merger arb, odd-lot
   tenders, SPAC trust arb, IPO access, CEF tenders, thrift conversions, 2024-26 updates on PEAD/closing auction/
   index rebalancing, VRP/0DTE); (B) crypto for a US/CA resident (legal perp venues, funding carry evidence, staking,
   airdrops, exchange arb, CA DFAL) + prediction-venue promos legal in CA.
2. Me: (i) base rates for "double in a month" from free price data (descriptive, pre-registered below);
   (ii) pre-registered crypto funding-carry backtest on Hyperliquid public funding history (proxy for US venues).
3. Rank, write table, inbox rows.

## Log
- 2026-11-18: data access check. Binance fapi 451 (restricted location), Bybit 403 (country block); OKX public API 200
  (but funding history only ~3 months); Hyperliquid info API 200 with hourly funding from 2023; Coinbase Exchange
  candles 200. yfinance 1.7.0 in .venv_quant.

## PRE-REGISTRATIONS (written 2026-11-18 BEFORE any outcome was computed)

### DBL -- base rates for "double in a month" (descriptive, no money trial)
- Question: how often has each retail-accessible vehicle returned >= +100% in one calendar month, and how often
  <= -50%? This bounds what "double in a month" needs.
- Vehicles (yfinance daily adj close, full history): SPY, QQQ, TQQQ, SOXL, UPRO, BTC-USD, ETH-USD, MSTR, COIN, GME.
- Metric: share of calendar months (month-end close to month-end close) with r >= +100%, r <= -50%, and the median
  / 95th pct monthly r. Plus the "all-in each month" ladder: P(reach 2x within 1, 3, 6 months before hitting 0.5x),
  daily path, from every month start.
- No success/kill: descriptive. Report counts with exact (Clopper-Pearson) 95% CIs.

### FC1 -- crypto funding carry (long spot, short perp), BTC + ETH
- Mechanism: perp longs pay shorts funding when perps trade above spot (leveraged long demand). Delta-neutral
  holder collects funding.
- Data: Hyperliquid public info API fundingHistory (hourly, from 2023-05) for BTC, ETH. Proxy for a US venue (Coinbase
  Derivatives perp-style futures) -- US-venue funding is NOT the same series; a pass here = "worth a forward paper log
  on the legal venue", never a trade.
- Baseline: cash at the 13-week T-bill yield (yfinance ^IRX, daily, accrued).
- Trials (2, family "fc"): T1 always-on carry, BTC and ETH 50/50. T2 regime filter: hold carry only when the
  trailing-7-day mean funding (known at decision, decisions daily 00:00 UTC, funding from the NEXT hour on) is > 0;
  otherwise cash at T-bill.
- Capital: perp leg at 2x leverage -> capital = notional x (1 + 1/2) = 1.5 x notional; return on capital =
  funding x notional / capital. Idle cash portion earns nothing (conservative).
- Costs: round trip (enter + exit both legs) charged at 1.0% of notional per entry (main), sensitivity 0.5% and 2.0%;
  T1 pays one round trip per year (annual re-entry, conservative). Liquidation check: max 1-day and 7-day BTC/ETH
  rise vs the ~50% move that liquidates a 2x short (report; rebalancing cost not modelled beyond the round trip).
- Split (time-based): DEV 2023-05-12..2025-04-30; HOLDOUT 2025-05-01..2026-11-17. Rule frozen now; holdout read once.
- SUCCESS: holdout annualized net return on capital minus ^IRX > 0 AND the 90% monthly-block-bootstrap CI of the
  monthly excess excludes 0. KILL: point estimate <= 0 or CI crosses 0 ("not distinguishable").
- Also report: the monthly return on $1.5k capital in $, to answer "does it grow fast".

## RESULTS (2026-11-18/19)

### DBL (scripts/tw_double_baserate.py -> results/trading_ways_survey/dbl_baserates.json)
Calendar months with >= +100% (and exact 95% CI); "2x before 0.5x within 21 trading days" going all-in from each month start:
| vehicle | months | months >= +100% | P(2x first, 1m) [CI] | P(0.5x first, 1m) | P(2x first, 6m) | P(0.5x first, 6m) |
|---|---|---|---|---|---|---|
| SPY | 406 | 0 [0, .009] | 0 | 0 | 0 | 0 |
| QQQ | 332 | 0 | 0 | 0 | 0 | .009 |
| TQQQ | 201 | 0 [0, .018] | 0 | .005 | .056 | .097 |
| UPRO | 209 | 0 | 0 | .005 | .015 | .049 |
| SOXL | 200 | 1 (2026-04 +165%) | .010 [.001, .036] | .015 | .210 | .190 |
| BTC-USD | 146 | 0 [0, .025] | 0 | 0 | .147 | .070 |
| ETH-USD | 108 | 0 | 0 | 0 | .229 | .171 |
| MSTR | 341 | 3 | .026 [.012, .050] | .021 | .149 | .128 |
| COIN | 67 | 0 | .015 | .015 | .226 | .194 |
| GME | 297 | 2 (incl. 2021-01 +1625%) | .010 | .007 | .065 | .103 |
Reading: no unlevered or 3x-levered liquid vehicle has EVER doubled in a calendar month except SOXL once; the most
volatile single names reach 2x-first in ~1-3% of month starts, about as often as they halve first. Doubling in a month
is a ~1-in-40-to-1-in-100 outcome, not a plan. Over 6 months the 2x-first and 0.5x-first rates are roughly equal for
the volatile names (SOXL .210 vs .190, COIN .226 vs .194): a coin flip with ruin on the other side.

### FC1 (scripts/tw_fc1_carry.py -> results/trading_ways_survey/fc1_carry.json). Main cost 1.0% round trip.
| trial | split | ann. return on capital | T-bill (^IRX) | ann. excess | monthly excess 90% CI | $/month on $1.5k |
|---|---|---|---|---|---|---|
| T1 always-on | dev 2023-05..2025-04 | 11.19% | 4.92% | +6.27% | [+0.207%, +0.826%] | 13.98 |
| T1 always-on | HOLDOUT 2025-05..2026-11 | 4.93% | 3.83% | +1.10% | [-0.018%, +0.198%] | 6.16 |
| T2 7d-filter | dev | 10.23% | 4.92% | +5.31% | [+0.053%, +0.802%] | 12.78 |
| T2 7d-filter | HOLDOUT | 2.94% | 3.83% | -0.89% | [-0.306%, +0.132%] | 3.67 |
- **Verdict: KILL both** (holdout CI crosses 0 / point < 0 = "not distinguishable" from T-bills). At 0.5% cost T1
  holdout CI [+0.015%, +0.206%]/mo excludes 0 but that was a sensitivity, not the pre-registered cost.
- Gross funding on notional (BTC/ETH 50/50) by year: 2023 16.8%, 2024 23.1%, 2025 9.6%, 2026 YTD 5.75%. Carry
  compressed (Hyperliquid's floor is the 0.01%/8h interest term; median hourly rate 1.25e-5 = that floor).
- Liquidation check: ETH max 7-day rise since 2023-05 +47.7%, max 30-day +69.6% (BTC +30.8% / +58.9%) -> a 2x short
  leg would have needed rebalancing (extra cost not modelled) -- the result is optimistic if anything.
- Proxy caveat: Hyperliquid bars US persons (researcher B); US perps (Coinbase nano perp-style, Kraken via Bitnomial)
  may not be open to CA (unknown). Not registered in edgelab (Kalshi-spec registry; this is a non-Kalshi family "fc",
  2 trials, recorded here + inbox).

### Researcher sweeps (2026-11-18; full reports summarized, labels kept)
A (equity/options): odd-lot tender carve-outs near-riskless but rare & capped at 99 sh (no frequency count found;
Fidelity voluntary-reorg fee ~$38 [UNVERIFIED, stale schedule]; Robinhood supports voluntary tenders [FULL]);
SPAC trust arb ~3.6% yield in 2022 [FULL wealthmanagement], 2025 "fallen" [UNVERIFIED]; Ritter IPO first-day mean/median
2025 29.3%/13.7% (n=90) [FULL Table 1] but retail allocation rationed, Robinhood 60-day block if flipped <30 d [FULL];
merger arb: MNA ETF 10-yr 2.76%/yr to 2026-03-31 [UNVERIFIED factsheet]; thrift conversions: last-year 5 conversions
+0.7% after one week (RP Financial 2024) [UNVERIFIED]; insider/Congress following: no net retail edge post-STOCK Act;
retail options: effective spreads up to 12.6%, retail 0DTE net losses mostly costs [UNVERIFIED]; XIV -92.6% on
2018-02-05 [ABS].
B (crypto/promos): Coinbase staking still blocked for CA; airdrops geoblock US (Dragonfly: $1.84-2.64B lost by US users
2020-24 [UNVERIFIED]); DFAL list = applications under review incl. Coinbase/Kraken/Gemini/RH/Crypto.com [FULL DFPI];
CME Micro BTC margin ~$2,800 (> budget) [UNVERIFIED]; CA-available prediction-venue sign-up offers (affiliate pages,
defirate CA updated 2026-11-05 [FULL]): FanDuel Predicts $100, DraftKings Predictions $150, Gemini up to $100,
ProphetX $75, Polymarket US $25-50, Underdog $50 entries -- credit non-withdrawable; hedge on Kalshi. Ninth Circuit
2026-08-28 (CEA does not preempt Nevada) and 2026-09-16 (tribal lands) raise tail risk for sports contracts in CA
[UNVERIFIED secondary].

## RANKING (for $1-2k, CA resident; monthly % of $1.5k)
| # | strategy | expected | ruin | evidence | CA legal | next step |
|---|---|---|---|---|---|---|
| 1 | Prediction-venue sign-up promos hedged on Kalshi | one-time ~$300-450 face (cash value lower, unknown) | ~0 | affiliate pages [UNVERIFIED] | yes (CFTC venues) | user reads each venue's terms (stake-returned?); compute hedge lock-in per offer |
| 2 | RSA scaling (more round-up-per-account brokers) | ~$15-30/good split/slot (estimate, 0 credited yet) | ~0 | own, pending 12-07..12-11 | yes | wait for first credits before adding brokers; fund Schwab cash |
| 3 | Odd-lot tender / exchange offers | tens-hundreds $/rare event | ~0 | SEC rule + blog cases | yes | EDGAR "odd lot" SC TO-I watcher; check voluntary-action fees |
| 4 | IPO access (RH) | unknown (rationed) | low | Ritter [FULL] unconditional | yes | opt-in only; no flip <30 d |
| 5 | SPAC trust arb / T-bills | ~T-bill | low | 2022 [FULL] | yes | not worth effort at $1.5k |
| 6 | Crypto funding carry | +0.41%/mo gross (T1 holdout, $6/mo) | med (venue, liq.) | own FC1: KILLED | US venue in CA unknown | dead |
| 7 | Merger arb | ~0.23%/mo (MNA 10y) | med (breaks) | ETF proxy | yes | dead for $1.5k |
| 8 | Leveraged directional (SOXL/MSTR/10x perps/options) | ~0 edge; P(2x in 1m) 0-2.6% | high | own DBL | yes | only as a capped lottery, not a strategy |
| 9 | Short vol / 0DTE / earnings / PEAD / overnight / index adds | negative | high | own + lit. | yes | dead |

## Learnings
### Doubling in one month is a 1-3% tail event even for the wildest liquid vehicles
DBL: all-in from month starts, P(2x before 0.5x within 1 month) = SOXL .010, MSTR .026, COIN .015; SPY/QQQ/TQQQ/BTC/ETH 0.
Source results/trading_ways_survey/dbl_baserates.json. Over 6 months 2x-first ~ 0.5x-first. Use this when a "double
fast" goal comes up: it is a lottery with roughly symmetric ruin, not a strategy.
### Crypto funding carry compressed to ~T-bill by 2025-26
FC1 holdout 2025-05..2026-11, BTC/ETH 50/50 on Hyperliquid funding, 2x perp leg, 1% round trip: 4.93%/yr on capital vs
T-bill 3.83%, monthly excess 90% CI [-0.018%, +0.198%] -> killed. Gross funding 23.1% (2024) -> 5.75% (2026 YTD).
Source results/trading_ways_survey/fc1_carry.json. Dev (2023-25) looked great (+6.3%/yr excess): regime decay.
### Hyperliquid funding history is free and reachable from the US; Binance/Bybit are geoblocked
Hyperliquid info API fundingHistory returns hourly funding from 2023-05-12 (POST, ~500 rows/call, 429 if <1 s spacing);
Binance fapi 451, Bybit 403, OKX only ~3 months. Hyperliquid's ToS bars US persons from trading.
### For a $1-2k account the best $/hour items are one-off promos and capped corporate actions, not market edges
Promos (CFTC prediction venues open in CA), RSA, odd-lot tenders each pay tens to hundreds of dollars per event at ~zero
ruin; every market strategy with a cited post-2015 edge pays T-bill-like returns (SPAC, merger arb, carry).
