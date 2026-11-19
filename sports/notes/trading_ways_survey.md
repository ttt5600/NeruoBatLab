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
