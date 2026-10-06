# quant — an automated trading harness that refuses to flatter you

A backtesting and strategy framework for daily-bar systematic trading, built so
that the evaluation is harder to fool than the strategy is to write.

The premise: retail algo-trading rarely fails because the strategy was bad. It
fails because the *backtest* was optimistic, and nobody found out until real
money was on the line. So the engineering effort here goes into the measurement
apparatus, not the signals.

```bash
python3.11 -m venv .venv_quant
.venv_quant/bin/pip install numpy pandas scipy matplotlib pytest
.venv_quant/bin/python -m pytest tests/ -q     # 67 tests
.venv_quant/bin/python scripts/run_demo.py     # the full experiment
```

## What's here

| module | role |
|---|---|
| `quant/data.py` | Adjusted daily bars from Yahoo's public chart endpoint. No API key. Disk cache keyed by **symbol and window**. |
| `quant/backtest.py` | Vectorised engine. Execution lag, turnover with drift, commissions, spread, slippage, borrow, financing. |
| `quant/strategies.py` | Seven strategies, each a pure function `prices -> target weights`. |
| `quant/stats.py` | Performance metrics plus the chance-corrections: PSR, Deflated Sharpe, PBO, block bootstraps. |
| `quant/walkforward.py` | Parameter selection on trailing data only, with an embargo. |
| `scripts/run_demo.py` | The three-part experiment below. |

## The three design decisions that matter

**1. The execution lag lives inside the engine.**
`target_weights.loc[t]` is the decision made at the close of day `t`; it earns
day `t+1`'s return. The `.shift()` that enforces this is in `run_backtest`, not
in any strategy, so a strategy author cannot forget it or quietly remove it to
improve a Sharpe. `test_execution_lag_defeats_the_classic_close_to_close_leak`
shows a signal built from today's close printing Sharpe **42.3** at `lag=0` and
**−0.17** at `lag=1`.

The engine's guarantee has a limit, and the tests state it explicitly: it
controls *when* a decision executes, not whether the decision used data that
existed yet. A strategy defined as `sign(return[t+1])` still prints 42.3. Only
`test_every_strategy_is_causal` — which checks that weights computed on
truncated history match weights computed on full history — catches that.

**2. Costs include the ones people forget.**
Constant weights are not free to maintain: positions drift with prices, so
turnover is measured against the *drifted* book, not the previous target.
Shorts accrue borrow; leverage above 1× accrues financing. Cash earns nothing,
which is deliberately conservative — a strategy that sits in cash gets no
T-bill yield and must beat buy-and-hold on price alone.

**3. Every metric is validated on a synthetic null before it ranks anything real.**
PBO must return ~0.5 on 200 coin-flip strategies and <0.2 when one has a real
edge. The Deflated Sharpe must refuse to certify the winner of a 500-way noise
sweep while the naive test reports >0.99. Those are tests, not assertions in
prose.

## The experiment (`scripts/run_demo.py`)

10 liquid ETFs, 2007-01-03 to 2026-09-11, 4954 trading days, net of costs.

### A. The same grid, scored two ways

A 22-configuration SMA crossover sweep:

| | Sharpe |
|---|---|
| Full-sample winner (SMA 10/150) | **0.821** |
| Same grid chosen walk-forward | 0.789 |
| Its **excess over buy-and-hold** | **−0.234** |

| null | verdict |
|---|---|
| DSR vs zero | **0.999** — certified |
| DSR on excess over buy-and-hold | **0.084** — rejected |

The winner is not a good strategy. It is a worse version of holding the index,
and testing it against zero hides that completely, because a long-only equity
rule collects the equity risk premium whether or not its timing does anything.

PBO = **0.635**, and the walk-forward parameter choice lurches from 20/50 to
100/250 to 5/50 across folds. That instability is the finding: a strategy whose
optimum moves every year does not have an optimum.

### B. Honest horse race — a priori parameters, net of costs

Nothing beat equal-weight buy-and-hold after correcting for multiple tests.

| strategy | Sharpe | Δ vs B&H | raw p | Holm p |
|---|---|---|---|---|
| `sixty_forty` | 0.776 | +0.141 | 0.096 | 0.672 |
| `inverse_vol` | 0.756 | +0.121 [−0.004, +0.242] | **0.028** | **0.221** |
| `buy_and_hold_EW` | 0.635 | — | — | — |
| `xsmom_12_1_top2` | 0.459 | −0.176 | 0.832 | 1.000 |
| `meanrev_5d` | −0.021 | −0.655 | 0.980 | 0.980 |

`inverse_vol` is the only raw p<0.05 — and it does not survive Holm. One hit in
eight tries is roughly one expected false positive.

Two details worth the price of admission:

- **`meanrev_5d` earns a gross Sharpe of 0.565 and a net Sharpe of −0.021.**
  135× annual turnover, 3.5%/yr in costs. The edge is real and the friction
  eats all of it. This is the most common way a good backtest becomes a losing
  account.
- **The significance test has to match the claim.** Risk-based allocation
  improves Sharpe by cutting volatility, not by raising return — so a test on
  *mean excess return* is structurally blind to it. `inverse_vol` scores
  p=0.944 on the mean test and p=0.028 on the Sharpe test. Same data, opposite
  conclusions; only one of them is asking the right question.

### C. Null control

Geometric Brownian motion with fixed drift — no timing signal exists by
construction. The same sweep still finds SMA 20/250 at Sharpe **1.248**, with
DSR-vs-zero of **1.000**. A strategy with provably zero skill gets certified.
It still loses to buy-and-hold (1.351), which is the only comparison that had
any content.

## A bug worth keeping in the record

The first full run silently reported on 2015–2026 instead of 2007–2026. The
per-symbol cache ignored the requested date range, so an earlier smoke test
that fetched SPY and TLT from 2015 capped every later call; `dropna(how="any")`
then truncated the whole panel to the shortest column. Eight years and the GFC
vanished, no error was raised, and every number was plausible.

Fixed in two places: the cache now stores maximal history and records the window
it fetched (`test_cache_is_not_poisoned_by_an_earlier_narrower_request`), and
`run_demo.py` refuses to run if the common window is more than a year shorter
than requested. Silent truncation is the failure mode to fear, because plausible
wrong numbers do not prompt anyone to look.

## Going live (`quant/execution.py`)

The seam between research and a real account. `plan_rebalance` turns target
weights into an order list against the positions you actually hold;
`PaperBroker` is the reference implementation and charges the same `CostModel`
the backtest assumed, so live results don't diverge for reasons unrelated to
the strategy.

```bash
.venv_quant/bin/python scripts/live_signal.py --strategy inverse_vol --equity 25000
.venv_quant/bin/python scripts/live_signal.py --positions "SPY=12,TLT=40" --cash 500
```

Three rails, all of them consequences of the findings above:

- **A no-trade band (default 2pp).** This is the `meanrev_5d` lesson made
  structural. On a fixed price path, 250 days of rebalancing a 3-asset book
  drops from **617 orders / $213.8k notional** to **28 orders / $112.4k** —
  order count falls 22×. Chasing the target exactly is how a real gross edge
  becomes a losing account.
- **`dry_run=True` by default.** Placing real orders is an explicit argument at
  the call site, never a config file's job.
- **Guardrails reject, never clamp.** A NaN weight, a non-positive price, a
  missing mark for a held position, or gross exposure over the cap raises
  `GuardrailError` naming the violation. Silently coercing bad input into
  something tradeable is how upstream bugs reach the market.
- **Buys are capped to what the account can actually pay for.** A buy for `N`
  of notional removes `N × (1 + fee)` from cash, so sizing on clean notional
  overspends on every rebalance and a gross-1.0 target walks cash negative —
  a margin call in a cash account, and silent leverage in any account. Found by
  a test asserting `cash >= 0`, not by inspection.

## Forward paper trading (`quant/paper.py`)

```bash
python scripts/paper_run.py init --strategy inverse_vol --cash 25000
python scripts/paper_run.py step        # once per trading day; idempotent
python scripts/paper_run.py report
```

A JSON journal of every day's plan, fills, costs and positions. `step` is
idempotent on the bar date, so a retrying cron cannot double-trade, and saves
are atomic so a crash mid-write can't corrupt the journal.

The point is not the paper P&L — a few months of it says almost nothing about
skill, and reading it as though it does repeats the error this repo exists to
prevent. The point is `divergence()`: **does the live path reproduce what the
backtest claims for the same days?** Replaying 2026-03-02 → 2026-09-11:

| | return | Sharpe | ann. turnover |
|---|---|---|---|
| paper | +2.828% | 0.606 | 2.15× |
| backtest | +2.564% | 0.555 | 2.80× |

Correlation **0.9990**, mean daily gap **+0.19 bps**. The paper path runs
slightly *ahead* because the band traded on 11 days out of 135 instead of
continuously, and the saved costs exceed the tracking error. That gap is
explained, which is the whole test — an unexplained gap means the live system
and the research system disagree, and neither number is trustworthy until you
know why.

### The journal that froze without an error

Three weeks after the go/no-go rule was sealed, the journal still ended at
2026-09-11, with zero live days. Two things had gone wrong. Nothing was running
`step`. And even if something had been, it would have done nothing: the cache
only went stale when a request asked for *older* history, never newer, so every
call got the 2026-09-11 bar back, `step` printed "already processed", and the
exit code was 0. It's the same kind of failure as the truncation bug above:
plausible output, no exception.

Now:

- **The cache refreshes forward.** It refetches once a new session has settled
  (17:00 New York) and drops any bar from an unsettled session, so an intraday
  quote never gets cached as a close. On a holiday, a fetch made after the
  close finds no bar and doesn't retry on every call.
- **`step` refuses stale data** (newest bar more than 3 business days old) and
  exits 2, so a scheduler sees the failure.
- **Missed days are caught up one bar at a time**, labelled `catchup`. Stepping
  only the newest bar would book three weeks as one "daily" return. Catch-up
  bars don't count as live evidence. Changing that would be a rule revision.
- **Positions carry across dividend re-adjustments.** A refresh rescales
  adjusted history: on 2026-10-06 SPY's 09-11 close moved −0.25% and TLT's
  −0.40%. Valuing the old share counts at new prices would book that as a loss
  (about −31 bps on the seam day). After that, the live path would track raw
  prices and miss every dividend, roughly 1 bp/day for 60/40, which is half the
  rule's tracking budget. Quantities are rescaled by old mark / new price for
  the same bar, so paper returns are total returns, like the backtest's.

`scripts/paper_cron.sh` runs `step` at 17:30 New York on weekdays via a
launchd agent (`com.jonathanwang.quant-paper-step`). If it fails, it posts a
macOS notification.

**The sealed rule can no longer be met on its date.** The 2026-10-06 → 2027-03-19
window holds about 115 trading days, short of the 126 live days required, so
`check` will say NO-GO on the decision date. That is the rule doing its job, so
it is left as sealed. With no further missed days, the 126th live day falls
around 2027-04-06. Moving the date is allowed through `supersede`, which keeps
the original on the record.

`live_signal.py` computes weights from the **most recent completed daily bar**,
matching the backtest's convention. Running it intraday and filling immediately
reintroduces the exact close-to-close leak the engine exists to prevent.

This module holds no credentials, makes no network calls, and ships no live
broker adapter — that stays in your environment, not in the repo.

## Honest limits

- **Daily close-to-close only.** Nothing here says anything about intraday.
- **Costs are modelled, not measured.** ~2.5 bps/trade suits liquid ETFs in
  retail size, and is far too optimistic for small caps or size.
- **No live broker integration**, deliberately. This measures whether an idea
  is worth trading; it does not route orders.
- **Survivorship bias in the universe.** These ten ETFs all still exist, which
  is itself a selection.
- **One market regime.** 2007–2026 is a single, largely bullish sample. Two
  decades of daily bars is a smaller effective sample than it looks.
