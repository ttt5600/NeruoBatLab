"""Vectorised daily backtester with an execution lag you cannot switch off by accident.

Timing convention (the one thing that decides whether a backtest is honest):

    target_weights.loc[t]  = the position the strategy CHOSE using information
                             available at the close of day t.
    It is executed at that close and earns the return of day t+1.

The ``.shift(lag)`` that enforces this lives *inside* :func:`run_backtest`, not
in any strategy. A strategy author therefore cannot forget it, and cannot
"improve" a Sharpe by quietly removing it. ``tests/test_backtest.py`` proves
the property directly: perturbing a future price leaves earlier P&L untouched.

Costs charged:
  * commission + half-spread + slippage on traded notional, both sides
  * stock-borrow on gross short notional, daily accrual
  * financing on gross exposure above 1x, daily accrual

Cash is assumed to earn nothing. That is deliberately conservative for timing
strategies: a model that sits in cash gets no T-bill yield, so it has to beat
buy-and-hold on price alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class CostModel:
    """Round-trip friction, in basis points of traded notional unless noted.

    Defaults are realistic-to-slightly-pessimistic for liquid US ETFs traded in
    retail size. They are NOT realistic for small caps, and they are wildly
    optimistic for anything intraday.
    """

    commission_bps: float = 0.5
    spread_bps: float = 1.0  # half-spread paid per side
    slippage_bps: float = 1.0  # market impact
    short_borrow_bps_annual: float = 50.0
    financing_bps_annual: float = 100.0  # cost of leverage above 1x

    @property
    def per_trade_bps(self) -> float:
        return self.commission_bps + self.spread_bps + self.slippage_bps


FRICTIONLESS = CostModel(0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class BacktestResult:
    gross_returns: pd.Series
    net_returns: pd.Series
    weights_held: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series

    @property
    def equity(self) -> pd.Series:
        return (1 + self.net_returns).cumprod()

    def __repr__(self) -> str:  # keeps notebook output readable
        from .stats import summarise

        s = summarise(self)
        bits = " ".join(f"{k}={v:.3f}" for k, v in s.items() if isinstance(v, float))
        return f"<BacktestResult n={len(self.net_returns)} {bits}>"


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    costs: CostModel = CostModel(),
    execution_lag: int = 1,
    max_gross_leverage: float = 3.0,
) -> BacktestResult:
    """Apply ``target_weights`` to ``prices`` and return gross/net performance.

    Parameters
    ----------
    prices
        Adjusted closes, one column per asset.
    target_weights
        Indexed like ``prices``. Row ``t`` is the decision made at the close of
        ``t``. Missing entries are treated as flat.
    execution_lag
        Bars between decision and the return it earns. ``1`` (the default) is
        the only honest setting for close-to-close daily data. ``0`` is allowed
        solely so tests can demonstrate the lookahead it creates.
    """
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0")

    rets = prices.pct_change()
    w = target_weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

    # An asset with no price today cannot be held today.
    w = w.where(prices.notna(), 0.0)

    gross = w.abs().sum(axis=1)
    over = gross > max_gross_leverage
    if over.any():
        w = w.div(np.where(over, gross / max_gross_leverage, 1.0), axis=0)

    held = w.shift(execution_lag)  # <-- the lag. Inside the engine, on purpose.
    held = held.fillna(0.0)

    gross_ret = (held * rets.fillna(0.0)).sum(axis=1)

    # Positions drift with prices between rebalances; turnover must price that
    # in, or a constant-weight portfolio looks free to maintain.
    drifted = held * (1 + rets.fillna(0.0))
    denom = (1 + gross_ret).replace(0.0, np.nan)
    drifted = drifted.div(denom, axis=0).fillna(0.0)

    turnover = (w - drifted).abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = w.iloc[0].abs().sum()  # initial entry is a real trade

    trade_cost = turnover * costs.per_trade_bps / 1e4

    short_notional = held.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_notional * costs.short_borrow_bps_annual / 1e4 / TRADING_DAYS

    excess_gross = (held.abs().sum(axis=1) - 1.0).clip(lower=0.0)
    financing = excess_gross * costs.financing_bps_annual / 1e4 / TRADING_DAYS

    total_costs = trade_cost + borrow + financing
    net_ret = gross_ret - total_costs

    valid = rets.notna().any(axis=1)
    keep = valid & (prices.index > prices.index[0])

    return BacktestResult(
        gross_returns=gross_ret[keep],
        net_returns=net_ret[keep],
        weights_held=held[keep],
        turnover=turnover[keep],
        costs=total_costs[keep],
    )
