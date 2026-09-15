"""Walk-forward evaluation: choose parameters using only the past.

This is the portfolio analogue of a leave-birds-out split. A full-sample grid
search reports the score of the configuration that best fits the sample you
scored it on -- the number is a property of the search, not of the strategy.
Walk-forward instead re-picks parameters on a trailing window and trades them
forward, so every return in the output series was earned by a configuration
chosen before that day existed.

The embargo matters and is easy to miss. A strategy with a 252-day lookback
computes its first out-of-sample signal from prices that lie inside the
training window. Without a gap, the split leaks. ``embargo_days`` should be at
least the longest lookback in the grid.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from .backtest import CostModel, run_backtest
from .stats import sharpe


def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of a ``{param: [values]}`` dict, as a list of kwargs."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    folds: pd.DataFrame
    param_grid_size: int
    all_is_sharpes: list = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.folds)


def grid_search_full_sample(
    prices: pd.DataFrame,
    strategy_fn,
    grid: dict,
    costs: CostModel = CostModel(),
) -> pd.DataFrame:
    """Score every configuration on the whole sample. The dishonest baseline.

    Useful only to quantify how much the honest number is lower. The winning
    row's Sharpe is an order statistic; feed the whole ``sharpe`` column to
    :func:`~quant.stats.deflated_sharpe_ratio` before believing any of it.
    """
    rows = []
    for params in expand_grid(grid):
        w = strategy_fn(prices, **params)
        res = run_backtest(prices, w, costs=costs)
        rows.append({**params, "sharpe": sharpe(res.net_returns),
                     "n_days": len(res.net_returns)})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def walk_forward(
    prices: pd.DataFrame,
    strategy_fn,
    grid: dict,
    train_days: int = 1260,
    test_days: int = 252,
    embargo_days: int = 252,
    costs: CostModel = CostModel(),
    min_train_days: int = 252,
) -> WalkForwardResult:
    """Re-select parameters each fold on trailing data, trade them forward.

    Returns the stitched out-of-sample series plus a per-fold table showing
    which configuration won each time. Parameter instability across folds is
    itself a finding: a strategy whose optimum jumps every year does not have
    an optimum, it has noise.
    """
    combos = expand_grid(grid)
    idx = prices.index
    n = len(idx)
    oos_parts, fold_rows, is_sharpes = [], [], []

    start = max(train_days, min_train_days)
    for test_start in range(start, n, test_days):
        test_end = min(test_start + test_days, n)
        if test_end - test_start < 20:
            break

        train_hi = max(0, test_start - embargo_days)
        train_lo = max(0, train_hi - train_days)
        if train_hi - train_lo < min_train_days:
            continue

        train_px = prices.iloc[train_lo:train_hi]
        best, best_sr = None, float("-inf")
        for params in combos:
            w = strategy_fn(train_px, **params)
            sr = sharpe(run_backtest(train_px, w, costs=costs).net_returns)
            is_sharpes.append(sr)
            if pd.notna(sr) and sr > best_sr:
                best, best_sr = params, sr
        if best is None:
            continue

        # Warm up the chosen config on history so its lookbacks are populated
        # on the first test day, then keep only the test slice.
        warm_lo = max(0, test_start - max(embargo_days, 300))
        w_full = strategy_fn(prices.iloc[warm_lo:test_end], **best)
        res = run_backtest(prices.iloc[warm_lo:test_end], w_full, costs=costs)
        oos = res.net_returns.loc[idx[test_start] : idx[test_end - 1]]
        if oos.empty:
            continue

        oos_parts.append(oos)
        fold_rows.append({
            "test_start": idx[test_start], "test_end": idx[test_end - 1],
            "train_start": idx[train_lo], "train_end": idx[max(train_hi - 1, 0)],
            "is_sharpe": best_sr, "oos_sharpe": sharpe(oos),
            "n_oos_days": len(oos), **{f"p_{k}": v for k, v in best.items()},
        })

    oos_returns = (
        pd.concat(oos_parts).sort_index() if oos_parts else pd.Series(dtype=float)
    )
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    return WalkForwardResult(
        oos_returns=oos_returns,
        folds=pd.DataFrame(fold_rows),
        param_grid_size=len(combos),
        all_is_sharpes=is_sharpes,
    )
