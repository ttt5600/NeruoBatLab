"""Strategy library. Each function maps prices -> target weights.

Contract: the row at index ``t`` may use prices up to and including ``t``, and
nothing after. Every rolling statistic here is causal for that reason. The
execution lag is applied by the engine, so none of these shift anything
themselves -- doing it twice would silently throw away a day of signal.

The docstrings say what published evidence each idea rests on, and where it is
known to fail. That ordering is roughly the order I would trust them in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _equal_weight_mask(mask: pd.DataFrame) -> pd.DataFrame:
    n = mask.sum(axis=1).replace(0, np.nan)
    return mask.div(n, axis=0).fillna(0.0)


# --------------------------------------------------------------------------
# benchmarks
# --------------------------------------------------------------------------
def buy_and_hold(prices: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Constant target weights. The benchmark almost nothing beats net of cost.

    Any strategy that cannot clear this, after the costs in ``CostModel``, is a
    worse version of doing nothing.
    """
    if weights is None:
        w = _equal_weight_mask(prices.notna())
    else:
        w = pd.DataFrame(
            {c: float(weights.get(c, 0.0)) for c in prices.columns}, index=prices.index
        ).where(prices.notna(), 0.0)
    return w


def sixty_forty(prices: pd.DataFrame, equity: str = "SPY", bond: str = "TLT") -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if equity in w:
        w[equity] = 0.6
    if bond in w:
        w[bond] = 0.4
    return w.where(prices.notna(), 0.0)


# --------------------------------------------------------------------------
# risk-based allocation -- no return forecast at all
# --------------------------------------------------------------------------
def inverse_vol(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Weight each asset by 1/vol. A "risk parity" that forecasts no returns.

    Volatility is far more forecastable than direction -- this is the most
    reliable free lunch in the file, and it needs no view on anything.
    """
    rets = prices.pct_change()
    vol = rets.rolling(lookback, min_periods=lookback // 2).std()
    inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    inv = inv.where(prices.notna())
    return inv.div(inv.sum(axis=1), axis=0).fillna(0.0)


def volatility_target(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = 60,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Overlay: scale any weight matrix to a constant forecast volatility.

    Realised vol is strongly autocorrelated (today's calm predicts tomorrow's
    calm), so this actually works -- it is the best-documented Sharpe
    improvement available, and it mostly earns it by shrinking before crashes
    rather than by picking winners.

    Note the scalar uses only trailing data, so it is causal.
    """
    rets = prices.pct_change()
    port = (weights * rets).sum(axis=1)
    realised = port.rolling(lookback, min_periods=lookback // 2).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / realised).replace([np.inf, -np.inf], np.nan)
    scale = scale.clip(upper=max_leverage).fillna(0.0)
    return weights.mul(scale, axis=0)


# --------------------------------------------------------------------------
# momentum -- the family with the most out-of-sample support
# --------------------------------------------------------------------------
def time_series_momentum(
    prices: pd.DataFrame, lookback: int = 252, long_only: bool = False
) -> pd.DataFrame:
    """Hold each asset long if its own trailing return is positive.

    Moskowitz, Ooi & Pedersen (2012) document this across 58 instruments and
    ~100 years. It is a genuine anomaly, but it is a *slow* one: the edge lives
    at multi-month horizons, and it pays for it with long flat stretches and
    sharp losses at trend reversals (2009, 2020 both hurt).
    """
    trail = prices / prices.shift(lookback) - 1
    sig = np.sign(trail)
    if long_only:
        sig = sig.clip(lower=0.0)
    sig = sig.where(prices.notna(), 0.0)
    n = prices.notna().sum(axis=1).replace(0, np.nan)
    return sig.div(n, axis=0).fillna(0.0)


def cross_sectional_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    n_long: int = 2,
    n_short: int = 0,
) -> pd.DataFrame:
    """Rank assets on trailing return, long the top, optionally short the bottom.

    The ``skip`` is not decoration: Jegadeesh (1990) showed the most recent
    month reverses, so 12-1 momentum skips it. Include it and you blend a
    momentum signal with a reversal signal and get neither.
    """
    trail = prices.shift(skip) / prices.shift(lookback) - 1
    trail = trail.where(prices.notna())
    ranks = trail.rank(axis=1, ascending=False)
    valid = trail.notna().sum(axis=1)

    longs = (ranks <= n_long) & trail.notna()
    w = _equal_weight_mask(longs)
    if n_short > 0:
        shorts = (ranks > (valid.values[:, None] - n_short)) & trail.notna()
        w = w - _equal_weight_mask(pd.DataFrame(shorts, index=w.index, columns=w.columns))
    return w.fillna(0.0)


# --------------------------------------------------------------------------
# mean reversion -- real, but the costs eat it
# --------------------------------------------------------------------------
def mean_reversion_zscore(
    prices: pd.DataFrame, lookback: int = 5, z_cap: float = 2.0
) -> pd.DataFrame:
    """Fade short-horizon moves: long what fell, short what rose.

    The effect is real in equity indices, but it rebalances daily. Watch what
    ``ann_turnover`` does to the net Sharpe -- this is the clearest example in
    the file of a gross edge that friction removes.
    """
    rets = prices.pct_change()
    mu = rets.rolling(lookback, min_periods=lookback).mean()
    sd = rets.rolling(lookback, min_periods=lookback).std()
    z = ((rets - mu) / sd).replace([np.inf, -np.inf], np.nan)
    sig = (-z).clip(-z_cap, z_cap) / z_cap
    sig = sig.where(prices.notna(), 0.0).fillna(0.0)
    n = prices.notna().sum(axis=1).replace(0, np.nan)
    return sig.div(n, axis=0).fillna(0.0)


# --------------------------------------------------------------------------
# the folklore one -- included as the overfitting exhibit
# --------------------------------------------------------------------------
def sma_crossover(
    prices: pd.DataFrame, fast: int = 50, slow: int = 200, long_only: bool = True
) -> pd.DataFrame:
    """Long when fast SMA > slow SMA.

    Two free integers over a few thousand daily bars is enough rope to fit noise
    precisely, which is why this is the strategy used in ``scripts/run_demo.py``
    to show a sweep winner collapsing out of sample. The 50/200 "golden cross"
    is folklore with a survivorship story attached, not a documented anomaly.
    """
    f = prices.rolling(fast, min_periods=fast).mean()
    s = prices.rolling(slow, min_periods=slow).mean()
    sig = np.sign(f - s)
    if long_only:
        sig = sig.clip(lower=0.0)
    sig = sig.where(prices.notna(), 0.0)
    n = prices.notna().sum(axis=1).replace(0, np.nan)
    return sig.div(n, axis=0).fillna(0.0)


REGISTRY = {
    "buy_and_hold": buy_and_hold,
    "sixty_forty": sixty_forty,
    "inverse_vol": inverse_vol,
    "time_series_momentum": time_series_momentum,
    "cross_sectional_momentum": cross_sectional_momentum,
    "mean_reversion_zscore": mean_reversion_zscore,
    "sma_crossover": sma_crossover,
}
