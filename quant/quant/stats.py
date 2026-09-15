"""Performance metrics, and the chance-corrections that stop them lying.

A Sharpe ratio read off the winner of a parameter sweep is not a Sharpe ratio;
it is the maximum of N draws, and the maximum of N draws from a zero-mean
distribution is positive by construction. This is the same failure that turns a
random-split probe into a +0.114 illusion: the number is inflated by the
selection that produced it, not by any underlying skill.

Three corrections are implemented, strongest last:

* :func:`probabilistic_sharpe_ratio` -- is this Sharpe distinguishable from a
  benchmark, given sample length, skew and fat tails?
* :func:`deflated_sharpe_ratio` -- the same test, with the benchmark raised to
  the level you would expect the *best of N trials* to reach by luck alone.
* :func:`probability_of_backtest_overfitting` -- across a whole strategy family,
  how often does the in-sample winner land below median out-of-sample?

References: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio";
Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest
Overfitting".
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats as sps

TRADING_DAYS = 252
_EULER = 0.5772156649015329


# --------------------------------------------------------------------------
# descriptive performance
# --------------------------------------------------------------------------
def ann_return(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Geometric (compound) annual growth rate -- not the arithmetic mean.

    The difference is volatility drag, and it is the gap between a backtest and
    a brokerage statement.
    """
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    total = float((1 + r).prod())
    if total <= 0:
        return -1.0
    return total ** (periods / len(r)) - 1


def ann_vol(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(r.dropna().std(ddof=1) * math.sqrt(periods))


def sharpe(r: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = r.dropna()
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float((r.mean() - rf / periods) / sd * math.sqrt(periods))


def max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough loss, as a negative fraction."""
    eq = (1 + r.dropna()).cumprod()
    if eq.empty:
        return float("nan")
    return float((eq / eq.cummax() - 1).min())


def calmar(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(r)
    if not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return ann_return(r, periods) / abs(mdd)


def summarise(result, periods: int = TRADING_DAYS) -> dict:
    """One-line performance dict for a BacktestResult (or a bare return Series)."""
    net = getattr(result, "net_returns", result)
    out = {
        "ann_return": ann_return(net, periods),
        "ann_vol": ann_vol(net, periods),
        "sharpe": sharpe(net, periods=periods),
        "max_drawdown": max_drawdown(net),
        "calmar": calmar(net, periods),
        "hit_rate": float((net > 0).mean()),
        "n_days": int(len(net)),
    }
    if hasattr(result, "turnover"):
        out["ann_turnover"] = float(result.turnover.mean() * periods)
        out["ann_cost_drag"] = float(result.costs.mean() * periods)
        out["sharpe_gross"] = sharpe(result.gross_returns, periods=periods)
    return out


# --------------------------------------------------------------------------
# chance-correction
# --------------------------------------------------------------------------
def probabilistic_sharpe_ratio(
    r: pd.Series, sr_benchmark: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    """P(true Sharpe > ``sr_benchmark``), adjusting for skew and kurtosis.

    ``sr_benchmark`` is annualised, as is the observed Sharpe. Returns a
    probability: 0.95 is the conventional bar.

    Non-normality matters here. Strategies that sell tails (short vol, carry,
    mean reversion) post high Sharpes with left-skewed, fat-tailed returns, and
    this correction discounts exactly those.
    """
    r = r.dropna()
    T = len(r)
    if T < 3:
        return float("nan")
    sr_hat = sharpe(r, periods=periods) / math.sqrt(periods)  # per-period
    sr_star = sr_benchmark / math.sqrt(periods)
    g3 = float(sps.skew(r, bias=False))
    g4 = float(sps.kurtosis(r, fisher=False, bias=False))  # non-excess

    denom = 1.0 - g3 * sr_hat + 0.25 * (g4 - 1.0) * sr_hat**2
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    z = (sr_hat - sr_star) * math.sqrt(T - 1) / math.sqrt(denom)
    return float(sps.norm.cdf(z))


def expected_max_sharpe(
    n_trials: int, sr_variance: float, periods: int = TRADING_DAYS
) -> float:
    """Annualised Sharpe the *best of* ``n_trials`` null strategies reaches by luck.

    ``sr_variance`` is the variance of the annualised Sharpes actually observed
    across the trials -- the spread of your own sweep tells you how much luck
    was available to harvest.
    """
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    n = float(n_trials)
    q1 = sps.norm.ppf(1.0 - 1.0 / n)
    q2 = sps.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(sr_variance) * ((1 - _EULER) * q1 + _EULER * q2))


def deflated_sharpe_ratio(
    r: pd.Series,
    all_trial_sharpes,
    periods: int = TRADING_DAYS,
) -> dict:
    """The honest significance of a swept winner.

    Pass the winner's return series and the annualised Sharpes of *every*
    configuration you tried -- including the ones you discarded. Silently
    dropping failed trials is what makes the raw Sharpe meaningless in the
    first place.

    Returns ``sr_observed``, the luck threshold ``sr_expected_max``, and
    ``dsr`` = P(skill | you ran this many trials). Below ~0.95, the result is
    consistent with a sweep over noise.
    """
    trials = np.asarray([s for s in np.asarray(all_trial_sharpes, dtype=float)
                         if np.isfinite(s)], dtype=float)
    n_trials = len(trials)
    sr_var = float(trials.var(ddof=1)) if n_trials > 1 else 0.0
    sr_star = expected_max_sharpe(n_trials, sr_var, periods)
    return {
        "sr_observed": sharpe(r, periods=periods),
        "n_trials": n_trials,
        "sr_trial_dispersion": math.sqrt(sr_var) if sr_var > 0 else 0.0,
        "sr_expected_max": sr_star,
        "dsr": probabilistic_sharpe_ratio(r, sr_star, periods),
        "psr_vs_zero": probabilistic_sharpe_ratio(r, 0.0, periods),
    }


def stationary_bootstrap_pvalue(
    r: pd.Series,
    n_boot: int = 5000,
    mean_block: int = 10,
    seed: int = 0,
) -> dict:
    """One-sided p-value for H0: mean return <= 0, under serial dependence.

    Politis-Romano stationary bootstrap: geometric block lengths, so volatility
    clustering and autocorrelation survive resampling. An iid bootstrap would
    shred that structure and report a p-value that is too small.
    """
    x = r.dropna().to_numpy(dtype=float)
    T = len(x)
    if T < 30:
        return {"p_value": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    p_new = 1.0 / mean_block
    centred = x - x.mean()  # impose H0

    idx = np.empty((n_boot, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, n_boot)
    jump = rng.random((n_boot, T)) < p_new
    fresh = rng.integers(0, T, (n_boot, T))
    for t in range(1, T):
        cont = (idx[:, t - 1] + 1) % T
        idx[:, t] = np.where(jump[:, t], fresh[:, t], cont)

    boot_means = centred[idx].mean(axis=1)
    obs = x.mean()
    return {
        "observed_mean": float(obs),
        "p_value": float((boot_means >= obs).mean()),
        "n_boot": n_boot,
        "mean_block": mean_block,
    }


def probability_of_backtest_overfitting(
    trial_returns: pd.DataFrame,
    n_blocks: int = 14,
    max_splits: int = 4000,
    seed: int = 0,
) -> dict:
    """PBO via combinatorially symmetric cross-validation.

    ``trial_returns``: one column per configuration tried, rows aligned in time.

    Chop time into ``n_blocks`` blocks; for every way of splitting them into
    half in-sample / half out-of-sample, pick the best config in-sample and
    record where it ranks out-of-sample. PBO is the fraction of splits where
    the in-sample champion lands in the bottom half out-of-sample.

    PBO near 0.5 means your selection procedure carries no information -- the
    winner is a coin flip. This is the portfolio analogue of checking that a
    layer chosen in-fold still wins out-of-fold.
    """
    df = trial_returns.dropna(how="all")
    if df.shape[1] < 2:
        return {"pbo": float("nan"), "n_splits": 0, "n_trials": int(df.shape[1])}
    if n_blocks % 2:
        n_blocks -= 1

    blocks = np.array_split(np.arange(len(df)), n_blocks)
    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    rng = np.random.default_rng(seed)
    if len(combos) > max_splits:
        combos = [combos[i] for i in rng.choice(len(combos), max_splits, replace=False)]

    vals = df.to_numpy(dtype=float)
    n_cfg = vals.shape[1]
    logits, below = [], 0
    for c in combos:
        is_rows = np.concatenate([blocks[i] for i in c])
        oos_rows = np.concatenate([blocks[i] for i in range(n_blocks) if i not in c])

        def _sr(rows):
            sub = vals[rows]
            sd = np.nanstd(sub, axis=0, ddof=1)
            sd = np.where((sd == 0) | ~np.isfinite(sd), np.nan, sd)
            return np.nanmean(sub, axis=0) / sd

        sr_is, sr_oos = _sr(is_rows), _sr(oos_rows)
        if not np.isfinite(sr_is).any() or not np.isfinite(sr_oos).any():
            continue
        best = int(np.nanargmax(sr_is))
        finite = np.isfinite(sr_oos)
        # relative rank of the IS winner among OOS results, in (0, 1)
        rank = (sr_oos[finite] <= sr_oos[best]).sum() / (finite.sum() + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1 - rank)))
        below += rank < 0.5

    n = len(logits)
    return {
        "pbo": below / n if n else float("nan"),
        "median_logit": float(np.median(logits)) if n else float("nan"),
        "n_splits": n,
        "n_trials": n_cfg,
    }


def bootstrap_sharpe_difference(
    r_strategy: pd.Series,
    r_benchmark: pd.Series,
    n_boot: int = 3000,
    mean_block: int = 10,
    seed: int = 0,
) -> dict:
    """Paired block bootstrap on the DIFFERENCE IN SHARPE, not in mean return.

    Testing the mean of ``strategy - benchmark`` answers the wrong question for
    anything risk-based. A 60/40 mix or an inverse-vol book can carry the same
    mean return at materially lower volatility: its Sharpe rises while its mean
    excess return sits near zero, so a mean-difference test reports p ~ 0.5 and
    you would wrongly conclude nothing happened.

    Days are resampled in blocks *jointly* for both series, preserving both the
    autocorrelation within each and the contemporaneous correlation between
    them -- the pairing is what makes this a like-for-like comparison.

    ``p_value`` is one-sided for H0: delta Sharpe <= 0.
    """
    joined = pd.concat([r_strategy, r_benchmark], axis=1).dropna()
    if len(joined) < 60:
        return {"delta_sharpe": float("nan"), "p_value": float("nan"), "n_boot": 0}
    a = joined.iloc[:, 0].to_numpy(dtype=float)
    b = joined.iloc[:, 1].to_numpy(dtype=float)
    T = len(a)

    rng = np.random.default_rng(seed)
    idx = np.empty((n_boot, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, n_boot)
    jump = rng.random((n_boot, T)) < (1.0 / mean_block)
    fresh = rng.integers(0, T, (n_boot, T))
    for t in range(1, T):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % T)

    def _sr(x):
        sd = x.std(axis=1, ddof=1)
        sd = np.where((sd == 0) | ~np.isfinite(sd), np.nan, sd)
        return x.mean(axis=1) / sd * math.sqrt(TRADING_DAYS)

    deltas = _sr(a[idx]) - _sr(b[idx])
    deltas = deltas[np.isfinite(deltas)]
    if not len(deltas):
        return {"delta_sharpe": float("nan"), "p_value": float("nan"), "n_boot": 0}
    obs = sharpe(pd.Series(a)) - sharpe(pd.Series(b))
    return {
        "delta_sharpe": float(obs),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_value": float((deltas <= 0).mean()),
        "n_boot": int(len(deltas)),
    }
