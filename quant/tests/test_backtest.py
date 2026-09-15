"""Tests that the harness cannot quietly lie.

Three things are worth testing here and the rest is bookkeeping:

1. the engine cannot leak the future into past P&L,
2. every shipped strategy is causal,
3. the overfitting metrics give the KNOWN answer on synthetic nulls.

(3) follows the rule that a metric has to be validated on a regime whose truth
you control before it is allowed to rank anything real. PBO and DSR are meant
to flag noise-fitting, so they are tested against deliberate noise first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest import FRICTIONLESS, CostModel, run_backtest
from quant.data import synthetic_prices
from quant.stats import (
    ann_return,
    bootstrap_sharpe_difference,
    deflated_sharpe_ratio,
    max_drawdown,
    probability_of_backtest_overfitting,
    sharpe,
    stationary_bootstrap_pvalue,
)
from quant.strategies import REGISTRY, buy_and_hold, sma_crossover


@pytest.fixture(scope="module")
def px():
    return synthetic_prices(n_days=1500, n_assets=4, seed=7)


# --------------------------------------------------------------------------
# 1. the engine cannot see the future
# --------------------------------------------------------------------------
def test_future_price_change_cannot_move_past_pnl(px):
    """Perturb one price deep in the sample; every P&L before it must be identical.

    This is the property that matters. If it ever fails, some rolling window or
    normalisation is reaching forward and every result in the repo is void.
    """
    w = sma_crossover(px, fast=20, slow=60)
    base = run_backtest(px, w).net_returns

    cut = 1000
    px2 = px.copy()
    px2.iloc[cut:] *= 1.35  # a large, obvious future shock
    w2 = sma_crossover(px2, fast=20, slow=60)
    pert = run_backtest(px2, w2).net_returns

    n = min(cut - 1, len(base), len(pert))
    np.testing.assert_allclose(base.iloc[:n], pert.iloc[:n], rtol=1e-12, atol=1e-14)


def test_execution_lag_defeats_the_classic_close_to_close_leak(px):
    """Trading on the return you just observed is free money -- unless lagged.

    This is the single most common backtest bug: compute a signal from today's
    close, then book today's return on it. With ``execution_lag=1`` the engine
    holds yesterday's decision, and the edge evaporates.
    """
    sig = np.sign(px.pct_change()).fillna(0.0) / px.shape[1]

    leaked = sharpe(run_backtest(px, sig, costs=FRICTIONLESS, execution_lag=0).net_returns)
    honest = sharpe(run_backtest(px, sig, costs=FRICTIONLESS, execution_lag=1).net_returns)

    assert leaked > 10, f"lag=0 should expose the leak, got Sharpe {leaked:.2f}"
    assert honest < 1, f"lag=1 should absorb it, got Sharpe {honest:.2f}"


def test_lag_cannot_save_you_from_a_strategy_that_reads_the_future(px):
    """Documents the limit of the engine's guarantee -- deliberately, not by accident.

    A signal defined as ``sign(return[t+1])`` is shifted straight back into
    alignment by the lag and prints a Sharpe of ~42. The engine enforces WHEN a
    decision is executed; it cannot know that the decision was made from data
    that did not exist yet. Only ``test_every_strategy_is_causal`` catches that,
    which is why that test is parametrised over the whole registry.
    """
    peek = np.sign(px.pct_change().shift(-1)).fillna(0.0) / px.shape[1]
    assert sharpe(run_backtest(px, peek, costs=FRICTIONLESS, execution_lag=1).net_returns) > 10


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_strategy_is_causal(name, px):
    """Weights on a truncated history must equal weights on the full history.

    Catches any centred window, full-sample z-score, or global normalisation --
    the ways lookahead usually arrives.
    """
    fn = REGISTRY[name]
    cut = 900
    full = fn(px).iloc[:cut]
    trunc = fn(px.iloc[:cut])
    common = full.index.intersection(trunc.index)
    assert len(common) > 100
    pd.testing.assert_frame_equal(
        full.loc[common], trunc.loc[common], check_exact=False, atol=1e-12
    )


# --------------------------------------------------------------------------
# 2. accounting is right
# --------------------------------------------------------------------------
def test_buy_and_hold_frictionless_equals_asset_return(px):
    one = px[["SYN0"]]
    res = run_backtest(one, buy_and_hold(one), costs=FRICTIONLESS)
    expected = one["SYN0"].pct_change().reindex(res.net_returns.index)
    np.testing.assert_allclose(res.net_returns, expected, rtol=1e-12)
    assert abs(ann_return(res.net_returns) - ann_return(expected)) < 1e-12


def test_costs_scale_linearly_with_turnover(px):
    """Doubling the fee must exactly double the cost drag, nothing else."""
    one = px[["SYN0"]]
    flip = pd.DataFrame(
        {"SYN0": np.where(np.arange(len(one)) % 2 == 0, 1.0, 0.0)}, index=one.index
    )
    c1 = run_backtest(one, flip, costs=CostModel(1.0, 0.0, 0.0, 0.0, 0.0))
    c2 = run_backtest(one, flip, costs=CostModel(2.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(c2.costs, 2 * c1.costs, rtol=1e-12)
    np.testing.assert_allclose(c1.costs, c1.turnover * 1.0 / 1e4, rtol=1e-12)
    assert c1.costs.sum() > 0


def test_constant_weights_still_cost_money_to_maintain(px):
    """Drift rebalancing is real turnover; a 2-asset constant mix is not free."""
    two = px[["SYN0", "SYN1"]]
    res = run_backtest(two, buy_and_hold(two), costs=CostModel())
    assert res.turnover.iloc[1:].sum() > 0
    assert res.net_returns.sum() < res.gross_returns.sum()


def test_gross_leverage_is_capped(px):
    big = pd.DataFrame(5.0, index=px.index, columns=px.columns)
    res = run_backtest(px, big, max_gross_leverage=2.0)
    assert res.weights_held.abs().sum(axis=1).max() <= 2.0 + 1e-9


def test_max_drawdown_known_answer():
    r = pd.Series([0.5, -0.5, -0.5, 1.0])  # 1 -> 1.5 -> .75 -> .375 -> .75
    assert abs(max_drawdown(r) - (0.375 / 1.5 - 1)) < 1e-12


# --------------------------------------------------------------------------
# 3. the overfitting metrics work on nulls whose truth we control
# --------------------------------------------------------------------------
def test_pbo_is_near_half_on_pure_noise():
    """200 coin-flip strategies: selection carries no information, so PBO ~ 0.5.

    If PBO came back low here the metric would be endorsing noise.
    """
    rng = np.random.default_rng(0)
    trials = pd.DataFrame(rng.normal(0, 0.01, (1200, 200)))
    out = probability_of_backtest_overfitting(trials, n_blocks=12, max_splits=600)
    assert 0.35 < out["pbo"] < 0.65, out


def test_pbo_is_low_when_one_strategy_is_genuinely_better():
    rng = np.random.default_rng(1)
    trials = pd.DataFrame(rng.normal(0, 0.01, (1200, 40)))
    trials[0] = rng.normal(0.0016, 0.01, 1200)  # a real, persistent edge
    out = probability_of_backtest_overfitting(trials, n_blocks=12, max_splits=600)
    assert out["pbo"] < 0.2, out


def test_deflated_sharpe_rejects_the_winner_of_a_noise_sweep():
    """Best of 500 zero-mean strategies posts Sharpe ~1.6. DSR must not certify it.

    The expected DSR here is ~0.5, not ~0. That is the point of the statistic:
    the winner of a noise sweep should land *at* the luck threshold, since the
    threshold is precisely the median of the best-of-N null distribution. What
    convicts the sweep is the gap -- naive PSR reads >0.99 while DSR sits at a
    coin flip, and 0.5 is nowhere near the 0.95 bar you would need to trade it.
    """
    rng = np.random.default_rng(3)
    trials = pd.DataFrame(rng.normal(0, 0.01, (1000, 500)))
    sharpes = trials.apply(sharpe)
    winner = trials[sharpes.idxmax()]

    assert sharpes.max() > 0.4, "sweep should surface a flattering Sharpe"
    out = deflated_sharpe_ratio(winner, sharpes.values)
    assert out["psr_vs_zero"] > 0.95, "naive test is fooled, as expected"
    assert 0.2 < out["dsr"] < 0.8, f"DSR should sit near the luck threshold: {out}"
    assert out["dsr"] < 0.95, "DSR must never clear the significance bar on noise"
    assert out["psr_vs_zero"] - out["dsr"] > 0.4, "deflation must bite"


def test_deflated_sharpe_accepts_a_real_edge():
    rng = np.random.default_rng(4)
    trials = pd.DataFrame(rng.normal(0, 0.01, (2500, 50)))
    trials[0] = rng.normal(0.0022, 0.01, 2500)  # Sharpe ~3.5 annualised
    sharpes = trials.apply(sharpe)
    out = deflated_sharpe_ratio(trials[sharpes.idxmax()], sharpes.values)
    assert out["dsr"] > 0.95, out


def test_bootstrap_pvalue_is_uniformish_under_the_null():
    rng = np.random.default_rng(5)
    ps = [
        stationary_bootstrap_pvalue(
            pd.Series(rng.normal(0, 0.01, 800)), n_boot=400, seed=i
        )["p_value"]
        for i in range(30)
    ]
    assert 0.25 < np.mean(ps) < 0.75, np.mean(ps)
    assert np.mean(np.array(ps) < 0.05) < 0.2


def test_bootstrap_pvalue_detects_a_real_mean():
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0.0015, 0.01, 1500))
    assert stationary_bootstrap_pvalue(r, n_boot=2000)["p_value"] < 0.01


def test_sharpe_difference_test_sees_what_the_mean_test_misses():
    """Same mean return, less volatility: a real Sharpe gain with zero mean excess.

    This is the regime the Sharpe-difference bootstrap exists for, and it is
    exactly what risk-based allocation does. A mean-difference test is blind
    here by construction -- the excess return really is ~0 -- so it lands near
    p=0.5 while the Sharpe test correctly rejects.
    """
    rng = np.random.default_rng(11)
    bench = pd.Series(rng.normal(0.0004, 0.012, 3000))
    strat = 0.5 * bench + 0.0002  # same mean, half the vol -- i.e. vol targeting

    out = bootstrap_sharpe_difference(strat, bench, n_boot=2000)
    assert out["delta_sharpe"] > 0.3, out
    assert out["p_value"] < 0.05, f"Sharpe test should reject: {out}"
    assert out["ci_low"] > 0, out

    p_mean = stationary_bootstrap_pvalue((strat - bench).dropna(), n_boot=2000)["p_value"]
    assert p_mean > 0.2, f"mean test should be blind here, got {p_mean}"


def test_sharpe_difference_test_is_honestly_underpowered_without_pairing():
    """The same effect, but on an INDEPENDENT series, must NOT reach significance.

    Pairing is where the power comes from. Against a benchmark it is correlated
    with, the bootstrap CI on a +0.52 Sharpe gain is [+0.51, +0.54]; strip the
    correlation and the identical point estimate widens to [-0.26, +1.38]. A
    test that still rejected here would be manufacturing confidence out of the
    resampling scheme rather than the data.
    """
    rng = np.random.default_rng(11)
    bench = pd.Series(rng.normal(0.0004, 0.012, 3000))
    indep = pd.Series(rng.normal(0.0004, 0.006, 3000))

    out = bootstrap_sharpe_difference(indep, bench, n_boot=2000)
    assert out["delta_sharpe"] > 0.3, "same point estimate as the paired case"
    assert out["ci_low"] < 0 < out["ci_high"], f"CI must span zero: {out}"
    assert out["p_value"] > 0.05, f"must not reject without pairing: {out}"


def test_sharpe_difference_test_is_calibrated_on_identical_processes():
    rng = np.random.default_rng(12)
    a = pd.Series(rng.normal(0.0004, 0.01, 2000))
    b = pd.Series(rng.normal(0.0004, 0.01, 2000))
    out = bootstrap_sharpe_difference(a, b, n_boot=2000)
    assert out["ci_low"] < 0 < out["ci_high"], out
    assert 0.05 < out["p_value"] < 0.95, out


# --------------------------------------------------------------------------
# 4. regression: the cache must be keyed by window, not just by symbol
# --------------------------------------------------------------------------
def test_cache_is_not_poisoned_by_an_earlier_narrower_request(tmp_path, monkeypatch):
    """A short early fetch must not truncate a later request for more history.

    This shipped as a live bug: the per-symbol cache ignored the date range, so
    a smoke test fetching SPY from 2015 silently capped every later 2007 query,
    and dropna() then cut the whole panel by eight years -- with no error. The
    only symptom was a suspiciously short backtest.
    """
    from quant import data as D

    monkeypatch.setattr(D, "CACHE", tmp_path)
    calls = []
    real = D._fetch_one

    def spy(sym, start, end, **kw):
        calls.append((sym, start))
        idx = pd.bdate_range(start, "2020-01-01")
        return pd.Series(np.linspace(100, 200, len(idx)), index=idx, name=sym)

    monkeypatch.setattr(D, "_fetch_one", spy)

    narrow = D.load_prices(["FAKE"], start="2015-01-01", end="2020-01-01")
    wide = D.load_prices(["FAKE"], start="2007-01-01", end="2020-01-01")

    assert wide.index[0] < pd.Timestamp("2008-01-01"), (
        f"cache returned truncated history: starts {wide.index[0].date()}"
    )
    assert len(wide) > len(narrow)

    # One fetch, not two: the fix works by always pulling maximal history, so a
    # cached file is a superset of any later request and no refetch is needed.
    assert len(calls) == 1, calls
    assert calls[0][1] == "1990-01-01", f"cache must store maximal history, got {calls[0]}"

    # A cache written with a narrow window (older format, or a hand-edited file)
    # must still be detected as stale and refetched.
    (tmp_path / "FAKE.meta.json").write_text('{"fetched_from": "2015-01-01"}')
    again = D.load_prices(["FAKE"], start="2007-01-01", end="2020-01-01")
    assert len(calls) == 2, "a narrow cached window must trigger a refetch"
    assert again.index[0] < pd.Timestamp("2008-01-01")
    assert real is not None
