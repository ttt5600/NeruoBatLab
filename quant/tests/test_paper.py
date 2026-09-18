"""Tests for the forward paper journal.

Idempotency is the one that matters operationally: a cron that retries, or a
human running `step` twice, must not double-trade. Everything else guards the
journal against silent corruption.
"""
from __future__ import annotations

import numpy as np
import pytest

from quant.backtest import CostModel
from quant.data import synthetic_prices
from quant.execution import GuardrailError, RebalancePolicy
from quant.paper import PaperAccount, backfill, divergence, step
from quant.strategies import inverse_vol

COSTS = CostModel()


def build(px):
    return inverse_vol(px, lookback=20)


@pytest.fixture
def px():
    return synthetic_prices(n_days=300, n_assets=4, seed=3)


@pytest.fixture
def acct():
    return PaperAccount.create("inverse_vol", 50_000.0, RebalancePolicy(band=0.02))


def test_step_is_idempotent_on_the_bar_date(acct, px):
    first = step(acct, px, build, COSTS)
    assert first is not None and first.n_orders > 0
    assert step(acct, px, build, COSTS) is None, "same bar must not trade twice"
    assert len(acct.entries) == 1

    cash, pos = acct.cash, dict(acct.positions)
    step(acct, px, build, COSTS)
    assert acct.cash == cash and acct.positions == pos


def test_force_overrides_idempotency(acct, px):
    step(acct, px, build, COSTS)
    assert step(acct, px, build, COSTS, force=True) is not None
    assert len(acct.entries) == 2


def test_state_round_trips_through_json(tmp_path, acct, px):
    backfill(acct, px, build, COSTS, start=px.index[250].strftime("%Y-%m-%d"))
    p = tmp_path / "acct.json"
    acct.save(p)
    back = PaperAccount.load(p)

    assert back.strategy == acct.strategy
    assert back.cash == pytest.approx(acct.cash)
    assert back.positions == pytest.approx(acct.positions)
    assert len(back.entries) == len(acct.entries)
    assert back.equity_curve().equals(acct.equity_curve())


def test_save_is_atomic_and_leaves_no_partial_file(tmp_path, acct, px):
    step(acct, px, build, COSTS)
    p = tmp_path / "acct.json"
    acct.save(p)
    acct.save(p)  # overwrite must not corrupt
    assert PaperAccount.load(p).strategy == "inverse_vol"
    assert not list(tmp_path.glob("*.tmp")), "temp file must be renamed away"


def test_backfill_only_ever_sees_past_prices(acct, px):
    """Replay must equal day-by-day stepping -- proof no future data leaked in."""
    n = backfill(acct, px, build, COSTS, start=px.index[280].strftime("%Y-%m-%d"))
    assert n > 5

    manual = PaperAccount.create("inverse_vol", 50_000.0, RebalancePolicy(band=0.02))
    for i in range(280, len(px)):
        step(manual, px.iloc[: i + 1], build, COSTS)

    assert manual.cash == pytest.approx(acct.cash)
    np.testing.assert_allclose(
        manual.equity_curve().to_numpy(), acct.equity_curve().to_numpy(), rtol=1e-12
    )


def test_band_means_most_days_do_not_trade(acct, px):
    """Most days are no-ops. How many depends on how jumpy the weights are.

    This fixture uses a deliberately short 20-day vol lookback on 4 synthetic
    assets, so the target weights themselves are noisy and it trades ~40% of
    days. The real configuration (60-day lookback, 10 ETFs) traded 10 of 135.
    The band is doing its job in both; the rate is a property of the strategy's
    weight stability, not of the band.
    """
    backfill(acct, px, build, COSTS, start=px.index[200].strftime("%Y-%m-%d"))
    traded = sum(1 for e in acct.entries if e.n_orders)
    assert traded < len(acct.entries) * 0.5, f"{traded}/{len(acct.entries)} days traded"


def test_cash_never_goes_negative(acct, px):
    """A long-only cash account must be able to pay for its own trades.

    Without a cash buffer, a gross-1.0 target spends every dollar on positions
    and the cost charge drives cash below zero on each rebalance -- silently
    levering a portfolio that was supposed to be unlevered.
    """
    backfill(acct, px, build, COSTS, start=px.index[200].strftime("%Y-%m-%d"))
    assert acct.cash >= 0, f"cash went negative: {acct.cash:,.2f}"
    for e in acct.entries:
        assert e.cash >= -1e-6, f"{e.bar_date}: cash {e.cash:,.2f}"


def test_divergence_tracks_the_backtest_closely(acct, px):
    backfill(acct, px, build, COSTS, start=px.index[200].strftime("%Y-%m-%d"))
    d = divergence(acct, px, build, COSTS)
    assert d["status"] == "ok"
    assert d["correlation"] > 0.95, d
    assert abs(d["mean_daily_gap_bps"]) < 5.0, d
    # The band is the point: the live path should trade LESS than the backtest.
    assert d["paper_ann_turnover"] < d["backtest_ann_turnover"]


def test_divergence_reports_insufficient_history_rather_than_guessing(acct, px):
    step(acct, px, build, COSTS)
    assert divergence(acct, px, build, COSTS)["status"] == "insufficient history"


def test_bad_mark_is_refused(acct, px):
    bad = px.copy()
    bad.iloc[-1, 0] = -1.0
    with pytest.raises(GuardrailError, match="bad mark"):
        step(acct, bad, build, COSTS)


def test_empty_prices_is_refused(acct, px):
    with pytest.raises(GuardrailError, match="no price data"):
        step(acct, px.iloc[:0], build, COSTS)


def test_backfilled_bars_are_not_counted_as_live_evidence(acct, px):
    """Replay must never satisfy a forward-evidence requirement.

    Backfill replays the period the strategy was selected on. If those bars
    counted toward a six-month out-of-sample bar, anyone could clear it
    instantly by replaying five years -- in-sample data laundered into an
    out-of-sample claim.
    """
    n = backfill(acct, px, build, COSTS, start=px.index[250].strftime("%Y-%m-%d"))
    assert n > 10
    assert len(acct.entries) == n
    assert acct.live_days == 0, "backfilled bars must not count as live"
    assert all(e.source == "backfill" for e in acct.entries)


def test_forward_steps_are_counted_as_live(acct, px):
    backfill(acct, px.iloc[:-1], build, COSTS, start=px.index[280].strftime("%Y-%m-%d"))
    before = acct.live_days
    step(acct, px, build, COSTS)
    assert acct.live_days == before + 1
    assert acct.entries[-1].source == "live"
