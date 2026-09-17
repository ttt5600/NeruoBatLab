"""Tests for the order layer. The band and the guardrails are the point.

The band is not a nicety: the horse race showed a strategy with a real gross
edge (Sharpe 0.565) turned into a loser (-0.021) purely by rebalancing too
often. ``test_band_cuts_turnover_materially`` measures that the band does what
it was added to do, rather than assuming it.
"""
from __future__ import annotations

import numpy as np
import pytest

from quant.backtest import CostModel
from quant.execution import (
    Account,
    GuardrailError,
    PaperBroker,
    RebalancePolicy,
    execute,
    plan_rebalance,
)

PX = {"SPY": 100.0, "TLT": 50.0, "GLD": 200.0}


def _acct(cash, **pos):
    return Account(cash=cash, positions=dict(pos))


# --------------------------------------------------------------------------
# the band
# --------------------------------------------------------------------------
def test_small_drift_produces_no_orders():
    """A position 1pp off target, with a 2pp band, must not trade."""
    acct = _acct(4_100.0, SPY=59.0)  # 5900 of 10000 = 0.59 vs target 0.60
    orders = plan_rebalance(acct, {"SPY": 0.60}, PX, RebalancePolicy(band=0.02))
    assert orders == []


def test_large_drift_produces_an_order_that_closes_the_gap():
    acct = _acct(6_000.0, SPY=40.0)  # 0.40 vs target 0.60
    orders = plan_rebalance(acct, {"SPY": 0.60}, PX, RebalancePolicy(band=0.02))
    assert len(orders) == 1
    o = orders[0]
    assert o.side == "buy"
    assert o.est_notional == pytest.approx(0.20 * 10_000, rel=1e-9)


def test_exit_is_never_suppressed_by_the_band():
    """Dropping to flat is a risk decision, not a rebalance -- it always trades."""
    acct = _acct(9_900.0, SPY=1.0)  # weight 0.01, far inside a 0.02 band
    orders = plan_rebalance(acct, {}, PX, RebalancePolicy(band=0.02, min_notional=50.0))
    assert len(orders) == 1
    assert orders[0].side == "sell" and orders[0].reason == "exit"


def test_dust_below_min_notional_is_skipped():
    acct = _acct(10_000.0)
    orders = plan_rebalance(acct, {"SPY": 0.003}, PX,
                            RebalancePolicy(band=0.0, min_notional=50.0))
    assert orders == []  # 0.003 * 10_000 = $30 < $50


def test_order_notional_is_capped():
    acct = _acct(1_000_000.0)
    orders = plan_rebalance(acct, {"SPY": 1.0}, PX,
                            RebalancePolicy(max_order_notional=25_000.0))
    assert orders[0].est_notional == pytest.approx(25_000.0)


def test_band_cuts_turnover_materially():
    """Drive a drifting book with band=0 and band=0.02; compare traded notional.

    This is the meanrev_5d lesson as an executable check: same target, same
    prices, and the only difference is how eagerly we chase the target.
    """
    target = {"SPY": 0.4, "TLT": 0.3, "GLD": 0.3}

    def traded(band):
        # Seed INSIDE, so both arms walk the identical price path. Comparing two
        # policies on different random draws measures the draws, not the policy.
        rng = np.random.default_rng(0)
        broker = PaperBroker(cash=100_000.0, costs=CostModel(), marks=dict(PX))
        total, count = 0.0, 0
        marks = dict(PX)
        for _ in range(250):
            for s in marks:
                marks[s] *= float(np.exp(rng.normal(0, 0.01)))
            broker.set_marks(marks)
            placed = execute(broker, target, RebalancePolicy(band=band), dry_run=False)
            total += sum(o.est_notional for o in placed)
            count += len(placed)
        return total, count

    eager_notional, eager_n = traded(0.0)
    banded_notional, banded_n = traded(0.02)

    assert banded_notional < eager_notional * 0.5, (
        f"band should halve traded notional: {banded_notional:,.0f} vs {eager_notional:,.0f}")
    # The order count collapses far harder than the notional: the band removes
    # the constant stream of tiny corrections while still making the few large
    # adjustments that actually matter.
    assert banded_n < eager_n * 0.1, f"order count: {banded_n} vs {eager_n}"


# --------------------------------------------------------------------------
# guardrails reject rather than clamp
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_weight_is_refused(bad):
    with pytest.raises(GuardrailError, match="target weight"):
        plan_rebalance(_acct(10_000.0), {"SPY": bad}, PX, RebalancePolicy())


def test_short_is_refused_unless_enabled():
    with pytest.raises(GuardrailError, match="allow_short"):
        plan_rebalance(_acct(10_000.0), {"SPY": -0.3}, PX, RebalancePolicy())
    ok = plan_rebalance(_acct(10_000.0), {"SPY": -0.3}, PX,
                        RebalancePolicy(allow_short=True, max_gross_leverage=1.0))
    assert ok[0].side == "sell"


def test_over_leverage_is_refused():
    with pytest.raises(GuardrailError, match="gross exposure"):
        plan_rebalance(_acct(10_000.0), {"SPY": 0.8, "TLT": 0.8}, PX, RebalancePolicy())


@pytest.mark.parametrize("px", [0.0, -5.0, float("nan")])
def test_bad_price_is_refused(px):
    with pytest.raises(GuardrailError, match="price is"):
        plan_rebalance(_acct(10_000.0), {"SPY": 0.5}, {"SPY": px}, RebalancePolicy())


def test_missing_price_for_a_held_position_is_refused():
    with pytest.raises(GuardrailError, match="no price available"):
        plan_rebalance(_acct(5_000.0, GLD=10.0), {"SPY": 0.5},
                       {"SPY": 100.0}, RebalancePolicy())


def test_zero_equity_is_refused():
    with pytest.raises(GuardrailError, match="equity"):
        plan_rebalance(_acct(0.0), {"SPY": 0.5}, PX, RebalancePolicy())


# --------------------------------------------------------------------------
# execution semantics
# --------------------------------------------------------------------------
def test_execute_is_dry_run_by_default():
    broker = PaperBroker(cash=10_000.0, marks=dict(PX))
    orders = execute(broker, {"SPY": 0.5})
    assert len(orders) == 1
    assert broker.fills == [], "dry_run must not place anything"
    assert broker.account().positions == {}


def test_execute_places_orders_when_explicitly_enabled():
    broker = PaperBroker(cash=10_000.0, marks=dict(PX))
    execute(broker, {"SPY": 0.5}, dry_run=False)
    assert broker.account().positions["SPY"] == pytest.approx(50.0)


def test_paper_broker_charges_costs_and_loses_value_to_them():
    broker = PaperBroker(cash=10_000.0, costs=CostModel(), marks=dict(PX))
    before = broker.account().equity(PX)
    execute(broker, {"SPY": 0.5}, dry_run=False)
    after = broker.account().equity(PX)
    assert broker.total_costs > 0
    assert after == pytest.approx(before - broker.total_costs, rel=1e-9)


def test_rebalance_converges_to_target_within_the_band():
    broker = PaperBroker(cash=100_000.0, costs=CostModel(), marks=dict(PX))
    target = {"SPY": 0.5, "TLT": 0.25, "GLD": 0.25}
    for _ in range(6):
        execute(broker, target, RebalancePolicy(band=0.02), dry_run=False)
    acct = broker.account()
    w = acct.weights(PX)
    for sym, tgt in target.items():
        assert abs(w.get(sym, 0.0) - tgt) < 0.03, (sym, w)
    assert plan_rebalance(acct, target, PX, RebalancePolicy(band=0.02)) == []


def test_selling_reduces_position_and_restores_cash():
    broker = PaperBroker(cash=10_000.0, costs=CostModel(0, 0, 0, 0, 0), marks=dict(PX))
    execute(broker, {"SPY": 1.0}, RebalancePolicy(max_order_notional=1e9), dry_run=False)
    assert broker.account().positions["SPY"] == pytest.approx(100.0)
    execute(broker, {}, RebalancePolicy(max_order_notional=1e9), dry_run=False)
    acct = broker.account()
    assert acct.positions == {}
    assert acct.cash == pytest.approx(10_000.0, rel=1e-9)
