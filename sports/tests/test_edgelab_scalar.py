"""edgelab SCORING_VERSION 2 (settlement-value scoring of Kalshi `scalar` settlements; notes/ops_edgelab_scalar.md).
Run: cd sports && ../.venv_quant/bin/python -m pytest tests/test_edgelab_scalar.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import edgelab  # noqa: E402
from edgelab import evaluator as E  # noqa: E402
from edgelab import fees as F  # noqa: E402
from edgelab import schema, vault  # noqa: E402

OCFG = {"kind": "result_equals_side", "key": "ticker", "value_col": "sv"}


def _bets(tickers, sides, price=0.70):
    n = len(tickers)
    t0 = pd.Timestamp("2026-08-01", tz="UTC")
    dec = t0 + pd.to_timedelta(np.arange(n), unit="D")
    return pd.DataFrame({"bet_id": [f"{t}|T7|{s}" for t, s in zip(tickers, sides)], "cluster": [f"ev{i}" for i in range(n)],
                         "event_start": dec + pd.Timedelta(days=7), "decided_at": dec, "inputs_as_of": dec,
                         "price": price, "price_source": "taker", "price_time": dec, "side": sides,
                         "stake": 1.0, "contracts": 1.0})


def _hidden(results, values, ocfg=OCFG):
    h = pd.DataFrame({"ticker": list(results), "result": list(results.values()), "sv": values})
    h.attrs = {"split": "dev", "outcome": ocfg}
    return h


def _fee(p):
    return float(F.fee_per_contract("kalshi_taker", np.array([p]), 1, "cent")[0])


def test_flag_is_off_by_default():
    assert edgelab.SCORING_VERSION == 1


def test_v1_legacy_scalar_is_a_loss_for_both_sides():
    b = _bets(["A", "A"], ["yes", "no"])
    h = _hidden({"A": "scalar"}, [0.25])
    r = vault.resolve_outcomes("x", b, h, None, scoring_version=1)
    assert r["outcome"].tolist() == [0.0, 0.0] and "payout" not in r.columns


def test_v2_scalar_pays_settlement_value_politics_025():
    b = _bets(["A", "A"], ["yes", "no"], price=0.70)
    h = _hidden({"A": "scalar"}, [0.25])
    r = schema.enforce(vault.resolve_outcomes("x", b, h, None, scoring_version=2))
    assert np.allclose(r["payout"], [0.25, 0.75]) and np.allclose(r["outcome"], [0.25, 0.75])
    cost, pnl, stake = E.cost_and_pnl(r, {})
    fee = _fee(0.70)
    assert np.allclose(pnl, [0.25 - 0.70 - fee, 0.75 - 0.70 - fee])     # count x (value - price) - fee
    assert np.allclose(stake, 0.70 + fee)


def test_v2_cricket_no_result_050_is_not_a_push():
    # cricket NR: every market settles at $0.50; an underdog YES bought at 0.30 makes +0.20 - fee, not 0 (push)
    b = _bets(["C"], ["yes"], price=0.30)
    h = _hidden({"C": "scalar"}, [0.5])
    r = schema.enforce(vault.resolve_outcomes("x", b, h, None, scoring_version=2))
    _, pnl, _ = E.cost_and_pnl(r, {})
    assert np.isclose(pnl[0], 0.5 - 0.30 - _fee(0.30))
    r1 = schema.enforce(vault.resolve_outcomes("x", b, h, None, scoring_version=1))
    assert np.isclose(E.cost_and_pnl(r1, {})[1][0], -(0.30 + _fee(0.30)))  # v1: scored as a full loss


@pytest.mark.parametrize("contracts", [True, False])
def test_binary_rows_bit_identical_v1_v2(contracts):
    rng = np.random.default_rng(0)
    n = 400
    tk = [f"M{i}" for i in range(n)]
    sides = rng.choice(["yes", "no"], n).tolist()
    res = dict(zip(tk, rng.choice(["yes", "no"], n)))
    b = _bets(tk, sides, price=np.round(rng.uniform(0.05, 0.95, n), 2))
    if not contracts:
        b = b.drop(columns="contracts")
    h = _hidden(res, np.where(np.array(list(res.values())) == "yes", 1.0, 0.0))
    r1 = schema.enforce(vault.resolve_outcomes("x", b, h, None, scoring_version=1))
    r2 = schema.enforce(vault.resolve_outcomes("x", b, h, None, scoring_version=2))
    assert r2["payout"].isna().all()
    p1, p2 = E.cost_and_pnl(r1, {})[1], E.cost_and_pnl(r2, {})[1]
    assert np.array_equal(p1, p2)
    s1, s2 = E.score(r1, 0.01, 50), E.score(r2, 0.01, 50)
    for k in ("net_roi", "ci", "p", "verdict", "half1", "half2", "losses"):
        assert s1[k] == s2[k], k


def test_void_push_unchanged_under_v2():
    b = _bets(["V", "W"], ["yes", "no"], price=0.55)
    b["bet_id"] = ["V|1", "W|1"]
    h = pd.DataFrame({"bet_id": ["V|1", "W|1"], "result": ["void", "no"]})
    h.attrs = {"split": "forward", "outcome": {"kind": "result_side", "key": "bet_id"}}
    r1 = vault.resolve_outcomes("x", b, h, None, scoring_version=1)
    r2 = vault.resolve_outcomes("x", b, h, None, scoring_version=2)
    assert r1["outcome"].tolist() == r2["outcome"].tolist() == [0.5, 1.0]
    assert np.array_equal(E.cost_and_pnl(schema.enforce(r1), {})[1], E.cost_and_pnl(schema.enforce(r2), {})[1])


def test_v2_result_side_scalar():
    b = _bets(["S"], ["no"], price=0.40)
    b["bet_id"] = ["S|k"]
    h = pd.DataFrame({"bet_id": ["S|k"], "result": ["scalar"], "sv": [0.5]})
    h.attrs = {"split": "forward", "outcome": {"kind": "result_side", "key": "bet_id", "value_col": "sv"}}
    r = schema.enforce(vault.resolve_outcomes("x", b, h, None, scoring_version=2))
    assert np.isclose(E.cost_and_pnl(r, {})[1][0], 0.5 - 0.40 - _fee(0.40))


def test_v2_scalar_without_value_fails_loudly():
    b = _bets(["A"], ["yes"])
    h = _hidden({"A": "scalar"}, [np.nan])
    with pytest.raises(vault.VaultError):
        vault.resolve_outcomes("x", b, h, None, scoring_version=2)
    h2 = _hidden({"A": "scalar"}, [0.25], ocfg={"kind": "result_equals_side", "key": "ticker"})  # no value_col
    with pytest.raises(vault.VaultError):
        vault.resolve_outcomes("x", b, h2, None, scoring_version=2)


def test_v2_scalar_needs_yes_no_side():
    b = _bets(["A"], ["home"])
    with pytest.raises(vault.VaultError):
        vault.resolve_outcomes("x", b, _hidden({"A": "scalar"}, [0.5]), None, scoring_version=2)


def test_rule_may_not_set_payout():
    b = _bets(["A"], ["yes"]).assign(payout=1.0)
    with pytest.raises(vault.VaultError):
        vault.resolve_outcomes("x", b, _hidden({"A": "yes"}, [1.0]), None, scoring_version=2)


def test_schema_rejects_fractional_outcome_without_matching_payout():
    b = _bets(["A"], ["yes"]).assign(outcome=0.25)
    with pytest.raises(schema.SchemaError):
        schema.enforce(b)
    with pytest.raises(schema.SchemaError):
        schema.enforce(b.assign(payout=0.30))
    schema.enforce(b.assign(payout=0.25))
