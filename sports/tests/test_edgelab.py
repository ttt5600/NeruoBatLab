"""edgelab: fees, PIT checks, rule-5 evaluator, registry (hash chain, amend, new-trial-on-edit), ladder, vault gates,
plus the real-data self-tests (golden E1 / L2, cheaters, a small placebo batch). Sandbox registries only.
Run: cd sports && ../.venv_quant/bin/python -m pytest tests/test_edgelab.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgelab import evaluator as E  # noqa: E402
from edgelab import fees as F  # noqa: E402
from edgelab import ladder, run, schema, selftest, vault  # noqa: E402
from edgelab.registry import Registry, RegistryError  # noqa: E402


# ------------------------------------------------------------------ fees
def test_kalshi_fee_examples():
    F.check_example()
    assert float(F.kalshi_fee_per_contract(0.97, 1)) == 0.01          # 1-lot rounds 0.2c up to a cent
    assert abs(float(F.kalshi_fee_per_contract(0.97, 1, rounding="exact")) - 0.07 * .97 * .03) < 1e-12
    assert float(F.fee_per_contract("kalshi_maker_quadratic", 0.5)) == 0.0
    assert float(F.fee_per_contract("forecastex", 0.9)) == 0.01


# ------------------------------------------------------------------ schema / PIT
def _rows(n=300, edge=0.0, seed=0, days=150):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    day = rng.integers(0, days, n)
    dec = t0 + pd.to_timedelta(day, unit="D")
    price = np.full(n, 0.5)
    out = (rng.random(n) < 0.5 + edge).astype(float)
    return pd.DataFrame({"bet_id": [f"b{i}" for i in range(n)], "cluster": dec.strftime("%Y-%m-%d"),
                         "event_start": dec + pd.Timedelta(hours=2), "decided_at": dec,
                         "inputs_as_of": dec - pd.Timedelta(hours=1), "price": price, "price_source": "book_quote",
                         "price_time": dec, "side": "home", "stake": 1.0, "outcome": out, "close_price": price})


def test_pit_clean_rows_pass():
    schema.enforce(_rows())


@pytest.mark.parametrize("col,delta,check", [
    ("inputs_as_of", pd.Timedelta(hours=3), "inputs_after_decision"),
    ("price_time", pd.Timedelta(minutes=5), "quote_after_decision"),
    ("decided_at", pd.Timedelta(hours=3), "decided_after_event_start"),
])
def test_pit_violations_raise_with_counts(col, delta, check):
    r = _rows()
    r.loc[:9, col] = r.loc[:9, "decided_at"] + delta
    if col == "decided_at":
        r["price_time"] = r[["price_time"]].min(axis=1)
        r["inputs_as_of"] = r["inputs_as_of"]
    with pytest.raises(schema.PITViolation) as e:
        schema.enforce(r)
    assert e.value.counts[check] == 10


def test_missing_timestamp_fails_closed():
    r = _rows()
    r["price_time"] = pd.NaT
    with pytest.raises(schema.PITViolation) as e:
        schema.enforce(r)
    assert e.value.counts["missing_timestamp"] == len(r)


def test_maker_fill_must_follow_post():
    r = _rows()
    r["price_source"] = "maker_tape"
    r.loc[:4, "price_time"] = r.loc[:4, "decided_at"] - pd.Timedelta(minutes=1)
    with pytest.raises(schema.PITViolation) as e:
        schema.enforce(r)
    assert e.value.counts["maker_fill_before_post"] == 5


# ------------------------------------------------------------------ evaluator (rule 5)
def test_evaluator_null_fails_and_edge_passes():
    null = E.score(schema.enforce(_rows(600, 0.0, 1)), bar=0.01, min_n=100)
    assert null["verdict"] == "fail"
    good = E.score(schema.enforce(_rows(3000, 0.06, 2, days=400)), bar=0.01, min_n=100)
    assert good["verdict"] in ("pass", "needs_audit") and all(good["checks"].values())
    assert good["ci"][0] < good["net_roi"] < good["ci"][1]                # observed ROI, not the resample mean


def test_concentration_rule():
    r = _rows(400, 0.0, 3)
    r["outcome"] = 0.0
    r.loc[r["cluster"] == r["cluster"].iloc[0], "outcome"] = 1.0         # all profit from one day
    r.loc[:150, "outcome"] = np.where(r.loc[:150, "cluster"] == r["cluster"].iloc[0], 1.0, 0.0)
    res = E.score(schema.enforce(r), bar=0.5, min_n=10)
    assert res["concentration"]["share"] is None or res["concentration"]["share"] > 0.3
    assert not res["checks"]["c_concentration_ok"]


def test_too_good_guard():
    res = E.score(schema.enforce(_rows(400, 0.25, 4)), bar=0.05, min_n=10)
    assert res["needs_audit"] and res["verdict"] in ("needs_audit", "fail")


def test_rule_may_not_set_its_own_pnl():
    r = _rows()
    r["pnl"] = 5.0
    with pytest.raises(ValueError):
        E.score(schema.enforce(r), bar=0.05, min_n=10)


# ------------------------------------------------------------------ registry + ladder
SPEC = {"id": "t_spec", "family": "fam_test", "hypothesis": "h", "dataset": "kalshi_temp_e1",
        "rule": "edgelab.rules.kalshi_wx:taker", "decision_time": "22:00", "price_model": "taker",
        "split": "vault:kalshi_temp_e1", "min_n": 10, "kill": {"kind": "roi_ci_below_zero", "after_n": 100}}


def test_registry_counts_and_new_trial_on_change(tmp_path):
    reg = Registry(tmp_path / "r.jsonl")
    a = reg.register(SPEC)
    assert reg.register(SPEC) == a                                       # identical -> same trial
    b = reg.register({**SPEC, "min_n": 11})                              # any change -> new trial
    assert a != b and reg.family_counts()["fam_test"] == 2
    reg.register({"id": "old", "family": "fam_test", "hypothesis": "x"}, origin="ledger_import",
                 initial_state="dead", n_trials=5)
    assert reg.family_bar("fam_test") == (0.05 / 7, 7)
    assert reg.verify()["ok"]


def test_registry_hash_chain_detects_edit(tmp_path):
    reg = Registry(tmp_path / "r.jsonl")
    reg.register(SPEC)
    reg.register({**SPEC, "id": "t2"})
    lines = reg.path.read_text().splitlines()
    lines[0] = lines[0].replace("fam_test", "fam_other")
    reg.path.write_text("\n".join(lines) + "\n")
    assert not reg.verify()["ok"]


def test_import_cannot_start_passed(tmp_path):
    reg = Registry(tmp_path / "r.jsonl")
    with pytest.raises(RegistryError):
        reg.register({"id": "x", "family": "f", "hypothesis": "h"}, origin="ledger_import", initial_state="dev_pass")


def test_ladder_user_only_edges():
    with pytest.raises(ladder.LadderError):
        ladder.check("forward_paper", "live_small", "evaluate")
    with pytest.raises(ladder.LadderError):
        ladder.check("forward_paper", "live_small", "user", None)
    with pytest.raises(ladder.LadderError):
        ladder.check("registered", "forward_paper", "evaluate")
    ladder.check("forward_paper", "live_small", "user", ladder.UserApproval.make("jonathan"))
    ladder.check("live", "dead", "retire")


# ------------------------------------------------------------------ real data (sandbox registries)
def test_vault_strips_outcomes_and_logs_access(tmp_path):
    reg = Registry(tmp_path / "r.jsonl")
    tid = reg.register(SPEC)
    frame, hidden, acc = vault.load("kalshi_temp_e1", "dev", tid, reg)
    assert "won" not in frame.columns and "won" in hidden.columns
    assert reg.accesses(tid) and pd.to_datetime(frame["date"]).max() <= pd.Timestamp("2025-06-30")
    with pytest.raises(vault.VaultError):
        vault.load("kalshi_temp_x1", "dev", tid, reg)                     # not its dataset
    with pytest.raises(vault.VaultError):
        vault.load("kalshi_temp_e1", "sealed", tid, reg)                  # no dev pass yet


def test_golden_cases():
    g = selftest.golden(log=lambda *a: None)
    assert g["e1_sealed"]["verdict"] == "pass", g
    assert g["l2_t1"]["verdict"] == "fail", g
    assert abs(g["e1_sealed"]["net_roi"] - 0.01455) < 2e-4               # lockbox: +1.45%, 1,844 bets
    assert g["e1_sealed"]["n"] == 1844


def test_cheaters_refused():
    c = selftest.cheaters(log=lambda *a: None)
    assert c["ok"], c


def test_placebo_small_batch():
    p = selftest.placebo(n_each=4, log=lambda *a: None)
    assert p["false_pass_rate"] <= 0.05
