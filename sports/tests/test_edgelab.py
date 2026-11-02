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


# ------------------------------------------------------------------ pre-set forward kill (rule 7) + forward adapters
def _fake_adapter(tmp_path, monkeypatch, outcomes):
    """A forward adapter on a fake dataset: one bet per outcome (1 win / 0 loss), price .9, maker_fill, 1 contract."""
    import types
    src = tmp_path / "fake_adapter.py"
    src.write_text("# fake adapter v1\n")
    mod = types.ModuleType("fake_fwd_adapter")
    mod.__file__ = str(src)
    mod.OUTCOME, mod.HIDDEN = {"kind": "result_side", "key": "bet_id"}, ["result"]
    t0 = pd.Timestamp("2026-11-03 02:00", tz="UTC")

    def load(params):
        n = len(outcomes)
        d = pd.DataFrame({"bet_id": [f"b{i}" for i in range(n)],
                          "date": pd.date_range("2026-11-03", periods=n, freq="D"),
                          "result": ["no" if o == 1 else "yes" for o in outcomes], "settled": True})
        return d, []

    def rows(frame, params):
        n = len(frame)
        dec = t0 + pd.to_timedelta(np.arange(n), unit="D")
        return pd.DataFrame({"bet_id": frame["bet_id"].to_numpy(), "cluster": frame["date"].dt.strftime("%Y-%m-%d"),
                             "event_start": dec + pd.Timedelta(hours=2), "decided_at": dec, "inputs_as_of": dec,
                             "price": 0.9, "price_source": "maker_fill", "price_time": dec + pd.Timedelta(hours=1),
                             "fill_deadline": dec + pd.Timedelta(hours=20), "side": "no", "stake": 0.9,
                             "contracts": 1.0, "close_price": 0.91})
    mod.load, mod.rows = load, rows
    monkeypatch.setitem(sys.modules, "fake_fwd_adapter", mod)
    real = vault.config()
    fake = {**real, "fake_fwd": {"loader": "none", "family": "fam_test", "hidden": [], "outcome": {"kind": "none"},
                                 "splits": {"dev": None, "sealed": None, "forward": ["2026-11-01", None]},
                                 "forward_adapters": {"t_fwd": {"module": "fake_fwd_adapter"}}}}
    monkeypatch.setattr(vault, "config", lambda: fake)
    return src


FWD_SPEC = {**SPEC, "id": "t_fwd", "dataset": "fake_fwd", "rule": "edgelab.rules.forward:unwired",
            "price_model": "maker_tape", "split": "vault:fake_fwd", "min_n": 5,
            "kill": {"kind": "roi_ci_below_zero", "after_n": 100}}


def _at_sealed_pass(reg):
    tid = reg.register(FWD_SPEC)
    ladder.transition(reg, tid, "sealed_pass", "evaluate", reason="test fixture")
    return tid


def test_setkill_refused_after_a_forward_bet_is_scored(tmp_path, monkeypatch):
    _fake_adapter(tmp_path, monkeypatch, [1, 1, 1])
    reg = Registry(tmp_path / "r.jsonl")
    tid = _at_sealed_pass(reg)
    out = run.evaluate(tid, "forward", reg, save_rows=False)
    assert out["n"] == 3 and out["kill"]["source"] == "spec"
    with pytest.raises(RegistryError, match="already has 3 forward bet"):
        reg.set_preset_kill(tid, [{"kind": "losses_in_first", "n": 30, "max_losses": 2}])


def test_setkill_once_and_never_edited(tmp_path, monkeypatch):
    _fake_adapter(tmp_path, monkeypatch, [])
    reg = Registry(tmp_path / "r.jsonl")
    tid = _at_sealed_pass(reg)
    run.evaluate(tid, "forward", reg, save_rows=False)              # n = 0 does not count as scored
    reg.set_preset_kill(tid, [{"kind": "losses_in_first", "n": 30, "max_losses": 2}])
    with pytest.raises(RegistryError, match="already has a pre-set"):
        reg.set_preset_kill(tid, [{"kind": "losses_in_first", "n": 30, "max_losses": 5}])
    with pytest.raises(RegistryError):                                # max_losses must be < n
        reg.set_preset_kill(reg.register({**FWD_SPEC, "min_n": 6}), [{"kind": "losses_in_first", "n": 3, "max_losses": 3}])
    assert reg.verify()["ok"]


def test_setkill_cli_refuses_second(tmp_path, monkeypatch):
    from edgelab import cli
    _fake_adapter(tmp_path, monkeypatch, [])
    p = tmp_path / "r.jsonl"
    _at_sealed_pass(Registry(p))
    args = ["--registry", str(p), "setkill", "t_fwd", "--kind", "losses_in_first", "--n", "50", "--max-losses", "5"]
    assert cli.main(args) == 0
    assert cli.main(args) == 1


def test_preset_kill_triggers_and_kills(tmp_path, monkeypatch):
    _fake_adapter(tmp_path, monkeypatch, [1, 0, 1, 0, 1, 1])          # 2 losses in the first 6
    reg = Registry(tmp_path / "r.jsonl")
    tid = _at_sealed_pass(reg)
    reg.set_preset_kill(tid, [{"kind": "losses_in_first", "n": 10, "max_losses": 1},
                              {"kind": "clv_negative_significant", "after_n": 100, "p": 0.05}])
    out = run.evaluate(tid, "forward", reg, save_rows=False)
    assert out["kill"]["source"] == "preset" and out["kill"]["triggered"] is True
    assert out["kill"]["rules"][0]["losses_in_first"] == 2 and out["kill"]["rules"][1]["triggered"] is False
    assert reg.state(tid) == "dead"
    from edgelab import migrate, report
    monkeypatch.setattr(migrate, "first_looks", lambda: {})
    ev = report.evidence(reg, tid)
    assert ev["preset_kill_triggered"] is True and ev["_preset_kill"]["source"] == "preset"


def test_preset_kill_not_triggered_at_the_limit(tmp_path, monkeypatch):
    _fake_adapter(tmp_path, monkeypatch, [1, 0, 1, 1, 1])             # 1 loss, max 1 -> alive
    reg = Registry(tmp_path / "r.jsonl")
    tid = _at_sealed_pass(reg)
    reg.set_preset_kill(tid, [{"kind": "losses_in_first", "n": 10, "max_losses": 1}])
    out = run.evaluate(tid, "forward", reg, save_rows=False)
    assert out["kill"]["triggered"] is False and reg.state(tid) == "forward_paper"


def test_losses_in_first_counts_only_the_window():
    r = _rows(40, 0.0, 5)
    r["outcome"] = 1.0
    order = np.argsort(r["decided_at"].to_numpy(), kind="stable")
    r.loc[order[-5:], "outcome"] = 0.0                                # 5 losses, all after the first 30
    k = run.kill_check({"kind": "losses_in_first", "n": 30, "max_losses": 2}, {"n": 40}, schema.enforce(r))
    assert k["triggered"] is False and k["rules"][0]["losses_in_first"] == 0


def test_forward_adapter_locked_after_first_scored(tmp_path, monkeypatch):
    src = _fake_adapter(tmp_path, monkeypatch, [1, 1])
    reg = Registry(tmp_path / "r.jsonl")
    tid = _at_sealed_pass(reg)
    run.evaluate(tid, "forward", reg, save_rows=False)
    src.write_text("# fake adapter v2 (edited after seeing forward results)\n")
    with pytest.raises(RegistryError, match="adapter changed"):
        run.evaluate(tid, "forward", reg, save_rows=False)


def test_kill_validation():
    from edgelab.registry import validate_kill
    with pytest.raises(RegistryError):
        validate_kill({"kind": "losses_in_first", "n": 30})                     # missing max_losses
    with pytest.raises(RegistryError):
        validate_kill({"kind": "clv_negative_significant", "after_n": 100, "p": 0.05, "x": 1})
    assert validate_kill({"kind": "roi_ci_below_zero", "after_n": "60"}) == {"kind": "roi_ci_below_zero", "after_n": 60}


def test_strategy_states_matches_resolution(tmp_path):
    reg = Registry(tmp_path / "r.jsonl")
    a = reg.register(SPEC)
    ladder.transition(reg, a, "dead", "retire", reason="t")
    st = reg.strategy_states()
    assert st["t_spec"]["trial_id"] == a and st["t_spec"]["state"] == "dead"
