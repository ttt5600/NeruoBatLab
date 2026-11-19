"""GOAL.md rules 5/6/7 amendment (user 2026-11-18, BAR_VERSION 2): stage bars, concentration, Sharpe guard, forward
sequential tests (placebo / planted-edge sanity), the SPRT kill added to existing kills, needs_audit on H1 acceptance."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgelab import evaluator as E  # noqa: E402
from edgelab import forward_test as FT  # noqa: E402
from edgelab import fees as F  # noqa: E402
from edgelab import run, schema, vault  # noqa: E402
from edgelab.registry import Registry, RegistryError  # noqa: E402

from test_edgelab import FWD_SPEC, SPEC, _at_sealed_pass, _fake_adapter, _rows  # noqa: E402


# ------------------------------------------------------------------ sealed bar .05 / 2^j
def test_sealed_bar_halves_per_shot(tmp_path, monkeypatch):
    real = vault.config()
    fake = {k: v for k, v in real.items()}
    fake["kalshi_temp_e1"] = {**real["kalshi_temp_e1"], "spent_unseals": [
        {"spec_id": "s1", "when": "2026-10-01"}, {"spec_id": "s2", "when": "2026-10-05"}]}
    monkeypatch.setattr(vault, "config", lambda: fake)
    reg = Registry(tmp_path / "r.jsonl")
    for sid in ("s1", "s2", "s3"):
        reg.register({**SPEC, "id": sid})
    assert reg.sealed_bar("fam_test", "s1") == (0.05 / 2, 1)
    assert reg.sealed_bar("fam_test", "s2") == (0.05 / 4, 2)
    assert reg.sealed_bar("fam_test", "s3") == (0.05 / 8, 3)          # next shot
    assert reg.sealed_bar("other_family", "x") == (0.05 / 2, 1)
    # family total of all possible shots stays <= .05
    assert sum(0.05 / 2 ** j for j in range(1, 60)) <= 0.05


def test_weather_next_sealed_shot_is_05_over_32(tmp_path):
    from edgelab import selftest
    reg = selftest._history_sandbox(tmp_path)
    shots = reg.sealed_shots("kalshi_weather_temp")
    assert [s for _, s in shots] == ["e1_taker", "e2_taker", "e4_maker_e1", "e4_maker_x1"]
    assert reg.sealed_bar("kalshi_weather_temp", "a_new_weather_spec") == (0.05 / 32, 5)
    assert reg.sealed_bar("kalshi_weather_temp", "e1_taker") == (0.05 / 2, 1)


# ------------------------------------------------------------------ rule 5(c) concentration v2
def test_concentration_v2_drop_top_week_and_cluster():
    r = _rows(400, 0.0, 3)
    r["outcome"] = 0.0
    top = r["cluster"].iloc[0]
    r.loc[r["cluster"] == top, "outcome"] = 1.0
    res = E.score(schema.enforce(r), bar=0.5, min_n=10, version=2)
    assert not res["checks"]["c_concentration_ok"]
    # a broad edge whose top week holds > 30% of a small total profit: v1 fails (c), v2 passes (c)
    r = _rows(1200, 0.0, 7, days=60)
    rng = np.random.default_rng(1)
    r["outcome"] = (rng.random(len(r)) < 0.53).astype(float)
    v1 = E.score(schema.enforce(r), bar=0.5, min_n=10, version=1)
    v2 = E.score(schema.enforce(r), bar=0.5, min_n=10, version=2)
    c = v2["concentration"]
    assert (c["roi_ex_top_week"] > 0 and c["roi_ex_top_cluster"] > 0) == v2["checks"]["c_concentration_ok"]
    assert "roi_ex_top_week" not in v1["concentration"]               # v1 output unchanged


def _nc_rows(n, loss_ratio, seed, price=0.96, days=None):
    """near-certain bets: price .96, loss prob = loss_ratio x break-even (1 - cost)."""
    rng = np.random.default_rng(seed)
    days = days or max(n // 2, 1)
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    day = np.sort(rng.integers(0, days, n))
    dec = t0 + pd.to_timedelta(day, unit="D")
    p = np.full(n, price)
    cost = p + F.kalshi_fee_per_contract(p, 1, rounding="exact")
    out = (rng.random(n) >= loss_ratio * (1 - cost)).astype(float)
    return pd.DataFrame({"bet_id": [f"b{i}" for i in range(n)], "cluster": dec.strftime("%Y-%m-%d"),
                         "event_start": dec + pd.Timedelta(hours=2), "decided_at": dec,
                         "inputs_as_of": dec - pd.Timedelta(hours=1), "price": p, "price_source": "taker",
                         "price_time": dec, "side": "no", "stake": 1.0, "outcome": out, "close_price": p + 0.005})


# ------------------------------------------------------------------ too-good guard v2
def test_no_sharpe_trigger_at_near_certain_but_roi_trigger_stays():
    r = schema.enforce(_nc_rows(2000, 0.3, 11, days=400))          # strong steady near-certain edge (~+2.8%)
    v1 = E.score(r, bar=0.01, min_n=100, fee={"rounding": "exact"}, version=1)
    v2 = E.score(r, bar=0.01, min_n=100, fee={"rounding": "exact"}, version=2)
    assert v1["sharpe_ann"] > 5 and v1["needs_audit"]
    assert v2["near_certain"] and not v2["sharpe_guard_applies"] and not v2["needs_audit"]
    hot = E.score(schema.enforce(_rows(400, 0.25, 4)), bar=0.05, min_n=10, version=2)   # 50c, ROI ~ +50%
    assert hot["needs_audit"]


# ------------------------------------------------------------------ forward sequential tests
def _stream_nc(rng, ratio, months=36, per_month=55, price_mix=(0.94, 0.96, 0.97, 0.98)):
    n = rng.poisson(per_month * months)
    day = np.sort(rng.integers(0, months * 30, n))
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    dec = t0 + pd.to_timedelta(day, unit="D")
    p = rng.choice(price_mix, n)
    cost = p + F.kalshi_fee_per_contract(p, 1, rounding="exact")
    out = (rng.random(n) >= np.clip(ratio * (1 - cost), 0, 1)).astype(float)
    df = pd.DataFrame({"cluster": dec.strftime("%Y-%m-%d"), "decided_at": dec, "event_start": dec + pd.Timedelta(hours=2),
                       "price": p, "outcome": out})
    return df, cost


def _rate(fn, reps):
    st = [fn(i)["status"] for i in range(reps)]
    return {s: st.count(s) / reps for s in set(st)}


def test_near_certain_placebo_rarely_accepts_and_planted_edge_accepts():
    rng = np.random.default_rng(20261118)
    plc = _rate(lambda i: FT.run_test(*_stream_nc(rng, 1.0), None, True), 300)
    assert plc.get("accept_h1", 0) <= 0.02                        # nominal .005 (iid, no month shock)
    assert plc.get("reject_h0", 0) >= 0.6                          # the kill side does work on a null
    big = _rate(lambda i: FT.run_test(*_stream_nc(rng, 0.4), None, True), 100)
    assert big.get("accept_h1", 0) >= 0.85                         # loss ratio .4 (~+3% ROI here)
    bad = _rate(lambda i: FT.run_test(*_stream_nc(rng, 1.4), None, True), 100)
    assert bad.get("reject_h0", 0) >= 0.95


def _stream_sp(rng, roi, months=24, per_month=200, lo=0.49, hi=0.535):
    n = per_month * months
    day = np.sort(rng.integers(0, months * 30, n))
    dec = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(day, unit="D")
    p = rng.uniform(lo, hi, n)
    cost = p + F.kalshi_fee_per_contract(p, 1, rounding="exact")
    out = (rng.random(n) < np.clip(cost * (1 + roi), 0, 1)).astype(float)
    return pd.DataFrame({"cluster": dec.strftime("%Y-%m-%d"), "decided_at": dec, "event_start": dec,
                         "price": p, "outcome": out}), cost


def test_sports_roi_placebo_and_planted_edge_and_scope():
    rng = np.random.default_rng(7)
    plc = _rate(lambda i: FT.run_test(*_stream_sp(rng, 0.0), None, True), 300)
    assert plc.get("accept_h1", 0) <= 0.05
    good = _rate(lambda i: FT.run_test(*_stream_sp(rng, 0.10), None, True), 100)
    assert good.get("accept_h1", 0) >= 0.85
    bad = _rate(lambda i: FT.run_test(*_stream_sp(rng, -0.05), None, True), 100)
    assert bad.get("reject_h0", 0) >= 0.9
    # scope: one 65c bet takes the whole trial out of the calibrated range -> no decision either way
    df, cost = _stream_sp(rng, -0.2)
    df.loc[df.index[0], "price"] = 0.65
    r = FT.run_test(df, cost, None, True)
    assert r["status"] == "out_of_scope" and not r["kills"] and not r["promotes"]
    r = FT.run_test(df, cost, {"kind": "sports_roi"}, True)
    assert r["status"] == "out_of_scope"


def test_sports_horizon_24_in_season_months():
    rng = np.random.default_rng(3)
    df, cost = _stream_sp(rng, 0.0, months=30, per_month=5)        # too few bets to decide
    r = FT.run_test(df, cost, None, True)
    assert r["status"] == "horizon_user_decides" and r["months_elapsed"] > 24


def test_forward_test_spec_validation_and_design_effect(tmp_path):
    assert FT.validate({"kind": "near_certain"})["fills_per_night"] == pytest.approx(55 / 30)
    with pytest.raises(FT.ForwardTestError):
        FT.validate({"kind": "sports_roi", "h1_roi": 0.02})        # uncalibrated alternative
    with pytest.raises(FT.ForwardTestError):
        FT.validate({"kind": "near_certain", "alpha": 0.05})
    reg = Registry(tmp_path / "r.jsonl")
    with pytest.raises(RegistryError):
        reg.register({**SPEC, "id": "bad_ft", "forward_test": {"kind": "magic"}})
    rng = np.random.default_rng(0)
    df, cost = _stream_nc(rng, 1.0, months=6)
    a = FT.run_test(df, cost, {"kind": "near_certain", "fills_per_night": 1.0}, True)
    b = FT.run_test(df, cost, {"kind": "near_certain", "fills_per_night": 10.0}, True)
    assert a["deff"] == pytest.approx(1.031) and b["deff"] == pytest.approx(1.31)
    assert b["llr_min"] == pytest.approx(a["llr_min"] * a["deff"] / b["deff"])
    assert a["params_source"] == "registration"
    assert FT.run_test(df, cost, None, True)["params_source"] == "adoption_default"


def test_first_crossing_is_final():
    llr = np.array([3.0, 3.0, -10.0])                               # crosses A at look 1, later dips below B
    r = FT.sprt_path(llr, np.arange(3), 1.0, 5.0, -2.0, np.ones(3, bool))
    assert r["status"] == "accept_h1" and r["decided_at_look"] == 1


# ------------------------------------------------------------------ integration: kills, audit, gate
def test_sprt_kill_added_and_fires(tmp_path, monkeypatch):
    _fake_adapter(tmp_path, monkeypatch, [0] * 9)                     # 9 straight losses at .90
    reg = Registry(tmp_path / "r.jsonl")
    tid = _at_sealed_pass(reg)
    out = run.evaluate(tid, "forward", reg, save_rows=False)
    assert out["forward_test"]["kind"] == "near_certain" and out["forward_test"]["status"] == "reject_h0"
    kinds = [r["kind"] for r in out["kill"]["rules"]]
    assert kinds == ["roi_ci_below_zero", "forward_sprt_h0"] and out["kill"]["triggered"] is True
    assert reg.state(tid) == "dead"


def test_existing_preset_kill_still_fires_when_sprt_continues(tmp_path, monkeypatch):
    _fake_adapter(tmp_path, monkeypatch, [1, 0, 1, 0, 1, 1])
    reg = Registry(tmp_path / "r.jsonl")
    tid = _at_sealed_pass(reg)
    reg.set_preset_kill(tid, [{"kind": "losses_in_first", "n": 10, "max_losses": 1}])
    out = run.evaluate(tid, "forward", reg, save_rows=False)
    sprt = [r for r in out["kill"]["rules"] if r["kind"] == "forward_sprt_h0"][0]
    assert sprt["triggered"] is False and out["forward_test"]["status"] == "continue"
    assert out["kill"]["rules"][0]["triggered"] is True and out["kill"]["triggered"] is True
    assert reg.state(tid) == "dead"


def test_accept_h1_needs_audit_and_sealed_gate(tmp_path, monkeypatch):
    from edgelab import cli
    _fake_adapter(tmp_path, monkeypatch, [1] * 175)                   # 175 straight wins at .90 -> H1
    p = tmp_path / "r.jsonl"
    reg = Registry(p)
    tid = _at_sealed_pass(reg)
    out = run.evaluate(tid, "forward", reg, save_rows=False)
    assert out["forward_test"]["status"] == "accept_h1" and out["forward_test"]["gate_ok"]
    assert out["needs_audit"] and out["forward_audit_required"] and out["forward_eligible_for_user"]
    assert reg.state(tid) == "forward_paper"                           # evaluate never promotes
    base = ["--registry", str(p), "approve", "t_fwd", "--to", "live_small", "--user-approved", "--by", "test"]
    assert cli.main(base) == 1                                         # no audit note -> refused
    assert cli.main(base + ["--audit-note", "rows inspected"]) == 0
    assert reg.state(tid) == "live_small"
    # same forward data on a trial that never reached sealed_pass (X1 / L32 case): cannot promote
    reg2 = Registry(tmp_path / "r2.jsonl")
    tid2 = reg2.register(FWD_SPEC)
    out2 = run.evaluate(tid2, "forward", reg2, save_rows=False)
    assert out2["forward_test"]["status"] == "accept_h1" and not out2["forward_test"]["gate_ok"]
    assert not out2["forward_eligible_for_user"]


def test_bar_version_1_unchanged(tmp_path, monkeypatch):
    _fake_adapter(tmp_path, monkeypatch, [0] * 9)
    reg = Registry(tmp_path / "r.jsonl")
    tid = _at_sealed_pass(reg)
    out = run.evaluate(tid, "forward", reg, save_rows=False, bar_version=1)
    assert "forward_test" not in out and "bar_version" not in out
    assert [r["kind"] for r in out["kill"]["rules"]] == ["roi_ci_below_zero"]
    assert reg.state(tid) == "forward_paper"
