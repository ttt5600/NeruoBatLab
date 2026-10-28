"""fund_stats.py (statistician gate): BH/Bonferroni arithmetic, trial accounting, halves parsing, every verdict path,
the power formula, and a smoke run on the real ledger. No network."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fund_stats as F  # noqa: E402


def row(name, p, ds="d", **kw):
    return {"strategy": name, "dataset": ds, "status": "ok", "p": p, **kw}


def filler(n, ds="filler"):
    return [row(f"f{i}", 0.5, ds, roi=-0.01) for i in range(n)]


# ------------------------------------------------------------------------------------------------ arithmetic
def test_bh_qvalues_match_hand_computation():
    ps = [0.01, 0.04, 0.03, 0.5]
    # sorted .01 .03 .04 .5 -> raw .04 .06 .0533 .5 -> monotone from the top: .04 .0533 .0533 .5
    q = F.bh_qvalues(ps)
    assert q == pytest.approx([0.04, 0.05333333, 0.05333333, 0.5])


def test_bh_q_capped_at_one_and_order_preserved():
    q = F.bh_qvalues([0.9, 0.99, 1.0])
    assert all(v <= 1.0 for v in q) and q[0] <= q[1] <= q[2]


def test_power_formula():
    n = F.n_required(0.02, 1.0)
    assert n == pytest.approx(((1.6448536 + 0.8416212) * 1.0 / 0.02) ** 2, rel=1e-6)
    assert math.isinf(F.n_required(0.0, 1.0))
    assert F.power_at(n, 0.02, 1.0) == pytest.approx(0.8, abs=1e-6)


def test_p_used_takes_most_conservative_and_floors_zero():
    assert F.p_used({"p": 0.001, "p_null": 0.016, "p_null_1per_cm": 0.12}) == 0.12
    assert F.p_used({"p": 0.0, "p_exact_vs_breakeven": 0.71}) == 0.71
    assert F.p_used({"p": 0.0}) == F.P_FLOOR
    # zero bootstrap p with a CI: normal approximation from the CI instead of the floor
    r = {"p": 0.0, "units": 10.0, "ci_lo": 5.0, "ci_hi": 15.0}
    assert F.p_used(r) == pytest.approx(1 - F.N.cdf(10 / (10 / 3.92)))
    assert F.p_used({"roi": 0.1}) is None


def test_halves_parsers():
    assert F.halves({"per_half": {"a": 0.5, "b": 0.6}}) == (0.5, 0.6)
    assert F.halves({"halves": {"<=14": [-0.2, 48], ">=15": [0.1, 1112]}}) == (-0.2, 0.1)
    assert F.halves({"half1": [0.01, 13], "half2": [0.02, 154]}) == (0.01, 0.02)
    assert F.halves({"units_h1": 2.8, "units_h2": 25.0}) == (2.8, 25.0)
    assert F.halves({"per_fold": {"2019": 1.0, "2020": 2.0, "2021": -1.0, "2022": 5.0}}) == (3.0, 4.0)
    assert F.halves({"sub": {"2006-12": 1.0, "2013-18": -1.0, "2019-24": 3.0}}) == (1.0, 1.0)
    assert F.halves({"roi": 0.1}) is None


def test_sign_consistency():
    assert F.sign_consistent({"roi": 0.1, "half1": 0.1, "half2": 0.2}) is True
    assert F.sign_consistent({"roi": 0.1, "half1": -0.1, "half2": 0.2}) is False
    assert F.sign_consistent({"roi": 0.1}) is None


# ------------------------------------------------------------------------------------------- trial accounting
def test_void_rows_cancel_the_earlier_trial_only():
    rows = [row("h1", 0.0, "x", sha="aa", roi=0.5), {"strategy": "h1", "dataset": "x", "sha": "aa",
                                                       "status": "invalidated"},
            row("h1", 0.9, "x", sha="aa", roi=-0.05),
            row("u4", 0.2, "y", trial_number=148, roi=0.1),
            {"strategy": "u4", "dataset": "y", "status": "void_bug", "voids_trial_number": 148},
            {"strategy": "bad", "dataset": "y", "status": "rejected: x"}]
    g = F.Gate(rows)
    assert g.n == 1 and g.by_name["h1"]["p"] == 0.9 and "u4" not in g.by_name


def test_bars_and_families():
    rows = [row("a", 0.0001, "d1", roi=0.1, bonferroni_bar=0.001), row("b", 0.02, "d1", roi=0.1)] + filler(98)
    g = F.Gate(rows)
    assert g.n == 100 and g.bonf == pytest.approx(0.0005)
    assert g.family["d1"] == 2 and g.dataset_bar(g.by_name["b"]) == 0.025
    assert g.dataset_bar(g.by_name["a"]) == 0.001  # the stricter pre-registered bar wins
    assert g.tier(g.by_name["a"]) == "ledger" and g.tier(g.by_name["b"]) == "dataset"


# ------------------------------------------------------------------------------------------------- verdicts
def gate_with(target, lockbox=(), fund_data=None):
    return F.Gate([target] + filler(50), list(lockbox), Path(fund_data) if fund_data else F.FUND_DATA)


def test_verdict_dead_paths():
    assert F.verdict(row("t", 0.4, roi=0.05, half1=1, half2=1), gate_with(row("t", 0.4, roi=0.05))) == "DEAD"
    g = gate_with(row("t", 0.0001, roi=-0.05, half1=1, half2=1))
    assert g.verdict("t") == "DEAD"  # negative effect
    g = gate_with(row("t", 0.0001, roi=0.05, half1=-1, half2=1))
    assert g.verdict("t") == "DEAD"  # halves flip
    assert g.verdict("not_in_ledger") == "DEAD"


def test_verdict_incomplete_without_halves():
    g = gate_with(row("t", 0.0001, roi=0.05))
    assert g.verdict("t") == "INCOMPLETE"
    assert [r["strategy"] for r in g.missing_halves()] == ["t"]


def test_verdict_lead_and_clv_caps_at_lead():
    g = gate_with(row("t", 0.0001, roi=0.05, half1=1, half2=1))
    assert g.verdict("t") == "LEAD"
    g = gate_with(row("t_clv", 0.0, metric="clv", clv=1.0, half1=1, half2=1))
    ev = g.evaluate(g.by_name["t_clv"], forward=("open_totals", None))
    assert ev["verdict"] == "LEAD" and any("CLV" in s for s in ev["reasons"])


def test_verdict_sealed_overrides_everything():
    t = row("t", 0.0001, roi=0.05, half1=1, half2=1)
    assert gate_with(t, [{"strategy": "t", "verdict": "FAIL"}]).verdict("t") == "DEAD"
    assert gate_with(t, [{"strategy": "t", "p": 0.001}]).verdict("t") == "PASSED"
    assert gate_with(t, [{"strategy": "t", "passed": False, "p": 0.001}]).verdict("t") == "DEAD"
    # alias: lockbox name differs from the dev trial name
    t2 = row("nba_simple_T5_tot_strict_open_novig", 0.001, roi=0.08, half1=1, half2=1)
    assert gate_with(t2, [{"strategy": "nba_L32_tot_open", "verdict": "FAIL (inconclusive)"}]).verdict(
        "nba_simple_T5_tot_strict_open_novig") == "DEAD"


def _write_settles(d: Path, xs):
    with (d / "open_totals_picks.jsonl").open("w") as fh:
        for x in xs:
            fh.write(json.dumps({"type": "settle", "clv_pts": x}) + "\n")


def test_verdict_paper_then_sealed_ask(tmp_path):
    t = row("t", 0.0001, roi=0.05, half1=1, half2=1)
    _write_settles(tmp_path, [1.0, -0.5, 2.0])  # 3 settled: forward exists but too small
    g = gate_with(t, fund_data=tmp_path)
    assert g.evaluate(g.by_name["t"], forward=("open_totals", None))["verdict"] == "PAPER"
    _write_settles(tmp_path, [1.0 + (i % 5) * 0.3 - 0.4 for i in range(120)])  # 120 settled, clearly positive
    g = gate_with(t, fund_data=tmp_path)
    ev = g.evaluate(g.by_name["t"], forward=("open_totals", None))
    assert ev["forward"]["passes"] and ev["tier"] in ("ledger", "dataset") and ev["verdict"] == "SEALED-ASK"
    # same forward evidence, but dev p misses both bars -> stays PAPER
    t_weak = row("t", 0.003, roi=0.05, half1=1, half2=1)
    g = F.Gate([t_weak] + [row(f"z{i}", 0.002, "d", roi=0.01, half1=1, half2=1) for i in range(30)] + filler(50),
               [], tmp_path)
    assert g.tier(g.by_name["t"]) == "none"
    assert g.evaluate(g.by_name["t"], forward=("open_totals", None))["verdict"] == "PAPER"


def test_forward_parser_failure_is_reported_not_raised(tmp_path):
    (tmp_path / "kalshi_paper.jsonl").write_text('{"cohort": "orig7", "result": "no", "no_price": "bad"}\nnot json\n')
    st = F.forward_stats(("weather_paper", "orig7"), tmp_path)
    assert st["n"] == 0 and st["passes"] is False


# ------------------------------------------------------------------------------------------- real ledger smoke
@pytest.mark.skipif(not F.LEDGER.exists(), reason="no ledger")
def test_real_ledger_smoke(tmp_path):
    g = F.Gate.load(fund_data=tmp_path)  # empty forward dir: deterministic
    assert g.n > 300 and len(g.family) > 30
    v = {e["id"]: e["verdict"] for e in F.lead_rows(g)}
    assert v["E1"] == "PASSED" and v["E4"] == "PASSED" and v["L32"] == "DEAD" and v["L37"] == "DEAD"
    assert "STATISTICIAN GATE" in F.report(g)
    json.loads(F.report(g, as_json=True))
