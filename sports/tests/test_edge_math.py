"""Unit tests for scripts/edge_math.py (hand-checked values)."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import edge_math as em  # noqa: E402


def test_odds_conversions():
    assert em.american_to_decimal(-110) == pytest.approx(1.909091, 1e-6)
    assert em.american_to_decimal(150) == pytest.approx(2.5)
    assert em.implied_prob(-110) == pytest.approx(110 / 210)
    assert em.implied_prob(2.5) == pytest.approx(0.4)
    assert em.implied_prob(0.37) == pytest.approx(0.37)
    assert em.prob_to_american(0.847457627118644) == pytest.approx(-555.56, abs=0.01)  # kicker break-even
    assert em.decimal_to_american(3.0) == pytest.approx(200)
    with pytest.raises(ValueError):
        em.american_to_decimal(50)


def test_devig_two_way_symmetric():
    for m in em.DEVIG:
        assert em.devig([-110, -110], m) == pytest.approx([0.5, 0.5])
    assert em.overround([em.implied_prob(-110)] * 2) == pytest.approx(0.047619, 1e-5)


def test_devig_methods_sum_to_one_and_order():
    odds = [1.30, 5.0, 9.0]  # favourite + two longshots, overround ~0.080
    for m in em.DEVIG:
        assert sum(em.devig(odds, m)) == pytest.approx(1.0, abs=1e-9)
    prop, powr, shin = (em.devig(odds, m) for m in ("proportional", "power", "shin"))
    # proportional leaves the most probability on the longshot; power/Shin move margin off it
    assert prop[2] > powr[2] and prop[2] > shin[2]
    assert prop[0] < powr[0] and prop[0] < shin[0]


def test_fees():
    assert em.fee_per_contract(0.5, "kalshi") == pytest.approx(0.0175)
    assert em.fee_per_contract(0.9, "kalshi") == pytest.approx(0.0063)
    assert em.fee_per_contract(0.9, "kalshi", contracts=1, round_fee=True) == pytest.approx(0.01)
    assert em.fee_per_contract(0.5, "kalshi", contracts=100, round_fee=True) == pytest.approx(0.0175)
    assert em.fee_per_contract(0.5, "kalshi_maker_fee") == pytest.approx(0.004375)
    assert em.fee_per_contract(0.5, "kalshi_maker") == 0
    assert em.fee_per_contract(0.5, "polymarket_us") == pytest.approx(0.017375)
    assert em.fee_per_contract(0.9, "forecastex") == pytest.approx(0.01)
    assert em.all_in_cost(odds=-110) == pytest.approx(0.52381, 1e-5)
    with pytest.raises(ValueError):
        em.fee_per_contract(0.5, "nyse")


def test_ev_and_kelly_identities():
    p, c = 0.92, 0.91
    b = em.net_odds(c)
    assert em.kelly(p, c) == pytest.approx((b * p - (1 - p)) / b)
    assert em.kelly(p, c) == pytest.approx((p - c) / (1 - c))
    assert em.ev_per_dollar(p, c) == pytest.approx(p / c - 1)
    assert em.ev_per_contract(p, c) == pytest.approx(0.01)
    # -110 coin flip: no edge at p = break-even
    c110 = em.all_in_cost(odds=-110)
    assert em.kelly(c110, c110) == pytest.approx(0)
    # classic: 55% at -110 -> f* = 5.5%
    assert em.kelly(0.55, c110) == pytest.approx(0.055, abs=1e-9)


def test_log_growth_max_at_kelly():
    p, c = 0.6, 0.5
    fk = em.kelly(p, c)
    assert fk == pytest.approx(0.2)
    g = em.log_growth(fk, p, c)
    assert g > em.log_growth(fk * 0.8, p, c) and g > em.log_growth(fk * 1.2, p, c)
    assert abs(em.log_growth(2 * fk, p, c)) < 0.15 * g  # 2x Kelly ~ zero growth (exact only in the diffusion limit)
    assert em.log_growth(0, p, c) == 0


def test_wilson():
    lo, hi = em.wilson(0, 10)
    assert lo == 0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = em.wilson(209, 219)  # E5 sealed: 10 losses / 219
    assert lo < 209 / 219 < hi
    assert 1 - hi == pytest.approx(0.02499, abs=1e-4)  # = lockbox loss_ci lower bound 0.02499
    assert 1 - lo == pytest.approx(0.08200, abs=1e-4)  # = lockbox loss_ci upper bound 0.08200


def test_power():
    # fund_economics.md: weather maker mu 1.5%, sigma .195 -> ~1,050 picks
    assert em.n_required(0.015, 0.195) == pytest.approx(1045, abs=5)
    # L32 true +6.4%, sigma 1.0 -> ~1,500
    assert em.n_required(0.064, 1.0) == pytest.approx(1509, abs=5)
    assert em.n_required(0, 1) == float("inf")
    assert em.n_required_roi(0.02, em.all_in_cost(odds=-110)) > 10000


def test_risk_of_ruin_kelly_rules():
    p, c = 0.6, 0.5
    fk = em.kelly(p, c)
    # full Kelly: P(ever halve) ~ 1/2 ; half Kelly ~ (1/2)^3
    assert em.risk_of_ruin(p, c, fk, 0.5) == pytest.approx(0.5, abs=0.08)
    assert em.risk_of_ruin(p, c, fk / 2, 0.5) == pytest.approx(0.125, abs=0.03)
    assert em.risk_of_ruin(p, c, 0.0, 0.5) == 0
    assert em.risk_of_ruin(0.4, c, 0.1, 0.5) == 1.0


def test_simulation_shapes():
    s = em.simulate_drawdown(0.6, 0.5, 0.1, n_bets=200, n_paths=200, seed=1)
    assert 0 < s["median_max_dd"] <= s["p95_max_dd"] < 1
    assert s["median_final"] > 1


def test_clv():
    assert em.clv_prob_points(0.50, 0.53) == pytest.approx(0.03)
    t = em.clv_test([0.01, 0.02, 0.03, 0.02])
    assert t["mean"] == pytest.approx(0.02) and t["p"] < 0.01
    t0 = em.clv_test([0.01, -0.01, 0.02, -0.02])
    assert t0["mean"] == pytest.approx(0) and t0["p"] == pytest.approx(0.5)


def test_bet_table_lcb_and_caps():
    t = em.bet_table(wins=209, n=219, price=0.8877, venue="forecastex", frac=0.5, bankroll=1000, liquidity=29)
    assert t["all_in_cost"] == pytest.approx(0.8977)
    assert t["p_lcb"] < t["p"]
    assert t["kelly_full_at_lcb"] < t["kelly_full"]
    assert t["stake_frac_after_pick_cap"] <= 0.05
    assert t["stake_usd_after_liquidity"] == pytest.approx(29 * 0.8977)
    t2 = em.bet_table(p=0.99, price=0.9, venue="none", frac=1, cap=1, picks_per_night=5, night_cap=0.1)
    assert t2["stake_frac_after_night_cap"] == pytest.approx(0.02)
    t3 = em.bet_table(p=0.5, odds=-110)
    assert t3["ev_per_dollar"] < 0 and t3["stake_frac_after_pick_cap"] == 0


def test_cli_runs(capsys):
    assert em.main(["bet", "--p", "0.92", "--price", "0.90", "--venue", "forecastex"]) == 0
    out = capsys.readouterr().out
    assert "all_in_cost" in out and "VERDICT" in out
    assert em.main(["devig", "--odds", "-120", "100"]) == 0
    assert em.main(["power", "--mu", "0.02", "--sigma", "1"]) == 0
    assert em.main(["clv", "--bet-prob", "0.5", "0.5", "--close-prob", "0.52", "0.51"]) == 0
    assert em.main(["ruin", "--p", "0.95", "--price", "0.9", "--bets", "50"]) == 0
