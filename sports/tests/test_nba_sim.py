"""Unit tests for scripts/nba_sim.py: minutes redistribution sums to 240, deterministic seed, delta sign sanity.
Synthetic inputs only (no data files, no network)."""
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nba_sim as ns  # noqa: E402

Row = namedtuple("Row", ["athlete_id", "name", "minutes"] + ns.STATS)
PARAMS = {"margin": {"const": 0.0, "mu0": 0.8, "dPfull": 1.4, "dAbs": 4.0},
          "total": {"const": 0.0, "tot0": 0.9, "Ofull": 0.18, "oAbs": 0.25},
          "sigma_m": 13.5, "sigma_t": 18.4, "rho": 0.0, "cv": {"reb": 0.16, "ast": 0.18}, "min_sd": 0.3,
          "kappa": {"pts": 60.0, "reb": 60.0, "ast": 60.0}}
PRIORS = {"bpm": {"b0": -10.0, "beta": [0.5] + [0.0] * 10, "repl48": [15.0, 13, 3, 2, 2.5, 6, 3.5, 2, 1.5, 1, 4.5],
                  "repl_ppm": 0.33},
          "role": {"G": [0.45, 0.12, 0.15], "W": [0.42, 0.17, 0.07], "B": [0.40, 0.27, 0.05]}}


def row(aid, pts, mins, ast=3, reb=5):
    v = {s: 0.0 for s in ns.STATS}
    v.update(points=pts, assists=ast, defensiveRebounds=reb, offensiveRebounds=1, fieldGoalsAttempted=pts / 2)
    return Row(aid, aid, mins, **v)


def team_state(st, T, star_pts, day0=700000, games=30):
    roster = [(f"{T}_star", star_pts, 36), (f"{T}_g", 14, 32), (f"{T}_w", 12, 30), (f"{T}_b", 10, 28),
              (f"{T}_w2", 9, 26), (f"{T}_s1", 8, 20), (f"{T}_s2", 7, 18), (f"{T}_s3", 6, 16), (f"{T}_s4", 4, 12)]
    for k in range(games):
        st.update(day0 + 2 * k, T, [row(a, p, m) for a, p, m in roster])
    return {a for a, _, _ in roster}


@pytest.mark.parametrize("seed", range(20))
def test_minutes_sum_to_240(seed):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(6, 15))
    base = {i: float(rng.uniform(2, 38)) for i in range(k)}
    roles = {i: "GWB"[int(rng.integers(0, 3))] for i in range(k)}
    absent = [(float(rng.uniform(10, 36)), "GWB"[int(rng.integers(0, 3))]) for _ in range(int(rng.integers(0, 3)))]
    fixed = {0: 30.0} if seed % 2 else None
    m = ns.project_minutes(base, roles, absent, fixed)
    assert sum(m.values()) == pytest.approx(240.0, abs=1e-6)
    assert all(v >= 0 for v in m.values())
    assert max(v for i, v in m.items() if not (fixed and i in fixed)) <= ns.CAP + 1e-6
    if fixed:
        assert m[0] == pytest.approx(30.0)


def test_absent_minutes_prefer_same_role():
    base = {"g1": 30.0, "g2": 20.0, "b1": 30.0, "b2": 20.0, "w1": 30.0, "w2": 20.0, "w3": 20.0, "w4": 20.0}
    roles = {k: k[0].upper() for k in base}
    m0 = ns.project_minutes(base, roles, [])
    m1 = ns.project_minutes(base, roles, [(30.0, "B")])
    assert sum(m1.values()) == pytest.approx(240.0)
    assert m1["b2"] - m0["b2"] > m1["g2"] - m0["g2"]


def test_side_inputs_sum_240_with_absence():
    st = ns.State()
    full = team_state(st, "AAA", 30)
    si = ns.side_inputs(st, 700100, PRIORS, full, full - {"AAA_star"}, team="AAA")
    assert sum(si["m"].values()) == pytest.approx(240.0)
    assert sum(si["m_full"].values()) == pytest.approx(240.0)
    assert "AAA_star" not in si["m"]


def test_deterministic_seed():
    st = ns.State()
    fh, fa = team_state(st, "HHH", 30), team_state(st, "AAA", 25)
    h = ns.side_inputs(st, 700100, PRIORS, fh, fh, team="HHH")
    a = ns.side_inputs(st, 700100, PRIORS, fa, fa, team="AAA")
    r1 = ns.price(PARAMS, 2.0, 225.0, h, a, n=3000, seed=42)
    r2 = ns.price(PARAMS, 2.0, 225.0, h, a, n=3000, seed=42)
    r3 = ns.price(PARAMS, 2.0, 225.0, h, a, n=3000, seed=43)
    assert np.array_equal(r1["margin"], r2["margin"]) and r1["p_home"] == r2["p_home"]
    assert np.array_equal(r1["players_h"]["pts"], r2["players_h"]["pts"])
    assert not np.array_equal(r1["margin"], r3["margin"])
    assert ((r1["hp"] != r1["ap"])).all()                       # OT resolves every tie


def test_delta_sign():
    st = ns.State()
    fh, fa = team_state(st, "HHH", 32), team_state(st, "AAA", 32)
    d = 700100
    H = ns.side_inputs(st, d, PRIORS, fh, fh, team="HHH")
    A = ns.side_inputs(st, d, PRIORS, fa, fa, team="AAA")
    H_out = ns.side_inputs(st, d, PRIORS, fh, fh - {"HHH_star"}, team="HHH")
    A_out = ns.side_inputs(st, d, PRIORS, fa, fa - {"AAA_star"}, team="AAA")
    base = ns.price(PARAMS, 0.0, 225.0, H, A, n=20000, seed=1, players=False)
    home_star_out = ns.price(PARAMS, 0.0, 225.0, H_out, A, n=20000, seed=1, players=False)
    away_star_out = ns.price(PARAMS, 0.0, 225.0, H, A_out, n=20000, seed=1, players=False)
    assert home_star_out["mu"] < base["mu"] < away_star_out["mu"]
    assert home_star_out["p_home"] < base["p_home"] < away_star_out["p_home"]
    assert home_star_out["tot"] < base["tot"]                    # star scorer out -> lower expected total
    assert base["p_home"] == pytest.approx(0.5, abs=0.02)         # symmetric teams, no home edge in mu0


def test_names_to_ids():
    ids, miss = ns._names_to_ids("jalen johnson, Nobody, Jalen", {"1": "Jalen Johnson", "2": "Jalen Wilson"})
    assert ids == {"1"}
    assert any("Nobody" in m for m in miss) and any("ambiguous" in m for m in miss)


PM = namedtuple("PM", ["athlete_id", "name", "minutes", "plusMinus"] + ns.STATS)


def test_v2_onoff_raises_a_positive_onoff_player_and_legacy_is_unchanged():
    """Same box line for two players; A's team wins his minutes by +10 and loses the rest, B is the reverse.
    v2 must rate A above B; legacy must rate them equal (box-only) -- the --legacy path stays v1."""
    st = ns.State()
    for k in range(40):
        base = {s: 0.0 for s in ns.STATS}
        base.update(points=20, fieldGoalsAttempted=16, turnovers=3)
        rows = [PM("A", "A", 36, +10, **base), PM("B", "B", 36, -10, **base)]
        rows += [PM(f"x{j}", f"x{j}", 168 / 8, 0, **{s: 0.0 for s in ns.STATS}) for j in range(8)]
        st.update(700000 + 2 * k, "TTT", rows, margin=0.0)
    try:
        ns.set_impact("v2", 0.5, 1500)
        ra, rb = (ns.player_rates(st, a, 700090, PRIORS)["r"] for a in ("A", "B"))
        assert ra > rb + 1.0
        ns.set_impact("legacy")
        la, lb = (ns.player_rates(st, a, 700090, PRIORS)["r"] for a in ("A", "B"))
        assert la == pytest.approx(lb)
    finally:
        ns.set_impact(ns.DEFAULT_MODE)


def test_update_without_margin_keeps_v1_state():
    st = ns.State()
    team_state(st, "AAA", 30, games=3)
    assert all("oo" not in s for s in st.ST.values())


# ------------------------------------------------------------------------------------------------ v2.1 components
def _restore():
    ns.set_impact(ns.DEFAULT_MODE)


def _ps(seed=0, n=300):
    """Synthetic player-seasons: plus-minus driven by points with a usage-confounded turnover penalty."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for season in (2021, 2022, 2023):
        for i in range(n):
            m = rng.uniform(400, 2500)
            rate = {s: rng.uniform(0.5, 2.0) * m / 48 for s in ns.STATS}
            pm = (0.4 * rate["points"] - 1.5 * rate["turnovers"] + rng.normal(0, 30)) * (1 + 0.3 * (season - 2022))
            rows.append({"season": season, "athlete_id": f"p{i}", "minutes": m, "plusMinus": pm, **rate})
    return pd.DataFrame(rows)


def test_fit_bpm_shrunk_usage_factor_and_ridge_limits():
    import nba_simple_model as nsm
    ps = _ps()
    pooled = nsm.fit_bpm(ps)
    same = ns.fit_bpm_shrunk(ps, pooled, None, 1.0)
    assert np.allclose(same["beta"], pooled["beta"])
    half = ns.fit_bpm_shrunk(ps, pooled, None, 0.5)
    for j in ns.USAGE:
        assert half["beta"][j] == pytest.approx(0.5 * pooled["beta"][j])
    huge = ns.fit_bpm_shrunk(ps, pooled, 1e9, 1.0)            # lam -> inf: back to the pooled betas
    assert np.allclose(huge["beta"], pooled["beta"], atol=1e-4)
    zero = ns.fit_bpm_shrunk(ps, pooled, 0.0, 1.0)            # lam 0: last season's own OLS (differs from pooled)
    assert not np.allclose(zero["beta"], pooled["beta"], atol=1e-3)


def test_recency_kernel_moves_points_not_impact():
    st = ns.State()
    try:
        ns.set_impact("v21", c=4.0, hs=14.0, lam=None, u=1.0, KA=0.0, KM=0.0, HP=180.0, KSHRINK=500.0, KROLE=150.0)
        for k in range(30):                                     # 10 pts for 25 games, then 30 for 5
            st.update(700000 + 2 * k, "AAA", [row("p", 10 if k < 25 else 30, 30)], margin=0.0)
        hot = ns.player_rates(st, "p", 700062, PRIORS)
        ns.set_impact("v21", c=0.0, lam=None, u=1.0, KA=0.0, KM=0.0, HP=180.0, KSHRINK=500.0, KROLE=150.0)
        cold = ns.player_rates(st, "p", 700062, PRIORS)
        assert hot["pts"] > cold["pts"] + 0.02                 # recent games weigh more in the stat projection
        assert hot["r"] == pytest.approx(cold["r"])            # the impact rating keeps the long kernel
    finally:
        _restore()


def test_arena_effect_off_by_default_and_shrunk_when_on():
    st = ns.State()
    try:
        ns.set_impact("v21", KA=500.0, c=0.0, KM=0.0)
        for k in range(20):                                     # scores 30 at arena H, 10 elsewhere
            v = "HHH" if k % 2 == 0 else "XXX"
            st.update(700000 + 2 * k, "AAA", [row("p", 30 if v == "HHH" else 10, 30)], margin=0.0, venue=v)
        at_h = ns.player_rates(st, "p", 700050, PRIORS, venue="HHH")["pts"]
        else_ = ns.player_rates(st, "p", 700050, PRIORS, venue="QQQ")["pts"]
        assert at_h > else_
        ns.set_impact("v21", KA=0.0, c=0.0, KM=0.0)
        assert ns.player_rates(st, "p", 700050, PRIORS, venue="HHH")["pts"] == pytest.approx(
            ns.player_rates(st, "p", 700050, PRIORS, venue=None)["pts"])
    finally:
        _restore()


def test_km_shrinks_early_season_minutes_to_last_season_mpg():
    st = ns.State()
    try:
        ns.set_impact("v21", KM=5.0, c=0.0, KA=0.0)
        for k in range(60):                                     # last season: 34 mpg
            st.update(700000 + 2 * k, "AAA", [row("p", 20, 34)])
        st.update(700400, "AAA", [row("p", 6, 14)])            # new season (gap > 90 d): two 14-min games
        st.update(700402, "AAA", [row("p", 6, 14)])
        e_km = ns.player_rates(st, "p", 700404, PRIORS)["em"]
        ns.set_impact("v21", KM=0.0, c=0.0, KA=0.0)
        e0 = ns.player_rates(st, "p", 700404, PRIORS)["em"]
        assert e0 < e_km < 34.0 and e_km > 25.0
    finally:
        _restore()


def test_presets_reset_extras_and_absence_order_is_deterministic():
    try:
        ns.set_impact("v21", KA=123.0)
        ns.set_impact("v2")
        assert ns.IMPACT == ns.PRESETS["v2"] and ns.impact_tag() == "v2A0.5K1500G0.2"
    finally:
        _restore()
    st = ns.State()
    full = team_state(st, "AAA", 30)
    avail = full - {"AAA_star", "AAA_g", "AAA_w"}
    a = ns.side_inputs(st, 700100, PRIORS, set(sorted(full)), set(sorted(avail)), team="AAA")["m"]
    b = ns.side_inputs(st, 700100, PRIORS, set(sorted(full, reverse=True)), set(sorted(avail, reverse=True)), team="AAA")["m"]
    assert all(a[k] == pytest.approx(b[k], abs=1e-9) for k in a)


def test_v22_minutes_map_preserves_total_cap_and_fixed():
    m = {"s1": 39.0, "s2": 35.0, "b": 30.0, "w": 28.0, "g": 26.0, "x1": 20.0, "x2": 16.0, "x3": 12.0, "x4": 10.0}
    m = {k: v * 240 / sum(m.values()) for k, v in m.items()}
    roles = {k: ("B" if k == "b" else "G") for k in m}
    try:
        ns.set_impact("v21")
        assert ns.adjust_minutes(m, roles) == m                      # v2.1: map off -> identity
        ns.set_impact("v22")
        out = ns.adjust_minutes(m, roles, fixed={"x4": m["x4"]})
        assert sum(out.values()) == pytest.approx(sum(m.values()), abs=1e-6)
        assert max(out.values()) <= ns.PRESETS["v22"]["CAP2"] + 1e-9
        assert out["x4"] == pytest.approx(m["x4"])                     # --minutes players untouched
        assert out["s1"] < m["s1"] and out["x3"] > m["x3"]            # top end compressed, bench up
    finally:
        _restore()


def test_v21_flag_reproduces_v21_player_sims():
    st = ns.State()
    full = team_state(st, "AAA", 30)
    try:
        ns.set_impact("v21")
        si = ns.side_inputs(st, 700100, PRIORS, full, full - {"AAA_g"}, team="AAA")
        assert si["m_pl"] == si["m"]
        ns.set_impact("v22")
        si2 = ns.side_inputs(st, 700100, PRIORS, full, full - {"AAA_g"}, team="AAA")
        assert si2["m"] == pytest.approx(si["m"]) and si2["Pav"] == pytest.approx(si["Pav"])   # team inputs = v2.1
        assert si2["m_pl"] != si2["m"]
    finally:
        _restore()
