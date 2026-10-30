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
