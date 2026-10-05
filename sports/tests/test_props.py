"""Strikeout-prop model: boxscore parsing, leakage, and the probability maths."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sports import mlb_box, props

FIX = Path(__file__).parent / "fixtures"


def test_boxscore_parse_known_start():
    with gzip.open(FIX / "mlb_boxscore_777245.json.gz", "rt") as f:
        b = json.load(f)
    P, B = mlb_box.parse_box(777245, b)
    p = pd.DataFrame(P)
    soroka = p[p["pitcher"] == "Michael Soroka"].iloc[0]
    assert soroka["started"] and soroka["team"] == "WSH" and soroka["opponent"] == "BOS"
    assert (soroka["k"], soroka["bf"], soroka["pitches"], soroka["outs"]) == (6, 24, 93, 12)
    assert p.groupby("team")["started"].sum().eq(1).all(), "exactly one starter per team"
    bat = pd.DataFrame(B).set_index("team")
    assert bat.loc["WSH", "bat_k"] == 8 and bat.loc["WSH", "bat_pa"] == 35


def test_innings_to_outs():
    assert mlb_box._ip_outs("5.2") == 17
    assert mlb_box._ip_outs("0.1") == 1
    assert mlb_box._ip_outs(None) is None


# ------------------------------------------------------------------ leakage
def synthetic_season(seed=0, n_days=200, n_teams=10):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    pitchers = {t: [f"{t}_p{j}" for j in range(5)] for t in teams}
    true_k = {p: rng.uniform(0.15, 0.32) for ps in pitchers.values() for p in ps}
    P, B, pk = [], [], 0
    t0 = pd.Timestamp("2023-04-01", tz="UTC")
    for d in range(n_days):
        order = rng.permutation(teams)
        for i in range(0, n_teams, 2):
            h, a = order[i], order[i + 1]
            start = t0 + pd.Timedelta(days=d, hours=int(rng.integers(17, 23)))
            for team, opp in ((h, a), (a, h)):
                sp = pitchers[team][d % 5]
                bf = int(rng.integers(15, 28))
                k = int(rng.binomial(bf, true_k[sp]))
                P.append(dict(game_pk=pk, team=team, opponent=opp, pitcher_id=sp, pitcher=sp,
                              started=True, k=k, bf=bf, start_utc=start, season=2023,
                              game_type="regular", date=start.normalize()))
                B.append(dict(game_pk=pk, team=opp, opponent=team, bat_k=k + int(rng.integers(0, 4)),
                              bat_pa=bf + 12, start_utc=start, season=2023, game_type="regular"))
            pk += 1
    return pd.DataFrame(P), pd.DataFrame(B), true_k


FEATS = ["tr_k", "tr_bf", "bf_sum", "bf_n", "tr_bat_k", "tr_bat_pa", "league_k", "p_pitcher", "p_opp"]


def test_strikeout_features_ignore_the_future_and_the_present():
    P, B, _ = synthetic_season()
    full = props.strikeout_features(P, B).set_index(["game_pk", "pitcher_id"])
    rng = np.random.default_rng(1)
    for pk in rng.choice(np.arange(300, P["game_pk"].max()), 8, replace=False):
        t = P.loc[P["game_pk"] == pk, "start_utc"].iloc[0]
        # Future removed, and THIS game's results scrambled: features must not move.
        P2 = P[P["start_utc"] <= t].copy()
        B2 = B[B["start_utc"] <= t].copy()
        P2.loc[P2["game_pk"] == pk, ["k", "bf"]] = [99, 99]
        B2.loc[B2["game_pk"] == pk, ["bat_k", "bat_pa"]] = [99, 99]
        cut = props.strikeout_features(P2, B2).set_index(["game_pk", "pitcher_id"])
        for key in full.loc[pk].index:
            a = full.loc[(pk, key), FEATS].astype(float)
            b = cut.loc[(pk, key), FEATS].astype(float)
            pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-12)


def test_model_recovers_true_strikeout_rates():
    """Positive control: late-season shrunk K rates must track the true ones."""
    P, B, true_k = synthetic_season(n_days=200)
    st = props.strikeout_features(P, B)
    late = st[st["start_utc"] > st["start_utc"].quantile(0.8)]
    est = late.groupby("pitcher_id")["p_pitcher"].last()
    truth = pd.Series(true_k).loc[est.index]
    assert np.corrcoef(est, truth)[0, 1] > 0.8


def test_p_over_matches_poisson_limit_and_is_monotone():
    from scipy import stats
    mu = np.array([5.3])
    assert props.p_over(mu, np.array([4.5]), r=1e6)[0] == pytest.approx(
        stats.poisson.sf(4, 5.3), abs=1e-4)
    ps = [props.p_over(mu, np.array([x]), 20.0)[0] for x in (3.5, 4.5, 5.5, 6.5)]
    assert all(a > b for a, b in zip(ps, ps[1:]))


def test_log5_identity():
    # A league-average opponent leaves the pitcher's rate unchanged.
    assert props.log5(np.array([0.22]), np.array([0.28]), np.array([0.22]))[0] == pytest.approx(0.28)
