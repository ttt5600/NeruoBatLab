"""Offline tests: parsers against captured real responses, and leakage checks.

The fixtures are real API responses (see scripts/capture_fixtures.py), so a
parser test failing after a re-capture means the API changed shape -- which,
for four undocumented APIs, is the most likely way this package breaks.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sports import features, market, mlb, nba, nfl, nhl, schema

FIX = Path(__file__).parent / "fixtures"


def load(name):
    with gzip.open(FIX / f"{name}.json.gz", "rt") as f:
        return json.load(f)


# ------------------------------------------------------------------ parsers
def test_mlb_parses_venue_weather_and_probable_pitchers():
    df = mlb.from_schedule_json([load("mlb_schedule_2025-07-04")])
    assert len(df) == 15
    coors = df[df["venue"] == "Coors Field"].iloc[0]
    assert coors["elevation_ft"] == 5190
    assert coors["home"] == "COL"
    assert df["home_starter"].notna().all(), "every final game had an announced probable"
    assert df["officials"].notna().all(), "home-plate umpire on every game"
    assert df["temp_f"].between(40, 115).all()
    # Closed roofs are reported through the weather condition string.
    closed = df[df["condition"] == "Roof Closed"]
    assert len(closed) >= 1 and set(closed["roof"]) <= {"retractable", "dome"}


def test_mlb_wind_string_parsing():
    assert mlb.parse_wind("12 mph, Out To CF") == (12.0, "Out To CF")
    assert mlb.parse_wind("0 mph, None") == (0.0, "None")
    assert mlb.parse_wind("Calm") == (0.0, "Calm")
    assert mlb.parse_wind(None) == (None, None)


def test_mlb_postseason_is_tagged():
    ws = mlb.from_schedule_json([load("mlb_schedule_2024-10-30")])
    assert len(ws) == 1 and ws["game_type"].iloc[0] == "post"
    assert (ws["home"].iloc[0], ws["away"].iloc[0]) == ("NYY", "LAD")


def test_nhl_week_and_starting_goalies():
    df = nhl.from_schedule_json([load("nhl_schedule_2025-01-13")])
    assert len(df) > 40
    assert df["season"].eq(2024).all(), "season labelled by start year"
    assert df["date"].min() == pd.Timestamp("2025-01-13")
    assert df["home_score"].notna().all()
    g = nhl.parse_starting_goalies(load("nhl_boxscore"))
    assert g["home_starter"] and g["away_starter"]
    assert isinstance(g["home_starter_id"], int)


def test_nba_drops_all_star_and_keeps_partial_pulls():
    days = [("2025-01-15", load("nba_scoreboard_2025-01-15")),
            ("2025-02-16", load("nba_scoreboard_2025-02-16"))]
    df = nba.from_scoreboard_json(days)
    # Two days of games is a partial pull; nothing real may be filtered out.
    assert len(df) == 11
    assert not df["home"].isin(["CHK", "KEN", "SHQ", "CAN"]).any(), "All-Star leaked in"
    assert df["season"].eq(2024).all(), "ESPN labels by end year; we use start year"
    assert (df["date"] == pd.Timestamp("2025-01-15")).all()
    blowout = df[(df["home"] == "LAC") & (df["away"] == "BKN")].iloc[0]
    assert (blowout["home_score"], blowout["away_score"]) == (126, 67)


def test_nfl_conversion_known_game():
    g = nfl.from_raw(pd.read_csv(FIX / "nfl_games_2024.csv.gz"))
    assert len(g) == 285 and (g["game_type"] == "post").sum() == 13
    opener = g[g["game_id"] == "2024_01_BAL_KC"].iloc[0]
    assert (opener["home_score"], opener["away_score"]) == (27, 20)
    assert opener["spread_line"] == 3.0, "positive = home favoured"
    assert opener["home_starter"] == "Patrick Mahomes"
    # 8:20pm Eastern on Sep 5 is 00:20 UTC on Sep 6.
    assert opener["start_utc"] == pd.Timestamp("2024-09-06 00:20", tz="UTC")


# ------------------------------------------------------------------ schema
def test_untagged_column_is_refused():
    df = schema.empty_games().assign(secret_feature=[])
    with pytest.raises(ValueError, match="untagged"):
        schema.conform(df)


def test_pregame_view_never_contains_scores():
    g = nfl.from_raw(pd.read_csv(FIX / "nfl_games_2024.csv.gz"))
    for as_of in ("pre", "lineup", "close"):
        cols = set(schema.pregame(g, as_of).columns)
        assert not cols & {"home_score", "away_score", "overtime", "attendance"}
    assert "spread_line" not in schema.pregame(g, "lineup").columns
    assert "spread_line" in schema.pregame(g, "close").columns
    assert "home_coach" in schema.pregame(g, "pre").columns


# ------------------------------------------------------------------ leakage
def synthetic_league(n_teams=8, n_seasons=3, games_per=30, seed=1):
    """Random schedule with a persistent strength per team, so form/Elo vary."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    strength = rng.normal(0, 3, n_teams)
    rows, t0 = [], pd.Timestamp("2020-01-01", tz="UTC")
    gid = 0
    for s in range(n_seasons):
        for k in range(games_per):
            perm = rng.permutation(n_teams)
            for j in range(0, n_teams, 2):
                h, a = perm[j], perm[j + 1]
                start = t0 + pd.Timedelta(days=400 * s + 2 * k + rng.integers(0, 2),
                                          hours=int(rng.integers(0, 6)))
                hs = max(0, int(round(20 + strength[h] + 1.5 + rng.normal(0, 7))))
                as_ = max(0, int(round(20 + strength[a] + rng.normal(0, 7))))
                rows.append(dict(league="NFL", game_id=f"g{gid:05d}", season=2020 + s,
                                 game_type="regular", date=start.tz_convert(None).normalize(),
                                 start_utc=start, home=teams[h], away=teams[a], neutral=False,
                                 home_score=hs, away_score=as_))
                gid += 1
    return schema.conform(pd.DataFrame(rows))


def test_features_are_causal():
    """Features for game g must not change when every later game is deleted."""
    g = synthetic_league()
    full = features.build(g).set_index("game_id")
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(np.arange(20, len(g)), 25, replace=False)):
        gid = g.iloc[i]["game_id"]
        # Keep only games that started strictly before g, plus g itself.
        cut = g[(g["start_utc"] < g.iloc[i]["start_utc"]) | (g["game_id"] == gid)]
        trunc = features.build(cut).set_index("game_id")
        a = full.loc[gid, features.FEATURE_COLUMNS].astype(float)
        b = trunc.loc[gid, features.FEATURE_COLUMNS].astype(float)
        pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-12)


def test_features_do_not_see_their_own_result():
    """Flipping a game's score must leave that game's features untouched."""
    g = synthetic_league()
    i = 150
    flipped = g.copy()
    flipped.loc[i, ["home_score", "away_score"]] = (
        g.loc[i, "away_score"] + 30, g.loc[i, "home_score"])
    a = features.build(g).set_index("game_id").loc[g.loc[i, "game_id"], features.FEATURE_COLUMNS]
    b = features.build(flipped).set_index("game_id").loc[g.loc[i, "game_id"], features.FEATURE_COLUMNS]
    pd.testing.assert_series_equal(a.astype(float), b.astype(float), check_names=False)


def test_elo_learns_real_strength_on_synthetic_league():
    """On a league with built-in strengths, Elo must beat a coin flip.

    The positive control: if this fails, a null result on real data would say
    nothing about the real data.
    """
    g = synthetic_league(n_seasons=4, games_per=60)
    f = features.build(g)
    late = f[f["season"] >= 2021]
    y = market.home_result(late)
    ok = y.notna()
    assert market.log_loss(late.loc[ok, "p_home_elo"], y[ok]) < np.log(2) - 0.01


def test_rest_days_and_back_to_back():
    g = synthetic_league()
    tg = features.add_team_history(features.team_games(g))
    first = tg.groupby(["team", "season"]).head(1)
    assert first["rest_days"].isna().all(), "season opener has no prior game"
    rest = tg["rest_days"].dropna()
    assert (rest > 0).all()
    assert tg.loc[tg["rest_days"] < 1.5, "b2b"].all()


# ------------------------------------------------------------------ market
def test_odds_arithmetic():
    assert market.american_to_prob(-150) == pytest.approx(0.6)
    assert market.american_to_prob(130) == pytest.approx(100 / 230)
    assert market.american_to_decimal(-110) == pytest.approx(1.9090909)
    pa, pb, over = market.devig_two_way(-110, -110)
    assert pa == pytest.approx(0.5) and over == pytest.approx(0.0476, abs=1e-4)


def test_paired_bootstrap_detects_a_better_forecast_and_not_an_equal_one():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.2, 0.8, 4000)
    y = (rng.uniform(size=4000) < p_true).astype(float)
    noisy = np.clip(p_true + rng.normal(0, 0.15, 4000), 0.01, 0.99)
    better = market.paired_logloss_bootstrap(p_true, noisy, y)
    assert better["observed_diff"] < 0 and better["p_model_not_better"] < 0.01
    same = market.paired_logloss_bootstrap(noisy, noisy, y)
    assert same["observed_diff"] == 0.0
