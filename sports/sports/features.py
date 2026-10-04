"""Leakage-safe game features: rest, recent form, and a pre-game Elo rating.

Every feature for game ``g`` is a function of games that *started before* ``g``
only. That is checked, not assumed: ``test_features_are_causal`` recomputes
each game's features on history truncated just before it and requires an exact
match -- the sports twin of ``test_every_strategy_is_causal`` in ``quant``.

The Elo parameters are a priori, taken from FiveThirtyEight's published
per-league settings rather than tuned here. Tuning them on the same games we
then evaluate would be the 22-config SMA sweep all over again.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# K-factor, home advantage (Elo points), fraction regressed to the mean between
# seasons. Values follow FiveThirtyEight's published per-league models.
ELO_PARAMS = {
    "NFL": dict(k=20.0, hfa=48.0, revert=1 / 3),
    "NBA": dict(k=20.0, hfa=100.0, revert=1 / 4),
    "MLB": dict(k=4.0, hfa=24.0, revert=1 / 3),
    "NHL": dict(k=6.0, hfa=50.0, revert=1 / 3),
}


def completed(games: pd.DataFrame) -> pd.DataFrame:
    return games[games["home_score"].notna() & games["away_score"].notna()]


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, ordered by start time."""
    base = ["league", "game_id", "season", "game_type", "date", "start_utc", "neutral"]
    h = games[base].assign(team=games["home"], opp=games["away"], is_home=True,
                           pf=games["home_score"], pa=games["away_score"])
    a = games[base].assign(team=games["away"], opp=games["home"], is_home=False,
                           pf=games["away_score"], pa=games["home_score"])
    tg = pd.concat([h, a], ignore_index=True)
    tg["margin"] = tg["pf"] - tg["pa"]
    tg["win"] = np.sign(tg["margin"]).map({1.0: 1.0, -1.0: 0.0, 0.0: 0.5})
    return tg.sort_values(["team", "start_utc", "game_id"]).reset_index(drop=True)


def add_team_history(tg: pd.DataFrame, form_window: int = 10) -> pd.DataFrame:
    """Rest and form from strictly earlier games of the same team.

    Within a season only: last season's final ten games say little about a
    rebuilt roster, and letting them in would make week 1 look better informed
    than it is.
    """
    tg = tg.copy()
    g = tg.groupby(["team", "season"], sort=False)
    prev = g["start_utc"].shift(1)
    tg["rest_days"] = (tg["start_utc"] - prev).dt.total_seconds() / 86400.0
    tg["b2b"] = tg["rest_days"] < 1.5
    tg["games_played"] = g.cumcount()
    tg["form_margin"] = g["margin"].transform(
        lambda s: s.shift(1).rolling(form_window, min_periods=1).mean())
    tg["form_win"] = g["win"].transform(
        lambda s: s.shift(1).rolling(form_window, min_periods=1).mean())
    tg["season_win_pct"] = g["win"].transform(lambda s: s.shift(1).expanding().mean())
    return tg


def elo(games: pd.DataFrame, league: str | None = None) -> pd.DataFrame:
    """Pre-game Elo for every completed game, in start order.

    Returns ``game_id, home_elo_pre, away_elo_pre, p_home_elo``. The rating
    stored for a game is the one held *before* it; the update from its result
    is applied afterwards and first affects the team's next game.
    """
    league = league or games["league"].iloc[0]
    p = ELO_PARAMS[league]
    k, hfa, revert = p["k"], p["hfa"], p["revert"]
    g = completed(games).sort_values(["start_utc", "game_id"])

    rating: dict[str, float] = {}
    season_of: dict[str, int] = {}
    out = []
    for row in g.itertuples(index=False):
        for t in (row.home, row.away):
            r = rating.get(t, 1500.0)
            if season_of.get(t, row.season) != row.season:
                r = 1500.0 + (1 - revert) * (r - 1500.0)
            rating[t], season_of[t] = r, row.season
        rh, ra = rating[row.home], rating[row.away]
        adv = 0.0 if bool(row.neutral) else hfa
        p_home = 1.0 / (1.0 + 10 ** (-(rh + adv - ra) / 400.0))
        out.append((row.game_id, rh, ra, p_home))

        s = 1.0 if row.home_score > row.away_score else 0.0 if row.home_score < row.away_score else 0.5
        rating[row.home] = rh + k * (s - p_home)
        rating[row.away] = ra - k * (s - p_home)

    return pd.DataFrame(out, columns=["game_id", "home_elo_pre", "away_elo_pre", "p_home_elo"])


def build(games: pd.DataFrame, form_window: int = 10) -> pd.DataFrame:
    """Game-level frame: the schema columns plus home_/away_ history and Elo."""
    done = completed(games)
    tg = add_team_history(team_games(done), form_window)
    feats = ["rest_days", "b2b", "games_played", "form_margin", "form_win", "season_win_pct"]
    home = tg[tg["is_home"]].set_index("game_id")[feats].add_prefix("home_")
    away = tg[~tg["is_home"]].set_index("game_id")[feats].add_prefix("away_")
    out = done.set_index("game_id").join(home).join(away).reset_index()
    out = out.merge(elo(done), on="game_id", how="left")
    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]
    return out.sort_values(["start_utc", "game_id"]).reset_index(drop=True)


FEATURE_COLUMNS = [
    "home_rest_days", "away_rest_days", "rest_diff", "home_b2b", "away_b2b",
    "home_games_played", "away_games_played", "home_form_margin", "away_form_margin",
    "home_form_win", "away_form_win", "home_season_win_pct", "away_season_win_pct",
    "home_elo_pre", "away_elo_pre", "p_home_elo",
]
"""Derived columns, all built from strictly earlier games. Safe as pre-game inputs."""
