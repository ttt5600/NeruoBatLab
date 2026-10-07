"""Bet tables: one row per bettable selection, real prices, pre-game features only.

Every dataset has the same shape so one scorer and one sandbox serve all of
them:

    opp_id      groups mutually exclusive selections (home/away, over/under)
    date        game date; season, fold
    selection   "home" | "away" | "over" | "under"
    dec_odds    decimal price actually quoted
    won, push   OUTCOME columns -- given to fit(), stripped before bet()
    <features>  everything else; each knowable before the bet

The split between DEV and LOCKBOX is fixed here, not by any agent. Agents only
ever receive dev rows. The lockbox is scored once per promoted strategy, by
a human command, and every promotion is counted.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sports import features, market, nfl, props

DATA = ROOT / "data"
CACHE = DATA / "lab"
OUTCOME_COLS = ["won", "push", "home_score", "away_score", "actual"]

SPLITS = {
    # dataset: (fold column, dev folds, lockbox folds)
    "nfl_spread": ("season", list(range(2006, 2020)), list(range(2020, 2026))),
    "nfl_total": ("season", list(range(2006, 2020)), list(range(2020, 2026))),
    "nfl_moneyline": ("season", list(range(2006, 2020)), list(range(2020, 2026))),
    "mlb_k_props": ("month", [4, 5, 6, 7], [8, 9]),
    "mlb_total": ("season", [2015, 2016, 2017, 2018, 2019, 2020], [2021]),
}
MIN_TRAIN_FOLDS = {"nfl_spread": 3, "nfl_total": 3, "nfl_moneyline": 3, "mlb_k_props": 1,
                   "mlb_total": 2}

K_LINES_URL = ("https://raw.githubusercontent.com/Msuresh32/pitcherKModel/HEAD/data/"
               "processed_ensemble_wf2025/bt_ensemble_2025_edges.csv")


# ---------------------------------------------------------------- NFL
def _qb_epa(first: int, last: int) -> pd.DataFrame:
    """Trailing QB passing EPA per game over his previous 8 games (strictly prior)."""
    pw = nfl.player_week(range(first, last + 1))
    qb = pw[pw["position"] == "QB"][["player_id", "game_id", "season", "week",
                                     "passing_epa", "attempts"]].copy()
    qb = qb.sort_values(["player_id", "season", "week"])
    g = qb.groupby("player_id")
    qb["qb_epa_per_game_l8"] = g["passing_epa"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean())
    qb["qb_attempts_l8"] = g["attempts"].transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
    qb["qb_career_games"] = g.cumcount()
    return qb[["player_id", "game_id", "qb_epa_per_game_l8", "qb_attempts_l8", "qb_career_games"]]


def _nfl_base() -> pd.DataFrame:
    raw = nfl.raw_games()
    raw = raw[raw["home_score"].notna()]
    # Features (Elo, rest, form) see EVERY completed game; only then keep the
    # games that have real prices. Filtering first would make Elo skip games.
    f = features.build(nfl.from_raw(raw))
    raw = raw[raw["home_spread_odds"].notna()]
    f = f[f["game_id"].isin(raw["game_id"])]
    extra = raw.set_index("game_id")[["week", "div_game", "home_spread_odds", "away_spread_odds",
                                      "over_odds", "under_odds"]]
    f = f.join(extra, on="game_id")
    qb = _qb_epa(int(f["season"].min()), int(f["season"].max()))
    for side in ("home", "away"):
        q = qb.rename(columns={"player_id": f"{side}_starter_id"}).rename(
            columns=lambda c: f"{side}_{c}" if c.startswith("qb_") else c)
        f = f.merge(q, on=["game_id", f"{side}_starter_id"], how="left")
    f["p_home_ml_mkt"], _, _ = market.devig_two_way(f["home_ml"], f["away_ml"])
    f["is_dome"] = f["roof"].isin(["dome", "closed"])
    return f


NFL_FEATURES = [
    "week", "div_game", "neutral", "is_dome", "roof", "surface", "temp_f", "wind_mph",
    "home_coach", "away_coach", "home_starter", "away_starter",
    "spread_line", "total_line", "home_ml", "away_ml", "p_home_ml_mkt",
    "home_spread_odds", "away_spread_odds", "over_odds", "under_odds",
] + features.FEATURE_COLUMNS + [
    f"{s}_{c}" for s in ("home", "away")
    for c in ("qb_epa_per_game_l8", "qb_attempts_l8", "qb_career_games")]


def _two_rows(f, sel_a, sel_b, odds_a, odds_b, won_a, push):
    keep = ["game_id", "season", "date", "start_utc", "home", "away", "home_score", "away_score"] + \
           [c for c in NFL_FEATURES if c not in ("home", "away")]
    base = f[keep].rename(columns={"game_id": "opp_id"})
    a = base.assign(selection=sel_a, dec_odds=market.american_to_decimal(f[odds_a]),
                    won=np.where(push, 0, won_a).astype(int), push=push.astype(int))
    b = base.assign(selection=sel_b, dec_odds=market.american_to_decimal(f[odds_b]),
                    won=np.where(push, 0, ~won_a & ~push).astype(int), push=push.astype(int))
    return pd.concat([a, b], ignore_index=True)


def nfl_spread(f):
    resid = (f["home_score"] - f["away_score"]) - f["spread_line"]
    return _two_rows(f, "home", "away", "home_spread_odds", "away_spread_odds",
                     (resid > 0).to_numpy(), (resid == 0).to_numpy())


def nfl_total(f):
    resid = (f["home_score"] + f["away_score"]) - f["total_line"]
    return _two_rows(f, "over", "under", "over_odds", "under_odds",
                     (resid > 0).to_numpy(), (resid == 0).to_numpy())


def nfl_moneyline(f):
    m = f["home_score"] - f["away_score"]
    return _two_rows(f, "home", "away", "home_ml", "away_ml", (m > 0).to_numpy(), (m == 0).to_numpy())


# ---------------------------------------------------------------- MLB K props
K_FEATURES = ["line", "pitcher", "team", "opponent", "is_home", "mu_model", "p_over_model",
              "p_over_mkt", "p_pitcher", "p_opp", "league_k", "expected_bf", "tr_bf",
              "rest_days", "pitches_last3", "ump_k_factor", "park_k_factor",
              "temp_f", "wind_mph", "condition", "venue", "over_odds_american",
              "under_odds_american", "over_book", "under_book"]


def mlb_k_props():
    from sports.http import fetch
    import io
    pit = pd.read_parquet(DATA / "mlb_pitching.parquet")
    bat = pd.read_parquet(DATA / "mlb_team_batting.parquet")
    st = props.context_features(props.strikeout_features(pit, bat), bat)
    fit = st[st["season"].between(2021, 2023)]
    league_bf = float(fit["bf"].mean())
    r = props.fit_dispersion(props.predict_mean(fit, league_bf).to_numpy(), fit["k"].to_numpy())
    s = st[st["season"] == 2025].copy()
    s["mu_model"] = props.predict_mean(s, league_bf)
    s["expected_bf"] = props.expected_bf(s, league_bf)

    lines = pd.read_csv(io.BytesIO(fetch(K_LINES_URL, binary=True)), low_memory=False,
                        usecols=["game_pk", "pitcher_id", "line", "over_odds", "under_odds",
                                 "over_bookmaker", "under_bookmaker", "fetched_at"])
    lines["fetched_at"] = pd.to_datetime(lines["fetched_at"], utc=True)
    L = lines.merge(s, on=["game_pk", "pitcher_id"], how="inner")
    L = L[L["fetched_at"] < L["start_utc"]].drop_duplicates(["game_pk", "pitcher_id", "line"]).copy()
    L["p_over_model"] = props.p_over(L["mu_model"], L["line"], r)
    L["p_over_mkt"], _, _ = market.devig_two_way(L["over_odds"], L["under_odds"])
    L["over_odds_american"], L["under_odds_american"] = L["over_odds"], L["under_odds"]
    L["over_book"], L["under_book"] = L["over_bookmaker"], L["under_bookmaker"]
    L["opp_id"] = L["game_pk"].astype(str) + "_" + L["pitcher_id"].astype(str) + "_" + L["line"].astype(str)
    L["month"] = pd.to_datetime(L["date"]).dt.month
    L["actual"] = L["k"]
    keep = ["opp_id", "game_pk", "pitcher_id", "season", "month", "date", "start_utc", "actual"] + K_FEATURES
    base = L[keep]
    over = (L["k"] > L["line"]).to_numpy()
    a = base.assign(selection="over", dec_odds=market.american_to_decimal(L["over_odds"]),
                    won=over.astype(int), push=0)
    b = base.assign(selection="under", dec_odds=market.american_to_decimal(L["under_odds"]),
                    won=(~over).astype(int), push=0)
    return pd.concat([a, b], ignore_index=True)


# ---------------------------------------------------------------- MLB totals x weather
MLB_TOTAL_FEATURES = [
    "home", "away", "home_pitcher", "away_pitcher", "venue", "venue_lat", "venue_lon",
    "elevation_ft", "roof", "surface", "day_night", "officials", "temp_f", "wind_mph",
    "wind_dir", "condition", "month", "ou_open", "ou_close", "line_move", "over_odds_american",
    "under_odds_american", "p_over_mkt", "ml_close_home", "ml_close_away"]


def mlb_total():
    from sports import sbr
    mlb = pd.read_parquet(DATA / "mlb_games.parquet")
    j = pd.concat([sbr.join_to_mlb(sbr.season(y), mlb) for y in range(2015, 2022)], ignore_index=True)
    j = j[j["score_match"]].copy()
    j["month"] = j["date"].dt.month
    j["line_move"] = j["ou_close"] - j["ou_open"]
    j["p_over_mkt"], _, _ = market.devig_two_way(j["over_odds"], j["under_odds"])
    j["over_odds_american"], j["under_odds_american"] = j["over_odds"], j["under_odds"]
    j["actual"] = j["home_score"] + j["away_score"]
    j["opp_id"] = j["game_id"].astype(str)
    base = j[["opp_id", "season", "date", "start_utc", "actual"] + MLB_TOTAL_FEATURES]
    resid = j["actual"] - j["ou_close"]
    push = (resid == 0).to_numpy()
    over = base.assign(selection="over", dec_odds=market.american_to_decimal(j["over_odds"]),
                       won=((resid > 0).to_numpy()).astype(int), push=push.astype(int))
    under = base.assign(selection="under", dec_odds=market.american_to_decimal(j["under_odds"]),
                        won=((resid < 0).to_numpy()).astype(int), push=push.astype(int))
    return pd.concat([over, under], ignore_index=True)


# ---------------------------------------------------------------- access
def build(name: str) -> pd.DataFrame:
    if name.startswith("nfl_"):
        return {"nfl_spread": nfl_spread, "nfl_total": nfl_total,
                "nfl_moneyline": nfl_moneyline}[name](_nfl_base())
    if name == "mlb_k_props":
        return mlb_k_props()
    if name == "mlb_total":
        return mlb_total()
    raise KeyError(name)


def load(name: str, split: str) -> pd.DataFrame:
    """split = "dev" | "lockbox". Cached per dataset after the first build."""
    if name not in SPLITS:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(SPLITS)}")
    p = CACHE / f"{name}.parquet"
    devonly = Path(__file__).resolve().parent / "devdata" / f"{name}.parquet"
    col, dev, lock = SPLITS[name]
    if not p.exists() and devonly.exists():
        # A machine with only the committed dev export (e.g. a cloud agent).
        # The lockbox is not in it, by construction.
        if split != "dev":
            raise FileNotFoundError("lockbox data is not available on this machine")
        return pd.read_parquet(devonly)
    CACHE.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        build(name).to_parquet(p, index=False)
    df = pd.read_parquet(p)
    folds = dev if split == "dev" else lock if split == "lockbox" else None
    if folds is None:
        raise ValueError(split)
    return df[df[col].isin(folds)].reset_index(drop=True)


def schema(name: str) -> dict:
    df = load(name, "dev")
    col, dev, lock = SPLITS[name]
    feats = [c for c in df.columns if c not in OUTCOME_COLS]
    return {"dataset": name, "fold_column": col, "dev_folds": dev,
            "rows": len(df), "opportunities": int(df["opp_id"].nunique()),
            "outcome_columns_hidden_from_bet": [c for c in OUTCOME_COLS if c in df.columns],
            "columns": {c: str(df[c].dtype) for c in feats}}
