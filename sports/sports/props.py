"""Pitcher strikeout props: a causal model and the bookkeeping to bet it honestly.

The model is deliberately small and structural, because every free parameter
is a chance to fit noise:

    K ~ NegBin(mean = E[BF] * p,  dispersion fitted on training years)
    p = log5(pitcher K/BF, opponent K/PA, league K/PA)

Every input is computed from games strictly before the start (``as-of``
joins on start time). Shrinkage toward the league rate stops a pitcher with
three starts from being trusted like one with thirty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

WINDOW_DAYS = 365
PRIOR_BF_PITCHER = 150.0   # pseudo-batters toward the league K rate
PRIOR_PA_TEAM = 600.0      # pseudo-PA toward the league K rate
PRIOR_STARTS_BF = 3.0      # pseudo-starts toward the league starter BF
BF_LOOKBACK = 10           # recent starts used for expected batters faced


def _trailing(df: pd.DataFrame, key: str, cols: list[str], days: int) -> pd.DataFrame:
    """Sum ``cols`` over each key's rows in the ``days`` before each row (exclusive)."""
    df = df.sort_values("start_utc")
    out = []
    for _, g in df.groupby(key, sort=False):
        g = g.set_index("start_utc")
        r = g[cols].rolling(f"{days}D", closed="left").sum()
        out.append(r.set_axis(g["row"].to_numpy()))
    return pd.concat(out).sort_index()


def strikeout_features(pitching: pd.DataFrame, batting: pd.DataFrame) -> pd.DataFrame:
    """One row per regular-season start with pre-game features and the outcome."""
    pit = pitching[pitching["game_type"] == "regular"].copy()
    bat = batting[batting["game_type"] == "regular"].copy()
    for d in (pit, bat):
        d["start_utc"] = pd.to_datetime(d["start_utc"], utc=True)

    # League K rate over the trailing window, from team batting (all games).
    lg = bat.groupby("start_utc")[["bat_k", "bat_pa"]].sum().sort_index()
    lg_roll = lg.rolling(f"{WINDOW_DAYS}D", closed="left").sum()
    league_k = (lg_roll["bat_k"] / lg_roll["bat_pa"]).rename("league_k")

    starts = pit[pit["started"]].copy()
    starts["row"] = np.arange(len(starts))

    # Pitcher K/BF from ALL his appearances (relief included) before this start.
    allp = pit.copy()
    allp["row"] = np.arange(len(allp))
    tr = _trailing(allp, "pitcher_id", ["k", "bf"], WINDOW_DAYS)
    allp = allp.join(tr.add_prefix("tr_"), on="row")
    starts = starts.merge(
        allp[["game_pk", "pitcher_id", "tr_k", "tr_bf"]], on=["game_pk", "pitcher_id"], how="left")

    # Expected batters faced: mean of his last N STARTS, shrunk to league starter mean.
    starts = starts.sort_values("start_utc")
    g = starts.groupby("pitcher_id")["bf"]
    starts["bf_sum"] = g.transform(lambda s: s.shift(1).rolling(BF_LOOKBACK, min_periods=1).sum())
    starts["bf_n"] = g.transform(lambda s: s.shift(1).rolling(BF_LOOKBACK, min_periods=1).count())

    # Opponent batting K/PA before this game.
    b = bat.copy()
    b["row"] = np.arange(len(b))
    trb = _trailing(b, "team", ["bat_k", "bat_pa"], WINDOW_DAYS)
    b = b.join(trb.add_prefix("tr_"), on="row")
    starts = starts.merge(
        b[["game_pk", "team", "tr_bat_k", "tr_bat_pa"]].rename(columns={"team": "opponent"}),
        on=["game_pk", "opponent"], how="left")

    starts = starts.merge(league_k, left_on="start_utc", right_index=True, how="left")
    L = starts["league_k"]
    starts["p_pitcher"] = ((starts["tr_k"].fillna(0) + PRIOR_BF_PITCHER * L)
                           / (starts["tr_bf"].fillna(0) + PRIOR_BF_PITCHER))
    starts["p_opp"] = ((starts["tr_bat_k"].fillna(0) + PRIOR_PA_TEAM * L)
                       / (starts["tr_bat_pa"].fillna(0) + PRIOR_PA_TEAM))
    return starts.reset_index(drop=True)


def log5(p_batter_side: np.ndarray, p_pitcher: np.ndarray, league: np.ndarray) -> np.ndarray:
    """Odds-ratio combination of a pitcher rate and an opponent rate."""
    a = p_pitcher * p_batter_side / league
    b = (1 - p_pitcher) * (1 - p_batter_side) / (1 - league)
    return a / (a + b)


def expected_bf(starts: pd.DataFrame, league_bf: float) -> pd.Series:
    return ((starts["bf_sum"].fillna(0) + PRIOR_STARTS_BF * league_bf)
            / (starts["bf_n"].fillna(0) + PRIOR_STARTS_BF))


def predict_mean(starts: pd.DataFrame, league_bf: float) -> pd.Series:
    p = log5(starts["p_opp"].to_numpy(), starts["p_pitcher"].to_numpy(),
             starts["league_k"].to_numpy())
    return pd.Series(p * expected_bf(starts, league_bf).to_numpy(), index=starts.index)


def fit_dispersion(mu: np.ndarray, k: np.ndarray) -> float:
    """MLE of the NegBin size parameter r (variance = mu + mu^2 / r)."""
    def nll(log_r):
        r = np.exp(log_r)
        return -stats.nbinom.logpmf(k, r, r / (r + mu)).sum()
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(nll, bounds=(np.log(1.0), np.log(1e4)), method="bounded")
    return float(np.exp(res.x))


def p_over(mu: np.ndarray, line: np.ndarray, r: float) -> np.ndarray:
    """P(K > line) for half-point lines, under NegBin(mu, r)."""
    mu = np.asarray(mu, float)
    return stats.nbinom.sf(np.floor(np.asarray(line, float)), r, r / (r + mu))
