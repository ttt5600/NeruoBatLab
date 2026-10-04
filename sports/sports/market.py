"""Odds arithmetic and the scoring rules used to compare a model to the market.

The market is the benchmark, the way buy-and-hold was in ``quant``. A model
that beats a coin flip has learned that good teams win -- which the closing
line already knows. The only comparison with content is model vs. market on
the same games, scored by a proper scoring rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def american_to_prob(ml) -> np.ndarray:
    """Implied probability of an American price, vig included. -150 -> 0.600, +130 -> 0.435."""
    ml = np.asarray(ml, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))


def american_to_decimal(ml) -> np.ndarray:
    ml = np.asarray(ml, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, 1.0 + 100.0 / -ml, 1.0 + ml / 100.0)


def devig_two_way(ml_a, ml_b) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Proportional devig of a two-sided market. Returns (p_a, p_b, overround).

    Proportional is the simplest method and slightly overstates longshots
    (the favourite-longshot bias lives in the vig). Good enough to benchmark
    against; refine to Shin or power devig if a result hinges on longshots.
    """
    qa, qb = american_to_prob(ml_a), american_to_prob(ml_b)
    s = qa + qb
    return qa / s, qb / s, s - 1.0


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def paired_logloss_bootstrap(p_model, p_market, y, n_boot: int = 2000,
                             block: int = 1, seed: int = 0) -> dict:
    """Bootstrap the per-game log-loss difference (model - market).

    Negative = model better. Games are resampled in pairs so both forecasts
    always face the same outcomes; that pairing is what lets a small edge
    show up at all, because most of each forecast's error is shared.
    """
    p_model, p_market, y = (np.asarray(a, float) for a in (p_model, p_market, y))
    ok = ~(np.isnan(p_model) | np.isnan(p_market) | np.isnan(y))
    p_model, p_market, y = p_model[ok], p_market[ok], y[ok]

    def per_game(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d = per_game(p_model) - per_game(p_market)
    rng = np.random.default_rng(seed)
    n = len(d)
    nb = int(np.ceil(n / block))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        boots[i] = d[idx].mean()
    return {
        "n": int(n),
        "observed_diff": float(d.mean()),
        "ci95": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        "p_model_not_better": float(np.mean(boots >= 0)),
    }


def home_result(games: pd.DataFrame) -> pd.Series:
    """1 home win, 0 away win, NaN for ties (scored as neither)."""
    m = games["home_score"] - games["away_score"]
    return pd.Series(np.where(m > 0, 1.0, np.where(m < 0, 0.0, np.nan)), index=games.index)
