"""OT1b: OT1 (edgelab/rules/opt_tails.py:taker, unchanged) with a MONOTONE, arbitrage-free options-implied P(S > K).
Prereg notes/ot1b_monotone.md (2026-11-15, forward only from 2026-11-16; no dev scoring). Frozen: any edit = new trial.

Input: the frozen OT1 smile tuple sm = (F_T, tau_D, xs, sig) from scripts/options_implied_tails.py:smile and tauK (yrs).
  1. 401-point grid uniform in log(K/F_T) over xs mapped to the Kalshi horizon: log(K/F_T) in xs[[0,-1]]*sqrt(tauK/tauD).
  2. Black-76 calls C_i at sigma = interp(x_i) (x = log(K/F_T) * sqrt(tauD/tauK), linear in xs, as OT1).
  3. call-spread probabilities q_j = -(C_{j+1}-C_j)/(K_{j+1}-K_j) at grid midpoints.
  4. weighted PAVA non-increasing in K (weights = K spacing; = slope of C's greatest convex minorant), clip [0, 1].
  5. P(S > K) = linear interpolation over the midpoints; outside -> NaN (no extrapolation).
The trading rule is OT1's taker, re-exported here so the trial id pins this module's source.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from .opt_tails import taker  # noqa: F401  (unchanged OT1 rule; this trial changes only the p_yes input)

GRID_N = 401


def _b76_call(F, K, sig, tau):
    sig = np.maximum(sig, 1e-6)
    sd = sig * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    return F * norm.cdf(d1) - K * norm.cdf(d1 - sd)


def pava_nonincreasing(y, w):
    """Weighted isotonic regression, non-increasing."""
    y, w = np.asarray(y, float), np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:          # violation of non-increasing -> pool
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            ww = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / ww); wts.append(ww); cnt.append(c1 + c2)
    return np.repeat(vals, cnt)


def p_above_grid(sm, tauK):
    """-> (K_mid, P(S > K_mid)) monotone non-increasing in [0, 1], or None."""
    FT, tauD, xs, sig = sm
    a = math.sqrt(tauK / tauD)
    lk = np.linspace(xs[0] * a, xs[-1] * a, GRID_N)
    K = FT * np.exp(lk)
    s = np.interp(lk / a, xs, sig)
    C = _b76_call(FT, K, s, tauK)
    dK = np.diff(K)
    q = -np.diff(C) / dK
    if not np.isfinite(q).all():
        return None
    P = np.clip(pava_nonincreasing(q, dK), 0.0, 1.0)
    return (K[:-1] + K[1:]) / 2, P


def p_above_mono(K, grid):
    if grid is None:
        return np.nan
    km, P = grid
    if not (np.isfinite(K) and km[0] <= K <= km[-1]):
        return np.nan
    return float(np.interp(K, km, P))
