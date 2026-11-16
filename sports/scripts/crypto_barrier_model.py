"""FROZEN barrier models + forward rule for Kalshi monthly crypto max/min one-touch ladders (notes/crypto_barrier.md).

Used identically by the DEV descriptive check (scripts/crypto_barrier.py) and the forward paper logger
(scripts/crypto_barrier_forward.py). Any change = a new trial (sha256 of this file is recorded in the notes).

    d      = ln(B/S) for MAX, ln(S/B) for MIN (> 0 while the barrier is untouched)
    sigma  = DVOL/100 (annualised, 365 d), tau in years
    M0     Gaussian reflection, risk-neutral drift (r = 0), BGK per-minute shift      (diagnostic)
    M1     driftless reflection on a unit-variance Student-t(4) terminal, BGK shift   (PRIMARY)
    M2     empirical day-block bootstrap of hourly (r, u, l) over the trailing 730 d   (second model)
Rule (taker): P_cons = min(P1_X, P2_X); fire iff P_cons - ask_X - fee(ask_X) >= MARGIN, ask in [.05, .95],
spread <= .05, tau >= 2 d.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

BGK = 0.5826
MINUTE_Y = 1.0 / (365 * 24 * 60)
T_NU = 4
MARGIN = 0.05
ASK_LO, ASK_HI = 0.05, 0.95
MAX_SPREAD = 0.05
MIN_TAU_D = 2.0
N_PATHS = 20000
BOOT_DAYS = 730


def dist(kind: str, spot: float, strike: float) -> float:
    return math.log(strike / spot) if kind == "max" else math.log(spot / strike)


def bgk(d: float, sigma: float) -> float:
    return d + BGK * sigma * math.sqrt(MINUTE_Y)


def p_m0(d: float, sigma: float, tau: float) -> float:
    """P(running max of X >= d'), X Brownian with drift m = -sigma^2/2 (MAX). For MIN the mirrored process -ln S has
    drift +sigma^2/2."""
    raise_if_bad(d, sigma, tau)
    dp = bgk(d, sigma)
    s = sigma * math.sqrt(tau)
    return float(_bm_max(dp, s, sigma, tau, -0.5 * sigma ** 2))


def p_m0_min(d: float, sigma: float, tau: float) -> float:
    raise_if_bad(d, sigma, tau)
    dp = bgk(d, sigma)
    s = sigma * math.sqrt(tau)
    return float(_bm_max(dp, s, sigma, tau, +0.5 * sigma ** 2))


def _bm_max(dp, s, sigma, tau, m):
    a = stats.norm.cdf((-dp + m * tau) / s)
    b = math.exp(2 * m * dp / sigma ** 2) * stats.norm.cdf((-dp - m * tau) / s)
    return min(1.0, a + b)


def p_m1(d: float, sigma: float, tau: float) -> float:
    raise_if_bad(d, sigma, tau)
    dp = bgk(d, sigma)
    s = sigma * math.sqrt(tau) * math.sqrt((T_NU - 2) / T_NU)   # scale of a unit-variance t(4) x sigma sqrt(tau)
    return float(min(1.0, 2.0 * stats.t.sf(dp / s, T_NU)))


def raise_if_bad(d, sigma, tau):
    if not (d > 0 and sigma > 0 and tau > 0):
        raise ValueError(f"bad inputs d={d} sigma={sigma} tau={tau}")


def hourly_blocks(ts: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray):
    """Hourly candles (ts = hour START, unix s, sorted, contiguous gaps allowed) -> list of complete UTC days, each a
    (24, 3) array of (r, u, l) relative to the previous hour's close. Days with any missing hour are dropped."""
    ts = np.asarray(ts, np.int64)
    prev_c = np.r_[np.nan, c[:-1]]
    ok_prev = np.r_[False, np.diff(ts) == 3600]
    r = np.log(c / prev_c)
    u = np.log(h / prev_c)
    lo = np.log(l / prev_c)
    day = ts // 86400
    out = []
    for dd in np.unique(day):
        m = (day == dd) & ok_prev
        if m.sum() == 24:
            out.append(np.stack([r[m], u[m], lo[m]], 1))
    return out


def m2_extremes(blocks: list, tau: float, seed: int, n: int = N_PATHS):
    """Bootstrap paths over tau -> (running max of log move, running min of log move), one value per path."""
    if tau <= 0 or len(blocks) < 300:
        raise ValueError(f"bad m2 inputs tau={tau} blocks={len(blocks)}")
    hours = max(1, int(math.ceil(tau * 365 * 24)))
    nd = int(math.ceil(hours / 24))
    rng = np.random.default_rng(seed % (2 ** 32))
    B = np.stack(blocks).astype(np.float32)       # (ndays, 24, 3)
    idx = rng.integers(0, len(B), size=(n, nd))
    P = B[idx].reshape(n, nd * 24, 3)[:, :hours]  # (n, hours, 3)
    cum = np.cumsum(P[:, :, 0], 1)
    before = np.concatenate([np.zeros((n, 1)), cum[:, :-1]], 1)
    return (before + P[:, :, 1]).max(1), (before + P[:, :, 2]).min(1)


def p_m2(ext: tuple, kind: str, d: float) -> float:
    """Share of bootstrap paths whose running max (MAX) / min (MIN) log move reaches d."""
    if d <= 0:
        raise ValueError(f"bad d={d}")
    return float((ext[0] >= d).mean()) if kind == "max" else float((ext[1] <= -d).mean())


def fee(a: float) -> float:
    return 0.07 * a * (1 - a)


def decide(yes_bid: float, yes_ask: float, p1_yes: float, p2_yes: float, tau_days: float) -> dict | None:
    """Frozen taker rule. Returns {'side','ask','p_cons','edge'} or None."""
    if yes_bid is None or yes_ask is None or not (0 < yes_bid < yes_ask < 1):
        return None
    if yes_ask - yes_bid > MAX_SPREAD + 1e-9 or tau_days < MIN_TAU_D:
        return None
    best = None
    for side, ask, pc in (("yes", yes_ask, min(p1_yes, p2_yes)), ("no", 1 - yes_bid, min(1 - p1_yes, 1 - p2_yes))):
        ask = round(ask, 4)
        if not (ASK_LO - 1e-9 <= ask <= ASK_HI + 1e-9):
            continue
        edge = pc - ask - fee(ask)
        if edge >= MARGIN - 1e-12 and (best is None or edge > best["edge"]):
            best = {"side": side, "ask": ask, "p_cons": pc, "edge": edge}
    return best
