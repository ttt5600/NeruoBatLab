"""FROZEN CB2 (Deribit smile barrier) + CB3 (MAX-rung NO structural rule) for Kalshi monthly crypto max/min ladders.

Prereg: notes/crypto_barrier_v2.md (forward only). Any change = a new trial (sha256 of this file is in the notes).

    smile_from_summary(rows, now)  Deribit get_book_summary_by_currency(kind=option) -> {T_years: (k[], w[])}
    w_at(smile, k, tau)            total variance at log-moneyness k, horizon tau (linear in k, then linear in T)
    p_ms(kind, S, B, smile, tau)   MAX: 2 D_call + C/B ; MIN: 2 D_put - Put/B ; Black with F = S (r = 0); BGK shift
    decide_cb3(...)                NO on MAX rungs with mid in [.05,.25] and M1 < mid/2
CB2's decision = crypto_barrier_model.decide(yes_bid, yes_ask, p_ms, p_m2, tau_d) (CB1's frozen rule, M1 -> MS).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

import crypto_barrier_model as M

YEAR_S = 365 * 86400
MIN_POINTS = 5
FD_H = 1e-3
CB3_MID_LO, CB3_MID_HI = 0.05, 0.25
CB3_RATIO = 0.5
_MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def expiry_ts(code: str) -> float:
    """'27NOV26' -> unix seconds of 08:00 UTC that day."""
    import calendar
    d, mon, yy = int(code[:-5]), _MON[code[-5:-2]], 2000 + int(code[-2:])
    return calendar.timegm((yy, mon, d, 8, 0, 0)) * 1.0


def smile_from_summary(rows: list[dict], now: float) -> dict:
    by = {}
    for r in rows:
        p = r.get("instrument_name", "").split("-")
        if len(p) != 4:
            continue
        iv, bid, F = r.get("mark_iv"), r.get("bid_price"), r.get("underlying_price")
        if not iv or iv <= 0 or not bid or bid <= 0 or not F or F <= 0:
            continue
        K, cp = float(p[2]), p[3]
        if (K < F and cp != "P") or (K >= F and cp != "C"):
            continue                                   # OTM only
        T = (expiry_ts(p[1]) - now) / YEAR_S
        if T <= 0:
            continue
        by.setdefault(T, []).append((math.log(K / F), (iv / 100.0) ** 2 * T))
    out = {}
    for T, pts in by.items():
        if len(pts) < MIN_POINTS:
            continue
        pts.sort()
        k = np.array([a for a, _ in pts])
        w = np.array([b for _, b in pts])
        k, ix = np.unique(k, return_index=True)
        out[T] = (k, w[ix])
    if not out:
        raise ValueError("empty smile")
    return dict(sorted(out.items()))


def _sig2_at(kw: tuple, k: float, T: float) -> float:
    ks, ws = kw
    return float(np.interp(k, ks, ws)) / T          # np.interp is flat beyond the ends -> flat sigma in the wings


def w_at(smile: dict, k: float, tau: float) -> float:
    Ts = list(smile)
    if tau <= Ts[0]:
        return _sig2_at(smile[Ts[0]], k, Ts[0]) * tau
    if tau >= Ts[-1]:
        return _sig2_at(smile[Ts[-1]], k, Ts[-1]) * tau
    i = int(np.searchsorted(Ts, tau))
    T1, T2 = Ts[i - 1], Ts[i]
    w1, w2 = _sig2_at(smile[T1], k, T1) * T1, _sig2_at(smile[T2], k, T2) * T2
    return w1 + (w2 - w1) * (tau - T1) / (T2 - T1)


def _black(S: float, K: float, w: float, call: bool) -> float:
    sw = math.sqrt(w)
    d1 = (math.log(S / K) + 0.5 * w) / sw
    d2 = d1 - sw
    if call:
        return S * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)
    return K * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def _px(S, K, smile, tau, call):
    return _black(S, K, w_at(smile, math.log(K / S), tau), call)


def p_ms(kind: str, S: float, B: float, smile: dict, tau: float) -> tuple[float, float]:
    """-> (P(touch), sigma at the barrier). Raises on bad inputs."""
    d = M.dist(kind, S, B)
    if not (d > 0 and tau > 0):
        raise ValueError(f"bad inputs d={d} tau={tau}")
    sig_b = math.sqrt(w_at(smile, math.log(B / S), tau) / tau)
    shift = M.BGK * sig_b * math.sqrt(M.MINUTE_Y)
    Bp = B * math.exp(shift if kind == "max" else -shift)
    h = FD_H * Bp
    if kind == "max":
        C = _px(S, Bp, smile, tau, True)
        D = -(_px(S, Bp + h, smile, tau, True) - _px(S, Bp - h, smile, tau, True)) / (2 * h)
        p = 2 * D + C / Bp
    else:
        P = _px(S, Bp, smile, tau, False)
        D = (_px(S, Bp + h, smile, tau, False) - _px(S, Bp - h, smile, tau, False)) / (2 * h)
        p = 2 * D - P / Bp
    return float(min(1.0, max(0.0, p))), sig_b


def decide_cb3(kind: str, yes_bid, yes_ask, p1_yes, tau_days: float) -> dict | None:
    """Frozen CB3 rule. Returns {'side','ask','mid','p1'} or None."""
    if kind != "max" or p1_yes is None or yes_bid is None or yes_ask is None or not (0 < yes_bid < yes_ask < 1):
        return None
    if yes_ask - yes_bid > M.MAX_SPREAD + 1e-9 or tau_days < M.MIN_TAU_D:
        return None
    mid = (yes_bid + yes_ask) / 2
    if not (CB3_MID_LO - 1e-9 <= mid <= CB3_MID_HI + 1e-9) or not (p1_yes < CB3_RATIO * mid):
        return None
    ask = round(1 - yes_bid, 4)
    if not (M.ASK_LO - 1e-9 <= ask <= M.ASK_HI + 1e-9):
        return None
    return {"side": "no", "ask": ask, "mid": mid, "p1": p1_yes, "edge": (1 - p1_yes) - ask - M.fee(ask)}
