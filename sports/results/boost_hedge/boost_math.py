"""boost_hedge valuation helpers (pure arithmetic; no network). Run: python3 boost_math.py

Kalshi taker fee = ceil-to-cent(0.07 * M * a * (1 - a)). Venue price q (what the user pays per $1 contract on the venue).
All locks are the WORST-CASE outcome after hedging (both outcomes equalised, then the min is reported).
"""
from __future__ import annotations

import math


def kfee(M: float, a: float) -> float:
    return math.ceil(round(0.07 * M * a * (1 - a) * 100, 6)) / 100


def profit_boost_lock(S: float, q: float, b: float, a: float, venue_fee: float = 0.0, boost_as_credit: bool = False,
                      credit_conv: float = 0.95) -> dict:
    """Cash stake S on the venue at price q, winnings (profit) boosted by fraction b; hedge M opposite on Kalshi at ask a.
    If the boosted part is paid as a non-cash credit, it is valued at credit_conv per $ (SR credit hedged ~0.95)."""
    N = S / q
    base = N * (1 - q)
    extra = base * b * (credit_conv if boost_as_credit else 1.0)
    W = base + extra - venue_fee                     # venue side wins: profit on the venue (fee paid at entry)
    M = base + extra + S                             # equalise: W' - fee - M a = -S - fee + M(1-a) -> M = W' + S
    M = math.ceil(M)                                 # whole Kalshi contracts
    f = kfee(M, a)
    win_v = W - M * a - f
    win_k = -S - venue_fee + M * (1 - a) - f
    return {"S": S, "q": q, "b": b, "a": a, "N_venue": round(N, 2), "M_kalshi": M, "kalshi_capital": round(M * a + f, 2),
            "lock": round(min(win_v, win_k), 2), "if_venue_wins": round(win_v, 2), "if_kalshi_wins": round(win_k, 2),
            "unhedged_EV_at_fair": round(S * (1 - q) * b * (credit_conv if boost_as_credit else 1.0) - venue_fee, 2)}


def price_boost_lock(N: int, q_boost: float, a: float, venue_fee_per_ct: float = 0.0) -> dict:
    """Venue sells N contracts at a boosted (reduced) price q_boost; hedge N opposite on Kalshi at ask a."""
    f = kfee(N, a)
    lock = N * (1 - q_boost - a) - f - N * venue_fee_per_ct
    return {"N": N, "q_boost": q_boost, "a": a, "lock": round(lock, 2), "capital": round(N * (q_boost + a) + f, 2)}


if __name__ == "__main__":
    print("profit boost, $25 stake, fair 1c-spread market (a = 1 - q + 0.01):")
    for q in (0.2, 0.35, 0.5, 0.65, 0.8):
        for b in (0.25, 0.5, 1.0):
            r = profit_boost_lock(25, q, b, round(1 - q + 0.01, 2))
            print(f"  q {q:.2f} b {b:.2f}: lock ${r['lock']:+.2f} (Kalshi capital ${r['kalshi_capital']:.0f}; "
                  f"unhedged EV ${r['unhedged_EV_at_fair']:+.2f})")
