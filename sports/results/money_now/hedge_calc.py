"""Hedge a prediction-venue promo credit with the opposite side on Kalshi; report the worst-case locked $.

Model (per promo):
  venue: spend credit B on side X at venue ask q (per $1 contract). n_v = floor(B / (q + fee_v_per_contract)).
    stake-returned (SR): a win pays n_v * $1 cash.         stake-not-returned (SNR): a win pays n_v * $1 - B.
  kalshi: buy N contracts of NOT-X at ask a (= 1 - Kalshi bid for X), taker fee ceil_cents(0.07 * N * a * (1 - a)).
  outcome X:     venue_payout - N * a - fee_k
  outcome not-X: N * (1 - a) - fee_k   (N * $1 payout minus cost)
  N is the integer that maximises min(outcome X, outcome not-X). Venue fees for the credit leg are passed in.
Qualifying cash trade (e.g. "trade $10 to unlock"): buy $Q of X on the venue and hedge the same way; its cost is the
  negative of the same min() with SR payout and real cash.
"""
from __future__ import annotations

import math


def kalshi_fee(n: int, p: float, rate: float = 0.07) -> float:
    return math.ceil(round(rate * n * p * (1 - p) * 100, 6)) / 100


def lock(credit: float, q: float, a: float, sr: bool, fee_v: float = 0.0, cash_cost: float = 0.0) -> dict:
    """credit = venue money spent at price q (credit or cash); a = Kalshi ask for the opposite side.
    fee_v = venue fee per contract; cash_cost = real cash spent on the venue leg (0 for pure credit)."""
    n_v = math.floor(credit / (q + fee_v) + 1e-9)
    payout_x = n_v * 1.0 - (0 if sr else credit)
    best = None
    for n in range(0, int(n_v * 1.2) + 2):
        fk = kalshi_fee(n, a) if n else 0.0
        win_x = payout_x - n * a - fk - cash_cost
        win_not = n * (1 - a) - fk - cash_cost
        m = min(win_x, win_not)
        if best is None or m > best["locked"]:
            best = {"locked": round(m, 2), "N_kalshi": n, "kalshi_cost": round(n * a, 2), "kalshi_fee": fk,
                    "venue_contracts": n_v, "if_X": round(win_x, 2), "if_notX": round(win_not, 2)}
    return best


if __name__ == "__main__":
    # sanity: $100 SR credit on a 0.58 side, Kalshi opposite ask 0.42 -> about +$100 minus spread/fees
    print("SR  100 @0.58 / K 0.42:", lock(100, 0.58, 0.42, sr=True))
    print("SNR 100 @0.58 / K 0.42:", lock(100, 0.58, 0.42, sr=False))
    print("SNR 100 @0.09 / K 0.92:", lock(100, 0.09, 0.92, sr=False))
    print("SR  100 @0.92 / K 0.09:", lock(100, 0.92, 0.09, sr=True))
    print("cash qualifier $10 @0.58 / K 0.42:", lock(10, 0.58, 0.42, sr=True, cash_cost=10))
