"""Verify a bet before placing it: EV after fees, edge, and the stake notes/risk.md allows. No orders.

    python scripts/ev_check.py --price 0.96 --fair 0.975 --venue kalshi --tier temp --bankroll 1000
    python scripts/ev_check.py --price -110 --fair 0.55 --venue book --tier judgment --bankroll 1000
    python scripts/ev_check.py --price 2.40 --fair-odds 2.25 2.10 3.40 --pick 0 --tier soccer   # fair from a sharp 1X2 (power de-vig)

price: Kalshi/Polymarket contract price in dollars (0-1), decimal odds (>1, <=50) or American (>=100 or <=-100).
fair : your probability; or --fair-odds = the sharp book's odds for every outcome (power de-vig) and --pick = which one.
Tiers (fractional Kelly, max % of bankroll per bet) mirror notes/risk.md; 'paper' always returns 0.
Rule: refuse anything with EV <= 0 after fees, and anything whose edge is below the tier's minimum.
"""
from __future__ import annotations

import argparse

import numpy as np

TIERS = {  # kelly fraction, cap (% bankroll), minimum edge (EV per $ staked)
    "temp": (0.25, 0.01, 0.005), "temp_x1": (0.25, 0.005, 0.003), "soccer": (0.5, 0.003, 0.02),
    "judgment": (0.125, 0.01, 0.02), "paper": (0.0, 0.0, 1.0)}


def to_cost(price: float, venue: str) -> tuple[float, float]:
    """-> (cost per $1 payout incl. fee, decimal odds)."""
    if 0 < price < 1:
        fee = 0.07 * price * (1 - price) if venue == "kalshi" else (0.01 if venue == "polymarket" else 0.0)
        c = price + fee
        return c, 1 / c
    dec = price if 1 < price <= 50 else (1 + price / 100 if price >= 100 else 1 + 100 / -price)
    return 1 / dec, dec


def power_devig(odds) -> np.ndarray:
    from scipy.optimize import brentq
    r = 1 / np.asarray(odds, float)
    k = brentq(lambda k: (r ** k).sum() - 1, 0.3, 4.0)
    return r ** k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=float, required=True)
    ap.add_argument("--fair", type=float)
    ap.add_argument("--fair-odds", type=float, nargs="+")
    ap.add_argument("--pick", type=int, default=0)
    ap.add_argument("--venue", default="book", choices=["kalshi", "polymarket", "book"])
    ap.add_argument("--tier", default="judgment", choices=list(TIERS))
    ap.add_argument("--bankroll", type=float, default=1000.0)
    a = ap.parse_args()
    p = float(power_devig(a.fair_odds)[a.pick]) if a.fair_odds else a.fair
    if p is None:
        ap.error("give --fair or --fair-odds")
    cost, dec = to_cost(a.price, a.venue)
    ev = p * dec - 1                      # expected return per $1 staked
    kelly = (p * dec - 1) / (dec - 1)     # = (p - c)/(1 - c)
    kf, cap, min_edge = TIERS[a.tier]
    stake = max(0.0, min(kf * kelly, cap)) * a.bankroll if ev > max(0, min_edge) else 0.0
    print(f"fair p {p:.4f} | cost {cost:.4f} (decimal {dec:.3f}) | EV {ev:+.2%} per $ | full Kelly {kelly:+.2%}")
    if ev <= 0:
        print("VERDICT: NO BET -- negative EV after fees")
    elif ev < min_edge:
        print(f"VERDICT: NO BET -- edge {ev:+.2%} below the '{a.tier}' minimum {min_edge:.1%}")
    else:
        print(f"VERDICT: BET ${stake:,.2f} ({stake / a.bankroll:.2%} of ${a.bankroll:,.0f}; tier '{a.tier}' = {kf:g} Kelly, cap {cap:.1%})"
              + (f"  ~{int(stake // a.price)} contracts" if 0 < a.price < 1 and stake else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
