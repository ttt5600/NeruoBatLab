"""Venue fee schedules (GOAL.md 'Costs'). Sources: notes/EDGE_MATH.md step 4, notes/MODELS_GUIDE.md 'Kalshi taker
fee as implemented', notes/execution_tests_2026-10-22.md (fee rounding).

Kalshi taker: 0.07 * C * P * (1 - P) per ORDER, rounded UP to the cent (scripts/kalshi_exec.py fee_total). Direct
members are charged at $0.0001 precision (docs fee_rounding; prod fills looked ~exact, HANDOFF_2026-10-25), so the
schedule takes a `rounding` argument: "cent" (the conservative published rule, per order of C contracts) or "exact".
Kalshi maker: 0 on series fee_type 'quadratic' (weather, TSA as checked in their notes); 0.0175 * C * P * (1 - P)
rounded up on 'quadratic_with_maker_fees'. ForecastEx: $0.01 per contract. Sportsbook: the fee is in the odds (0).
"""
from __future__ import annotations

import math

import numpy as np

KALSHI_TAKER = 0.07
KALSHI_MAKER_QWMF = 0.0175
FORECASTEX_PER_CONTRACT = 0.01
PMUS_TAKER = 0.0695


def _ceil_cent(x: np.ndarray) -> np.ndarray:
    return np.ceil(np.round(x * 100, 6)) / 100


def kalshi_fee_per_contract(p, contracts: int = 1, rate: float = KALSHI_TAKER, rounding: str = "cent") -> np.ndarray:
    """Fee per contract for an order of `contracts` contracts at price p (dollars)."""
    p = np.asarray(p, float)
    raw = rate * contracts * p * (1 - p)
    if rounding == "exact":
        return raw / contracts
    if rounding != "cent":
        raise ValueError(rounding)
    return _ceil_cent(raw) / contracts


def fee_per_contract(schedule: str, p, contracts: int = 1, rounding: str = "cent") -> np.ndarray:
    p = np.asarray(p, float)
    if schedule == "kalshi_taker":
        return kalshi_fee_per_contract(p, contracts, KALSHI_TAKER, rounding)
    if schedule == "kalshi_maker_quadratic":
        return np.zeros_like(p)
    if schedule == "kalshi_maker_qwmf":
        return kalshi_fee_per_contract(p, contracts, KALSHI_MAKER_QWMF, rounding)
    if schedule == "pmus_taker":   # Polymarket US taker 0.0695*C*p*(1-p) (docs.polymarket.us/fees.md, 2026-10-28)
        return kalshi_fee_per_contract(p, contracts, PMUS_TAKER, rounding)
    if schedule == "forecastex":
        return np.full_like(p, FORECASTEX_PER_CONTRACT)
    if schedule in ("book", "none"):
        return np.zeros_like(p)
    raise KeyError(f"unknown fee schedule {schedule!r}")


DEFAULT_SCHEDULE = {"taker": "kalshi_taker", "maker_tape": "kalshi_maker_quadratic", "book_quote": "book",
                    "maker_fill": "kalshi_maker_quadratic"}   # maker_fill = a REAL prod maker fill (forward adapters)


def check_example() -> None:
    # 0.07 x 0.9535 x 0.0465 = 0.0031 (EDGE_MATH E1 worked example); 1-lot rounds up to 1c.
    assert math.isclose(float(kalshi_fee_per_contract(0.9535, 1, rounding="exact")), 0.0031036, rel_tol=1e-3)
    assert float(kalshi_fee_per_contract(0.9535, 1)) == 0.01
    assert math.isclose(float(kalshi_fee_per_contract(0.50, 100)), 0.0175)
