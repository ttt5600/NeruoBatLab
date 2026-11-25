"""Continuous forecasts: percentiles -> Metaculus CDF, the official validity rules, and CDF aggregation.

The percentile->CDF construction and `standardize_cdf` are a dependency-free port of
forecasting_tools/data_models/numeric_report.py (Metaculus forecasting-tools 0.3.7, MIT License, Copyright (c) 2024
CodexVeritas; full text in LICENSES/forecasting-tools-MIT.txt). The same algorithm is published in the Metaculus API
docs (docs/openapi.yml, "CDF generation details"). tests/test_cdf.py checks parity against the installed package when
it is importable.

Official acceptance rules (docs/openapi.yml, read 2026-11-25), with n = inbound_outcome_count (200 for numeric):
  * len(cdf) == n + 1
  * strictly increasing by at least 0.01 / n per step (0.00005 when n = 200)
  * no step larger than 0.2 * (200 / n)  [standardization guide]; and never larger than 0.59 [ContinuousForecast schema]
  * closed lower bound -> cdf[0] == 0.0, open lower bound -> cdf[0] >= 0.001
  * closed upper bound -> cdf[-1] == 1.0, open upper bound -> cdf[-1] <= 0.999
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .questions import Question

DEFAULT_INBOUND = 200
MAX_PMF_DEFAULT = 0.2
SCHEMA_MAX_STEP = 0.59


@dataclass(frozen=True)
class Spec:
    range_min: float
    range_max: float
    zero_point: float | None
    open_lower: bool
    open_upper: bool
    cdf_size: int = 201
    is_date: bool = False

    @classmethod
    def from_question(cls, q: Question) -> "Spec":
        return cls(float(q.range_min), float(q.range_max), q.zero_point, q.open_lower_bound, q.open_upper_bound,
                   q.cdf_size, q.type == "date")

    @property
    def inbound(self) -> int:
        return self.cdf_size - 1

    @property
    def min_step(self) -> float:
        return 0.01 / self.inbound

    @property
    def max_step(self) -> float:
        return min(MAX_PMF_DEFAULT * (DEFAULT_INBOUND / self.inbound), SCHEMA_MAX_STEP)


def max_pmf_value(cdf_size: int, include_wiggle_room: bool = True) -> float:
    cap = MAX_PMF_DEFAULT * (DEFAULT_INBOUND / (cdf_size - 1))
    return cap * 0.95 if include_wiggle_room else cap


# ---------------------------------------------------------------------------------------------- scaling
def nominal_to_location(value: float, s: Spec) -> float:
    """Real-world value -> position on the CDF x-axis (0 = range_min, 1 = range_max; may fall outside [0, 1])."""
    if s.zero_point is not None:
        deriv_ratio = (s.range_max - s.zero_point) / (s.range_min - s.zero_point)
        if value == s.zero_point:
            value += 1e-10
        return float((np.log((value - s.range_min) * (deriv_ratio - 1) + (s.range_max - s.range_min))
                      - np.log(s.range_max - s.range_min)) / np.log(deriv_ratio))
    return float((value - s.range_min) / (s.range_max - s.range_min))


def location_to_nominal(loc: float, s: Spec) -> float:
    if s.zero_point is None:
        v = s.range_min + (s.range_max - s.range_min) * loc
    else:
        deriv_ratio = (s.range_max - s.zero_point) / (s.range_min - s.zero_point)
        v = s.range_min + (s.range_max - s.range_min) * (deriv_ratio ** loc - 1) / (deriv_ratio - 1)
    if np.isnan(v):
        raise ValueError(f"nominal value is NaN at location {loc}")
    return float(v)


def x_axis(s: Spec) -> list[float]:
    return [location_to_nominal(i / (s.cdf_size - 1), s) for i in range(s.cdf_size)]


def date_to_value(d: str | datetime) -> float:
    if isinstance(d, str):
        d = datetime.fromisoformat(d.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


# ---------------------------------------------------------------------------------------------- percentiles -> CDF
def _check_declared(pcts: list[tuple[float, float]], s: Spec, strict: bool = True) -> list[tuple[float, float]]:
    if len(pcts) < 2:
        raise ValueError("need at least 2 percentiles")
    for p, v in pcts:
        if not 0 <= p <= 1 or np.isnan(p) or np.isnan(v):
            raise ValueError(f"bad percentile ({p}, {v})")
    for (p0, v0), (p1, v1) in zip(pcts, pcts[1:]):
        if p0 >= p1:
            raise ValueError("percentiles must be strictly increasing")
        if v0 > v1:
            raise ValueError("values must be non-decreasing")
    if s.zero_point is not None:
        if s.range_min <= s.zero_point:
            raise ValueError("lower bound must exceed the zero point")
        for _, v in pcts:
            if v < s.zero_point:
                raise ValueError(f"value {v} below zero point {s.zero_point}")
    if not strict:
        return pcts
    for (p0, _), (p1, _) in zip(pcts, pcts[1:]):
        if abs(p1 - p0) < 5e-05:
            raise ValueError("percentiles too close together")
    rng = s.range_max - s.range_min
    if not any(s.range_min - 0.25 * rng <= v <= s.range_max + 0.25 * rng for _, v in pcts):
        raise ValueError("no declared percentile within the question range +/- 25%")
    if any(v < s.range_min - 2 * rng or v > s.range_max + 2 * rng for _, v in pcts):
        raise ValueError("some declared percentiles far exceed the question bounds")
    # nudge repeated values apart (forecasting-tools _check_and_update_repeating_values)
    counts = Counter(v for _, v in pcts)
    out = []
    for p, v in pcts:
        if counts[v] <= 1:
            out.append((p, v))
        elif s.range_min < v < s.range_max:
            out.append((p, v - (1 - p) * 1e-6))
        elif v >= s.range_max:
            out.append((p, s.range_max + 1e-10 * p))
        else:
            out.append((p, s.range_min - 1e-10 * (1 - p)))
    return out


def _add_bounds(pcts: list[tuple[float, float]], s: Spec) -> list[tuple[float, float]]:
    d = {p * 100: v for p, v in pcts}
    pmax, pmin = max(d), min(d)
    rng = abs(s.range_max - s.range_min)
    buffer = 1 if rng > 100 else 0.01 * rng
    for k, v in list(d.items()):
        if not s.open_lower and v <= s.range_min + buffer:
            d[k] = s.range_min + buffer
        if not s.open_upper and v >= s.range_max - buffer:
            d[k] = s.range_max - buffer
    if s.open_upper:
        if s.range_max > d[pmax]:
            d[100 - 0.5 * (100 - pmax)] = s.range_max
    else:
        d[100] = s.range_max
    if s.open_lower:
        if s.range_min < d[pmin]:
            d[0.5 * pmin] = s.range_min
    else:
        d[0] = s.range_min
    return [(k / 100, v) for k, v in sorted(d.items())]


def _cdf_at(loc: float, points: list[tuple[float, float]]) -> float:
    prev = points[0]
    for cur in points[1:]:
        if prev[0] - 1e-10 <= loc <= cur[0] + 1e-10:
            r = prev[1] + (cur[1] - prev[1]) * (loc - prev[0]) / (cur[0] - prev[0])
            if np.isnan(r):
                raise ValueError(f"NaN cdf at {loc}")
            return float(r)
        prev = cur
    raise ValueError(f"cdf location {loc} not covered by the declared percentiles")


def percentiles_to_cdf(pcts: list[tuple[float, float]], s: Spec, standardize: bool = True) -> list[float]:
    """Declared (percentile in (0,1), nominal value) pairs -> a CDF of length s.cdf_size, linearly interpolated in
    the question's (possibly log) scale, then standardized to the official rules."""
    pcts = sorted(pcts)
    if s.zero_point is not None:   # an LLM value at/below the zero point would hit log(<=0); clamp just above it
        eps = 1e-6 * (s.range_min - s.zero_point)
        pcts = [(p, max(v, s.zero_point + eps)) for p, v in pcts]
    pcts = _check_declared(pcts, s)
    bounded = _add_bounds(pcts, s)
    points = [(nominal_to_location(v, s), p) for p, v in bounded]
    cdf = [_cdf_at(i / (s.cdf_size - 1), points) for i in range(s.cdf_size)]
    return standardize_cdf(cdf, s) if standardize else cdf


def standardize_cdf(cdf: list[float] | np.ndarray, s: Spec) -> list[float]:
    """Metaculus standardization: no mass outside closed bounds, >= 0.1% outside open bounds, a 1% uniform floor,
    and per-step mass capped (official cap with 5% wiggle room, and below the schema's 0.59)."""
    cdf = np.asarray(cdf, dtype=float).copy()
    lo_to = 0.0 if s.open_lower else cdf[0]
    hi_to = 1.0 if s.open_upper else cdf[-1]
    mass = hi_to - lo_to
    n = len(cdf)
    for i in range(n):
        F = (cdf[i] - lo_to) / mass
        loc = i / (n - 1)
        if s.open_lower and s.open_upper:
            cdf[i] = 0.988 * F + 0.01 * loc + 0.001
        elif s.open_lower:
            cdf[i] = 0.989 * F + 0.01 * loc + 0.001
        elif s.open_upper:
            cdf[i] = 0.989 * F + 0.01 * loc
        else:
            cdf[i] = 0.99 * F + 0.01 * loc
    pmf = np.diff(cdf, prepend=0, append=1)
    cap = min(max_pmf_value(n), SCHEMA_MAX_STEP * 0.95)

    def cap_pmf(scale: float) -> np.ndarray:
        return np.concatenate([pmf[:1], np.minimum(cap, scale * pmf[1:-1]), pmf[-1:]])

    def capped_sum(scale: float) -> float:
        return float(cap_pmf(scale).sum())

    lo = hi = scale = 1.0
    expansions = 0
    while capped_sum(hi) < 1.0:
        if expansions >= 100:
            raise ValueError("could not standardize CDF (non-increasing input?)")
        hi *= 1.2
        expansions += 1
    for _ in range(100):
        scale = 0.5 * (lo + hi)
        sm = capped_sum(scale)
        if sm < 1.0:
            lo = scale
        else:
            hi = scale
        if sm == 1.0 or (hi - lo) < 2e-5:
            break
    pmf = cap_pmf(scale)
    pmf[1:-1] *= (cdf[-1] - cdf[0]) / pmf[1:-1].sum()
    out = np.round(np.cumsum(pmf)[:-1], 10)
    return out.tolist()


# ---------------------------------------------------------------------------------------------- validation
def validate_cdf(cdf: list[float], s: Spec, tol: float = 1e-8) -> list[str]:
    """Return the list of official-rule violations (empty list = the API should accept it)."""
    errs: list[str] = []
    if len(cdf) != s.cdf_size:
        return [f"length {len(cdf)} != {s.cdf_size}"]
    a = np.asarray(cdf, dtype=float)
    if not np.all(np.isfinite(a)):
        return ["non-finite values"]
    if a.min() < 0 or a.max() > 1:
        errs.append("values outside [0, 1]")
    steps = np.diff(a)
    if steps.min() < s.min_step - tol:
        errs.append(f"min step {steps.min():.3g} < {s.min_step:.3g} (must increase by >= 0.01/n)")
    if steps.max() > s.max_step + tol:
        errs.append(f"max step {steps.max():.3g} > {s.max_step:.3g}")
    if s.open_lower:
        if a[0] < 0.001 - tol:
            errs.append(f"open lower bound needs cdf[0] >= 0.001 (got {a[0]:.6g})")
    elif a[0] != 0.0:
        errs.append(f"closed lower bound needs cdf[0] == 0.0 (got {a[0]!r})")
    if s.open_upper:
        if a[-1] > 0.999 + tol:
            errs.append(f"open upper bound needs cdf[-1] <= 0.999 (got {a[-1]:.6g})")
    elif a[-1] != 1.0:
        errs.append(f"closed upper bound needs cdf[-1] == 1.0 (got {a[-1]!r})")
    return errs


def repair_cdf(cdf: list[float], s: Spec) -> list[float]:
    """Re-standardize a CDF that fails validation (e.g. a pointwise median), keeping its shape."""
    if not validate_cdf(cdf, s):
        return list(cdf)
    a = np.maximum.accumulate(np.clip(np.asarray(cdf, dtype=float), 0, 1))
    if not s.open_lower:
        a = a - a[0]
    if not s.open_upper and a[-1] > 0:
        a = a / a[-1]
    return standardize_cdf(a, s)


# ---------------------------------------------------------------------------------------------- aggregation / summaries
def median_cdf(cdfs: list[list[float]], s: Spec) -> list[float]:
    """Pointwise median of CDFs on the shared x-axis (what forecasting-tools does), then repaired if needed. The
    pointwise median of strictly increasing sequences is itself strictly increasing by at least the common minimum
    step, so the repair only triggers on the step-size cap."""
    if not cdfs:
        raise ValueError("no CDFs to aggregate")
    for c in cdfs:
        if len(c) != s.cdf_size:
            raise ValueError("CDFs of different length")
    med = np.median(np.asarray(cdfs, dtype=float), axis=0).tolist()
    return repair_cdf(med, s)


def cdf_to_pmf(cdf: list[float]) -> np.ndarray:
    """pmf[0] = P(below range), pmf[1..n] = inbound buckets, pmf[n+1] = P(above range) (Metaculus get_pmf layout)."""
    return np.diff(np.asarray(cdf, dtype=float), prepend=0.0, append=1.0)


def quantile(cdf: list[float], s: Spec, q: float) -> float:
    """Nominal value at CDF height q (linear in location space). Clamped to the range ends."""
    a = np.asarray(cdf, dtype=float)
    locs = np.linspace(0, 1, s.cdf_size)
    if q <= a[0]:
        return location_to_nominal(0.0, s)
    if q >= a[-1]:
        return location_to_nominal(1.0, s)
    return location_to_nominal(float(np.interp(q, a, locs)), s)


def readable(cdf: list[float], s: Spec, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> str:
    parts = []
    for q in qs:
        v = quantile(cdf, s, q)
        if s.is_date:
            parts.append(f"p{int(q * 100)}={datetime.fromtimestamp(v, tz=timezone.utc).date().isoformat()}")
        else:
            parts.append(f"p{int(q * 100)}={v:.6g}")
    return ", ".join(parts)
