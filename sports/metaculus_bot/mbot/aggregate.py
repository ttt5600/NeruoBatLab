"""Aggregation of independent samples + calibration guards.

Evidence used (notes/metaculus_bot_tourney.md sec 3, from LessWrong a82q6yd8zKpYk56cF and Bowen1314 RESEARCH.md):
  * 86% of Fall-2025 prize winners aggregated several runs, typically a median of 3-7  -> median here.
  * clipping binaries to ~[0.03, 0.97] was the strongest within-winner differentiator (r +.48) -> clip on by default
    ([0.02, 0.98] as briefed; tighten via MBOT_BINARY_CLIP_LO/HI).
  * extremizing correlated with WORSE results (r -.30); a bot fell 20th -> 63rd after adding logit aggregation +
    extremizing -> extremize factor defaults to 1.0 (off). The knob exists only so a future backtest can test it.
"""
from __future__ import annotations

import math
import statistics


def median_binary(ps: list[float]) -> float:
    if not ps:
        raise ValueError("no samples")
    for p in ps:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"bad probability {p}")
    return float(statistics.median(ps))


def extremize_binary(p: float, factor: float) -> float:
    """Logit-space scaling; factor 1.0 is the identity."""
    if factor == 1.0 or p in (0.0, 1.0):
        return p
    lo = math.log(p / (1 - p)) * factor
    return 1.0 / (1.0 + math.exp(-lo))


def clip_binary(p: float, lo: float, hi: float) -> float:
    return min(max(p, lo), hi)


def median_mc(samples: list[dict[str, float]], options: list[str]) -> dict[str, float]:
    """Per-option median across samples, renormalized to sum 1."""
    if not samples:
        raise ValueError("no samples")
    med = {o: float(statistics.median([s[o] for s in samples])) for o in options}
    tot = sum(med.values())
    if tot <= 0:
        return {o: 1.0 / len(options) for o in options}
    return {o: v / tot for o, v in med.items()}


def floor_mc(probs: dict[str, float], floor: float) -> dict[str, float]:
    """Give every option at least `floor` (scaled down to 0.2/n for long option lists so the floors never take more
    than 20% of the mass), then renormalize so the values sum to exactly 1.0 (the API requires sum == 1)."""
    n = len(probs)
    f = min(floor, 0.2 / n) if n else floor
    raw = {o: max(v, f) for o, v in probs.items()}
    tot = sum(raw.values())
    out = {o: v / tot for o, v in raw.items()}
    # renormalizing can pull a floored option a hair under f; fix by taking the deficit from the largest option
    keys = list(out)
    big = max(keys, key=lambda k: out[k])
    for o in keys:
        if out[o] < f and o != big:
            out[big] -= f - out[o]
            out[o] = f
    out[big] += 1.0 - sum(out.values())
    return out
