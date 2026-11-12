"""Hyped-launch unders (notes/game_release_dose.md): UNDER every in-season NFL/NBA/NHL/CFB game at the CLOSING total on
days 0..6 after a video-game launch cluster whose launch-week revenue H >= params.h_min (default $600M, fixed in the
pre-registration). Frame: loader game_release_hype (dev rows <= 2022-12-31 from SBR/lab; sealed rows 2023-01-01..
2026-10-31 from ESPN lines, real under price). Timing convention identical to edgelab.rules.game_release.
"""
from __future__ import annotations

import pandas as pd

from .game_release import unders


def hyped_unders(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    p = params or {}
    f = frame[frame["H"] >= float(p.get("h_min", 600e6))]
    return unders(f, {"window": int(p.get("window", 7))})
