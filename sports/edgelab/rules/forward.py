"""Forward-only strategies whose logs do not yet produce settled, priced, time-stamped bet rows.

unwired() returns zero bets (fail closed: n = 0 -> no verdict can pass) and is the registered rule until a real
adapter is written -- which is a new rule module hash, i.e. a NEW trial, by design.
"""
from __future__ import annotations

import pandas as pd

from ..schema import REQUIRED


def unwired(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=[c for c in REQUIRED if c != "outcome"])
