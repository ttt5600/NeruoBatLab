"""Child process: load one strategy file, fit on train, stake the slate."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    strategy, td = Path(sys.argv[1]), Path(sys.argv[2])
    train = pd.read_parquet(td / "train.parquet")
    slate = pd.read_parquet(td / "slate.parquet")
    spec = importlib.util.spec_from_file_location("strategy", strategy)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.fit(train.copy())
    stakes = pd.Series(mod.bet(model, slate.copy()))
    if len(stakes) != len(slate):
        raise ValueError(f"bet() returned {len(stakes)} stakes for {len(slate)} rows")
    pd.DataFrame({"stake": stakes.to_numpy(dtype=float)}).to_parquet(td / "stakes.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
