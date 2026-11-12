"""Frozen L36 CFB totals model predictions for 2026 games (walk-forward, results only). Copies 2022-25 games_full into
cfb_espn/ and refetches 2026 there (does not touch data/espn/cfb). Output cfb_preds_2026.parquet (event_id, pred_total)."""
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import l36_cfb_sealed as L  # noqa: E402
from sports import espn  # noqa: E402

D = OUT / "cfb_espn"
D.mkdir(exist_ok=True)
for y in (2022, 2023, 2024, 2025):
    if not (D / f"games_full_{y}.parquet").exists():
        shutil.copy(L.ESPN / f"games_full_{y}.parquet", D / f"games_full_{y}.parquet")
L.ESPN = D
if not (D / "games_full_2026.parquet").exists() or "--refetch" in sys.argv:
    L.ESPN_SEASONS = [2026]
    L.fetch_games()
    L.ESPN_SEASONS = [2022, 2023, 2024, 2025, 2026]
P = L.predictions()
P = P[P.season == 2026]
cols = [c for c in ("event_id", "date", "home", "away", "week", "pred_total", "pred_margin", "points", "fbs_game") if c in P.columns]
print(P.columns.tolist())
P[cols].to_parquet(OUT / "cfb_preds_2026.parquet", index=False)
print("2026 preds", len(P), "with pred_total", int(P.pred_total.notna().sum()))
