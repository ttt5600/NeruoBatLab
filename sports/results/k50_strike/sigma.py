"""Per-league predictive SD for the ladder-optimal variant, fit on the PRIOR season only (no dev outcome):
sd(final - book OPEN) for totals and home margin, regular season, DraftKings preferred else first provider."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

E = Path(__file__).resolve().parents[2] / "data/espn"
out = {}
for lg, yr, gf in (("nba", 2026, "games_2026"), ("nfl", 2025, "games_2025"), ("cfb", 2025, "games_full_2025")):
    G = pd.read_parquet(E / lg / f"{gf}.parquet")
    G = G[(G.season_type.astype(str).str.contains("regular", case=False)) & G.home_score.notna()]
    Ls = pd.read_parquet(E / lg / (f"lines_full_{yr}.parquet" if lg == "cfb" else f"lines_{yr}.parquet"))
    Ls["pri"] = (~Ls.provider.astype(str).str.contains("Draft", case=False)).astype(int)
    Ls = Ls.sort_values("pri").drop_duplicates("event_id")
    d = G.merge(Ls, on="event_id")
    tot = (d.home_score + d.away_score - d.total_open).dropna()
    mar = (d.home_score - d.away_score + d.home_spread_open).dropna()
    out[lg] = dict(season=yr, n_total=len(tot), sd_total=round(float(tot.std()), 2), bias_total=round(float(tot.mean()), 2),
                   n_spread=len(mar), sd_spread=round(float(mar.std()), 2), providers=d.provider.value_counts().head(3).to_dict())
print(json.dumps(out, indent=1))
json.dump(out, open(Path(__file__).resolve().parent / "sigma.json", "w"), indent=1)
