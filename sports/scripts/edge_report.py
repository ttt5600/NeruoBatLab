"""How much edge is REAL? Honest per-strategy edge estimates from DEV data only.

    python scripts/edge_report.py [--stake 50]     # -> lab/reports/edge_report.csv

Raw dev ROI is the best number we could find after searching, so it overstates
the truth. For each candidate this reports:

  roi_raw      dev walk-forward ROI (same code path as lab/run.py dev; no ledger write)
  ci (season)  95% interval from resampling whole SEASONS (games in a season share
               rules, refs, weather regimes -- a per-bet bootstrap is too narrow)
  halves       ROI in the first and second half of the dev seasons: an edge found
               in 2006-19 should show up in both, not just one hot stretch
  roi_tauX     Bayesian shrinkage of roi_raw toward "no edge" (prior mean = -4.5%,
               the vig) with prior sd X of TRUE edges. A rule found by searching is
               pulled back in proportion to its noise. tau 2% ~ typical published
               betting anomalies; tau 5% = generous. roi_tau2 is the planning number.
               (Fitting tau from the 72 scan rules does not work: they reuse the
               same games, so their ROIs are correlated and spread LESS than
               independent noise -- var 0.00083 vs 0.00109 -- which gives tau = 0.)
  $/season     bets per season x stake x roi_shrunk (and the season CI)

Lockbox numbers (already scored by a human) are read from lab/lockbox.jsonl and
shown as-is; this script never scores the lockbox.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from lab import datasets as D
from lab.run import walk_forward

STRATS = ["div_unders", "div_unders_late", "west_coast_night", "bye_fade"]
OUT = ROOT / "lab" / "reports" / "edge_report.csv"


def pnl(t: pd.DataFrame) -> pd.Series:
    t = t[t["stake"] > 0]
    return pd.Series(np.where(t["push"] == 1, 0.0, np.where(t["won"] == 1, t["dec_odds"] - 1, -1.0)) * t["stake"],
                     index=t.index)


def season_boot(t: pd.DataFrame, n=5000, seed=0) -> np.ndarray:
    """ROI under resampling of whole seasons."""
    g = t.groupby("season").agg(p=("profit", "sum"), s=("stake", "sum"))
    idx = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    return g["p"].to_numpy()[idx].sum(1) / g["s"].to_numpy()[idx].sum(1)


PRIOR_MEAN = -0.045
TAUS = (0.02, 0.05)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stake", type=float, default=50.0, help="$ per bet")
    a = ap.parse_args()
    rows = []
    for name in STRATS:
        path = ROOT / "lab" / "strategies" / f"{name}.py"
        ds = next(l.split("=")[1].strip().strip('"\'') for l in path.read_text().splitlines()
                  if l.startswith("DATASET"))
        _, t = walk_forward(path, ds, "dev")
        t = t[t["stake"] > 0].assign(profit=lambda x: pnl(x))
        n, roi = len(t), t["profit"].sum() / t["stake"].sum()
        boots = season_boot(t)
        seasons = sorted(t["season"].unique())
        mid = seasons[len(seasons) // 2]
        h1, h2 = t[t["season"] < mid], t[t["season"] >= mid]
        # sampling variance of ROI from the season bootstrap (captures clustering)
        se2 = np.var(boots)
        shr = {tau: PRIOR_MEAN + tau**2 / (tau**2 + se2) * (roi - PRIOR_MEAN) for tau in TAUS}
        per_season = n / len(seasons)
        rows.append({
            "strategy": name, "dataset": ds, "bets": n, "bets_per_season": round(per_season, 1),
            "win_rate": float((t["won"] == 1).mean()),
            "breakeven": float((1 / t["dec_odds"]).mean()),
            "roi_raw": roi, "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
            f"roi_{seasons[0]}_{mid - 1}": h1["profit"].sum() / max(h1["stake"].sum(), 1),
            f"roi_{mid}_{seasons[-1]}": h2["profit"].sum() / max(h2["stake"].sum(), 1),
            "seasons_pos": f"{int((t.groupby('season')['profit'].sum() > 0).sum())}/{len(seasons)}",
            **{f"roi_tau{int(tau * 100)}": v for tau, v in shr.items()},
            "usd_season": per_season * a.stake * shr[0.02],
            "usd_season_lo": per_season * a.stake * np.percentile(boots, 2.5),
            "usd_season_hi": per_season * a.stake * np.percentile(boots, 97.5)})
    r = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n$ figures at ${a.stake:.0f}/bet. usd_season uses roi_tau2; CI is season-resampled.")
    lb = ROOT / "lab" / "lockbox.jsonl"
    if lb.exists():
        print("\nAlready-scored lockbox results (out-of-sample, human-run):")
        for l in lb.read_text().splitlines():
            x = json.loads(l)
            print(f"  {x['strategy']:<28} {x['dataset']:<12} bets {x['bets']:>5}  ROI {x['roi']:+.2%}  "
                  f"p {x['p']:.4f}  units {x['units']:+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
