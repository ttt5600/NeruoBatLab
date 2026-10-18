"""MLB favourite-longshot bias at the OPEN vs the CLOSE (SBRO moneylines 2010-2021).

    python scripts/mlb_ml_open.py [--ledger]

Literature: the bias is present in MLB opening odds and gone by the close (Harvard DASH
thesis snippet, see reports/Sports betting edge research.md). Pre-registered 2026-10-17:
  DEV = 2010-2018; SEALED = 2019-2021 (never loaded here)
  M1 fav60_open  : back the side whose OPEN moneyline implies >= 0.60 (raw, with vig), at the open price
  M2 fav60_close : same rule at the CLOSE price (control)
  2 trials on dataset mlb_ml_sbr -> bar p < 0.025, bootstrap over games.
Outcome from SBR final scores (ties impossible in MLB). Postponed/blank prices dropped.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sports import sbr
from sports.market import american_to_decimal

LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "mlb_ml_sbr"
DEV = range(2010, 2019)


def frame() -> pd.DataFrame:
    g = pd.concat([sbr.season(y) for y in DEV], ignore_index=True)
    g = g.dropna(subset=["away_runs", "home_runs", "ml_open_away", "ml_open_home", "ml_close_away", "ml_close_home"])
    g = g[(g.away_runs != g.home_runs)]
    # SBR sometimes writes blank / 0 / tiny codes for missing lines: keep real American odds only
    for c in ["ml_open_away", "ml_open_home", "ml_close_away", "ml_close_home"]:
        g = g[g[c].abs() >= 100]
    return g.reset_index(drop=True)


def bets(g: pd.DataFrame, when: str) -> pd.DataFrame:
    rows = []
    for side, opp, won in (("home", "away", g.home_runs > g.away_runs), ("away", "home", g.away_runs > g.home_runs)):
        dec = np.asarray(american_to_decimal(g[f"ml_{when}_{side}"]), float)
        rows.append(pd.DataFrame({"game": g.index, "season": g.season, "dec": dec, "won": won.to_numpy(),
                                  "open_dec": np.asarray(american_to_decimal(g[f"ml_open_{side}"]), float)}))
    b = pd.concat(rows)
    b = b[1 / b.open_dec >= 0.60]  # selection always on the OPEN price, so M1/M2 bet the same games
    b["pnl"] = np.where(b.won, b.dec - 1, -1.0)
    return b


def score(b: pd.DataFrame, rng) -> dict:
    per_game = b.groupby("game")["pnl"].sum().to_numpy()
    boots = np.array([rng.choice(per_game, len(per_game)).sum() for _ in range(4000)])
    per = b.groupby("season")["pnl"].sum().round(2)
    return {"bets": int(len(b)), "staked": float(len(b)), "units": float(b.pnl.sum()), "roi": float(b.pnl.mean()),
            "ci_lo": float(np.quantile(boots, .025)), "ci_hi": float(np.quantile(boots, .975)),
            "p": float(np.mean(boots <= 0)), "folds_positive": int((per > 0).sum()),
            "per_fold": {str(k): v for k, v in per.items()}}


def main() -> int:
    g = frame()
    print(f"DEV MLB games with open+close moneylines: {len(g)} ({g.season.min()}-{g.season.max()})")
    imp = 1 / np.asarray(american_to_decimal(g.ml_open_home), float) + 1 / np.asarray(american_to_decimal(g.ml_open_away), float)
    print(f"open overround median {np.median(imp) - 1:+.3f} (sanity: ~0.02-0.05)")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    recs = []
    for name, when in (("fav60_open", "open"), ("fav60_close", "close")):
        r = score(bets(g, when), np.random.default_rng(0))
        sha = hashlib.sha256(("mlb_ml_open:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        print(f"{name:12s} bets {r['bets']:5d}  units {r['units']:+7.1f}  ROI {r['roi']:+6.2%}  "
              f"95%CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]  p {r['p']:.4f}  seasons+ {r['folds_positive']}/9  {r['per_fold']}")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"mlb_{name}", "sha": sha,
                     "dataset": DATASET, "hypothesis": f"pre-registered in scripts/mlb_ml_open.py ({name})",
                     "source": "FLB in MLB opening odds (Harvard DASH thesis)", "status": "ok",
                     "trial_number": len(shas), **r})
    print(f"{DATASET} trials {len(shas)} -> bar p < {0.05 / len(shas):.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
