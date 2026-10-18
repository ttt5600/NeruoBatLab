"""MLB: follow vs fade the open->close moneyline move (SBRO 2010-2018 dev).

    python scripts/mlb_ml_move.py [--ledger]

Pre-registered 2026-10-17 (dataset mlb_ml_sbr, trials 3-4; same DEV/SEALED split as mlb_ml_open.py):
  move = de-vigged home win prob at CLOSE minus at OPEN.
  M3 follow_move : |move| >= 0.04 -> bet the side the line moved TOWARD, at the close price ("sharp money")
  M4 fade_move   : |move| >= 0.04 -> bet the side the line moved AWAY from, at the close price (Moskowitz 2021:
                   open->close moves are reversed by outcomes)
Bootstrap over games.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from sports.market import american_to_decimal
import mlb_ml_open as M

LEDGER = ROOT / "lab" / "ledger.jsonl"


def fair(h, a):
    ih, ia = 1 / np.asarray(american_to_decimal(h), float), 1 / np.asarray(american_to_decimal(a), float)
    return ih / (ih + ia)


def main() -> int:
    g = M.frame()
    move = fair(g.ml_close_home, g.ml_close_away) - fair(g.ml_open_home, g.ml_open_away)
    print(f"DEV games {len(g)}; |move| >= 0.04 in {(np.abs(move) >= 0.04).mean():.1%}; median |move| {np.median(np.abs(move)):.3f}")
    dh = np.asarray(american_to_decimal(g.ml_close_home), float)
    da = np.asarray(american_to_decimal(g.ml_close_away), float)
    won_h = (g.home_runs > g.away_runs).to_numpy()
    big = np.abs(move) >= 0.04
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == M.DATASET}
    recs = []
    for name, home_side in (("follow_move", move > 0), ("fade_move", move < 0)):
        dec = np.where(home_side, dh, da)
        won = np.where(home_side, won_h, ~won_h)
        b = g[big].assign(pnl=np.where(won, dec - 1, -1.0)[big])
        b = b.assign(game=b.index)
        r = M.score(b, np.random.default_rng(0))
        sha = hashlib.sha256(("mlb_ml_move:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        print(f"{name:12s} bets {r['bets']:5d}  units {r['units']:+7.1f}  ROI {r['roi']:+6.2%}  "
              f"95%CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]  p {r['p']:.4f}  seasons+ {r['folds_positive']}/9")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"mlb_{name}", "sha": sha,
                     "dataset": M.DATASET, "hypothesis": f"pre-registered in scripts/mlb_ml_move.py ({name})",
                     "source": "sharp-money lore vs Moskowitz 2021", "status": "ok", "trial_number": len(shas), **r})
    print(f"{M.DATASET} trials {len(shas)} -> bar p < {0.05 / len(shas):.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
