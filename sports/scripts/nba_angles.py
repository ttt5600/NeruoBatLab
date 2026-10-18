"""Popular NBA betting angles on 12 seasons of closing lines (SBR-scraped odds, public repo).

    python scripts/nba_angles.py [--ledger]

Data: data/nba_odds/OddsData.sqlite from github.com/kyleskom/NBA-Machine-Learning-Sports-Betting
(tables odds_YYYY-YY: Date, Home, Away, OU, Spread (= points HOME is favoured by), ML_Home/ML_Away
(American), Points (total), Win_Margin, Days_Rest_* (1 = played yesterday)). Gitignored, re-download:
curl -sL -o data/nba_odds/OddsData.sqlite https://raw.githubusercontent.com/kyleskom/NBA-Machine-Learning-Sports-Betting/master/Data/OddsData.sqlite

Pre-registered 2026-10-17. DEV = 2007-08 .. 2018-19; SEALED = 2019-20 onward (never loaded here).
Spread/total bets priced at -110 (the file has no spread/total prices); moneylines at their own price.
  N1 rest_edge_ats  : one team on a back-to-back (rest 1), other rest >= 2 -> rested team ATS
  N2 big_spread_dog : |spread| >= 12 -> underdog ATS
  N3 heavy_fav_ml   : moneyline <= -400 -> favourite ML
  N4 b2b_unders     : either team on a back-to-back -> under
  N5 home_dog_ml    : home team is the ML underdog -> home ML
5 trials on dataset nba_sbr -> bar p < 0.01; bootstrap over games.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sports.market import american_to_decimal

DB = ROOT / "data" / "nba_odds" / "OddsData.sqlite"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nba_sbr"
DEV = [f"{y}-{str(y + 1)[2:]}" for y in range(2007, 2019)]
STD = 1 + 100 / 110  # -110


def frame() -> pd.DataFrame:
    c = sqlite3.connect(DB)
    fs = []
    for s in DEV:
        d = pd.read_sql(f'select * from "odds_{s}"', c)
        d["season"] = int(s[:4])
        fs.append(d)
    d = pd.concat(fs, ignore_index=True)
    for k in ["OU", "Spread", "ML_Home", "ML_Away", "Points", "Win_Margin"]:
        d[k] = pd.to_numeric(d[k], errors="coerce")
    d = d.dropna(subset=["OU", "Spread", "ML_Home", "ML_Away", "Points", "Win_Margin"]).reset_index(drop=True)
    # The file's Spread is the favourite's line UNSIGNED (>0 in 97-98% of rows whoever is favoured; checked
    # 2026-10-17). Sign it from the moneyline favourite: + = home favoured. corr with margin 0.2-0.3 -> 0.41-0.50.
    d["Spread"] = d.Spread.abs() * np.where(d.ML_Home < d.ML_Away, 1, -1)
    return d


def ats(d, home_side: pd.Series) -> np.ndarray:
    """+1 if the chosen side covers. home covers when margin > spread (spread = home favoured by)."""
    cover = d.Win_Margin - d.Spread
    won = np.where(home_side, cover > 0, cover < 0)
    push = cover == 0
    return np.where(push, 0.0, np.where(won, STD - 1, -1.0))


def ml(d, home_side: pd.Series) -> np.ndarray:
    dec = np.where(home_side, american_to_decimal(d.ML_Home), american_to_decimal(d.ML_Away)).astype(float)
    won = np.where(home_side, d.Win_Margin > 0, d.Win_Margin < 0)
    return np.where(won, dec - 1, -1.0)


def trials(d):
    rh, ra = d.Days_Rest_Home, d.Days_Rest_Away
    fav_home = d.ML_Home < d.ML_Away
    out = {}
    m = ((rh == 1) & (ra >= 2)) | ((ra == 1) & (rh >= 2))
    out["rest_edge_ats"] = (m, ats(d, ra == 1))           # back the HOME side when the away team is tired
    m = d.Spread.abs() >= 12
    out["big_spread_dog"] = (m, ats(d, d.Spread < 0))     # home is the dog when spread < 0
    m = np.minimum(d.ML_Home, d.ML_Away) <= -400
    out["heavy_fav_ml"] = (m, ml(d, fav_home))
    m = (rh == 1) | (ra == 1)
    tot = d.Points - d.OU
    out["b2b_unders"] = (m, np.where(tot == 0, 0.0, np.where(tot < 0, STD - 1, -1.0)))
    m = d.ML_Home > 0
    out["home_dog_ml"] = (m, ml(d, pd.Series(True, index=d.index)))
    return out


def score(pnl: np.ndarray, season: pd.Series, rng) -> dict:
    boots = np.array([rng.choice(pnl, len(pnl)).sum() for _ in range(4000)])
    per = pd.Series(pnl, index=season.index).groupby(season).sum().round(2)
    return {"bets": int(len(pnl)), "staked": float(len(pnl)), "units": float(pnl.sum()), "roi": float(pnl.mean()),
            "ci_lo": float(np.quantile(boots, .025)), "ci_hi": float(np.quantile(boots, .975)),
            "p": float(np.mean(boots <= 0)), "folds_positive": int((per > 0).sum()),
            "per_fold": {str(k): v for k, v in per.items()}}


def main() -> int:
    d = frame()
    print(f"DEV NBA games {len(d)} ({d.season.min()}-{d.season.max() + 1})")
    over = 1 / american_to_decimal(d.ML_Home) + 1 / american_to_decimal(d.ML_Away) - 1
    print(f"sanity: corr(spread, margin) {np.corrcoef(d.Spread, d.Win_Margin)[0, 1]:+.2f} (expect > 0); "
          f"ML overround median {np.median(over):+.3f}; home covers {(d.Win_Margin > d.Spread).mean():.3f}; "
          f"overs {(d.Points > d.OU).mean():.3f}")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    recs = []
    for name, (m, pnl) in trials(d).items():
        m = np.asarray(m, bool)
        r = score(np.asarray(pnl)[m], d.season[m], np.random.default_rng(0))
        sha = hashlib.sha256(("nba_angles:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        print(f"{name:15s} bets {r['bets']:5d}  units {r['units']:+7.1f}  ROI {r['roi']:+6.2%}  "
              f"95%CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]  p {r['p']:.4f}  seasons+ {r['folds_positive']}/12")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"nba_{name}", "sha": sha,
                     "dataset": DATASET, "hypothesis": f"pre-registered in scripts/nba_angles.py ({name})",
                     "source": "popular NBA angles", "status": "ok", "trial_number": len(shas), **r})
    print(f"{DATASET} trials {len(shas)} -> bar p < {0.05 / len(shas):.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
