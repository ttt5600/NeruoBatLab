"""NBA: a rating model decorrelated from the book, bet only where it disagrees (Hubacek et al. 2019).

    python scripts/nba_model.py [--ledger]

Same data/split as nba_angles.py (DEV 2007-08..2018-19; spread signed from the ML favourite).
Pre-registered 2026-10-17, trials 6-7 on dataset nba_sbr:
  N6 elo_disagree10 : margin-of-victory Elo (K=20, home +100, 538 MOV multiplier, 1/4 regression to 1505
                      between seasons), updated game by game in date order (rating used = BEFORE the game).
                      Bet the ML side where |p_elo - p_mkt| >= 0.10 (p_mkt = proportionally de-vigged ML).
  N7 logit_edge3    : walk-forward logistic (fit on all earlier dev seasons, >= 3) of home win on
                      logit(p_mkt), elo_diff/400, rest_home, rest_away; bet the side whose model prob
                      exceeds p_mkt by >= 0.03, at its ML price. First 3 seasons are training only.
Bootstrap over games. Bar = 0.05 / (#nba_sbr trials).
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
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sports.market import american_to_decimal
import nba_angles as N

LEDGER = ROOT / "lab" / "ledger.jsonl"


def with_dates(d: pd.DataFrame) -> pd.DataFrame:
    md = d.Date.astype(str).str[-4:]
    mon = md.str[:2].astype(int)
    yr = np.where(mon >= 8, d.season, d.season + 1)
    d = d.assign(date=pd.to_datetime(pd.Series(yr).astype(str).values + md.values, format="%Y%m%d"))
    return d.sort_values(["date"], kind="stable").reset_index(drop=True)


def elo(d: pd.DataFrame) -> np.ndarray:
    r, last_season, out = {}, None, np.empty(len(d))
    for i, g in enumerate(d.itertuples()):
        if g.season != last_season:
            r = {k: 0.75 * v + 0.25 * 1505 for k, v in r.items()}
            last_season = g.season
        h, a = r.get(g.Home, 1505.0), r.get(g.Away, 1505.0)
        diff = h + 100 - a
        out[i] = diff
        p = 1 / (1 + 10 ** (-diff / 400))
        mov = abs(g.Win_Margin)
        w = 1.0 if g.Win_Margin > 0 else 0.0
        mult = ((mov + 3) ** 0.8) / (7.5 + 0.006 * (diff if w else -diff))
        delta = 20 * mult * (w - p)
        r[g.Home], r[g.Away] = h + delta, a - delta
    return out


def main() -> int:
    d = with_dates(N.frame())
    dh, da = np.asarray(american_to_decimal(d.ML_Home), float), np.asarray(american_to_decimal(d.ML_Away), float)
    ph = (1 / dh) / (1 / dh + 1 / da)
    d["elo_diff"] = elo(d)
    p_elo = 1 / (1 + 10 ** (-d.elo_diff / 400))
    won_h = (d.Win_Margin > 0).to_numpy()
    print(f"DEV games {len(d)}; Brier market {np.mean((ph - won_h) ** 2):.4f}  Elo {np.mean((p_elo - won_h) ** 2):.4f} "
          f"(sanity: Elo should be close to but worse than the market)")

    def pnl(home_side, mask):
        dec = np.where(home_side, dh, da)
        won = np.where(home_side, won_h, ~won_h)
        return np.where(won, dec - 1, -1.0)[mask]

    res = {}
    gap = p_elo - ph
    m = np.abs(gap) >= 0.10
    res["elo_disagree10"] = (pnl(gap > 0, m), d.season[m])

    X = np.column_stack([np.log(ph / (1 - ph)), d.elo_diff / 400, d.Days_Rest_Home.clip(upper=4), d.Days_Rest_Away.clip(upper=4)])
    pm = np.full(len(d), np.nan)
    seasons = sorted(d.season.unique())
    for s in seasons[3:]:
        tr, te = (d.season < s).to_numpy(), (d.season == s).to_numpy()
        pm[te] = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], won_h[tr]).predict_proba(X[te])[:, 1]
    e = pm - ph
    m = ~np.isnan(pm) & (np.abs(e) >= 0.03)
    res["logit_edge3"] = (pnl(e > 0, m), d.season[m])

    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == N.DATASET}
    recs = []
    for name, (p, s) in res.items():
        r = N.score(p, s, np.random.default_rng(0))
        sha = hashlib.sha256(("nba_model:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        print(f"{name:15s} bets {r['bets']:5d}  units {r['units']:+7.1f}  ROI {r['roi']:+6.2%}  "
              f"95%CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]  p {r['p']:.4f}  seasons+ {r['folds_positive']}/{len(r['per_fold'])}")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"nba_{name}", "sha": sha,
                     "dataset": N.DATASET, "hypothesis": f"pre-registered in scripts/nba_model.py ({name})",
                     "source": "Hubacek et al. 2019 decorrelation", "status": "ok", "trial_number": len(shas), **r})
    print(f"{N.DATASET} trials {len(shas)} -> bar p < {0.05 / len(shas):.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
