"""Do 2-leg parlays of high-probability legs make money? (user idea, 2026-10-15) -- DEV data only.

    python scripts/parlay_test.py

PRE-REGISTERED (written before any parlay number existed):
  * high-probability leg = decimal price <= 1.50 (implied >= 66.7% incl. vig)
  * legs are paired at random (seed 0) within the same day (props) / same season-week (NFL), never two
    legs from the same game -- a cross-game parlay is priced as d1*d2 by every book, so this is the
    real payout. Same-game parlays are priced by a correlation model we have no prices for: untested.
  * payout uses the leg multiplier m = dec (win) / 1 (push) / 0 (loss); parlay pays m1*m2
  * one quote per prop (player-game-market-side): the alphabetically first book -- no line shopping,
    because best-of-N quotes in this data are mostly stale (see notes/results.md 10-14)
  * 2 trials (props, NFL moneyline) -> bar one-sided p < 0.025 AND ROI > 0; SE clustered by date
  * data windows: props = non-sealed seasons of data/frozen/2026-10-14; NFL = lab nfl_moneyline DEV 2006-19

The mechanism being tested: for independent legs E[m1*m2] = E[m1]*E[m2], so parlay ROI =
(1+r1)(1+r2)-1. Parlays cannot create an edge -- they square whatever the legs have, good or bad.
The only way they help is if high-probability legs are themselves +EV (favourite-longshot bias).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab"))
sys.path.insert(0, str(ROOT / "scripts"))

LEDGER = ROOT / "lab" / "reports" / "parlay_ledger.jsonl"
HP = 1.50
N_TRIALS = 2


def cluster_roi(profit: np.ndarray, cluster: np.ndarray):
    """ROI with SE clustered on `cluster` (ratio estimator; every bet stakes 1u)."""
    df = pd.DataFrame({"p": profit, "c": cluster})
    g = df.groupby("c")["p"].agg(["sum", "size"])
    n, roi = len(df), df.p.mean()
    r = g["sum"] - roi * g["size"]
    se = np.sqrt(len(g) / (len(g) - 1) * (r ** 2).sum()) / n
    t = roi / se
    return roi, se, t, float(stats.norm.sf(t))


def make_pairs(legs: pd.DataFrame, group: str, game: str, seed: int = 0) -> pd.DataFrame:
    """Random cross-game pairs within each `group`; returns parlay rows (m = m1*m2)."""
    rng = np.random.default_rng(seed)
    out = []
    for key, g in legs.groupby(group):
        g = g.iloc[rng.permutation(len(g))]
        pool = list(g.itertuples(index=False))
        while len(pool) >= 2:
            a = pool.pop()
            j = next((k for k, b in enumerate(pool) if getattr(b, game) != getattr(a, game)), None)
            if j is None:
                break
            b = pool.pop(j)
            out.append({"group": key, "m": a.m * b.m, "dec": a.dec * b.dec, "m1": a.m, "m2": b.m})
    return pd.DataFrame(out)


def prop_legs() -> pd.DataFrame:
    import pattern_mine as pm
    snap = pm.snapshot_dir(None)
    d = pm.load(snap, sealed_only=False)
    man = json.loads((snap / "MANIFEST.json").read_text())
    dates = []
    for lg, sy in man["sealed"].items():
        for f in sorted((snap / lg).glob("games_*.parquet")):
            if int(f.stem.split("_")[1]) == sy:
                continue
            g = pd.read_parquet(f, columns=["event_id", "start_utc"])
            dates.append(g.assign(league=lg))
    dates = pd.concat(dates).drop_duplicates(["league", "event_id"])
    dates["date"] = pd.to_datetime(dates["start_utc"]).dt.tz_convert("America/New_York").dt.date.astype(str)
    d = d.merge(dates[["league", "event_id", "date"]], on=["league", "event_id"], how="inner")
    d = d.sort_values("provider").drop_duplicates(["cluster", "side"])  # one quote per prop side
    d["m"] = 1.0 + d["profit"]
    d["game"] = d["league"] + "|" + d["event_id"].astype(str)
    return d[["date", "game", "league", "dec", "m", "cluster"]]


def nfl_legs() -> pd.DataFrame:
    import datasets as D
    d = D.load("nfl_moneyline", "dev")
    d = d[d["dec_odds"].notna()].copy()
    d["m"] = np.where(d["won"] == 1, d["dec_odds"], np.where(d["push"] == 1, 1.0, 0.0))
    d["dec"] = d["dec_odds"]
    d["week"] = d["season"].astype(str) + "-" + d["opp_id"].str.split("_").str[1]
    d["game"] = d["opp_id"]
    return d[["week", "game", "dec", "m", "season"]]


def report(name, legs, group):
    rows = []
    # singles by price bucket: where (if anywhere) are legs +EV?
    b = pd.cut(legs["dec"], [1, 1.25, 1.5, 1.8, 2.0, 2.5, 100])
    for k, g in legs.groupby(b, observed=True):
        roi, se, t, _ = cluster_roi(g["m"].values - 1, g[group].values)
        rows.append(f"    single {str(k):>13}: n={len(g):>7,}  ROI {roi:+.2%}  (t={t:+.2f})")
    hp = legs[legs["dec"] <= HP]
    r1, *_ = cluster_roi(hp["m"].values - 1, hp[group].values)
    pr = make_pairs(hp, group, "game")
    roi, se, t, p = cluster_roi(pr["m"].values - 1, pr["group"].values)
    pred = (1 + r1) ** 2 - 1
    print(f"\n{name}: legs at dec <= {HP}: n={len(hp):,}, single ROI {r1:+.2%}")
    print("\n".join(rows))
    print(f"  2-leg cross-game parlays: n={len(pr):,}  ROI {roi:+.2%} +- {se:.2%}  t={t:+.2f}  p={p:.3g}"
          f"   | compounding prediction (1+r)^2-1 = {pred:+.2%}   hit rate {np.mean(pr['m'] > 0):.1%}")
    return {"trial": name, "hp_max_dec": HP, "legs": int(len(hp)), "single_roi": r1, "parlays": int(len(pr)),
            "roi": roi, "se": se, "t": t, "p": p, "pred_roi": pred, "hit_rate": float(np.mean(pr["m"] > 0)),
            "passes": bool(roi > 0 and p < 0.05 / N_TRIALS)}


def main():
    recs = [report("props_hp_parlay", prop_legs(), "date"), report("nfl_ml_hp_parlay", nfl_legs(), "week")]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with LEDGER.open("a") as f:
        for r in recs:
            f.write(json.dumps({"ts": ts, **r}, default=float) + "\n")
    print(f"\nbar: one-sided p < {0.05 / N_TRIALS:.3f} and ROI > 0 -> PASS: {sum(r['passes'] for r in recs)}/{len(recs)}")


if __name__ == "__main__":
    main()
