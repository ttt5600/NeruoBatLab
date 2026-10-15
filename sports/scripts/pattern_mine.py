"""Work backwards through a FROZEN snapshot: which simple prop patterns made money at the real prices?

    python scripts/pattern_mine.py discover [--snapshot 2026-10-14]   # non-sealed seasons only
    python scripts/pattern_mine.py confirm --confirm                  # survivors, sealed seasons, ONCE

A pattern is a cell: league x side (over/under) x up to two of
    market | price bucket | line level | line move | price move | book
scored as ROI of betting 1u on every graded prop side in the cell at its quoted price.

Honesty rules:
  * reads only data/frozen/<snapshot>/ (checksummed); sealed seasons from its MANIFEST
  * one player-game-market is ONE observation: the same prop at 8 books (or several
    lines) is clustered, so duplicate quotes cannot fake significance
  * Benjamini-Hochberg across every cell tested; survivors need q < 0.05, ROI > 0,
    >= MIN_CLUSTERS distinct props, and the same sign in both halves of discovery
  * survivors are written to a file BEFORE confirm runs; confirm scores exactly
    those on the sealed seasons, once, Bonferroni over the survivor count
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

FROZEN = ROOT / "data" / "frozen"
REPORTS = ROOT / "lab" / "reports"
SURV = REPORTS / "pattern_survivors.json"
CONF = REPORTS / "pattern_confirm.jsonl"
MIN_CLUSTERS = 300
DIMS = ["market_n", "price_b", "line_b", "line_move", "price_move", "provider"]


def snapshot_dir(name: str | None) -> Path:
    snaps = sorted(p for p in FROZEN.iterdir() if p.is_dir())
    return FROZEN / name if name else snaps[-1]


def load(snap: Path, sealed_only: bool) -> pd.DataFrame:
    man = json.loads((snap / "MANIFEST.json").read_text())
    sealed = man["sealed"]
    parts = []
    for lg, sy in sealed.items():
        for f in sorted((snap / lg).glob("graded_*.parquet")):
            y = int(f.stem.split("_")[1])
            if (y == sy) != sealed_only:
                continue
            g = pd.read_parquet(f, columns=["event_id", "provider", "athlete_id", "market_n", "side", "line",
                                            "line_open", "dec", "dec_open", "profit", "last_updated"])
            g = g[g["side"].isin(["over", "under"]) & g["profit"].notna() & (g["dec"] > 1)]
            parts.append(g.assign(league=lg, season=y))
    d = pd.concat(parts, ignore_index=True)
    d["cluster"] = d["league"] + "|" + d["event_id"].astype(str) + "|" + d["athlete_id"].astype(str) + "|" + d["market_n"]
    d["price_b"] = pd.cut(d["dec"], [1, 1.6, 1.8, 1.87, 1.95, 2.05, 2.3, 100],
                          labels=["<1.6", "1.6-1.8", "1.8-1.87", "1.87-1.95", "1.95-2.05", "2.05-2.3", ">2.3"]).astype(str)
    # line level: tercile of the line within (league, market), cut points fixed from discovery data
    d["line_move"] = np.sign(d["line"] - d["line_open"]).map({-1: "down", 0: "flat", 1: "up"}).fillna("na")
    pm = np.sign(d["dec_open"] - d["dec"])  # >0: this side's price shortened (money came in on it)
    d["price_move"] = pm.map({-1: "drifted", 0: "flat", 1: "shortened"}).fillna("na")
    d["half"] = 0
    return d


def add_line_bins(d: pd.DataFrame, cuts: dict | None = None) -> tuple[pd.DataFrame, dict]:
    if cuts is None:
        cuts = {k: list(np.unique(np.quantile(g["line"], [1 / 3, 2 / 3])))
                for k, g in d.groupby(["league", "market_n"])}
    lab = []
    for (lg, m), g in d.groupby(["league", "market_n"]):
        c = cuts.get((lg, m)) or cuts.get(f"{lg}|{m}")
        if c is None:
            lab.append(pd.Series("na", index=g.index))
            continue
        lab.append(pd.Series(np.select([g["line"] <= c[0], g["line"] <= c[-1]], ["low", "mid"], "high"), index=g.index))
    d["line_b"] = pd.concat(lab)
    return d, cuts


def cell_stats(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """ROI and cluster-robust t per cell."""
    c = d.groupby(keys + ["cluster"], observed=True)["profit"].agg(["sum", "size"]).reset_index()
    g = c.groupby(keys, observed=True)
    out = g.agg(units=("sum", "sum"), bets=("size", "sum"), clusters=("sum", "size")).reset_index()
    out["roi"] = out["units"] / out["bets"]
    c = c.merge(out[keys + ["roi"]], on=keys)
    c["r"] = c["sum"] - c["roi"] * c["size"]          # ratio-estimator residual per cluster
    v = c.groupby(keys, observed=True).agg(var=("r", "var"), mean_n=("size", "mean")).reset_index()
    out = out.merge(v, on=keys)
    se = np.sqrt(out["var"] / out["clusters"]) / out["mean_n"]
    out["t"] = out["roi"] / se
    out["p"] = 2 * stats.norm.sf(out["t"].abs())
    return out


def bh(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p)
    q = np.empty(len(p))
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def combos():
    yield []
    for k in (1, 2):
        yield from (list(c) for c in itertools.combinations(DIMS, k))


def cmd_discover(snap: Path) -> None:
    d = load(snap, sealed_only=False)
    d, cuts = add_line_bins(d)
    # halves: split discovery chronologically within each league
    d["t"] = pd.to_datetime(d["last_updated"], utc=True)
    med = d.groupby("league")["t"].transform("median")
    d["half"] = (d["t"] > med).astype(int)
    print(f"discovery rows {len(d):,}, distinct props {d['cluster'].nunique():,}, by league/season:")
    print(d.groupby(["league", "season"]).agg(rows=("profit", "size"), roi=("profit", "mean")).round(4).to_string())
    cells = []
    for extra in combos():
        keys = ["league", "side"] + extra
        s = cell_stats(d, keys)
        s = s[s["clusters"] >= MIN_CLUSTERS]
        s["cell"] = s.apply(lambda r: " & ".join(f"{k}={r[k]}" for k in keys), axis=1)
        s["keys"] = [json.dumps({k: str(r[k]) for k in keys}) for _, r in s.iterrows()]
        cells.append(s[["cell", "keys", "bets", "clusters", "units", "roi", "t", "p"]])
    c = pd.concat(cells, ignore_index=True)
    c["q"] = bh(c["p"].to_numpy())
    # same sign in both halves
    def half_roi(keys_json, h):
        k = json.loads(keys_json)
        m = (d["half"] == h)
        for kk, vv in k.items():
            m &= d[kk].astype(str) == vv
        return d.loc[m, "profit"].mean()
    cand = c[(c["q"] < 0.05) & (c["roi"] > 0)].copy()
    cand["roi_h1"] = [half_roi(k, 0) for k in cand["keys"]]
    cand["roi_h2"] = [half_roi(k, 1) for k in cand["keys"]]
    surv = cand[(cand["roi_h1"] > 0) & (cand["roi_h2"] > 0)].sort_values("t", ascending=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 95)
    print(f"\n{len(c):,} cells tested (>= {MIN_CLUSTERS} distinct props each); "
          f"BH q<0.05 & ROI>0: {len(cand)}; also positive in both halves: {len(surv)}")
    print("\nTOP 30 SURVIVORS (discovery only):")
    print(surv[["cell", "bets", "clusters", "units", "roi", "t", "q", "roi_h1", "roi_h2"]].head(30)
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nworst 10 cells (what the other tail looks like):")
    print(c.sort_values("t")[["cell", "clusters", "roi", "t"]].head(10).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    REPORTS.mkdir(parents=True, exist_ok=True)
    c.to_csv(REPORTS / "pattern_cells.csv", index=False)
    if CONF.exists():
        print("\nconfirm has already been run; survivors file left unchanged")
        return
    SURV.write_text(json.dumps({"snapshot": snap.name, "n_cells_tested": len(c),
                                "line_cuts": {f"{k[0]}|{k[1]}": v for k, v in cuts.items()},
                                "survivors": surv[["cell", "keys", "roi", "t", "clusters"]].to_dict("records")},
                               indent=1))
    print(f"\nwrote {len(surv)} survivors -> {SURV.relative_to(ROOT)} (these, and only these, go to confirm)")


def cmd_confirm(snap: Path, confirm: bool) -> None:
    if not confirm:
        print("confirm scores the sealed seasons ONCE; pass --confirm")
        return
    if CONF.exists():
        print("already confirmed:\n" + CONF.read_text())
        return
    sv = json.loads(SURV.read_text())
    d = load(snap, sealed_only=True)
    cuts = {tuple(k.split("|", 1)): v for k, v in sv["line_cuts"].items()}
    d, _ = add_line_bins(d, cuts)
    k = max(len(sv["survivors"]), 1)
    rows = []
    for s in sv["survivors"]:
        keys = json.loads(s["keys"])
        m = pd.Series(True, index=d.index)
        for kk, vv in keys.items():
            m &= d[kk].astype(str) == vv
        sub = d[m]
        if sub.empty:
            rows.append({"cell": s["cell"], "dev_roi": s["roi"], "sealed_bets": 0})
            continue
        st = cell_stats(sub.assign(_all="all"), ["_all"]).iloc[0]
        rows.append({"cell": s["cell"], "dev_roi": s["roi"], "sealed_bets": int(st["bets"]),
                     "sealed_clusters": int(st["clusters"]), "sealed_units": float(st["units"]),
                     "sealed_roi": float(st["roi"]), "sealed_t": float(st["t"]), "sealed_p": float(st["p"]),
                     "passes": bool(st["roi"] > 0 and st["p"] / 2 < 0.05 / k)})
    r = pd.DataFrame(rows)
    with CONF.open("w") as fh:
        for rec in rows:
            fh.write(json.dumps(rec) + "\n")
    pd.set_option("display.width", 250, "display.max_colwidth", 95)
    print(f"sealed-season confirmation of {len(rows)} survivors (one-sided Bonferroni bar p < {0.05 / k:.4f}; "
          f"sealed_p is TWO-sided, pass = ROI > 0 and sealed_p/2 < bar):")
    print(r.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nPASS: {int(r.get('passes', pd.Series(dtype=bool)).fillna(False).sum())}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["discover", "confirm"])
    ap.add_argument("--snapshot")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    snap = snapshot_dir(a.snapshot)
    cmd_discover(snap) if a.cmd == "discover" else cmd_confirm(snap, a.confirm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
