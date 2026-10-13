"""NBA player props: bet a book's price when it beats the OTHER books' no-vig consensus.

    python scripts/consensus_props.py dev        # 2023: thresholds, staleness, by book/market
    python scripts/consensus_props.py sealed --confirm   # 2024 once, with the rule frozen below

The +EV method with the most practitioner and academic support (market-making
logic, Buchdahl CLV evidence): a price that disagrees with the consensus of
several books is more likely the book's error than the consensus's.

For each (game, player, market, line) quoted by >= MIN_BOOKS books with both sides:
  consensus p_over = median of the OTHER books' de-vigged p_over (leave-one-out)
  EV(side at book b) = p_side_consensus * decimal_price_b - 1
Bet the side if EV >= threshold. Graded against ESPN box scores.

PROTOCOL (fixed 2026-10-14 before any result was computed):
  dev     2023 season (choose threshold and filters here)
  sealed  2024 season, scored ONCE with the frozen rule; 2025-26 have one book only.
  Risk being tested in dev: 'current' prices may be stale quotes; staleness =
  this book's last_updated minus the median last_updated of the other books.
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

DATA = ROOT / "data" / "espn" / "nba"
LOG = ROOT / "lab" / "reports" / "consensus_props_sealed.jsonl"
DEV, SEALED = 2023, 2024
MIN_BOOKS = 3
# frozen rule for the sealed run -- edit ONLY before `sealed` has ever been run
RULE = {"ev_min": 0.04, "max_stale_h": 6.0, "min_books": 3}


def pairs(season: int) -> pd.DataFrame:
    g = pd.read_parquet(DATA / f"graded_{season}.parquet")
    g = g[g["side"].isin(["over", "under"]) & g["athlete_id"].notna() & g["line"].notna()]
    key = ["event_id", "provider", "athlete_id", "market_n", "line"]
    o = g[g["side"] == "over"].drop_duplicates(key).set_index(key)
    u = g[g["side"] == "under"].drop_duplicates(key).set_index(key)
    j = o[["dec", "actual", "last_updated", "won", "push"]].rename(columns={"dec": "dec_o", "won": "won_o"}) \
        .join(u[["dec"]].rename(columns={"dec": "dec_u"}), how="inner").reset_index()
    j = j[(j["dec_o"] > 1) & (j["dec_u"] > 1) & j["actual"].notna()]
    io, iu = 1 / j["dec_o"], 1 / j["dec_u"]
    j["p_o"] = io / (io + iu)                      # this book's no-vig P(over)
    j["hold"] = io + iu - 1
    return j


def consensus(j: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out median of the other books' p_over and last_updated, per quote."""
    grp = ["event_id", "athlete_id", "market_n", "line"]
    j = j.copy()
    j["lu"] = pd.to_datetime(j["last_updated"], utc=True).astype("int64") / 3.6e12  # hours
    out_p, out_lu, out_n = [], [], []
    for _, g in j.groupby(grp, sort=False):
        p, lu = g["p_o"].to_numpy(), g["lu"].to_numpy()
        n = len(g)
        for i in range(n):
            if n - 1 < MIN_BOOKS - 1:
                out_p.append(np.nan); out_lu.append(np.nan); out_n.append(n - 1)
                continue
            m = np.ones(n, bool); m[i] = False
            out_p.append(np.median(p[m])); out_lu.append(np.median(lu[m])); out_n.append(n - 1)
    idx = np.concatenate([g.index.to_numpy() for _, g in j.groupby(grp, sort=False)])
    j.loc[idx, "p_cons"] = out_p
    j.loc[idx, "lu_cons"] = out_lu
    j.loc[idx, "n_other"] = out_n
    j["stale_h"] = j["lu_cons"] - j["lu"]           # >0: this book's price is older than the others'
    j["ev_o"] = j["p_cons"] * j["dec_o"] - 1
    j["ev_u"] = (1 - j["p_cons"]) * j["dec_u"] - 1
    return j.dropna(subset=["p_cons"])


def bets(j: pd.DataFrame, ev_min: float, max_stale_h: float | None = None) -> pd.DataFrame:
    over = j["ev_o"] >= j["ev_u"]
    ev = np.where(over, j["ev_o"], j["ev_u"])
    dec = np.where(over, j["dec_o"], j["dec_u"])
    push = j["push"] == 1
    won = np.where(over, j["won_o"] == 1, (j["won_o"] == 0) & ~push)
    b = j.assign(bet_over=over, ev=ev, dec=dec, profit=np.where(push, 0.0, np.where(won, dec - 1, -1.0)))
    b = b[b["ev"] >= ev_min]
    if max_stale_h is not None:
        b = b[b["stale_h"].abs() <= max_stale_h]
    # one bet per player-market-game: the best-EV quote across books/lines
    return b.sort_values("ev", ascending=False).drop_duplicates(["event_id", "athlete_id", "market_n"])


def summary(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"bets": 0}
    by_day = b.groupby(pd.to_datetime(b["last_updated"], utc=True).dt.date)["profit"].sum().to_numpy()
    boots = np.random.default_rng(0).choice(by_day, (4000, len(by_day))).sum(1)
    return {"bets": len(b), "units": round(float(b["profit"].sum()), 1), "roi": round(float(b["profit"].mean()), 4),
            "ev_claimed": round(float(b["ev"].mean()), 4), "p": float((boots <= 0).mean()),
            "ci_lo": round(float(np.percentile(boots, 2.5)), 1), "ci_hi": round(float(np.percentile(boots, 97.5)), 1)}


def cmd_dev() -> None:
    j = consensus(pairs(DEV))
    pd.set_option("display.width", 220)
    print(f"{DEV}: {len(j):,} quotes with >= {MIN_BOOKS - 1} other books; "
          f"median hold {j['hold'].median():.3f}; books {j['provider'].nunique()}")
    print("\nthreshold sweep (dev only):")
    rows = [{"ev_min": t, **summary(bets(j, t))} for t in (0.0, 0.02, 0.04, 0.06, 0.08, 0.12)]
    print(pd.DataFrame(rows).to_string(index=False))
    b = bets(j, 0.04)
    print("\nstaleness (hours this book's price is OLDER than the others') vs ROI, EV>=4%:")
    b["stale_bin"] = pd.cut(b["stale_h"], [-1e9, -6, -1, 1, 6, 24, 1e9])
    print(b.groupby("stale_bin", observed=True).agg(n=("profit", "size"), roi=("profit", "mean"),
                                                    ev=("ev", "mean")).round(3).to_string())
    print("\nby book (EV>=4%, |stale|<=6h):")
    bb = bets(j, 0.04, 6.0)
    print(bb.groupby("provider").agg(n=("profit", "size"), units=("profit", "sum"), roi=("profit", "mean"),
                                     ev=("ev", "mean")).sort_values("n", ascending=False).round(3).to_string())
    print("\nby market:")
    print(bb.groupby("market_n").agg(n=("profit", "size"), roi=("profit", "mean")).sort_values(
        "n", ascending=False).head(12).round(3).to_string())
    print("\nby side:")
    print(bb.groupby("bet_over").agg(n=("profit", "size"), roi=("profit", "mean")).round(3).to_string())


def cmd_sealed(confirm: bool) -> None:
    if not confirm:
        print("the sealed season is scored once, by a human: pass --confirm")
        return
    if LOG.exists():
        print("already scored:", LOG.read_text().strip())
        return
    b = bets(consensus(pairs(SEALED)), RULE["ev_min"], RULE["max_stale_h"])
    rec = {"season": SEALED, "rule": RULE, **summary(b)}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(rec) + "\n")
    print(rec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["dev", "sealed"])
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    cmd_dev() if a.cmd == "dev" else cmd_sealed(a.confirm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
