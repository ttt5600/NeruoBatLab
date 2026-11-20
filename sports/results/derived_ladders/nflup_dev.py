"""NFLUP devB scorer (prereg notes/derived_ladders.md). Loads devB only via calib.load(part='devB')."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib
OUT = Path(__file__).resolve().parent
def frame():
    fs = []
    for s in ["KXNFLPASSTDS", "KXNFLTD"]:
        d, cut = calib.load(s, "devB"); fs.append(d)
    d = pd.concat(fs, ignore_index=True)
    d["game"] = d.event_ticker.str.split("-").str[1]
    return d[d.valid].copy()
def pick(d, upper=True):
    k = (d.floor_strike >= 1.5) if upper else (d.floor_strike == 0.5)
    b = d[k & d.ask.between(0.05, 0.30)].copy()
    b["tc"] = b.ask + calib.fee(b.ask, b.mult); b["pnl"] = b.y - b.tc
    return b
def boot(b, n=10000, seed=0):
    g = b.groupby("game").agg(p=("pnl", "sum"), c=("tc", "sum")); P, C = g.p.to_numpy(), g.c.to_numpy()
    r = np.random.default_rng(seed); idx = r.integers(0, len(g), (n, len(g))); x = P[idx].sum(1) / C[idx].sum(1)
    return np.percentile(x, [2.5, 97.5]).round(4).tolist(), float((x <= 0).mean())
def boot_diff(a, c, n=10000, seed=1):
    ga = a.groupby("game").agg(p=("pnl", "sum"), c=("tc", "sum")); gc = c.groupby("game").agg(p=("pnl", "sum"), c=("tc", "sum"))
    g = ga.join(gc, lsuffix="a", rsuffix="c", how="outer").fillna(0)
    r = np.random.default_rng(seed); idx = r.integers(0, len(g), (n, len(g)))
    x = g.pa.to_numpy()[idx].sum(1) / g.ca.to_numpy()[idx].sum(1) - g.pc.to_numpy()[idx].sum(1) / g.cc.to_numpy()[idx].sum(1)
    return np.percentile(x, [2.5, 97.5]).round(4).tolist(), float((x <= 0).mean())
if __name__ == "__main__":
    d = frame()
    print("devB valid markets", len(d), "games", d.game.nunique(), "start range", d.start_utc.min(), d.start_utc.max())
    print("stale min median", d.stale_min.median(), "| T-start check done in calib.load")
    b, c = pick(d), pick(d, upper=False)
    for name, x in [("NFLUP", b), ("C1 (1+ rungs)", c)]:
        ci, p = boot(x)
        print(f"{name}: n {len(x)} games {x.game.nunique()} wins {int(x.y.sum())} hit {x.y.mean():.3f} mean ask {x.ask.mean():.3f} "
              f"ROI {x.pnl.sum()/x.tc.sum():+.4f} game-CI {ci} p {p:.4f}")
    dci, dp = boot_diff(b, c); print("NFLUP - C1 ROI diff CI", dci, "p", dp)
    print(b.groupby(["series", "floor_strike"]).agg(n=("y", "size"), hit=("y", "mean"), ask=("ask", "mean"), pnl=("pnl", "sum")).round(3).to_string())
    b["wk"] = pd.to_datetime(b.start_utc).dt.tz_convert("America/New_York").dt.to_period("W")
    print("by week pnl:", b.groupby("wk").pnl.sum().round(2).to_dict())
    b.to_parquet(OUT / "nflup_devB_bets.parquet")
