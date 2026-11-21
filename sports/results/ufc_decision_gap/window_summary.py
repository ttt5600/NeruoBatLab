"""Price-only descriptive summary of the horn -> announcement window (DEV). No outcomes."""
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
W = pd.read_parquet(OUT / "window.parquet"); w = W[W.has_live].copy()
w["pK_h30"] = np.nanmean(np.c_[w.pxA_h30, 1 - w.pxB_h30], axis=1)
w["fav"] = np.maximum(w.pK_h30, 1 - w.pK_h30)
w["wvol"] = w.wvolA + w.wvolB; w["wtr"] = w.wtrA + w.wtrB
w["active"] = w.wtr > 0
ok = lambda b, a: (b > 0) & (a < 1)
w["two_sided"] = ok(w.bidA, w.askA) & ok(w.bidB, w.askB)
w["spr_min"] = np.fmin(np.where(ok(w.bidA, w.askA), w.sprA, np.nan), np.where(ok(w.bidB, w.askB), w.sprB, np.nan))
w["yr"] = np.where(w.card < "2026-01-01", "2025", "2026H1")
print("fights", len(w), "| any trade in window", w.active.mean().round(3), "| both books two-sided at t_dec", w.two_sided.mean().round(3))
q = lambda s: s.quantile([.1, .25, .5, .75, .9]).round(3).tolist()
for g, x in w.groupby("yr"):
    print(f"== {g} n {len(x)}: window contracts (both tickers) p10..p90 {q(x.wvol)}; trades {q(x.wtr)}")
    print(f"   tighter-book spread at t_dec {q(x.spr_min.dropna())} (n {x.spr_min.notna().sum()}); favourite price at horn+30 {q(x.fav.dropna())}")
    print(f"   share fav >= .95: {(x.fav >= .95).mean():.3f}; .80-.95: {((x.fav >= .8) & (x.fav < .95)).mean():.3f}; < .80: {(x.fav < .8).mean():.3f}")
u = w[w.active & (w.fav < .9)].copy()
for a, b in (("t_jump", "t_horn"), ("t_jump", "t_results"), ("t_jump", "t_uw")):
    u[f"{a}-{b}"] = (u[a] - u[b]).dt.total_seconds()
print("contested (fav < .90 at horn+30, traded in window) n", len(u))
print(u[["t_jump-t_horn", "t_jump-t_results", "t_jump-t_uw"]].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
w.to_parquet(OUT / "window_summary.parquet")
