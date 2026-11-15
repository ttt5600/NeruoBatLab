"""Pre-registered certainty test (notes/energy_monthly.md): late-month pinned rungs on ERCOT hub monthly RT averages.
DEV months 2012-01..2023-12 only (2011 = burn-in history). No Kalshi prices exist historically -> this measures only
violations (losses) and flag counts. Bound windows end strictly before day 1 of the decision month (no lookahead)."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
GRID = {"HB_HOUSTON": [20, 30, 35, 40, 50, 65, 100],
        "HB_NORTH": [20, 25, 30, 35, 45, 60, 100],
        "HB_WEST": [0, 15, 25, 30, 40, 60, 100]}
RS = [10, 7, 5, 3, 2, 1]
MULT, LPAD = 1.25, 25.0

h = pd.read_parquet(OUT / "hubs.parquet")
h["date"] = pd.to_datetime(h["Delivery Date"], format="%m/%d/%Y")
h["p"] = h["Settlement Point Price"].astype(float)
d = (h.groupby(["Settlement Point Name", "date"])["p"].agg(["sum", "count"]).reset_index()
     .rename(columns={"Settlement Point Name": "hub"}))
d["mean"] = d["sum"] / d["count"]
assert d["count"].between(92, 100).all(), d[~d["count"].between(92, 100)]

rows = []
for hub, g in d.groupby("hub"):
    g = g.set_index("date").sort_index()
    roll = {R: g["mean"].rolling(R).mean() for R in RS}
    for mstart in pd.date_range("2012-01-01", "2023-12-01", freq="MS"):
        mend = mstart + pd.offsets.MonthEnd(0)
        gm = g.loc[mstart:mend]
        if len(gm) != mend.day:
            continue
        final = round(gm["sum"].sum() / gm["count"].sum(), 2)
        N = gm["count"].sum()
        for R in RS:
            past = roll[R].loc[: mstart - pd.Timedelta(days=1)].dropna()
            U, L = MULT * past.max(), min(past.min(), 0) - LPAD
            dday = mend.day - R + 1
            el = gm.loc[: mstart + pd.Timedelta(days=dday - 2)]   # days 1..d-1
            ne, A = el["count"].sum(), el["sum"].sum()
            nr = N - ne
            Fhi, Flo = (A + nr * U) / N, (A + nr * L) / N
            for K in GRID[hub]:
                side = "NO" if Fhi <= K else ("YES" if Flo > K else None)
                if side is None:
                    continue
                win = (final <= K) if side == "NO" else (final > K)
                rows.append(dict(hub=hub, month=mstart.strftime("%Y-%m"), R=R, K=K, side=side, A_partial=A / max(ne, 1),
                                 Fhi=Fhi, Flo=Flo, final=final, win=win, dist=abs(final - K), U=U))
r = pd.DataFrame(rows)
r.to_csv(OUT / "pin_dev.csv", index=False)
print("flags", len(r), "losses", (~r.win).sum())
print(r.groupby(["R", "side"]).agg(n=("win", "size"), losses=("win", lambda x: (~x).sum()), med_dist=("dist", "median")))
nm = r.month.nunique()
print("hub-months", r.groupby("hub").month.nunique().to_dict())
per = r[r.R == 5].groupby(["hub", "month"]).size()
print("flags per hub-month at R=5: mean", per.reindex(pd.MultiIndex.from_product([GRID, sorted(r.month.unique())]), fill_value=0).mean())
yrs = r[r.R == 5].assign(y=r.month.str[:4]).groupby("y").size()
print("R=5 flags by year", yrs.to_dict())
print("U by year (R=5,NORTH)", r[(r.R == 5) & (r.hub == "HB_NORTH")].assign(y=r.month.str[:4]).groupby("y").U.max().round(0).to_dict())
print(r[~r.win])
