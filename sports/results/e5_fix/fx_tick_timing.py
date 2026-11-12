"""Direct FX fill timing (11-10/11-11 events only; IBKR HMDS keeps ticks only ~1 day after expiry).
For every pick with a print strictly below its bid: first through-print time (local), the METAR running max at that
moment vs the threshold X, and the final METAR max (= settlement)."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/agent-harness/lib"))
from traps import check_epoch_seconds  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from fx_f2 import TZ  # noqa: E402

OUTD = Path(__file__).parent
TICKS, PICKS, OUT = (sys.argv[1:4] if len(sys.argv) > 3 else
                     ("fx_ticks_recent.parquet", "ib_in_recent.csv", "fx_tick_timing.csv"))
t = pd.read_parquet(OUTD / TICKS)
p = pd.read_csv(OUTD / PICKS)
p["code"] = p.ticker.str[2:5]
p["X"] = p.ticker.str.split("_").str[-1].astype(float)
o = pd.read_parquet(OUTD / "asos_obs.parquet")
o["stn"] = o.stn.astype(str).str[-3:]
t = t[t.status == "ok"].copy()
t["tsec"] = (t.ts - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
check_epoch_seconds(t.tsec, "tick ts")
rows = []
for r in p.itertuples():
    x = t[(t.ticker == r.ticker) & (t.side == r.side)].sort_values("tsec")
    if not len(x):
        continue
    z = TZ[r.code]
    thr = x[x.price < r.side_bid - 1e-9]
    w = o[(o.stn == r.code)]
    mid = pd.Timestamp(r.event_date, tz=z)
    day = w[(w.valid >= mid) & (w.valid < mid + pd.Timedelta(days=1))]
    fmax = day.tmpf.max()
    row = {"ticker": r.ticker, "side": r.side, "bid": r.side_bid, "X": r.X, "n_ticks": len(x),
           "first_tick_local": x.ts.iloc[0].tz_convert(z).strftime("%m-%d %H:%M"), "final_metar_max": fmax,
           "won_by_metar": (fmax <= r.X) if r.side == "NO" else (fmax > r.X)}
    if r.side == "NO":
        avail = day.valid + pd.Timedelta(minutes=15)
        rmax = day.tmpf.cummax()
        for lab, lvl in (("trig2_local", r.X + 1 - 2), ("decided_local", r.X + 1)):
            h = avail[rmax >= lvl]
            row[lab] = h.iloc[0].tz_convert(z).strftime("%m-%d %H:%M") if len(h) else None
    if len(thr):
        f = thr.ts.iloc[0]
        seen = day[day.valid + pd.Timedelta(minutes=15) <= f].tmpf.max()
        row.update({"first_through_local": f.tz_convert(z).strftime("%m-%d %H:%M"), "through_px": thr.price.iloc[0],
                    "runmax_at_fill": seen, "gap_to_lose_F": (r.X + 1 - seen) if r.side == "NO" else None,
                    "vol_through": float(thr["size"].sum())})
    rows.append(row)
d = pd.DataFrame(rows)
d.to_csv(OUTD / OUT, index=False)
print(d.to_string())
