"""Nowcast N1 (frozen in notes/mortgage_rates.md): mu = P0 + beta * (A - B), vintage D-1 only.

Calibration weeks D in 2023-01-05 .. 2026-09-10 -> beta, residuals, margin m. Then N1 for every Kalshi event D.
Target = FIRST print of PMMS for D (earliest Wednesday vintage that contains D, i.e. vintage D+6)."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
V = R / "vintages"


def vint(sid, wed):
    f = V / f"{sid}_{wed.date()}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d.columns = ["date", "v"]
    d["date"] = pd.to_datetime(d["date"])
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.dropna().set_index("date")["v"]


def row(D):
    D = pd.Timestamp(D)
    w = D - pd.Timedelta(days=1)                      # Wednesday vintage D-1 (published strictly before T = D 14:00Z)
    ob, pm = vint("OBMMIC30YF", w), vint("MORTGAGE30US", w)
    if ob is None or pm is None:
        return None
    assert ob.index.max() <= w and pm.index.max() <= w, "lookahead"   # traps: nothing dated after the vintage
    A = ob[(ob.index >= D - pd.Timedelta(days=7)) & (ob.index <= D - pd.Timedelta(days=1))]
    B = ob[(ob.index >= D - pd.Timedelta(days=14)) & (ob.index <= D - pd.Timedelta(days=8))]
    p0d = D - pd.Timedelta(days=7)
    if p0d not in pm.index or len(A) == 0 or len(B) == 0:
        return None
    first = None
    nxt = vint("MORTGAGE30US", D + pd.Timedelta(days=6))
    if nxt is not None and D in nxt.index:
        first = float(nxt[D])
    return dict(D=D, P0=float(pm[p0d]), A=float(A.mean()), B=float(B.mean()), nA=len(A), nB=len(B),
                A_last=str(A.index.max().date()), first=first)


if __name__ == "__main__":
    cal = [r for D in pd.date_range("2023-01-05", "2026-09-10", freq="W-THU") if (r := row(D))]
    c = pd.DataFrame(cal).dropna(subset=["first"])
    x, y = (c.A - c.B).values, (c["first"] - c.P0).values
    beta = float((x * y).sum() / (x * x).sum())
    c["mu"] = c.P0 + beta * (c.A - c.B)
    c["e"] = c["first"] - c.mu
    m = math.ceil(round(c.e.abs().max() * 100, 6)) / 100 + 0.02
    # first print vs current value (revision check)
    cur = pd.read_csv(R / "raw" / "MORTGAGE30US.csv"); cur.columns = ["date", "v"]
    cur = cur.assign(date=pd.to_datetime(cur.date)).set_index("date").v
    c["rev"] = c.D.map(cur) - c["first"]
    stats = dict(n=len(c), beta=beta, e_sd=float(c.e.std()), e_mad=float(c.e.abs().median()),
                 e_q95=float(c.e.abs().quantile(.95)), e_q99=float(c.e.abs().quantile(.99)), e_max=float(c.e.abs().max()),
                 naive_sd=float(y.std()), naive_max=float(np.abs(y).max()), margin_m=m,
                 n_revised=int((c.rev.abs() > 1e-9).sum()), rev_max=float(c.rev.abs().max()),
                 A_last_counts=c.A_last.map(lambda s: pd.Timestamp(s).day_name()).value_counts().to_dict())
    print(json.dumps(stats, indent=1))
    print(c.loc[c.e.abs().nlargest(5).index, ["D", "P0", "A", "B", "mu", "first", "e"]])
    c.to_csv(R / "calib.csv", index=False)
    ev = [r for D in pd.date_range("2026-09-17", "2026-11-19", freq="W-THU") if (r := row(D))]
    k = pd.DataFrame(ev)
    k["mu"] = k.P0 + beta * (k.A - k.B)
    k.loc[k.D > "2026-10-29", "first"] = np.nan   # sealed/forward outcome stays unread
    k["lo_yes_max"] = k.mu - m        # YES on 'greater K' iff K + 0.01 <= mu - m
    k["hi_no_min"] = k.mu + m         # NO on 'greater K' iff K >= mu + m
    k.to_csv(R / "nowcast_events.csv", index=False)
    json.dump(stats, open(R / "calib.json", "w"), indent=1)
    print(k[["D", "P0", "A", "B", "mu", "lo_yes_max", "hi_no_min", "A_last"]])
