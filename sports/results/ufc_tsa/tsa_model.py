"""KXTSAD nowcast: daily TSA screenings. History = data/e1x/tsa_daily_*.csv (2023-01-01..2026-10-29) + KXTSAD
expiration_value (first-published) for later days. Two candidate forecasts, chosen on PRE-DEV residuals only:
  W: F = Y(d-7) * sum(Y over last 7 known days) / sum(Y over those days - 7)
  A: F = Y(d-364) * sum(Y over last 14 known days) / sum(Y over those days - 364)
Known days at decision time: x <= d-2 always; d-1 only if published (KXTSAD settlement_ts(d-1) <= T, for d-1 >= 10-09;
before that, simulated: d-1 known iff d is Tue..Fri)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
D = OUT.parents[1] / "data/e1x"
def series():
    s = pd.concat([pd.read_csv(p) for p in sorted(D.glob("tsa_daily_*.csv"))])
    s["date"] = pd.to_datetime(s.date, format="%m/%d/%Y")
    s = s.drop_duplicates("date").set_index("date").n.astype(float).sort_index()
    from census import load
    t = load("KXTSAD"); t = t[t.status == "finalized"].copy()
    t["date"] = pd.to_datetime(t.event_ticker.str.split("-").str[1], format="%y%b%d")
    t["st"] = pd.to_datetime(t.settlement_ts, format="ISO8601")
    e = t.groupby("date").agg(ev=("expiration_value", "first"), st=("st", "max"), ot=("ot", "min"))
    e["ev"] = e.ev.str.replace(",", "").astype(float)
    ov = e.ev.reindex(s.index).dropna()
    chk = (ov - s.reindex(ov.index)).abs()
    print("CSV vs KXTSAD first-published overlap days", len(ov), "max abs diff", chk.max())
    s = pd.concat([s, e.ev[~e.index.isin(s.index)]]).sort_index()
    return s, e
def known_through(d, e, T):
    """Last known day at decision time T for target day d."""
    prev = d - pd.Timedelta(days=1)
    if prev in e.index:
        return prev if e.loc[prev, "st"] <= T else d - pd.Timedelta(days=2)
    return prev if d.dayofweek in (1, 2, 3, 4) else d - pd.Timedelta(days=2)
def fc(s, d, last, kind):
    days = pd.date_range(end=last, periods=7 if kind == "W" else 14)
    lag = pd.Timedelta(days=7 if kind == "W" else 364)
    if not set(days).issubset(s.index) or not set(days - lag).issubset(s.index) or (d - lag) not in s.index: return np.nan
    return s[d - lag] * s[days].sum() / s[days - lag].sum()
if __name__ == "__main__":
    s, e = series()
    rows = []
    for d in pd.date_range("2024-01-15", "2026-10-08"):
        T = pd.Timestamp(d.date()).tz_localize("UTC") + pd.Timedelta(hours=16)
        last = known_through(d, e, T)
        rows.append({"date": d, "y": s.get(d, np.nan), "W": fc(s, d, last, "W"), "A": fc(s, d, last, "A")})
    r = pd.DataFrame(rows).dropna()
    for k in ("W", "A"):
        r["r" + k] = np.log(r.y / r[k])
    print("pre-dev residual days", len(r))
    for k in ("W", "A"):
        x = r["r" + k]
        print(k, "MAE %.4f  sd %.4f  q01 %.4f q05 %.4f q95 %.4f q99 %.4f" % (x.abs().mean(), x.std(), *x.quantile([.01, .05, .95, .99])))
        x2 = r[r.date >= "2025-07-01"]["r" + k]
        print("  last 15 months MAE %.4f sd %.4f" % (x2.abs().mean(), x2.std()))
    r.to_parquet(OUT / "tsa_resid_predev.parquet")
