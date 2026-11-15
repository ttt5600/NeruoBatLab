"""Exp B secondary: per-bet ROI vs net CLV against the Kalshi pre-start close, on existing strategy rows.
net_clv = p_close(our side) - cost; x = net_clv / cost (= close-implied EV per $). OLS roi ~ a + b x, date-cluster bootstrap.
Run from sports/ after calib.py (and nba_fetch.py for NBA)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
R = OUT.parent / "kalshi_openers_xleague"
sys.path.insert(0, str(R))
from kx import side_x  # noqa: E402

RNG = np.random.default_rng(7)
EP = pd.Timestamp("1970-01-01", tz="UTC")
M = pd.read_parquet(OUT / "markets.parquet")
close = {}
for r in M.itertuples():
    if r.ask - r.bid <= 0.04 + 1e-9 and r.age_min <= 90:
        close[r.ticker] = (r.bid + r.ask) / 2
if (OUT / "nba_candles.parquet").exists():  # NBA: last candle <= start
    c = pd.read_parquet(OUT / "nba_candles.parquet")
    km = pd.read_parquet(OUT / "nba_kmarkets.parquet")
    g = pd.read_parquet(OUT / "nba_games.parquet")
    g["start_s"] = (g.start_utc - EP) // pd.Timedelta(seconds=1)
    j = c.merge(km[["ticker", "key"]], on="ticker").merge(g[["key", "start_s"]], on="key")
    j = j[(j.end <= j.start_s) & (j.start_s - j.end <= 90 * 60) & (j.ask - j.bid <= 0.04 + 1e-9)]
    j = j.sort_values("end").groupby("ticker").tail(1)
    close.update(dict(zip(j.ticker, (j.bid + j.ask) / 2)))

rows = []
for lg in ["cfb", "nfl", "nba"]:
    k = pd.read_csv(OUT.parent / f"k50_strike/rows_{lg}.csv")
    k = k[(k.status == "bet") & k.roi.notna() & k.ticker.notna()].drop_duplicates(["ticker", "pick"])
    for r in k.itertuples():
        if r.ticker not in close:
            continue
        ev = r.ticker.rsplit("-", 1)[0]
        _, sg = side_x(ev, r.ticker, float("nan") if pd.isna(r.S) else r.S, r.mk)
        yes = (r.pick == "up") == (sg > 0)
        p = close[r.ticker] if yes else 1 - close[r.ticker]
        rows.append(dict(src=f"k50_{lg}", date=r.date, roi=r.roi, cost=r.cost, p=p))
cb = pd.read_csv(OUT.parent / "cbb_opener_model/kalshi_rows.csv")
cb = cb[cb.bet == True]  # noqa: E712
ckm = pd.read_parquet(R / "cbb_kmarkets.parquet")
tk = {(e, s): t for e, s, t in zip(ckm.event, ckm.strike, ckm.ticker)}
for r in cb.itertuples():
    t = tk.get((r.event, r.S))
    if t is None or t not in close or pd.isna(r.roi):
        continue
    p = close[t] if r.side == "over" else 1 - close[t]
    rows.append(dict(src="cbb_opener", date=str(r.tip)[:10], roi=r.roi, cost=r.cost, p=p))
D = pd.DataFrame(rows)
D["x"] = (D.p - D.cost) / D.cost
D.to_csv(OUT / "reg_rows.csv", index=False)


def ols(d):
    X = np.c_[np.ones(len(d)), d.x.to_numpy()]
    return np.linalg.lstsq(X, d.roi.to_numpy(), rcond=None)[0]


def boot(d, B=2000):
    dates = d.date.unique()
    gi = {k: v.index.to_numpy() for k, v in d.groupby("date")}
    out = []
    for _ in range(B):
        s = RNG.choice(dates, len(dates))
        out.append(ols(d.loc[np.concatenate([gi[x] for x in s])]))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


res = []
for name, d in [("all", D)] + list(D.groupby("src")):
    d = d.reset_index(drop=True)
    a, b = ols(d)
    ci = boot(d)
    res.append(dict(src=name, n=len(d), dates=d.date.nunique(), mean_roi=d.roi.mean(), mean_x=d.x.mean(),
                    sd_x=d.x.std(), slope=b, slope_lo=ci[0, 1], slope_hi=ci[1, 1], icpt=a, icpt_lo=ci[0, 0],
                    icpt_hi=ci[1, 0]))
T = pd.DataFrame(res)
T.to_csv(OUT / "reg.csv", index=False)
print(T.round(3).to_string())

# (c) bias under OUR selection: outcome of our side minus its close prob, date-cluster bootstrap
D["resid"] = (D.roi > 0).astype(float) - D.p
sel = []
for name, d in [("all", D)] + list(D.groupby("src")):
    gs = d.groupby("date").resid.agg(["sum", "count"])
    s, n = gs["sum"].to_numpy(), gs["count"].to_numpy()
    idx = RNG.integers(0, len(s), size=(2000, len(s)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    sel.append(dict(src=name, n=len(d), bias_c=100 * s.sum() / n.sum(), lo_c=100 * np.percentile(bs, 2.5),
                    hi_c=100 * np.percentile(bs, 97.5)))
S = pd.DataFrame(sel)
S.to_csv(OUT / "selection_bias.csv", index=False)
print(S.round(2).to_string())
