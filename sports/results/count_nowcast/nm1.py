"""NM1 (KXTRUMPNOMNUM) dev: congress.gov civilian receivedDate nowcast at Thursday 15:00Z."""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from kx_throttle import kget
R = Path("results/count_nowcast"); E = pd.Timestamp("1970-01-01", tz="UTC")
fee = lambda p: math.ceil(round(0.07 * p * (1 - p) * 100, 6)) / 100
n = pd.DataFrame(json.load(open(R / "congress_noms_119.json")))
n["rd"] = pd.to_datetime(n.receivedDate); n = n[n.nominationType.map(lambda d: bool(d.get("isCivilian")))]
m = pd.DataFrame(json.load(open(R / "mk/KXTRUMPNOMNUM.json"))).drop_duplicates("ticker")
m["k"] = m.floor_strike.astype(float)
pr = m.rules_primary.str.extract(r"from (\w+ \d+, \d{4}) through (\w+ \d+, \d{4})")
m["ws"] = pd.to_datetime(pr[0], format="%b %d, %Y"); m["we"] = pd.to_datetime(pr[1], format="%b %d, %Y")
for c in ("open_time", "close_time"): m[c] = pd.to_datetime(m[c], utc=True, format="ISO8601")
m = m[m.result.isin(["yes", "no"]) & (m.we <= "2026-11-14")].copy()
cache = R / "nm_candles.json"; cc = json.loads(cache.read_text()) if cache.exists() else {}
for tk, o, c in zip(m.ticker, m.open_time, m.close_time):
    if tk in cc: continue
    p = {"start_ts": (o - E) // pd.Timedelta(seconds=1) - 3600, "end_ts": (c - E) // pd.Timedelta(seconds=1) + 3600, "period_interval": 60}
    cs = (kget(f"/series/KXTRUMPNOMNUM/markets/{tk}/candlesticks", p) or {}).get("candlesticks", [])
    if not cs: cs = (kget(f"/historical/markets/{tk}/candlesticks", p) or {}).get("candlesticks", [])
    cc[tk] = cs
cache.write_text(json.dumps(cc))
def quote(tk, T):
    t = (T - E) // pd.Timedelta(seconds=1)
    c = [x for x in cc.get(tk, []) if t - 6 * 3600 < x["end_period_ts"] <= t]
    if not c: return np.nan, np.nan
    g = lambda s: (lambda v: np.nan if v is None else float(v))(c[-1][s].get("close", c[-1][s].get("close_dollars")))
    return g("yes_bid"), g("yes_ask")
rows = []
for r in m.itertuples():
    T = (r.ws + pd.Timedelta(days=4)).tz_localize("UTC") + pd.Timedelta(hours=15)
    C = ((n.rd >= r.ws) & (n.rd <= r.ws + pd.Timedelta(days=3))).sum()
    tot = ((n.rd >= r.ws) & (n.rd <= r.we)).sum()
    yb, ya = quote(r.ticker, T) if r.close_time > T else (np.nan, np.nan)
    rows.append(dict(ticker=r.ticker, ws=r.ws, k=r.k, C=C, tot=tot, result=r.result, open_at_T=r.close_time > T, yb=yb, ya=ya,
                     lowest=r.k == m[m.event_ticker == r.event_ticker].k.min()))
f = pd.DataFrame(rows); f.to_parquet(R / "nm1_frame.parquet")
print("dev rungs", len(f), "weeks", f.ws.nunique(), "open at T", f.open_at_T.sum(), "with quote", f.ya.notna().sum())
print("nowcast C vs final civ count (by week):"); print(f.groupby("ws")[["C", "tot"]].first().T.to_string())
f["noask"] = 1 - f.yb
bets = []
for r in f.itertuples():
    if not np.isfinite(r.yb): continue
    if r.C == 0 and r.k >= 1:
        a = r.noask
        if a >= 0.50 and (1 - 0.065) - a - fee(a) >= 0.02: bets.append(("N", r.ticker, r.ws, "no", a, r.result == "no"))
    if r.C >= r.k and 0.50 <= r.ya <= 0.98: bets.append(("Y", r.ticker, r.ws, "yes", r.ya, r.result == "yes"))
b = pd.DataFrame(bets, columns=["arm", "ticker", "ws", "side", "ask", "win"])
if len(b):
    b["cost"] = b.ask + b.ask.map(fee); b["pnl"] = b.win - b.cost
    print(b.to_string()); print(b.groupby("arm").agg(n=("pnl", "size"), wins=("win", "sum"), roi=("pnl", lambda s: s.sum() / b.loc[s.index, "cost"].sum())).to_string())
else: print("no bets")
pl = f[f.lowest & f.yb.notna()].copy(); pl = pl[(pl.noask >= 0.50) & ((1 - 0.065) - pl.noask - pl.noask.map(fee) >= 0.02)]
if len(pl):
    pl["cost"] = pl.noask + pl.noask.map(fee); pl["pnl"] = (pl.result == "no") - pl.cost
    print("placebo n", len(pl), "wins", (pl.result == "no").sum(), "ROI", round(pl.pnl.sum() / pl.cost.sum(), 4))
# descriptive: YES quote on rungs above C when C==0, and on rungs <= C
d = f[f.yb.notna()]
print("C==0 rungs: YES bid", d[d.C == 0].yb.describe().round(3).to_dict(), "results yes:", (d[d.C == 0].result == "yes").sum())
print("C>=k rungs: YES ask", d[d.C >= d.k].ya.describe().round(3).to_dict())
