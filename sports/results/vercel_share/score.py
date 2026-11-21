"""Score VS1/VS2 on DEV (prereg notes/vercel_share.md): rule + placebos, day-cluster bootstrap, top-5 contributors,
Brier(model) vs Brier(mid) at T1, far-rung staleness, VS2 book clearing time. Uses the edgelab rule functions on the edgelab
loader frame (dev rows only exist in the frame)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from edgelab.loaders import kalshi_vercel_share
from edgelab.rules import vercel_share as R
sys.path.insert(0, os.path.expanduser("~/agent-harness/lib")); from traps import check_epoch_seconds, check_no_lookahead, too_good
rng = np.random.default_rng(20261121)
F, _ = kalshi_vercel_share()
F = F[(F.date >= "2026-09-11") & (F.date <= "2026-10-25")].copy()
res = dict(zip(F.ticker, F.result)); ser = dict(zip(F.ticker, F.series))
def settle(b):
    if b.empty: return b
    b = b.copy(); b["fee"] = b.price.map(R.fee); b["cost"] = b.price + b.fee
    b["win"] = [(s == res[t]) for s, t in zip(b.side, b.bet_id)]
    b["pnl"] = b.win.astype(float) - b.cost; b["series"] = b.bet_id.map(ser); return b
def boot(b, n=10000):
    if b.empty: return (np.nan, np.nan, np.nan)
    g = b.groupby("cluster").agg(p=("pnl", "sum"), c=("cost", "sum")); P, C = g.p.values, g.c.values
    idx = rng.integers(0, len(g), (n, len(g))); r = P[idx].sum(1) / C[idx].sum(1)
    return (P.sum() / C.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5))
def summ(name, b):
    roi, lo, hi = boot(b)
    d = dict(arm=name, n=len(b), days=b.cluster.nunique() if len(b) else 0, losses=int((~b.win).sum()) if len(b) else 0,
             mean_price=round(b.price.mean(), 3) if len(b) else np.nan, roi=round(100 * roi, 2) if len(b) else np.nan,
             lo=round(100 * lo, 2) if len(b) else np.nan, hi=round(100 * hi, 2) if len(b) else np.nan,
             yes=int((b.side == "yes").sum()) if len(b) else 0)
    return d
out = {}
arms = {"VS1 M": R.vs1_far(F, {"pcol": "p_m"}), "VS1 P-persist": R.vs1_far(F, {"pcol": "p_persist"}),
        "VS1 P-clim": R.vs1_far(F, {"pcol": "p_clim"}), "VS2 V": R.vs2_post(F, {"vcol": "V"}), "VS2 P-V1": R.vs2_post(F, {"vcol": "V1"})}
# trap checks: candle epochs in seconds; VS1 quote at/before T1 and before close; VS2 quote after D+1 01:00Z and before close
check_epoch_seconds([k["end_period_ts"] for cs in F.attrs["candles"].values() for k in cs], "candle end_period_ts")
for k, b in arms.items():
    if b.empty: continue
    check_no_lookahead(b, "price_time", "event_start")
    if k.startswith("VS1"):
        assert (pd.to_datetime(b.price_time, utc=True) <= pd.to_datetime(b.decided_at, utc=True)).all(), "VS1 quote after T1"
    else:
        d1 = pd.to_datetime(b.cluster).dt.tz_localize("UTC") + pd.Timedelta(days=1, hours=1)
        assert (pd.to_datetime(b.price_time, utc=True).values >= d1.values).all(), "VS2 quote before D+1 01:00Z"
tab = []
for k, b in arms.items():
    b = settle(b); arms[k] = b; tab.append(summ(k, b))
    if len(b): b.to_csv(f"bets_{k.replace(' ', '_')}.csv", index=False)
T = pd.DataFrame(tab); print(T.to_string(index=False)); out["arms"] = tab
for t in tab:
    m = too_good("roi_pct", t["roi"] if t["roi"] == t["roi"] else -1, {"roi_pct": 15})
    if m: print("TOO-GOOD FLAG", t["arm"], m)
for k in ("VS1 M", "VS2 V"):
    b = arms[k]
    if b.empty: continue
    print(f"\n{k} by series:"); print(b.groupby("series").agg(n=("pnl", "size"), pnl=("pnl", "sum"), cost=("cost", "sum"), losses=("win", lambda w: (~w).sum())).round(3).to_string())
    top = b.groupby("cluster").pnl.sum().sort_values(ascending=False)
    print(f"{k} top-5 days by pnl:", top.head(5).round(3).to_dict(), "total", round(b.pnl.sum(), 3), "ex-top-day ROI",
          round(100 * (b.pnl.sum() - top.iloc[0]) / (b.cost.sum() - b[b.cluster == top.index[0]].cost.sum()), 2))
    print(f"{k} worst-5 days:", top.tail(5).round(3).to_dict())
    print(b.sort_values("pnl").head(8)[["bet_id", "side", "price", "p_model", "win", "pnl"]].to_string(index=False))
# ---- Brier at T1 + far-rung staleness (descriptive, prereg secondary)
C = F.attrs["candles"]; rows = []
for r in F.itertuples(index=False):
    t = R._ts(r.T1)
    if pd.Timestamp(r.close) <= pd.Timestamp(r.T1): continue
    yb, ya, te = R._q(C.get(r.ticker, []), t, t - 6 * 3600)
    if not (np.isfinite(yb) and np.isfinite(ya)) or yb <= 0 or ya >= 1 or ya - yb > 0.10 or not np.isfinite(r.p_m): continue
    rows.append(dict(cluster=str(pd.Timestamp(r.date).date()), series=r.series, p=r.p_m, pp=r.p_persist, pc=r.p_clim, mid=(yb + ya) / 2,
                     y=float(r.result == "yes")))
Bq = pd.DataFrame(rows)
if len(Bq):
    for c in ("p", "pp", "pc", "mid"): Bq["b_" + c] = (Bq[c] - Bq.y) ** 2
    g = Bq.groupby("cluster")[["b_p", "b_pp", "b_pc", "b_mid"]].agg(["sum", "count"])
    cl = Bq.cluster.unique(); idx = rng.integers(0, len(cl), (10000, len(cl)))
    gs = Bq.groupby("cluster")[["b_p", "b_pp", "b_pc", "b_mid"]].sum().loc[cl]; gn = Bq.groupby("cluster").size().loc[cl].values
    def bd(a, bcol):
        d = (gs[a].values - gs[bcol].values); r = d[idx].sum(1) / gn[idx].sum(1)
        return round(d.sum() / gn.sum(), 4), round(np.percentile(r, 2.5), 4), round(np.percentile(r, 97.5), 4)
    print(f"\nBrier on {len(Bq)} two-sided rungs (spread<=.10) at T1, {len(cl)} days: model {Bq.b_p.mean():.4f}, P-persist {Bq.b_pp.mean():.4f}, "
          f"P-clim {Bq.b_pc.mean():.4f}, Kalshi mid {Bq.b_mid.mean():.4f}")
    print(" Brier(model)-Brier(mid)", bd("b_p", "b_mid"), " Brier(model)-Brier(P-persist)", bd("b_p", "b_pp"))
    far = Bq[(Bq.p >= .85) | (Bq.p <= .15)].copy(); far["p_side"] = np.where(far.p >= .85, far.p, 1 - far.p)
    far["mid_side"] = np.where(far.p >= .85, far.mid, 1 - far.mid); far["won"] = np.where(far.p >= .85, far.y, 1 - far.y)
    print(f" far rungs (model p>=.85/<=.15) n {len(far)}: mean model p_side {far.p_side.mean():.3f}, mean mid_side {far.mid_side.mean():.3f}, realised {far.won.mean():.3f}")
    out["brier"] = dict(n=len(Bq), days=len(cl), model=Bq.b_p.mean(), persist=Bq.b_pp.mean(), clim=Bq.b_pc.mean(), mid=Bq.b_mid.mean(),
                        diff_model_mid=bd("b_p", "b_mid"), far_n=len(far), far_p=far.p_side.mean(), far_mid=far.mid_side.mean(), far_real=far.won.mean())
# ---- spreads at T1 (all rungs with any quote): two-sided share, median spread on two-sided rungs, far-rung neglect
sp = []
for r in F.itertuples(index=False):
    t = R._ts(r.T1)
    if pd.Timestamp(r.close) <= pd.Timestamp(r.T1): continue
    yb, ya, te = R._q(C.get(r.ticker, []), t, t - 6 * 3600)
    if te is None: continue
    sp.append(dict(series=r.series, two=(yb > 0) & (ya < 1), spread=ya - yb, p=r.p_m))
SP = pd.DataFrame(sp)
if len(SP):
    SP["fam"] = np.where(SP.series == "KXOPENSOURCESHARE", "open_share", "lab")
    print("\nQuotes at T1:"); print(SP.groupby("fam").agg(n=("two", "size"), two_sided=("two", "mean"),
          med_spread_two=("spread", lambda x: x[SP.loc[x.index, "two"]].median())).round(3).to_string())
    out["spread"] = SP.groupby("fam").agg(n=("two", "size"), two_sided=("two", "mean")).to_dict()
# ---- VS2 descriptive: book clearing time after D+1 00:00Z (outcome-side quote >= .95)
cl_rows = []
for r in F.itertuples(index=False):
    d1 = R._ts(pd.Timestamp(r.date).tz_localize("UTC") + pd.Timedelta(days=1)); cz = R._ts(r.close)
    cs = [k for k in C.get(r.ticker, []) if d1 < k["end_period_ts"] <= cz]
    if not cs: continue
    hit = None
    for k in cs:
        yb, ya = R._v(k, "yes_bid"), R._v(k, "yes_ask")
        ok = (r.result == "yes" and np.isfinite(yb) and yb >= .95) or (r.result == "no" and np.isfinite(ya) and ya <= .05)
        if ok: hit = (k["end_period_ts"] - d1) / 3600; break
    cl_rows.append(dict(series=r.series, ticker=r.ticker, hours_window=(cz - d1) / 3600, cleared_h=hit,
                        dist=abs(r.X - r.V) if np.isfinite(r.V) else np.nan))
CL = pd.DataFrame(cl_rows)
if len(CL):
    CL["fam"] = np.where(CL.series == "KXOPENSOURCESHARE", "open_share(close 10ET D+1)", "lab(close 23:59ET D)")
    print("\nVS2 clearing (rungs with candles after D+1 00:00Z):")
    print(CL.groupby("fam").agg(n=("ticker", "size"), window_h=("hours_window", "median"), cleared_any=("cleared_h", lambda x: x.notna().mean()),
          cleared_by_1h=("cleared_h", lambda x: (x <= 1).mean()), med_h=("cleared_h", "median")).round(3).to_string())
    near = CL[CL.dist <= 1.0]
    print(" rungs within 1.0 pt of V:", len(near), "cleared any", round(near.cleared_h.notna().mean(), 3))
    CL.to_csv("vs2_clearing.csv", index=False)
json.dump(out, open("score_dev.json", "w"), default=str, indent=1)
