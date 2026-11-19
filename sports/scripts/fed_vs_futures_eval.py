"""fed_vs_futures: dev evaluation of rule FT (taker NO on futures-near-zero outcomes) + FM (maker). Prereg notes/fed_vs_futures.md."""
import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

R = Path(__file__).resolve().parent.parent / "results" / "fed_vs_futures"
ET = "America/New_York"
P_MAX, MARGIN, MIN_BID = 0.05, 0.03, 0.03

ms = pd.DataFrame(json.load(open(R / "markets.json")))
cd = json.load(open(R / "candles.json"))
zq = pd.read_csv(R / "zq_front.csv", parse_dates=["Date"]).set_index("Date")["Close"].dropna()
effr = pd.read_csv(R / "raw" / "effr.csv", parse_dates=["observation_date"]).set_index("observation_date")["EFFR"]
effr = pd.to_numeric(effr, errors="coerce").dropna()


def bucket(t):
    t = t.lower()
    if "maintain" in t or "no change" in t or "0bps" in t or "no cut" in t:
        return 0
    s = -1 if "cut" in t else 1
    if ">25" in t:
        return 2 * s  # >= 50
    if ">50" in t:
        return 3 * s
    if "50" in t:
        return 2 * s
    return s


ms["b"] = ms.yes_sub_title.map(bucket)
# 23MAY has 50 and >50 hike buckets: treat H50 as exactly +50 (b=2), TH50 as >=75 (b=3). Others: +-2 means >= 50.
ms["A"] = pd.to_datetime(ms.close_time, utc=True).dt.tz_convert(ET).dt.normalize().dt.tz_localize(None)

# ---- ZQ roll diagnostic: months without a meeting -> ZQ ~ 100 - EFFR monthly avg
cal_effr = effr.reindex(pd.date_range("2023-01-01", "2025-12-31"), method="ffill")
mm = set(ms.A.dt.to_period("M"))
diag = []
for d, px in zq.items():
    per = d.to_period("M")
    if per in mm or d.year < 2023:
        continue
    diag.append((d, px, 100 - cal_effr[str(per)].mean(), d.day))
dg = pd.DataFrame(diag, columns=["d", "zq", "fair", "day"])
dg["err_bp"] = (dg.zq - dg.fair) * 100
print("roll diag (non-meeting months): |err| bp median", dg.err_bp.abs().median().round(2),
      "max", dg.err_bp.abs().max().round(1), "n", len(dg))
print(dg.sort_values("err_bp", key=abs).tail(5).to_string())


def fut_p(A, D):
    """P_f per bucket using ZQ close of last trading day < D, EFFR <= D-2. None if invalid."""
    z = zq[zq.index < D]
    if not len(z):
        return None, None
    zd, px = z.index[-1], z.iloc[-1]
    if (zd.year, zd.month) != (A.year, A.month):
        return None, None
    e = effr[effr.index <= D - pd.Timedelta(days=2)]
    r0 = e.iloc[-1]
    N = calendar.monthrange(A.year, A.month)[1]
    days = pd.date_range(A.replace(day=1), periods=N)
    obs = e.reindex(days, method="ffill")
    obs[days > D - pd.Timedelta(days=2)] = np.nan
    pre = obs.fillna(r0)[days <= A].sum()
    d2 = (days > A).sum()
    if d2 == 0:
        return None, None
    r_post = ((100 - px) * N - pre) / d2
    delta = (r_post - r0) * 100 / 25
    k0 = int(np.floor(delta)); f = delta - k0
    return {k0: 1 - f, k0 + 1: f}, (zd, px, r0, r_post, delta * 25)


def pb(probs, b):
    """Kalshi bucket b: +-2 means >= 50 (or ==50 when a +-3 bucket exists in the event)."""
    if b in (0, 1, -1):
        return probs.get(b, 0.0)
    s = np.sign(b)
    return sum(v for k, v in probs.items() if s * k >= abs(b))


rows = []
for ev, g in ms.groupby("event_ticker"):
    A = g.A.iloc[0]
    has3 = (g.b.abs() == 3).any()
    for k in (14, 7, 1):
        D = A - pd.Timedelta(days=k)
        probs, info = fut_p(A, D)
        if probs is None:
            rows.append({"event": ev, "k": k, "valid": False})
            continue
        tdec = pd.Timestamp(D).tz_localize(ET).replace(hour=15).tz_convert("UTC").timestamp()
        tnext = (pd.Timestamp(A - pd.Timedelta(days={14: 7, 7: 1, 1: 0}[k])).tz_localize(ET).replace(hour=15)
                 .tz_convert("UTC").timestamp()) if k > 1 else pd.Timestamp(g.close_time.iloc[0]).timestamp()
        for _, m in g.iterrows():
            b = m.b
            if has3 and abs(b) == 2:
                pf = probs.get(b, 0.0)
            else:
                pf = pb(probs, b)
            cs = [c for c in cd.get(m.ticker, []) if c["end_period_ts"] <= tdec]
            r = {"event": ev, "k": k, "valid": True, "ticker": m.ticker, "b": b, "pf": round(pf, 4),
                 "yes": int(m.result == "yes"), "zq_date": info[0].date(), "delta_bp": round(info[4], 1), "d2": calendar.monthrange(A.year, A.month)[1] - A.day}
            if cs and tdec - cs[-1]["end_period_ts"] <= 86400:
                c = cs[-1]
                yb = float(c["yes_bid"]["close"]); ya = float(c["yes_ask"]["close"])
                vol24 = sum(float(x["volume"]) for x in cs if x["end_period_ts"] > tdec - 86400)
                later = [x for x in cd[m.ticker] if tdec < x["end_period_ts"] <= tnext]
                hi = max([float(x["price"]["high"]) for x in later if x["price"].get("high") is not None] or [np.nan])
                r.update(yes_bid=yb, yes_ask=ya, vol24=vol24, later_hi=hi)
            rows.append(r)
F = pd.DataFrame(rows)
F.to_csv(R / "frame.csv", index=False)
v = F[F.valid]
print("cells valid", v[["event", "k"]].drop_duplicates().shape[0], "of", F[["event", "k"]].drop_duplicates().shape[0])
print("market-cells valid", len(v), "with quote", v.yes_bid.notna().sum())

# ---- calibration (descriptive): futures P vs Kalshi mid vs realised, by pf band
q = v[v.yes_bid.notna()].copy()
q["mid"] = (q.yes_bid + q.yes_ask) / 2
q["band"] = pd.cut(q.pf, [-0.01, 0.0001, 0.05, 0.2, 0.5, 0.8, 0.95, 1.01])
print(q.groupby("band", observed=True).agg(n=("yes", "size"), pf=("pf", "mean"), mid=("mid", "mean"),
                                           bid=("yes_bid", "mean"), real=("yes", "mean")).round(3).to_string())

# ---- rule FT
t = q[(q.pf <= P_MAX) & (q.yes_bid >= q.pf + MARGIN) & (q.yes_bid >= MIN_BID)].copy()
t["price"] = 1 - t.yes_bid
t["fee"] = 0.07 * t.price * (1 - t.price)
t["stake"] = t.price + t.fee
t["pnl"] = np.where(t.yes == 0, 1 - t.stake, -t.stake)
t.to_csv(R / "bets_FT.csv", index=False)
n, L = len(t), int(t.yes.sum())
print(f"\nFT bets n={n} losses={L} events={t.event.nunique()}")
if n:
    roi = t.pnl.sum() / t.stake.sum()
    lo, hi = (beta.ppf(0.025, L, n - L + 1) if L else 0.0), beta.ppf(0.975, L + 1, n - L)
    be = 1 - t.stake.mean()
    rng = np.random.default_rng(0)
    evs = t.event.unique()
    bs = []
    for _ in range(10000):
        s = rng.choice(evs, len(evs))
        tt = pd.concat([t[t.event == e] for e in s])
        bs.append(tt.pnl.sum() / tt.stake.sum())
    print(f"ROI {roi:+.4f} meeting-boot CI [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}]; "
          f"loss rate {L/n:.3f} CP [{lo:.3f},{hi:.3f}] vs break-even {be:.3f}; mean yes_bid {t.yes_bid.mean():.3f}; "
          f"median vol24 {t.vol24.median()}; total vol24 {t.vol24.sum()}")
    print(t.groupby("k").agg(n=("pnl", "size"), L=("yes", "sum"), pnl=("pnl", "sum")).to_string())
    print(t[["ticker", "k", "pf", "yes_bid", "yes_ask", "vol24", "yes", "pnl", "delta_bp"]].to_string())

# ---- rule FM (maker)
m = q[q.pf <= P_MAX].copy()
m["offer"] = np.maximum(m.yes_bid + 0.01, m.yes_ask - 0.01)
m = m[(m.offer >= m.pf + MARGIN) & (m.offer >= MIN_BID) & (m.offer < m.yes_ask + 1e-9)]
m["filled"] = m.later_hi >= m.offer - 1e-9
m["pnl"] = np.where(m.yes == 0, m.offer, -(1 - m.offer))
m.to_csv(R / "bets_FM.csv", index=False)
print(f"\nFM candidates {len(m)}; filled {int(m.filled.sum())} losses filled {int(m[m.filled].yes.sum())} "
      f"unfilled {int((~m.filled).sum())} losses unfilled {int(m[~m.filled].yes.sum())}; "
      f"filled pnl/contract {m[m.filled].pnl.mean() if m.filled.any() else float('nan'):+.4f}")
