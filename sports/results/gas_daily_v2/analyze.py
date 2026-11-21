"""After `edgelab evaluate g2_gas_daily_far-54b11381 --split dev`: placebos, descriptives D1-D4 (prereg notes/gas_daily_v2.md).
DEV only (frame contains dev events only)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from edgelab.rules.gas_daily import g2_far, _q, _ts, _fee
from edgelab import evaluator as EV
D = ROOT / "data" / "gas_daily_v2"
fr = pd.read_parquet(D / "gd_frame.parquet")
fr = fr[fr.result.isin(["yes", "no"])].copy()
fr.attrs["candles"] = json.loads((D / "gd_candles.json").read_text())
res = dict(zip(fr.ticker, fr.result))
elig = fr[fr.p_m.notna()].copy(); elig.attrs = fr.attrs
print("rows", len(fr), "eligible rows", len(elig), "events", elig.event_ticker.nunique(), elig.date.min().date(), elig.date.max().date())

def score(rows, name):
    if rows.empty:
        print(f"{name}: 0 bets"); return None
    r = rows.copy()
    r["outcome"] = [1.0 if res[t] == s else 0.0 for t, s in zip(r.bet_id, r.side)]
    cost, pnl, stake = EV.cost_and_pnl(r, {"rounding": "cent"})
    r["cost"], r["pnl"] = cost, pnl
    bt = EV.bootstrap(r.cluster.astype(str).to_numpy(), pnl, stake, 7e-4)
    roi = pnl.sum() / stake.sum()
    top = r.groupby("cluster").pnl.sum().sort_values(ascending=False)
    ex = (pnl.sum() - top.iloc[0]) / stake[r.cluster.to_numpy() != top.index[0]].sum()
    print(f"{name}: n {len(r)} dates {r.cluster.nunique()} losses {(r.outcome == 0).sum()} mean cost {cost.mean():.3f} "
          f"ROI {roi:+.4f} [{bt['ci'][0]:+.4f}, {bt['ci'][1]:+.4f}] p {bt['p']:.4f}  ex-top-date {ex:+.4f}  "
          f"sides {r.side.value_counts().to_dict()}")
    return r

out = {}
for name, params in [("G2", {}), ("P-RW", {"pcol": "p_rw"}), ("P-MOM", {"pcol": "p_mom"}), ("P-BASE", {"mode": "base"})]:
    out[name] = score(g2_far(elig, params), name)
g = out["G2"]
if g is not None:
    g.to_parquet("g2_dev_bets.parquet", index=False)
    top = g.groupby("cluster").agg(n=("pnl", "size"), pnl=("pnl", "sum"), losses=("outcome", lambda x: int((x == 0).sum())))
    print("D4 top-5 dates by pnl:\n", top.sort_values("pnl", ascending=False).head(5).to_string())
    print("D4 bottom-5 dates:\n", top.sort_values("pnl").head(5).to_string())
    print("G2 bets by NO-ask band:", g.groupby(pd.cut(g.price, [0, .9, .95, .97, .98, .99, 1])).agg(n=("pnl","size"), roi=("pnl", "sum")).to_string())

# ---- D3 availability at T
rows = []
for r in elig.itertuples(index=False):
    t = _ts(r.T)
    yb, ya, te = _q(fr.attrs["candles"].get(r.ticker, []), t, t - 6 * 3600)
    rows.append((r.ticker, r.event_ticker, r.date, r.X, r.p_m, r.mu_m, yb, ya, res[r.ticker], float(r.expiration_value)))
q = pd.DataFrame(rows, columns=["ticker", "ev", "date", "X", "p", "mu", "yb", "ya", "res", "v"])
q["y"] = (q.res == "yes").astype(float)
far = q[(q.p <= .03) | (q.p >= .97)]
print("D3: rungs with quote at T", int(q.yb.notna().sum()), "of", len(q))
lo = q[q.p <= .03]; hi = q[q.p >= .97]
print(f"D3: p<=.03 rungs {len(lo)}, with a quote {lo.yb.notna().sum()}, YES bid >= .01 {(lo.yb >= .01).sum()}; "
      f"p>=.97 rungs {len(hi)}, YES ask <= .99 {(hi.ya <= .99).sum()}")
print("D3: realized YES rate on p<=.03 rungs", round(lo.y.mean(), 4), "mean p", round(lo.p.mean(), 4),
      "| NO rate on p>=.97 rungs", round(1 - hi.y.mean(), 4), "mean 1-p", round(1 - hi.p.mean(), 4))
print("D3: yes_bid distribution on p<=.03 quoted rungs", lo.yb.describe().round(3).to_dict())

# ---- D2 Benter stacking on two-sided rungs at T
from scipy.special import logit
import statsmodels.api as sm
two = q[q.yb.notna() & q.ya.notna() & (q.yb > 0) & (q.ya < 1)].copy()
two["mid"] = (two.yb + two.ya) / 2
two = two[(two.mid >= .03) & (two.mid <= .97)].copy()
two["pm"] = two.p.clip(1e-4, 1 - 1e-4)
Xs = np.c_[np.ones(len(two)), logit(two.mid), logit(two.pm)]
fit = sm.Logit(two.y.to_numpy(), Xs).fit(disp=0)
rng = np.random.default_rng(0); dates = two.date.unique(); coefs = []
for _ in range(1000):
    pick = rng.choice(dates, len(dates)); bb = pd.concat([two[two.date == d] for d in pick])
    try:
        coefs.append(sm.Logit(bb.y.to_numpy(), np.c_[np.ones(len(bb)), logit(bb.mid), logit(bb.pm)]).fit(disp=0).params[2])
    except Exception:
        pass
print(f"D2 Benter: n {len(two)} rungs / {two.date.nunique()} dates; coef logit(mid) {fit.params[1]:.3f}, coef logit(p_M) "
      f"{fit.params[2]:.3f} date-boot 95% [{np.quantile(coefs, .025):.3f}, {np.quantile(coefs, .975):.3f}]")
bm = np.mean((two.mid - two.y) ** 2); bp = np.mean((two.pm - two.y) ** 2)
d = (two.pm - two.y) ** 2 - (two.mid - two.y) ** 2
dd = d.groupby(two.date).sum(); boot = [dd.sample(len(dd), replace=True, random_state=i).sum() / len(two) for i in range(2000)]
print(f"D2 Brier mid {bm:.4f} model {bp:.4f} diff(model-mid) {d.mean():+.4f} [{np.quantile(boot,.025):+.4f}, {np.quantile(boot,.975):+.4f}]")

# D1 moved to d1_timing.py (vectorised)
