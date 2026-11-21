"""D1 (prereg in notes/ufc_decision_gap.md): judges model MR vs Kalshi window price, DEV decision fights only."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
OUT = Path(__file__).resolve().parent
W = pd.read_parquet(OUT / "window.parquet"); F = pd.read_parquet(OUT / "fights_timed.parquet")
P = pd.read_parquet(OUT / "judges_test_preds.parquet")
R = pd.read_parquet(OUT / "rounds.parquet").drop_duplicates("fid")[["fid", "f1", "f2"]]
import sys; sys.path.insert(0, str(OUT.parent / "ufc_tsa")); from ufc_hist import nkey
d = W[W.has_live].merge(F[["event_ticker", "fid", "ufA", "ufB", "resA", "resB", "winner"]], on="event_ticker").merge(R, on="fid", how="left").merge(P, on="fid", how="left")
assert (d.card < "2026-07-01").all()
d["a_is_f1"] = d.ufA.map(nkey) == d.f1.map(nkey)
d["pM"] = np.where(d.a_is_f1, d.p_f1, 1 - d.p_f1)
d["y"] = d.resA.map({"yes": 1.0, "no": 0.0})
ok = lambda b, a: (b > 0) & (a < 1)
mA = np.where(ok(d.bidA, d.askA), d.midA, np.nan); mB = np.where(ok(d.bidB, d.askB), 1 - d.midB, np.nan)
d["pK"] = np.nanmean(np.c_[mA, mB], axis=1)
fb = np.nanmean(np.c_[d.pxA_h30, 1 - d.pxB_h30], axis=1)
d["pK_src"] = np.where(np.isnan(d.pK), "trade", "quote"); d["pK"] = np.where(np.isnan(d.pK), fb, d.pK)
d["pK_late"] = np.nanmean(np.c_[d.pxA_late, 1 - d.pxB_late], axis=1)
d["pK_pre"] = np.nanmean(np.c_[d.pxA_pre, 1 - d.pxB_pre], axis=1)
print("DEV decision fights with live timing", len(d), "| y missing (draw/scalar)", d.y.isna().sum(), "| pM missing", d.pM.isna().sum(), "| pK missing", d.pK.isna().sum())
print("Kalshi result vs UFCStats winner agree:", (d.y == (d.winner.map(nkey) == d.ufA.map(nkey)).astype(float))[d.y.notna()].mean())
d = d[d.y.notna() & d.pM.notna() & d.pK.notna()].copy()
print("analysis n", len(d), "cards", d.card.nunique(), "pK source", d.pK_src.value_counts().to_dict())
eps = 1e-3
def br(p): return (p - d.y) ** 2
def ll(p): p = np.clip(p, eps, 1 - eps); return -(d.y * np.log(p) + (1 - d.y) * np.log(1 - p))
cards = d.card.values; uc = np.unique(cards); idx = {c: np.where(cards == c)[0] for c in uc}
rng = np.random.default_rng(20261121)
B = [np.concatenate([idx[c] for c in rng.choice(uc, len(uc))]) for _ in range(2000)]
def ci(v):
    v = np.asarray(v); bs = [v[b].mean() for b in B]; return round(v.mean(), 4), np.round(np.percentile(bs, [2.5, 97.5]), 4).tolist()
for nm in ("pK", "pK_late", "pK_pre", "pM"):
    x = d[nm].fillna(0.5)
    print(f"{nm:8s} Brier {ci(br(x))}  logloss {ci(ll(x))}  acc {((x > .5) == (d.y == 1)).mean():.3f}")
print("Brier(pM)-Brier(pK)", ci(br(d.pM) - br(d.pK)), " logloss diff", ci(ll(d.pM) - ll(d.pK)))
print("Brier(pM)-Brier(pK_late)", ci(br(d.pM) - br(d.pK_late.fillna(.5))))
# stacking
def lg(p): return logit(np.clip(p, .01, .99))
def fit(ix, cols):
    X = np.c_[[lg(d[c].values[ix]) for c in cols]].T; y = d.y.values[ix]
    f = lambda w: -np.mean(y * np.log(np.clip(expit(X @ w), 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - expit(X @ w), 1e-9, 1)))
    return minimize(f, np.ones(len(cols)) * .5, method="BFGS").x
w = fit(np.arange(len(d)), ["pK", "pM"]); wb = np.array([fit(b, ["pK", "pM"]) for b in B[:1000]])
print("stack a(pK), b(pM):", np.round(w, 3), "b CI", np.round(np.percentile(wb[:, 1], [2.5, 97.5]), 3), "a CI", np.round(np.percentile(wb[:, 0], [2.5, 97.5]), 3))
# leave-one-card-out blend
pb = np.zeros(len(d))
for c in uc:
    tr = np.where(cards != c)[0]; te = idx[c]; ww = fit(tr, ["pK", "pM"])
    pb[te] = expit(np.c_[lg(d.pK.values[te]), lg(d.pM.values[te])] @ ww)
d["pBlend"] = pb
print("LOCO blend Brier", ci(br(d.pBlend)), " Brier(blend)-Brier(pK)", ci(br(d.pBlend) - br(d.pK)))
# where do they disagree
d["dis"] = (d.pM - d.pK)
for lo, hi in ((-1, -.3), (-.3, -.1), (-.1, .1), (.1, .3), (.3, 1)):
    s = d[(d.dis > lo) & (d.dis <= hi)]
    if len(s): print(f"pM-pK in ({lo},{hi}] n {len(s)} mean pK {s.pK.mean():.3f} mean pM {s.pM.mean():.3f} y {s.y.mean():.3f}")
for lo, hi in ((0, .1), (.1, .3), (.3, .7), (.7, .9), (.9, 1.01)):
    s = d[(d.pK >= lo) & (d.pK < hi)]
    if len(s): print(f"pK bin [{lo},{hi}) n {len(s)} mean pK {s.pK.mean():.3f} y {s.y.mean():.3f} pM {s.pM.mean():.3f}")
d.to_parquet(OUT / "d1_dev.parquet")
