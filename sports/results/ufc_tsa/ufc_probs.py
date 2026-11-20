"""Model method-given-win probabilities for every matched Kalshi fight (PIT features; model fit on UFCStats 2012-2025 only).
M  = full model (fighter history + base); P0 = base-rate-shape placebo (weight-class base rates + 5-rd + women only).
No Kalshi prices used here."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from ufc_hist import fights, long_table, asof, base_rates, feats, design, FEATS
BASE = ["base_KO", "base_SUB", "base_DEC", "sched5", "women"]
def dbase(X):
    Z = X[BASE].copy()
    for c in BASE[:3]: Z[c] = np.log(Z[c].clip(1e-3, 1))
    return Z.values
if __name__ == "__main__":
    tr = pd.read_parquet(OUT / "ufc_train.parquet")
    M = LogisticRegression(max_iter=3000, C=1.0).fit(design(tr), tr.y)
    P0 = LogisticRegression(max_iter=3000, C=1.0).fit(dbase(tr), tr.y)
    fr = fights(); L, cols = long_table(fr)
    br, brall = base_rates(fr, pd.Timestamp("2026-01-01"))
    j = pd.read_parquet(OUT / "ufc_fights.parquet"); j = j[j.fid.notna()]
    rows = []
    for r in j.itertuples():
        A = asof(L, cols, r.ufA, r.card); B = asof(L, cols, r.ufB, r.card)
        for side, (w, l, code) in (("A", (A, B, r.codeA)), ("B", (B, A, r.codeB))):
            f = feats(w, l, r.wclass, r.sched5, r.women, br, brall)
            f.update({"event_ticker": r.event_ticker, "fight": r.fight, "card": r.card, "split": r.split, "side": side, "code": code, "n_hist": w["n"]})
            rows.append(f)
    X = pd.DataFrame(rows)
    for tag, mdl, D in (("m", M, design), ("b", P0, dbase)):
        P = mdl.predict_proba(D(X))
        for i, c in enumerate(mdl.classes_): X[f"{tag}_{c}"] = P[:, i]
    X.to_parquet(OUT / "ufc_probs.parquet")
    print(X.groupby("split").size().to_dict(), "debut fighters share", round((X.n_hist == 0).mean(), 3))
    print(X[["m_KO", "m_SUB", "m_DEC", "b_KO", "b_SUB", "b_DEC"]].describe().round(3).to_string())
