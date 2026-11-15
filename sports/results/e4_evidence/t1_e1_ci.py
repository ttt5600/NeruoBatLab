"""T1: E1-only CI re-derived from the frozen sealed inputs of lockbox kalshi_temp_maker_M3_N10.
NOT a new scoring: same frozen variant(), asserts the data sha + the recorded E1 numbers, never writes lab/lockbox.jsonl.
Run from sports/: ../.venv_quant/bin/python results/e4_evidence/t1_e1_ci.py"""
import sys, json, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "..")
import kalshi_temp_maker as K

sha = hashlib.sha256(K.PICKS_S.read_bytes() + K.TRADES_S.read_bytes()).hexdigest()[:12]
assert sha == "e980f9c66792", sha
p = pd.read_parquet(K.PICKS_S); p["date"] = pd.to_datetime(p["date"])
tr = pd.read_parquet(K.TRADES_S)
mk, bet, filled = K.variant(p, tr, None, "M3_N10")
q = K.maker_price(p, "3")
out = {"sha": sha, "pooled_reproduced": dict(picks=len(p), fill=round(filled.mean(), 4), roi=round(mk[bet].mean(), 6))}
m = (p.cohort == "E1").to_numpy()
e = p[m].reset_index(drop=True); mk, f, q = mk[m], filled[m], q[m]
won = e.won.to_numpy()
rec = dict(picks=len(e), fill=round(float(f.mean()), 4), roi=round(float(mk[f].mean()), 4))
assert rec == {"picks": 1844, "fill": 0.8574, "roi": 0.0287}, rec
out["E1_reproduced"] = rec
out["E1_filled_losses"] = int(((won == 0) & f).sum()); out["E1_unfilled_losses"] = int(((won == 0) & ~f).sum())
out["E1_n_dates"] = int(e.date.nunique()); out["E1_date_range"] = [str(e.date.min().date()), str(e.date.max().date())]
hi = q >= 0.98 - 1e-9
out["E1_fill_by_q"] = {"q>=.98": [int(f[hi].sum()), int(hi.sum())], "q<.98": [int(f[~hi].sum()), int((~hi).sum())]}
out["E1_roi_by_q_filled"] = {"q>=.98": float(mk[f & hi].mean()), "q<.98": float(mk[f & ~hi].mean())}

dates = e.date.dt.strftime("%Y-%m-%d").to_numpy()
u, inv = np.unique(dates, return_inverse=True)
B = 4000
idx = np.random.default_rng(0).integers(0, len(u), size=(B, len(u)))
QS = (.025, .05, .5, .95, .975)


def ci(x, d):
    sx = np.bincount(inv, weights=x, minlength=len(u)); sd = np.bincount(inv, weights=d, minlength=len(u))
    r = sx[idx].sum(1) / sd[idx].sum(1)
    return [round(float(np.quantile(r, a)), 5) for a in QS]


out["quantiles"] = list(QS)
out["E1_sealed_roi_per_unit_cost"] = dict(point=float(mk[f].mean()), ci=ci(mk, f.astype(float)))
pn1 = np.where(won == 1, 1 - q, -q) * f  # 1 contract each, ROI per $ staked
out["E1_sealed_roi_per_dollar_1ct"] = dict(point=float(pn1.sum() / (q * f).sum()), ci=ci(pn1, q * f))

fl = np.where(f & (won == 0))[0]; fw = np.where(f & (won == 1))[0]; fa = np.where(f)[0]
for f_live in (0.617, 0.68):
    n_keep = int(round(f_live * len(e)))
    res = {"A_adverse": [], "B_neutral": [], "C_q98_only": []}
    rng = np.random.default_rng(1)
    # C: shortfall placed on q>=.98 winners only (live: q<.98 filled 14/17, q>=.98 15/30)
    fw98 = np.where(f & (won == 1) & hi)[0]
    for b in range(B):
        ka = np.zeros(len(e), bool); ka[fl] = True; ka[rng.choice(fw, n_keep - len(fl), replace=False)] = True
        kb = np.zeros(len(e), bool); kb[rng.choice(fa, n_keep, replace=False)] = True
        kc = f.copy(); drop = len(fa) - n_keep; kc[rng.choice(fw98, min(drop, len(fw98)), replace=False)] = False
        for name, k in (("A_adverse", ka), ("B_neutral", kb), ("C_q98_only", kc)):
            sx = np.bincount(inv, weights=mk * k, minlength=len(u)); sd = np.bincount(inv, weights=k.astype(float), minlength=len(u))
            res[name].append(sx[idx[b]].sum() / sd[idx[b]].sum())
    pa = float((mk[fl].sum() + mk[fw].mean() * (n_keep - len(fl))) / n_keep)
    out[f"E1_filladj_f{f_live}"] = {"A_point": pa, **{k: [round(float(np.quantile(v, a)), 5) for a in QS] for k, v in res.items()}}
json.dump(out, open("results/e4_evidence/t1_e1_ci.json", "w"), indent=1)
print(json.dumps(out, indent=1))
