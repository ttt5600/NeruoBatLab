"""T2b (added after pre-reg, disclosed): X1-cohort CI re-derived from the frozen sealed inputs of lockbox
kalshi_temp_maker_M3_N10, same guard as t1 (sha + recorded X1 numbers asserted; no lockbox write).
Run from sports/: ../.venv_quant/bin/python results/e4_evidence/t2b_x1_sealed_ci.py"""
import sys, json, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, "..")
import kalshi_temp_maker as K

sha = hashlib.sha256(K.PICKS_S.read_bytes() + K.TRADES_S.read_bytes()).hexdigest()[:12]
assert sha == "e980f9c66792", sha
p = pd.read_parquet(K.PICKS_S); p["date"] = pd.to_datetime(p["date"])
mk, bet, f = K.variant(p, pd.read_parquet(K.TRADES_S), None, "M3_N10")
m = (p.cohort == "X1").to_numpy()
x = p[m]; mk, f = mk[m], f[m]
rec = dict(picks=len(x), fill=round(float(f.mean()), 4), roi=round(float(mk[f].mean()), 4))
assert rec == {"picks": 2464, "fill": 0.761, "roi": 0.0167}, rec
u, inv = np.unique(x.date.dt.strftime("%Y-%m-%d").to_numpy(), return_inverse=True)
idx = np.random.default_rng(0).integers(0, len(u), size=(4000, len(u)))
sx = np.bincount(inv, weights=mk, minlength=len(u)); sd = np.bincount(inv, weights=f.astype(float), minlength=len(u))
r = sx[idx].sum(1) / sd[idx].sum(1)
out = {"X1_reproduced": rec, "n_dates": len(u), "losses_filled": int(((x.won.to_numpy() == 0) & f).sum()),
       "ci90": [round(float(np.quantile(r, a)), 5) for a in (.05, .95)],
       "ci95": [round(float(np.quantile(r, a)), 5) for a in (.025, .975)]}
json.dump(out, open("results/e4_evidence/t2b_x1_sealed_ci.json", "w"), indent=1)
print(json.dumps(out))
