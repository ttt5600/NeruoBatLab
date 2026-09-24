"""Paired bootstrap over BIRDS -- the only independent unit -- for run15 vs the field.

Writes analysis/run15_bootstrap.json so figures read the intervals instead of retyping them.
"""
import sys, json, warnings, numpy as np
from pathlib import Path
sys.path.insert(0, "zfeval/experiments"); sys.path.insert(0, "zfeval")
warnings.filterwarnings("ignore")
from calltype11 import collect, KEEP11
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
rows = [r for r in collect() if r[3] in KEEP11]
birds = np.array([r[1].lower() for r in rows])
tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
y = np.array([classes.index(t) for t in tt])

MODELS = {"run11": ("ct11_run11_emb.npy", 3), "run15_combined": ("ct11_run15_combined_emb.npy", 3),
          "aves-base-bio": ("ct11_aves_emb.npy", 3), "aves-base-core": ("ct11_aves-base-core_emb.npy", 3),
          "birdaves-biox-base": ("ct11_birdaves-biox-base_emb.npy", 1)}

def per_clip(f, layer):
    E = np.ascontiguousarray(np.load(FEAT / f)[:, layer])
    ok = np.zeros(len(y), bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(E, y, birds):
        sc = StandardScaler().fit(E[tr])
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial").fit(sc.transform(E[tr]), y[tr])
        ok[te] = clf.predict(sc.transform(E[te])) == y[te]
    return ok

R = {}
for m, (f, l) in MODELS.items():
    R[m] = per_clip(f, l); print(f"  {m:<22} acc {R[m].mean():.4f}  (layer {l})")

rng = np.random.default_rng(0); ub = np.unique(birds); B = 2000
bidx = [np.concatenate([np.where(birds == b)[0] for b in rng.choice(ub, len(ub), replace=True)])
        for _ in range(B)]
def boot(m): return np.array([R[m][i].mean() for i in bidx])

print("\npaired bootstrap over 48 birds, 2000 resamples")
OUTJ = {"note": "delta = run15 minus the named model; paired bootstrap over the 48 birds, 2000 resamples, seed 0", "acc": {m: float(R[m].mean()) for m in R}, "vs": {}}
base = "run15_combined"
for other in ["run11", "aves-base-bio", "aves-base-core", "birdaves-biox-base"]:
    d = boot(base) - boot(other)
    lo, hi = np.percentile(d, [2.5, 97.5])
    obs = R[base].mean() - R[other].mean()
    v = "RESOLVED" if (lo > 0 or hi < 0) else "not distinguishable"
    print(f"  run15 - {other:<22}{obs:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  {v}")
    OUTJ["vs"][other] = dict(delta=float(obs), lo=float(lo), hi=float(hi), resolved=bool(lo > 0 or hi < 0))
import json as _j
(Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis/run15_bootstrap.json").write_text(_j.dumps(OUTJ, indent=2))
print("wrote run15_bootstrap.json")
