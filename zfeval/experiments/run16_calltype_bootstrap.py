"""run16 (run15 + 4 GPUs realised = 4x audio per update) vs run11, run15 and all six AVES checkpoints.

Paired bootstrap over the 48 birds (the only independent unit), 2000 resamples, seed 0 -- same
procedure as aves_all_vs_run11_bootstrap.py. Every model at its own best 11-class layer, so this is
the scoreboard's footing. run16's best layer (7) was picked on the reported metric, so it is ALSO
scored at layer 3 -- the layer run11 and run15 use -- as a no-selection check.

Writes analysis/run16_bootstrap.json.
"""
import sys, json, warnings, numpy as np
from pathlib import Path
sys.path.insert(0, "zfeval/experiments"); sys.path.insert(0, "zfeval")
warnings.filterwarnings("ignore")
from calltype11 import collect, KEEP11
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

A = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
VAR = json.loads((A / "aves_variants_calltype.json").read_text())
R15 = json.loads((A / "run15_calltype.json").read_text())
R16 = json.loads((A / "run16_compute4x_calltype.json").read_text())
rows = [r for r in collect() if r[3] in KEEP11]
birds = np.array([r[1].lower() for r in rows])
tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
y = np.array([classes.index(t) for t in tt])
FILE = {"run11": "ct11_run11_emb.npy", "aves-base-bio": "ct11_aves_emb.npy"}

LAYER = {m: VAR["models"][m]["best_11"]["layer"] for m in VAR["models"]}
LAYER["run15_combined"] = R15["best_layer11"]
LAYER["run16_compute4x"] = R16["best_layer11"]


def per_clip(name, layer):
    E = np.ascontiguousarray(np.load(FEAT / FILE.get(name, f"ct11_{name}_emb.npy"))[:, layer])
    ok = np.zeros(len(y), bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(E, y, birds):
        sc = StandardScaler().fit(E[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(E[tr]), y[tr])
        ok[te] = clf.predict(sc.transform(E[te])) == y[te]
    return ok


R = {m: per_clip(m, l) for m, l in LAYER.items()}
R["run16_compute4x@L3"] = per_clip("run16_compute4x", 3)
rng = np.random.default_rng(0); ub = np.unique(birds)
idx = [np.concatenate([np.where(birds == b)[0] for b in rng.choice(ub, len(ub))]) for _ in range(2000)]
BOOT = {m: np.array([ok[i].mean() for i in idx]) for m, ok in R.items()}

out = {"note": "delta = run16 minus the named model; paired bootstrap over 48 birds, 2000 resamples, "
               "seed 0; every model at its own best 11-class layer",
       "layers": {**LAYER, "run16_compute4x@L3": 3},
       "acc": {m: float(ok.mean()) for m, ok in R.items()}, "vs": {}}
others = ["run11", "run15_combined"] + [m for m in VAR["models"] if m != "run11"]
for base in ["run16_compute4x", "run16_compute4x@L3"]:
    out["vs"][base] = {}
    print(f"\n{base}  acc {R[base].mean():.4f}")
    for o in others:
        d = BOOT[base] - BOOT[o]
        lo, hi = np.percentile(d, [2.5, 97.5]); obs = R[base].mean() - R[o].mean()
        res = bool(lo > 0 or hi < 0)
        out["vs"][base][o] = dict(delta=float(obs), lo=float(lo), hi=float(hi), resolved=res)
        print(f"  minus {o:<22} (L{LAYER[o]:>2}, {R[o].mean():.4f})  {obs:+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  {'RESOLVED' if res else 'not distinguishable'}")
(A / "run16_bootstrap.json").write_text(json.dumps(out, indent=2))
print("\nwrote run16_bootstrap.json")
