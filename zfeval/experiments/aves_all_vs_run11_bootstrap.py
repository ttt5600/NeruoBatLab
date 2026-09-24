"""Every AVES checkpoint vs run11, individually, paired bootstrap over the 48 birds.

aves_variants_calltype.json only bootstrapped run11 against the BEST checkpoint of each family.
This answers the narrower question: does EACH checkpoint -- including the oldest -- beat run11
by more than the noise? Each model at its own best layer from that file, so the comparison is
on the same footing as the scoreboard.
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
rows = [r for r in collect() if r[3] in KEEP11]
birds = np.array([r[1].lower() for r in rows])
tt = np.array([r[3] for r in rows]); classes = sorted(set(tt))
y = np.array([classes.index(t) for t in tt])
FILE = {"run11": "ct11_run11_emb.npy", "aves-base-bio": "ct11_aves_emb.npy"}

def per_clip(name):
    f = FILE.get(name, f"ct11_{name}_emb.npy")
    layer = VAR["models"][name]["best_11"]["layer"]
    E = np.ascontiguousarray(np.load(FEAT / f)[:, layer])
    ok = np.zeros(len(y), bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(E, y, birds):
        sc = StandardScaler().fit(E[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(E[tr]), y[tr])
        ok[te] = clf.predict(sc.transform(E[te])) == y[te]
    return ok, layer

R = {m: per_clip(m) for m in VAR["models"]}
rng = np.random.default_rng(0); ub = np.unique(birds)
idx = [np.concatenate([np.where(birds == b)[0] for b in rng.choice(ub, len(ub))]) for _ in range(2000)]
base = np.array([R["run11"][0][i].mean() for i in idx])
out = {"note": "delta = checkpoint minus run11, paired bootstrap over 48 birds, 2000 resamples", "vs_run11": {}}
print(f"{'checkpoint':<22}{'released':>10}{'layer':>7}{'acc':>8}{'minus run11':>13}   95% interval")
for m in VAR["models"]:
    if m == "run11": continue
    ok, layer = R[m]
    d = np.array([ok[i].mean() for i in idx]) - base
    lo, hi = np.percentile(d, [2.5, 97.5]); obs = ok.mean() - R["run11"][0].mean()
    rel = "2022 AVES" if m.startswith("aves") else "BirdAVES"
    out["vs_run11"][m] = dict(layer=layer, acc=float(ok.mean()), delta=float(obs), lo=float(lo), hi=float(hi),
                              resolved=bool(lo > 0 or hi < 0))
    print(f"{m:<22}{rel:>10}{layer:>7}{ok.mean():>8.4f}{obs:>+13.4f}   [{lo:+.4f}, {hi:+.4f}]  "
          f"{'REAL' if lo > 0 else 'within noise'}")
(A / "aves_all_vs_run11_bootstrap.json").write_text(json.dumps(out, indent=2))
