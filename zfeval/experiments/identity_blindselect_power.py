"""Does the identity target resolve the five encoders at all? Bootstrap before believing any rho."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, "zfeval/experiments"); sys.path.insert(0, "zfeval")
import calltype11 as C11, calltype_blindselect as BS
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")

K, NFOLD, SEED, NBOOT = 450, 5, 0, 400
rows = [r for r in C11.collect() if r[3] in C11.KEEP11]
birds = np.array([r[1].lower() for r in rows]); ub = sorted(set(birds))
y = np.array([ub.index(b) for b in birds])
ENC = BS.ENCODERS

per_clip = {}
for nm in ENC:
    E = np.load(C11.FEAT / f"ct11_{nm}_emb.npy")[:, 3].astype(np.float64)
    cv = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    ok = np.zeros(len(y), bool)
    for tr, te in cv.split(E, y):
        sc = StandardScaler().fit(E[tr]); Xtr, Xte = sc.transform(E[tr]), sc.transform(E[te])
        lab = fcluster(linkage(Xtr, method="ward"), t=K, criterion="maxclust") - 1
        C = BS.centroids(Xtr, lab, K)
        maj = np.array([np.bincount(y[tr][lab == c], minlength=len(ub)).argmax()
                        if (lab == c).any() else np.bincount(y[tr]).argmax() for c in range(K)])
        ok[te] = (maj[BS.assign(Xte, C)] == y[te])
    per_clip[nm] = ok
    print(f"  {nm:<20} acc {ok.mean():.4f}")

rng = np.random.default_rng(0); n = len(y)
boot = {nm: [] for nm in ENC}
idx = rng.integers(0, n, size=(NBOOT, n))
for b in range(NBOOT):
    for nm in ENC: boot[nm].append(per_clip[nm][idx[b]].mean())
print("\n[bootstrap over clips, 400 resamples]")
for nm in ENC:
    a = np.array(boot[nm]); lo, hi = np.percentile(a, [2.5, 97.5])
    print(f"  {nm:<20} {per_clip[nm].mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]  width {hi-lo:.4f}")

print("\n[pairwise: is the difference resolvable?]")
order = sorted(ENC, key=lambda e: -per_clip[e].mean())
for i in range(len(order)-1):
    a, b_ = order[i], order[i+1]
    d = np.array(boot[a]) - np.array(boot[b_])
    lo, hi = np.percentile(d, [2.5, 97.5])
    sig = "RESOLVED" if lo > 0 or hi < 0 else "not distinguishable"
    print(f"  {a:<20} - {b_:<20} {per_clip[a].mean()-per_clip[b_].mean():+.4f}  [{lo:+.4f},{hi:+.4f}]  {sig}")

# how fragile is a rho built on these five points?
acc = np.array([per_clip[e].mean() for e in ENC])
print(f"\n[spread] top4 range {np.ptp(acc[np.argsort(-acc)][:4]):.4f}   full range {np.ptp(acc):.4f}")
rh = []
for b in range(NBOOT):
    a = np.array([boot[e][b] for e in ENC])
    rh.append(spearmanr(a, acc).statistic)
rh = np.array(rh)
print(f"[rho of the truth against ITSELF under resampling] median {np.median(rh):+.2f} "
      f"95% [{np.percentile(rh,2.5):+.2f}, {np.percentile(rh,97.5):+.2f}]")
print("   -> this is the ceiling any criterion could reach here; if it is not near +1.0,")
print("      the target cannot rank these encoders and every rho above is noise.")
