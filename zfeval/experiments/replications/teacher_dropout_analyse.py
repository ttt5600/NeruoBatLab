"""Label-change rates under a k=200 codebook fit on train-mode pass A. Full Lloyd k-means (not MiniBatch) as primary."""
import json, time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score as ami

D = Path("/Users/jonathanwang/.claude/jobs/63c218d9/tmp")
A, B, E = (np.load(D / f"{n}.npy") for n in ("A_train_s11", "B_train_s22", "E_eval"))
assert A.shape == B.shape == E.shape and np.isfinite(A).all() and np.isfinite(B).all() and np.isfinite(E).all()
print("frames", A.shape)


def cos(x, y):
    return float(np.mean((x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))))


def chance(la, lb, k):
    pa = np.bincount(la, minlength=k) / len(la); pb = np.bincount(lb, minlength=k) / len(lb)
    return float(1 - (pa * pb).sum())


def rates(km, tag):
    la, lb, le = km.predict(A), km.predict(B), km.predict(E)
    r = dict(codebook=tag, used_clusters=int(len(np.unique(la))),
             changed_A_vs_B=float(np.mean(la != lb)), changed_A_vs_E=float(np.mean(la != le)),
             changed_B_vs_E=float(np.mean(lb != le)),
             ami_A_B=float(ami(la, lb)), ami_A_E=float(ami(la, le)),
             chance_change_A_vs_B=chance(la, lb, 200),
             adjacent_frame_change_A=float(np.mean(la[1:] != la[:-1])),
             adjacent_frame_change_E=float(np.mean(le[1:] != le[:-1])))
    return r, la


out = {"cos_A_B": cos(A, B), "cos_A_E": cos(A, E), "cos_B_E": cos(B, E)}
t0 = time.time()
km = KMeans(n_clusters=200, init="k-means++", n_init=1, max_iter=300, tol=1e-4, random_state=7, algorithm="lloyd").fit(A)
r, la = rates(km, "KMeans lloyd k++ rs7"); r["fit_s"] = round(time.time() - t0); r["n_iter"] = int(km.n_iter_)
out["primary"] = r; print(json.dumps(r), flush=True)

# codebook-seed floor: same data, another full-Lloyd seed
km2 = KMeans(n_clusters=200, n_init=1, max_iter=300, random_state=8, algorithm="lloyd").fit(A)
out["ami_A_codebook_rs7_vs_rs8"] = float(ami(la, km2.predict(A)))
r2, _ = rates(km2, "KMeans lloyd k++ rs8"); out["primary_seed2"] = r2; print(json.dumps(r2), flush=True)

# method variant: the diagnostic's MiniBatchKMeans settings on this stretch (to localise any divergence)
mb = MiniBatchKMeans(n_clusters=200, batch_size=8192, n_init=3, random_state=0, max_iter=200).fit(A)
r3, _ = rates(mb, "MiniBatchKMeans bs8192 ninit3 rs0"); out["minibatch_variant"] = r3; print(json.dumps(r3), flush=True)

# symmetric variant: codebook fit on the eval-mode pass instead
kme = KMeans(n_clusters=200, n_init=1, max_iter=300, random_state=7, algorithm="lloyd").fit(E)
le, la_e, lb_e = kme.predict(E), kme.predict(A), kme.predict(B)
out["codebook_on_E"] = dict(changed_A_vs_B=float(np.mean(la_e != lb_e)), changed_A_vs_E=float(np.mean(la_e != le)))
print(json.dumps(out["codebook_on_E"]))

# shuffled-frame floor: B labels permuted in time
rng = np.random.default_rng(0)
lb = km.predict(B)
out["shuffled_B_change_vs_A"] = float(np.mean(la != rng.permutation(lb)))
print({k: v for k, v in out.items() if not isinstance(v, dict)})
(D / "result.json").write_text(json.dumps(out, indent=1))
