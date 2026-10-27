import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans
D = Path("/Users/jonathanwang/.claude/jobs/63c218d9/tmp")
A, B, E = (np.load(D / f"{n}.npy") for n in ("A_train_s11", "B_train_s22", "E_eval"))
def go(km, tag):
    km.fit(A); la, lb, le = km.predict(A), km.predict(B), km.predict(E)
    inert = float(((A - km.cluster_centers_[la]) ** 2).sum(1).mean())
    print(f"{tag:28s} A_vs_B {np.mean(la!=lb):.4f}  A_vs_E {np.mean(la!=le):.4f}  mean sq dist {inert:.2f}", flush=True)
for rs in (7, 9):
    go(KMeans(n_clusters=200, n_init=1, max_iter=300, random_state=rs, algorithm="lloyd"), f"Lloyd rs{rs}")
for rs in (0, 1, 2):
    go(MiniBatchKMeans(n_clusters=200, batch_size=8192, n_init=3, random_state=rs, max_iter=200), f"MiniBatch rs{rs}")
