#!/usr/bin/env python
"""POST HOC controls for Part 1 (not in the pre-committed list; written after E1-E5 were seen).

Is the ensemble gain specific to combining strong, different encoders, or does averaging any probes help?
  K1 = run16 three seeds (same recipe)            vs best seed
  K2 = run19 two seeds                            vs best seed
  K3 = whisper-base + run16_seed2 (weakest of ours, 0.8198)  vs whisper-base
  K4 = whisper-base + hf-hubert-base-ls960 (stock speech HuBERT, 0.8271) vs whisper-base
  K5 = whisper-base + whisper-small (same family) vs whisper-base
Same protocol as ensemble_calltype.py; 95% intervals only (exploratory).

    python zfeval/experiments/ensemble_tet/ensemble_controls.py -> ensemble_tet/ensemble_controls.json
"""
from __future__ import annotations
import json
import numpy as np
from joblib import Parallel, delayed
from common import task, probe, pct, CPU, N_JOBS, HERE

K = {"K1": ("run16_compute4x", "run16_seed2", "run16_seed3"), "K2": ("run19_avesteacher", "run19_seed2"),
     "K3": ("hf-whisper-base-enc", "run16_seed2"), "K4": ("hf-whisper-base-enc", "hf-hubert-base-ls960"),
     "K5": ("hf-whisper-base-enc", "hf-whisper-small-enc")}


def main():
    y, birds, classes, BOOT = task()
    tags = sorted({t for v in K.values() for t in v})
    P = dict(zip(tags, Parallel(n_jobs=N_JOBS)(delayed(probe)(t, "best", y, birds) for t in tags)))
    hit = lambda p: (p.argmax(1) == y).astype(float)
    res = dict(status="POST HOC controls, not pre-committed", results={})
    for name, mem in K.items():
        best = max(mem, key=lambda t: CPU[t]["acc11_cpu"])
        ha, hb = hit(np.mean([P[t] for t in mem], 0)), hit(P[best])
        bs = np.array([ha[s].mean() - hb[s].mean() for s in BOOT]); lo, hi = pct(bs, 0.05)
        res["results"][name] = dict(members=list(mem), best_member=best, acc_ens=float(ha.mean()), acc_best=float(hb.mean()),
                                    diff=float(ha.mean() - hb.mean()), lo95=lo, hi95=hi)
        print(f"{name} {'+'.join(mem)}: {ha.mean():.4f} vs {best} {hb.mean():.4f} {ha.mean() - hb.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    (HERE / "ensemble_controls.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
