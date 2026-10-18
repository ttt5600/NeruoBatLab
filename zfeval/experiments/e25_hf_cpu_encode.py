#!/usr/bin/env python
"""CPU re-encode of the E25 Hugging Face encoders' 11-class call-type clips (helper for calltype_cpu_rescore.py).

calltype_cpu_rescore.py cannot encode these itself: the 3.10 framework interpreter has no transformers, and its
spec()/load_model() path (detection_variants -> HFFrames/WhisperFrames) is the frame-pass interface, which is NOT
the protocol the E25 caches were made with (no feature extractor / do_normalize, a different Whisper frame count,
and no AST at all). The E25 caches came from e25_hf_encoders.embed(); this script calls that same function with
device 'cpu', so the only thing that changes is the device.

    HF_HUB_OFFLINE=1 ~/zf_labelset/envs/e25/bin/python zfeval/experiments/e25_hf_cpu_encode.py [--only t1,t2]
    python zfeval/experiments/calltype_cpu_rescore.py --only <same tags>      # scores with sklearn 1.7.2

Per tag: (1) encode a few LONG clips (>= 10 conv frames, where MPS SDPA is correct) on CPU and require them to match
the MPS cache features/ct11_<tag>_emb.npy (max |diff| relative to the clip's max |value| < --tol); stop if not.
(2) encode all 3412 clips on CPU -> features/ct11_<tag>_cpuemb.npy. (3) per-clip max |CPU - MPS| over layers and
dims, counted as changed at > 1e-2 (calltype_cpu_rescore's threshold), split by clip length in conv frames.
Summary -> analysis/e25_hf_cpu_encode.json. Never overwrites an existing _cpuemb.npy.
"""
from __future__ import annotations
import argparse, json, math, sys, time, warnings
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from calltype11 import collect, KEEP11                                     # noqa: E402
from e25_hf_encoders import MODELS, embed                                  # noqa: E402

FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ANA / "e25_hf_cpu_encode.json"
SR = 16000


def n_samples16(path):
    """Length after aves_calltype.load_audio's resample to 16 kHz (torchaudio Resample: ceil(n * 16000 / sr))."""
    import torchaudio
    i = torchaudio.info(str(path))
    return int(math.ceil(i.num_frames * SR / i.sample_rate)) if i.sample_rate != SR else int(i.num_frames)


def conv_frames(n):
    n = max(n, 400)                                    # embed() pads clips shorter than 400 samples
    return (n - 400) // 320 + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(MODELS))
    ap.add_argument("--n-verify", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rows = [r for r in collect() if r[3] in KEEP11]
    paths = [r[0] for r in rows]
    assert len(paths) == 3412, "cohort changed"
    nfr = np.array([conv_frames(n_samples16(p)) for p in paths])
    long_idx = np.flatnonzero(nfr >= 10)
    rng = np.random.default_rng(0)
    vidx = np.sort(rng.choice(long_idx, size=a.n_verify, replace=False))
    print(f"[cohort] {len(paths)} clips; conv frames: 1 -> {(nfr == 1).sum()}, 2-8 -> {((nfr >= 2) & (nfr <= 8)).sum()}, "
          f">=9 -> {(nfr >= 9).sum()}; verify clips {vidx.tolist()} (frames {nfr[vidx].tolist()})", flush=True)
    res = json.loads(OUT.read_text()) if OUT.exists() else dict(
        why="E25 HF encoders re-encoded on CPU with e25_hf_encoders.embed (MPS SDPA bug for 2-8 frame inputs)",
        models={})
    for tag in [t for t in a.only.split(",") if t]:
        hf_id = MODELS[tag]
        M = np.load(FEAT / f"ct11_{tag}_emb.npy")
        cpu_p = FEAT / f"ct11_{tag}_cpuemb.npy"
        print(f"=== {tag} ({hf_id})  MPS cache {M.shape}", flush=True)
        # (1) long clips must reproduce the MPS cache
        Ev, _ = embed(tag, hf_id, [paths[i] for i in vidx], "cpu")
        dv = np.abs(Ev - M[vidx]).max(axis=(1, 2))
        rel = dv / np.abs(M[vidx]).max(axis=(1, 2))
        print(f"    verify long clips: max|diff| {dv.max():.2e}  max rel {rel.max():.2e}  per clip "
              f"{[f'{x:.1e}' for x in dv]}", flush=True)
        if rel.max() > a.tol:
            print(f"    [{tag}] long clips do NOT reproduce the MPS cache (rel {rel.max():.2e} > {a.tol}); skipped",
                  flush=True)
            res["models"][tag] = dict(verify_ok=False, verify_idx=vidx.tolist(), verify_max_abs=float(dv.max()),
                                      verify_max_rel=float(rel.max()))
            OUT.write_text(json.dumps(res, indent=1))
            continue
        # (2) every clip on CPU
        if cpu_p.exists():
            E = np.load(cpu_p); print(f"    [cache] {cpu_p.name} exists, not re-encoded", flush=True)
        else:
            t0 = time.time()
            E, info = embed(tag, hf_id, paths, "cpu")
            tmp = cpu_p.with_name(cpu_p.stem + ".partial.npy")
            np.save(tmp, E); tmp.rename(cpu_p)
            print(f"    wrote {cpu_p.name} {E.shape} in {time.time() - t0:.0f}s", flush=True)
        assert np.allclose(E[vidx], Ev, atol=1e-5, rtol=0), "full pass disagrees with the verify pass"
        # (3) what changed vs MPS
        d = np.abs(E - M).max(axis=2)                  # (N, layers)
        dc = d.max(1)
        ch = dc > 1e-2
        buckets = {"1": nfr == 1, "2-8": (nfr >= 2) & (nfr <= 8), ">=9": nfr >= 9}
        by = {k: dict(n=int(b.sum()), changed=int((ch & b).sum()), max_abs=float(dc[b].max()),
                      median_abs=float(np.median(dc[b]))) for k, b in buckets.items()}
        res["models"][tag] = dict(hf_id=hf_id, verify_ok=True, verify_idx=vidx.tolist(),
                                  verify_max_abs=float(dv.max()), verify_max_rel=float(rel.max()),
                                  n_clips_changed=int(ch.sum()), max_abs_diff=float(dc.max()),
                                  median_abs_diff=float(np.median(dc)), by_conv_frames=by,
                                  max_abs_diff_per_layer=[float(x) for x in d.max(0)])
        OUT.write_text(json.dumps(res, indent=1))
        print(f"    [{tag}] clips changed (>1e-2) {ch.sum()} / {len(ch)}; max|diff| {dc.max():.3e}, median "
              f"{np.median(dc):.2e}; by frames " + "; ".join(f"{k}: {v['changed']}/{v['n']} max {v['max_abs']:.2e}"
                                                           for k, v in by.items()), flush=True)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
