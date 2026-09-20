#!/usr/bin/env python
"""Ground truth exact to the sample, so model timing error can be separated from annotator jitter.

Every timing number we have is measured against hand-drawn SoundSep boundaries. Those boundaries
have their own error, and it is unmeasured, so "onset |err| 8.8 ms" could be a good model judged
against sloppy labels or a sloppy model judged against good labels. There is no way to tell from
real data alone.

The fix is to build the audio. A real call is windowed to zero outside its own energy support and
mixed into real call-free background at a KNOWN sample offset. The true onset is then the first
sample where the added signal is nonzero -- true by construction, with no annotator in the loop --
and the true offset likewise. That makes three things measurable for the first time:

  * onset precision with no label noise, which upper-bounds how much of the 8.8 ms is the model
  * OFFSET precision, which every previous run reported only at the 20 ms grid
  * whether the +20 ms offset bias / 1.20 duration ratio is the model running long or the
    annotator stopping early

Two conditions separate intrinsic acuity from crowding: ISOLATED spaces calls far apart, NATURAL
draws gaps from the real inter-call distribution (p5 = 16 ms, below one frame).

Guards against measuring our own artifacts:
  * the probe is trained ONLY on real frames from the first 60% of the recording; donors come from
    the last 40%, so no synthetic call or background was seen in probe training
  * background is taken >= 200 ms from any annotation, in the one recording that was annotated
    EXHAUSTIVELY -- elsewhere "unannotated" is 62.6% contaminated and would poison the negatives
  * background segments are crossfaded at seams and every seam position is kept, so false
    positives can be checked for clustering at our own splices
  * the call taper is 1 ms, 1/20th of a frame, reported so it is auditable against the ~8 ms result
"""
from __future__ import annotations
import csv, gc, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zfeval import metrics as mx                                           # noqa: E402
from subframe_onset import events_from_curve                               # noqa: E402

SR, SRC_SR, HOP, RF = 16000, 44100, 320, 400
SPAN = int(round(79434253 * SR / SRC_SR))
AUDIO = Path.home() / "zf_labelset/audio/111021-000.wav"
SEGMENTS = Path.home() / "Downloads/Segments Data.csv"
WEIGHTS = ROOT.parent / "release/zf_hubert_run11/weights/zf_hubert_run11_encoder.pt"
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"

PROBE_FRAC = 0.60            # probe trains on real frames below this fraction of the recording
TAPER_MS = 1.0               # raised-cosine edge on the inserted call
XFADE_MS = 20.0              # crossfade between stitched background segments
BG_CLEAR_MS = 200.0          # background must be this far from any annotation
CANVAS_SEC = 120.0
SNRS = [20.0, 10.0, 5.0, 0.0, -5.0, -10.0]
LAYERS = [0, 6]
SEED = 11021

# Why there are two truth modes.
#   CORE  the inserted signal IS the loud core and truth is its edges. Boundaries are unambiguous,
#         and this measures the encoder's acuity on a step-like edge.
#   TAILS the inserted signal keeps the call's quiet onset/offset tails, but truth is still the loud
#         core. That is the geometry of the real annotation: a human marks the core, while the audio
#         keeps decaying past the mark. If the +20 ms late-offset bias and 1.20 duration ratio seen
#         on real data reappear HERE and not in CORE, the bias is a property of where the boundary
#         is drawn, not a defect of the model -- and that would explain why finding 023's offset
#         shrink was rejected by the tuner in every fold.
TRUTH_FRAC, SIGNAL_FRAC = 0.5, 0.12


def load_intervals():
    rows = list(csv.DictReader(open(SEGMENTS)))
    iv = np.array([[float(r["StartIndex"]), float(r["StopIndex"])] for r in rows])
    iv = iv[~np.isnan(iv).any(1)]
    iv = iv[iv[:, 1] > iv[:, 0]] * SR / SRC_SR
    iv = np.clip(iv, 0, SPAN)
    iv = iv[np.argsort(iv[:, 0])]
    out = [iv[0]]
    for a, b in iv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append(np.array([a, b]))
    return np.array(out)


def taper(x, n):
    """Raised-cosine fade on both ends, in place on a copy."""
    x = x.copy()
    if n > 0 and len(x) > 2 * n:
        w = 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))
        x[:n] *= w
        x[-n:] *= w[::-1]
    return x


def harvest_calls(rec, iv, lo_s, rng, n_max=400, mode="core"):
    """Clean, isolated donor calls, windowed to zero outside their own energy support.

    The support is found on the excerpt's own envelope rather than from the annotation, so the
    inserted signal's boundaries do not inherit the annotator's.
    """
    pad = int(0.060 * SR)
    donors = []
    for a, b in iv:
        a, b = int(a), int(b)
        dur = (b - a) / SR
        if a < lo_s or not (0.040 <= dur <= 0.250):
            continue
        if a - pad < 0 or b + pad >= len(rec):
            continue
        seg = rec[a - pad:b + pad]
        # local SNR: call span vs the two pads flanking it
        call = seg[pad:-pad]
        bg = np.concatenate([seg[:pad], seg[-pad:]])
        rc, rb = float(np.sqrt((call ** 2).mean())), float(np.sqrt((bg ** 2).mean()))
        if rb <= 0 or rc <= 0:
            continue
        snr = 20 * np.log10(rc / rb)
        if snr < 10.0:                       # only clean donors, so SNR is set by us not inherited
            continue
        # energy support on a 2 ms envelope, threshold halfway (in dB) between bg and peak
        w = int(0.002 * SR)
        env = np.sqrt(np.convolve(seg ** 2, np.ones(w) / w, mode="same"))
        edb = 20 * np.log10(env + 1e-12)
        thr = 20 * np.log10(rb + 1e-12) + 0.5 * (edb.max() - 20 * np.log10(rb + 1e-12))
        on = np.where(edb > thr)[0]
        if len(on) < w:
            continue
        s0, s1 = int(on[0]), int(on[-1]) + 1
        if (s1 - s0) / SR < 0.030:
            continue
        if mode == "core":
            donors.append((taper(seg[s0:s1], int(TAPER_MS / 1000 * SR)), 0, s1 - s0))
        else:
            # a looser threshold defines the SIGNAL extent; the core above stays the TRUTH
            base = 20 * np.log10(rb + 1e-12)
            thr2 = base + SIGNAL_FRAC * (edb.max() - base)
            on2 = np.where(edb > thr2)[0]
            w0, w1 = int(on2[0]), int(on2[-1]) + 1
            if w0 > s0 or w1 < s1:
                continue
            donors.append((taper(seg[w0:w1], int(TAPER_MS / 1000 * SR)), s0 - w0, s1 - w0))
    rng.shuffle(donors)
    return donors[:n_max]


def harvest_background(rec, iv, lo_s, min_len_s=1.5):
    """Gaps in the exhaustive annotation, backed off by BG_CLEAR_MS on both sides."""
    clear = int(BG_CLEAR_MS / 1000 * SR)
    segs = []
    edges = np.concatenate([[0], iv.reshape(-1), [SPAN]]).astype(int)
    for k in range(0, len(edges) - 1, 2):   # even k are GAPS: [0,iv0), [iv0_end,iv1), ...
        a, b = edges[k] + clear, edges[k + 1] - clear
        if a < lo_s:
            continue
        if (b - a) / SR >= min_len_s:
            segs.append((a, b))
    return segs


def build_canvas(rec, donors, bg_segs, snr_db, condition, rng):
    """Stitch background, crossfade seams, insert calls at known sample offsets."""
    n = int(CANVAS_SEC * SR)
    xf = int(XFADE_MS / 1000 * SR)
    win = 0.5 * (1 - np.cos(np.pi * np.arange(xf) / xf))
    canvas = np.zeros(n, dtype=np.float32)
    seams, pos = [], 0
    while pos < n:
        a, b = bg_segs[rng.integers(len(bg_segs))]
        seg = rec[a:b]
        take = min(len(seg), n - pos + xf)
        seg = seg[:take]
        if pos == 0:
            canvas[:len(seg)] = seg
            pos = len(seg)
        else:
            ov = min(xf, len(seg), n - pos + xf)
            st = pos - ov
            canvas[st:st + ov] = canvas[st:st + ov] * win[:ov][::-1] + seg[:ov] * win[:ov]
            rest = seg[ov:]
            e = min(n, st + ov + len(rest))
            canvas[st + ov:e] = rest[:e - st - ov]
            seams.append(st)
            pos = e
        if len(seg) < xf:
            break
    bg_only = canvas.copy()

    # insertion positions
    guard = int(0.5 * SR)
    truth, t = [], guard
    di = 0
    while t < n - guard and di < len(donors):
        c, t_on, t_off = donors[di]; di += 1
        L = len(c)
        if t + L >= n - guard:
            break
        seg_bg = canvas[t:t + L]
        rb = float(np.sqrt((seg_bg ** 2).mean()))
        rc = float(np.sqrt((c ** 2).mean()))
        if rb <= 0 or rc <= 0:
            continue
        g = (rb * 10 ** (snr_db / 20)) / rc
        canvas[t:t + L] += g * c
        truth.append([(t + t_on) / SR, (t + t_off) / SR])
        if condition == "isolated":
            gap = rng.uniform(0.6, 1.4)
        else:                                # natural: real inter-call gap distribution
            gap = float(np.clip(rng.lognormal(np.log(0.09), 1.0), 0.016, 3.0))
        t = t + L + int(gap * SR)
    return canvas, bg_only, np.array(truth), np.array(seams, dtype=int)


def frame_pass(model, x, device, layers=LAYERS, chunk_sec=20.0):
    nF = (len(x) - RF) // HOP + 1
    F = {l: np.zeros((nF, 768), dtype=np.float32) for l in layers}
    C = int(chunk_sec * SR)
    filled = 0
    for c in range((len(x) + C - 1) // C):
        lo = c * C
        seg = x[lo:min(len(x), lo + C + RF - HOP)]
        if len(seg) < RF:
            break
        mu, var = float(seg.mean()), float(seg.var())
        xin = ((seg - mu) / np.sqrt(var + 1e-5)).astype(np.float32)
        with torch.no_grad():
            feats, _ = model.extract_features(torch.from_numpy(xin).unsqueeze(0).to(device), None)
        i0 = lo // HOP
        take = min(feats[0].shape[1], C // HOP, nF - i0)
        for l in layers:
            F[l][i0:i0 + take] = feats[l][0, :take].cpu().numpy()
        filled += take
    if filled < nF - 2:
        raise RuntimeError(f"only filled {filled}/{nF}")
    return F, nF


def match_both(pred, true, tol_s):
    """One-to-one by nearest onset within tolerance; returns signed onset AND offset errors."""
    used = np.zeros(len(pred), bool)
    on_e, off_e, dur_p, dur_t = [], [], [], []
    for ts, te in true:
        best, bd = -1, tol_s + 1
        for j, (ps, pe) in enumerate(pred):
            if used[j]:
                continue
            d = abs(ps - ts)
            if d < bd:
                best, bd = j, d
        if best >= 0 and bd <= tol_s:
            used[best] = True
            ps, pe = pred[best]
            on_e.append(ps - ts); off_e.append(pe - te)
            dur_p.append(pe - ps); dur_t.append(te - ts)
    return (np.array(on_e), np.array(off_e), np.array(dur_p), np.array(dur_t), used)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-mode", choices=["core", "tails"], default="core")
    ap.add_argument("--conditions", default="isolated,natural")
    ap.add_argument("--snrs", default=None, help="comma list; default all")
    A = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    print(f"[device] {device}", flush=True)

    iv = load_intervals()
    need = min(SPAN + 20 * SR, sf.info(AUDIO).frames)
    rec, _ = sf.read(AUDIO, dtype="float32", frames=need)
    lo_s = int(PROBE_FRAC * SPAN)

    donors = harvest_calls(rec, iv, lo_s, rng, mode=A.truth_mode)
    print(f"[mode] truth={A.truth_mode}"
          + ("  (signal keeps quiet tails; truth is the loud core)" if A.truth_mode == "tails"
             else "  (signal and truth are both the loud core)"))
    bg_segs = harvest_background(rec, iv, lo_s)
    dd = np.array([(d[2] - d[1]) / SR for d in donors])
    print(f"[donors] {len(donors)} calls from the last {100*(1-PROBE_FRAC):.0f}% of the recording, "
          f"duration median {np.median(dd)*1000:.0f} ms  p5 {np.percentile(dd,5)*1000:.0f}  "
          f"p95 {np.percentile(dd,95)*1000:.0f}")
    bl = sum(b - a for a, b in bg_segs) / SR
    print(f"[background] {len(bg_segs)} call-free stretches, {bl:.0f} s total", flush=True)
    if len(donors) < 50 or bl < 60:
        raise RuntimeError("not enough donor material")

    # ---- probe trained ONLY on real frames from the early region
    FR = np.load(FEAT / "frames_30min.npz", allow_pickle=True)
    y_all, centers = FR["y"], FR["centers"]
    tr = centers < lo_s
    print(f"[probe] training on {tr.sum()} real frames (< {PROBE_FRAC:.0%} of recording), "
          f"prevalence {y_all[tr].mean():.4f}", flush=True)
    probes = {}
    for l in LAYERS:
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0))
        est.fit(FR[f"F{l}"][tr].astype(np.float32), y_all[tr])
        probes[l] = est
    del FR; gc.collect()

    from torchaudio.models import hubert_base
    model = hubert_base()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu",
                                     weights_only=False)["state_dict"], strict=True)
    model = model.eval().to(device)

    # decoder settings held FIXED across all conditions -- tuning per SNR would manufacture the curve
    DEC = dict(thr=0.5, min_dur_s=0.030, merge_gap_s=0.0, smooth=1)
    out = {"config": dict(taper_ms=TAPER_MS, xfade_ms=XFADE_MS, bg_clear_ms=BG_CLEAR_MS,
                          canvas_sec=CANVAS_SEC, probe_frac=PROBE_FRAC, seed=SEED,
                          decoder=DEC, snrs=SNRS, layers=LAYERS,
                          n_donors=len(donors), background_sec=bl),
           "conditions": {}}

    t0 = time.time()
    snr_list = [float(x) for x in A.snrs.split(",")] if A.snrs else SNRS
    out["config"]["truth_mode"] = A.truth_mode
    out["config"]["snrs"] = snr_list
    for cond in A.conditions.split(","):
        out["conditions"][cond] = {}
        for snr in snr_list:
            r2 = np.random.default_rng(SEED + int(snr * 10) + (0 if cond == "isolated" else 7))
            canvas, bg_only, truth, seams = build_canvas(rec, donors, bg_segs, snr, cond, r2)
            F, nF = frame_pass(model, canvas, device)
            t = (np.arange(nF) * HOP + RF / 2) / SR
            gaps = np.diff(truth[:, 0]) - (truth[:-1, 1] - truth[:-1, 0])
            ent = {"n_true": int(len(truth)),
                   "median_gap_ms": float(np.median(gaps) * 1000) if len(gaps) else None,
                   "p5_gap_ms": float(np.percentile(gaps, 5) * 1000) if len(gaps) else None,
                   "layers": {}}
            for l in LAYERS:
                p = probes[l].predict_proba(F[l])[:, 1]
                # frame-level truth on the same grid, for AUC/AP
                fy = np.zeros(nF, dtype=int)
                for a, b in truth:
                    fc = t
                    fy[(fc >= a) & (fc <= b)] = 1
                s = mx.score(fy, p, "synthetic")
                pred = events_from_curve(t, p, DEC["thr"], DEC["min_dur_s"],
                                         DEC["merge_gap_s"], DEC["smooth"], interp=True)
                on_e, off_e, dp, dt_, used = match_both(pred, truth, tol_s=0.050)
                fp = pred[~used] if len(pred) else np.zeros((0, 2))
                seam_t = seams / SR
                near_seam = 0
                if len(fp) and len(seam_t):
                    near_seam = int(sum(np.min(np.abs(seam_t - s0)) < 0.050 for s0 in fp[:, 0]))
                ent["layers"][f"L{l}"] = dict(
                    frame_auc=s.auc, frame_ap=s.ap,
                    n_pred=int(len(pred)), n_match=int(len(on_e)),
                    recall=float(len(on_e) / max(len(truth), 1)),
                    precision=float(len(on_e) / max(len(pred), 1)),
                    onset_mae_ms=float(np.median(np.abs(on_e)) * 1000) if len(on_e) else None,
                    onset_bias_ms=float(np.median(on_e) * 1000) if len(on_e) else None,
                    onset_p90_ms=float(np.percentile(np.abs(on_e), 90) * 1000) if len(on_e) else None,
                    offset_mae_ms=float(np.median(np.abs(off_e)) * 1000) if len(off_e) else None,
                    offset_bias_ms=float(np.median(off_e) * 1000) if len(off_e) else None,
                    offset_p90_ms=float(np.percentile(np.abs(off_e), 90) * 1000) if len(off_e) else None,
                    duration_ratio_median=float(np.median(dp / dt_)) if len(dp) else None,
                    n_fp=int(len(fp)), fp_near_seam=near_seam,
                    seam_frac=float(near_seam / max(len(fp), 1)))
                e = ent["layers"][f"L{l}"]
                print(f"  {cond:9s} {snr:+5.0f} dB L{l}  AUC {s.auc:.3f} AP {s.ap:.3f}  "
                      f"R {e['recall']:.3f} P {e['precision']:.3f}  "
                      f"on {e['onset_mae_ms'] or float('nan'):5.1f} ({e['onset_bias_ms'] or float('nan'):+5.1f})  "
                      f"off {e['offset_mae_ms'] or float('nan'):5.1f} ({e['offset_bias_ms'] or float('nan'):+5.1f})  "
                      f"dur×{e['duration_ratio_median'] or float('nan'):.2f}  "
                      f"seamFP {near_seam}/{len(fp)}  [{time.time()-t0:.0f}s]", flush=True)
            # background-only false-alarm floor, same decoder, once per condition
            if snr == snr_list[0]:
                Fb, nb = frame_pass(model, bg_only, device)
                tb = (np.arange(nb) * HOP + RF / 2) / SR
                ent["background_only"] = {}
                for l in LAYERS:
                    pb = probes[l].predict_proba(Fb[l])[:, 1]
                    pe = events_from_curve(tb, pb, DEC["thr"], DEC["min_dur_s"],
                                           DEC["merge_gap_s"], DEC["smooth"], interp=True)
                    ent["background_only"][f"L{l}"] = dict(
                        n_false_events=int(len(pe)),
                        per_min=float(len(pe) / (len(bg_only) / SR / 60)),
                        frac_frames_over_thr=float((pb > DEC["thr"]).mean()))
                    print(f"    [bg-only] L{l}: {len(pe)} false events "
                          f"({ent['background_only'][f'L{l}']['per_min']:.1f}/min), "
                          f"{100*ent['background_only'][f'L{l}']['frac_frames_over_thr']:.1f}% frames over thr",
                          flush=True)
                del Fb; gc.collect()
            out["conditions"][cond][f"{snr}"] = ent
            del F, canvas, bg_only; gc.collect()

    sfx = "" if A.truth_mode == "core" else "_tails"
    (ANA / f"synth_timing{sfx}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ANA/f'synth_timing{sfx}.json'}")


if __name__ == "__main__":
    main()
