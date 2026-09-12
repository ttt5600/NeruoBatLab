"""Frame probabilities -> vocalization intervals, and honest event-level scoring.

Overlap-only matching flatters any detector that emits long intervals: on run11 layer 6 it reads
F1 0.840 and the same predictions score 0.736 under a 20 ms onset collar. Both are reported.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import median_filter

HOP, SR = 320, 16000
MS_PER_FRAME = HOP / SR * 1000.0


def runs(mask):
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return np.stack([np.where(d == 1)[0], np.where(d == -1)[0]], 1)


def postproc(iv, min_dur, merge_gap):
    if len(iv) == 0:
        return iv
    if merge_gap > 0 and len(iv) > 1:
        out = [iv[0].tolist()]
        for s, e in iv[1:]:
            if s - out[-1][1] <= merge_gap:
                out[-1][1] = e
            else:
                out.append([s, e])
        iv = np.array(out)
    return iv[(iv[:, 1] - iv[:, 0]) >= min_dur]


def decode(p, thr, min_dur, merge_gap, smooth, thr_low=None, shrink=0):
    """Smooth -> threshold (optionally with hysteresis) -> min duration -> bridge gaps -> shrink."""
    q = median_filter(p, size=int(smooth), mode="nearest") if smooth > 1 else p
    if thr_low is None or thr_low >= thr:
        iv = runs(q > thr)
    else:
        cand = runs(q > thr_low)
        iv = np.array([[a, b] for a, b in cand if (q[a:b] > thr).any()]) if len(cand) \
            else np.zeros((0, 2), int)
    iv = postproc(iv, min_dur, merge_gap)
    if shrink and len(iv):
        iv = iv.copy()
        iv[:, 1] = np.maximum(iv[:, 0] + 1, iv[:, 1] - shrink)
    return iv


def match(pred, true, tol_frames=None):
    """One-to-one pairing by a two-pointer sweep over sorted, non-overlapping intervals.

    Equivalent to greedy-by-overlap on real data and O(n+m) rather than building and argsorting an
    n x m matrix -- the tuner calls this ~170k times.
    tol_frames=None accepts any overlap; a number requires the onset within that many frames.
    """
    n, m = len(pred), len(true)
    if n == 0 or m == 0:
        return 0, np.array([]), np.array([])
    i = j = 0
    on, off = [], []
    while i < n and j < m:
        ps, pe = pred[i]; ts, te = true[j]
        ok = (pe > ts and te > ps) if tol_frames is None else (abs(ps - ts) <= tol_frames)
        if ok:
            on.append(ps - ts); off.append(pe - te); i += 1; j += 1
        elif (ps + pe) < (ts + te):
            i += 1
        else:
            j += 1
    return len(on), np.array(on, float), np.array(off, float)


def prf(pred, true, tol_frames=None):
    n, on, off = match(pred, true, tol_frames)
    p = n / len(pred) if len(pred) else 0.0
    r = n / len(true) if len(true) else 0.0
    return dict(precision=p, recall=r, f1=(2 * p * r / (p + r) if p + r else 0.0),
                n_pred=len(pred), n_true=len(true), n_match=n,
                onset_mae_ms=float(np.median(np.abs(on)) * MS_PER_FRAME) if len(on) else None,
                onset_bias_ms=float(np.median(on) * MS_PER_FRAME) if len(on) else None,
                offset_mae_ms=float(np.median(np.abs(off)) * MS_PER_FRAME) if len(off) else None,
                offset_bias_ms=float(np.median(off) * MS_PER_FRAME) if len(off) else None)


def overlap_pairs(pred, true):
    pairs, j = [], 0
    for i, (ps, pe) in enumerate(pred):
        while j < len(true) and true[j, 1] <= ps:
            j += 1
        k = j
        while k < len(true) and true[k, 0] < pe:
            if true[k, 1] > ps:
                pairs.append((i, k))
            k += 1
    return pairs


def taxonomy(pred, true):
    """Deletions / insertions / merges / fragmentations. 'Recall 0.81' says nothing about whether
    the misses are isolated or whole runs collapsed into one interval."""
    P = overlap_pairs(pred, true)
    pt, tp = {}, {}
    for i, j in P:
        pt.setdefault(i, []).append(j); tp.setdefault(j, []).append(i)
    return dict(n_pred=len(pred), n_true=len(true),
                deletions=int(sum(1 for j in range(len(true)) if j not in tp)),
                insertions=int(sum(1 for i in range(len(pred)) if i not in pt)),
                merges=int(sum(1 for v in pt.values() if len(v) > 1)),
                fragmentations=int(sum(1 for v in tp.values() if len(v) > 1)))


def tolerance_sweep(pred, true, tolerances_ms=(20, 50, 100, 200, 500)):
    out = {"overlap": prf(pred, true, None)}
    for t in tolerances_ms:
        out[f"{t}ms"] = prf(pred, true, int(round(t / MS_PER_FRAME)))
    return out


def recall_by_duration(pred, true, bins_ms=((0, 50), (50, 80), (80, 120), (120, 200), (200, 1e9))):
    """Short calls are the real weakness: a 50 ms call is 2.5 frames at 20 ms resolution."""
    hit = {j for _, j in overlap_pairs(pred, true)}
    dur = (true[:, 1] - true[:, 0]) * MS_PER_FRAME
    out = {}
    for lo, hi in bins_ms:
        m = np.where((dur >= lo) & (dur < hi))[0]
        out[f"{lo}-{'inf' if hi > 1e8 else hi} ms"] = dict(
            n=int(len(m)), recall=float(np.mean([j in hit for j in m])) if len(m) else None)
    return out


def tune_decoder(p, y_true_intervals, blocks, grid, objective_tol_ms=50):
    """Choose decoder knobs OUT OF FOLD, then apply them unchanged to the held-out block.

    Tuned on overlap-F1 the optimum sits on long sloppy intervals, so the default objective is the
    50 ms collar. If the chosen parameters land on a grid edge the reported score is a floor, not a
    maximum -- that happened twice here -- so the edge is flagged.
    """
    tol = int(round(objective_tol_ms / MS_PER_FRAME))
    blocks = np.asarray(blocks)
    uniq = np.unique(blocks)
    per, chosen = [], []
    for b in uniq:
        best, bf = None, -1.0
        for params in grid:
            f1s = []
            for b2 in uniq:
                if b2 == b:
                    continue
                m = blocks == b2
                lo = np.where(m)[0][0]
                tv = _clip_to(y_true_intervals, np.where(m)[0])
                f1s.append(prf(decode(p[m], **params) + lo, tv, tol)["f1"])
            v = float(np.mean(f1s))
            if v > bf:
                bf, best = v, params
        m = blocks == b
        lo = np.where(m)[0][0]
        tv = _clip_to(y_true_intervals, np.where(m)[0])
        pr = decode(p[m], **best) + lo
        sc = prf(pr, tv, tol); sc_ov = prf(pr, tv, None)
        per.append(dict(block=int(b), params=best, collar=sc, overlap=sc_ov))
        chosen.append(best)
    edges = _grid_edges(chosen, grid)
    agg = {}
    for key in ("precision", "recall", "f1"):
        agg[f"collar_{key}"] = float(np.mean([x["collar"][key] for x in per]))
        agg[f"overlap_{key}"] = float(np.mean([x["overlap"][key] for x in per]))
    return dict(overall=agg, per_block=per, on_grid_edge=edges,
                objective=f"collar {objective_tol_ms} ms")


def _clip_to(intervals, idx):
    lo, hi = idx[0], idx[-1]
    s = intervals[(intervals[:, 0] >= lo) & (intervals[:, 0] <= hi)]
    return np.clip(s, lo, hi + 1)


def _grid_edges(chosen, grid):
    keys = list(grid[0])
    edges = {}
    for k in keys:
        vals = sorted({g[k] for g in grid})
        picked = {c[k] for c in chosen}
        if picked & {vals[0], vals[-1]}:
            edges[k] = dict(picked=sorted(picked), grid_min=vals[0], grid_max=vals[-1],
                            warning="chosen value sits on the grid edge -- widen the grid, "
                                    "the reported score is a floor not a maximum")
    return edges
