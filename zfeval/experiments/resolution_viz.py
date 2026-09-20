#!/usr/bin/env python
"""Figures for the temporal-resolution study."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
VIZ = ANA / "viz"
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
CALL, NOISE, ACC, GREY = "#c9430c", "#1a5fa8", "#0a6c7e", "#8a8f98"


def fig_resolution():
    R = json.loads((ANA / "resolution_sweep.json").read_text())
    FB = json.loads((ANA / "frame_baselines.json").read_text())
    sizes = sorted([int(k[1:]) for k in R if k.startswith("w")], reverse=True)
    x = np.arange(len(sizes) + 1)
    lab = [f"{s}" for s in sizes] + ["20\n(frame)"]

    def series(key, ld="center"):
        v = []
        for s in sizes:
            r = R[f"w{s}"].get(ld)
            v.append(r[key] if r else np.nan)
        return np.array(v)

    best_w = np.maximum(series("windowed_L0"), series("windowed_L6"))
    best_c = np.maximum(series("continuous_L0"), series("continuous_L6"))
    mel = series("logmel"); ener = series("logenergy")
    fr = FB["full"]
    best_w = np.append(best_w, np.nan)                       # no windowed arm at one frame
    best_c = np.append(best_c, max(fr["hubert_L0"]["auc"], fr["hubert_L6"]["auc"]))
    mel = np.append(mel, fr["logmel_100ms"]["auc"])
    ener = np.append(ener, fr["logenergy"]["auc"])

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    a = ax[0]
    a.plot(x, best_c, "o-", color=CALL, lw=2.2, label="HuBERT, continuous (full context)")
    a.plot(x, best_w, "s--", color=NOISE, lw=1.9, label="HuBERT, windowed (segment only)")
    a.plot(x, mel, "^-", color="#2a9d3f", lw=1.5, label="log-mel")
    a.plot(x, ener, ":", color=GREY, lw=1.5, label="log-energy")
    a.set_xticks(x); a.set_xticklabels(lab)
    a.set_xlabel("analysis segment (ms)"); a.set_ylabel("AUC")
    a.set_title('"Is the CENTRE of this segment inside a call?"\n'
                "prevalence held near 0.12 at every size, so these are comparable", fontsize=11)
    a.grid(alpha=.3); a.legend(fontsize=9, loc="lower right"); a.set_ylim(0.5, 1.0)

    b = ax[1]
    ctx = np.array([R[f"w{s}"]["center"]["context_benefit"] for s in sizes])
    b.axhline(0, color="#444", lw=1)
    b.bar(np.arange(len(sizes)), ctx, color=[ACC if v > 0 else "#b03060" for v in ctx], width=.62)
    b.set_xticks(np.arange(len(sizes))); b.set_xticklabels([str(s) for s in sizes])
    b.set_xlabel("analysis segment (ms)")
    b.set_ylabel("AUC(continuous) − AUC(windowed)")
    b.set_title("Does surrounding context help or hurt?\n"
                "teal = context helps · red = the isolated segment is better", fontsize=11)
    b.grid(alpha=.3, axis="y")
    for i, v in enumerate(ctx):
        b.text(i, v + (0.004 if v > 0 else -0.010), f"{v:+.3f}", ha="center", fontsize=8.5)
    fig.tight_layout(); fig.savefig(VIZ / "08_resolution_sweep.png", dpi=135); plt.close(fig)
    print("  wrote 08_resolution_sweep.png")


def fig_onset():
    S = json.loads((ANA / "subframe_onset.json").read_text())
    O = json.loads((ANA / "onset_probe.json").read_text())
    arms = S["arms"]
    names = ["coarse_nearest", "coarse_interp", f"shifted_K{S['K']}_nearest", f"shifted_K{S['K']}_interp"]
    pretty = ["20 ms grid\nnearest frame", "20 ms grid\n+ interpolation",
              "5 ms grid\n(4 passes)", "5 ms grid\n+ interpolation"]
    cost = ["1x", "1x", "4x", "4x"]

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.9))
    a = ax[0]
    med = [arms[n]["med_abs_err_ms"] for n in names]
    bars = a.bar(range(4), med, color=[GREY, ACC, GREY, ACC], width=.62)
    for i, (v, c) in enumerate(zip(med, cost)):
        a.text(i, v + .3, f"{v:.1f} ms\n{c} compute", ha="center", fontsize=9)
    a.set_xticks(range(4)); a.set_xticklabels(pretty, fontsize=9)
    a.set_ylabel("median |onset error| (ms)"); a.set_ylim(0, 16)
    a.set_title("Interpolation is free and does most of the work", fontsize=11)
    a.grid(alpha=.3, axis="y")

    b = ax[1]
    bias = [arms[n]["bias_ms"] for n in names]
    b.axhline(0, color="#444", lw=1)
    b.bar(range(4), bias, color=[GREY, ACC, GREY, ACC], width=.62)
    for i, v in enumerate(bias):
        b.text(i, v + (.3 if v > 0 else -.7), f"{v:+.1f}", ha="center", fontsize=9)
    b.set_xticks(range(4)); b.set_xticklabels(pretty, fontsize=9)
    b.set_ylabel("signed bias (ms)")
    b.set_title("The nearest-frame rule reports onsets LATE\nby half a frame; interpolation removes it",
                fontsize=11)
    b.grid(alpha=.3, axis="y")

    c = ax[2]
    tols = [5, 10, 20, 50, 100]
    for n, pr, col, ls in zip(names, pretty, [GREY, ACC, "#6b8fb5", CALL], ["--", "-", "--", "-"]):
        c.plot(tols, [arms[n]["F1_by_tol"][f"{t}ms"] for t in tols], ls, marker="o",
               color=col, lw=1.9, label=pr.replace("\n", " "))
    bp = O["layers"]["L0"]["decoders"]["boundary_peak_parabolic"]
    c.plot(tols, [bp[f"{t}ms"]["F1"] for t in tols], "-", marker="s", color="#7b3fa0", lw=1.6,
           label="L0 boundary probe + peak-pick")
    c.set_xscale("log"); c.set_xticks(tols); c.set_xticklabels([str(t) for t in tols])
    c.set_xlabel("onset tolerance (ms)"); c.set_ylabel("event F1")
    c.set_title("Below a 50 ms collar the decoder choice\nstarts to matter", fontsize=11)
    c.grid(alpha=.3); c.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(VIZ / "09_onset_resolution.png", dpi=135); plt.close(fig)
    print("  wrote 09_onset_resolution.png")


def fig_curve():
    d = np.load(FEAT / "subframe_curves.npz")
    tc, pc, tf, pf, iv = d["t_coarse"], d["p_coarse"], d["t_fine"], d["p_fine"], d["intervals_s"]
    # a stretch with several closely spaced calls
    gaps = iv[1:, 0] - iv[:-1, 1]
    k = int(np.argmin(np.abs(gaps - 0.06)))
    t0, t1 = iv[k, 0] - 0.35, iv[min(k + 3, len(iv) - 1), 1] + 0.35
    mc = (tc >= t0) & (tc <= t1); mf = (tf >= t0) & (tf <= t1)
    fig, ax = plt.subplots(figsize=(13, 4.0))
    for s, e in iv[(iv[:, 1] > t0) & (iv[:, 0] < t1)]:
        ax.axvspan(s, e, color=CALL, alpha=.16, lw=0)
    ax.plot(tf[mf], pf[mf], "-", color=ACC, lw=1.5, label="5 ms grid (4 shifted passes)")
    ax.plot(tc[mc], pc[mc], "o-", color="#444", lw=1.3, ms=4.5, label="20 ms grid (1 pass)")
    ax.axhline(0.4, color="#b03060", ls="--", lw=1.2, label="decision threshold")
    ax.set_xlim(t0, t1); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("time (s)"); ax.set_ylabel("P(call)")
    ax.set_title("Shaded = annotated calls. The 20 ms grid can only place an onset at a dot; "
                 "interpolating between dots recovers most of what the finer grid gives.",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(VIZ / "10_probability_curve.png", dpi=135); plt.close(fig)
    print("  wrote 10_probability_curve.png")


if __name__ == "__main__":
    VIZ.mkdir(parents=True, exist_ok=True)
    fig_resolution(); fig_onset(); fig_curve()
    print("done")
