#!/usr/bin/env python
"""Generate 08_calltype_errors.ipynb -- inside the call-type score: which call types each encoder
gets wrong, how confidently, how far each clip sits from its own call type's cluster, and what the
mistakes look like and sound like. Built to answer one question: does continued pretraining (DAPT)
from AVES help call type, and where?

Every number is computed at run time from the cached clip embeddings and calltype_errors.npz
(zfeval/experiments/calltype_errors.py, which asserts it reproduces the board's accuracies).

    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 notebooks/build_calltype_errors_notebook.py
    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m jupyter nbconvert --to notebook \
        --execute --inplace notebooks/08_calltype_errors.ipynb
"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def co(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Call-type errors: where each encoder goes wrong, and what DAPT changes

**The question.** Domain-adaptive pretraining (DAPT) starts from AVES-bio and keeps pretraining on
zebra finch audio. If specialisation helps, it should help most on the species-specific task, call
type. The board gives one accuracy per model. This notebook opens that number up:

1. **Confusion heatmaps**: which call types each model confuses with which.
2. **What DAPT changed**: per-type recall of DAPT minus its parent AVES-bio, with bird-bootstrap
   intervals.
3. **Confidence**: when the probe is wrong, how sure was it, and was the true type its second
   guess?
4. **Distance to the cluster**: where a misclassified clip sits relative to its own call type's
   centre and to the centre of the type it was called.
5. **Spectrograms** of the errors, next to typical members of both types, plus `show(i)` to look at
   and listen to any clip.
6. **Every DAPT checkpoint, per call type**: does any DAPT recipe help any call type consistently?

**Models (sections 1-5).** AVES-bio, the parent every DAPT run starts from. E23 DAPT
(`daptv4two_2e4s2_step15000`: AVES + 15,000 updates on zebra finch with two-teacher targets, seed 2),
the highest DAPT point on call type. run16, our best from-scratch recipe. Each is read at its own
reported best layer.

**Protocol.** The board's: 3,412 clips, 11 call types, 48 birds; leave-birds-out
StratifiedGroupKFold(5, seed 0); StandardScaler + logistic regression; mean-pooled clip embeddings.
Every prediction is out of fold. Birds are the independent unit, so every interval here is a
bird bootstrap (2,000 resamples, seed 0) and every difference quoted is the observed one.

**Call types** (Elie & Theunissen 2016): Ag = Wsst (aggressive), Be = begging, DC = distance call,
Di = distress, LT = long tonal, Ne = nest, So = song, Te = tet, Th = thuk, Tu = tuck, Wh = whine.
Be and LT are mostly chick calls. Di has 18 clips from 11 birds, so read its intervals before its
bars.
""")

md(r"""
> **Embeddings: CPU, not MPS (2026-10-16).** On this Mac, torch 2.8's MPS attention returns wrong outputs for
> inputs of 2-8 frames (about 45-185 ms), which is most call-type clips. Every number here is computed from CPU
> embeddings (`features/ct11_<tag>_cpuemb.npy`, `analysis/calltype_cpu_rescore.json`), which agree with float64 to
> about 1e-5. The board's original MPS numbers are kept unchanged beside them; the correction is finding 076.
""")

co(r"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import soundfile as sf
from scipy import signal
from IPython.display import Audio, display
warnings.filterwarnings("ignore")

ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
FEAT = Path.home() / "zf_labelset/zf_detection_dataset_v1/features"
Z = np.load(ANA / "calltype_errors.npz")
classes = [str(c) for c in Z["classes"]]; K = len(classes)
y, birds, paths = Z["y"], Z["birds"], Z["paths"]
TAGS = [str(t) for t in Z["tags"]]
NAME = {"aves-base-bio": "AVES-bio (parent)", "daptv4two_2e4s2_step15000": "E23 DAPT",
        "run16_compute4x": "run16 (scratch)"}
P = {t: Z[f"P__{t}"] for t in TAGS}
R = {t: Z[f"R_lda__{t}"] for t in TAGS}          # decoder-aligned geometry (see section 4)
Rraw = {t: Z[f"R__{t}"] for t in TAGS}
pred = {t: P[t].argmax(1) for t in TAGS}
AV, DA, R16 = TAGS
ORDER = ["DC", "Te", "Ne", "Th", "Tu", "Ag", "Wh", "Di", "So", "Be", "LT"]   # chick-dominant last
oi = [classes.index(c) for c in ORDER]

# palette: sequential blue for magnitude, blue<->red through grey for differences
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#a32a29", "#e34948", "#f3b3b2", "#f0efec", "#b7d3f6", "#3987e5", "#184f95"])
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})

# bird bootstrap, shared by every interval below: the same 2,000 resamples of the 48 birds
rng = np.random.default_rng(0)
ub = np.unique(birds); bidx = {b: np.where(birds == b)[0] for b in ub}
BOOT = [np.concatenate([bidx[b] for b in rng.choice(ub, len(ub))]) for _ in range(2000)]

def per_type_recall(hit, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    n = np.bincount(y[idx], minlength=K); h = np.bincount(y[idx], weights=hit[idx], minlength=K)
    with np.errstate(invalid="ignore"):
        return h / n

def recall_diff(hit_a, hit_b):
    obs = per_type_recall(hit_a) - per_type_recall(hit_b)
    bs = np.array([per_type_recall(hit_a, s) - per_type_recall(hit_b, s) for s in BOOT])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    oa = hit_a.mean() - hit_b.mean()
    ba = np.array([hit_a[s].mean() - hit_b[s].mean() for s in BOOT])
    return obs, lo, hi, (oa, *np.percentile(ba, [2.5, 97.5]))

for t in TAGS:
    print(f"{NAME[t]:20s} layer {int(Z[f'layer__{t}'])}  accuracy {(pred[t] == y).mean():.4f}  "
          f"errors {(pred[t] != y).sum()} of {len(y)}")
print("\nclips per type:", {c: int((y == classes.index(c)).sum()) for c in ORDER})
print("birds per type:", {c: len(set(birds[y == classes.index(c)])) for c in ORDER})
""")

md(r"""
## 1. Confusion heatmaps

Rows are the true call type, columns what the probe called it; each row sums to 100%. The diagonal
is per-type recall. Off-diagonal cells of 3% or more are labelled.
""")

co(r"""
def confusion(t):
    M = np.zeros((K, K)); np.add.at(M, (y, pred[t]), 1)
    return M[np.ix_(oi, oi)]

fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
for ax, t in zip(axes, TAGS):
    M = confusion(t); Mn = M / M.sum(1, keepdims=True)
    im = ax.imshow(Mn, cmap=SEQ, vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            if i == j or Mn[i, j] >= 0.03:
                ax.text(j, i, f"{100 * Mn[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if Mn[i, j] > 0.55 else INK)
    ax.set_xticks(range(K)); ax.set_xticklabels(ORDER); ax.set_yticks(range(K)); ax.set_yticklabels(ORDER)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{NAME[t]}  (acc {(pred[t] == y).mean():.3f})", loc="left", color=INK)
    for s in ax.spines.values(): s.set_visible(False)
fig.colorbar(im, ax=axes, shrink=0.8, label="share of the true type's clips")
plt.show()
""")

md(r"""
## 2. What DAPT changed, call type by call type

Left: the confusion matrix of E23 DAPT minus AVES-bio's, in percentage points of each true type.
Blue on the diagonal = DAPT recalls that type more often; red off the diagonal = DAPT makes that
particular mistake more often. Right: per-type recall difference with its 95% bird-bootstrap
interval. A bar whose interval crosses zero is not distinguishable from no change.
""")

co(r"""
hit = {t: (pred[t] == y).astype(float) for t in TAGS}
fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.05, 1]})
Ma, Md = confusion(AV), confusion(DA)
Dm = 100 * (Md / Md.sum(1, keepdims=True) - Ma / Ma.sum(1, keepdims=True))
lim = np.abs(Dm).max()
im = a0.imshow(Dm, cmap=DIV, vmin=-lim, vmax=lim)
for i in range(K):
    for j in range(K):
        if abs(Dm[i, j]) >= 2:
            a0.text(j, i, f"{Dm[i, j]:+.0f}", ha="center", va="center", fontsize=7.5, color=INK)
a0.set_xticks(range(K)); a0.set_xticklabels(ORDER); a0.set_yticks(range(K)); a0.set_yticklabels(ORDER)
a0.set_xlabel("predicted"); a0.set_ylabel("true")
a0.set_title("E23 DAPT minus AVES-bio (percentage points of each true type)", loc="left", color=INK)
for s in a0.spines.values(): s.set_visible(False)
fig.colorbar(im, ax=a0, shrink=0.8)

rows = []
for other, lab in ((AV, "DAPT - AVES-bio"), (R16, "DAPT - run16")):
    obs, lo, hi, (oa, la, ha) = recall_diff(hit[DA], hit[other])
    for c in ORDER:
        i = classes.index(c)
        rows.append(dict(contrast=lab, type=c, diff=obs[i], lo=lo[i], hi=hi[i],
                         resolved=bool(lo[i] > 0 or hi[i] < 0)))
    rows.append(dict(contrast=lab, type="ALL", diff=oa, lo=la, hi=ha, resolved=bool(la > 0 or ha < 0)))
RD = pd.DataFrame(rows)

d = RD[RD.contrast == "DAPT - AVES-bio"].reset_index(drop=True)
yy = np.arange(len(d))[::-1]
for k, r in d.iterrows():
    col = "#184f95" if (r.resolved and r["diff"] > 0) else "#a32a29" if r.resolved else MUTED
    a1.plot([r.lo, r.hi], [yy[k]] * 2, color=col, lw=2, solid_capstyle="round")
    a1.plot(r["diff"], yy[k], "o", color=col, ms=6 if r.type != "ALL" else 8)
a1.axvline(0, color=MUTED, lw=1)
a1.set_yticks(yy); a1.set_yticklabels(d.type)
a1.set_xlabel("recall difference, DAPT minus AVES-bio (95% bird bootstrap)")
a1.set_title("Per-type recall: DAPT vs its parent", loc="left", color=INK)
a1.xaxis.grid(True, color=GRID); a1.set_axisbelow(True)
plt.tight_layout(); plt.show()
display(RD.pivot(index="type", columns="contrast", values="diff").loc[ORDER + ["ALL"]]
        .style.format("{:+.3f}"))
display(RD[RD.resolved][["contrast", "type", "diff", "lo", "hi"]]
        .style.format({"diff": "{:+.3f}", "lo": "{:+.3f}", "hi": "{:+.3f}"}).hide(axis="index"))
""")

md(r"""
## 3. The mistakes, ranked

The most frequent confusions per model, with how many different birds the errors come from. An
error pattern carried by one or two birds is about those birds' voices, not about the call types.
""")

co(r"""
def top_confusions(t, n=10):
    wrong = pred[t] != y
    df = pd.DataFrame(dict(true=np.array(classes)[y[wrong]], called=np.array(classes)[pred[t][wrong]],
                           bird=birds[wrong]))
    g = df.groupby(["true", "called"]).agg(n=("bird", "size"), birds=("bird", "nunique")).reset_index()
    tot = {c: int((y == classes.index(c)).sum()) for c in classes}
    g["share_of_true"] = g.apply(lambda r: r.n / tot[r.true], axis=1)
    return g.sort_values("n", ascending=False).head(n).reset_index(drop=True)

for t in TAGS:
    print(NAME[t])
    display(top_confusions(t).style.format({"share_of_true": "{:.1%}"}).hide(axis="index"))
""")

md(r"""
## 4. Confidence: how sure was the probe when it was wrong?

Left: the probability the probe gave its own answer, for correct clips and for errors. A probe that
knows when it is unsure puts its errors at low confidence. Right: for the errors, where the true
type ranked among the 11. Rank 2 means the right answer was its second guess, a near miss.
""")

co(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
bins = np.linspace(0, 1, 21)
cols = {AV: "#2a78d6", DA: "#eda100", R16: "#eb6834"}
for t in TAGS:
    conf = P[t].max(1); wrong = pred[t] != y
    axes[0].hist(conf[wrong], bins=bins, histtype="step", lw=2, color=cols[t], label=f"{NAME[t]}, errors")
    axes[0].hist(conf[~wrong], bins=bins, histtype="step", lw=1, ls="--", color=cols[t],
                 label=f"{NAME[t]}, correct", density=False, alpha=0.6)
axes[0].set_yscale("log"); axes[0].set_xlabel("probability of the predicted type"); axes[0].set_ylabel("clips (log)")
axes[0].legend(fontsize=7.5, frameon=False); axes[0].set_title("Confidence, correct vs wrong", loc="left")

ranks = {}
for t in TAGS:
    wrong = np.where(pred[t] != y)[0]
    order = np.argsort(-P[t][wrong], axis=1)
    ranks[t] = np.array([int(np.where(o == y[i])[0][0]) + 1 for o, i in zip(order, wrong)])
w = 0.27
for k, t in enumerate(TAGS):
    cnt = np.bincount(ranks[t], minlength=12)[2:12]
    axes[1].bar(np.arange(2, 12) + (k - 1) * w, cnt / len(ranks[t]), width=w, color=cols[t], label=NAME[t])
axes[1].set_xticks(range(2, 12)); axes[1].set_xlabel("rank of the true type among the errors")
axes[1].set_ylabel("share of errors"); axes[1].legend(fontsize=8, frameon=False)
axes[1].set_title("Was the right answer the second guess?", loc="left")
plt.tight_layout(); plt.show()

rows = []
for t in TAGS:
    wrong = pred[t] != y; conf = P[t].max(1); ptrue = P[t][np.arange(len(y)), y]
    rows.append(dict(model=NAME[t], errors=int(wrong.sum()),
                     median_conf_when_wrong=np.median(conf[wrong]),
                     median_conf_when_right=np.median(conf[~wrong]),
                     wrong_with_conf_over_90=float((conf[wrong] > 0.9).mean()),
                     true_type_2nd_guess=float((ranks[t] == 2).mean()),
                     median_p_true_when_wrong=np.median(ptrue[wrong])))
display(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in
        ["median_conf_when_wrong", "median_conf_when_right", "median_p_true_when_wrong"]} |
        {"wrong_with_conf_over_90": "{:.1%}", "true_type_2nd_guess": "{:.1%}"}).hide(axis="index"))

# per type, for DAPT: how confident are the errors of each type
wrong = pred[DA] != y; conf = P[DA].max(1)
pt = pd.DataFrame(dict(type=np.array(classes)[y], wrong=wrong, conf=conf))
display(pt[pt.wrong].groupby("type").conf.agg(["size", "median"]).loc[[c for c in ORDER if c in set(pt[pt.wrong].type)]]
        .rename(columns={"size": "errors", "median": "median confidence"}).T.style.format("{:.2f}"))
""")

md(r"""
## 5. Distance to the call-type cluster

For each clip, the distance to the centre of every call type, computed out of fold (centres and
scales from the training birds only). Distances are in units of each type's own spread: **1.0 = as
far from the centre as a typical member of that type**.

Two geometries are measured:

- **Raw**: the standardised embedding itself. This is the geometry a label-free clustering would
  see.
- **Decoder-aligned**: a 10-dimensional shrinkage-LDA space fitted on the training fold, the
  directions that separate call types. This is closer to what the probe uses.

The first table is the sanity check: nearest-centroid accuracy in each geometry, against the
probe. The gap between raw and LDA is how much of the call-type structure is *not* visible as
clusters in the raw embedding.
""")

co(r"""
rows = []
for t in TAGS:
    rows.append(dict(model=NAME[t], probe=(pred[t] == y).mean(),
                     nearest_centroid_raw=(Rraw[t].argmin(1) == y).mean(),
                     nearest_centroid_lda=(R[t].argmin(1) == y).mean()))
display(pd.DataFrame(rows).style.format({c: "{:.4f}" for c in
        ["probe", "nearest_centroid_raw", "nearest_centroid_lda"]}).hide(axis="index"))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
for ax, t in zip(axes, TAGS):
    i = np.arange(len(y)); wrong = pred[t] != y
    rt = R[t][i, y]; rp = R[t][i, pred[t]]
    # for correct clips the predicted type IS the true type; use their nearest OTHER type instead
    Ro = R[t].copy(); Ro[i, y] = np.inf; ro = Ro.min(1)
    ax.scatter(rt[~wrong], ro[~wrong], s=5, color="#b9b8b3", alpha=0.5, lw=0, label="correct (y = nearest other type)")
    ax.scatter(rt[wrong], rp[wrong], s=9, color="#e34948", alpha=0.75, lw=0, label="error (y = the type it was called)")
    m = max(np.percentile(rt, 99), np.percentile(ro, 99))
    ax.plot([0, m], [0, m], color=MUTED, lw=1, ls=":")
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    ax.set_xlabel("distance to its TRUE type's centre (type spreads)")
    ax.set_title(NAME[t], loc="left", color=INK)
axes[0].set_ylabel("distance to the other type's centre")
axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
fig.suptitle("Decoder-aligned geometry: below the dotted line a clip sits nearer the wrong centre than its own",
             x=0.01, ha="left", color=INK)
plt.tight_layout(); plt.show()

rows = []
for t in TAGS:
    i = np.arange(len(y)); wrong = pred[t] != y
    rt = R[t][i, y]; rp = R[t][i, pred[t]]
    rows.append(dict(model=NAME[t],
                     median_dist_to_own_centre_correct=np.median(rt[~wrong]),
                     median_dist_to_own_centre_errors=np.median(rt[wrong]),
                     errors_inside_called_type=float((rp[wrong] < 1).mean()),
                     errors_still_inside_own_type=float((rt[wrong] < 1).mean())))
display(pd.DataFrame(rows).style.format({"median_dist_to_own_centre_correct": "{:.2f}",
        "median_dist_to_own_centre_errors": "{:.2f}", "errors_inside_called_type": "{:.1%}",
        "errors_still_inside_own_type": "{:.1%}"}).hide(axis="index"))
print("'inside' = within the typical spread (distance < 1) of that type's centre, decoder-aligned geometry")
""")

md(r"""
## 6. What the mistakes look and sound like

For E23 DAPT's four most frequent confusions, each row shows:

- two **typical** clips of the true type (correct, nearest its centre);
- the four most **confident** errors (true type → called type);
- two typical clips of the type it was called.

Spectrograms run 0-10 kHz, each clip on its own time axis. Titles give the bird, the probe's
probability for the type it chose, and the distance to the true type's centre.

To look at and listen to any clip, call `show(i)` with its row number (printed in the titles).
""")

co(r"""
def spec(path, fmax=10000):
    w, sr = sf.read(path, always_2d=True); w = w.mean(1)
    f, tt, S = signal.spectrogram(w, sr, nperseg=512, noverlap=448, window="hann")
    k = f <= fmax
    return tt, f[k], 10 * np.log10(S[k] + 1e-12), len(w) / sr

def draw(ax, i, t=DA, title=None):
    tt, f, S, dur = spec(paths[i])
    vmax = np.percentile(S, 99.5)
    ax.pcolormesh(tt, f / 1000, S, cmap="magma", vmin=vmax - 55, vmax=vmax, shading="auto")
    ax.set_xticks([]); ax.set_yticks([0, 5, 10] if ax.get_subplotspec().is_first_col() else [])
    ax.set_title(title or f"#{i} {birds[i]}", fontsize=7, loc="left")
    ax.text(0.98, 0.04, f"{1000 * dur:.0f} ms", transform=ax.transAxes, ha="right", fontsize=6.5, color="white")

def typical(c, t=DA, n=2):
    i = np.where((y == c) & (pred[t] == y))[0]
    return i[np.argsort(R[t][i, c])[:n]]

g = top_confusions(DA, 4)
fig, axes = plt.subplots(len(g), 8, figsize=(17, 2.0 * len(g) + 0.6))
for r, row in g.iterrows():
    ct, cp = classes.index(row.true), classes.index(row.called)
    errs = np.where((y == ct) & (pred[DA] == cp))[0]
    errs = errs[np.argsort(-P[DA][errs, cp])[:4]]
    cells = [(i, f"typical {row.true}") for i in typical(ct)] + \
            [(i, f"{row.true}->{row.called}") for i in errs] + \
            [(i, f"typical {row.called}") for i in typical(cp)]
    for k, ax in enumerate(axes[r]):
        if k >= len(cells):
            ax.axis("off"); continue
        i, lab = cells[k]
        draw(ax, i, title=f"{lab}  #{i} {birds[i]}\np={P[DA][i].max():.2f}  d_true={R[DA][i, y[i]]:.1f}")
    axes[r][0].set_ylabel("kHz", fontsize=7)
fig.suptitle("E23 DAPT's most frequent confusions: typical true type | confident errors | typical called type",
             x=0.01, ha="left", color=INK)
plt.tight_layout(); plt.show()
""")

co(r"""
def show(i):
    # spectrogram, audio, and what every model made of clip i
    fig, ax = plt.subplots(figsize=(6, 2.6))
    draw(ax, i, title=f"#{i}  {Path(paths[i]).name}  true type {classes[y[i]]}")
    ax.set_ylabel("kHz"); plt.show()
    w, sr = sf.read(paths[i], always_2d=True)
    display(Audio(w.mean(1), rate=sr))
    rows = []
    for t in TAGS:
        top = np.argsort(-P[t][i])[:3]
        rows.append(dict(model=NAME[t], called=classes[top[0]],
                         top3=", ".join(f"{classes[c]} {P[t][i, c]:.2f}" for c in top),
                         p_true=P[t][i, y[i]], d_true=R[t][i, y[i]], d_called=R[t][i, top[0]]))
    display(pd.DataFrame(rows).style.format({"p_true": "{:.3f}", "d_true": "{:.2f}",
                                             "d_called": "{:.2f}"}).hide(axis="index"))

# example: DAPT's single most confident error
wrong = np.where(pred[DA] != y)[0]
show(int(wrong[np.argmax(P[DA][wrong].max(1))]))
""")

md(r"""
## 7. Errors every model makes, and errors DAPT fixed or introduced

Clips all three encoders get wrong, with high confidence, are candidates for ambiguous calls or
label errors rather than model failures. Clips that AVES-bio gets wrong and DAPT gets right (and
the reverse) are where DAPT's change actually lives.
""")

co(r"""
W = {t: pred[t] != y for t in TAGS}
allw = W[AV] & W[DA] & W[R16]
fixed = W[AV] & ~W[DA]; broke = ~W[AV] & W[DA]
print(f"all three wrong: {allw.sum()} clips ({len(set(birds[allw]))} birds); "
      f"of those, all three agree on the same wrong type: "
      f"{(allw & (pred[AV] == pred[DA]) & (pred[DA] == pred[R16])).sum()}")
print(f"DAPT fixed (AVES wrong, DAPT right): {fixed.sum()}   DAPT broke (AVES right, DAPT wrong): {broke.sum()}"
      f"   net {fixed.sum() - broke.sum():+d}")
tab = pd.DataFrame({"fixed": pd.Series(np.array(classes)[y[fixed]]).value_counts(),
                    "broke": pd.Series(np.array(classes)[y[broke]]).value_counts()}).fillna(0).astype(int)
tab["net"] = tab.fixed - tab.broke
display(tab.reindex(ORDER).fillna(0).astype(int).T)

agree = np.where(allw & (pred[AV] == pred[DA]) & (pred[DA] == pred[R16]))[0]
agree = agree[np.argsort(-np.minimum.reduce([P[t][agree].max(1) for t in TAGS]))][:8]
if len(agree):
    fig, axes = plt.subplots(1, len(agree), figsize=(2.1 * len(agree), 2.3))
    for ax, i in zip(np.atleast_1d(axes), agree):
        draw(ax, i, title=f"#{i} {birds[i]}\nlabel {classes[y[i]]}, all say {classes[pred[DA][i]]}")
    fig.suptitle("All three encoders confidently agree on a different type: worth a listen as possible label errors",
                 x=0.01, ha="left", fontsize=10, color=INK)
    plt.tight_layout(); plt.show()
""")

md(r"""
## 8. Every DAPT checkpoint, call type by call type

Sections 1-7 use one DAPT run. The real question is whether *any* way of continuing AVES on zebra
finch moves call type, and whether different DAPT runs move the *same* call types. A type that
improves in one seed and not in its replicate is noise.

Each row is a DAPT checkpoint, read at its own reported best layer exactly as the board does. Each
cell is its recall minus its parent's on that call type (AVES-bio; BirdAVES biox-base for E14); a
dot marks a 95% bird-bootstrap interval that excludes zero. The rightmost column is overall
accuracy.

The rows are not independent replications. They share a parent and most share a recipe, and 11
types times many rows is a lot of tests. Read a column as a pattern only if it holds across
different recipes *and* across the seed pairs (E19/E23, E21/E24, E18/E20).
""")

co(r"""
DB = json.loads((ANA / "dapt_calltype_by_type.json").read_text())
LABEL = {
    "dapt5e5_step15000": "R1 ZF only, lr 5e-5", "dapt1e4_step15000": "R1 ZF only, lr 1e-4",
    "daptshort_5e5_step2500": "R2 ZF only, 2,500 updates", "daptlr_1e5_step15000": "R2 ZF only, lr 1e-5",
    "daptfreeze_5e5_step15000": "R2 ZF only, CNN frozen",
    "daptreplay_5e5_step15000": "R2 replay x1 (19% FSD50K)", "daptreplay3_5e5_step15000": "E5 replay x3 (42% FSD50K)",
    "daptv3aves_5e5_step15000": "E17 AVES targets, lr 5e-5", "daptv3aves_2e4_step15000": "E18 AVES targets, lr 2e-4",
    "daptv3aves_2e4s2_step15000": "E20 = E18, seed 2", "daptv3aves_2e4m8_step15000": "E28a E18 + AVES masking",
    "daptv3aves_2e4s2m8_step15000": "E28b E20 + AVES masking",
    "daptv4two_2e4_step15000": "E19 two-teacher targets", "daptv4two_2e4s2_step15000": "E23 = E19, seed 2",
    "daptv4aveso_2e4_step15000": "E21 AVES-only control", "daptv4aveso_2e4s2_step15000": "E24 = E21, seed 2",
    "daptv4two45k_2e4_step45000": "E22 E19 x3 longer (45k)", "daptv4bird_2e4_step15000": "E14 E19 from BirdAVES",
    "aves-base-core": "download: AVES core", "aves-base-all": "download: AVES all", "aves-base-nonbio": "download: AVES nonbio",
    "birdaves-biox-base": "download: BirdAVES biox-base", "birdaves-biox-large": "download: BirdAVES biox-large",
    "birdaves-bioxn-large": "download: BirdAVES bioxn-large",
}

def heat(block, keys, title, ax):
    H = pd.DataFrame([[block[t]["vs_parent"][c]["diff"] for c in ORDER + ["ALL"]] for t in keys], index=keys, columns=ORDER + ["ALL"])
    S = pd.DataFrame([[block[t]["vs_parent"][c]["resolved"] for c in ORDER + ["ALL"]] for t in keys], index=keys, columns=ORDER + ["ALL"])
    lim = 100 * np.nanmax(np.abs(H.drop(columns=["Di", "ALL"]).values))
    im = ax.imshow(np.clip(100 * H.values, -lim, lim), cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
    for r in range(H.shape[0]):
        for c in range(H.shape[1]):
            v = 100 * H.values[r, c]
            ax.text(c, r, f"{v:+.0f}" + ("\u2022" if S.values[r, c] else ""), ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.8 * lim else INK)
    ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns)
    ax.set_yticks(range(H.shape[0]))
    ax.set_yticklabels([f"{LABEL.get(t, t)}  (L{block[t]['layer']}, {block[t]['acc11']:.3f})"
                        + ("  vs BirdAVES" if block[t]["parent"] != "aves-base-bio" else "") for t in keys], fontsize=7.5)
    ax.axvline(len(ORDER) - 0.5, color="white", lw=3)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    for sp in ax.spines.values(): sp.set_visible(False)
    return im

dk = list(DB["dapt"]["best"]); dl = list(DB["downloads"]["best"])
fig, axes = plt.subplots(2, 1, figsize=(11.5, 0.36 * (len(dk) + len(dl)) + 3), gridspec_kw={"height_ratios": [len(dk), len(dl)]})
im = heat(DB["dapt"]["best"], dk, "DAPT checkpoints minus their parent, per call type, each at its own best layer  (\u2022 = 95% interval excludes 0)", axes[0])
heat(DB["downloads"]["best"], dl, "Control: the other downloaded AVES models minus AVES-bio, same reading", axes[1])
fig.colorbar(im, ax=axes, shrink=0.6, label="recall difference (percentage points; Di clipped)")
plt.show()

P8 = DB["pattern"]
tab = pd.DataFrame({k: {c: f"{P8[k][c]['above_point']} up / {P8[k][c]['below_point']} down  (med {P8[k][c]['median']:+.3f})"
                        for c in ORDER + ["ALL"]} for k in ("dapt_best", "dapt_L3", "downloads_best", "downloads_L3")}).T
tab.index = ["DAPT, own best layers (17, AVES-bio parent)", "DAPT, both at layer 3", "downloads, own best layers",
             "downloads, both at layer 3"]
display(tab[["Te", "Ne", "Th", "Tu", "Ag", "Wh", "Be", "LT", "ALL"]])
print("A per-type DAPT effect is only DAPT-specific if the DAPT rows differ from the downloads' rows: any other encoder")
print("compared with AVES-bio's particular layer moves the same types the same way.")
""")

md(r"""
## 9. Does specialising longer help call type? The round-1 trajectories

Round 1 continued AVES-bio on zebra finch audio only, saving a checkpoint every 2,500 updates at two
learning rates. These checkpoints were scored for detection (notebook 03) but never for call type
until now. If specialisation helps call type, accuracy should rise with steps on zebra finch.
Each point is the checkpoint minus AVES-bio (observed difference, 95% bird bootstrap, from
`<tag>_bootstrap.json`), at its own best layer (left) and at the fixed layer 3 (right; no layer
selection).
""")

co(r"""
R1 = DB["round1"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
col = {"5e5": "#2a78d6", "1e4": "#eb6834"}
rows = []
for k, (key, title) in enumerate((("best", "at each checkpoint's best layer"), ("L3", "both at layer 3 (no layer selection)"))):
    ax = axes[k]
    for lr in ("5e5", "1e4"):
        steps = [2500, 5000, 7500, 10000, 12500, 15000]
        d = [R1[key][f"dapt{lr}_step{s_}"] for s_ in steps]
        st = np.array(steps) + (-150 if lr == "5e5" else 150)
        ax.errorbar(st, [x["diff"] for x in d], yerr=[[x["diff"] - x["lo"] for x in d], [x["hi"] - x["diff"] for x in d]],
                    fmt="o-", color=col[lr], lw=1.5, ms=5, capsize=0, label=f"lr {lr.replace('e', 'e-')}")
        for s_, x in zip(steps, d):
            rows.append(dict(reading=key, lr=lr, step=s_, diff=x["diff"], lo=x["lo"], hi=x["hi"], resolved=x["resolved"]))
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("updates of continued pretraining on zebra finch"); ax.set_title(title, loc="left")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
axes[0].set_ylabel("call-type accuracy minus AVES-bio"); axes[0].legend(frameon=False)
fig.suptitle("Round-1 DAPT (zebra finch only): call type along the trajectory", x=0.01, ha="left", color=INK)
plt.tight_layout(); plt.show()
display(pd.DataFrame(rows).style.format({"diff": "{:+.4f}", "lo": "{:+.4f}", "hi": "{:+.4f}"}).hide(axis="index"))
""")

md(r"""
**Reading the trajectories.** Continuing AVES-bio on zebra finch audio alone never raised call type,
at any of the 12 checkpoints or either learning rate. The point estimates sit at or below the parent
throughout, more clearly at the fixed layer 3 than at each checkpoint's own best layer. No single
checkpoint is resolved from the parent, and there is no rising trend to extend. More specialisation
on the same audio is not the road to a call-type gain.
""")

md(r"""
## 10. Where the headroom is

One cell that collects the numbers the conclusions rest on: how the errors are spread over call
types, and how close the best frozen probes are to what supervised fine-tuning reaches on the same
folds (`finetune_calltype.json`: the whole encoder trained on the call-type labels, 2 seeds).
""")

co(r"""
soft = ["Te", "Ne", "Th", "Tu"]
si = [classes.index(c) for c in soft]
for t in TAGS:
    w = pred[t] != y
    print(f"{NAME[t]:20s} {w.sum()} errors; {np.isin(y[w], si).mean():.0%} of them are true {'/'.join(soft)} "
          f"(these types are {np.isin(y, si).mean():.0%} of the clips); recall on Tu {per_type_recall(hit[t])[classes.index('Tu')]:.2f}")
FT = json.loads((ANA / "finetune_calltype.json").read_text())["summary"]["arms"]
ft = FT["full_ft"]["aves"]
print(f"\nAVES-bio fully fine-tuned on call type: {ft['mean']:.4f} (seeds: "
      + ", ".join(f"{v:.4f}" for v in ft["per_seed_mean"].values())
      + f"); same-script frozen layer 3: {FT['frozen_L3']['aves']['mean']:.4f}")
print("note: finetune_calltype.json averages accuracy over folds, the board pools clips; the two differ slightly")
print(f"best frozen points on the board: E23 DAPT {(pred[DA] == y).mean():.4f}, AVES-bio {(pred[AV] == y).mean():.4f}")
""")

md(r"""
## 11. Are the soft-call errors about the calls, or about new birds?

Every number above is leave-birds-out: the probe never heard the test bird. Here the same
embeddings are probed with a random clip split (StratifiedKFold(5), seed 0), so the test birds'
other calls are in training. A call type whose recall jumps under the random split is
confused because birds say it differently, not because the type is acoustically ambiguous.
(A random split lets the probe learn each test bird's voice and inflates call type; the 2023
AVES notebook used one, finding 034. It is used here only as a diagnostic, never as a score.)
""")

co(r"""
BG = DB["bird_generalisation"]
rows = {}
for t in (AV, DA):
    rows[(NAME[t], "random split")] = [BG[t]["random_split"][c] for c in ORDER] + [BG[t]["acc_random_split"]]
    rows[(NAME[t], "leave birds out")] = [BG[t]["leave_birds_out"][c] for c in ORDER] + [BG[t]["acc_leave_birds_out"]]
RS = pd.DataFrame(rows, index=ORDER + ["ALL"]).T
display(RS.style.format("{:.2f}").background_gradient(cmap=SEQ, vmin=0.3, vmax=1, axis=None))
for t in (AV, DA):
    b = BG[t]
    print(f"{NAME[t]:18s} errors the leaky split removes: soft calls (Te/Ne/Th/Tu) {b['share_soft_errors_removed']:.0%} "
          f"({b['soft_errors_lbo']} -> {b['soft_errors_random']}), other types {b['share_other_errors_removed']:.0%} "
          f"({b['other_errors_lbo']} -> {b['other_errors_random']})")
g = BG["E23_extra_gain_on_random_split"]
print(f"\nE23 minus AVES-bio: random split {g['gain_random']:+.4f}, leave birds out {g['gain_lbo']:+.4f}; "
      f"difference {g['diff']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}] ({'resolved' if g['resolved'] else 'not distinguishable'})")
print("Every bird in this test set is also in DAPT's pretraining audio (finding 028). The random split also lets clips")
print("from the same recording session into training, so it is more than 'the probe heard this bird's voice'.")
""")

md(r"""
## What this does and does not license

Every number behind these statements is printed in the cell it cites.

**Supported.**

- **The errors live in four call types.** Tuck, thuk, nest and tet (Tu, Th, Ne, Te) carry most of
  every encoder's mistakes, against a much smaller share of the clips (section 10). The largest
  single confusion is tuck called thuk, and tuck is recalled less than half the time by all three
  encoders (sections 1 and 3).
- **DAPT moves the boundary between these types; it does not sharpen it.** E23 DAPT's overall gain
  over AVES-bio is not resolved (section 2). Across every DAPT checkpoint the per-type pattern
  repeats: Ag and Th recall go up, Te and Tu go down, and overall accuracy does not move (section 8).
  That is a trade between neighbouring types, not new separability.
- **The soft-call errors are mostly about new birds.** When the test birds' other calls are in
  training, tuck recall nearly doubles (section 11). DAPT's advantage over its parent is larger on
  that leaky split than across birds, and its tuck recall across birds goes down. Every bird in
  this test set is in DAPT's pretraining audio, so the most economical reading is that continued
  pretraining on these 48 birds teaches their voices more than it teaches call categories.
- **The mistakes are confident, and usually near misses.** About half of the errors have
  confidence above 0.9, and the true type is most often the probe's second guess (section 4). This
  is a representation limit; no threshold on the probe recovers it.
- **Call types are separable, but not clusters.** Nearest-centroid classification in the raw
  embedding is far below the linear probe, and most of the gap closes in the LDA space (section 5).
  For label-free unit discovery this matters: clustering the raw embedding will not return call
  types as its clusters, even where a linear readout finds them.
- **Specialising longer on zebra finch alone does not help either.** The 12 round-1 checkpoints
  (AVES-bio continued on zebra finch only, two learning rates, every 2,500 updates) sit at or below
  the parent on call type throughout, none resolved (section 9).
- **The headroom on this task is small.** Fine-tuning all of AVES on the call-type labels reaches
  about what the best frozen probes reach (section 10). A DAPT gain would have to come from tuck,
  thuk and nest across birds, because that is where the remaining errors are.

**Not supported.**

- Per-type differences from one checkpoint (sections 2-7) are single runs. The evidence for the
  DAPT pattern is section 8's consistency, and those checkpoints share a parent and mostly a recipe,
  so they are not independent replications.
- "All three wrong and agreeing" clips (section 7) are candidates for label errors, not
  established ones. They need listening: use the error browser's *All three wrong* view.
- Section 11's random split is a diagnostic of where errors come from, never a score.

**What it suggests for DAPT.** Specialising on the same birds the test uses rewards learning their
voices. The DAPT change that follows from these numbers is one that removes voice, not one that adds
more of the same audio: E33 in `harness/experiments.yaml` (pitch-shifted input with fixed targets,
the bird version of ContentVec), with a frozen-encoder pre-test (`e33_pitch_pretest.py`) that checks
first whether pitch is a voice axis that call type does not need.
""")

nb["cells"] = C
out = Path(__file__).resolve().parent / "08_calltype_errors.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
