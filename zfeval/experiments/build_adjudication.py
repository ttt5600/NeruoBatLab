#!/usr/bin/env python
"""Build a BLIND adjudication page for the 52 high-confidence false positives.

The open question: the 52 eval-B windows the probe scored p>0.944 but the hand labels call
"noise" are acoustically indistinguishable from true positives (flatness p=0.41, pitch p=0.25,
peak dB p=0.13). Either they are genuine model errors, or they are real calls nobody annotated.
No acoustic measure settles it; listening does.

But "listen to 52 clips the model got wrong" is not a measurement -- there is nothing to compare
the judgements to, and knowing they are the model's errors biases the listener toward hearing
errors. So the page also carries PLANTED CONTROLS, matched on loudness and shuffled in:

  52  high-confidence FPs        p>0.944, hand label "noise"
  18  true positives            hand label "call", probe agreed -- known real calls
  18  true negatives            hand label "noise", probe agreed -- known non-calls

All 88 look identical in the page, in shuffled order, with no label, no probability, no tier.
The answer key is written to a LOCAL file and never appears in the HTML, so the blinding is real
and not decorative. The read-out is a rate comparison:

  FPs judged "call" at ~ the TP control rate  -> they are unannotated calls, and detection
                                                 precision was underestimated
  FPs judged "call" at ~ the TN control rate  -> they are genuine model errors

Controls are matched on window loudness because the FPs are loud (-37.96 dB median vs -41.95 for
true negatives): unmatched controls would be quieter as a group and the listener could separate
the arms on volume alone, which would measure nothing.
"""
from __future__ import annotations
import base64, csv, io, json
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TMP = Path.home() / ".claude/jobs/63c218d9/tmp"
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
AUDIO = Path.home() / "zf_labelset/audio/111021-000.wav"
SR, WIN = 16000, 16000
CTX = 2.0                      # seconds of context per clip, window centred
HI = 0.944
N_TP, N_TN = 18, 18
SEED = 11021


def wav_b64(x: np.ndarray) -> str:
    """PCM16 WAV as a data URI. Peak-normalised per clip: these are being judged by ear, and a
    quiet call must be audible. Loudness judgements are NOT made from the player -- the page shows
    the measured dB instead."""
    x = np.asarray(x, dtype=np.float32)
    pk = float(np.abs(x).max())
    if pk > 0:
        x = 0.95 * x / pk
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def spec_b64(ctx: np.ndarray, win_lo: float, win_hi: float) -> str:
    """JPEG, not PNG. A spectrogram is noise-like, so PNG's lossless compression barely helps:
    88 of them came to 23 MB and the page cap is 16 MB. JPEG at q=72 costs nothing a listener
    would notice on a 560 px image and cuts that by an order of magnitude."""
    from scipy.signal import spectrogram
    f, t, S = spectrogram(ctx, SR, nperseg=512, noverlap=512 - 128, mode="psd")
    S = 10 * np.log10(S + 1e-12)
    fig, ax = plt.subplots(figsize=(5.9, 2.2), dpi=132)
    fig.patch.set_facecolor("#0c0c10")
    ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
              extent=[0, len(ctx) / SR, f[0] / 1000, f[-1] / 1000],
              vmin=np.percentile(S, 30), vmax=np.percentile(S, 99.7))
    ax.set_ylim(0, 8)
    for xv in (win_lo, win_hi):
        ax.axvline(xv, color="#5ff", lw=1.3, alpha=.85)
    ax.set_yticks([0, 4, 8]); ax.set_yticklabels(["0", "4", "8k"], fontsize=8, color="#8a8a95")
    ax.set_xticks([]); ax.tick_params(length=2, colors="#8a8a95")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=0.12)
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", facecolor="#0c0c10", pil_kwargs=dict(quality=72, optimize=True))
    plt.close(fig)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    d = np.load(TMP / "evalB_err.npz", allow_pickle=True)
    starts, outcome, pb, yb, bids, en = (d["starts"], d["outcome"].astype(str), d["pb"],
                                         d["yb"], d["bids"].astype(str), d["energy"])
    rec, sr = sf.read(AUDIO, dtype="float32")
    assert sr == SR

    fp = np.where((outcome == "FP") & (pb > HI))[0]
    print(f"high-confidence FPs: {len(fp)}  (median {np.median(en[fp]):.1f} dB)")

    rng = np.random.default_rng(SEED)
    # Loudness-matched controls: for each drawn control, take the unused candidate closest in dB to
    # a randomly chosen FP. Nearest-UNUSED, not nearest-overall -- a scan that can reuse a
    # candidate biases the matched set, which is exactly the bug that once gave a null control a
    # spurious 0.926 win rate on this dataset.
    def matched(pool, n):
        pool = list(pool); chosen = []
        for tgt in rng.choice(en[fp], size=n, replace=False):
            j = min(pool, key=lambda i: abs(en[i] - tgt))
            pool.remove(j); chosen.append(j)
        return np.array(chosen)

    tp = matched(np.where((outcome == "TP"))[0], N_TP)
    tn = matched(np.where((outcome == "TN"))[0], N_TN)
    for nm, ix in [("TP ctrl", tp), ("TN ctrl", tn)]:
        print(f"{nm}: n={len(ix)} median {np.median(en[ix]):.1f} dB")

    items = ([(i, "high_conf_FP") for i in fp] + [(i, "control_TP") for i in tp]
             + [(i, "control_TN") for i in tn])
    order = rng.permutation(len(items))

    cards, key = [], []
    half = int(CTX * SR / 2)
    for n, oi in enumerate(order):
        i, kind = items[oi]
        s = int(starts[i])
        mid = s + WIN // 2
        lo = max(0, mid - half); hi = min(len(rec), lo + int(CTX * SR))
        lo = max(0, hi - int(CTX * SR))
        ctx = rec[lo:hi]
        cid = f"c{n:03d}"
        cards.append(dict(id=cid, audio=wav_b64(ctx),
                          img=spec_b64(ctx, (s - lo) / SR, (s - lo + WIN) / SR),
                          db=round(float(en[i]), 1), t=round(float(s / SR), 1)))
        key.append(dict(id=cid, kind=kind, row=int(i), window_id=bids[i],
                        t_in_file_s=float(s / SR), p=float(pb[i]), y=int(yb[i]),
                        outcome=outcome[i], db=float(en[i])))
        if n % 20 == 0:
            print(f"  built {n}/{len(items)}", flush=True)

    (ANA / "adjudication_key.json").write_text(json.dumps(
        dict(note="ANSWER KEY -- deliberately absent from the published page",
             hi_conf_threshold=HI, seed=SEED, context_sec=CTX,
             counts={k: sum(1 for r in key if r["kind"] == k) for k in
                     ("high_conf_FP", "control_TP", "control_TN")},
             cards=key), indent=2))
    (TMP / "adjudication_cards.json").write_text(json.dumps(cards))
    mb = len(json.dumps(cards)) / 1e6
    print(f"\n{len(cards)} cards, payload {mb:.1f} MB")
    print("key  ->", ANA / "adjudication_key.json")
    print("cards->", TMP / "adjudication_cards.json")


if __name__ == "__main__":
    main()
