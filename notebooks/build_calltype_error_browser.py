#!/usr/bin/env python
"""Build docs/calltype_error_browser.html: a self-contained page to click through every call-type
error made by AVES-bio, E23 DAPT and run16, with each clip's spectrogram, audio, and what each model
made of it (probability, true-type rank, distance to the true and called type's centre).

Reads analysis/calltype_errors.npz (zfeval/experiments/calltype_errors.py, which reproduces the
board's accuracies) and the original wavs. Spectrograms (55 dB range) are WebP, audio is 16 kHz 16-bit WAV,
both embedded as data URIs so the page works offline and as an Artifact.

    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 notebooks/build_calltype_error_browser.py
"""
import base64, io, json, wave
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from PIL import Image
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parent.parent
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ROOT / "docs/calltype_error_browser.html"
NAME = {"aves-base-bio": "AVES-bio", "daptv4two_2e4s2_step15000": "E23 DAPT", "run16_compute4x": "run16"}
ROLE = {"aves-base-bio": "downloaded parent", "daptv4two_2e4s2_step15000": "AVES-bio continued on zebra finch",
        "run16_compute4x": "ours, from scratch"}
FULL = {"Ag": "Wsst (aggressive)", "Be": "Begging", "DC": "Distance call", "Di": "Distress", "LT": "Long tonal",
        "Ne": "Nest", "So": "Song", "Te": "Tet", "Th": "Thuk", "Tu": "Tuck", "Wh": "Whine"}
ORDER = ["DC", "Te", "Ne", "Th", "Tu", "Ag", "Wh", "Di", "So", "Be", "LT"]
MAGMA = (colormaps["magma"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def spectro_webp(w, sr, h=112, wpx=208):
    f, t, S = signal.spectrogram(w, sr, nperseg=512, noverlap=448, window="hann")
    k = f <= 10000
    S = 10 * np.log10(S[k] + 1e-12)
    vmax = np.percentile(S, 99.5)
    S = np.clip((S - (vmax - 55)) / 55, 0, 1)[::-1]
    img = Image.fromarray(MAGMA[(S * 255).astype(np.uint8)]).resize((wpx, h), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, "WEBP", quality=60, method=6)
    return "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()


def wav_uri(w, sr, target=16000):
    w = signal.resample_poly(w, target, sr) if sr != target else w
    w = w / max(1e-9, np.abs(w).max()) * 0.9                       # peak-normalised for listening only
    pcm = (w * 32767).astype("<i2")
    b = io.BytesIO()
    with wave.open(b, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(target); f.writeframes(pcm.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    Z = np.load(ANA / "calltype_errors.npz")
    classes = [str(c) for c in Z["classes"]]; y = Z["y"]; birds = Z["birds"]; paths = Z["paths"]
    tags = [str(t) for t in Z["tags"]]
    P = {t: Z[f"P__{t}"] for t in tags}; R = {t: Z[f"R_lda__{t}"] for t in tags}
    pred = {t: P[t].argmax(1) for t in tags}
    wrong_any = np.zeros(len(y), bool)
    for t in tags:
        wrong_any |= pred[t] != y
    # typical members: the 3 correct clips nearest their own type's centre, per model and type
    typical = {}
    for t in tags:
        typical[t] = {}
        for c in range(len(classes)):
            i = np.where((y == c) & (pred[t] == y))[0]
            typical[t][classes[c]] = [int(j) for j in i[np.argsort(R[t][i, c])[:3]]]
    keep = sorted(set(np.where(wrong_any)[0].tolist()) |
                  {j for t in tags for v in typical[t].values() for j in v})

    clips, media = [], {}
    for n, i in enumerate(keep):
        w, sr = sf.read(paths[i], always_2d=True); w = w.mean(1)
        media[i] = (spectro_webp(w, sr), wav_uri(w, sr))
        per = {}
        for t in tags:
            top = np.argsort(-P[t][i])[:3]
            per[NAME[t]] = dict(c=classes[top[0]], p=round(float(P[t][i, top[0]]), 3),
                                pt=round(float(P[t][i, y[i]]), 3),
                                rk=int(np.where(np.argsort(-P[t][i]) == y[i])[0][0]) + 1,
                                top=[[classes[k], round(float(P[t][i, k]), 3)] for k in top],
                                dt=round(float(R[t][i, y[i]]), 2), dc=round(float(R[t][i, top[0]]), 2))
        clips.append(dict(i=int(i), f=Path(paths[i]).name, b=str(birds[i]), y=classes[y[i]],
                          ms=int(round(1000 * len(w) / sr)), m=per))
        if n % 100 == 0:
            print(f"  {n}/{len(keep)}", flush=True)

    conf = {}
    for t in tags:
        M = np.zeros((len(classes), len(classes)), int); np.add.at(M, (y, pred[t]), 1)
        conf[NAME[t]] = {a: {b: int(M[classes.index(a), classes.index(b)]) for b in ORDER} for a in ORDER}
    data = dict(order=ORDER, full=FULL, models=[NAME[t] for t in tags],
                role={NAME[t]: ROLE[t] for t in tags},
                layer={NAME[t]: int(Z[f"layer__{t}"]) for t in tags},
                acc={NAME[t]: round(float((pred[t] == y).mean()), 4) for t in tags},
                nclips=int(len(y)), nbirds=int(len(set(birds))),
                counts={c: int((y == classes.index(c)).sum()) for c in ORDER},
                conf=conf, typical={NAME[t]: typical[t] for t in tags}, clips=clips)
    img = {str(i): m[0] for i, m in media.items()}
    aud = {str(i): m[1] for i, m in media.items()}
    tpl = (Path(__file__).resolve().parent / "calltype_error_browser.template.html").read_text()
    html = (tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
               .replace("/*__IMG__*/null", json.dumps(img, separators=(",", ":")))
               .replace("/*__AUD__*/null", json.dumps(aud, separators=(",", ":"))))
    OUT.write_text(html)
    print(f"{len(keep)} clips ({int(wrong_any.sum())} errors of any model) -> {OUT}  "
          f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
