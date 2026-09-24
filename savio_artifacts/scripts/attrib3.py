"""Replicate load_16k_mono EXACTLY -- the earlier check measured RMS at native rate.

load_16k_mono resamples to 16 kHz BEFORE testing the silence floor. resample_poly applies an
anti-alias filter, so a clip whose energy lives entirely above 8 kHz is loud at 44.1 kHz and
silent after the decimation. That is the leading candidate for the 8 files the native-rate
scan could not account for -- and if it is not, there is a rejection path I have not found.
"""
import soundfile as sf, numpy as np, glob
from scipy.signal import resample_poly
from math import gcd
SR, SIL, MIN_SAMPLES = 16000, -70.0, int(0.10 * 16000)
def rms_db(x):
    if x.size == 0: return -np.inf
    return 20.0 * np.log10(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-12)
files = sorted(glob.glob("/global/scratch/users/jonathanswang/external/fsd50k/FSD50K.dev_audio/*.wav"))
silent_native, silent_only_after, short, unread = [], [], [], []
for f in files:
    try:
        x, sr = sf.read(f, dtype="float32", always_2d=True)
    except Exception:
        unread.append(f); continue
    x = x.mean(axis=1)
    if not np.isfinite(x).all():
        continue
    before = rms_db(x)
    if sr != SR:
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    if x.size < MIN_SAMPLES:
        short.append((f.split("/")[-1], x.size)); continue
    after = rms_db(x)
    if after < SIL:
        (silent_native if before < SIL else silent_only_after).append(
            (f.split("/")[-1], before, after, sr))
print("total skipped by the real guard: {}".format(
    len(silent_native) + len(silent_only_after) + len(short) + len(unread)))
print("  silent at native rate too:        {}".format(len(silent_native)))
print("  SILENT ONLY AFTER RESAMPLING:     {}".format(len(silent_only_after)))
print("  too short after resampling:       {}".format(len(short)))
print("  unreadable:                       {}".format(len(unread)))
for n, b, a, sr in silent_only_after[:12]:
    print("   POST-RESAMPLE {}  native={:.1f} dB -> 16k={:.1f} dB  (sr={})".format(n, b, a, sr))
for n, sz in short[:6]:
    print("   SHORT {}  {} samples".format(n, sz))
