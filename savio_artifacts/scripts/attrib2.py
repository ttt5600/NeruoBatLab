import soundfile as sf, numpy as np, glob
SR, SIL = 16000, -70.0
def rms_db(x):
    if x.size == 0: return -np.inf
    return 20.0 * np.log10(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-12)
files = sorted(glob.glob("/global/scratch/users/jonathanswang/external/fsd50k/FSD50K.dev_audio/*.wav"))
truly, cancelled, nonfinite = [], [], []
for f in files:
    try:
        x, sr = sf.read(f, dtype="float32", always_2d=True)
    except Exception:
        continue
    mono = x.mean(axis=1)
    if not np.isfinite(mono).all():
        nonfinite.append(f); continue
    m = rms_db(mono)
    if m >= SIL:
        continue
    ch = max(rms_db(x[:, c]) for c in range(x.shape[1]))
    rec = (f.split("/")[-1], m, ch, x.shape[1])
    (cancelled if ch >= SIL else truly).append(rec)
print("mono-RMS below {} dB: {}".format(SIL, len(truly) + len(cancelled)))
print("  truly silent (every channel quiet too): {}".format(len(truly)))
print("  CANCELLED BY MONO DOWNMIX (a channel is loud): {}".format(len(cancelled)))
print("  non-finite: {}".format(len(nonfinite)))
for n, m, c, nc in cancelled[:10]:
    print("   CANCEL {}  mono={:.1f} dB  maxch={:.1f} dB  ch={}".format(n, m, c, nc))
for n, m, c, nc in truly[:6]:
    print("   SILENT {}  mono={:.1f} dB  maxch={:.1f} dB  ch={}".format(n, m, c, nc))
