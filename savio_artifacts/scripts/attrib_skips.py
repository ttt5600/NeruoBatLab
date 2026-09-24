import soundfile as sf, numpy as np, glob, sys
SR, MIN_S, SIL = 16000, 0.10, -70.0
files = sorted(glob.glob("/global/scratch/users/jonathanswang/external/fsd50k/FSD50K.dev_audio/*.wav"))
short, bad_hdr, cand = [], [], []
for f in files:
    try:
        i = sf.info(f)
        d = i.frames / i.samplerate
    except Exception as e:
        bad_hdr.append((f, str(e))); continue
    if d < MIN_S: short.append((f, d))
print(f"total {len(files)}  header-unreadable {len(bad_hdr)}  shorter than {MIN_S}s: {len(short)}")
for f, d in short[:10]: print(f"   SHORT {f.split(chr(47))[-1]}  {d*1000:.1f} ms")
for f, e in bad_hdr[:5]: print(f"   HDR   {f.split(chr(47))[-1]}  {e[:60]}")
