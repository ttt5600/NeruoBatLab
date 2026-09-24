import torch, numpy as np
C = "/global/scratch/users/jonathanswang/temp_files/combined_zf_fsd/data/spectrogram"
obj = torch.load(C + "/label/label_train.pt", map_location="cpu", weights_only=False)
print("[keys]     " + str(list(obj.keys())))
for k, v in obj.items():
    if k == "labels":
        continue
    print("[{}] {}".format(k, str(v)[:200]))
per = obj["labels"]
print("[files]    {} label sequences".format(len(per)))
lens = np.array([len(np.asarray(x).ravel()) for x in per])
lab = np.concatenate([np.asarray(x).ravel() for x in per])
print("[labels]   {:,} frames  dtype={}  min={}  max={}".format(lab.shape[0], lab.dtype, lab.min(), lab.max()))
print("[per-file] shortest={:,}  longest={:,}  median={:,} frames".format(lens.min(), lens.max(), int(np.median(lens))))
print("[empty]    {} files with zero frames".format(int((lens == 0).sum())))
u, c = np.unique(lab, return_counts=True)
print("[clusters] {} distinct of 200  -- empty: {}".format(len(u), 200 - len(u)))
print("[usage]    min={:,}  max={:,}  median={:,}".format(c.min(), c.max(), int(np.median(c))))
print("[balance]  largest = {:.2f}% of frames (uniform = 0.50%)".format(c.max() / len(lab) * 100))
print("[top5 %]   " + str((np.sort(c)[::-1][:5] / len(lab) * 100).round(2).tolist()))
p = c / c.sum(); H = float(-(p * np.log(p + 1e-12)).sum())
print("[entropy]  {:.3f} / {:.3f} max  -> efficiency {:.3f}".format(H, np.log(200), H / np.log(200)))
rows = [l.split() for l in open(C + "/tsv/ZF_test_pipeline_train.tsv").read().splitlines()[1:]]
audio_h = sum(int(r[-1]) for r in rows) / 16000 / 3600
print("[tsv]      {} files  {:.2f} h at 16 kHz".format(len(rows), audio_h))
print("[fps]      {:.2f} frames/s implied  ({:.2f} h of labels)".format(lab.shape[0] / (audio_h * 3600), lab.shape[0] / 50 / 3600))
# which files are ZF vs FSD -- the mix is the whole point of this corpus
zf = [i for i, r in enumerate(rows) if "fsd" not in r[0].lower()]
fs = [i for i, r in enumerate(rows) if "fsd" in r[0].lower()]
print("[mix]      ZF files={} ({:.1f} h)   FSD files={} ({:.1f} h)".format(
    len(zf), sum(int(rows[i][-1]) for i in zf) / 16000 / 3600,
    len(fs), sum(int(rows[i][-1]) for i in fs) / 16000 / 3600))
if len(lens) == len(rows):
    zf_fr = sum(lens[i] for i in zf); fs_fr = sum(lens[i] for i in fs)
    print("[frames]   ZF={:,} ({:.1f}%)  FSD={:,} ({:.1f}%)".format(
        zf_fr, zf_fr / lab.shape[0] * 100, fs_fr, fs_fr / lab.shape[0] * 100))
