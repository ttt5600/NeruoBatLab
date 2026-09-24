import torch, numpy as np, os
S = "/global/scratch/users/jonathanswang/temp_files"
CAND = [("combined k=200", S + "/combined_zf_fsd/data/spectrogram/label/label_train.pt", 200),
        ("run11 k=100",    S + "/run5-4-26-full/data/spectrogram/label/label_train.pt", 100),
        ("replay k=100",   S + "/replay_fsd50k/data/spectrogram/label/label_train.pt", 100),
        ("mix smoke",      S + "/mix_zf_fsd50k1/label/label_train.pt", None)]
print("{:<16}{:>7}{:>14}{:>8}{:>9}{:>10}{:>9}{:>8}".format(
    "corpus", "k", "frames", "used", "largest", "entropy", "eff", "perplex"))
for name, p, k in CAND:
    if not os.path.exists(p):
        print("{:<16}  (absent)".format(name)); continue
    o = torch.load(p, map_location="cpu", weights_only=False)
    per = o["labels"] if isinstance(o, dict) and "labels" in o else o
    lab = np.concatenate([np.asarray(x).ravel() for x in per]) if isinstance(per, (list, tuple)) else np.asarray(per).ravel()
    u, c = np.unique(lab, return_counts=True)
    kk = k or (int(lab.max()) + 1)
    pr = c / c.sum(); H = float(-(pr * np.log(pr + 1e-12)).sum())
    print("{:<16}{:>7}{:>14,}{:>8}{:>8.2f}%{:>10.3f}{:>9.3f}{:>8.1f}".format(
        name, kk, lab.shape[0], len(u), c.max() / len(lab) * 100, H, H / np.log(kk), float(np.exp(H))))
    tiny = int((c / c.sum() < 1e-4).sum())
    print("{:<16}  clusters under 0.01% of frames: {} of {}   smallest={} frames".format("", tiny, kk, c.min()))
