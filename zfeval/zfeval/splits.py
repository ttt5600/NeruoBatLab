"""Splits, and an honest report of what each one actually holds out."""
from __future__ import annotations
import re, collections
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut


def bird_map_from_clipnames(names):
    """bird <-> datecode from curated clip filenames: Bird_datecode-CallType-idx.wav

    Case-normalizes: HPiHPi4748 and HpiHpi4748 are one bird. 'Unknown000' is a placeholder.
    """
    pat = re.compile(r"^([A-Za-z]+\d+)_(\d{6})-(.+)\.wav$")
    pairs, variants = [], collections.defaultdict(set)
    for n in names:
        n = n.split("/")[-1]
        if n.startswith("._"):
            continue
        m = pat.match(n)
        if m:
            b = m.group(1).lower()
            variants[b].add(m.group(1))
            pairs.append((b, m.group(2)))
    b2d, d2b = collections.defaultdict(set), collections.defaultdict(set)
    for b, d in pairs:
        b2d[b].add(d); d2b[d].add(b)
    return dict(bird_to_dates={k: sorted(v) for k, v in b2d.items()},
                date_to_birds={k: sorted(v) for k, v in d2b.items()},
                case_duplicates={k: sorted(v) for k, v in variants.items() if len(v) > 1})


def bird_components(bird_to_dates):
    """Minimal groups of dates such that no bird spans two groups.

    Colony recordings put several birds on one date (median 2, max 7) and one bird on several
    dates (median 5), so holding out a recording does NOT hold out its birds. Dates sharing a bird
    must travel together; the connected components are the finest bird-disjoint split available.
    """
    dates = sorted({d for ds in bird_to_dates.values() for d in ds})
    parent = {d: d for d in dates}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for ds in bird_to_dates.values():
        ds = sorted(ds)
        for x in ds[1:]:
            ra, rb = find(ds[0]), find(x)
            if ra != rb:
                parent[ra] = rb
    comp = collections.defaultdict(list)
    for d in dates:
        comp[find(d)].append(d)
    return [sorted(v) for v in comp.values()]


def bird_leakage(groups, dates, date_to_birds, n_splits=5, seed=0, y=None):
    """What fraction of test-fold birds also appear in training under this grouping?

    On our corpus, recording-grouped CV leaks 96.8% of test birds into training. That is not
    automatically fatal -- for detection it costs -0.002 AUC -- but it must be measured, not
    assumed away.
    """
    groups, dates = np.asarray(groups), np.asarray(dates)
    y = np.zeros(len(groups), int) if y is None else np.asarray(y)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    frac = []
    for tr, te in cv.split(np.zeros((len(y), 1)), y, groups=groups):
        b_tr = {b for d in dates[tr] if d in date_to_birds for b in date_to_birds[d]}
        b_te = {b for d in dates[te] if d in date_to_birds for b in date_to_birds[d]}
        if b_te:
            frac.append(len(b_te & b_tr) / len(b_te))
    return dict(mean_leaked_fraction=float(np.mean(frac)) if frac else None,
                per_fold=[float(f) for f in frac])


def contiguous_blocks(n_frames, block_frames, n_blocks=5):
    """Temporal blocks for within-recording CV.

    Random frame splits leak: adjacent 20 ms frames are near-identical, so the model sees the
    answer from the other side of the fold boundary. Blocks must be contiguous.
    """
    return np.minimum(np.arange(n_frames) // block_frames, n_blocks - 1)


def make_cv(kind, n_splits=5, seed=0):
    if kind == "leave_one_group_out":
        return LeaveOneGroupOut()
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def choose_cv(groups, starts=None, n_splits=5, seed=0, block_sec=60.0, sr=16000):
    """Pick a valid split for this dataset and SAY WHAT IT IS.

    A grouped CV needs at least n_splits groups. A single-recording window set (e.g. a file tiled
    into 1 s windows) has one group, and StratifiedGroupKFold silently produces empty folds there.
    Falling back without saying so would be worse than crashing: the reported number would look
    like a leave-recordings-out score and would not be one.

    Preference order:
      >= n_splits groups        -> StratifiedGroupKFold, leave-recordings-out
      1 group + window starts   -> contiguous TIME blocks, so neighbouring windows cannot straddle
                                   a fold boundary (adjacent windows overlap in context)
      1 group, no starts        -> StratifiedKFold, clearly labelled as within-recording random
    """
    from sklearn.model_selection import StratifiedKFold
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups >= n_splits:
        return (StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
                groups, f"leave-recordings-out ({n_groups} recordings, {n_splits}-fold)")
    if starts is not None:
        starts = np.asarray(starts, dtype=float)
        blk = (starts / (block_sec * sr)).astype(int)
        _, blk = np.unique(blk, return_inverse=True)
        nb = len(np.unique(blk))
        if nb >= n_splits:
            return (StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
                    blk, f"contiguous {block_sec:g}s time blocks within {n_groups} recording(s) "
                         f"({nb} blocks) -- NOT a recording holdout")
    return (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), None,
            f"random {n_splits}-fold WITHIN {n_groups} recording(s) -- NOT a recording holdout, "
            f"adjacent windows share context so this is optimistic")
