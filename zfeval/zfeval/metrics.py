"""Scores that refuse to be quoted without their context."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve


@dataclass
class Score:
    """A metric bundle that always carries its split and its majority rate.

    Accuracy on this task is meaningless alone -- eval A's majority class is 62.6% and eval B's is
    67.6%, so an 'accuracy of 0.685' can be below chance-after-prior. The dataclass makes it
    awkward to report one without the other.
    """
    split: str
    n: int
    majority: float
    auc: float
    ap: float
    acc: float
    prevalence: float
    layer: str | None = None
    note: str = ""

    def __str__(self):
        L = f"[{self.layer}] " if self.layer else ""
        return (f"{L}{self.split}: n={self.n} AUC={self.auc:.4f} AP={self.ap:.4f} "
                f"acc={self.acc:.4f} (majority {self.majority:.4f})")

    def to_dict(self):
        return asdict(self)


def score(y, p, split: str, layer: str | None = None, thr: float = 0.5, note: str = "") -> Score:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    prev = float(y.mean())
    return Score(split=split, n=int(len(y)), majority=float(max(prev, 1 - prev)),
                 auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                 acc=float(accuracy_score(y, p > thr)), prevalence=prev, layer=layer, note=note)


def operating_points(y, p, false_alarm_rates=(0.01, 0.05, 0.10, 0.20)):
    """AUC translated into 'at this false-alarm budget, how many calls do you keep'."""
    y = np.asarray(y).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    out = {}
    for fa in false_alarm_rates:
        r = float(np.interp(fa, fpr, tpr))
        tp, fp = r * n_pos, fa * n_neg
        out[f"fa_{fa:g}"] = dict(recall=r, calls_found=round(tp), false_alarms=round(fp),
                                 precision=float(tp / (tp + fp)) if tp + fp else 0.0)
    return out


def ranking_error(auc: float) -> float:
    """1 - AUC. Easier to compare than AUC: 0.9267 -> 0.9774 is a 69% error reduction."""
    return 1.0 - auc


def paired_bootstrap(y, p_a, p_b, groups=None, block: int | None = None, n: int = 2000,
                     seed: int = 0, metric=roc_auc_score):
    """Interval on (metric(p_a) - metric(p_b)), resampling the unit that is actually independent.

    groups -> cluster bootstrap over those groups (use for recording-grouped CV).
    block  -> moving-block bootstrap of that many consecutive rows (use for tiled windows or
              frames, where neighbours are correlated and a plain row bootstrap understates the
              interval).
    Exactly one of groups/block must be given.
    """
    y = np.asarray(y).astype(int)
    p_a, p_b = np.asarray(p_a, float), np.asarray(p_b, float)
    if (groups is None) == (block is None):
        raise ValueError("give exactly one of groups= or block=")
    rng = np.random.default_rng(seed)
    deltas = []
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        idx = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n):
            sel = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
            if len(np.unique(y[sel])) < 2:
                continue
            deltas.append(metric(y[sel], p_a[sel]) - metric(y[sel], p_b[sel]))
    else:
        nb = max(1, len(y) // block)
        for _ in range(n):
            starts = rng.integers(0, max(1, len(y) - block), nb)
            sel = np.concatenate([np.arange(s, s + block) for s in starts])
            sel = sel[sel < len(y)]
            if len(np.unique(y[sel])) < 2:
                continue
            deltas.append(metric(y[sel], p_a[sel]) - metric(y[sel], p_b[sel]))
    d = np.array(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta=float(d.mean()), lo=float(lo), hi=float(hi), n_resamples=int(len(d)),
                verdict=("a_better" if lo > 0 else "b_better" if hi < 0 else "not_distinguishable"))
