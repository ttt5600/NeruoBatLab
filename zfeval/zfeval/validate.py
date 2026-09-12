"""Input guards. Each one exists because its absence produced a wrong number at least once."""
from __future__ import annotations
import numpy as np


class ValidationError(Exception):
    pass


def annotated_span(n_samples: int, intervals: np.ndarray, sr: int, name: str = "recording"):
    """Truncate a recording to the span someone actually listened to.

    111021-000 is 80.6 min long and only the first 30.0 min are annotated. Scoring the whole file
    labels 50 minutes of unheard audio as negative, which dropped frame AUC from 0.968 to 0.849
    and was only noticed because the voiced fraction printed 5.7% instead of the known 15.2%.
    """
    if len(intervals) == 0:
        raise ValidationError(f"{name}: no annotation intervals")
    end = int(np.ceil(intervals[:, 1].max() * sr)) if intervals.dtype.kind == "f" \
        else int(intervals[:, 1].max())
    if end > n_samples:
        raise ValidationError(f"{name}: annotations run past the audio "
                              f"({end} > {n_samples} samples)")
    return end, end < n_samples


def clean_intervals(iv: np.ndarray, max_time: float, name: str = "annotations"):
    """Drop NaN, non-positive and out-of-range intervals, then merge overlaps.

    NaN first: every comparison against NaN is False, so an `offset <= onset` filter passes a NaN
    row straight through and it then poisons every downstream sum. BirdPark row 179 is exactly
    that case.
    """
    iv = np.asarray(iv, dtype=float)
    if iv.ndim != 2 or iv.shape[1] != 2:
        raise ValidationError(f"{name}: expected (n,2), got {iv.shape}")
    n0 = len(iv)
    nan = np.isnan(iv).any(axis=1)
    iv = iv[~nan]
    bad = iv[:, 1] <= iv[:, 0]
    oob = (iv[:, 0] < 0) | (iv[:, 1] > max_time)
    iv = iv[~(bad | oob)]
    if not len(iv):
        raise ValidationError(f"{name}: no usable intervals survived cleaning")
    iv = iv[np.argsort(iv[:, 0])]
    merged = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    report = dict(n_in=int(n0), n_nan=int(nan.sum()), n_nonpositive=int(bad.sum()),
                  n_out_of_range=int(oob.sum()), n_merged=len(merged))
    return np.array(merged), report


def labeled_only(rows, label_field="labeled"):
    """labeled=0 means NOBODY LISTENED. It is not a negative and must never be defaulted to 0."""
    keep = [r for r in rows if str(r.get(label_field, "")).strip() == "1"]
    dropped = len(rows) - len(keep)
    if not keep:
        raise ValidationError("no rows with labeled=1")
    return keep, dropped


def audio_alignment(local_db: np.ndarray, reference_db: np.ndarray, name: str = "windows",
                    min_corr: float = 0.99, max_median_diff: float = 1.0):
    """Confirm locally cut windows are the same windows a feature extractor saw.

    Cheap and decisive: recompute each window's dB and compare. If a rate conversion or an offset
    is wrong, this catches it before any model number is produced.
    """
    if len(local_db) != len(reference_db):
        raise ValidationError(f"{name}: length mismatch {len(local_db)} vs {len(reference_db)}")
    corr = float(np.corrcoef(local_db, reference_db)[0, 1])
    md = float(np.median(np.abs(local_db - reference_db)))
    if corr < min_corr or md > max_median_diff:
        raise ValidationError(f"{name}: windows do not match the extracted ones "
                              f"(corr={corr:.6f}, median |diff|={md:.3f} dB)")
    return dict(corr=corr, median_abs_diff_db=md)


def finite(X: np.ndarray, name: str = "features"):
    if not np.isfinite(X).all():
        n = int((~np.isfinite(X)).sum())
        raise ValidationError(f"{name}: {n} non-finite values")
    return X
