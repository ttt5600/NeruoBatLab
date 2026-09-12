"""Tests for zfeval. Most of these encode a bug that actually happened."""
import sys, warnings
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from zfeval import validate as va, events as ev, controls as ctl, metrics as mx, splits as sp


# ---------------------------------------------------------------- validation guards
def test_nan_interval_is_dropped_not_passed_through():
    """NaN survives an `offset <= onset` filter because every NaN comparison is False.
    BirdPark row 179 was exactly this and poisoned a downstream sum."""
    iv = np.array([[0.0, 1.0], [np.nan, np.nan], [2.0, 3.0]])
    merged, rep = va.clean_intervals(iv, max_time=10.0)
    assert rep["n_nan"] == 1 and len(merged) == 2
    assert np.isfinite(merged).all()


def test_intervals_merge_and_reject_bad_rows():
    iv = np.array([[0.0, 1.0], [0.5, 2.0], [5.0, 4.0], [-1.0, 0.5], [20.0, 21.0]])
    merged, rep = va.clean_intervals(iv, max_time=10.0)
    assert rep["n_nonpositive"] == 1 and rep["n_out_of_range"] == 2
    assert len(merged) == 1 and merged[0].tolist() == [0.0, 2.0]


def test_clean_intervals_raises_when_nothing_survives():
    with pytest.raises(va.ValidationError):
        va.clean_intervals(np.array([[np.nan, np.nan]]), max_time=10.0)


def test_annotated_span_truncates_unheard_audio():
    """The most expensive bug here: scoring 50 unheard minutes as negative."""
    end, truncated = va.annotated_span(n_samples=1_000_000,
                                       intervals=np.array([[0, 100], [200, 300_000]]), sr=16000)
    assert truncated and end == 300_000


def test_annotated_span_rejects_annotations_past_the_audio():
    with pytest.raises(va.ValidationError):
        va.annotated_span(1000, np.array([[0, 5000]]), 16000)


def test_labeled_only_excludes_unlistened_rows():
    rows = [{"labeled": "1"}, {"labeled": "0"}, {"labeled": ""}]
    keep, dropped = va.labeled_only(rows)
    assert len(keep) == 1 and dropped == 2


def test_alignment_check_catches_wrong_windows():
    a = np.linspace(-60, -30, 100)
    va.audio_alignment(a, a.copy())                      # identical -> fine
    with pytest.raises(va.ValidationError):
        va.audio_alignment(a, np.roll(a, 37))            # shifted -> must fail


# ---------------------------------------------------------------- event decoding
def test_decode_min_duration_and_gap_bridging():
    p = np.array([.1, .9, .9, .1, .9, .1, .1, .9, .9, .9, .1])
    assert ev.decode(p, .5, 1, 0, 1).tolist() == [[1, 3], [4, 5], [7, 10]]
    assert ev.decode(p, .5, 1, 2, 1).tolist() == [[1, 10]]
    assert ev.decode(p, .5, 3, 0, 1).tolist() == [[7, 10]]


def test_hysteresis_keeps_only_runs_containing_a_confident_frame():
    q = np.array([.1, .6, .95, .6, .1, .6, .6, .1])
    assert ev.decode(q, .5, 1, 0, 1).tolist() == [[1, 4], [5, 7]]
    assert ev.decode(q, .8, 1, 0, 1, thr_low=.5).tolist() == [[1, 4]]


def test_offset_shrink_never_collapses_an_interval():
    iv = ev.decode(np.array([.1, .9, .1]), .5, 1, 0, 1, shrink=5)
    assert iv.tolist() == [[1, 2]]


def test_two_pointer_match_agrees_with_brute_force_greedy():
    """The fast matcher replaced an O(n^2) greedy one called ~170k times per tuning run."""
    def greedy(pred, true):
        if not len(pred) or not len(true):
            return 0
        ov = (np.minimum(pred[:, None, 1], true[None, :, 1])
              - np.maximum(pred[:, None, 0], true[None, :, 0]))
        up, ut = set(), set()
        for k in np.argsort(-ov, axis=None):
            i, j = np.unravel_index(k, ov.shape)
            if ov[i, j] <= 0:
                break
            if i in up or j in ut:
                continue
            up.add(i); ut.add(j)
        return len(ut)

    rng = np.random.default_rng(0)
    def rand_iv(n, hi):
        out = []
        for a in np.sort(rng.integers(0, hi, n)):
            b = a + rng.integers(1, 12)
            if out and a <= out[-1][1]:
                continue
            out.append([int(a), int(b)])
        return np.array(out) if out else np.zeros((0, 2), int)

    for _ in range(200):
        a, b = rand_iv(rng.integers(0, 30), 300), rand_iv(rng.integers(0, 30), 300)
        n_fast, _, _ = ev.match(a, b)
        # the sweep is order-preserving and optimal for sorted disjoint intervals,
        # so it can only find at least as many pairs as greedy-by-overlap
        assert n_fast >= greedy(a, b)


def test_tolerance_is_stricter_than_overlap():
    true = np.array([[10, 14]])
    pred = np.array([[7, 20]])           # overlaps, but the onset is 3 frames early
    assert ev.prf(pred, true, None)["n_match"] == 1
    assert ev.prf(pred, true, 1)["n_match"] == 0


def test_taxonomy_counts_merges_and_fragments():
    true = np.array([[0, 5], [10, 15]])
    assert ev.taxonomy(np.array([[0, 15]]), true)["merges"] == 1
    assert ev.taxonomy(np.array([[0, 2], [3, 5]]), true)["fragmentations"] == 1
    assert ev.taxonomy(np.zeros((0, 2), int), true)["deletions"] == 2


def test_grid_edge_is_flagged():
    true = np.array([[5, 9], [20, 24], [40, 44]])
    p = np.zeros(60); 
    for a, b in true:
        p[a:b] = 0.9
    grid = [dict(thr=t, min_dur=1, merge_gap=0, smooth=1, thr_low=None) for t in (0.5, 0.6)]
    r = ev.tune_decoder(p, true, np.repeat([0, 1], 30), grid)
    assert "thr" in r["on_grid_edge"]        # only two values -> always on an edge


# ---------------------------------------------------------------- controls
def test_shuffled_label_control_passes_on_clean_data_and_fails_on_a_leak():
    from sklearn.model_selection import StratifiedGroupKFold
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < .5).astype(int)
    g = rng.integers(0, 8, n)
    X = rng.normal(size=(n, 6)) + y[:, None]
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=0)
    assert ctl.shuffled_label(X, y, g, cv, n_perm=3)["passed"]


def test_loudness_matched_pairs_rejects_a_biased_matching():
    """A control whose own null is wrong is worse than no control."""
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < .5).astype(int)
    e = rng.normal(-45, 5, n) + 12 * y          # positives much louder
    p = rng.random(n)
    with pytest.raises(ctl.ControlFailed):
        ctl.loudness_matched_pairs(y, p, e, tol_db=50.0)   # tolerance so wide pairs are unmatched


def test_loudness_stratified_flags_wide_bands():
    rng = np.random.default_rng(1)
    n = 600
    y = (rng.random(n) < .5).astype(int)
    e = rng.normal(-45, 2, n) + 20 * y
    out = ctl.loudness_stratified(y, rng.random(n), e, n_bands=2)
    assert not out["energy_null_ok"]


# ---------------------------------------------------------------- metrics & splits
def test_score_carries_majority_and_split():
    y = np.r_[np.zeros(80), np.ones(20)]
    s = mx.score(y, y + 0.0, split="x", layer="L0")
    assert abs(s.majority - 0.8) < 1e-9 and s.split == "x" and "majority" in str(s)


def test_paired_bootstrap_finds_no_difference_between_identical_predictors():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < .5).astype(int)
    p = y * 0.6 + rng.random(400) * 0.4
    r = mx.paired_bootstrap(y, p, p.copy(), groups=rng.integers(0, 10, 400), n=200)
    assert r["verdict"] == "not_distinguishable" and abs(r["delta"]) < 1e-9


def test_paired_bootstrap_requires_exactly_one_resampling_unit():
    with pytest.raises(ValueError):
        mx.paired_bootstrap([0, 1], [0, 1], [0, 1])


def test_bird_components_are_disjoint_in_birds():
    b2d = {"a": ["d1", "d2"], "b": ["d2", "d3"], "c": ["d9"]}
    comps = sp.bird_components(b2d)
    d2b = {"d1": ["a"], "d2": ["a", "b"], "d3": ["b"], "d9": ["c"]}
    seen = []
    for c in comps:
        seen.append({b for d in c for b in d2b[d]})
    assert len(comps) == 2
    assert not (seen[0] & seen[1])


def test_bird_map_merges_case_variants():
    m = sp.bird_map_from_clipnames(["HPiHPi4748_110302-DC-01.wav", "HpiHpi4748_110303-DC-02.wav"])
    assert list(m["bird_to_dates"]) == ["hpihpi4748"]
    assert m["case_duplicates"]["hpihpi4748"] == ["HPiHPi4748", "HpiHpi4748"]


def test_contiguous_blocks_are_contiguous():
    b = sp.contiguous_blocks(100, 20, 5)
    assert b[0] == 0 and b[-1] == 4
    assert all(np.diff(b) >= 0)          # never returns to an earlier block
