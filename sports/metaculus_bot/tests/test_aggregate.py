import math

import pytest

from mbot import aggregate as A


def test_median_binary_and_clip():
    assert A.median_binary([0.1, 0.9, 0.3]) == 0.3
    assert A.median_binary([0.2, 0.4]) == pytest.approx(0.3)
    assert A.clip_binary(0.995, 0.02, 0.98) == 0.98
    assert A.clip_binary(0.001, 0.02, 0.98) == 0.02
    with pytest.raises(ValueError):
        A.median_binary([])
    with pytest.raises(ValueError):
        A.median_binary([1.2])


def test_extremize_is_identity_by_default():
    for p in (0.03, 0.4, 0.77):
        assert A.extremize_binary(p, 1.0) == p
    assert A.extremize_binary(0.8, 1.5) > 0.8 and A.extremize_binary(0.2, 1.5) < 0.2


def test_median_mc_and_floor():
    opts = ["A", "B", "C"]
    s = [{"A": 0.7, "B": 0.3, "C": 0.0}, {"A": 0.6, "B": 0.4, "C": 0.0}, {"A": 0.9, "B": 0.1, "C": 0.0}]
    m = A.median_mc(s, opts)
    assert math.isclose(sum(m.values()), 1.0) and m["C"] == 0.0
    f = A.floor_mc(m, 0.02)
    assert min(f.values()) >= 0.02 - 1e-12
    assert sum(f.values()) == pytest.approx(1.0, abs=1e-12)
    assert f["A"] > f["B"] > f["C"]


def test_floor_scales_down_for_many_options():
    probs = {f"o{i}": (0.9 if i == 0 else 0.1 / 39) for i in range(40)}
    f = A.floor_mc(probs, 0.02)
    floor = 0.2 / 40
    assert min(f.values()) >= floor - 1e-12
    assert sum(f.values()) == pytest.approx(1.0, abs=1e-12)
    assert f["o0"] > 0.75          # floors never take more than ~20% of the mass
