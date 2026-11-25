import numpy as np
import pytest

from mbot import cdf as C
from mbot.parsing import parse_percentiles


def _spec(questions, qid):
    return C.Spec.from_question(questions[qid])


def test_linear_open_bounds(questions, llm_out):
    s = _spec(questions, 8003)
    cdf = C.percentiles_to_cdf(parse_percentiles(llm_out["numeric_ok"]), s)
    assert len(cdf) == 201 and C.validate_cdf(cdf, s) == []
    assert cdf[0] >= 0.001 and cdf[-1] <= 0.999
    assert 4.0 < C.quantile(cdf, s, 0.5) < 4.2


def test_log_scaled_closed_lower(questions, llm_out):
    s = _spec(questions, 8004)
    cdf = C.percentiles_to_cdf(parse_percentiles(llm_out["numeric_commas"]), s)
    assert C.validate_cdf(cdf, s) == [] and cdf[0] == 0.0
    assert 1500 < C.quantile(cdf, s, 0.5) < 2200


def test_discrete_and_date(questions, llm_out):
    s = _spec(questions, 8005)
    cdf = C.percentiles_to_cdf(parse_percentiles(llm_out["discrete_ok"]), s)
    assert len(cdf) == 21 and C.validate_cdf(cdf, s) == [] and cdf[0] == 0.0
    assert min(np.diff(cdf)) >= 0.01 / 20 - 1e-9
    sd = _spec(questions, 8006)
    cdf = C.percentiles_to_cdf(parse_percentiles(llm_out["date_ok"], is_date=True), sd)
    assert C.validate_cdf(cdf, sd) == []
    assert "p50=2027-0" in C.readable(cdf, sd)


def _doc_nominal_to_location(v, lo, hi, zp):
    # verbatim formula from docs/openapi.yml "nominal_location_to_cdf_location"
    if zp is not None:
        r = (hi - zp) / (lo - zp)
        return (np.log((v - lo) * (r - 1) + (hi - lo)) - np.log(hi - lo)) / np.log(r)
    return (v - lo) / (hi - lo)


@pytest.mark.parametrize("lo,hi,zp", [(200.0, 1e8, 0.0), (3.0, 5.5, None), (1.0, 50.0, -10.0)])
def test_axis_matches_documented_formula(lo, hi, zp):
    # Note: the continuous_range inside the openapi.yml *example* post (213.6328 at index 1) does not follow the
    # documented formula (gives 213.5624); it looks like stale sample data, so the formula itself is the oracle.
    s = C.Spec(lo, hi, zp, True, True)
    xs = C.x_axis(s)
    assert xs[0] == pytest.approx(lo) and xs[-1] == pytest.approx(hi)
    for i in (1, 37, 100, 199):
        assert _doc_nominal_to_location(xs[i], lo, hi, zp) == pytest.approx(i / 200, abs=1e-9)
        assert C.nominal_to_location(xs[i], s) == pytest.approx(i / 200, abs=1e-9)


def test_concentrated_forecast_is_capped():
    s = C.Spec(0.0, 100.0, None, True, True)
    cdf = C.percentiles_to_cdf([(0.1, 50.0), (0.5, 50.1), (0.9, 50.2)], s)
    assert C.validate_cdf(cdf, s) == []
    assert max(np.diff(cdf)) <= 0.2


def test_validator_catches_violations():
    s = C.Spec(0.0, 10.0, None, open_lower=False, open_upper=True)
    good = C.percentiles_to_cdf([(0.1, 2.0), (0.9, 8.0)], s)
    assert C.validate_cdf(good, s) == []
    assert "length" in C.validate_cdf(good[:-1], s)[0]
    bad = list(good); bad[0] = 0.01
    assert any("closed lower" in e for e in C.validate_cdf(bad, s))
    bad = list(good); bad[-1] = 0.9995
    assert any("open upper" in e for e in C.validate_cdf(bad, s))
    bad = list(good); bad[100] = bad[99]
    assert any("min step" in e for e in C.validate_cdf(bad, s))
    flat = list(np.linspace(0, 0.3, 201)); flat[150:] = list(np.linspace(0.9, 0.95, 51))
    assert any("max step" in e for e in C.validate_cdf(flat, s))


def test_median_cdf_valid(questions, llm_out):
    s = _spec(questions, 8003)
    a = C.percentiles_to_cdf([(0.1, 3.2), (0.5, 3.4), (0.9, 3.6)], s)
    b = C.percentiles_to_cdf([(0.1, 4.0), (0.5, 4.1), (0.9, 4.2)], s)
    c = C.percentiles_to_cdf([(0.1, 5.0), (0.5, 5.2), (0.9, 5.4)], s)
    m = C.median_cdf([a, b, c], s)
    assert C.validate_cdf(m, s) == []
    assert 4.0 < C.quantile(m, s, 0.5) < 4.3


def test_far_out_of_range_rejected():
    s = C.Spec(0.0, 10.0, None, True, True)
    with pytest.raises(ValueError):
        C.percentiles_to_cdf([(0.1, 100.0), (0.9, 200.0)], s)


# ------------------------------------------------------------------ parity with the official package (optional)
CASES = [
    (C.Spec(3.0, 5.5, None, True, True), [(0.1, 3.85), (0.2, 3.95), (0.4, 4.07), (0.6, 4.15), (0.8, 4.27), (0.9, 4.38)]),
    (C.Spec(100.0, 1e5, 0.0, False, True), [(0.1, 1200), (0.2, 1450), (0.4, 1800), (0.6, 2150), (0.8, 2900), (0.9, 3600)]),
    (C.Spec(-0.5, 19.5, None, False, True, cdf_size=21), [(0.1, 2), (0.2, 3), (0.4, 5), (0.6, 6), (0.8, 8), (0.9, 10)]),
    (C.Spec(0.0, 100.0, None, False, False), [(0.05, 0.0), (0.5, 40.0), (0.95, 100.0)]),
]


@pytest.mark.parametrize("spec,pcts", CASES)
def test_parity_with_forecasting_tools(spec, pcts):
    ft = pytest.importorskip("forecasting_tools.data_models.numeric_report")
    nd = ft.NumericDistribution(
        declared_percentiles=[ft.Percentile(percentile=p, value=v) for p, v in pcts],
        open_upper_bound=spec.open_upper, open_lower_bound=spec.open_lower, upper_bound=spec.range_max,
        lower_bound=spec.range_min, zero_point=spec.zero_point, cdf_size=spec.cdf_size)
    theirs = [p.percentile for p in nd.get_cdf()]
    ours = C.percentiles_to_cdf(pcts, spec)
    assert np.max(np.abs(np.asarray(theirs) - np.asarray(ours))) < 1e-9
