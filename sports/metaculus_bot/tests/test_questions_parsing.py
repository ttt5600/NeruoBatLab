import math

import pytest

from mbot.parsing import ParseError, claims_resolved, parse_binary, parse_multiple_choice, parse_percentiles
from mbot.questions import route
from conftest import NOW


def test_fixture_types_and_router(questions):
    fam = {qid: q.family for qid, q in questions.items()}
    assert fam[8001] == "binary"
    assert fam[8002] == "multiple_choice"
    assert fam[8003] == "continuous" and questions[8003].cdf_size == 201
    assert fam[8004] == "continuous" and questions[8004].zero_point == 0.0 and not questions[8004].open_lower_bound
    assert fam[8005] == "continuous" and questions[8005].cdf_size == 21          # discrete: inbound_outcome_count 20
    assert fam[8006] == "continuous" and questions[8006].type == "date"
    # group post unpacked into its sub-questions, group text inherited
    assert questions[8091].label == "Team A" and questions[8091].description == "Group description."
    assert questions[8007].already_forecast and not questions[8001].already_forecast
    assert questions[8008].minutes_left(NOW) == pytest.approx(5.0)


def test_router_unsupported(questions):
    q = questions[8001]
    q2 = type(q)(**{**q.__dict__, "type": "conditional"})
    assert route(q2) == "unsupported"
    q3 = type(q)(**{**questions[8003].__dict__, "range_min": None})
    assert route(q3) == "unsupported"
    q4 = type(q)(**{**questions[8002].__dict__, "options": ["only one"]})
    assert route(q4) == "unsupported"


def test_parse_binary(llm_out):
    assert parse_binary(llm_out["binary_ok"]) == pytest.approx(0.78)
    assert parse_binary(llm_out["binary_two_lines"]) == pytest.approx(0.35)       # last line wins
    with pytest.raises(ParseError):
        parse_binary(llm_out["binary_bad"])
    assert claims_resolved(llm_out["binary_resolved_claim"]) and not claims_resolved(llm_out["binary_ok"])


def test_parse_mc(llm_out):
    opts = ["Democratic", "Republican", "Other"]
    d = parse_multiple_choice(llm_out["mc_ok"], opts)
    assert d == pytest.approx({"Democratic": 0.55, "Republican": 0.40, "Other": 0.05})
    d2 = parse_multiple_choice(llm_out["mc_prefixed"], opts)
    assert math.isclose(sum(d2.values()), 1.0)
    with pytest.raises(ParseError):
        parse_multiple_choice(llm_out["mc_missing"], opts)


def test_parse_percentiles(llm_out):
    pts = parse_percentiles(llm_out["numeric_ok"])
    assert [p for p, _ in pts] == pytest.approx([0.1, 0.2, 0.4, 0.6, 0.8, 0.9]) and pts[0][1] == 3.85
    assert parse_percentiles(llm_out["numeric_commas"])[0][1] == 1200
    assert parse_percentiles(llm_out["numeric_suffix"])[0][1] == pytest.approx(1.2e9)
    with pytest.raises(ParseError):
        parse_percentiles(llm_out["numeric_decreasing"])
    dates = parse_percentiles(llm_out["date_ok"], is_date=True)
    assert dates[0][1] == 1801440000.0         # 2027-02-01T00:00Z
