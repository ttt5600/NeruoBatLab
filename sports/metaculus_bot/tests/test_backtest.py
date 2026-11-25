"""Backtest scoring checked against hand calculations of the official formulas."""
import json
import math

import numpy as np
import pytest

from conftest import FIX
from mbot import backtest as bt
from mbot.forecaster import Forecaster
from mbot.llm import MockLLM
from mbot.questions import questions_from_post
from mbot.research import Researcher


@pytest.fixture
def resolved():
    return {q.question_id: q for q in bt.load_fixtures(FIX / "resolved")}


def flat_cdf(n=200, lo=0.05, hi=0.95):
    return list(np.linspace(lo, hi, n + 1))


def test_binary_and_mc_baseline_by_hand(resolved):
    s = bt.score(resolved[8101], 0.8)                      # resolved YES, CP p_yes = 0.7
    assert s["baseline"] == pytest.approx(100 * math.log2(1.6))          # 67.81
    assert s["rel_cp"] == pytest.approx(100 * math.log(0.8 / 0.7))       # 13.35
    s = bt.score(resolved[8101], 0.5)
    assert s["baseline"] == pytest.approx(0.0)                           # p = 1/K scores exactly 0
    s = bt.score(resolved[8102], {"A": 0.1, "B": 0.6, "C": 0.2, "D": 0.1})
    assert s["baseline"] == pytest.approx(100 * math.log(2.4) / math.log(4))   # 63.15
    assert s["rel_cp"] is None


def test_continuous_bucket_and_baseline_by_hand(resolved):
    q = resolved[8103]                                     # range 3.0..5.5, both bounds open, resolved 4.2
    assert bt.bucket_index(q) == 96                        # loc 0.48 -> int(0.48*200 + 1 - 1e-10) = 96
    s = bt.score(q, flat_cdf())                            # in-range mass 0.9 spread evenly = the reference density
    assert s["baseline"] == pytest.approx(0.0, abs=1e-9)
    peaked = flat_cdf()
    peaked = [c + (0.002 if i >= 96 else 0) for i, c in enumerate(peaked)]   # +0.002 mass in bucket 96
    s = bt.score(q, peaked)
    assert s["baseline"] == pytest.approx(100 * math.log((0.0045 + 0.002) / 0.0045) / 2)


def test_bucket_edges(resolved):
    import dataclasses
    q = resolved[8103]
    at = lambda r: bt.bucket_index(dataclasses.replace(q, resolution=r))  # noqa: E731
    assert at("below_lower_bound") == 0 and at("above_upper_bound") == 201
    assert at("3.0") == 1 and at("5.5") == 200 and at("2.0") == 0 and at("9") == 201
    s = bt.score(dataclasses.replace(q, resolution="above_upper_bound"), flat_cdf())
    assert s["baseline"] == pytest.approx(0.0)            # 0.05 left above the range vs reference 0.05


def test_bootstrap_ci():
    m, lo, hi = bt.bootstrap_ci([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5 and lo < 2.5 < hi
    assert math.isnan(bt.bootstrap_ci([])[0])


def test_run_backtest_end_to_end(cfg, tmp_path, resolved):
    def respond(prompt, i):
        if "FORMAT: BINARY" in prompt:
            return "Probability: 80%"
        if "FORMAT: MULTIPLE_CHOICE" in prompt:
            return "A: 10%\nB: 60%\nC: 20%\nD: 10%"
        return "Percentile 10: 3.6\nPercentile 20: 3.8\nPercentile 40: 4.1\nPercentile 60: 4.3\nPercentile 80: 4.6\nPercentile 90: 4.9"
    fc = Forecaster([MockLLM(responder=respond)], Researcher(tools=[], use_news=False), cfg)
    qs = bt.load_fixtures(FIX / "resolved")
    out = tmp_path / "scores.jsonl"
    summ = bt.run_backtest(fc, qs, out, limit=50, printer=lambda s: None)
    assert summ["n_scored"] == 3 and summ["skipped"]["unresolved_or_annulled"] == 1
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    b = next(r for r in rows if r["question_id"] == 8101)
    assert b["baseline"] == pytest.approx(100 * math.log2(1.6), abs=1e-3)
    assert json.loads((tmp_path / "scores.jsonl.summary.json").read_text())["n_scored"] == 3
    # the knowledge-cutoff guard drops questions opened before the given date
    summ = bt.run_backtest(fc, qs, tmp_path / "s2.jsonl", min_open_date="2027-01-01", printer=lambda s: None)
    assert summ["n_scored"] == 0 and summ["skipped"]["before_min_open_date"] == 3


class FakeAPI:
    def __init__(self, posts):
        self.posts, self.calls = posts, []

    def list_posts(self, project, statuses="open", page_size=50, max_pages=20, extra=None):
        self.calls.append((project, statuses))
        return self.posts


def test_fetch_resolved_keeps_only_usable(tmp_path):
    posts = [json.loads(p.read_text()) for p in sorted((FIX / "resolved").glob("*.json"))]
    hidden = json.loads(json.dumps(posts[0]))
    hidden["id"] = hidden["question"]["post_id"] = 9199
    hidden["question"]["id"] = 8199
    hidden["question"]["resolution"] = None                # Restricted tier hides the resolution
    api = FakeAPI(posts + [hidden])
    summ = bt.fetch_resolved(api, 32979, tmp_path / "fx", limit=10)
    assert api.calls == [(32979, "resolved")]
    assert summ["kept"] == 4 and summ["resolution_null"] == 1
    idx = [json.loads(l) for l in (tmp_path / "fx" / "index.jsonl").read_text().splitlines()]
    assert {r["question_id"] for r in idx} == {8101, 8102, 8103, 8104}
    assert len(bt.load_fixtures(tmp_path / "fx")) == 4
