"""End to end: fixture questions -> forecaster (stub LLM) -> runner -> a fake Metaculus transport. No network."""
import copy
import json
import re

import pytest

from conftest import NOW
from mbot import cdf as C
from mbot.forecaster import Forecaster
from mbot.ledger import Ledger, SpendCapExceeded, SpendGuard
from mbot.llm import MockLLM
from mbot.metaculus_api import MetaculusAPI
from mbot.research import Researcher
from mbot.runner import Runner


class ScriptedLLM(MockLLM):
    """A test double that is NOT the user-facing mock provider, so the runner lets it 'submit' to the fake API."""
    provider = "scripted"


def scripted_answers(llm_out):
    """Per-format answers that differ by sample index, so the median is actually exercised."""
    def respond(prompt, i):
        k = i % 3
        if "FORMAT: BINARY" in prompt:
            return f"reasoning\nProbability: {[60, 70, 99.5][k]}%"
        if "FORMAT: MULTIPLE_CHOICE" in prompt:
            return llm_out["mc_ok"] if k != 2 else llm_out["mc_prefixed"]
        if "measles" in prompt:
            return llm_out["numeric_commas"]
        if "executive orders" in prompt:
            return llm_out["discrete_ok"]
        if "CEO" in prompt:
            return llm_out["date_ok"]
        return llm_out["numeric_ok"]
    return respond


class FakeMetaculus:
    """Minimal stand-in for www.metaculus.com/api: listing, post detail, forecast + comment POSTs."""

    def __init__(self, posts, fail_forecast_post=None, land_despite_error=False):
        self.posts = {p["id"]: copy.deepcopy(p) for p in posts}
        self.forecasts, self.comments, self.requests = [], [], []
        self.fail_forecast_post = fail_forecast_post
        self.land_despite_error = land_despite_error

    def _questions(self, post):
        if post.get("question"):
            return [post["question"]]
        return (post.get("group_of_questions") or {}).get("questions") or []

    def __call__(self, method, url, params=None, json=None):
        self.requests.append((method, url))
        if method == "GET" and url.endswith("/posts/"):
            return 200, {"next": None, "results": [copy.deepcopy(p) for p in self.posts.values()]}
        m = re.search(r"/posts/(\d+)/$", url)
        if method == "GET" and m:
            return 200, copy.deepcopy(self.posts[int(m.group(1))])
        if method == "POST" and url.endswith("/questions/forecast/"):
            qid = json[0]["question"]
            if self.fail_forecast_post == qid:
                if self.land_despite_error:
                    self._mark(qid, json[0])
                return 502, "bad gateway"
            self._mark(qid, json[0])
            return 201, ""
        if method == "POST" and url.endswith("/comments/create/"):
            self.comments.append(json)
            return 201, {"id": len(self.comments)}
        return 404, "nope"

    def _mark(self, qid, body):
        self.forecasts.append(body)
        for p in self.posts.values():
            for q in self._questions(p):
                if q["id"] == qid:
                    q["my_forecasts"] = {"latest": {"forecast_values": [0.5, 0.5]}, "history": []}


def _build(cfg, llm, ledger=None):
    ledger = ledger or Ledger(cfg.ledger_path, now_fn=lambda: NOW)
    guard = SpendGuard(ledger, cfg.monthly_cap_usd, now_fn=lambda: NOW)
    return Forecaster([llm], Researcher(tools=[], use_news=False), cfg, guard=guard, ledger=ledger), ledger


# ---------------------------------------------------------------------------------------------- forecaster
def test_forecaster_every_type(cfg, questions, llm_out):
    fc, _ = _build(cfg, ScriptedLLM(responder=scripted_answers(llm_out)))
    for qid in (8001, 8002, 8003, 8004, 8005, 8006):
        fr = fc.forecast(questions[qid], run_id="t")
        assert fr.ok, (qid, fr.error)
        assert fr.comment and "Samples" in fr.comment and len(fr.samples) == 3
        if fr.family == "continuous":
            assert C.validate_cdf(fr.final, C.Spec.from_question(questions[qid])) == []
            assert fr.payload["continuous_cdf"] == fr.final
    b = fc.forecast(questions[8001])
    assert b.final == 0.70                                   # median of 0.60, 0.70, 0.995
    b_hi = Forecaster([ScriptedLLM(responder=["Probability: 99.9%"])], Researcher(use_news=False), cfg).forecast(questions[8001])
    assert b_hi.final == 0.98                                # clipped
    mc = fc.forecast(questions[8002])
    assert sum(mc.final.values()) == pytest.approx(1.0) and min(mc.final.values()) >= 0.02 - 1e-12


def test_forecaster_drops_bad_samples_and_fails_cleanly(cfg, questions, llm_out):
    llm = ScriptedLLM(responder=[llm_out["binary_bad"], "Probability: 40%", llm_out["binary_bad"]])
    fc, _ = _build(cfg, llm)
    fr = fc.forecast(questions[8001])
    assert fr.ok and fr.final == 0.40 and sum(s.ok for s in fr.samples) == 1
    cfg.min_valid_samples = 2
    fr = _build(cfg, ScriptedLLM(responder=[llm_out["binary_bad"], "Probability: 40%", llm_out["binary_bad"]]))[0].forecast(questions[8001])
    assert not fr.ok and "valid samples" in fr.error
    fr = _build(cfg, ScriptedLLM(fail_on={0, 1, 2}))[0].forecast(questions[8001])
    assert not fr.ok


def test_spend_cap_stops_before_calling(cfg, questions):
    cfg.monthly_cap_usd = 1.0
    llm = ScriptedLLM(responder=["Probability: 40%"], fake_cost_usd=0.4)
    fc, ledger = _build(cfg, llm)
    fr = fc.forecast(questions[8001])                  # 3 x $0.40 worst case > $1: only 2 calls may start
    assert fr.ok and len(llm.calls) == 2 and "spend cap" in fr.warnings[0]
    assert ledger.spent_month(NOW) == pytest.approx(0.8)
    with pytest.raises(SpendCapExceeded):              # next question: $0.80 + $0.40 > $1 -> no call at all
        fc.forecast(questions[8091])
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------------------------- runner
def _runner(cfg, posts, llm, dry_run, **fake_kw):
    fake = FakeMetaculus(posts, **fake_kw)
    api = MetaculusAPI("tok", http=fake, min_interval_s=0, sleep=lambda s: None)
    fc, ledger = _build(cfg, llm)
    return Runner(cfg, fc, ledger, api, dry_run=dry_run, now_fn=lambda: NOW, printer=lambda s: None), fake, ledger


def test_dry_run_posts_nothing(cfg, open_posts, llm_out):
    r, fake, ledger = _runner(cfg, open_posts, ScriptedLLM(responder=scripted_answers(llm_out)), dry_run=True)
    s = r.run(tournaments=["minibench"])
    assert s["forecast_ok"] == 8 and s["submitted"] == 0
    assert not [m for m, _ in fake.requests if m == "POST"]
    assert ledger.submitted_question_ids() == set()
    reasons = {row.get("question_id"): row.get("reason") for row in ledger.rows() if row["kind"] == "skip"}
    assert reasons[8007] == "already_forecast_on_metaculus" and reasons[8008].startswith("closing_in_5.0")


def test_mock_provider_cannot_submit(cfg, open_posts):
    with pytest.raises(ValueError, match="mock"):
        _runner(cfg, open_posts, MockLLM(), dry_run=False)


def test_live_run_submits_once_never_twice(cfg, open_posts, llm_out):
    llm = ScriptedLLM(responder=scripted_answers(llm_out))
    r, fake, ledger = _runner(cfg, open_posts, llm, dry_run=False)
    s1 = r.run(tournaments=["minibench"])
    assert s1["submitted"] == 8 and len(fake.forecasts) == 8 and len(fake.comments) == 8
    assert all(c["is_private"] and c["included_forecast"] for c in fake.comments)
    n_calls = len(llm.calls)
    # second pass, same ledger and same fake server: nothing is forecast or posted again
    s2 = r.run(tournaments=["minibench"])
    assert s2["candidates"] == 0 and len(fake.forecasts) == 8 and len(llm.calls) == n_calls
    # even with a wiped ledger, Metaculus' own my_forecasts record blocks a resubmit
    r.ledger = Ledger(cfg.state_dir / "other.jsonl", now_fn=lambda: NOW)
    s3 = r.run(tournaments=["minibench"])
    assert s3["candidates"] == 0 and len(fake.forecasts) == 8


def test_recheck_blocks_a_race(cfg, open_posts, llm_out):
    """Another process forecasts the question between listing and posting: the pre-POST re-read catches it."""
    llm = ScriptedLLM(responder=scripted_answers(llm_out))
    r, fake, ledger = _runner(cfg, [p for p in open_posts if p["id"] == 9001], llm, dry_run=False)
    orig = r.fc.forecast

    def forecast_then_race(q, **kw):
        out = orig(q, **kw)
        fake._mark(q.question_id, {"question": q.question_id})          # someone else got there first
        return out
    r.fc.forecast = forecast_then_race
    s = r.run(tournaments=["minibench"])
    assert s["outcomes"] == {"8001": "already"} and len(fake.forecasts) == 1   # only the racer's


def test_ambiguous_post_failure_is_not_retried(cfg, open_posts, llm_out):
    one = [p for p in open_posts if p["id"] == 9001]
    # (a) 502 and the forecast did NOT land: logged as submit_error, no submitted row, next run may try again
    r, fake, ledger = _runner(cfg, one, ScriptedLLM(responder=scripted_answers(llm_out)), dry_run=False,
                              fail_forecast_post=8001)
    s = r.run(tournaments=["minibench"])
    assert s["outcomes"] == {"8001": "submit_error"} and ledger.submitted_question_ids() == set()
    assert sum(1 for m, u in fake.requests if m == "POST" and u.endswith("/questions/forecast/")) == 1
    # (b) 502 but it DID land: recorded as submitted (the re-read sees it), comment still posted
    r, fake, ledger = _runner(cfg, one, ScriptedLLM(responder=scripted_answers(llm_out)), dry_run=False,
                              fail_forecast_post=8001, land_despite_error=True)
    s = r.run(tournaments=["minibench"])
    assert s["outcomes"] == {"8001": "submitted"} and ledger.submitted_question_ids() == {8001}
    assert len(fake.comments) == 1


def test_spend_cap_stops_the_run(cfg, open_posts):
    cfg.monthly_cap_usd = 0.5
    cfg.n_samples = 1
    llm = ScriptedLLM(responder=lambda p, i: "Probability: 40%" if "BINARY" in p else "x", fake_cost_usd=0.2)
    r, fake, ledger = _runner(cfg, [p for p in open_posts if p["id"] in (9001, 9009)], llm, dry_run=False)
    s = r.run(tournaments=["minibench"])
    assert s["spend_cap_hit"] and s["submitted"] == 2 and len(llm.calls) == 2
    assert ledger.spent_month(NOW) == pytest.approx(0.4)


def test_ledger_rows_have_required_fields(cfg, open_posts, llm_out):
    r, fake, ledger = _runner(cfg, open_posts, ScriptedLLM(responder=scripted_answers(llm_out)), dry_run=True)
    r.run(tournaments=["minibench"])
    rows = [x for x in ledger.rows() if x["kind"] == "forecast"]
    assert rows
    for x in rows:
        for k in ("question_id", "type", "inputs", "samples", "final", "cost_usd", "ts"):
            assert k in x
    json.dumps(rows)


def test_cli_offline_dry_run(tmp_path, monkeypatch, capsys):
    from mbot.__main__ import main
    rc = main(["run", "--dry-run", "--models", "mock:m", "--n-samples", "2", "--no-tools", "--state-dir", str(tmp_path),
               "--questions-file", "tests/fixtures/open_posts.json"])
    out = capsys.readouterr().out
    assert rc == 0 and '"forecast_ok"' in out
    assert main(["run", "--models", "mock:m", "--state-dir", str(tmp_path)]) == 2       # no token -> refuses
