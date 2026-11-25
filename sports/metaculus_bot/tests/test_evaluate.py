"""The pre-registered go/kill rule, zone by zone, and the score collector against a fake listing."""
import copy
import json
from datetime import datetime, timezone

from conftest import FIX, NOW
from mbot import evaluate as ev
from mbot.ledger import Ledger


def test_rule_zones():
    ok_spend = {"2026-12": 30.0}
    assert ev.decide(10, 10, [5.0] * 10, ok_spend, 50)["verdict"] == "PENDING"
    assert ev.decide(60, 58, [6.0] * 45, ok_spend, 50)["verdict"] == "GO"            # 96.7% coverage, mean +6
    assert ev.decide(60, 58, [2.0] * 45, ok_spend, 50)["verdict"] == "HOLD"           # mean in [0, 5)
    assert ev.decide(60, 56, [6.0] * 45, ok_spend, 50)["verdict"] == "HOLD"           # coverage 93.3%
    assert ev.decide(60, 58, [2.0] * 45, ok_spend, 50, previous_verdict="HOLD")["verdict"] == "KILL"
    assert ev.decide(60, 58, [-1.0] * 45, ok_spend, 50)["verdict"] == "KILL"          # mean < 0
    assert ev.decide(60, 50, [9.0] * 45, ok_spend, 50)["verdict"] == "KILL"           # coverage 83%
    assert ev.decide(60, 58, [9.0] * 45, {"2026-12": 50.5}, 50)["verdict"] == "KILL"  # over the cap
    early = ev.decide(5, 2, [9.0], ok_spend, 50)                                      # too few to judge coverage
    assert early["verdict"] == "PENDING"


def test_ci_reading():
    r = ev.decide(60, 60, [10.0, -9.0] * 25, {}, 50)
    assert r["mean_spot_peer"] == 0.5 and r["ci_reading"] == "not distinguishable from 0"


class FakeAPI:
    def __init__(self, posts):
        self.posts = posts

    def list_posts(self, t, statuses="open", page_size=50, max_pages=20, extra=None):
        assert set(statuses) == {"closed", "resolved"}
        return self.posts


def test_collect_counts_coverage_and_scores(tmp_path):
    base = json.loads((FIX / "resolved" / "9101.json").read_text())
    posts = []
    for i, (forecast, score, res) in enumerate([(True, 12.5, "yes"), (True, -3.0, "no"), (False, None, "yes"),
                                                (True, 40.0, "annulled")]):
        p = copy.deepcopy(base)
        p["id"] = p["question"]["post_id"] = 9500 + i
        p["question"]["id"] = 8500 + i
        p["question"]["resolution"] = res
        if forecast:
            p["question"]["my_forecasts"] = {"latest": {"forecast_values": [0.4, 0.6]},
                                             "score_data": {"spot_peer_score": score}}
        posts.append(p)
    old = copy.deepcopy(base)                     # opened before the bot's first live run: not eligible
    old["id"], old["question"]["id"], old["question"]["open_time"] = 9600, 8600, "2026-01-01T00:00:00Z"
    posts.append(old)
    led = Ledger(tmp_path / "l.jsonl", now_fn=lambda: NOW)
    led.append({"kind": "run_start", "dry_run": True})
    led.append({"kind": "run_start", "dry_run": False})
    since = ev.first_live_run(led)
    since = since.replace(day=1)                  # fixtures open 2026-11-30 12:00; NOW is 13:00 the same day
    got = ev.collect(FakeAPI(posts), led, ["minibench"], since, now=datetime(2026, 12, 1, tzinfo=timezone.utc))
    assert got["eligible"] == 4 and got["forecast"] == 3 and sorted(got["scores"]) == [-3.0, 12.5]
