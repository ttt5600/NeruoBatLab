"""kalshi_paper.cmd_pick under network failures (e4_ops_diag, 2026-11-15).

Replays the failure that lost every pick for the nights logged 11-13 and 11-14 UTC: one open-meteo ReadTimeout
raised out of cmd_pick and aborted the whole tick, for every city, every hour of the decision window.
No network: requests.get and wx helpers are monkeypatched.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kalshi_paper as kp  # noqa: E402

NOW = datetime(2026, 11, 14, 4, 3, 5, tzinfo=timezone.utc)  # 22:03 CT -> CT cities due


class FrozenDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


class Resp:
    def __init__(self, obj):
        self.obj = obj

    def json(self):
        return self.obj


def _market(ev):
    # bracket 80-81 F, forecasts ~70 -> rejected by > 3F -> a pick
    return {"ticker": f"{ev}-B80.5", "strike_type": "between", "floor_strike": 80, "cap_strike": 81,
            "yes_bid_dollars": "0.03", "yes_ask_dollars": "0.04"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(kp, "LOG", tmp_path / "kalshi_paper.jsonl")
    monkeypatch.setattr(kp, "DONE", tmp_path / "kalshi_paper_done.json")
    monkeypatch.setattr(kp, "datetime", FrozenDT)
    monkeypatch.setattr(kp.wx, "station_coords", lambda st: (30.0, -97.0))
    monkeypatch.setattr(kp, "PAPER_CITIES", {"KXHIGHAUS": ("KAUS", "America/Chicago"),
                                             "KXHIGHCHI": ("KMDW", "America/Chicago")})
    state = {"meteo_fail": set(), "calls": 0}

    def fake_get(url, headers=None, timeout=None, params=None):
        state["calls"] += 1
        if url.endswith("/markets"):
            return Resp({"markets": [_market(params["event_ticker"])]})
        if url.endswith("/orderbook"):
            return Resp({"orderbook_fp": {"yes_dollars": [["0.03", "50"]]}})
        raise AssertionError(url)

    def fake_fc(lat, lon, tz, day, **kw):
        if state["meteo_fail"]:
            raise requests.exceptions.ReadTimeout("HTTPSConnectionPool(host='api.open-meteo.com'): Read timed out")
        return {m: 70.0 for m in kp.MODELS}

    monkeypatch.setattr(kp.requests, "get", fake_get)
    monkeypatch.setattr(kp, "live_forecasts", fake_fc)
    monkeypatch.setattr(kp, "_retry", lambda fn, tries=3, wait=5.0, sleep=None:
                        _real_retry(fn, tries, wait, sleep=lambda s: None))
    return state


_real_retry = kp._retry


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def test_meteo_timeout_does_not_crash_and_city_stays_retryable(env):
    env["meteo_fail"] = {"*"}
    failed = kp.cmd_pick(due_only=True)               # before the fix this raised ReadTimeout
    assert set(failed) == {"KXHIGHAUS", "KXHIGHCHI"}
    assert _rows(kp.LOG) == []
    assert not kp.DONE.exists() or json.loads(kp.DONE.read_text()) == []   # not marked done -> retried
    env["meteo_fail"] = set()                          # next hourly tick, open-meteo back
    assert kp.cmd_pick(due_only=True) == []
    assert {r["city"] for r in _rows(kp.LOG)} == {"KXHIGHAUS", "KXHIGHCHI"}
    assert {r["event"] for r in _rows(kp.LOG)} == {"KXHIGHAUS-26NOV14", "KXHIGHCHI-26NOV14"}


def test_one_city_failing_does_not_block_the_other(env, monkeypatch):
    calls = {"n": 0}

    def flaky(lat, lon, tz, day, **kw):  # first city always times out, second works
        calls["n"] += 1
        if calls["n"] <= 3:
            raise requests.exceptions.ReadTimeout("timed out")
        return {m: 70.0 for m in kp.MODELS}

    monkeypatch.setattr(kp, "live_forecasts", flaky)
    failed = kp.cmd_pick(due_only=True)
    assert failed == ["KXHIGHAUS"]
    assert [r["city"] for r in _rows(kp.LOG)] == ["KXHIGHCHI"]
    assert json.loads(kp.DONE.read_text()) == ["KXHIGHCHI-26NOV14"]


def test_retry_recovers_from_transient_timeout(env, monkeypatch):
    calls = {"n": 0}

    def once_then_ok(lat, lon, tz, day, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ConnectTimeout("handshake timed out")
        return {m: 70.0 for m in kp.MODELS}

    monkeypatch.setattr(kp, "live_forecasts", once_then_ok)
    assert kp.cmd_pick(due_only=True) == []
    assert len(_rows(kp.LOG)) == 2
