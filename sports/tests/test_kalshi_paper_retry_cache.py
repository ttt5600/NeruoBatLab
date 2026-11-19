"""ops_robustness_1118: kalshi_paper `retry` pass, LOST alert, and the om_fetch in-window cache. No network."""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import om_fetch  # noqa: E402
from test_kalshi_paper_resilience import NOW, _rows, env, kp  # noqa: E402,F401  (env = shared fixture)


@pytest.fixture
def lenv(env, tmp_path, monkeypatch):
    monkeypatch.setattr(kp, "LOST", tmp_path / "kalshi_paper_lost.jsonl")
    kp.FAIL_EV.clear()  # `retry` runs as a new process
    return env


def _clock(step):
    it = iter(range(0, 10 ** 6, step))
    return lambda: next(it)


def test_retry_pass_recovers_city_after_tick_failure(lenv, monkeypatch):
    """11-17: DEN failed all 3 tries in both window ticks. `retry` (run.sh, after exec) re-tries over minutes."""
    lenv["meteo_fail"] = {"*"}
    assert set(kp.cmd_pick(due_only=True)) == {"KXHIGHAUS", "KXHIGHCHI"}
    kp.FAIL_EV.clear()
    passes = {"n": 0}

    def sleep(s):
        passes["n"] += 1
        if passes["n"] == 2:  # open-meteo back before the 2nd retry pass
            lenv["meteo_fail"] = set()
    left = kp.cmd_retry(kp.pending_due(), sleep=sleep, clock=_clock(1))
    assert left == [] and passes["n"] == 2
    assert {r["city"] for r in _rows(kp.LOG)} == {"KXHIGHAUS", "KXHIGHCHI"}
    assert not kp.LOST.exists()


def test_retry_noop_when_nothing_pending(lenv):
    assert kp.cmd_pick(due_only=True) == []
    assert kp.pending_due() == []
    assert kp.cmd_retry(kp.pending_due(), sleep=lambda s: (_ for _ in ()).throw(AssertionError), clock=_clock(1)) == []


def test_lost_city_alerts_once_when_last_window_hour(lenv, monkeypatch):
    lenv["meteo_fail"] = {"*"}
    sent = []
    monkeypatch.setattr(om_fetch, "notify", lambda t, m: sent.append(m))
    late = datetime(2026, 11, 14, 5, 30, tzinfo=timezone.utc)  # 23:30 CT: last window hour

    class Late(datetime):
        @classmethod
        def now(cls, tz=None):
            return late
    monkeypatch.setattr(kp, "datetime", Late)
    left = kp.cmd_retry(["KXHIGHAUS", "KXHIGHCHI"], sleep=lambda s: None, clock=_clock(400))
    assert set(left) == {"KXHIGHAUS", "KXHIGHCHI"}
    evs = {json.loads(l)["event"] for l in kp.LOST.read_text().splitlines()}
    assert evs == {"KXHIGHAUS-26NOV14", "KXHIGHCHI-26NOV14"}
    assert len(sent) == 2
    kp.cmd_retry(["KXHIGHAUS"], sleep=lambda s: None, clock=_clock(400))
    assert len(sent) == 2  # once per city-night


def test_lost_not_raised_in_first_window_hour(lenv, monkeypatch):
    lenv["meteo_fail"] = {"*"}
    monkeypatch.setattr(om_fetch, "notify", lambda t, m: (_ for _ in ()).throw(AssertionError("alerted")))
    kp.cmd_retry(["KXHIGHAUS"], sleep=lambda s: None, clock=_clock(400))  # NOW = 22:03 CT, 23:xx tick still to come
    assert not kp.LOST.exists()


class _R:
    def __init__(self, obj):
        self.obj = obj

    def json(self):
        return self.obj


def _boom(*a, **k):
    raise requests.exceptions.ReadTimeout("Read timed out. (read timeout=10)")


def test_om_cache_used_only_inside_window_and_same_params(tmp_path, monkeypatch):
    monkeypatch.setattr(om_fetch, "CACHE_DIR", tmp_path)
    hourly = {"time": ["2026-11-14T00:00"], "temperature_2m_gfs_seamless": [70.0]}
    args = (30.0, -97.0, "America/Chicago", kp.MODELS)
    assert om_fetch.fetch_hourly(*args, get=lambda *a, **k: _R({"hourly": hourly})) == (hourly, "live")
    now = datetime.now(timezone.utc)
    h, s = om_fetch.fetch_hourly(*args, since=now - timedelta(minutes=30), get=_boom)
    assert h == hourly and s.startswith("cache@")
    with pytest.raises(requests.exceptions.ReadTimeout):  # fetched before the window opened -> not used
        om_fetch.fetch_hourly(*args, since=now + timedelta(minutes=1), get=_boom)
    with pytest.raises(requests.exceptions.ReadTimeout):  # different params (other city) -> no cache
        om_fetch.fetch_hourly(31.0, -97.0, "America/Chicago", kp.MODELS, since=now - timedelta(hours=1), get=_boom)
    with pytest.raises(requests.exceptions.ReadTimeout):  # no window (validate / dry) -> never cache
        om_fetch.fetch_hourly(*args, get=_boom)
    assert not list(tmp_path.glob("*.tmp"))


def test_live_forecasts_inputs_unchanged(monkeypatch, tmp_path):
    """The frozen rule's input: same endpoint, same 5 models, same params as before this change."""
    monkeypatch.setattr(om_fetch, "CACHE_DIR", tmp_path)
    seen = {}

    def get(url, headers=None, timeout=None, params=None):
        seen.update(url=url, params=params, timeout=timeout)
        return _R({"hourly": {"time": [f"2026-11-15T{h:02d}:00" for h in range(24)],
                              **{f"temperature_2m_{m}": [60.0 + h for h in range(24)] for m in kp.MODELS}}})
    monkeypatch.setattr(om_fetch.requests, "get", get)
    src = []
    out = kp.live_forecasts(30.0, -97.0, "America/Chicago", date(2026, 11, 15), src=src)
    assert seen["url"] == "https://api.open-meteo.com/v1/forecast"
    assert seen["params"] == {"latitude": 30.0, "longitude": -97.0, "models": ",".join(kp.MODELS),
                              "hourly": "temperature_2m", "temperature_unit": "fahrenheit",
                              "timezone": "America/Chicago", "forecast_days": 3}
    assert seen["timeout"] == (30, 60)
    assert out == {m: 78.0 for m in kp.MODELS} and src == ["live"]
