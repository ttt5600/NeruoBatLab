"""ops_robustness_1118: e4p_paper open-meteo fetch (same inputs, in-window cache) and the MISSED alert. No network."""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e4p_paper as ep  # noqa: E402
import om_fetch  # noqa: E402


class _R:
    def __init__(self, obj):
        self.obj = obj

    def json(self):
        return self.obj


HOURLY = {"time": [f"2026-11-19T{h:02d}:00" for h in range(24)],
          **{f"temperature_2m_{m}": [50.0 + h for h in range(24)] for m in ep.MODELS}}


def test_e4p_uses_same_params_and_falls_back_to_in_window_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(om_fetch, "CACHE_DIR", tmp_path)
    seen = {}

    def ok(url, headers=None, timeout=None, params=None):
        seen.update(url=url, params=params)
        return _R({"hourly": HOURLY})
    monkeypatch.setattr(om_fetch.requests, "get", ok)
    st, tz, lat, lon = ep.CITIES["KXHIGHNY"]
    src = []
    a = ep.live_forecasts(lat, lon, tz, date(2026, 11, 19), src=src)   # e.g. the E4 job's fetch at 22:03
    assert seen["url"] == ep.OM and seen["params"]["models"] == ",".join(ep.MODELS) and src == ["live"]

    def boom(*a, **k):
        raise requests.exceptions.ReadTimeout("Read timed out. (read timeout=15)")
    monkeypatch.setattr(om_fetch.requests, "get", boom)
    since = datetime.now(timezone.utc) - timedelta(minutes=10)
    src = []
    b = ep.live_forecasts(lat, lon, tz, date(2026, 11, 19), since=since, src=src)
    assert a == b and src[0].startswith("cache@")
    with pytest.raises(requests.exceptions.ReadTimeout):              # cache older than this window -> not used
        ep.live_forecasts(lat, lon, tz, date(2026, 11, 19), since=datetime.now(timezone.utc) + timedelta(seconds=5))


def test_missed_city_day_alerts(tmp_path, monkeypatch):
    monkeypatch.setattr(ep, "LOG", tmp_path / "e4p.jsonl")
    monkeypatch.setattr(ep, "CITIES", {"KXHIGHNY": ep.CITIES["KXHIGHNY"]})
    t = datetime(2026, 11, 19, 3, 55, tzinfo=timezone.utc)   # 22:55 ET: past the 22:00-22:50 window
    monkeypatch.setattr(ep, "now_utc", lambda: t)
    sent = []
    monkeypatch.setattr(om_fetch, "notify", lambda title, msg: sent.append((title, msg)))
    assert ep.cmd_tick(None) == 0
    rows = [json.loads(l) for l in ep.LOG.read_text().splitlines()]
    assert [r["status"] for r in rows] == ["missed"]
    assert len(sent) == 1 and "KXHIGHNY" in sent[0][1]
    assert ep.cmd_tick(None) == 0 and len(sent) == 1               # once per city-day
