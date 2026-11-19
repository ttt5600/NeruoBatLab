"""ops_robustness_1118: forecastex_paper.cmd_tick skips a station missing from ForecastEx's public category tree
(UHBKF since ~11-17) with one logged reason per station-day, and the other stations still run. No network."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import forecastex_paper as fp  # noqa: E402

NOW = datetime(2026, 11, 18, 4, 31, tzinfo=timezone.utc)  # 22:31 CT, 23:31 ET


class Frozen(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


class FakeFX:
    def __init__(self, env=None):
        pass

    def underlying_conids(self):
        return {"UHNYC": 1, "UHMDW": 2}  # BKF absent

    def strikes(self, code, max_age=30.0):
        if code == "BKF":
            raise AssertionError("BKF must be skipped before any strikes() call")
        return {}


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k in ("DATA", "LOG", "STATE", "LOCK"):
        monkeypatch.setattr(fp, k, tmp_path / getattr(fp, k).name)
    monkeypatch.setattr(fp, "STATIONS", ["BKF", "NYC", "MDW"])
    monkeypatch.setattr(fp, "ForecastEx", FakeFX)
    monkeypatch.setattr(fp, "datetime", Frozen)
    monkeypatch.setattr(fp, "station_meta", lambda st, c: (0.0, 0.0, "America/Chicago"))
    monkeypatch.setattr(fp, "q008_tick", lambda *a, **k: [])
    monkeypatch.setattr(fp, "settle", lambda now=None: 0)
    evald = []
    monkeypatch.setattr(fp, "evaluate", lambda fxc, st, code, day, now: (evald.append(code) or ([], f"{code} {day}: 3 strikes, 0 rule picks, 5 models")))
    return evald


def test_missing_station_skipped_with_reason_once_per_day(env, capsys):
    assert fp.cmd_tick() == 0
    out = capsys.readouterr().out
    assert "BKF 2026-11-18: SKIP UHBKF is not listed" in out and "Error" not in out and "ERROR" not in out
    assert env == ["NYC", "MDW"]                                   # other stations still evaluated
    st = json.loads(fp.STATE.read_text())
    assert "BKF:2026-11-18" not in st["done"] and st["skip_logged"] == ["BKF:2026-11-18"]
    assert fp.cmd_tick() == 0
    assert "BKF" not in capsys.readouterr().out                    # logged once per station-day


def test_tree_fetch_failure_falls_back_to_per_station_path(env, monkeypatch, capsys):
    def boom(self):
        raise RuntimeError("tree down")
    monkeypatch.setattr(FakeFX, "underlying_conids", boom)
    assert fp.cmd_tick() == 0
    out = capsys.readouterr().out
    assert "category tree ERROR" in out
    assert env == ["BKF", "NYC", "MDW"]
