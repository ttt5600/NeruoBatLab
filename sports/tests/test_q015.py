"""Offline tests for Q015 (scripts/q015_approval_avg.py, scripts/q015_logger.py). No network."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import q015_approval_avg as Q  # noqa: E402
import q015_logger as L  # noqa: E402


@pytest.mark.parametrize("row,expect", [
    ({"yes_sub_title": "41.0 to 41.2"}, (410, 412)),
    ({"yes_sub_title": "40.5% to 40.7%"}, (405, 407)),
    ({"yes_sub_title": "44.7% or higher"}, (447, 10 ** 4)),
    ({"yes_sub_title": "37% or lower"}, (-1, 370)),
    ({"yes_sub_title": "43.2 or below"}, (-1, 432)),
    ({"yes_sub_title": "Above 39.4", "strike_type": "greater", "floor_strike": 39.4}, (395, 10 ** 4)),
    ({"yes_sub_title": "", "strike_type": "less", "floor_strike": float("nan"), "cap_strike": 38.1}, (-1, 380)),
    ({"yes_sub_title": "", "strike_type": "between", "floor_strike": 38.4, "cap_strike": 38.4}, (384, 384)),
])
def test_yes_set(row, expect):
    assert Q.yes_set(row) == expect


def _toy_hist():
    d = pd.date_range("2017-01-27", "2021-09-30", freq="D")
    v = 42 + 0.1 * ((pd.Series(range(len(d))) * 7919) % 5 - 2).cumsum().clip(-30, 30).values
    return pd.DataFrame({"date": d, "approve": v})


def test_pmf_and_partition():
    pm = Q.pmf(_toy_hist(), 2)
    assert abs(sum(pm.values()) - 1) < 1e-9
    ladder = [(-1, 399), (400, 402), (403, 405), (406, 10 ** 4)]  # partitions the line
    assert abs(sum(Q.p_yes(40.3, lo, hi, pm) for lo, hi in ladder) - 1) < 1e-9


def test_decision_ts():
    c = 1_800_000_000
    assert Q.decision_ts(c, "close-24h") == c - 86400
    assert Q.decision_ts(c, "close-24h+6m") == c - 86400 + 360


def test_logger_tick_offline(tmp_path, monkeypatch):
    close = "2026-10-30T15:00:00Z"
    now = datetime(2026, 10, 30, 13, 10, tzinfo=timezone.utc)  # T4 tau = 13:00Z
    mk = [{"ticker": f"KXAPRPOTUS-26OCT30-{x}", "event_ticker": "KXAPRPOTUS-26OCT30", "close_time": close,
           "yes_sub_title": s, "strike_type": st, "floor_strike": f, "cap_strike": cp, "yes_bid_dollars": b,
           "yes_ask_dollars": a, "volume_fp": "10"}
          for x, s, st, f, cp, b, a in [("38.4", "38.3 to 38.5", "between", 38.3, 38.5, "0.67", "0.68"),
                                        ("38.1", "38.0 to 38.2", "between", 38.0, 38.2, "0.07", "0.08")]]

    def fake_fetch(url, params=None, cache=True, **kw):
        if "votehub" in url:
            return [{"id": "p1", "created_at": "2026-10-29", "pollster": "X", "answers": [{"choice": "Approve", "pct": 39}]}]
        if params and params.get("status") == "settled":
            return {"markets": [{"expiration_value": "38.4", "close_time": "2026-10-29T16:59:00Z"}]}
        if params and params.get("series_ticker") == "KXAPRPOTUS":
            return {"markets": mk}
        return {"markets": []}

    monkeypatch.setattr(L, "fetch", fake_fetch)
    monkeypatch.setattr(L, "OUT", tmp_path)
    _toy_hist().to_parquet(tmp_path / "rcp_history.parquet", index=False)
    monkeypatch.setattr(L.Q, "D", tmp_path)
    out = L.tick(now)
    assert out["snapshots"] == 2 and out["new_polls"] == 1 and out["picks_logged"] == 2
    out2 = L.tick(now)  # same window: no duplicate picks, no duplicate polls
    assert out2["picks_logged"] == 0 and out2["new_polls"] == 0
    early = L.tick(datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc))  # before tau -> nothing
    assert early["picks_logged"] == 0
