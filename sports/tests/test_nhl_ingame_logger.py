"""Offline tests for the NHL in-game forward logger, its edgelab loader and rule T1. No network."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

SPORTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPORTS / "scripts"))
sys.path.insert(0, str(SPORTS))
import nhl_ingame_logger as L  # noqa: E402
from edgelab.loaders import nhl_frame_from_log  # noqa: E402
from edgelab.rules.nhl_ingame import t1_late_taker_edge3  # noqa: E402
from edgelab import schema  # noqa: E402

T0 = datetime(2026, 11, 9, 1, 0, tzinfo=timezone.utc)


def game(state="LIVE", period=3, sec=600, inter=False, hs=2, as_=1, ptype="REG"):
    return {"id": 2026020300, "gameDate": "2026-11-08", "startTimeUTC": "2026-11-09T00:00:00Z", "gameState": state,
            "period": period, "periodDescriptor": {"number": period, "periodType": ptype},
            "clock": {"secondsRemaining": sec, "inIntermission": inter, "running": True},
            "homeTeam": {"abbrev": "LAK", "score": hs}, "awayTeam": {"abbrev": "TBL", "score": as_}}


def market(tk, b, a):
    return {"ticker": tk, "yes_bid_dollars": f"{b:.4f}", "yes_ask_dollars": f"{a:.4f}"}


@pytest.fixture
def tmp_out(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "OUT", tmp_path / "p.jsonl")
    monkeypatch.setattr(L, "RESULTS", tmp_path / "r.jsonl")
    monkeypatch.setattr(L, "STATE", tmp_path / "s.json")
    return tmp_path


def fake(games, markets, settled=None, fail=None):
    calls = []

    def get(url, params=None):
        calls.append(url)
        if fail and fail in url:
            raise ConnectionError("boom")
        if url == L.NHL_NOW:
            return {"currentDate": "2026-11-08", "games": games}
        if url.endswith("/markets"):
            return {"markets": markets}
        if "/candlesticks" in url:
            return {"candlesticks": [{"end_period_ts": 1794182340, "yes_bid": {"close_dollars": "0.55"},
                                      "yes_ask": {"close_dollars": "0.57"}}]}
        if "/markets/" in url:
            tk = url.rsplit("/", 1)[1]
            return {"market": {"status": "finalized", "result": (settled or {}).get(tk, "")}}
        raise AssertionError(url)
    get.calls = calls
    return get


def test_event_ticker_uses_kalshi_codes():
    assert L.event_ticker(game()) == "KXNHLGAME-26NOV08TBLA"


def test_minutes_left_intermission_and_ot():
    assert L.minutes_left(game(period=3, sec=600)) == 10.0
    assert L.minutes_left(game(period=2, sec=900, inter=True)) == 20.0   # intermission clock is NOT period time
    assert L.minutes_left(game(period=4, ptype="OT")) is None
    assert L.minutes_left(game(period=1, sec=1200)) == 60.0


def test_idle_no_games_makes_one_call_then_none(tmp_out):
    g = fake([], [])
    assert L.tick(get=g, now=T0) == 0
    assert g.calls == [L.NHL_NOW]
    g2 = fake([], [])
    assert L.tick(get=g2, now=T0 + timedelta(minutes=1)) == 0
    assert g2.calls == []                                     # cached schedule: idle exit, no network
    assert not (tmp_out / "p.jsonl").exists()


def test_api_errors_exit_1_without_crashing(tmp_out):
    assert L.tick(get=fake([game()], [], fail="nhle"), now=T0) == 1
    rc = L.tick(get=fake([game()], [], fail="/markets"), now=T0)
    assert rc == 1 and not (tmp_out / "p.jsonl").exists()


def test_live_tick_row_and_settlement(tmp_out):
    mk = [market("KXNHLGAME-26NOV08TBLA-LA", 0.80, 0.82), market("KXNHLGAME-26NOV08TBLA-TB", 0.18, 0.20)]
    assert L.tick(get=fake([game()], mk), now=T0, clock=lambda: T0) == 0
    r = json.loads((tmp_out / "p.jsonl").read_text().splitlines()[0])
    assert r["kind"] == "tick" and r["minutes_left"] == 10.0 and r["margin"] == 1
    assert r["anchor_src"] == "candle" and r["p_pre"] == pytest.approx(0.56)
    assert r["as_of"]["anchor"] <= r["as_of"]["kalshi"] <= r["decided_at"]
    assert 0.5 < r["p_model"] < 1
    # game ends -> final row, results fetched
    g = fake([game(state="OFF")], mk, settled={"KXNHLGAME-26NOV08TBLA-LA": "yes", "KXNHLGAME-26NOV08TBLA-TB": "no"})
    assert L.tick(get=g, now=T0 + timedelta(hours=1)) == 0
    kinds = [json.loads(x)["kind"] for x in (tmp_out / "p.jsonl").read_text().splitlines()]
    assert kinds == ["tick", "final"]
    res = [json.loads(x) for x in (tmp_out / "r.jsonl").read_text().splitlines()]
    assert {x["ticker"]: x["result"] for x in res} == {"KXNHLGAME-26NOV08TBLA-LA": "yes", "KXNHLGAME-26NOV08TBLA-TB": "no"}


def _row(t, hs, as_, hb, ha, p, sec, period=3):
    ts = (T0 + timedelta(seconds=t)).isoformat()
    return {"kind": "tick", "decided_at": ts, "as_of": {"nhl": ts, "kalshi": ts, "anchor": "2026-11-08T23:59:00+00:00"},
            "gid": "g1", "game_date": "2026-11-08", "home_ticker": "H", "away_ticker": "A", "period": period,
            "period_type": "REG", "intermission": False, "minutes_left": sec / 60, "home_score": hs, "away_score": as_,
            "margin": hs - as_, "p_model": p, "home_bid": hb, "home_ask": ha, "away_bid": round(1 - ha - 0.01, 2),
            "away_ask": round(1 - hb, 2)}


def test_rule_cooldown_one_per_side_and_pit():
    rows = [_row(0, 1, 1, 0.48, 0.50, 0.50, 840),
            _row(60, 2, 1, 0.70, 0.72, 0.85, 800),      # goal seen -> cooldown (and a jump)
            _row(120, 2, 1, 0.70, 0.72, 0.85, 760),     # still < 180 s after the goal
            _row(240, 2, 1, 0.74, 0.76, 0.86, 700),     # cooldown over, but mid jumped vs t=0 (<= 300 s back)
            _row(300, 2, 1, 0.74, 0.76, 0.87, 660),     # first eligible: home YES edge ~ .097
            _row(360, 2, 1, 0.74, 0.76, 0.88, 620),     # same backed team -> capped
            _row(620, 2, 1, 0.78, 0.79, 0.88, 500),
            {"kind": "final", "gid": "g1", "decided_at": (T0 + timedelta(seconds=2000)).isoformat()}]
    res = [{"ticker": "H", "result": "yes"}, {"ticker": "A", "result": "no"}]
    f = nhl_frame_from_log(rows, res)
    assert f["recent_goal"].tolist()[:3] == [False, True, True]
    b = t1_late_taker_edge3(f.drop(columns=["result"]))
    assert len(b) == 1
    x = b.iloc[0]
    assert x["bet_id"].startswith("H|") and x["side"] == "yes" and x["price"] == 0.76
    assert x["bet_id"] == "H|" + str(int(T0.timestamp()) + 300) + "|yes"
    assert x["close_price"] == pytest.approx(0.785)               # +300 s mark = mid at t=620 (window 600..720)
    b["outcome"] = 1.0
    schema.enforce(b)                                              # PIT: inputs <= decided < event_start
    # unsettled game -> dropped entirely
    assert nhl_frame_from_log(rows, res[:1]).empty


def test_rule_ignores_ot_and_early_periods():
    rows = [_row(0, 1, 1, 0.30, 0.32, 0.6, 600, period=2), _row(60, 1, 1, 0.30, 0.32, 0.6, 1100)]
    f = nhl_frame_from_log(rows, [{"ticker": "H", "result": "yes"}, {"ticker": "A", "result": "no"}])
    assert t1_late_taker_edge3(f.drop(columns=["result"])).empty
