"""Offline tests for the NFL F1L live paper logger (no network, no executor)."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nfl_fade_live as F  # noqa: E402

START = datetime(2026, 11, 8, 18, 0, tzinfo=timezone.utc)          # 13:00 ET
T_PLAY = int((START + timedelta(minutes=40)).timestamp())


def sb(state="in"):
    return {"events": [{"id": "401", "date": "2026-11-08T18:00Z", "status": {"type": {"state": state}},
                        "competitions": [{"competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "WSH"}},
                            {"homeAway": "away", "team": {"abbreviation": "JAX"}}]}]}]}


def summary(delta=0.10):
    wc = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    return {"winprobability": [{"homeWinPercentage": 0.50, "playId": "1"}, {"homeWinPercentage": 0.50 + delta, "playId": "2"},
                               {"homeWinPercentage": 0.52 + delta, "playId": "3"}],
            "drives": {"previous": [{"plays": [{"id": "1", "wallclock": wc(T_PLAY - 40)}, {"id": "2", "wallclock": wc(T_PLAY)},
                                               {"id": "3", "wallclock": wc(T_PLAY + 40)}]}]}}


@pytest.fixture
def tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "OUT", tmp_path / "o.jsonl")
    monkeypatch.setattr(F, "STATE", tmp_path / "s.json")
    return tmp_path


def fake(bid=0.40, ask=0.42, delta=0.10, state="in"):
    calls = []

    def get(url, params=None):
        calls.append(url)
        if url.endswith("/scoreboard"):
            return sb(state)
        if url.endswith("/summary"):
            return summary(delta)
        if "/markets/" in url:
            return {"market": {"status": "active", "yes_bid_dollars": f"{bid:.4f}", "yes_ask_dollars": f"{ask:.4f}"}}
        raise AssertionError(url)
    get.calls = calls
    return get


def test_games_kalshi_codes():
    g, = F.games(sb())
    assert g["event_ticker"] == "KXNFLGAME-26NOV08JACWAS" and g["tickers"]["home"] == "KXNFLGAME-26NOV08JACWAS-WAS"
    assert g["game_day"] == "2026-11-08"


def test_big_play_side_is_team_whose_wp_fell():
    bp, = F.big_plays(summary(0.10))
    assert bp["play_id"] == "2" and bp["t"] == T_PLAY and bp["side"] == "away"      # home WP rose -> fade toward away
    assert F.big_plays(summary(0.07)) == []
    assert F.big_plays(summary(-0.09))[0]["side"] == "home"


def test_subtick_pending_signal_late_noquote(tmp):
    st = {}
    at = lambda s: (lambda: datetime.fromtimestamp(T_PLAY + s, timezone.utc))  # noqa: E731
    rows, sigs, _ = F.subtick(st, fake(), at(20))
    assert [r["kind"] for r in rows] == ["big_play"] and sigs == []                 # too early (< t+60)
    rows, sigs, _ = F.subtick(st, fake(), at(65))
    assert [r["kind"] for r in rows] == ["signal"]
    s = sigs[0]
    assert s["strategy"] == "NFLF" and s["price"] == 0.40 and s["ticker"].endswith("-JAC") and s["t_play"] == T_PLAY
    rows, sigs, _ = F.subtick(st, fake(), at(80))
    assert rows == [] and sigs == []                                                # one signal per play
    st2 = {}
    F.subtick(st2, fake(), at(20))
    rows, _, _ = F.subtick(st2, fake(), at(130))
    assert rows[0]["kind"] == "late"
    st3 = {}
    F.subtick(st3, fake(), at(20))
    rows, sigs, _ = F.subtick(st3, fake(bid=0.40, ask=0.47), at(70))
    assert rows[0]["kind"] == "no_quote" and sigs == []


def test_grade_f1_mechanics():
    sig = {"key": "k", "ticker": "T", "bid": 0.40, "t_play": T_PLAY,
           "decided_at": datetime.fromtimestamp(T_PLAY + 65, timezone.utc).isoformat()}
    cs = [{"end": T_PLAY + 60 + 60 * i, "bid": 0.40, "ask": 0.42, "ask_low": 0.41} for i in range(10)]
    cs[2]["ask_low"] = 0.39                                                         # trade-through in the window
    for c in cs:
        if c["end"] >= T_PLAY + 480:
            c["bid"] = 0.45
    g = F.grade_one(sig, cs)
    assert g["f1"] == "fill" and g["live_fill"] and g["exit"] == 0.45
    assert g["f1_pnl"] == pytest.approx((0.45 - 0.07 * 0.45 * 0.55) - (0.40 + 0.0175 * 0.4 * 0.6), abs=1e-4)
    cs[2]["ask_low"] = 0.41
    assert F.grade_one(sig, cs)["f1"] == "no_fill"


def test_tick_idle_and_live_calls_hook(tmp):
    got = []
    hook = lambda s: got.append(s) or 0  # noqa: E731
    g = fake(state="pre")
    assert F.tick(get=g, now=START - timedelta(hours=3), clock=lambda: START - timedelta(hours=3), hook=hook) == 0
    assert got == []                                                                # idle: no hook call
    clock_t = [T_PLAY + 62]
    clk = lambda: datetime.fromtimestamp(clock_t[0], timezone.utc)  # noqa: E731

    def sleep(s):
        clock_t[0] += s
    assert F.tick(get=fake(), now=clk(), clock=clk, sleep=sleep, hook=hook) == 0
    assert len(got) >= 3 and sum(len(x) for x in got) == 1                         # hook every sub-tick, 1 signal
    kinds = [json.loads(l)["kind"] for l in (tmp / "o.jsonl").read_text().splitlines()]
    assert kinds == ["big_play", "signal"]
