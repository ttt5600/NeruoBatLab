"""Offline tests for the read-only lock queue logger (no network)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lock_queue_logger as L  # noqa: E402

START = "2026-11-10T01:00Z"           # 20:00 ET on Nov 9
EV = "26NOV09INDDAL"


def game(state="in", name="STATUS_IN_PROGRESS", completed=False, period=4, hs=110, as_=105, gid="1", winner=None, clock="5:00"):
    home = {"homeAway": "home", "team": {"abbreviation": "DAL"}, "score": str(hs)}
    away = {"homeAway": "away", "team": {"abbreviation": "IND"}, "score": str(as_)}
    if winner:
        (home if winner == "home" else away)["winner"] = True
    return {"id": gid, "date": START, "status": {"period": period, "displayClock": clock,
                                                 "type": {"state": state, "name": name, "completed": completed}},
            "competitions": [{"competitors": [home, away]}]}


FINAL = dict(state="post", name="STATUS_FINAL", completed=True, clock="0.0", winner="home")


class Net:
    def __init__(self):
        self.g = game()
        self.calls = []
        self.book = {"orderbook_fp": {"yes_dollars": [["0.9800", "50.00"], ["0.9900", "1200.00"]],
                                      "no_dollars": [["0.0100", "300.00"]]}}
        self.status = "active"
        self.trades = []

    def __call__(self, url, params=None):
        self.calls.append(url)
        if url == L.ESPN:
            return {"events": [self.g]}
        if url.endswith("/orderbook"):
            return self.book
        if url.endswith("/markets/trades"):
            return {"trades": self.trades, "cursor": ""}
        if url.endswith("/markets"):
            if params["event_ticker"].startswith("KXNBATOTAL"):
                return {"markets": [{"ticker": f"KXNBATOTAL-{EV}-{k}", "floor_strike": k - 0.5, "strike_type": "greater",
                                     "status": "active"} for k in (210, 220, 230)]}
            return {"markets": [{"ticker": f"KXNBAGAME-{EV}-DAL", "status": "active"},
                                {"ticker": f"KXNBAGAME-{EV}-IND", "status": "active"}]}
        if "/markets/" in url:
            return {"market": {"status": self.status}}
        raise AssertionError(url)


@pytest.fixture
def tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "OUT", tmp_path / "o.jsonl")
    monkeypatch.setattr(L, "STATE", tmp_path / "s.json")
    return tmp_path


def rows():
    return [json.loads(x) for x in L.OUT.read_text().splitlines()]


def parsed(g):
    return L.parse_games({"events": [g]})


def test_parse_event_ticker_uses_et_date_and_kalshi_codes():
    g = game()
    g["competitions"][0]["competitors"][0]["team"]["abbreviation"] = "GS"
    out = parsed(g)[0]
    assert out["ev"] == "26NOV09INDGSW" and out["espn_date"] == "20261109"


def test_clock_zero_in_progress_is_not_a_final_lock():
    g = parsed(game(period=4, clock="0.0", hs=120, as_=100))[0]
    assert not g["final"] and g["winner"] is None


def test_final_needs_status_final_and_clean_winner():
    assert parsed(game(**FINAL))[0]["winner"] == "DAL"
    assert parsed(game(**{**FINAL, "winner": "away"}))[0]["winner"] is None     # flag disagrees with score
    assert parsed(game(**{**FINAL, "winner": None}, hs=100, as_=100))[0]["winner"] is None
    assert not parsed(game(state="post", name="STATUS_POSTPONED", completed=True))[0]["final"]


def test_over_margin_boundary():
    n, st = Net(), {}
    r = L.detect(st, parsed(game(hs=110, as_=108)), n, 1000.0)      # 218: K=209.5 needs 213, K=219.5 needs 223
    assert [x["K"] for x in r] == [209.5] and r[0]["late_detect"] is True
    r2 = L.detect(st, parsed(game(hs=112, as_=111)), n, 1015.0)     # 223 = exactly 219.5 + 3.5
    assert [x["K"] for x in r2] == [219.5] and r2[0]["late_detect"] is False


def test_ml_lock_once_and_only_winner():
    n, st = Net(), {}
    L.detect(st, parsed(game()), n, 1000.0)
    final = parsed(game(**FINAL))
    ml = [x for x in L.detect(st, final, n, 1015.0) if x["series"] == "KXNBAGAME"]
    assert [x["ticker"] for x in ml] == [f"KXNBAGAME-{EV}-DAL"] and ml[0]["late_detect"] is False
    assert L.detect(st, final, n, 1030.0) == []


def test_book_summary_ask_from_no_bids():
    s = L.book_summary(Net().book)
    assert (s["yes_99"], s["yes_98"], s["best_bid_c"], s["best_ask_c"], s["no_1c"]) == (1200.0, 50.0, 99, 99, 300.0)
    empty = L.book_summary({"orderbook_fp": {}})
    assert empty["book_empty"] and empty["best_ask_c"] is None and empty["yes_99"] == 0


def lock(**kw):
    return {"ticker": "X", "series": "S", "ltype": "over", "K": 1.0, "ts": 1000.0, "late": False, "game": "1", "ev": EV,
            "snaps": [], "tape": [], "seen": [], **kw}


def test_trades_deduped_across_snapshots():
    n, lk = Net(), lock()
    n.trades = [{"trade_id": "a", "created_time": "2026-11-10T03:00:00.5Z", "yes_price_dollars": "0.9900", "count_fp": "7.00",
                 "taker_side": "no"}]
    assert L.take(lk, 0, "snapshot", n, lambda: 1001.0)["new_trades"][0][1:4] == [99, 7.0, "no"]
    assert L.take(lk, 15, "snapshot", n, lambda: 1016.0)["new_trades"] == []


def test_closed_market_stops_book_keeps_tape():
    n, lk = Net(), lock(snaps=[15, 30], tape=[900])
    n.status = "closed"
    assert L.take(lk, 0, "snapshot", n, lambda: 1001.0)["market_status"] == "closed"
    assert lk["snaps"] == [] and lk["tape"] == [900]


def test_error_row_does_not_kill_run():
    def boom(url, params=None):
        raise RuntimeError("503")
    assert "503" in L.take(lock(), 0, "snapshot", boom, lambda: 1001.0)["error"]


def clocked(start, step=1.0):
    t = {"v": start}

    def sleep(s):
        t["v"] += max(s, step)
    return (lambda: t["v"]), sleep


def seeded(now):
    L.save_state({"sched": {"fetched": now - 10, "games": {"1": {"start": int(L.epoch(START)), "espn_date": "20261109"}}}})


def test_cheap_exit_makes_no_network_call_outside_window(tmp):
    n, now = Net(), L.epoch(START) - 3600
    seeded(now)
    assert L.tick(n, *clocked(now)) == 0
    assert n.calls == [] and not L.OUT.exists()


def test_early_game_one_scoreboard_call_then_exit(tmp):
    n, now = Net(), L.epoch(START) + 80 * 60
    n.g = game(period=2)
    seeded(now)
    assert L.tick(n, *clocked(now)) == 0
    assert n.calls == [L.ESPN] and not L.OUT.exists()


def test_tick_snapshots_at_delays_and_report(tmp, capsys):
    n, now = Net(), L.epoch(START) + 7200
    n.g = game(hs=110, as_=108)
    seeded(now)
    n.trades = [{"trade_id": "t1", "created_time": L.iso(now + 400), "yes_price_dollars": "0.9900", "count_fp": "2000.00", "taker_side": "no"},
                {"trade_id": "t2", "created_time": L.iso(now + 400), "yes_price_dollars": "0.9900", "count_fp": "999.00", "taker_side": "yes"}]
    L.tick(n, *clocked(now), max_s=320)
    out = rows()
    locks = [r for r in out if r["kind"] == "lock"]
    assert len(locks) == 1 and locks[0]["K"] == 209.5
    snaps = [r for r in out if r["kind"] == "snapshot"]
    assert [r["delay_s"] for r in snaps] == list(L.SNAP_S)
    assert all(r["as_of_ts"] >= r["decided_ts"] + r["delay_s"] for r in snaps)
    assert snaps[0]["yes_99"] == 1200.0 and snaps[0]["decided_at"] == locks[0]["decided_at"]
    L.report(L.OUT, all_locks=True)                     # the single lock is late_detect, so include it
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("KXNBATOTAL") and x.split()[1] == "0"][0].split()
    # queue 1200, V 2000 (the 999 taker-buy print is excluded), share (2000 - 1200) / 2000 = 0.400
    assert line[6] == "2000" and line[7] == "0.400"
    L.report(L.OUT)
    assert "KXNBATOTAL" not in capsys.readouterr().out.split("V>q")[-1]
