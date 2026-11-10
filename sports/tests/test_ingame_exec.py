"""ingame_exec.py (NHL late taker + NFLF maker fade) against the fake Kalshi exchange of test_lead_exec: no network,
no key, no real orders. Covers the 3-flag gate, caps, staleness / price-move guards, idempotency, NFLF window + exit,
settlement + CLV, kill rules, kill files, and the NHL logger's live rule signals."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kalshi_exec as kx  # noqa: E402
import lead_exec as lx  # noqa: E402
import ingame_exec as ix  # noqa: E402
import order_guard as og  # noqa: E402
from test_lead_exec import FakeExchange, FakeClient, write_env  # noqa: E402

NOW = datetime(2026, 11, 9, 2, 30, tzinfo=timezone.utc)
HT, AT = "KXNHLGAME-26NOV08TBLA-LA", "KXNHLGAME-26NOV08TBLA-TB"
NF = "KXNFLGAME-26NOV08KCBUF-KC"
ARM = dict(EXEC_MODE="live", LEAD_EXEC_MODE="live", KALSHI_KEY_ID="x", KALSHI_PRIVATE_KEY_PATH="/nope")


@pytest.fixture
def ex(tmp_path, monkeypatch):
    kx.set_root(tmp_path, deploy=tmp_path / "deploy")
    (tmp_path / "deploy").mkdir()
    monkeypatch.setattr(kx, "notify", lambda *a, **k: None)
    e = FakeExchange()
    e.fee_type = "quadratic_with_maker_fees"
    for t in (HT, AT, NF):
        e.markets[t] = {"ticker": t, "status": "active", "close_time": "2026-11-10T00:00:00Z"}
    book(e, HT, 0.80, 0.82)
    book(e, AT, 0.18, 0.20)
    book(e, NF, 0.40, 0.42)
    return e


def book(e, t, yb, ya):
    e.books[t] = [[f"{yb:.2f}", "100"]]                 # YES bids
    e.no_books[t] = [[f"{1 - ya:.2f}", "100"]]          # NO bids -> YES ask = 1 - NO bid


def nhl_sig(age=3.0, price=0.82, side="yes", t=HT, backed="home", gid="g1"):
    d = (NOW - timedelta(seconds=age)).isoformat()
    return {"strategy": "NHL", "key": f"NHL|{gid}|{backed}", "ticker": t, "side": side, "price": price,
            "decided_at": d, "source_row": f"{gid}|{d}", "gid": gid, "backed": backed, "game_day": "2026-11-08"}


def nfl_sig(age=2.0, price=0.40, play="p1", t_play=None):
    d = NOW - timedelta(seconds=age)
    tp = int((t_play or (d - timedelta(seconds=60))).timestamp())
    return {"strategy": "NFLF", "key": f"NFLF|e1|{play}", "ticker": NF, "side": "yes", "price": price,
            "decided_at": d.isoformat(), "t_play": tp, "play_id": play, "gid": "e1", "game_day": "2026-11-08"}


def run(e, sigs, at=NOW, key=True):
    return ix.cmd_tick(sigs, client=FakeClient(e, key=key), clock=lambda: at)


def rows(**f):
    return [r for r in lx.journal() if all(r.get(k) == v for k, v in f.items())]


def posts(e):
    return [c for c in e.calls if c[0] == "POST"]


def auth_calls(e):
    return [c for c in e.calls if c[1].startswith("/portfolio")]


# ---------------------------------------------------------------- gate
@pytest.mark.parametrize("envkv", [
    {}, dict(EXEC_MODE="live"), dict(EXEC_MODE="live", LEAD_EXEC_MODE="live"),         # today's .env: still dry
    dict(EXEC_MODE="live", LEAD_LIVE_NHL="1"), dict(LEAD_EXEC_MODE="live", LEAD_LIVE_NHL="1"),
    dict(EXEC_MODE="live", LEAD_EXEC_MODE="live", LEAD_LIVE_NFLF="1"),                   # wrong strategy armed
    dict(EXEC_MODE="live", LEAD_EXEC_MODE="live", LEAD_LIVE_NHL="true")])
def test_gate_dry_unless_all_three(ex, tmp_path, envkv):
    write_env(tmp_path, KALSHI_KEY_ID="x", KALSHI_PRIVATE_KEY_PATH="/nope", **envkv)
    assert run(ex, [nhl_sig()]) == 0
    assert posts(ex) == [] and auth_calls(ex) == []
    o = rows(event="order")[0]
    assert o["mode"] == "dry" and o["status"] == "dry" and o["count"] == 1


def test_armed_nhl_sends_ioc_at_signal_plus_1c(ex, tmp_path):
    write_env(tmp_path, **ARM, LEAD_LIVE_NHL="1")
    assert run(ex, [nhl_sig()]) == 0
    (_, _, _, body), = posts(ex)
    assert body["side"] == "bid" and body["price"] == "0.8300" and body["count"] == "1"
    assert body["time_in_force"] == "immediate_or_cancel" and "post_only" not in body
    assert body["client_order_id"] == ix.coid("NHL", "live", "NHL|g1|home")
    o = lx.fold(lx.journal(), "live")["NHL|g1|home"]
    assert o["status"] == "filled" and o["filled"] == 1 and o["decided_at"] and o["source_row"].startswith("g1|")


def test_nhl_no_side_is_sell_yes(ex, tmp_path):
    write_env(tmp_path, **ARM, LEAD_LIVE_NHL="1")
    book(ex, AT, 0.18, 0.20)                       # NO at 1 - 0.18 = 0.82
    run(ex, [nhl_sig(t=AT, side="no", price=0.82, backed="home")])
    (_, _, _, body), = posts(ex)
    assert body["side"] == "ask" and body["price"] == "0.1700"      # buy NO at 0.83 == sell YES at 0.17


def test_sizing_one_dollar(ex, tmp_path):
    write_env(tmp_path)
    book(ex, HT, 0.09, 0.10)
    run(ex, [nhl_sig(price=0.10)])
    o = rows(event="order")[0]
    # limit 0.11: unit 0.11 + 0.07*.11*.89 = 0.1169 -> 8 contracts, cost 0.88 + fee ceil(0.0548) 0.06 = 0.94
    assert o["count"] == 8 and o["cost"] <= 1.00 and o["limit_price"] == 0.11
    book(ex, HT, 0.97, 0.98)
    run(ex, [nhl_sig(price=0.98, backed="away", t=HT)])
    o = rows(event="order")[-1]                    # limit 0.99: 1 contract, cost 0.99 + 0.01 fee = $1.00 (allowed)
    assert o["count"] == 1 and o["cost"] <= 1.00


# ---------------------------------------------------------------- staleness / price move
def test_stale_signal_refused(ex, tmp_path):
    write_env(tmp_path, **ARM, LEAD_LIVE_NHL="1")
    run(ex, [nhl_sig(age=21)])
    assert posts(ex) == [] and rows(event="order")[0]["status"] == "skip_stale_signal"


def test_price_moved_refused(ex, tmp_path):
    write_env(tmp_path, **ARM, LEAD_LIVE_NHL="1")
    book(ex, HT, 0.84, 0.85)                       # ask 0.85 vs signal 0.82 = 3c
    run(ex, [nhl_sig()])
    assert posts(ex) == [] and rows(event="order")[0]["status"] == "skip_price_moved"
    book(ex, HT, 0.83, 0.84)                       # 2c is allowed
    run(ex, [nhl_sig(backed="away", t=HT)])
    assert len(posts(ex)) == 1


# ---------------------------------------------------------------- caps
def test_nhl_day_and_season_caps(ex, tmp_path, monkeypatch):
    monkeypatch.setattr(og, "MAX_ORDERS", 100)   # 7 games on one fake ticker would (rightly) trip the breaker
    write_env(tmp_path, **ARM, LEAD_LIVE_NHL="1")
    for i in range(7):
        run(ex, [nhl_sig(gid=f"g{i}")])
    st = lx.fold(lx.journal(), "live")
    money = [o for o in st.values() if lx.committed(o) > 0]
    assert sum(lx.committed(o) for o in money) <= 5.00 + 1e-9
    assert len(money) == 6 and [o["status"] for o in st.values()][-1] == "skip_day"   # 6 x $0.83 = $4.98
    # season: pretend $39.50 already spent on other days
    for i in range(40):
        lx.append({"event": "order", "ts": "2026-11-01T00:00:00+00:00", "mode": "live", "strategy": "NHL",
                   "key": f"NHL|old{i}|home", "ticker": "X", "side": "yes", "status": "filled", "cost": 0.99 if i < 39 else 0.89,
                   "filled": 1, "game_day": "2026-11-01", "decided_at": "2026-11-01T00:00:00+00:00", "settled": True})
    run(ex, [nhl_sig(gid="g99")], at=NOW + timedelta(days=1))  # stale vs NOW+1d -> use a fresh signal instead
    sig = nhl_sig(gid="g98")
    sig["decided_at"] = (NOW + timedelta(days=1)).isoformat()
    sig["game_day"] = "2026-11-09"
    run(ex, [sig], at=NOW + timedelta(days=1))
    assert lx.fold(lx.journal(), "live")["NHL|g98|home"]["status"] == "skip_season"


def test_cash_reserve(ex, tmp_path):
    write_env(tmp_path, **ARM, LEAD_LIVE_NHL="1")
    ex.balance = 80.50
    run(ex, [nhl_sig()])
    assert posts(ex) == [] and rows(event="order")[0]["status"] == "skip_cash"


def test_nflf_caps_order_and_gameday(ex, tmp_path, monkeypatch):
    monkeypatch.setattr(og, "MAX_ORDERS", 100)   # 7 games on one fake ticker would (rightly) trip the breaker
    write_env(tmp_path, **ARM, LEAD_LIVE_NFLF="1")
    for i in range(7):
        run(ex, [nfl_sig(play=f"p{i}")])
    st = lx.fold(lx.journal(), "live")
    assert all(lx.committed(o) <= 2.00 for o in st.values())
    assert sum(lx.committed(o) for o in st.values()) <= 10.00 + 1e-9
    assert any(o["status"] == "skip_day" for o in st.values())


# ---------------------------------------------------------------- idempotency
def test_same_key_never_twice(ex, tmp_path):
    write_env(tmp_path, **ARM, LEAD_LIVE_NHL="1")
    run(ex, [nhl_sig()])
    run(ex, [nhl_sig(age=1)])
    assert len(posts(ex)) == 1


def test_crash_after_post_adopts_exchange_order(ex, tmp_path):
    write_env(tmp_path, **ARM, LEAD_LIVE_NHL="1")
    cid = ix.coid("NHL", "live", "NHL|g1|home")
    ex.orders.append({"order_id": "o9", "client_order_id": cid, "ticker": HT, "status": "executed",
                      "fill_count_fp": "1", "initial_count_fp": "1", "taker_fill_cost_dollars": "0.82",
                      "taker_fees_dollars": "0.02"})
    run(ex, [nhl_sig()])
    assert posts(ex) == []
    assert lx.fold(lx.journal(), "live")["NHL|g1|home"]["filled"] == 1


def test_foreign_position_blocks(ex, tmp_path):
    write_env(tmp_path, **ARM, LEAD_LIVE_NHL="1")
    ex.positions[HT] = 3
    run(ex, [nhl_sig()])
    assert posts(ex) == [] and rows(event="order")[-1]["status"] == "skip_exchange_has_position"


# ---------------------------------------------------------------- kill files / rules
@pytest.mark.parametrize("where", ["lead", "kx", "disable"])
def test_kill_files(ex, tmp_path, where):
    write_env(tmp_path, **ARM, LEAD_LIVE_NHL="1")
    p = {"lead": lx.lead_kill_path(), "kx": tmp_path / "KILL", "disable": lx.disable_path("NHL")}[where]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    run(ex, [nhl_sig()])
    assert posts(ex) == []


def _settled(n_loss, n_win, s="NHL", mode="live", pnl_l=-0.82, pnl_w=0.16):
    out = {}
    for i in range(n_loss + n_win):
        w = i >= n_loss
        out[f"k{i}"] = {"strategy": s, "settled": True, "pnl": pnl_w if w else pnl_l, "won": w, "cost": 0.84,
                        "status": "filled", "ts": f"2026-11-{1 + i % 28:02d}T00:00:00+00:00",
                        "game_day": f"2026-11-{1 + i % 28:02d}", "clv": 0.0}
    return out


@pytest.mark.parametrize("paid,limit,bad", [(0.42, 0.45, False), (0.45, 0.45, False), (0.47, 0.45, False),
                                            (0.48, 0.45, True), (0.34, 0.45, True)])
def test_fill_price_check_is_one_sided(paid, limit, bad):
    assert lx.fill_price_bad(paid, limit) is bad       # 11-08: 0.42 vs limit 0.45 falsely wrote LEAD_KILL


def test_kill_rules():
    assert ix.kill_verdict(_settled(10, 5), "NHL").startswith("10 losses")
    assert ix.kill_verdict(_settled(9, 21), "NHL") is None
    assert ix.kill_verdict(_settled(8, 5, s="NFLF", pnl_l=-0.3), "NFLF").startswith("8 losses")
    assert "P&L" in ix.kill_verdict(_settled(7, 0, s="NFLF", pnl_l=-2.2), "NFLF")
    # after 100: heavy losers -> ROI CI below 0
    st = _settled(9, 21)                                   # first 30: 9 losses (early rule not met)
    for i in range(70):                                    # next 70: 60 losses
        st[f"z{i}"] = {**st["k0"], "pnl": -0.84 if i < 60 else 0.16, "ts": f"2026-12-{1 + i % 28:02d}T00:00:00+00:00",
                       "game_day": f"2026-12-{1 + i % 28:02d}"}
    assert "ROI" in ix.kill_verdict(st, "NHL")
    st = _settled(9, 91, pnl_w=0.30)
    for o in st.values():
        o["clv"] = -0.03
    assert "CLV" in ix.kill_verdict(st, "NHL")


def test_kill_rule_writes_disable_in_live(ex, tmp_path):
    write_env(tmp_path, **ARM, LEAD_LIVE_NFLF="1")
    for i in range(8):
        lx.append({"event": "order", "ts": f"2026-11-0{1 + i % 8}T00:00:00+00:00", "mode": "live", "strategy": "NFLF",
                   "key": f"NFLF|x|{i}", "ticker": NF, "side": "yes", "status": "filled", "cost": 2.0, "filled": 4,
                   "game_day": "2026-11-01", "decided_at": "2026-11-01T00:00:00+00:00"})
        lx.append({"event": "settle", "ts": "2026-11-02T00:00:00+00:00", "mode": "live", "key": f"NFLF|x|{i}",
                   "ticker": NF, "result": "exited", "won": False, "pnl": -0.2})
    run(ex, [])
    assert lx.disable_path("NFLF").exists()
    run(ex, [nfl_sig()])
    assert posts(ex) == [] and rows(event="order")[-1]["status"] == "skip_disabled"


# ---------------------------------------------------------------- NFLF lifecycle
def test_nflf_live_rest_cancel_exit(ex, tmp_path):
    write_env(tmp_path, **ARM, LEAD_LIVE_NFLF="1")
    ex.fill_mode = "rest"
    send = ex.send

    def send_cancel(method, url, *a):                     # the fake exchange honours DELETE
        if method == "DELETE":
            for o_ in ex.orders:
                if url.rstrip("/").endswith(o_["order_id"]):
                    o_["status"] = "canceled"
        return send(method, url, *a)
    ex.send = send_cancel
    sig = nfl_sig()
    run(ex, [sig])
    (_, _, _, body), = posts(ex)
    assert body["post_only"] and body["side"] == "bid" and body["price"] == "0.4000"
    assert body["expiration_time"] == sig["t_play"] + 240 and int(body["count"]) == 4   # $2 / (0.40 + 0.0168)
    # someone sells through us: mark the order partly filled, then the window ends
    o = ex.orders[0]
    o.update(fill_count_fp="3", taker_fill_cost_dollars="1.20", taker_fees_dollars="0.0")
    run(ex, [], at=datetime.fromtimestamp(sig["t_play"] + 241, timezone.utc))
    assert any(c[0] == "DELETE" for c in ex.calls)
    r = lx.fold(lx.journal(), "live")[sig["key"]]
    assert r["status"] == "partial" and r["filled"] == 3 and r["cost"] == pytest.approx(1.20)
    run(ex, [], at=datetime.fromtimestamp(sig["t_play"] + 300, timezone.utc))
    assert len(posts(ex)) == 1                             # no exit before t + 480
    book(ex, NF, 0.43, 0.45)
    ex.positions[NF] = 3                                   # the exit is capped at the exchange-held YES
    run(ex, [], at=datetime.fromtimestamp(sig["t_play"] + 485, timezone.utc))
    exit_body = posts(ex)[-1][3]
    assert exit_body["side"] == "ask" and exit_body["count"] == "3" and exit_body["price"] == "0.4300"
    r = lx.fold(lx.journal(), "live")[sig["key"]]
    assert r["settled"] and r["result"] == "exited"
    assert r["exit_proceeds"] == pytest.approx(3 * 0.43 - kx.fee_total(3, 0.43))   # = 1.29 - 0.06
    assert r["pnl"] == pytest.approx(r["exit_proceeds"] - 1.20, abs=1e-6)
    run(ex, [], at=datetime.fromtimestamp(sig["t_play"] + 520, timezone.utc))
    assert len(posts(ex)) == 2                             # closed: no second exit


def test_nflf_would_cross_and_dry_simulation(ex, tmp_path):
    write_env(tmp_path)
    book(ex, NF, 0.40, 0.40)
    run(ex, [nfl_sig()])
    assert rows(event="order")[0]["status"] == "skip_would_cross"
    book(ex, NF, 0.40, 0.42)
    sig = nfl_sig(play="p2")
    run(ex, [sig])
    assert lx.fold(lx.journal(), "dry")[sig["key"]]["status"] == "dry_resting"
    book(ex, NF, 0.37, 0.39)                       # ask 0.39 <= q - 1c -> trade-through
    run(ex, [], at=datetime.fromtimestamp(sig["t_play"] + 120, timezone.utc))
    assert lx.fold(lx.journal(), "dry")[sig["key"]]["status"] == "dry_filled"
    book(ex, NF, 0.45, 0.47)
    run(ex, [], at=datetime.fromtimestamp(sig["t_play"] + 490, timezone.utc))
    r = lx.fold(lx.journal(), "dry")[sig["key"]]
    assert r["settled"] and r["result"] == "exited" and r["pnl"] > 0
    assert posts(ex) == [] and auth_calls(ex) == []


# ---------------------------------------------------------------- NHL settle + CLV
def test_nhl_marks_and_settle_with_clv(ex, tmp_path):
    write_env(tmp_path, **ARM, LEAD_LIVE_NHL="1")
    sig = nhl_sig(age=0)
    run(ex, [sig])
    book(ex, HT, 0.85, 0.87)
    run(ex, [], at=NOW + timedelta(seconds=90))
    book(ex, HT, 0.88, 0.90)
    run(ex, [], at=NOW + timedelta(seconds=330))
    ex.markets[HT].update(status="finalized", result="yes")
    run(ex, [], at=NOW + timedelta(hours=1))
    r = lx.fold(lx.journal(), "live")[sig["key"]]
    assert r["mark60"] == 0.86 and r["mark300"] == 0.89
    assert r["settled"] and r["won"] and r["clv"] == pytest.approx(0.89 - r["avg_price"], abs=1e-4)
    assert r["pnl"] == pytest.approx(1 - r["cost"], abs=1e-6)


def test_lead_exec_ignores_ingame_rows(ex, tmp_path):
    write_env(tmp_path, **ARM, LEAD_LIVE_NHL="1")
    run(ex, [nhl_sig()])
    st = lx.fold(lx.journal(), "live")
    assert lx.open_exposure(st) == 0.0          # not in the L39/TSA/DUL $40 open cap
    n = len(lx.journal())
    ex.markets[HT].update(status="finalized", result="yes")
    lx.settle(FakeClient(ex), NOW + timedelta(days=2))
    assert len(lx.journal()) == n               # lead_exec.settle leaves NHL rows to ingame_exec


# ---------------------------------------------------------------- NHL logger -> rule signals
def test_logger_rule_signals_one_per_side(tmp_path, monkeypatch):
    import nhl_ingame_logger as L
    from test_nhl_ingame_logger import _row
    monkeypatch.setattr(L, "OUT", tmp_path / "p.jsonl")
    st, hist, got = {}, [], []
    # p_model 0.90 vs home ask 0.80 -> edge 0.9 - 0.8 - 0.0112 = 0.089 >= 0.03 at every tick in the last 15 min
    for i, sec in enumerate((1000, 880, 820, 760)):
        r = _row(i * 60, 2, 1, 0.78, 0.80, 0.90, sec)
        hist.append(r)
        got.append(L.rule_signals([r], st, hist=list(hist)))
    assert [len(g) for g in got] == [0, 1, 0, 0]       # 1000 s left = 16.7 min (outside); first trigger at 880 s
    s = got[1][0]
    assert s["key"] == "NHL|g1|home" and s["side"] == "yes" and s["price"] == 0.80 and s["ticker"] == "H"
    assert s["decided_at"] == hist[1]["decided_at"] and s["source_row"] == f"g1|{hist[1]['decided_at']}"


def test_logger_hook_off_outside_deployed(monkeypatch):
    import nhl_ingame_logger as L
    monkeypatch.delenv("INGAME_EXEC_HOOK", raising=False)
    assert L.exec_hook_on() is False                     # worktree copy never calls the deployed executor
