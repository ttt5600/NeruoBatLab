"""lead_exec.py against a fake Kalshi exchange: no network, no key, no real orders.

Covers: live gating (needs EXEC_MODE=live AND LEAD_EXEC_MODE=live), per-order / month / week / season caps, the global
$40 open cap and the $80 cash reserve, idempotency (re-run, deterministic client_order_id, exchange-side order),
kill switches and the per-strategy kill rules, the DUL 30-90 min window, strike choice, price gate and CLV, maker
order bodies (NO and YES side), the post-only cross pre-check and the fill price check.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kalshi_exec as kx  # noqa: E402
import lead_exec as lx  # noqa: E402

NOW = datetime(2026, 11, 29, 6, 0, tzinfo=timezone.utc)
CLOSE = "2026-12-01T04:59:59Z"
RAIN = "KXRAINNYCM-26NOV-6"
TSA_NO = "KXTSAW-26DEC06-A2.80"
TSA_YES = "KXTSAW-26DEC06-A1.90"
EVT = "KXNFLTOTAL-26NOV29WASNYG"


class FakeExchange:
    def __init__(self):
        self.calls, self.markets, self.books, self.no_books = [], {}, {}, {}
        self.orders, self.positions, self.trades = [], {}, {}
        self.balance, self.fee_type, self.fill_mode = 500.0, "quadratic", "rest"
        self.fill_price_override = None

    def send(self, method, url, headers, params, data, timeout):
        path = urlparse(url).path.replace(kx.BASE, "", 1)
        params = dict(params or {})
        self.calls.append((method, path, params, json.loads(data) if data else None))
        if path.endswith("/orderbook"):
            t = path.split("/")[2]
            return 200, {"orderbook_fp": {"yes_dollars": self.books.get(t, []), "no_dollars": self.no_books.get(t, [])}}
        if path.startswith("/series/"):
            return 200, {"series": {"fee_type": self.fee_type}}
        if path == "/markets/trades":
            return 200, {"trades": self.trades.get(params["ticker"], [])}
        if path == "/markets":
            return 200, {"markets": [m for m in self.markets.values() if m.get("event_ticker") == params.get("event_ticker")]}
        if path.startswith("/markets/"):
            t = path.split("/")[2]
            return (200, {"market": self.markets[t]}) if t in self.markets else (404, {})
        if path == "/portfolio/balance":
            return 200, {"balance_dollars": f"{self.balance:.4f}"}
        if path == "/portfolio/orders":
            rows = [o for o in self.orders if (not params.get("ticker") or o["ticker"] == params["ticker"])
                    and (not params.get("status") or o["status"] == params["status"])]
            return 200, {"orders": rows, "cursor": ""}
        if path == "/portfolio/positions":
            return 200, {"market_positions": [{"ticker": t, "position_fp": str(q)} for t, q in self.positions.items()
                                              if not params.get("ticker") or t == params["ticker"]], "cursor": ""}
        if path == "/portfolio/events/orders" and method == "POST":
            b = json.loads(data)
            if any(o["client_order_id"] == b["client_order_id"] for o in self.orders):
                return 409, {"code": "conflict"}
            n = float(b["count"])
            yes_px = float(b["price"])
            own = (1 - yes_px) if b["side"] == "ask" else yes_px
            if self.fill_price_override is not None:
                own = self.fill_price_override
            filled = 0.0 if (b.get("post_only") and self.fill_mode == "rest") else n
            o = {"order_id": f"o{len(self.orders)}", "client_order_id": b["client_order_id"], "ticker": b["ticker"],
                 "status": "resting" if filled < n and b.get("post_only") else "executed",
                 "fill_count_fp": str(filled), "initial_count_fp": str(n),
                 "taker_fill_cost_dollars": f"{filled * own:.4f}", "taker_fees_dollars": "0"}
            self.orders.append(o)
            return 201, {"order_id": o["order_id"], "fill_count": filled, "remaining_count": n - filled}
        if method == "DELETE":
            return 200, {"reduced_by": 1}
        return 404, {}


class FakeClient(kx.Kalshi):
    def __init__(self, ex, key=True):
        super().__init__({"KALSHI_KEY_ID": "x", "KALSHI_PRIVATE_KEY_PATH": "/nope"} if key else {}, "prod",
                         transport=ex.send, sleep=lambda s: None)

    def _sign(self, method, path):
        return {}


@pytest.fixture
def ex(tmp_path, monkeypatch):
    kx.set_root(tmp_path, deploy=tmp_path / "deploy")
    (tmp_path / "deploy").mkdir()
    monkeypatch.setattr(kx, "notify", lambda *a, **k: None)
    monkeypatch.setattr(lx, "L39_PICKS", tmp_path / "l39.jsonl")
    monkeypatch.setattr(lx, "TSA_PAPER", tmp_path / "tsa.jsonl")
    monkeypatch.setattr(lx, "SCHEDULE", tmp_path / "sched.csv")
    e = FakeExchange()
    for t in (RAIN, TSA_NO, TSA_YES):
        e.markets[t] = {"ticker": t, "status": "active", "close_time": CLOSE}
        e.books[t] = [["0.01", "100"]]          # NO ask 0.99
        e.no_books[t] = [["0.97", "50"]]        # NO bid 0.97 -> YES ask 0.03
    return e


def write_env(tmp: Path, **kv):
    (tmp / ".env").write_text("".join(f"{k}={v}\n" for k, v in kv.items()))


def jw(p: Path, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def l39_pick(t=RAIN, q=0.98, age_h=1.0, d="2026-11-28"):
    return {"ticker": t, "event": t.rsplit("-", 1)[0], "maker_q": q, "d": d, "ts": (NOW - timedelta(hours=age_h)).isoformat()}


def tsa_post(t=TSA_NO, side="no", q=0.96, age_h=2.0):
    return {"kind": "post", "ticker": t, "event": "KXTSAW-26DEC06", "side": side, "q": q,
            "ts": (NOW - timedelta(hours=age_h)).isoformat()}


def orders(mode="dry"):
    return {t: o for t, o in lx.fold(lx.journal(), mode).items()}


def run(ex, live=False, now=NOW, key=None):
    return lx.cmd_run(["--live"] if live else [], client=FakeClient(ex, key=live if key is None else key), now=now)


# ------------------------------------------------------------------------------------------------ gating / kills
def test_live_needs_both_env_flags(tmp_path, ex):
    write_env(tmp_path, EXEC_MODE="live")
    assert lx.lead_mode(kx.env(), True) == "dry"
    write_env(tmp_path, EXEC_MODE="live", LEAD_EXEC_MODE="live")
    assert lx.lead_mode(kx.env(), True) == "live"
    assert lx.lead_mode(kx.env(), False) == "dry"


def test_dry_never_posts(tmp_path, ex):
    jw(lx.L39_PICKS, [l39_pick()])
    run(ex)
    assert not [c for c in ex.calls if c[0] in ("POST", "DELETE")]
    o = orders()[RAIN]
    assert o["status"] == "dry" and o["count"] == 3 and o["cost"] == pytest.approx(2.94)
    assert o["tag"] == "lead_L39" and o["decided_at"] and o["price_ts"] and o["source_ts"]


@pytest.mark.parametrize("which", ["lead", "global"])
def test_kill_files_stop_everything(tmp_path, ex, which):
    jw(lx.L39_PICKS, [l39_pick()])
    (lx.lead_kill_path() if which == "lead" else tmp_path / "KILL").parent.mkdir(parents=True, exist_ok=True)
    (lx.lead_kill_path() if which == "lead" else tmp_path / "KILL").write_text("x")
    run(ex)
    assert lx.journal() == []


def test_disable_file_stops_one_strategy(tmp_path, ex):
    jw(lx.L39_PICKS, [l39_pick()])
    jw(lx.TSA_PAPER, [tsa_post()])
    kx.DATA.mkdir(parents=True, exist_ok=True)
    lx.disable_path("L39").write_text("x")
    run(ex)
    assert set(orders()) == {TSA_NO}


# ------------------------------------------------------------------------------------------------ L39
def test_l39_stale_pick_skipped(ex):
    jw(lx.L39_PICKS, [l39_pick(age_h=4)])
    run(ex)
    assert orders()[RAIN]["status"] == "skip_stale_signal"


def test_l39_would_cross_skipped(ex):
    jw(lx.L39_PICKS, [l39_pick(q=0.99)])            # NO ask 0.99 <= q -> post_only would cross
    run(ex)
    assert orders()[RAIN]["status"] == "skip_would_cross"


def test_l39_month_cap(ex):
    picks = [l39_pick(t=f"KXRAINNYCM-26NOV-{i}") for i in range(7)]
    for p in picks:
        ex.markets[p["ticker"]] = {"ticker": p["ticker"], "status": "active", "close_time": CLOSE}
        ex.books[p["ticker"]] = [["0.01", "100"]]
    jw(lx.L39_PICKS, picks)
    run(ex)
    rows = sorted(orders().values(), key=lambda o: o["ticker"])
    assert [o["count"] for o in rows] == [3, 3, 3, 3, 3, 0, 0]          # 5 x 2.94 = 14.70 <= 15
    assert rows[-1]["status"] == "skip_period"
    assert lx.spent_in(orders(), "L39", "2026-11") <= 15


def test_l39_idempotent_rerun(ex):
    jw(lx.L39_PICKS, [l39_pick()])
    run(ex)
    run(ex, now=NOW + timedelta(minutes=15))
    assert len([r for r in lx.journal() if r.get("event") == "order"]) == 1


# ------------------------------------------------------------------------------------------------ TSA
def test_tsa_sizing_and_sides(ex):
    jw(lx.TSA_PAPER, [tsa_post(q=0.96), tsa_post(t=TSA_YES, side="yes", q=0.30)])
    ex.no_books[TSA_YES] = [["0.60", "10"]]         # YES ask 0.40 > q 0.30 -> no cross
    run(ex)
    o = orders()
    assert o[TSA_NO]["count"] == 1 and o[TSA_NO]["cost"] == pytest.approx(0.96)
    assert o[TSA_YES]["count"] == 3 and o[TSA_YES]["cost"] == pytest.approx(0.90)
    assert lx.order_body(o[TSA_NO] | {"maker_q": 0.96}, "c")["side"] == "ask"
    assert lx.order_body(o[TSA_NO] | {"maker_q": 0.96}, "c")["price"] == "0.0400"
    b = lx.order_body(o[TSA_YES] | {"maker_q": 0.30}, "c")
    assert b["side"] == "bid" and b["price"] == "0.3000" and b["post_only"] is True


def test_tsa_week_cap(ex):
    posts = [tsa_post(t=f"KXTSAW-26DEC06-A2.{i:02d}") for i in range(10)]
    for p in posts:
        ex.markets[p["ticker"]] = {"ticker": p["ticker"], "status": "active", "close_time": CLOSE}
        ex.books[p["ticker"]] = [["0.01", "100"]]
    jw(lx.TSA_PAPER, posts)
    run(ex)
    rows = sorted(orders().values(), key=lambda o: o["ticker"])
    assert sum(o["count"] for o in rows) == 8 and rows[-1]["status"] == "skip_period"


def test_tsa_stale_post(ex):
    jw(lx.TSA_PAPER, [tsa_post(age_h=30)])
    run(ex)
    assert orders()[TSA_NO]["status"] == "skip_stale_signal"


# ------------------------------------------------------------------------------------------------ global caps
def test_global_open_cap(ex):
    rows = [{"event": "order", "ts": NOW.isoformat(), "mode": "dry", "strategy": "DUL", "ticker": f"X{i}",
             "status": "dry", "cost": 2.0, "filled": 1, "count": 1} for i in range(19)]   # $38 open
    kx.DATA.mkdir(parents=True, exist_ok=True)
    jw(lx.log_path(), rows)
    jw(lx.L39_PICKS, [l39_pick()])
    run(ex)
    o = orders()[RAIN]
    assert o["count"] == 2 and o["cost"] <= 2.0 + 1e-9                     # $2 left of the $40


def test_cash_reserve(tmp_path, ex):
    write_env(tmp_path, EXEC_MODE="live", LEAD_EXEC_MODE="live")
    jw(lx.L39_PICKS, [l39_pick(), l39_pick(t=TSA_NO, q=0.96)])
    ex.balance = 82.5                                                       # $2.50 above the reserve
    run(ex, live=True)
    o = orders("live")
    assert o[RAIN]["count"] == 2 and o[RAIN]["status"] == "resting"         # 2 x .98 = 1.96
    assert o[TSA_NO]["status"] == "skip_cash"                               # .54 left < .96
    assert ex.balance - sum(lx.committed(x) for x in o.values()) >= lx.CASH_RESERVE


# ------------------------------------------------------------------------------------------------ live placement
def test_live_maker_writeahead_and_idempotent(tmp_path, ex):
    write_env(tmp_path, EXEC_MODE="live", LEAD_EXEC_MODE="live")
    jw(lx.L39_PICKS, [l39_pick()])
    run(ex, live=True)
    posts = [c for c in ex.calls if c[0] == "POST"]
    assert len(posts) == 1
    b = posts[0][3]
    assert b["side"] == "ask" and b["price"] == "0.0200" and b["post_only"] and b["count"] == "3"
    assert b["client_order_id"] == lx.coid("L39", "live", RAIN)
    evs = [r.get("status") for r in lx.journal() if r.get("ticker") == RAIN]
    assert evs[0] == "submitting"                                           # written before the POST
    run(ex, live=True, now=NOW + timedelta(hours=1))
    assert len([c for c in ex.calls if c[0] == "POST"]) == 1
    assert orders("live")[RAIN]["cost"] == pytest.approx(2.94)              # resting bid stays committed


def test_exchange_side_order_blocks(tmp_path, ex):
    write_env(tmp_path, EXEC_MODE="live", LEAD_EXEC_MODE="live")
    jw(lx.L39_PICKS, [l39_pick()])
    ex.positions[RAIN] = -2
    run(ex, live=True)
    assert orders("live")[RAIN]["status"] == "skip_exchange_has_order"
    assert not [c for c in ex.calls if c[0] == "POST"]


def test_fill_price_check_writes_lead_kill(tmp_path, ex):
    write_env(tmp_path, EXEC_MODE="live", LEAD_EXEC_MODE="live")
    jw(lx.L39_PICKS, [l39_pick(), l39_pick(t=TSA_NO, q=0.96)])
    ex.fill_mode, ex.fill_price_override = "fill", 0.02                    # wrong convention
    run(ex, live=True)
    assert lx.lead_kill_path().exists()
    assert len([c for c in ex.calls if c[0] == "POST"]) == 1


# ------------------------------------------------------------------------------------------------ kill rules
def _settled(strategy, wins, mode="live", clv=None, cost=1.0):
    st = {}
    for i, w in enumerate(wins):
        st[f"T{i}"] = {"strategy": strategy, "ticker": f"T{i}", "ts": (NOW + timedelta(minutes=i)).isoformat(), "status": "filled",
                       "filled": 1, "cost": cost, "settled": True, "won": w, "pnl": (1 - cost) if w else -cost,
                       "clv": clv}
    return st


def test_kill_rules():
    assert lx.kill_verdict(_settled("L39", [True] * 50 + [False]), "L39") is None
    assert lx.kill_verdict(_settled("L39", [True] * 50 + [False, False]), "L39")
    assert lx.kill_verdict(_settled("L39", [True] * 100 + [False, False]), "L39") is None   # after the first 100
    assert lx.kill_verdict(_settled("TSA", [False] + [True] * 28 + [False]), "TSA")
    assert lx.kill_verdict(_settled("TSA", [False] + [True] * 29 + [False]), "TSA") is None
    assert lx.kill_verdict(_settled("DUL", [False] * 39, clv=-0.1, cost=0.5), "DUL") is None  # < 40 settled
    assert lx.kill_verdict(_settled("DUL", [True] * 19 + [False] * 21, clv=0.01, cost=0.5), "DUL")   # ROI < 0
    assert lx.kill_verdict(_settled("DUL", [True] * 25 + [False] * 15, clv=-0.01, cost=0.5), "DUL")  # CLV < 0
    assert lx.kill_verdict(_settled("DUL", [True] * 25 + [False] * 15, clv=0.01, cost=0.5), "DUL") is None


def test_kill_rule_disables_in_live_not_dry(tmp_path, ex):
    kx.DATA.mkdir(parents=True, exist_ok=True)
    for mode in ("dry", "live"):
        for i in range(2):
            lx.append({"event": "order", "ts": NOW.isoformat(), "mode": mode, "strategy": "L39", "ticker": f"L{i}",
                       "status": "filled", "filled": 1, "cost": 0.98, "count": 1})
            lx.append({"event": "settle", "ts": NOW.isoformat(), "mode": mode, "ticker": f"L{i}", "result": "yes",
                       "won": False, "pnl": -0.98})
    assert lx.apply_kill_rules("dry", {}, NOW)[0].startswith("[dry]")
    assert not lx.disable_path("L39").exists()
    lx.apply_kill_rules("live", {}, NOW)
    assert lx.disable_path("L39").exists()


# ------------------------------------------------------------------------------------------------ DUL
SCHED_COLS = "game_id,season,game_type,week,gameday,gametime,away_team,home_team,div_game\n"


def dul_setup(ex, kick_et="20:15", week=12, div=1, asks=None):
    lx.SCHEDULE.write_text(SCHED_COLS + f"2026_{week}_WAS_NYG,2026,REG,{week},2026-11-28,{kick_et},WAS,NYG,{div}\n")
    kick = datetime(2026, 11, 28, int(kick_et[:2]), int(kick_et[3:]), tzinfo=lx.ET)
    exp = (kick + timedelta(hours=3)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    evt = "KXNFLTOTAL-26NOV28WASNYG"
    quotes = asks or {"38": (0.62, 0.63), "41": (0.50, 0.52), "44": (0.47, 0.48), "47": (0.30, 0.31)}
    for k, (yb, ya) in quotes.items():
        t = f"{evt}-{k}"
        ex.markets[t] = {"ticker": t, "event_ticker": evt, "status": "active", "floor_strike": float(k) - 0.5,
                         "yes_bid_dollars": f"{yb}", "yes_ask_dollars": f"{ya}", "close_time": "2026-12-01T00:00:00Z",
                         "expected_expiration_time": exp, "updated_time": "2026-11-29T00:30:00Z"}
        ex.books[t] = [[f"{yb}", "500"]]
    return kick.astimezone(timezone.utc), evt


def test_dul_window_and_strike(ex):
    kick, evt = dul_setup(ex)
    for mins, expect in ((100, 0), (20, 0)):
        run(ex, now=kick - timedelta(minutes=mins))
        assert not orders(), mins
    run(ex, now=kick - timedelta(minutes=60))
    (o,) = orders().values()
    # mids: 38 .625, 41 .51, 44 .475 -> 41 is closest to .50 (|.01| < |.025|)
    assert o["ticker"] == f"{evt}-41" and o["strike"] == 40.5 and o["side"] == "no"
    assert o["avg_price"] == 0.50 and o["kickoff"] == kick.isoformat(timespec="seconds")
    unit = 0.50 + 0.07 * 0.25
    assert o["count"] == int(2 // unit) and o["cost"] <= 2.0
    run(ex, now=kick - timedelta(minutes=45))                              # one bet per game
    assert len([r for r in lx.journal() if r.get("event") == "order"]) == 1


def test_dul_filters_non_div_and_early_week(ex):
    kick, _ = dul_setup(ex, div=0)
    run(ex, now=kick - timedelta(minutes=60))
    kick, _ = dul_setup(ex, week=9)
    run(ex, now=kick - timedelta(minutes=60))
    assert not orders()


def test_dul_price_gate_then_retry_in_window(ex):
    kick, evt = dul_setup(ex, asks={"41": (0.40, 0.60)})                   # mid .50, NO ask .60 > .56
    run(ex, now=kick - timedelta(minutes=80))
    assert list(orders().values())[0]["status"] == "skip_price"
    ex.markets[f"{evt}-41"].update(yes_bid_dollars="0.47", yes_ask_dollars="0.53")
    ex.books[f"{evt}-41"] = [["0.47", "500"]]                               # NO ask .53
    run(ex, now=kick - timedelta(minutes=50))
    o = orders()[f"{evt}-41"]
    assert o["status"] == "dry" and o["avg_price"] == 0.53


def test_dul_kickoff_mismatch(ex):
    kick, evt = dul_setup(ex)
    for m in ex.markets.values():
        m["expected_expiration_time"] = "2026-11-29T22:00:00Z"
    run(ex, now=kick - timedelta(minutes=60))
    assert list(orders().values())[0]["status"] == "skip_kickoff_mismatch"


def test_dul_season_cap(ex):
    kx.DATA.mkdir(parents=True, exist_ok=True)
    jw(lx.log_path(), [{"event": "order", "ts": "2026-11-20T00:00:00+00:00", "mode": "dry", "strategy": "DUL",
                        "ticker": f"D{i}", "game_id": f"g{i}", "status": "dry", "cost": 1.99, "filled": 3, "count": 3,
                        "settled": True} for i in range(30)] +
       [{"event": "settle", "ts": "2026-11-21T00:00:00+00:00", "mode": "dry", "ticker": f"D{i}", "result": "no",
         "won": True, "pnl": 1.01} for i in range(30)])                    # $59.70 spent, all settled
    kick, _ = dul_setup(ex)
    run(ex, now=kick - timedelta(minutes=60))
    (o,) = [x for x in orders().values() if x.get("game_id") == "2026_12_WAS_NYG"]
    assert o["status"] == "skip_period"                                     # $0.30 left < 1 contract


def test_dul_settle_clv(ex):
    kick, evt = dul_setup(ex)
    run(ex, now=kick - timedelta(minutes=60))
    t = f"{evt}-41"
    ex.markets[t]["result"] = "no"
    ex.trades[t] = [{"created_time": "2026-11-29T01:10:00Z", "no_price_dollars": "0.5400", "yes_price_dollars": "0.46"},
                    {"created_time": "2026-11-29T00:50:00Z", "no_price_dollars": "0.5200", "yes_price_dollars": "0.48"}]
    assert lx.settle(FakeClient(ex, key=False), kick + timedelta(hours=4)) == 1
    o = orders()[t]
    assert o["won"] and o["close_no"] == 0.54 and o["clv"] == pytest.approx(0.04)
    assert o["pnl"] == pytest.approx(o["filled"] - o["cost"])
