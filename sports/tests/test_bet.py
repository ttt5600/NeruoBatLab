"""bet.py (unified router) against a fake Kalshi client: no network, no key, no orders.

Covers: units -> contracts sizing (maker/taker fee), tier per-bet caps, tier-A night cap incl. kalshi_exec spend,
daily/weekly loss stops (worst case incl. open risk), monthly tier-C cap, KILL file, shard refusal, ESPN start
lookup + expiry, external logging/settling, API settle, allocate dry-run vs --confirm, legacy journal read.
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bet  # noqa: E402

NOW = datetime(2026, 10, 27, 20, 0, tzinfo=timezone.utc)   # 16:00 ET
NBA = "KXNBAGAME-26OCT27PORLAL-POR"
WX = "KXHIGHNY-26OCT28-T67"


class FakeKalshi:
    def __init__(self, shards=None, exchange_index=3, yes=(0.54, 0.56), fee_type="quadratic"):
        self.shards = shards if shards is not None else {0: 100.0, 1: 0.0, 2: 0.0, 3: 100.0}
        self.ei, self.yes, self.fee_type = exchange_index, yes, fee_type
        self.sent, self.posts, self.order_rows, self.results = [], [], {}, {}

    def market(self, t):
        m = {"ticker": t, "status": "active", "title": t, "close_time": "2026-11-10T03:00:00Z",
             "yes_bid_dollars": str(self.yes[0]), "yes_ask_dollars": str(self.yes[1]),
             "no_bid_dollars": f"{1 - self.yes[1]:.2f}", "no_ask_dollars": f"{1 - self.yes[0]:.2f}",
             "result": self.results.get(t, "")}
        if self.ei is not None:
            m["exchange_index"] = self.ei
        return m

    def series_fee_type(self, s):
        return self.fee_type

    def request(self, method, path, params=None, body=None, auth=True):
        if path == "/portfolio/balance":
            return 200, {"balance": int(sum(self.shards.values()) * 100),
                         "balance_breakdown": [{"exchange_index": k, "balance": f"{v:.4f}"} for k, v in self.shards.items()]}
        if path == "/portfolio/target_balance_allocation":
            if method == "POST":
                self.posts.append(body)
                return 200, {}
            return 200, {"allocations": [], "resting_margin_reservation": "sum"}
        if path.startswith("/portfolio/orders/"):
            return 200, {"order": self.order_rows[path.rsplit("/", 1)[1]]}
        raise AssertionError(path)

    def create_order(self, body):
        self.sent.append(body)
        oid = f"o{len(self.sent)}"
        return 201, {"order": {"order_id": oid, "status": "resting"}}


def espn_get(start="2026-10-28T02:30Z"):
    return lambda url: {"events": [{"date": start, "competitions": [{"competitors": [
        {"homeAway": "home", "team": {"abbreviation": "LAL"}}, {"homeAway": "away", "team": {"abbreviation": "POR"}}]}]}]}


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    bet.set_root(tmp_path)
    monkeypatch.setattr(bet, "exec_night_spend", lambda now: 0.0)
    yield tmp_path


def args(**kw):
    d = dict(ticker=NBA, side="yes", units=None, contracts=None, price=None, tier="C", strategy="judgment",
             taker=False, expire=None, dry=False)
    d.update(kw)
    return Namespace(**d)


def ext(**kw):
    d = dict(venue="robinhood", market="X ML", side="yes", price=0.5, units=0.25, contracts=None, tier="C",
             strategy="judgment", result=None)
    d.update(kw)
    return Namespace(**d)


# ---------------------------------------------------------------- sizing
def test_size_maker_quadratic_no_fee():
    assert bet.size(0.25, None, 0.50, False, "quadratic") == (25, 12.5)


def test_size_taker_includes_fee_and_fits_units():
    n, cost = bet.size(0.5, None, 0.50, True, None)
    assert cost <= 25.0 and n == int(25 / (0.5 + 0.07 * 0.25))
    n2, cost2 = bet.size(None, n + 1, 0.50, True, None)
    assert cost2 > 25.0


def test_size_contracts_passthrough():
    assert bet.size(None, 10, 0.55, False, "quadratic") == (10, 5.5)


# ---------------------------------------------------------------- caps
def test_tier_per_bet_caps():
    assert bet.check_caps("C", 12.5, [], NOW) == []
    assert "tier C per-bet cap" in bet.check_caps("C", 12.51, [], NOW)[0]
    assert bet.check_caps("B", 25.0, [], NOW) == []
    assert bet.check_caps("B", 25.5, [], NOW)


def test_tier_a_night_cap_counts_exec_spend():
    assert bet.check_caps("A", 30, [], NOW, exec_spend=20) == []
    assert "night cap" in bet.check_caps("A", 30, [], NOW, exec_spend=20.01)[0]


def _settled(pnl, tier="C", when=NOW, stake=12.5):
    return {"id": str(pnl), "ts": bet._iso(when), "tier": tier, "status": "external", "stake_usd": stake,
            "settled": True, "result": "loss" if pnl < 0 else "win", "pnl_usd": pnl, "settled_ts": bet._iso(when)}


def test_daily_stop_uses_realized_plus_open_risk():
    bs = [_settled(-50, "B"), _settled(-50, "B"), _settled(-25, "C")]          # -2.5u today
    assert bet.check_caps("C", 12.5, bs, NOW) == []                            # worst -2.75u
    bs.append({"id": "o", "ts": bet._iso(NOW), "tier": "C", "status": "external", "stake_usd": 12.5})  # open 0.25u
    assert bet.check_caps("C", 12.5, bs, NOW) == []                            # worst exactly -3.0u: allowed
    bs.append({"id": "o2", "ts": bet._iso(NOW), "tier": "B", "status": "external", "stake_usd": 1.0})
    assert any("daily loss stop" in w for w in bet.check_caps("C", 12.5, bs, NOW))


def test_daily_stop_boundary_exactly_minus_3u_allowed():
    bs = [_settled(-125, "B")]                                                  # -2.5u
    assert bet.check_caps("B", 25.0, bs, NOW) == []                            # -3.0u exactly: allowed


def test_daily_resets_but_weekly_binds():
    y = NOW - timedelta(days=2)
    bs = [_settled(-150, "B", y), _settled(-140, "B", y)]                      # -5.8u two days ago
    why = bet.check_caps("B", 25.0, bs, NOW)
    assert not any("daily" in w for w in why) and any("weekly" in w for w in why)
    old = NOW - timedelta(days=8)
    assert bet.check_caps("B", 25.0, [_settled(-290, "B", old)], NOW) == []


def test_tier_a_losses_do_not_hit_manual_stops():
    assert bet.check_caps("C", 12.5, [_settled(-500, "A")], NOW) == []


def test_monthly_c_cap():
    last_month = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)
    bs = [_settled(-12.5, "C", NOW - timedelta(days=d)) for d in (8, 9, 10, 11, 12, 13, 14)]  # -1.75u this month
    assert bet.check_caps("C", 12.5, bs, NOW) == []                            # -2.0u exactly
    bs.append(_settled(-0.01, "C", NOW - timedelta(days=15)))
    assert any("monthly tier-C" in w for w in bet.check_caps("C", 12.5, bs, NOW))
    assert bet.check_caps("C", 12.5, [_settled(-90, "C", last_month)], NOW) == []


# ---------------------------------------------------------------- place
def test_place_maker_sizes_and_expires_at_espn_start(root):
    c = FakeKalshi()
    assert bet.cmd_place(args(units=0.25, price=0.55), c, NOW, espn_get()) == 0
    body = c.sent[0]
    assert body["count"] == "22" and body["price"] == "0.55" and body["post_only"] is True
    assert body["expiration_time"] == int(datetime(2026, 10, 28, 2, 30, tzinfo=timezone.utc).timestamp())
    b = next(iter(bet.bets().values()))
    assert b["tier"] == "C" and b["stake_usd"] == 12.1 and b["shard"] == 3 and b["status"] == "placed"


def test_place_no_side_sends_yes_price_complement():
    c = FakeKalshi()
    assert bet.cmd_place(args(side="no", units=0.25, price=0.44), c, NOW, espn_get()) == 0
    assert c.sent[0]["side"] == "ask" and c.sent[0]["price"] == "0.56"


def test_place_refuses_when_shard_lacks_cash(capsys):
    c = FakeKalshi(shards={0: 500.0, 1: 0.0, 2: 0.0, 3: 5.0})
    assert bet.cmd_place(args(units=0.25, price=0.55), c, NOW, espn_get()) == 1
    out = capsys.readouterr().out
    assert "REFUSED" in out and "shard 3" in out and not c.sent
    assert next(iter(bet.bets().values()))["status"] == "refused"


def test_shard_fallback_table_when_api_silent():
    assert bet.shard_of({}, "KXMLBGAME-26OCT28MILTB-TB") == (3, "table")
    assert bet.shard_of({}, "KXHIGHNY-26OCT28-T67") == (0, "table")
    assert bet.shard_of({"exchange_index": 0}, "KXNBAGAME-X") == (0, "api")       # API wins


def test_place_refuses_over_tier_cap_and_kill(root):
    c = FakeKalshi()
    assert bet.cmd_place(args(units=0.5, price=0.55, tier="C"), c, NOW, espn_get()) == 1
    assert bet.cmd_place(args(units=0.5, price=0.55, tier="B"), c, NOW, espn_get()) == 0
    (root / "KILL").write_text("x")
    assert bet.cmd_place(args(units=0.1, price=0.55), c, NOW, espn_get()) == 1
    assert len(c.sent) == 1


def test_place_refuses_crossing_maker_and_missing_start():
    c = FakeKalshi()
    assert bet.cmd_place(args(units=0.25, price=0.56), c, NOW, espn_get()) == 1          # at the ask
    assert bet.cmd_place(args(units=0.25, price=0.55), c, NOW, lambda u: {"events": []}) == 1  # no start
    assert bet.cmd_place(args(units=0.25, price=0.55, expire="2026-10-28T02:30:00Z"), c, NOW,
                         lambda u: {"events": []}) == 0
    assert bet.cmd_place(args(units=0.25, price=0.55), c, NOW + timedelta(hours=7), espn_get()) == 1  # started


def test_non_sports_maker_expires_at_close():
    c = FakeKalshi(exchange_index=0)
    assert bet.cmd_place(args(ticker=WX, side="no", units=0.2, price=0.3, tier="B"), c, NOW, espn_get()) == 0
    assert c.sent[0]["expiration_time"] == int(datetime(2026, 11, 10, 3, tzinfo=timezone.utc).timestamp())


def test_dry_sends_nothing():
    c = FakeKalshi()
    assert bet.cmd_place(args(units=0.25, price=0.55, dry=True), c, NOW, espn_get()) == 0
    assert not c.sent and not bet.bets()


# ---------------------------------------------------------------- external + settle + pnl
def test_log_external_over_cap_is_logged_with_breach(capsys):
    assert bet.cmd_log_external(ext(units=1, price=0.55), NOW) == 0
    b = next(iter(bet.bets().values()))
    assert b["stake_usd"] == 50 and b["contracts"] is None and b["cap_breach"]
    assert "CAP BREACH" in capsys.readouterr().out


def test_external_settle_win_and_loss():
    bet.cmd_log_external(ext(units=1, price=0.55, tier="C"), NOW)
    bid = next(iter(bet.bets()))
    assert bet._settle_one(bid, "win", NOW) == 0
    assert bet.bets()[bid]["pnl_usd"] == round(50 / 0.55 - 50, 2)
    assert bet._settle_one(bid, "loss", NOW) == 1                              # no double settle
    bet.cmd_log_external(ext(units=0.25, result="loss"), NOW)
    assert sorted(b["pnl_usd"] for b in bet.bets().values()) == [-12.5, 40.91]


def test_api_settle_from_order_fills():
    c = FakeKalshi()
    bet.cmd_place(args(units=0.25, price=0.55), c, NOW, espn_get())
    c.order_rows["o1"] = {"status": "executed", "fill_count_fp": "22.00", "maker_fees_dollars": "0"}
    c.results[NBA] = "yes"
    bet.cmd_settle(Namespace(id=None, result=None), c, NOW)
    b = next(iter(bet.bets().values()))
    assert b["result"] == "win" and b["pnl_usd"] == round(22 - 22 * 0.55, 2)


def test_api_settle_unfilled_expired_is_void():
    c = FakeKalshi()
    bet.cmd_place(args(units=0.25, price=0.55), c, NOW, espn_get())
    c.order_rows["o1"] = {"status": "canceled", "fill_count_fp": "0"}
    bet.cmd_settle(Namespace(id=None, result=None), c, NOW)
    b = next(iter(bet.bets().values()))
    assert b["result"] == "void-unfilled" and b["pnl_usd"] == 0 and bet.risk(b) == 0


def test_pnl_summary_and_legacy_rows(root):
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "manual_bets.jsonl").write_text(json.dumps(
        {"ts": "2026-10-26T16:00:00+00:00", "ticker": "KXNHLTOTAL-X", "side": "no", "price": 0.45, "count": 4,
         "max_cost": 1.8, "client_order_id": "manual-abcdef0123456789", "order_id": "z", "status": "placed"}) + "\n")
    bet.cmd_log_external(ext(units=1, price=0.55, venue="unknown", market="POR@LAL POR ML"), NOW)
    lines = "\n".join(bet.summarize(list(bet.bets().values()), NOW))
    assert "legacy-manual" in lines and "unknown" in lines and "monthly" not in lines
    assert "month  C realized +0.00u, open 1.00u" in lines


def test_allocate_dry_run_vs_confirm():
    c = FakeKalshi()
    assert bet.cmd_allocate(["0=70", "3=30"], False, c) == 0 and c.posts == []
    assert bet.cmd_allocate(["0=70", "3=30"], True, c) == 0
    assert c.posts == [{"allocations": [{"exchange_index": 0, "percent": 70}, {"exchange_index": 3, "percent": 30}]}]
    assert bet.cmd_allocate(["0=70", "3=20"], True, c) == 2


def test_parse_event():
    assert bet.parse_event(NBA) == ("KXNBAGAME", "2026-10-27", "PORLAL")
    assert bet.parse_event("KXMLBGAME-26OCT282000MILTB-TB") == ("KXMLBGAME", "2026-10-28", "MILTB")
