"""venues/forecastex_client.py against a mocked HTTP layer: no network, no session, no real orders.

Covers: ticker parse/make, public discovery + book (complement fill, nan sizes), session snapshot book with sizes, gating
(no --live / no session / no account / paper-vs-live account mismatch / KILL), caps, post-only cross guard, order body
(BUY only, LMT, conid of the bought side), write-ahead journal + one decision per tag, reply prompt not auto-confirmed,
ambiguous POST never re-sent, cancel path, Venue protocol.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from venues import get_venue  # noqa: E402
from venues.base import TradingDisabled, Venue  # noqa: E402
from venues import forecastex_client as fxc  # noqa: E402

MKT = "UHLGA_102926_63"
TREE = {"categories": {"g1": {"name": "NYC", "markets": [
    {"symbol": "UHLGA", "exchange": "FORECASTX", "conid": 853400786, "product_conid": 853400791},
    {"symbol": "ULLGA", "exchange": "FORECASTX", "conid": 1, "product_conid": 2}]}}}


def cc(strike, by=None, ay=None, bn=None, an=None):
    d = {"strike": strike, "expiration": "20261030", "measured_period": "Oct29'26 11:59 PM ET", "open_interest": 5,
         "contracts": [{"conid_yes": 1000 + int(strike), "conid_no": 2000 + int(strike), "last_trade_time": 1}]}
    for k, v in (("bid_yes", by), ("ask_yes", ay), ("bid_no", bn), ("ask_no", an)):
        if v is not None:
            d[k] = v
    return d


class Fake:
    def __init__(self, authenticated=False):
        self.calls = []
        self.auth = authenticated
        self.order_reply = [{"order_id": "777", "order_status": "Submitted"}]
        self.raise_on_post = False
        self.lst = {"complex_contracts": [cc(63, by=0.04, ay=0.10, bn=0.90, an=0.96), cc(64, ay=0.10, bn=0.90),
                                          cc(60, by=0.83, an=0.17)]}

    def __call__(self, method, url, headers, params, data, timeout):
        u = urlparse(url)
        self.calls.append({"method": method, "url": url, "path": u.path, "params": params,
                           "body": json.loads(data) if data else None})
        p = u.path
        if p.endswith("/category/tree"):
            return 200, TREE
        if p.endswith("/prediction/contract/list"):
            return 200, self.lst
        if p.endswith("/iserver/auth/status"):
            return 200, {"authenticated": self.auth}
        if p.endswith("/iserver/marketdata/snapshot"):
            return 200, [{"conid": 1063, "84": "0.05", "86": "0.09", "88": "120", "85": "1.2K"},
                         {"conid": 2063, "84": "0.91", "86": "0.95", "88": "1,200", "85": "120"}]
        if method == "POST" and "/orders" in p:
            if self.raise_on_post:
                raise fxc.TransportError("timeout")
            return 200, self.order_reply
        if method == "POST" and "/iserver/reply/" in p:
            return 200, [{"order_id": "778", "order_status": "Submitted"}]
        if method == "DELETE":
            return 200, {"msg": "Request was submitted", "order_id": "777"}
        if p.endswith("/positions/0"):
            return 200, [{"conid": 2063, "position": 3, "listingExchange": "FORECASTX"}]
        return 404, {"error": "not mocked"}


def client(tmp_path, fake=None, env=None, live=False):
    env = {"IBKR_ACCOUNT_ID": "DU1234567"} if env is None else env
    return fxc.ForecastEx(env=env, live=live, transport=fake or Fake(), sleep=lambda s: None, deploy=tmp_path,
                          journal=tmp_path / "orders.jsonl")


def posts(f):
    return [c for c in f.calls if c["method"] == "POST" and "/orders" in c["path"]]


def test_ticker_roundtrip():
    assert fxc.parse_ticker(MKT) == ("LGA", date(2026, 10, 29), 63.0)
    assert fxc.make_ticker("LGA", date(2026, 10, 29), 63.0) == MKT
    assert fxc.event_day({"measured_period": "Oct29'26 11:59 PM ET"}) == date(2026, 10, 29)
    assert fxc.event_day({"time_specifier": "2026.10.28"}) == date(2026, 10, 28)
    with pytest.raises(ValueError):
        fxc.parse_ticker("KXHIGHNY-26OCT29-B63")


def test_event_day_central_station_uses_question_not_measured_period():
    # 2026-10-28 live feed: a Central station's Oct-28 contract carries measured_period "Oct29'26 12:59 AM ET"
    c = {"measured_period": "Oct29'26 12:59 AM ET", "expiration": "20261029",
         "contracts": [{"question": "Will the highest temperature in San Antonio (SAT)(KSAT) exceed 80 F on "
                                    "October 28, 2026? (maximum) (max) (hottest)?"}]}
    assert fxc.event_day(c) == date(2026, 10, 28)
    pac = {"measured_period": "Oct28'26 2:59 AM ET",
           "contracts": [{"question": "Will ... (KLAX) exceed 68 F on October 28, 2026? (maximum)?"}]}
    assert fxc.event_day(pac) == date(2026, 10, 28)


def test_protocol_and_registry(tmp_path):
    assert isinstance(client(tmp_path), Venue)
    assert get_venue("forecastex", env={}, transport=Fake(), deploy=tmp_path).name == "forecastex"


def test_public_book_complement_and_nan_size(tmp_path):
    c = client(tmp_path)
    b = c.book(MKT)
    assert b.best("yes") == (0.04, 0.10) and b.best("no") == (0.90, 0.96)
    assert math.isnan(b.bids[0][1])
    b64 = c.book("UHLGA_102926_64")           # YES bid missing, NO ask missing -> only YES ask (from feed)
    assert b64.best("yes") == (None, 0.10) and b64.best("no") == (0.90, None)
    b60 = c.book("UHLGA_102926_60")           # YES ask missing but NO bid missing too -> YES bid only
    assert b60.best("yes") == (0.83, None)
    assert c.book("UHLGA_102926_99") is None


def test_session_book_has_sizes(tmp_path):
    f = Fake(authenticated=True)
    b = client(tmp_path, f).book(MKT)
    assert b.best("yes") == (0.05, 0.09)
    assert b.bids[0][1] == 120 and b.asks[0][1] == 1200


@pytest.mark.parametrize("env,live,auth,msg", [
    ({"IBKR_ACCOUNT_ID": "DU1"}, False, True, "--live"),
    ({"IBKR_ACCOUNT_ID": "DU1"}, True, False, "session"),
    ({}, True, True, "IBKR_ACCOUNT_ID"),
    ({"IBKR_ACCOUNT_ID": "U9999999"}, True, True, "not a paper"),
    ({"IBKR_ACCOUNT_ID": "U9999999", "IBKR_MODE": "live"}, True, True, "FX_ALLOW_LIVE_ACCOUNT"),
    ({"IBKR_ACCOUNT_ID": "DU1", "IBKR_MODE": "live", "FX_ALLOW_LIVE_ACCOUNT": "1"}, True, True, "paper account"),
])
def test_gating(tmp_path, env, live, auth, msg):
    f = Fake(authenticated=auth)
    with pytest.raises(TradingDisabled, match=msg):
        client(tmp_path, f, env=env, live=live).place_limit(MKT, "no", 0.93, 1)
    assert not posts(f)


def test_kill_file(tmp_path):
    (tmp_path / "KILL").write_text("")
    f = Fake(authenticated=True)
    with pytest.raises(TradingDisabled, match="KILL"):
        client(tmp_path, f, live=True).place_limit(MKT, "no", 0.93, 1)
    assert not posts(f)


def test_paper_order_body_and_journal(tmp_path):
    f = Fake(authenticated=True)
    c = client(tmp_path, f, live=True)
    r = c.place_limit(MKT, "no", 0.93, 2, post_only=True, tag="fx-1")
    assert r.status == "placed" and r.order_id == "777"
    [p] = posts(f)
    assert p["path"].endswith("/iserver/account/DU1234567/orders")
    assert p["body"] == {"orders": [{"conid": 2063, "orderType": "LMT", "side": "BUY", "price": 0.93, "quantity": 2,
                                     "tif": "DAY", "cOID": "fx-1"}]}
    ev = [json.loads(l)["event"] for l in (tmp_path / "orders.jsonl").read_text().splitlines()]
    assert ev == ["intent", "response"]
    again = c.place_limit(MKT, "no", 0.93, 2, tag="fx-1")
    assert again.status == "rejected" and "already used" in again.detail and len(posts(f)) == 1


def test_post_only_cross_and_caps(tmp_path):
    f = Fake(authenticated=True)
    c = client(tmp_path, f, live=True)
    assert "would cross" in c.place_limit(MKT, "no", 0.95, 1).detail      # session NO ask 0.95
    assert c.place_limit(MKT, "no", 0.95, 1, post_only=False).status == "placed"
    assert "FX_MAX_CONTRACTS" in c.place_limit(MKT, "no", 0.10, 11).detail
    assert "FX_MAX_ORDER_USD" in c.place_limit(MKT, "no", 0.90, 6).detail
    assert "cent" in c.place_limit(MKT, "no", 0.935, 1).detail
    c2 = client(tmp_path / "x", Fake(authenticated=True), env={"IBKR_ACCOUNT_ID": "DU1", "FX_MAX_DAILY_USD": "1"}, live=True)
    assert "daily cap" in c2.place_limit(MKT, "no", 0.90, 2).detail
    assert len(posts(f)) == 1


def test_reply_prompt_not_auto_confirmed(tmp_path):
    f = Fake(authenticated=True)
    f.order_reply = [{"id": "abc", "message": ["You are about to buy an event contract..."]}]
    r = client(tmp_path, f, live=True).place_limit(MKT, "no", 0.93, 1)
    assert r.status == "rejected" and "FX_AUTO_CONFIRM" in r.detail
    assert not [c for c in f.calls if "/iserver/reply/" in c["path"]]
    f2 = Fake(authenticated=True)
    f2.order_reply = f.order_reply
    r2 = client(tmp_path / "y", f2, env={"IBKR_ACCOUNT_ID": "DU1", "FX_AUTO_CONFIRM": "1"}, live=True).place_limit(MKT, "no", 0.93, 1)
    assert r2.status == "placed" and r2.order_id == "778"


def test_ambiguous_post_not_resent(tmp_path):
    f = Fake(authenticated=True)
    f.raise_on_post = True
    r = client(tmp_path, f, live=True).place_limit(MKT, "no", 0.93, 1)
    assert r.status == "ambiguous" and len(posts(f)) == 1


def test_dry_needs_no_session_and_sends_nothing(tmp_path):
    f = Fake(authenticated=False)
    r = client(tmp_path, f, live=False).place_limit(MKT, "yes", 0.05, 1, dry=True)
    assert r.status == "dry" and json.loads(r.detail)["orders"][0]["conid"] == 1063 and not posts(f)


def test_cancel_and_positions(tmp_path):
    f = Fake(authenticated=True)
    c = client(tmp_path, f, live=True)
    assert c.cancel("777", MKT)
    assert [x for x in f.calls if x["method"] == "DELETE"][0]["path"].endswith("/iserver/account/DU1234567/order/777")
    assert c.positions(MKT)[0]["position"] == 3
    with pytest.raises(TradingDisabled):
        client(tmp_path, Fake(authenticated=True), live=False).cancel("777", MKT)
