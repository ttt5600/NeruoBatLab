"""venues/polymarket_us_client.py against a mocked HTTP layer: no network, no real key, no real orders.

Covers: read-only without keys, live gating (no --live / no key / bad perms / KILL), request signing (verifiable with the
public key, path without query), order body (intent, post-only, AUTOMATIC, GTD, NO price convention), post-only cross
guard, tick/qty/caps, write-ahead journal + one decision per tag, ambiguous POST never re-sent, retry on 429 for GETs only.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from venues import get_venue  # noqa: E402
from venues.base import Book, TradingDisabled, Venue  # noqa: E402
from venues import polymarket_us_client as pmc  # noqa: E402

NOW = datetime(2026, 10, 27, 23, 0, tzinfo=timezone.utc)
SLUG = "tc-temp-nychigh-2026-10-28-gte70lt71f"


class Fake:
    """Routes (method, path) like the Polymarket US Retail API; records every call."""

    def __init__(self):
        self.calls = []
        self.post_status = 200
        self.raise_on_post = False
        self.get_429 = 0
        self.book = {"marketData": {"bids": [{"px": {"value": "0.03"}, "qty": "500"}],
                                    "offers": [{"px": {"value": "0.05"}, "qty": "300"}], "state": "MARKET_STATE_OPEN"}}
        self.market = {"market": {"slug": SLUG, "active": True, "closed": False, "orderPriceMinTickSize": 0.01,
                                  "minimumTradeQty": 1, "feeCoefficient": "0.0695"}}

    def __call__(self, method, url, headers, params, data, timeout):
        u = urlparse(url)
        self.calls.append({"method": method, "host": u.netloc, "path": u.path, "headers": headers, "params": params,
                           "body": json.loads(data) if data else None})
        if method == "GET" and self.get_429 > 0:
            self.get_429 -= 1
            return 429, {"message": "Too Many Requests"}
        if u.path.endswith("/book"):
            return 200, self.book
        if u.path.startswith("/v1/market/slug/"):
            return 200, self.market
        if u.path == "/v1/markets":
            return 200, {"markets": [{"slug": SLUG}]}
        if u.path == "/v1/account/balances":
            return 200, {"balances": [{"currentBalance": 100, "buyingPower": 100}]}
        if u.path == "/v1/orders/open":
            return 200, {"orders": [{"id": "o1", "marketSlug": SLUG}]}
        if u.path == "/v1/portfolio/positions":
            return 200, {"positions": {SLUG: {"netPosition": "-3"}}}
        if u.path == "/v1/portfolio/activities":
            return 200, {"activities": [{"type": "ACTIVITY_TYPE_TRADE", "trade": {"id": "t1", "price": "0.04"}}]}
        if method == "POST" and u.path == "/v1/orders":
            if self.raise_on_post:
                raise pmc.TransportError("connection reset")
            return self.post_status, ({"id": "ord-1"} if self.post_status == 200 else {"message": "bad"})
        if u.path.endswith("/cancel"):
            return 200, {}
        return 404, {}

    def posts(self):
        return [c for c in self.calls if c["method"] == "POST" and c["path"] == "/v1/orders"]


@pytest.fixture
def keyed(tmp_path):
    sk = ed25519.Ed25519PrivateKey.generate()
    seed = sk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    pub = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    kf = tmp_path / "pmus.secret"
    kf.write_text(base64.b64encode(seed + pub).decode())  # 64-byte secret, as the portal may issue
    os.chmod(kf, 0o600)
    env = {"POLYMARKET_US_KEY_ID": "kid-123", "POLYMARKET_US_SECRET_PATH": str(kf)}
    return env, sk.public_key(), tmp_path


def client(env, tmp_path, live=True, fake=None):
    fake = fake or Fake()
    c = pmc.PolymarketUS(env=env, live=live, transport=fake, sleep=lambda s: None, deploy=tmp_path,
                         journal=tmp_path / "data" / "pmus_orders.jsonl")
    return c, fake


def test_interface_and_registry(tmp_path):
    c = get_venue("polymarket_us", env={}, deploy=tmp_path)
    assert isinstance(c, Venue) and c.name == "polymarket_us"


def test_read_only_without_keys(tmp_path):
    c, f = client({}, tmp_path, live=False)
    b = c.book(SLUG)
    assert b.best("yes") == (0.03, 0.05) and b.best("no") == (0.95, 0.97)
    assert b.depth("no") == 500 and b.depth("yes") == 300
    assert all(x["host"] == "gateway.polymarket.us" for x in f.calls)
    assert all("X-PM-Signature" not in x["headers"] for x in f.calls)
    with pytest.raises(TradingDisabled):
        c.balance()


def test_place_without_live_is_dry_and_sends_nothing(keyed):
    env, _, tmp = keyed
    c, f = client(env, tmp, live=False)
    r = c.place_limit(SLUG, "no", 0.95, 3, now=NOW)
    assert r.status == "dry" and not f.posts()
    with pytest.raises(TradingDisabled):
        c.cancel("o1", SLUG)


def test_live_without_key_refuses(tmp_path):
    c, f = client({}, tmp_path, live=True)
    with pytest.raises(TradingDisabled):
        c.place_limit(SLUG, "no", 0.95, 3, now=NOW)
    assert not f.posts()


def test_kill_file_and_perms(keyed):
    env, _, tmp = keyed
    c, f = client(env, tmp)
    (tmp / "KILL").write_text("x")
    with pytest.raises(TradingDisabled, match="KILL"):
        c.place_limit(SLUG, "no", 0.95, 3, now=NOW)
    (tmp / "KILL").unlink()
    os.chmod(env["POLYMARKET_US_SECRET_PATH"], 0o644)
    with pytest.raises(TradingDisabled, match="chmod"):
        c.place_limit(SLUG, "no", 0.95, 3, now=NOW)
    assert not f.posts()


def test_signature_verifies_and_excludes_query(keyed):
    env, pub, tmp = keyed
    c, f = client(env, tmp)
    c.open_orders(SLUG)
    call = [x for x in f.calls if x["path"] == "/v1/orders/open"][0]
    h = call["headers"]
    assert call["host"] == "api.polymarket.us" and h["X-PM-Access-Key"] == "kid-123"
    pub.verify(base64.b64decode(h["X-PM-Signature"]), f"{h['X-PM-Timestamp']}GET/v1/orders/open".encode())
    assert call["params"] == {"slugs": [SLUG]}


def test_order_body_no_side_post_only(keyed):
    env, _, tmp = keyed
    c, f = client(env, tmp)
    r = c.place_limit(SLUG, "no", 0.95, 3, expire=NOW + timedelta(hours=10), now=NOW, tag="t1")
    assert r.status == "placed" and r.order_id == "ord-1"
    b = f.posts()[0]["body"]
    assert b["intent"] == "ORDER_INTENT_BUY_SHORT" and b["participateDontInitiate"] is True
    assert b["manualOrderIndicator"] == "MANUAL_ORDER_INDICATOR_AUTOMATIC"
    assert b["price"] == {"value": "0.05", "currency": "USD"}          # default convention: YES-side price
    assert b["tif"] == "TIME_IN_FORCE_GOOD_TILL_DATE" and b["goodTillTime"] == "2026-10-28T09:00:00Z"
    env2 = dict(env, PMUS_PRICE_CONVENTION="nominal")
    c2, f2 = client(env2, tmp / "x")
    c2.place_limit(SLUG, "no", 0.95, 3, now=NOW)
    assert f2.posts()[0]["body"]["price"]["value"] == "0.95"


def test_post_only_cross_tick_and_caps(keyed):
    env, _, tmp = keyed
    c, f = client(env, tmp)
    assert "cross" in c.place_limit(SLUG, "no", 0.97, 1, now=NOW).detail       # NO ask is 0.97
    assert "tick" in c.place_limit(SLUG, "no", 0.955, 1, now=NOW).detail
    assert "PMUS_MAX_ORDER_USD" in c.place_limit(SLUG, "no", 0.95, 6, now=NOW).detail   # $5.70 > $5
    assert "PMUS_MAX_CONTRACTS" in c.place_limit(SLUG, "yes", 0.04, 11, now=NOW).detail
    assert not f.posts()
    env3 = dict(env, PMUS_MAX_DAILY_USD="5", PMUS_MAX_ORDER_USD="5")
    c3, f3 = client(env3, tmp / "d")
    assert c3.place_limit(SLUG, "no", 0.95, 4, now=NOW).status == "placed"      # $3.80
    assert "daily cap" in c3.place_limit(SLUG, "no", 0.95, 2, now=NOW).detail    # +$1.90 > $5
    assert len(f3.posts()) == 1


def test_one_decision_per_tag_and_write_ahead(keyed):
    env, _, tmp = keyed
    c, f = client(env, tmp)
    assert c.place_limit(SLUG, "no", 0.95, 1, now=NOW, tag="E1-x").status == "placed"
    r2 = c.place_limit(SLUG, "no", 0.95, 1, now=NOW, tag="E1-x")
    assert r2.status == "rejected" and "already" in r2.detail and len(f.posts()) == 1
    rows = [json.loads(x) for x in c.journal.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["intent", "result"]


def test_ambiguous_post_not_resent(keyed):
    env, _, tmp = keyed
    c, f = client(env, tmp)
    f.raise_on_post = True
    r = c.place_limit(SLUG, "no", 0.95, 1, now=NOW, tag="amb")
    assert r.status == "ambiguous" and len(f.posts()) == 1
    f.raise_on_post, f.post_status = False, 503
    r = c.place_limit(SLUG, "no", 0.95, 1, now=NOW, tag="amb2")
    assert r.status == "ambiguous" and len(f.posts()) == 2      # 503 on an order POST: sent once, not retried


def test_get_retries_429(keyed):
    env, _, tmp = keyed
    c, f = client(env, tmp)
    f.get_429 = 2
    assert c.balance()["balances"][0]["buyingPower"] == 100
    assert len([x for x in f.calls if x["path"] == "/v1/account/balances"]) == 3


def test_account_reads(keyed):
    env, _, tmp = keyed
    c, f = client(env, tmp, live=False)   # account reads need the key, not --live
    assert c.positions()[0]["slug"] == SLUG
    assert c.fills()[0]["id"] == "t1"
    p = [x for x in f.calls if x["path"] == "/v1/portfolio/activities"][0]["params"]
    assert p["types"] == ["ACTIVITY_TYPE_TRADE"]


def test_fee():
    c = pmc.PolymarketUS(env={}, transport=Fake(), sleep=lambda s: None)
    assert c.fee(0.5, 100, taker=True) == pytest.approx(1.7375)
    assert c.fee(0.5, 100, taker=False) == 0.0
