"""Live odds: Pinnacle (sharp reference) vs Kalshi game contracts and ESPN-listed books.

The idea with the most evidence behind it (Buchdahl's 88k-bet CLV study, Hubacek
2019): the Pinnacle price with its margin removed is the best available estimate
of the true probability. A price elsewhere that beats it by more than costs is a
+EV bet, and whether it ALSO beats Pinnacle's CLOSING price is how we find out,
fast, whether the edge is real (closing line value).

Sources, all read-only, no keys:
  pinnacle   guest.api.arcadia.pinnacle.com -- the public API pinnacle.com's own
             web app uses. Odds endpoints answer from the US; we do not bypass
             anything that refuses (its league-info endpoint is geo-blocked).
  kalshi     api.elections.kalshi.com -- official public market data.
  espn       sports.core.api.espn.com -- whatever books ESPN lists for a game.

Polling is one request per league per endpoint; keep the loop at >= 5 minutes.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

PIN = "https://guest.api.arcadia.pinnacle.com/0.1"
PIN_HEADERS = {"User-Agent": "Mozilla/5.0", "X-API-Key": "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"}  # public web-app key
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_FEE = 0.07  # taker fee coefficient: fee/contract ~= 0.07 * p * (1-p), rounded up per order

LEAGUES = {  # our name -> (pinnacle league id, kalshi game series, espn league key)
    "nfl": (889, "KXNFLGAME", "nfl"),
    "cfb": (880, "KXNCAAFGAME", "cfb"),
    "nba": (487, "KXNBAGAME", "nba"),
    "nhl": (1456, "KXNHLGAME", "nhl"),
    "mlb": (246, "KXMLBGAME", "mlb"),
    "wnba": (578, "KXWNBAGAME", "wnba"),
}

_s = requests.Session()


def _get(url, **kw):
    r = _s.get(url, timeout=30, **kw)
    r.raise_for_status()
    return r.json()


def am_to_dec(a: float) -> float:
    return 1 + (a / 100 if a > 0 else 100 / -a)


def devig(p1: float, p2: float) -> tuple[float, float]:
    """Multiplicative de-vig of two implied probabilities."""
    t = p1 + p2
    return p1 / t, p2 / t


def devig_power(p1: float, p2: float) -> tuple[float, float]:
    """Power de-vig: fair_i = p_i**k with k s.t. they sum to 1. Takes more margin off the
    longshot than multiplicative does (favourite-longshot bias), so it is the
    conservative choice when the bet is ON a longshot."""
    lo, hi = 1.0, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if p1 ** k + p2 ** k > 1:
            lo = k
        else:
            hi = k
    return p1 ** k, p2 ** k


# ---------------------------------------------------------------- pinnacle

def pinnacle(league: str) -> pd.DataFrame:
    """One row per (game, market, line): both sides' prices and the fair (no-vig) probability."""
    lid = LEAGUES[league][0]
    mu = _get(f"{PIN}/leagues/{lid}/matchups", headers=PIN_HEADERS)
    games = {}
    for m in mu:
        if m.get("type") != "matchup" or m.get("parentId") or m.get("units", "Regular") != "Regular":
            continue
        side = {p["alignment"]: p["name"] for p in m.get("participants", [])}
        if "home" in side and "away" in side:
            games[m["id"]] = {"start": pd.Timestamp(m["startTime"]), "home": side["home"], "away": side["away"],
                              "live": bool(m.get("isLive"))}
    rows = []
    for k in _get(f"{PIN}/leagues/{lid}/markets/straight", headers=PIN_HEADERS):
        g = games.get(k["matchupId"])
        if g is None or k.get("period") != 0 or k.get("status") != "open" or k.get("isAlternate"):
            continue
        pr = {p["designation"]: p for p in k["prices"]}
        typ = k.get("type")
        if typ == "moneyline" and {"home", "away"} <= set(pr):
            a, b = "home", "away"
        elif typ == "spread" and {"home", "away"} <= set(pr):
            a, b = "home", "away"
        elif typ in ("total", "team_total") and {"over", "under"} <= set(pr):
            a, b = "over", "under"
        else:
            continue
        d1, d2 = am_to_dec(pr[a]["price"]), am_to_dec(pr[b]["price"])
        f1, f2 = devig(1 / d1, 1 / d2)
        q1, q2 = devig_power(1 / d1, 1 / d2)
        rows.append({"league": league, "pin_id": k["matchupId"], **g, "market": typ,
                     "side_for": k.get("side") if typ == "team_total" else None,
                     "line": pr[a].get("points"), "side1": a, "side2": b,
                     "price1": pr[a]["price"], "price2": pr[b]["price"], "fair1": f1, "fair2": f2,
                     "pfair1": q1, "pfair2": q2,
                     "hold": 1 / d1 + 1 / d2 - 1,
                     "max_risk": next((l["amount"] for l in k.get("limits", []) if l["type"] == "maxRiskStake"), None)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- kalshi

def kalshi(league: str) -> pd.DataFrame:
    series = LEAGUES[league][1]
    out, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        d = _get(f"{KALSHI}/markets", params=params)
        for m in d.get("markets", []):
            f = lambda k: float(m[k]) if m.get(k) not in (None, "") else np.nan
            out.append({"ticker": m["ticker"], "event": m["event_ticker"], "team": m.get("yes_sub_title"),
                        "yes_bid": f("yes_bid_dollars"), "yes_ask": f("yes_ask_dollars"),
                        "no_bid": f("no_bid_dollars"), "no_ask": f("no_ask_dollars"),
                        "volume": f("volume_fp") if m.get("volume_fp") else m.get("volume"),
                        "expected_end": m.get("expected_expiration_time")})
        cursor = d.get("cursor")
        if not cursor or not d.get("markets"):
            break
    return pd.DataFrame(out)


def kalshi_fee(p: float) -> float:
    return KALSHI_FEE * p * (1 - p)


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", "", str(s).lower().replace("st.", "state").replace("&", "and"))
    return re.sub(r"\s+", " ", s).strip()


def match_kalshi(pin: pd.DataFrame, kal: pd.DataFrame) -> pd.DataFrame:
    """Attach the Pinnacle moneyline game to each Kalshi team market.

    Kalshi names teams by city prefix ("Los Angeles C", "New York J"); a Kalshi event
    (both teams) matches the one Pinnacle game whose two participants each START with
    one of the event's two names, starting within 2 days of the Kalshi expected end.
    """
    ml = pin[pin["market"] == "moneyline"].copy()
    ml["et_date"] = ml["start"].dt.tz_convert("America/New_York").dt.date
    rows = []
    for ev, g in kal.groupby("event"):
        if len(g) != 2:
            continue
        names = [_norm(t) for t in g["team"]]
        # game date is in the event ticker: KXNFLGAME-26OCT19WASSF -> 2026-10-19 (US date)
        dm = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", ev)
        day = pd.to_datetime(f"20{dm.group(1)}-{dm.group(2)}-{dm.group(3)}", format="%Y-%b-%d").date() if dm else None
        cand = []
        for _, p in ml.iterrows():
            h, a = _norm(p["home"]), _norm(p["away"])
            if day is None or p["et_date"] != day:
                continue
            hit = {n: ("home" if h.startswith(n) else "away" if a.startswith(n) else None) for n in names}
            if set(hit.values()) == {"home", "away"}:
                cand.append((p, hit))
        if len(cand) != 1:
            continue
        p, hit = cand[0]
        for (_, k), n in zip(g.iterrows(), names):
            home = hit[n] == "home"
            rows.append({**k.to_dict(), "pin_id": p["pin_id"], "start": p["start"], "home": p["home"],
                         "away": p["away"], "fair": p["fair1"] if home else p["fair2"],
                         "pfair": p["pfair1"] if home else p["pfair2"],
                         "pin_price": p["price1"] if home else p["price2"],
                         "pin_hold": p["hold"], "pin_max_risk": p["max_risk"]})
    return pd.DataFrame(rows)


def kalshi_edges(m: pd.DataFrame) -> pd.DataFrame:
    """EV per $1 risked of buying YES / NO at the ask, after the taker fee.

    ev_* uses the CONSERVATIVE fair value: of the multiplicative and power de-vigs,
    whichever makes the bet look worse.
    """
    if m.empty:
        return m
    m = m.copy()
    lo, hi = np.minimum(m["fair"], m["pfair"]), np.maximum(m["fair"], m["pfair"])
    m["fee_yes"] = kalshi_fee(m["yes_ask"])
    m["ev_yes"] = (lo - m["yes_ask"] - m["fee_yes"]) / (m["yes_ask"] + m["fee_yes"])
    m["fee_no"] = kalshi_fee(m["no_ask"])
    m["ev_no"] = ((1 - hi) - m["no_ask"] - m["fee_no"]) / (m["no_ask"] + m["fee_no"])
    return m


def now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))
