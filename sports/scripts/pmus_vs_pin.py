"""Polymarket US (CFTC exchange, available in CA) game-winner prices vs Pinnacle fair. DESCRIPTIVE snapshot, not a trial.

    python scripts/pmus_vs_pin.py [nba nfl nhl cfb ...]

Data: gateway.polymarket.us public endpoints (documented, no key): /v2/leagues/{slug}/events and
/v1/markets/{slug}/bbo. One instrument per game: LONG = one team (bestAsk), the other team = SHORT
(cost 1 - bestBid). Taker fee 0.0695 * p * (1-p) per contract (docs.polymarket.us/fees, eff. 2026-10-19).
Fair = Pinnacle moneyline, conservative of power / multiplicative de-vig for the side. Pre-game only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

from sports.live import pinnacle

GW = "https://gateway.polymarket.us"
THETA = 0.0695


def n(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def pm_games(league: str) -> list[dict]:
    d = requests.get(f"{GW}/v2/leagues/{league}/events", params={"limit": 200}, timeout=30).json()
    out = []
    for e in d.get("events", []):
        for m in e.get("markets", []):
            if m.get("marketType") != "moneyline" or m.get("closed") or not m.get("active"):
                continue
            sides = m.get("marketSides", [])
            lo = next((s for s in sides if s.get("long")), None)
            sh = next((s for s in sides if not s.get("long")), None)
            if not lo or not sh:
                continue
            out.append({"slug": m["slug"], "start": pd.Timestamp(m.get("gameStartTime") or e.get("startTime")),
                        "long": lo.get("team") or {"name": lo["description"]},
                        "short": sh.get("team") or {"name": sh["description"]}})
    return out


def bbo(slug: str):
    import time
    for i in range(4):
        time.sleep(0.15 * (1 + 4 * i))  # stay far under the 20 req/s public limit
        r = requests.get(f"{GW}/v1/markets/{slug}/bbo", timeout=30)
        if r.ok and r.text.strip():
            break
    else:
        return None, None, None, None, f"http {r.status_code}"
    md = r.json().get("marketData", {})
    f = lambda k: float(md[k]["value"]) if md.get(k) else None
    return f("bestBid"), f("bestAsk"), md.get("bidDepth"), md.get("askDepth"), md.get("state")


def team_hit(team: dict, pin_name: str) -> bool:
    p = n(pin_name)
    keys = [n(team.get("name")), n(f"{team.get('safeName', '')}{team.get('name', '')}"), n(team.get("alias"))]
    return any(k and (p.endswith(k) or k == p) for k in keys)


def main(leagues):
    rows = []
    for lg in leagues:
        try:
            pin = pinnacle(lg)
        except Exception as e:
            print(lg, "pinnacle failed", e)
            continue
        ml = pin[(pin.market == "moneyline") & (~pin.live)]
        games = pm_games(lg)
        print(f"{lg}: polymarket US moneylines {len(games)}, pinnacle moneylines {len(ml)}")
        for g in games:
            cand = ml[(ml.start - g["start"]).abs() <= pd.Timedelta("3h")]
            hit = None
            for _, p in cand.iterrows():
                if team_hit(g["long"], p.home) and team_hit(g["short"], p.away):
                    hit = (p, "home")
                elif team_hit(g["long"], p.away) and team_hit(g["short"], p.home):
                    hit = (p, "away")
            if hit is None:
                continue
            p, lside = hit
            bid, ask, bd, ad, st = bbo(g["slug"])
            if st != "MARKET_STATE_OPEN":
                continue
            fl = (p.fair1, p.pfair1) if lside == "home" else (p.fair2, p.pfair2)
            for side, cost, fair in (("long", ask, min(fl)), ("short", None if bid is None else 1 - bid, min(1 - fl[0], 1 - fl[1]))):
                if cost is None or not 0 < cost < 1:
                    continue
                c = cost + THETA * cost * (1 - cost)
                rows.append({"league": lg, "game": f"{p.away} @ {p.home}", "start": g["start"],
                             "team": (g["long"] if side == "long" else g["short"]).get("name"), "cost": cost,
                             "fair": fair, "ev": fair / c - 1, "mid": None if bid is None or ask is None else (bid + ask) / 2,
                             "fair_mid": min(fl) if side == "long" else None})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no matched quotes")
        return
    lg_mid = df.dropna(subset=["mid"])
    print(f"matched quotes {len(df)} ({df.game.nunique()} games); |PM mid - Pinnacle fair| median "
          f"{(lg_mid.mid - lg_mid.fair_mid).abs().median()*100:.2f}c")
    print(f"EV after taker fee: median {df.ev.median()*100:+.2f}%, max {df.ev.max()*100:+.2f}%; "
          f">= 2%: {(df.ev >= .02).sum()}, >= 1%: {(df.ev >= .01).sum()}, > 0: {(df.ev > 0).sum()}")
    print(df.sort_values("ev", ascending=False).head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or ["nba", "nfl", "nhl", "cfb"])
