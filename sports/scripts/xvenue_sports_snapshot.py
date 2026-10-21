"""One-off snapshot: Kalshi vs Polymarket game-winner prices for upcoming NFL/NBA games (descriptive, read-only).

    python scripts/xvenue_sports_snapshot.py [nfl|nba]

Polymarket data is the public gamma API (events?series_id from /sports: NFL 12185, NBA 10345); it covers the
international book. Any trade must go through the US-regulated Polymarket product, which may not list every game.
Matching: Polymarket nickname -> Kalshi ticker code (tables below) + Kalshi event date within 1 day of the Polymarket
game time. Two-venue arb: team A YES on one venue + team B YES on the other at the asks, plus fees
(Kalshi 0.07*p*(1-p); Polymarket assumed 0.01/contract, UNVERIFIED).
"""
from __future__ import annotations

import json
import sys

import pandas as pd
import requests

K = "https://api.elections.kalshi.com/trade-api/v2"
G = "https://gamma-api.polymarket.com/events"
SER = {"nfl": "KXNFLGAME", "nba": "KXNBAGAME"}
PSER = {"nfl": 12185, "nba": 10345}
NFL = {"Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF", "Panthers": "CAR", "Bears": "CHI", "Bengals": "CIN",
       "Browns": "CLE", "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB", "Texans": "HOU", "Colts": "IND",
       "Jaguars": "JAX", "Chiefs": "KC", "Rams": "LA", "Chargers": "LAC", "Raiders": "LV", "Dolphins": "MIA", "Vikings": "MIN",
       "Patriots": "NE", "Saints": "NO", "Giants": "NYG", "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "Seahawks": "SEA",
       "49ers": "SF", "Buccaneers": "TB", "Titans": "TEN", "Commanders": "WAS"}
NBA = {"Hawks": "ATL", "Celtics": "BOS", "Nets": "BKN", "Hornets": "CHA", "Bulls": "CHI", "Cavaliers": "CLE", "Mavericks": "DAL",
       "Nuggets": "DEN", "Pistons": "DET", "Warriors": "GSW", "Rockets": "HOU", "Pacers": "IND", "Clippers": "LAC", "Lakers": "LAL",
       "Grizzlies": "MEM", "Heat": "MIA", "Bucks": "MIL", "Timberwolves": "MIN", "Pelicans": "NOP", "Knicks": "NYK", "Thunder": "OKC",
       "Magic": "ORL", "76ers": "PHI", "Suns": "PHX", "Trail Blazers": "POR", "Kings": "SAC", "Spurs": "SAS", "Raptors": "TOR",
       "Jazz": "UTA", "Wizards": "WAS"}


def kfee(p):
    return 0.07 * p * (1 - p)


def main() -> int:
    lg = sys.argv[1] if len(sys.argv) > 1 else "nfl"
    codes = NFL if lg == "nfl" else NBA
    km = requests.get(f"{K}/markets", params={"series_ticker": SER[lg], "status": "open", "limit": 1000}, timeout=30).json()["markets"]
    kal = {}
    for m in km:
        kal.setdefault(m["event_ticker"], {})[m["ticker"].rsplit("-", 1)[-1]] = m
    ev = requests.get(G, params={"series_id": PSER[lg], "closed": "false", "limit": 300}, timeout=30).json()
    rows, unmatched = [], 0
    now = pd.Timestamp.now(tz="UTC")
    for e in ev:
        t = pd.to_datetime(e.get("endDate"), utc=True, errors="coerce")
        if t is pd.NaT or t < now:
            continue
        ml = [m for m in e.get("markets", []) if m.get("sportsMarketType") == "moneyline"]
        if not ml:
            continue
        m = ml[0]
        outs = json.loads(m["outcomes"]) if isinstance(m.get("outcomes"), str) else m.get("outcomes", [])
        if len(outs) != 2 or m.get("bestBid") is None or m.get("bestAsk") is None:
            continue
        ca, cb = codes.get(outs[0]), codes.get(outs[1])
        hit = None
        for ek, ms in kal.items():
            if ca in ms and cb in ms:
                kd = pd.to_datetime(ek.split("-")[1][:7], format="%y%b%d", errors="coerce")
                if kd is not pd.NaT and abs((kd - t.tz_convert("America/New_York").tz_localize(None).normalize()).days) <= 1:
                    hit = ms
                    break
        if hit is None:
            unmatched += 1
            continue
        a, b = hit[ca], hit[cb]
        ka, kb = float(a["yes_ask_dollars"]), float(b["yes_ask_dollars"])
        pbid, pask = float(m["bestBid"]), float(m["bestAsk"])
        arb = min(ka + kfee(ka) + (1 - pbid) + 0.01, pask + 0.01 + kb + kfee(kb))
        kmid = (float(a["yes_bid_dollars"]) + ka) / 2
        rows.append((e["title"], outs[0], round(kmid, 3), round((pbid + pask) / 2, 3), round(arb, 4), str(t)[:16]))
    print(f"{lg}: {len(rows)} upcoming games matched, {unmatched} Polymarket games unmatched")
    for r in sorted(rows, key=lambda r: -abs(r[2] - r[3])):
        print(f"  {r[5]} {r[0][:34]:34s} {r[1][:12]:12s} Kalshi {r[2]:.3f} Poly {r[3]:.3f} diff {r[2] - r[3]:+.3f}  "
              f"2-venue cost {r[4]}" + ("  <-- ARB?" if r[4] < 1 else ""))
    if rows:
        d = sorted(abs(r[2] - r[3]) for r in rows)
        print(f"median |diff| {d[len(d) // 2]:.3f}; arb candidates {sum(r[4] < 1 for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
