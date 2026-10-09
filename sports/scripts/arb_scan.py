"""Scan for arbitrage between Kalshi and Polymarket, and inside each venue. Read-only.

    python scripts/arb_scan.py            # one snapshot -> data/arb_scan.jsonl
    python scripts/arb_scan.py report     # how often / how big, over all snapshots

Public endpoints only, no accounts, no orders. Two kinds of arb are checked:

  cross  : same proposition on both venues (e.g. "Bills win AFC East"):
           buy YES on one + NO on the other for < $1 after fees.
  dutch  : one venue, mutually exclusive outcomes (the 4 teams of a division):
           buy YES on every outcome for < $1 total after fees.

Exchanges match you against other users, so (unlike sportsbooks) they do not
limit winning accounts. The real risks are settlement-rule differences between
venues, fee changes, and capital locked until the market resolves -- hence the
annualized return next to every hit. Polymarket's international venue is not
open to US residents; check access before acting on a cross-venue hit.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

LOG = Path(__file__).resolve().parents[1] / "data" / "arb_scan.jsonl"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
H = {"User-Agent": "Mozilla/5.0"}
KALSHI_FEE = 0.07   # taker fee per contract = 0.07 * p * (1 - p)  (assumed; verify current schedule)
POLY_FEE = 0.0      # taker fee rate on Polymarket; set from the venue's current schedule
MIN_EDGE = 0.005    # report hits worth >= 0.5c per $1 contract pair

# (label, kalshi series, regex on Polymarket event title, poly tag)
PAIRS = [(f"NFL {c} {d}", f"KXNFL{c}{d.upper()}", rf"{c} {d} Champion", "nfl")
         for c in ("AFC", "NFC") for d in ("East", "North", "South", "West")] + [
    ("Super Bowl", "KXSB", r"Pro Football: \d{4} Champion$", "nfl"),
    ("World Series", "KXMLBWS", r"World Series Champion", "mlb"),
    ("Stanley Cup", "KXNHL", r"(Stanley Cup|NHL) Champion", "nhl"),
    ("NBA title", "KXNBA", r"NBA Champion", "nba"),
]


def kfee(p):
    return KALSHI_FEE * p * (1 - p)


def nick(name: str) -> str:
    """Team key = last word of the name ("Buffalo Bills" -> "bills"). Kalshi often uses the city."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).split()[-1] if name else ""


def kalshi_event(series):
    ms = requests.get(f"{KALSHI}/markets", headers=H, timeout=30,
                      params={"series_ticker": series, "status": "open", "limit": 200}).json().get("markets", [])
    out = {}
    for m in ms:
        ask, bid = float(m.get("yes_ask_dollars") or 0), float(m.get("yes_bid_dollars") or 0)
        name = m.get("yes_sub_title") or m.get("subtitle") or ""
        out.setdefault(m["event_ticker"], []).append(
            {"name": name, "ticker": m["ticker"], "ask": ask, "bid": bid,
             "close": m.get("close_time") or m.get("expiration_time")})
    return out


def poly_events(tag):
    evs, off = [], 0
    while off < 1000:
        page = requests.get(f"{GAMMA}/events", headers=H, timeout=30,
                            params={"tag_slug": tag, "closed": "false", "limit": 100, "offset": off}).json()
        evs += page
        if len(page) < 100:
            break
        off += 100
    return evs


def poly_team(question):
    m = re.match(r"Will (?:the )?(.+?) win", question or "")
    return m.group(1) if m else ""


def years_to(close):
    try:
        t = datetime.fromisoformat(str(close).replace("Z", "+00:00"))
        return max((t - datetime.now(timezone.utc)).days, 1) / 365
    except ValueError:
        return None


def scan():
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    hits, checked = [], 0
    poly_cache = {}
    for label, series, title_re, tag in PAIRS:
        kev = kalshi_event(series)
        if tag not in poly_cache:
            poly_cache[tag] = poly_events(tag)
        pev = [e for e in poly_cache[tag] if re.search(title_re, e.get("title", ""))]
        pm = {}
        for e in pev:
            for m in e.get("markets", []):
                if m.get("bestAsk") is not None and m.get("active") and not m.get("closed"):
                    pm[poly_team(m["question"]).lower()] = {"q": m["question"], "ask": float(m["bestAsk"]),
                                                         "bid": float(m.get("bestBid") or 0), "event": e["title"]}
        for ev, ks in kev.items():
            yrs = years_to(ks[0]["close"]) if ks else None
            # dutch book inside Kalshi
            if len(ks) >= 2 and all(0 < k["ask"] < 1 for k in ks):
                cost = sum(k["ask"] + kfee(k["ask"]) for k in ks)
                checked += 1
                if cost < 1 - MIN_EDGE:
                    hits.append({"ts": now, "kind": "dutch_kalshi", "label": label, "event": ev,
                                 "cost": round(cost, 4), "edge": round(1 - cost, 4), "years": yrs})
            for k in ks:
                cand = [n for n in pm if k["name"] and n.startswith(k["name"].lower())]
                p = pm[cand[0]] if len(cand) == 1 else None
                if not p:
                    continue
                checked += 1
                # YES Kalshi + NO Poly ; YES Poly + NO Kalshi
                legs = [("yes_kalshi+no_poly", k["ask"] + kfee(k["ask"]) + (1 - p["bid"]) * (1 + POLY_FEE), k["ask"] > 0 and p["bid"] > 0),
                        ("yes_poly+no_kalshi", p["ask"] * (1 + POLY_FEE) + (1 - k["bid"]) + kfee(1 - k["bid"]), k["bid"] > 0 and p["ask"] < 1)]
                for kind, cost, valid in legs:
                    if valid and cost < 1 - MIN_EDGE:
                        edge = 1 - cost
                        hits.append({"ts": now, "kind": kind, "label": label, "kalshi": k["ticker"],
                                     "kalshi_name": k["name"], "poly_q": p["q"], "poly_event": p["event"],
                                     "k_bid": k["bid"], "k_ask": k["ask"], "p_bid": p["bid"], "p_ask": p["ask"],
                                     "cost": round(cost, 4), "edge": round(edge, 4), "years": yrs,
                                     "annualized": round(edge / cost / yrs, 3) if yrs else None})
        print(f"{label:16s} kalshi events {len(kev)}  poly markets matched {len(pm)}", flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps({"ts": now, "snapshot": True, "checked": checked, "hits": len(hits)}) + "\n")
        for h in hits:
            f.write(json.dumps(h) + "\n")
    print(f"\n{checked} comparisons, {len(hits)} hits >= {MIN_EDGE:.3f}")
    for h in sorted(hits, key=lambda h: -h["edge"])[:15]:
        print(f"  {h['kind']:20s} {h['label']:14s} {h.get('kalshi_name', h.get('event')):22s} "
              f"edge {h['edge']:.3f}  ann {h.get('annualized')}  | {h.get('poly_q', '')}")


def report():
    rows = [json.loads(l) for l in LOG.read_text().splitlines()] if LOG.exists() else []
    snaps = [r for r in rows if r.get("snapshot")]
    hits = [r for r in rows if not r.get("snapshot")]
    print(f"{len(snaps)} snapshots, {len(hits)} hits; by kind:",
          {k: sum(h['kind'] == k for h in hits) for k in {h['kind'] for h in hits}})


if __name__ == "__main__":
    report() if (sys.argv[1:] or [""])[0] == "report" else scan()
