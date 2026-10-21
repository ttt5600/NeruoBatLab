"""Cross-venue 2026 midterm scan: Kalshi vs Polymarket (read-only public data). No orders.

    python scripts/xvenue_elections.py            # table of every matched race + any two-venue arbitrage after fees

Kalshi: SENATE{ST} / GOVPARTY{ST} (party markets: -R / -D), KXSENATE / KXHOUSE (control).
Polymarket: gamma public-search "{State} Senate Election Winner" / "{State} Governor Election Winner" events,
markets "Will the Republicans/Democrats win ...". Polymarket is traded ONLY via the US-regulated product (never the
international site); its prices here are read-only public data.
Arb check: buy R on one venue at its ask + D on the other at its ask; pays $1 whichever party wins (ignores
independents/third parties -> flagged). Kalshi fee 0.07*p*(1-p); Polymarket US fee assumed 0.01 per contract
(UNVERIFIED -- check the US product's fee schedule before acting).
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "xvenue_elections.jsonl"
K = "https://api.elections.kalshi.com/trade-api/v2"
PM = "https://gamma-api.polymarket.com/public-search"
ST = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado", "CT": "Connecticut",
      "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
      "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
      "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
      "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
      "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
      "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}
PM_FEE = 0.01


def kfee(p):
    return 0.07 * p * (1 - p)


def kalshi(series: str) -> dict:
    ms = requests.get(f"{K}/markets", params={"series_ticker": series, "status": "open", "limit": 50}, timeout=20).json().get("markets", [])
    out = {}
    for m in ms:
        if "26" not in m["event_ticker"]:
            continue
        side = m["ticker"].rsplit("-", 1)[-1]
        if side in ("R", "D"):
            out[side] = (float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 1), float(m.get("volume_fp") or 0))
    return out


def poly(state: str, office: str) -> dict:
    q = f"{state} {office} Election Winner"
    d = requests.get(PM, params={"q": q, "limit_per_type": 5}, timeout=20).json()
    for e in d.get("events", []):
        t = (e.get("title") or "").lower()
        if state.lower() in t and office.lower() in t and "winner" in t and "2026" in (e.get("slug") or "") + (e.get("endDate") or ""):
            out = {}
            for m in e.get("markets", []):
                qn = (m.get("question") or "").lower()
                bid, ask = m.get("bestBid"), m.get("bestAsk")
                if bid is None or ask is None:
                    continue
                if "republican" in qn:
                    out["R"] = (float(bid), float(ask), float(m.get("volume") or 0))
                elif "democrat" in qn:
                    out["D"] = (float(bid), float(ask), float(m.get("volume") or 0))
            return out
    return {}


def main() -> int:
    rows = []
    for ab, name in ST.items():
        for office, kser in (("Senate", f"SENATE{ab}"), ("Governor", f"GOVPARTY{ab}")):
            k = kalshi(kser)
            if not k:
                continue
            p = poly(name, office)
            time.sleep(0.2)
            r = {"race": f"{ab} {office}", "k": k, "p": p}
            if "R" in k and "R" in p:
                km, pm = (k["R"][0] + k["R"][1]) / 2, (p["R"][0] + p["R"][1]) / 2
                r["k_R_mid"], r["p_R_mid"], r["diff"] = round(km, 3), round(pm, 3), round(km - pm, 3)
            arbs = []
            for a, b in (("k", "p"), ("p", "k")):
                A, B = (k, p) if a == "k" else (p, k)
                if "R" in A and "D" in B:
                    ra, da = A["R"][1], B["D"][1]
                    cost = ra + da + (kfee(ra) if a == "k" else PM_FEE) + (kfee(da) if b == "k" else PM_FEE)
                    arbs.append((f"R@{a}+D@{b}", round(cost, 4)))
            r["best_arb_cost"] = min(arbs, key=lambda x: x[1]) if arbs else None
            rows.append(r)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with OUT.open("a") as f:
        for r in rows:
            f.write(json.dumps({"ts": ts, **r}) + "\n")
    m = [r for r in rows if "diff" in r]
    print(f"{len(rows)} Kalshi races, {len(m)} matched on Polymarket; |R mid diff| median "
          f"{sorted(abs(r['diff']) for r in m)[len(m) // 2] if m else 0:.3f}")
    for r in sorted(m, key=lambda r: -abs(r["diff"])):
        flag = " <-- DIVERGE" if abs(r["diff"]) >= 0.03 else ""
        print(f"  {r['race']:14s} Kalshi R {r['k_R_mid']:.3f}  Poly R {r['p_R_mid']:.3f}  diff {r['diff']:+.3f}  "
              f"best 2-venue cost {r['best_arb_cost']}{flag}")
    arb = [r for r in rows if r["best_arb_cost"] and r["best_arb_cost"][1] < 1.0]
    print("ARBITRAGE (cost < $1 after fees):", [(r["race"], r["best_arb_cost"]) for r in arb] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
