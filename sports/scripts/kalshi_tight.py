"""Shared helpers for the Q004 FIRST-TIGHT-QUOTE addendum of nba_open_totals.py and open_lines_logger.py (2026-10-28).
Paper only -- reads the public market + orderbook endpoints, never places orders. Pre-registration text lives in
the two loggers' docstrings; this module only implements it.

Terms (per Kalshi event = one game x one series, e.g. KXNBATOTAL-26OCT28OKCDAL):
  two-sided row = a market with yes_bid > 0 and yes_ask < 1.
  ANCHOR        = the two-sided row whose YES mid is nearest 0.50 (ladders: the at-the-money strike; ML: the closer
                  of the two team markets). TIGHT = anchor yes_ask - yes_bid <= TIGHT_MAX (4c).
  censored      = the first-two-sided / first-tight observation happened at our FIRST sighting of the event and the
                  event's Kalshi open_time is more than `slack` older than that sighting (we missed the real moment).
  depth         = from GET /markets/{ticker}/orderbook (bids only on both sides; buying YES lifts NO bids at 1-q).
  fee           = exact Kalshi taker fee for a 100-contract order: ceil(100 * 0.07 p (1-p) to the cent) / 100.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TIGHT_MAX = 0.04
FEE_ORDER_N = 100


def fee_exact(p: float, n: int = FEE_ORDER_N) -> float:
    """Per-contract taker fee of an n-contract order, Kalshi rounding (total rounded UP to the cent)."""
    return math.ceil(round(n * 0.07 * p * (1 - p) * 100, 6)) / 100 / n


def two_sided(r: dict) -> bool:
    b, a = r.get("yes_bid"), r.get("yes_ask")
    return b is not None and a is not None and np.isfinite(b) and np.isfinite(a) and b > 0 and a < 1


def anchor(rows: list[dict]) -> dict | None:
    c = [r for r in rows if two_sided(r)]
    return min(c, key=lambda r: abs((r["yes_bid"] + r["yes_ask"]) / 2 - 0.5)) if c else None


def book(get, base: str, ticker: str) -> dict | None:
    """{'yes': [(price, qty)] best first, 'no': [...]} -- bid ladders; None on failure."""
    try:
        d = get(f"{base}/markets/{ticker}/orderbook")
    except Exception:  # noqa: BLE001 -- depth is a logged extra, never fatal
        return None
    o = d.get("orderbook_fp") or d.get("orderbook") or {}
    out = {}
    for s in ("yes", "no"):
        lv = [(float(p) if float(p) <= 1 else float(p) / 100, float(q)) for p, q in (o.get(f"{s}_dollars") or o.get(s) or [])]
        out[s] = sorted(lv, reverse=True)
    return out


def buy_depth(bk: dict | None, yes: bool) -> dict:
    """Depth for BUYING `yes` (True) or NO: offers = the other side's bids at 1-q. Contracts and $ at the best price
    and within 1c of it."""
    if not bk:
        return {}
    lv = bk["no" if yes else "yes"]
    if not lv:
        return {"best_ask": None, "ct_best": 0.0, "usd_best": 0.0, "ct_1c": 0.0, "usd_1c": 0.0}
    q0 = lv[0][0]
    best = [(q, n) for q, n in lv if q >= q0 - 1e-9]
    near = [(q, n) for q, n in lv if q >= q0 - 0.0101]
    return {"best_ask": round(1 - q0, 4), "ct_best": sum(n for _, n in best),
            "usd_best": round(sum(n * (1 - q) for q, n in best), 2),
            "ct_1c": sum(n for _, n in near), "usd_1c": round(sum(n * (1 - q) for q, n in near), 2)}


def bid_depth(bk: dict | None, yes: bool) -> dict:
    if not bk or not bk["yes" if yes else "no"]:
        return {}
    lv = bk["yes" if yes else "no"]
    q0 = lv[0][0]
    best = [(q, n) for q, n in lv if q >= q0 - 1e-9]
    return {"best_bid": round(q0, 4), "ct_best": sum(n for _, n in best), "usd_best": round(sum(n * q for q, n in best), 2)}


def track(rec: dict | None, rows: list[dict], t: pd.Timestamp, slack: pd.Timedelta, get, base: str) -> dict:
    """Update one event's listing record with this tick's quotes. Returns the (new) record."""
    ots = [pd.Timestamp(r["open_time"]) for r in rows if r.get("open_time")]
    first = rec is None
    if first:
        ot = min(ots) if ots else None
        rec = {"first_seen": t.isoformat(), "open_time": ot.isoformat() if ot is not None else None,
               "listing_censored": bool(ot is None or t - ot > slack), "first_two_sided": None, "first_tight": None}
    a = anchor(rows)
    if a is None:
        return rec
    ot = pd.Timestamp(rec["open_time"]) if rec.get("open_time") else None
    lag = lambda: None if ot is None else round((t - ot).total_seconds() / 3600, 3)  # noqa: E731
    snap = {"t": t.isoformat(), "ticker": a["ticker"], "strike": a.get("strike"), "team": a.get("team"),
            "yes_bid": a["yes_bid"], "yes_ask": a["yes_ask"], "spread": round(a["yes_ask"] - a["yes_bid"], 4),
            "h_after_open": lag(), "censored": bool(first and rec["listing_censored"])}
    if rec["first_two_sided"] is None:
        rec["first_two_sided"] = snap
    if rec["first_tight"] is None and snap["spread"] <= TIGHT_MAX + 1e-9:
        bk = book(get, base, a["ticker"])
        rec["first_tight"] = {**snap, "depth_yes_ask": buy_depth(bk, True), "depth_no_ask": buy_depth(bk, False),
                              "depth_yes_bid": bid_depth(bk, True)}
    return rec


def roi(result: str, ask: float, fee: float) -> float:
    return (1 - ask - fee) / (ask + fee) if result == "win" else -1.0


def pct(xs, q):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.percentile(xs, q)) if xs else float("nan")


def report(tight: dict, settled: list[dict], decision_depths: list[float], label: str) -> None:
    """Print the Q004 section: listing timing, first-tight depth, pick_tight results."""
    mk = list(tight.get("markets", {}).values())
    print(f"\n-- Q004 first-tight addendum ({label}) --")
    two = [m["first_two_sided"] for m in mk if m.get("first_two_sided") and not m["first_two_sided"]["censored"]]
    ft = [m["first_tight"] for m in mk if m.get("first_tight") and not m["first_tight"]["censored"]]
    print(f"  events tracked {len(mk)}; listing observed {sum(not m['listing_censored'] for m in mk)}; "
          f"first two-sided (uncensored) {len(two)}; first tight (uncensored) {len(ft)}")
    if two:
        print(f"  first two-sided: median {pct([x['h_after_open'] for x in two], 50):.2f} h after open_time, "
              f"median spread {pct([x['spread'] for x in two], 50) * 100:.0f}c")
    if ft:
        ya = [x["depth_yes_ask"].get("usd_best") for x in ft if x.get("depth_yes_ask")]
        na = [x["depth_no_ask"].get("usd_best") for x in ft if x.get("depth_no_ask")]
        print(f"  first tight: median {pct([x['h_after_open'] for x in ft], 50):.2f} h after open_time "
              f"(P90 {pct([x['h_after_open'] for x in ft], 90):.1f} h); $ at best YES ask median {pct(ya, 50):.0f} "
              f"P10 {pct(ya, 10):.0f}; NO ask median {pct(na, 50):.0f} P10 {pct(na, 10):.0f}")
    if decision_depths:
        print(f"  our side within 1c at decision ticks (n {len(decision_depths)}): median ${pct(decision_depths, 50):.0f} "
              f"P10 ${pct(decision_depths, 10):.0f}")
    picks = list(tight.get("picks", {}).values())
    made = [p for p in picks if p.get("pick")]
    print(f"  pick_tight decided {len(picks)}, picks {len(made)}, settled {len(settled)}")
    if settled:
        w = sum(s["result"] == "win" for s in settled)
        r = [s["roi"] for s in settled]
        c = [s["clv"] for s in settled if s.get("clv") is not None]
        print(f"  pick_tight {w}-{len(settled) - w}  ROI @ask+fee {np.mean(r):+.1%}  CLV vs last pre-start Pinnacle "
              f"{np.mean(c):+.4f} (sd {np.std(c):.4f}, n {len(c)})")
        if len(settled) >= 100:
            ok = np.mean(c) > 0 and np.mean(r) > 0
            print("  Q004 RULE (>=100): CLV > 0 and ROI > 0 -> " + ("SEALED ASK" if ok else "no ask (fails)"))
        else:
            print(f"  Q004 rule checked at 100 settled pick_tight ({len(settled)} so far)")
