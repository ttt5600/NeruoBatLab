"""Paper-trade the Kalshi consensus-reject rule on live markets. No orders are placed.

    python scripts/kalshi_paper.py pick      # run ~10pm local: log tomorrow's picks + book depth
    python scripts/kalshi_paper.py settle    # any time later: fill in results from settled markets
    python scripts/kalshi_paper.py report

Rule (lockbox-validated, kalshi_reject_safe_check): for tomorrow's brackets in
each city, take each model's forecast max over hours <= 18:00 local; if ALL
of GFS, ECMWF, ICON, GEM, NBM (>= 3 available) miss the bracket by > 3F,
buy NO at 1 - best YES bid, if the YES ask > 1c and the quote is two-sided
with a spread <= 10c.

The point of paper trading is the one thing the backtest could not see:
how many contracts were actually resting at the price. Each pick records the
depth within 1c and 2c of the best price.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import requests

from sports import wx

LOG = Path(__file__).resolve().parents[1] / "data" / "kalshi_paper.jsonl"
MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "ncep_nbm_conus"]
MARGIN = 3.0
H = {"User-Agent": "Mozilla/5.0"}


def live_forecasts(lat, lon, tz, day):
    r = requests.get("https://api.open-meteo.com/v1/forecast", headers=H, timeout=60, params={
        "latitude": lat, "longitude": lon, "models": ",".join(MODELS), "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit", "timezone": tz, "forecast_days": 3}).json()["hourly"]
    times = [datetime.fromisoformat(t) for t in r["time"]]
    keep = [i for i, t in enumerate(times) if t.date() == day and t.hour <= 18]
    out = {}
    for m in MODELS:
        v = [r.get(f"temperature_2m_{m}", [None] * len(times))[i] for i in keep]
        v = [x for x in v if x is not None]
        out[m] = max(v) if len(v) >= 15 else None
    return out


def rejected(st, floor_s, cap_s, fcs):
    hi, lo = max(fcs), min(fcs)
    if st == "greater":
        return hi + MARGIN < floor_s + 1
    if st == "less":
        return lo - MARGIN > cap_s - 1
    return hi + MARGIN < floor_s or lo - MARGIN > cap_s


def book_depth(ticker):
    """Contracts available to BUY NO = resting YES bids. Depth within 1c/2c of best."""
    d = requests.get(f"{wx.API}/markets/{ticker}/orderbook", headers=H, timeout=30).json()
    ob = d.get("orderbook_fp") or d.get("orderbook") or {}
    bids = ob.get("yes_dollars") or ob.get("yes") or []
    levels = sorted(((float(p) if float(p) <= 1 else float(p) / 100, float(q)) for p, q in bids), reverse=True)
    if not levels:
        return None, 0.0, 0.0
    best = levels[0][0]
    d1 = sum(q for p, q in levels if p >= best - 0.0101)
    d2 = sum(q for p, q in levels if p >= best - 0.0201)
    return best, d1, d2


def cmd_pick(days_ahead: int = 1, log: bool = True):
    now = datetime.now(timezone.utc)
    picks = []
    for series, (station, tz) in wx.CITIES.items():
        tomorrow = (now.astimezone(ZoneInfo(tz)) + timedelta(days=days_ahead)).date()
        ev = f"{series}-{tomorrow.strftime('%y%b%d').upper()}"
        ms = requests.get(f"{wx.API}/markets", headers=H, timeout=30,
                          params={"event_ticker": ev, "limit": 100}).json().get("markets", [])
        if not ms:
            print(f"{series}: no open event {ev}")
            continue
        fc = live_forecasts(*wx.station_coords(station), tz, tomorrow)
        vals = [v for v in fc.values() if v is not None]
        if len(vals) < 3:
            print(f"{series}: only {len(vals)} model forecasts; skipping")
            continue
        for m in ms:
            ask = float(m.get("yes_ask_dollars") or 0)
            bid = float(m.get("yes_bid_dollars") or 0)
            fl = float(m["floor_strike"]) if m.get("floor_strike") is not None else np.nan
            ca = float(m["cap_strike"]) if m.get("cap_strike") is not None else np.nan
            if not (bid > 0 and ask < 1 and ask - bid <= 0.10 and ask > 0.01):
                continue
            if not rejected(m.get("strike_type"), fl, ca, vals):
                continue
            best, d1, d2 = book_depth(m["ticker"])
            picks.append({"logged_at": now.isoformat(timespec="seconds"), "city": series, "event": ev,
                          "ticker": m["ticker"], "strike_type": m.get("strike_type"), "floor": fl, "cap": ca,
                          "forecasts": fc, "yes_bid": bid, "yes_ask": ask, "no_price": round(1 - bid, 4),
                          "fee_per_contract": round(wx.FEE_RATE * (1 - bid) * bid, 4),
                          "depth_1c": d1, "depth_2c": d2, "result": None})
    if log:
        with LOG.open("a") as f:
            for p in picks:
                f.write(json.dumps(p) + "\n")
    for p in picks:
        print(f"PICK {p['ticker']}: buy NO @ {p['no_price']:.2f}  depth {p['depth_1c']:.0f} (1c) / "
              f"{p['depth_2c']:.0f} (2c) contracts  forecasts {p['forecasts']}")
    print(f"{len(picks)} picks " + (f"logged to {LOG}" if log else "(validation only, not logged)"))


def cmd_settle():
    rows = [json.loads(l) for l in LOG.read_text().splitlines()] if LOG.exists() else []
    for r in rows:
        if r["result"] is None:
            m = requests.get(f"{wx.API}/markets/{r['ticker']}", headers=H, timeout=30).json().get("market", {})
            if m.get("result") in ("yes", "no"):
                r["result"] = m["result"]
    LOG.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cmd_report()


def cmd_report():
    rows = [json.loads(l) for l in LOG.read_text().splitlines()] if LOG.exists() else []
    done = [r for r in rows if r["result"]]
    units = sum((1 - r["no_price"] - r["fee_per_contract"]) / (r["no_price"] + r["fee_per_contract"])
                if r["result"] == "no" else -1.0 for r in done)
    print(f"{len(rows)} paper picks, {len(done)} settled, {sum(r['result'] == 'no' for r in done)} wins, "
          f"{units:+.2f} units; median depth within 1c: "
          f"{np.median([r['depth_1c'] for r in rows]) if rows else 0:.0f} contracts")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "validate":  # today's open markets: exercises the live path, logs nothing
        cmd_pick(days_ahead=0, log=False)
    else:
        {"pick": cmd_pick, "settle": cmd_settle, "report": cmd_report}[cmd]()
