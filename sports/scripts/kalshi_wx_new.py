"""More volume for the proven E1/E2 rule: census of NEW Kalshi daily weather series. DEV only, no orders.

    python scripts/kalshi_wx_new.py census [--ledger]

Pre-registered 2026-10-28, BEFORE any price, forecast or settlement of a candidate series was loaded:
  Candidates : (a) any daily-HIGH series (KXHIGH*) with listed events that is not one of the 24 traded
               (wx.CITIES + kalshi_paper.X1_CITIES); (b) any other DAILY weather series with a forecastable
               numeric outcome (snowfall, max wind/gust, ...), excluding lows (X2 dead) and daily rain (R1 dead).
  Rule       : E1/E2 frozen, unchanged (kalshi_reject_safe_check / wx_expand.picks): 5 models, lead-1, max over
               local hours <= 18, reject if every model (>= 3) misses the bracket by > 3F; buy NO at 1 - yes_bid if
               yes_ask > 0.01, two-sided quote, spread <= 10c; decision 22:00 local the day before.
               Non-temperature series would need their own pre-registered margin -- none qualified (see census).
  Split      : per series, DEV = earliest 70% of its settled event-days; SEALED = later 30% (never loaded here).
  Scoring    : taker = exact fee 0.07*p*(1-p) unrounded; maker = M3 (NO mid floored, <= ask-1c, tape fill, fee 0
               on fee_type quadratic, sensitivity 0.0175*q*(1-q)); one-sided bootstrap over station-days.
  Trials     : one per qualifying series x {taker, maker}; bar p < 0.05 / #trials. Minimum 60 DEV event-days per
               series or it is reported as underpowered and not scored.

Census result 2026-10-28 (458 Climate-and-Weather series): 0 qualifying candidates -> 0 trials scored.
  - All 24 traded KXHIGH* series are open (12 markets each). The other daily-high tickers have no events at all
    (21 international KXHIGHT* e.g. EGLL/CYYZ/RJTT, KXHIGHTKSAN, KXDVHIGH, KXDENHIGH, KXHOUHIGH, KXHIGHOU) or are
    retired legacy duplicates (KXHIGHHOU 2024-12..2025-01, KXCITIESWEATHER 2025-02..03, KXHIGHUS 2023).
  - No daily snowfall, wind or gust series exist (snow is monthly/seasonal only; L39 covers monthly).
  - Weekly KXAVGT* (11 cities) ran 2026-08-24..09-27 only (8 events each) and has nothing open: not daily, no market.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import requests

from sports import wx

LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "kalshi_wx_new"
H = {"User-Agent": "Mozilla/5.0"}
X1 = ["KXHIGHTATL", "KXHIGHTBOS", "KXHIGHTDAL", "KXHIGHTDC", "KXHIGHTEWR", "KXHIGHTHOU", "KXHIGHTLV", "KXHIGHTMIN",
      "KXHIGHTNOLA", "KXHIGHTOKC", "KXHIGHTPHX", "KXHIGHTSAN", "KXHIGHTSATX", "KXHIGHTSDF", "KXHIGHTSEA", "KXHIGHTSFO",
      "KXHIGHTTTN"]  # = kalshi_paper.X1_CITIES keys (copied to avoid importing a live-path module)
TRADED = set(wx.CITIES) | set(X1)
DEAD = re.compile(r"^(KX)?(LOW|RAIN)")  # lows (X2) and daily rain (R1) already declared dead


def _get(path, **p):
    return requests.get(f"{wx.API}{path}", params=p, headers=H, timeout=30).json()


def census() -> dict:
    ser = [x for x in _get("/series", limit=5000)["series"] if x.get("category") == "Climate and Weather"]
    daily = sorted(x["ticker"] for x in ser if x.get("frequency") == "daily" and x["ticker"] not in TRADED
                   and not DEAD.match(x["ticker"]))
    rows = {}
    for s in daily:
        op = _get("/markets", series_ticker=s, status="open", limit=200).get("markets", [])
        st = _get("/markets", series_ticker=s, status="settled", limit=1000).get("markets", [])
        hi = _get("/historical/markets", series_ticker=s, limit=1000).get("markets", [])
        evs = sorted({m["event_ticker"] for m in st + hi})
        rows[s] = {"open_markets": len(op), "settled_events": len(evs), "first": evs[0] if evs else None,
                   "last": evs[-1] if evs else None,
                   "qualifies": bool(op) and len(evs) >= 60 / 0.7 and s.startswith("KXHIGH")}
        print(s, rows[s])
    traded_open = {s: len(_get("/markets", series_ticker=s, status="open", limit=200).get("markets", [])) for s in sorted(TRADED)}
    return {"n_series": len(ser), "candidates": rows, "qualifying": [s for s, r in rows.items() if r["qualifies"]],
            "traded_open_markets": traded_open}


if __name__ == "__main__":
    if (sys.argv[1:2] or [""])[0] != "census":
        print(__doc__)
        raise SystemExit(1)
    r = census()
    print(f"{r['n_series']} weather series; qualifying new daily series: {r['qualifying'] or 'NONE'}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": "kalshi_wx_new_census",
                                "sha": "census", "dataset": DATASET, "status": "no_trial",
                                "hypothesis": "pre-registered in scripts/kalshi_wx_new.py: E1/E2 frozen rule on new daily weather series",
                                "source": "volume expansion for E1/E2", "trials_scored": 0,
                                "qualifying": r["qualifying"], "candidates_checked": len(r["candidates"])}) + "\n")
        print("appended census row to ledger (0 trials)")
