"""Does Kalshi's new settlement source (The Weather Company, from 2026-08-14) agree with the NWS climate report?

    python scripts/wx_source_check.py [--all]     # 7 traded series (default) or all 48 US temp series

Found 2026-10-19: every Kalshi KXHIGH*/KXLOW* series' rules switched from NWS to "The Weather Company"
(weather.com/kalshi) on 2026-08-14 (KXHIGHNY: NWS through 08-13, TWC from 08-14). The consensus-reject rule was
validated on NWS-settled history, so we compare Kalshi's settled value (expiration_value) with the official NWS
daily max/min from ACIS (data.rcc-acis.org, keyless) for the same station + date, before (control: should match)
and after the switch. Descriptive data check, not a trial.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests

from sports.http import fetch

API = "https://api.elections.kalshi.com/trade-api/v2"
SWITCH = pd.Timestamp("2026-08-14")
TRADED = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHAUS", "KXHIGHDEN", "KXHIGHLAX", "KXHIGHPHIL"]


def all_series() -> list[str]:
    d = fetch(f"{API}/series", {"limit": 5000}, cache=False)["series"]
    s = [x["ticker"] for x in d if re.match(r"^KX(HIGH|LOW)T?[A-Z]{2,5}$", x["ticker"]) and x.get("category") == "Climate and Weather"]
    live = []
    for t in s:
        if fetch(f"{API}/markets", {"series_ticker": t, "status": "open", "limit": 1}, cache=False).get("markets"):
            live.append(t)
    return live


def settled(series: str) -> pd.DataFrame:
    rows = []
    for path, extra in ((f"{API}/historical/markets", {}), (f"{API}/markets", {"status": "settled"})):
        cur = None
        while True:
            p = {"series_ticker": series, "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            d = fetch(path, p, cache="historical" in path)
            rows += d.get("markets", [])
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("ticker")
    df["value"] = pd.to_numeric(df["expiration_value"], errors="coerce")
    df["station"] = df["rules_primary"].str.extract(r"\(CLI([A-Z]{3})\)", expand=False)
    df["src"] = np.where(df["rules_primary"].str.contains("Weather Company", na=False), "TWC", "NWS")
    df["date"] = pd.to_datetime(df["event_ticker"].str.split("-").str[-1], format="%y%b%d", errors="coerce")
    g = df.groupby("event_ticker").agg(date=("date", "first"), value=("value", "max"), station=("station", "first"),
                                       src=("src", "first"))
    return g.dropna(subset=["value", "date"]).reset_index()


def acis(station: str, start, end, elem: str) -> pd.Series:
    r = requests.get("https://data.rcc-acis.org/StnData", params={"sid": f"K{station}", "sdate": str(start.date()),
                     "edate": str(end.date()), "elems": elem, "output": "json"}, timeout=60).json()
    s = pd.Series({pd.Timestamp(d): pd.to_numeric(v, errors="coerce") for d, v in r.get("data", [])})
    return s


def main() -> int:
    series = all_series() if "--all" in sys.argv else TRADED
    out = []
    for s in series:
        ev = settled(s)
        if ev.empty or ev.station.isna().all():
            print(f"{s}: no station in rules")
            continue
        st = ev.station.dropna().mode()[0]
        ev = ev[ev.date >= SWITCH - pd.Timedelta(days=60)]
        nws = acis(st, ev.date.min(), ev.date.max(), "maxt" if "HIGH" in s else "mint")
        ev["nws"] = ev.date.map(nws)
        ev["diff"] = ev.value - ev.nws
        for src, g in ev.dropna(subset=["nws"]).groupby("src"):
            out.append({"series": s, "station": st, "src": src, "days": len(g), "exact": round((g["diff"] == 0).mean(), 3),
                        "within1": round((g["diff"].abs() <= 1).mean(), 3), "mean_diff": round(g["diff"].mean(), 2),
                        "max_abs": g["diff"].abs().max(), "gt3": int((g["diff"].abs() > 3).sum())})
        print(pd.DataFrame([o for o in out if o["series"] == s]).to_string(index=False, header=False), flush=True)
    r = pd.DataFrame(out)
    print("\nSUMMARY by source (days pooled):")
    for src, g in r.groupby("src"):
        w = g.days
        print(f"  {src}: series {len(g)} days {w.sum()} exact {np.average(g.exact, weights=w):.3f} within1F "
              f"{np.average(g.within1, weights=w):.3f} mean diff {np.average(g.mean_diff, weights=w):+.2f}F "
              f"days |diff|>3F {g.gt3.sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
