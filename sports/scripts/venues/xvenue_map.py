"""Cross-venue map: what we trade/paper on Kalshi vs Polymarket US equivalents, with a LIVE depth/spread snapshot.

    python scripts/venues/xvenue_map.py            # weather highs (24 Kalshi cities) + NBA/NHL/NFL winners + totals
    python scripts/venues/xvenue_map.py weather    # only weather
    python scripts/venues/xvenue_map.py sports     # only sports

Read-only, public endpoints only (Kalshi api.elections.kalshi.com, Polymarket US gateway). Writes the raw rows to
data/xvenue_map_<UTC stamp>.json. Descriptive snapshot, not a trial.

Book conventions (both venues reported as the YES instrument):
  spread = yes_ask - yes_bid; bid_qty = contracts at the best YES bid (what a NO taker hits), ask_qty = at the best YES ask.
  Kalshi orderbook lists YES bids and NO bids; YES ask = 1 - best NO bid, its size = that NO bid's size.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from polymarket_us_client import PolymarketUS  # noqa: E402

KAPI = "https://api.elections.kalshi.com/trade-api/v2"
H = {"User-Agent": "Mozilla/5.0 (research; read-only)"}

# Kalshi daily-high series we paper/trade (sports/wx.py CITIES = orig7; kalshi_paper.X1_CITIES = 17 more)
KALSHI_HIGH = {
    "KXHIGHNY": ("KNYC", "America/New_York"), "KXHIGHCHI": ("KMDW", "America/Chicago"),
    "KXHIGHMIA": ("KMIA", "America/New_York"), "KXHIGHAUS": ("KAUS", "America/Chicago"),
    "KXHIGHDEN": ("KDEN", "America/Denver"), "KXHIGHLAX": ("KLAX", "America/Los_Angeles"),
    "KXHIGHPHIL": ("KPHL", "America/New_York"),
    "KXHIGHTATL": ("KATL", "America/New_York"), "KXHIGHTBOS": ("KBOS", "America/New_York"),
    "KXHIGHTDAL": ("KDFW", "America/Chicago"), "KXHIGHTDC": ("KDCA", "America/New_York"),
    "KXHIGHTEWR": ("KEWR", "America/New_York"), "KXHIGHTHOU": ("KHOU", "America/Chicago"),
    "KXHIGHTLV": ("KLAS", "America/Los_Angeles"), "KXHIGHTMIN": ("KMSP", "America/Chicago"),
    "KXHIGHTNOLA": ("KMSY", "America/Chicago"), "KXHIGHTOKC": ("KOKC", "America/Chicago"),
    "KXHIGHTPHX": ("KPHX", "America/Phoenix"), "KXHIGHTSAN": ("KSAN", "America/Los_Angeles"),
    "KXHIGHTSATX": ("KSAT", "America/Chicago"), "KXHIGHTSDF": ("KSDF", "America/Kentucky/Louisville"),
    "KXHIGHTSEA": ("KSEA", "America/Los_Angeles"), "KXHIGHTSFO": ("KSFO", "America/Los_Angeles"),
    "KXHIGHTTTN": ("KTTN", "America/New_York"),
}
# Polymarket US daily-high event prefixes by station (docs faqs/weather-faqs.md + market text)
PM_HIGH = {"KNYC": "nychigh", "KMDW": "mdwhigh", "KMIA": "miahigh", "KLAX": "laxhigh", "KSFO": "sfohigh", "KPHL": "phlhigh"}
PM_LOW_TRY = {"KNYC": "nyclow", "KMDW": "mdwlow", "KMIA": "mialow", "KLAX": "laxlow", "KSFO": "sfolow", "KPHL": "phllow"}
SPORTS = {"nba": ("KXNBAGAME", "KXNBATOTAL"), "nhl": ("KXNHLGAME", "KXNHLTOTAL"), "nfl": ("KXNFLGAME", "KXNFLTOTAL")}
ALIAS = {"was": ["WSH", "WAS"], "wsh": ["WSH", "WAS"], "la": ["LA", "LAR"], "lar": ["LA", "LAR"], "nj": ["NJ", "NJD"],
         "tb": ["TB", "TBL"], "sj": ["SJ", "SJS"], "lv": ["LV", "LVR", "VGK"], "vgk": ["VGK", "LV"], "utah": ["UTA"],
         "uta": ["UTA", "UTAH"], "gs": ["GSW"], "gsw": ["GSW", "GS"], "no": ["NO", "NOP"], "nop": ["NOP", "NO"],
         "sa": ["SAS", "SA"], "sas": ["SAS", "SA"], "ny": ["NYK", "NY"], "nyk": ["NYK", "NY"], "jax": ["JAX", "JAC"],
         "mon": ["MTL", "MON"], "mtl": ["MTL", "MON"], "nas": ["NSH", "NAS"], "nsh": ["NSH", "NAS"]}

_last = [0.0]


def kget(path, params=None, retries=5):
    for i in range(retries):
        w = _last[0] + 0.12 - time.monotonic()
        if w > 0:
            time.sleep(w)
        _last[0] = time.monotonic()
        r = requests.get(KAPI + path, params=params, headers=H, timeout=30)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(1.5 * 2 ** i)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    return None


def kbook(ticker: str) -> dict:
    d = kget(f"/markets/{ticker}/orderbook") or {}
    ob = d.get("orderbook_fp") or d.get("orderbook") or {}
    cv = lambda xs: sorted(((float(p) if float(p) <= 1 else float(p) / 100, float(q)) for p, q in xs or []), reverse=True)  # noqa: E731
    yb, nb = cv(ob.get("yes_dollars") or ob.get("yes")), cv(ob.get("no_dollars") or ob.get("no"))
    return {"yes_bid": yb[0][0] if yb else None, "bid_qty": yb[0][1] if yb else 0.0,
            "yes_ask": round(1 - nb[0][0], 4) if nb else None, "ask_qty": nb[0][1] if nb else 0.0}


def pbook(pm: PolymarketUS, slug: str) -> dict:
    b = pm.book(slug)
    if not b:
        return {"yes_bid": None, "yes_ask": None, "bid_qty": 0.0, "ask_qty": 0.0}
    return {"yes_bid": b.bids[0][0] if b.bids else None, "bid_qty": b.bids[0][1] if b.bids else 0.0,
            "yes_ask": b.asks[0][0] if b.asks else None, "ask_qty": b.asks[0][1] if b.asks else 0.0}


def kalshi_strike(km: dict):
    f = lambda k: None if km.get(k) is None else float(km[k])  # noqa: E731
    return km.get("strike_type"), f("floor_strike"), f("cap_strike")


def pm_strike(slug: str):
    import polymarket_us as legacy  # bracket parser shared with venues_weather.py
    t = legacy.parse_bracket(slug)
    if not t:
        return None
    st, fl, ca = t
    return st, (None if fl != fl else fl), (None if ca != ca else ca)


# ------------------------------------------------------------------------------------------------- weather
def weather(pm: PolymarketUS) -> list[dict]:
    rows = []
    now = datetime.now(timezone.utc)
    for series, (station, tz) in KALSHI_HIGH.items():
        pmc = PM_HIGH.get(station)
        if not pmc:
            rows.append({"kind": "wx_high", "series": series, "station": station, "pm": None})
            continue
        local = now.astimezone(ZoneInfo(tz)).date()
        for d in (local, local + timedelta(days=1)):
            ev = pm.event(f"temp-{pmc}-{d.isoformat()}")
            kev = f"{series}-{d.strftime('%y%b%d').upper()}"
            kms = (kget("/markets", {"event_ticker": kev, "limit": 100}) or {}).get("markets", [])
            kmap = {kalshi_strike(k): k for k in kms}
            for m in (ev or {}).get("markets", []):
                key = pm_strike(m["slug"])
                km = kmap.get(key)
                r = {"kind": "wx_high", "series": series, "station": station, "pm": pmc, "date": d.isoformat(),
                     "slug": m["slug"], "strike": key, "ticker": km and km["ticker"], "matched": km is not None}
                r.update({f"p_{k}": v for k, v in pbook(pm, m["slug"]).items()})
                if km:
                    r.update({f"k_{k}": v for k, v in kbook(km["ticker"]).items()})
                rows.append(r)
            print(f"  {series:12s} {d}: pm brackets {len((ev or {}).get('markets', []))}, kalshi {len(kms)}", flush=True)
    # lows + rain: inventory only (we do not trade lows -- X2 failed dev; Kalshi rain we paper is MONTHLY totals)
    for station, code in PM_LOW_TRY.items():
        tz = next(v[1] for v in KALSHI_HIGH.values() if v[0] == station)
        d0 = now.astimezone(ZoneInfo(tz)).date()
        n = max(len((pm.event(f"temp-{code}-{d.isoformat()}") or {}).get("markets", [])) for d in (d0, d0 + timedelta(days=1)))
        rows.append({"kind": "wx_low_inventory", "station": station, "pm": code, "date": d0.isoformat(), "n_brackets": n})
    return rows


def rain_inventory(pm: PolymarketUS) -> list[dict]:
    today = datetime.now(timezone.utc).date()
    out = []
    for d in (today, today + timedelta(days=1)):
        for q in (f"rain {d.strftime('%b')} {d.day}", "Where will it rain"):
            for m in pm.search(q):
                s = m.get("slug", "")
                if "rain-daily" in s and s not in {r["slug"] for r in out}:
                    r = {"kind": "rain_daily", "slug": s, "station": s.rsplit("-", 1)[-1].upper()}
                    r.update({f"p_{k}": v for k, v in pbook(pm, s).items()})
                    out.append(r)
    return out


# ------------------------------------------------------------------------------------------------- sports
def codes(abbr: str) -> list[str]:
    a = (abbr or "").lower()
    return ALIAS.get(a, [a.upper()])


def sports(pm: PolymarketUS, horizon_h: float = 60) -> list[dict]:
    rows = []
    now = pd.Timestamp.now(tz="UTC")
    for lg, (kgame, ktot) in SPORTS.items():
        evs = requests.get(f"https://gateway.polymarket.us/v2/leagues/{lg}/events", params={"limit": 200},
                           headers=H, timeout=30).json().get("events", [])
        evs = [e for e in evs if not e.get("closed") and pd.Timestamp(e["startTime"]) - now < pd.Timedelta(hours=horizon_h)
               and pd.Timestamp(e["startTime"]) > now]
        kg = (kget("/markets", {"series_ticker": kgame, "status": "open", "limit": 1000}) or {}).get("markets", [])
        kt = (kget("/markets", {"series_ticker": ktot, "status": "open", "limit": 1000}) or {}).get("markets", [])
        kev_g, kev_t = {}, {}
        for m in kg:
            kev_g.setdefault(m["event_ticker"], []).append(m)
        for m in kt:
            kev_t.setdefault(m["event_ticker"], []).append(m)
        print(f"  {lg}: polymarket US games in {horizon_h:.0f} h: {len(evs)}; kalshi open winner events {len(kev_g)}, "
              f"total events {len(kev_t)}", flush=True)
        for e in evs:
            # team codes from the moneyline's two sides ('participants' can be empty -> all() would match anything)
            sides = next((m.get("marketSides", []) for m in e.get("markets", []) if m.get("marketType") == "moneyline"), [])
            teams = [(s.get("team") or {}).get("abbreviation") for s in sides]
            if len(teams) != 2 or not all(teams):
                teams = [p.get("abbreviation") for p in (e.get("participants") or [])][:2]
            day = pd.Timestamp(e["startTime"]).tz_convert("America/New_York")
            days = {day.strftime("%y%b%d").upper(), (day - pd.Timedelta(days=1)).strftime("%y%b%d").upper()}

            def find(evmap):
                if len(teams) != 2 or not all(teams):
                    return None
                for et in evmap:
                    tail = et.split("-", 1)[1]
                    if tail[:7] in days and all(any(c in tail[7:] for c in codes(t)) for t in teams):
                        return et
                return None
            g_et, t_et = find(kev_g), find(kev_t)
            full = pm.event(e["slug"]) or e
            ml = [m for m in full.get("markets", []) if m.get("marketType") == "moneyline"]
            # full-game totals only: slug ends '-<date>-<n>pt5'; period totals carry '-1h-', '-4q-', ...
            tots = [m for m in full.get("markets", []) if m.get("marketType") == "totals"
                    and m["slug"].rsplit("-", 1)[-1].replace("pt", "").isdigit() and m["slug"].split("-")[-2].isdigit()]
            base = {"league": lg, "game": e.get("title"), "start": e["startTime"]}
            if ml:
                r = dict(base, kind="winner", slug=ml[0]["slug"], kalshi_event=g_et, matched=g_et is not None)
                r.update({f"p_{k}": v for k, v in pbook(pm, ml[0]["slug"]).items()})
                if g_et:
                    km = sorted(kev_g[g_et], key=lambda m: m["ticker"])[0]
                    r["ticker"] = km["ticker"]
                    r.update({f"k_{k}": v for k, v in kbook(km["ticker"]).items()})
                rows.append(r)
            if tots:   # the most balanced line on each venue (mid nearest 0.5)
                mid = lambda b: None if b["yes_bid"] is None or b["yes_ask"] is None else (b["yes_bid"] + b["yes_ask"]) / 2  # noqa: E731
                def qmid(m):
                    try:
                        return (float(m["bestBidQuote"]["value"]) + float(m["bestAskQuote"]["value"])) / 2
                    except (KeyError, TypeError, ValueError):
                        return 9.0
                tots = sorted(tots, key=lambda m: abs(qmid(m) - 0.5))[:2]
                best = None
                for m in tots:
                    b = pbook(pm, m["slug"])
                    if mid(b) is not None and (best is None or abs(mid(b) - .5) < abs(mid(best[1]) - .5)):
                        best = (m, b)
                r = dict(base, kind="total", kalshi_event=t_et, matched=t_et is not None, pm_lines=len(full.get("markets", [])))
                if best:
                    r["slug"] = best[0]["slug"]
                    r.update({f"p_{k}": v for k, v in best[1].items()})
                if t_et:
                    def kmid(m):
                        b, a = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 1)
                        return (b + a) / 2 if b > 0 and a < 1 else 9.0
                    ks = sorted(kev_t[t_et], key=lambda m: abs(kmid(m) - 0.5))
                    r["ticker"] = ks[0]["ticker"]
                    r["k_lines"] = len(kev_t[t_et])
                    r.update({f"k_{k}": v for k, v in kbook(ks[0]["ticker"]).items()})
                rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------------- summary
def summarize(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        print("no rows")
        return
    for v in ("p", "k"):
        if f"{v}_yes_ask" in df:
            df[f"{v}_spread"] = df[f"{v}_yes_ask"] - df[f"{v}_yes_bid"]
    q = lambda s: "n/a" if s.dropna().empty else f"{s.dropna().median():.3f}"  # noqa: E731
    qq = lambda s: "n/a" if s.dropna().empty else f"{s.dropna().median():.0f}"  # noqa: E731
    wx = df[df.kind == "wx_high"]
    if not wx.empty:
        print(f"\nWEATHER HIGHS: Kalshi series {wx.series.nunique()}, with a Polymarket US twin "
              f"{wx[wx.pm.notna()].series.nunique()} ({', '.join(sorted(wx[wx.pm.notna()].station.unique()))})")
        w = wx[wx.pm.notna() & wx.slug.notna()]
        print(f"  PM brackets {len(w)}, matched 1:1 to a Kalshi bracket {int(w.matched.sum())}")
        m = w[w.matched]
        for lab, sub in (("all", m), ("tail (PM YES ask <= .10)", m[m.p_yes_ask <= .10])):
            print(f"  {lab:26s} n={len(sub):3d}  median spread PM {q(sub.p_spread)} vs K {q(sub.get('k_spread', pd.Series(dtype=float)))}"
                  f" | median qty at YES bid PM {qq(sub.p_bid_qty)} vs K {qq(sub.get('k_bid_qty', pd.Series(dtype=float)))}"
                  f" | at YES ask PM {qq(sub.p_ask_qty)} vs K {qq(sub.get('k_ask_qty', pd.Series(dtype=float)))}")
        one = m[(m.p_yes_bid.isna()) | (m.p_yes_ask.isna())]
        onek = m[(m.k_yes_bid.isna()) | (m.k_yes_ask.isna())] if "k_yes_bid" in m else m.iloc[:0]
        print(f"  one-sided/empty books among matched: PM {len(one)}/{len(m)}, Kalshi {len(onek)}/{len(m)}")
        # rows for 'tomorrow' in a city whose local clock is 20:00-23:59 = the E1 decision window right now
        now = datetime.now(timezone.utc)
        dw = m[[(lambda tz: now.astimezone(ZoneInfo(tz)).hour >= 20 and r.date == (now.astimezone(ZoneInfo(tz)).date()
                 + timedelta(days=1)).isoformat())(KALSHI_HIGH[r.series][1]) for r in m.itertuples()]]
        if not dw.empty:
            print(f"  IN DECISION WINDOW now ({', '.join(sorted(dw.station.unique()))}, n={len(dw)}): qty at YES bid "
                  f"(= NO taker size) PM {list(dw.p_bid_qty.round(0))} vs K {list(dw.k_bid_qty.round(0))}")
    for kind in ("winner", "total"):
        s = df[df.kind == kind]
        for lg, sub in s.groupby("league"):
            mm = sub[sub.matched]
            print(f"{kind.upper():6s} {lg}: PM games {len(sub)}, matched to Kalshi {len(mm)} | median spread PM {q(sub.p_spread)} "
                  f"K {q(mm.get('k_spread', pd.Series(dtype=float)))} | median top qty bid/ask PM {qq(sub.p_bid_qty)}/{qq(sub.p_ask_qty)} "
                  f"K {qq(mm.get('k_bid_qty', pd.Series(dtype=float)))}/{qq(mm.get('k_ask_qty', pd.Series(dtype=float)))}")
    lows = df[df.kind == "wx_low_inventory"]
    if not lows.empty:
        print("PM daily LOW brackets (tomorrow): " + ", ".join(f"{r.pm}={r.n_brackets}" for r in lows.itertuples()))
    rain = df[df.kind == "rain_daily"]
    if not rain.empty:
        print(f"PM daily RAIN (any METAR rain code) markets: {len(rain)} stations {sorted(rain.station.unique())}; "
              f"median spread {q(rain.p_yes_ask - rain.p_yes_bid)}")


def main(argv) -> int:
    pm = PolymarketUS(env={})
    what = argv[0] if argv else "all"
    rows = []
    if what in ("all", "weather"):
        print("weather ...", flush=True)
        rows += weather(pm)
        rows += rain_inventory(pm)
    if what in ("all", "sports"):
        print("sports ...", flush=True)
        rows += sports(pm)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    out = ROOT / "data" / f"xvenue_map_{stamp}.json"
    out.write_text(json.dumps(rows, default=str))
    summarize(rows)
    print(f"\nrows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
