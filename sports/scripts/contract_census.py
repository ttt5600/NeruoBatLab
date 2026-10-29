"""Q010 contract census across venues (READ-ONLY public data; no keys, no orders, no account calls).

    ../.venv_quant/bin/python scripts/contract_census.py fetch      # cache raw pages to data/census/raw_<date>/
    ../.venv_quant/bin/python scripts/contract_census.py build      # family table -> data/census/census_<date>.csv

Sources:
  Kalshi        https://api.elections.kalshi.com/trade-api/v2  /series (all), /events?status=open&with_nested_markets=true
                (cursor paged, <= 5 req/s). Per-market fields used: volume_24h_fp, open_interest_fp, yes_bid/ask_dollars.
  Polymarket US https://gateway.polymarket.us/v1/markets?closed=false (offset paged) -- list gives best bid/ask and
                feeCoefficient; /markets/{slug}/bbo (sampled, <= SAMPLE_PER_FAMILY per family) gives openInterest and
                lifetime sharesTraded (no 24h volume field exists on the public gateway).
  ForecastEx    https://forecastex.com/api/products (catalog + per-product open_interest) and the latest daily price file
                /api/download?type=prices&date=YYYYMMDD (pair_quantity per contract for that day; trades, no bid/ask).
Family = Kalshi series_ticker; Polymarket first two slug tokens; ForecastEx product_id.
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import median

import requests

ROOT = Path(__file__).resolve().parents[1]
DAY = "2026-10-29"
RAW = ROOT / "data" / "census" / f"raw_{DAY}"
OUT = ROOT / "data" / "census" / f"census_{DAY}.csv"
KAPI = "https://api.elections.kalshi.com/trade-api/v2"
PM = "https://gateway.polymarket.us/v1"
FX = "https://forecastex.com/api"
H = {"User-Agent": "Mozilla/5.0 (research; read-only census)"}
SAMPLE_PER_FAMILY = 4
_last = [0.0]


def get(url, params=None, min_interval=0.21, retries=6):
    for a in range(retries):
        w = _last[0] + min_interval - time.monotonic()
        if w > 0:
            time.sleep(w)
        _last[0] = time.monotonic()
        try:
            r = requests.get(url, params=params, headers=H, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** a)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** a)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r
    raise RuntimeError(f"gave up {url} {params}")


def fetch_kalshi():
    p = RAW / "kalshi_series.json"
    if not p.exists():
        series, cur = [], None
        while True:
            prm = {"limit": 1000}
            if cur:
                prm["cursor"] = cur
            d = get(f"{KAPI}/series", prm).json()
            series += d.get("series") or []
            cur = d.get("cursor")
            if not cur or not d.get("series"):
                break
        p.write_text(json.dumps(series))
        print("kalshi series", len(series))
    i, cur = 0, None
    while True:
        f = RAW / f"kalshi_events_{i:04d}.json"
        if f.exists():
            d = json.loads(f.read_text())
        else:
            prm = {"status": "open", "limit": 200, "with_nested_markets": "true"}
            if cur:
                prm["cursor"] = cur
            d = get(f"{KAPI}/events", prm).json()
            f.write_text(json.dumps(d))
        cur = d.get("cursor")
        i += 1
        if i % 25 == 0:
            print("kalshi events pages", i, flush=True)
        if not cur or not d.get("events"):
            break
    print("kalshi event pages", i)


def fetch_pm():
    off, i = 0, 0
    while True:
        f = RAW / f"pm_markets_{i:04d}.json"
        if f.exists():
            ms = json.loads(f.read_text())
        else:
            r = get(f"{PM}/markets", {"limit": 500, "offset": off, "closed": "false"}, min_interval=0.07)
            ms = (r.json() if r else {}).get("markets", [])
            f.write_text(json.dumps(ms))
        i += 1
        off += 500
        if len(ms) < 500:
            break
    print("pm market pages", i)
    # sample BBO per family
    allm = pm_markets()
    fam = defaultdict(list)
    for m in allm:
        fam[pm_family(m)].append(m)
    bf = RAW / "pm_bbo_sample.json"
    got = json.loads(bf.read_text()) if bf.exists() else {}
    for k, ms in fam.items():
        step = max(1, len(ms) // SAMPLE_PER_FAMILY)
        for m in ms[::step][:SAMPLE_PER_FAMILY]:
            if m["slug"] in got:
                continue
            r = get(f"{PM}/markets/{m['slug']}/bbo", min_interval=0.07)
            got[m["slug"]] = (r.json() if r else {}).get("marketData") if r else None
    bf.write_text(json.dumps(got))
    print("pm bbo sampled", len(got))


def fetch_fx():
    p = RAW / "fx_products.json"
    if not p.exists():
        data, page = [], 1
        while True:
            body = get(f"{FX}/products", {"page": page}).json().get("body", {})
            chunk = body.get("data", [])
            data += json.loads(chunk) if isinstance(chunk, str) else chunk
            page += 1
            if page > int(body.get("total_pages") or 0):
                break
        p.write_text(json.dumps(data))
        print("fx products", len(data))
    for back in range(1, 6):
        ds = (date.fromisoformat(DAY) - timedelta(days=back)).strftime("%Y%m%d")
        f = RAW / f"fx_prices_{ds}.csv"
        if f.exists():
            return
        r = get(f"{FX}/download", {"type": "prices", "date": ds})
        if r is not None and len(r.content) > 1000:
            f.write_bytes(r.content)
            print("fx prices", ds, len(r.content))
            return


# ---------------------------------------------------------------- build
def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


DATE_TOK = re.compile(r"^(\d{4}|\d{2,4}[a-z]{3}\d{0,4}|[a-z]{3}\d{2,4}|\d{6,8})$")


def pm_family(m):
    """Product prefix + league/topic, e.g. astatc-nfl, tc-temp, aaagaspc-usgas (482 families on 10-29)."""
    return "-".join(m["slug"].split("-")[:2])


def pm_markets():
    ms = []
    for f in sorted(RAW.glob("pm_markets_*.json")):
        ms += json.loads(f.read_text())
    return ms


# Hand annotation (2026-10-29) of the families assessed in notes/contract_census_2026-10-29.md. Key = exact family or a
# prefix ending in '*'. Value = (external reference forecast, fund status vs FUND_BOOK s3). Unlisted = "not assessed".
ANNOT = {
    "KXHIGH*": ("NWS/NBM/GFS/ECMWF/HRRR/GEM point forecasts", "LIVE E1/E2/E4"),
    "KXLOWT*": ("same forecast models", "DEAD X2 (lows)"),
    "KXRAIN": ("NWS PoP / NBM QPF", "DEAD R1/R2 (sealed FAIL)"),
    "KXRAINWKND": ("NWS PoP / NBM QPF", "dead-adjacent R1 (weekend horizon untested)"),
    "KXAAAGASD": ("AAA prior day + RBOB futures", "DEAD N6 (0 bets)"),
    "KXAAAGASD*": ("AAA state prior day", "dead-adjacent N6 (state dailies untested)"),
    "KXAAAGASW": ("AAA daily (partial obs) + RBOB futures pass-through", "dead-adjacent N6; new mechanism (RBOB lead) untested; top-15 G-GAS"),
    "KXAAAGASM": ("AAA daily (partial obs) + RBOB futures pass-through", "dead-adjacent N6; untested; top-15 G-GAS"),
    "KXCPI*": ("Cleveland Fed inflation nowcast", "DEAD N2 + econ consensus-reject"),
    "KXTSA*": ("TSA daily throughput (partial obs of the week)", "N7 deprioritized, never tested; top-15 G-TSA"),
    "KXTRUMPMENTION": ("none (transcripts after)", "DEAD M3 taker / Q003 maker"),
    "KXEARNINGSMENTION*": ("none", "DEAD-adjacent mentions (M3/Q003)"),
    "KXRT": ("Rotten Tomatoes live Tomatometer + review count (partial obs)", "NEW top-15 G-RT"),
    "KXMC": ("Metacritic live score (partial obs)", "NEW top-15 G-RT (bundled)"),
    "KXTRUMPAPPROVE": ("RCP average recomputable from listed polls + poll release cadence", "NEW top-15 G-APPROVAL"),
    "KXAPRPOTUS": ("RCP average recomputable from listed polls", "NEW top-15 G-APPROVAL"),
    "KXTRUMPVH": ("VoteHub average recomputable", "NEW top-15 G-APPROVAL"),
    "KXGENERICBALLOTVOTEHUB": ("VoteHub average recomputable", "NEW top-15 G-APPROVAL"),
    "KXTRUTHSOCIAL": ("running post count (Roll Call/Factba.se, partial obs) + base rate", "NEW top-15 G-TRUTH"),
    "KXTRUTHSOCIALD": ("running post count (partial obs)", "NEW top-15 G-TRUTH"),
    "KXBTCD": ("Deribit options IV + CF BRTI spot", "NEW top-15 G-CRYPTO (L19 K-FAV fav-maker DEAD on same series)"),
    "KXETHD": ("Deribit options IV + CF ETH index", "NEW top-15 G-CRYPTO"),
    "KXBTC": ("Deribit options IV", "NEW top-15 G-CRYPTO"),
    "KXETH": ("Deribit options IV", "NEW top-15 G-CRYPTO"),
    "KXFEDDECISION": ("CME FedWatch (30-day fed funds futures)", "NEW top-15 G-FED (social sweep #9, untested)"),
    "KXFED": ("CME FedWatch", "NEW top-15 G-FED"),
    "KXWTI": ("NYMEX/ICE WTI futures + CME options IV", "NEW top-15 G-COMMOD"),
    "KXWTIW": ("WTI futures + options IV", "NEW top-15 G-COMMOD"),
    "KXBRENTW": ("Brent futures + options IV", "NEW top-15 G-COMMOD"),
    "KXGOLDW": ("COMEX gold futures + options IV", "NEW top-15 G-COMMOD"),
    "KXNATGASW": ("Henry Hub futures + options IV", "NEW top-15 G-COMMOD"),
    "KXYTVIEWSW": ("YouTube Charts daily views (partial obs)", "NEW top-15 G-YOUTUBE"),
    "KXALBUMEQUIV": ("HitsDailyDouble first-week projections + Spotify/Kworb daily streams", "NEW top-15 G-CHARTS"),
    "KXTOPSONG": ("Spotify/Apple daily charts, Kworb, Hits projections", "NEW top-15 G-CHARTS"),
    "KXTOPALBUM": ("HitsDailyDouble projections", "NEW top-15 G-CHARTS"),
    "KXBBSINGLE": ("Spotify/Kworb daily streams", "NEW top-15 G-CHARTS"),
    "KXHORMUZWEEKLY": ("IMF PortWatch daily transit calls (partial obs, lagged)", "NEW top-15 G-PORTWATCH"),
    "KXPANAMAWEEKLY": ("IMF PortWatch", "NEW top-15 G-PORTWATCH"),
    "KXSUEZWEEKLY": ("IMF PortWatch", "NEW top-15 G-PORTWATCH"),
    "KXBABELMANDEBWEEKLY": ("IMF PortWatch", "NEW top-15 G-PORTWATCH"),
    "KXTXERCOTPEAKD": ("ERCOT 7-day hourly load forecast", "NEW top-15 G-ERCOT"),
    "KXMISOINDW": ("MISO day-ahead LMPs (partial obs)", "NEW top-15 G-ERCOT"),
    "KXERCOTNORTHW": ("ERCOT day-ahead SPPs", "NEW top-15 G-ERCOT"),
    "KXPJMDOMW": ("PJM day-ahead LMPs", "NEW top-15 G-ERCOT"),
    "KXNETFLIXRANKSHOW*": ("FlixPatrol daily Netflix top-10 (partial obs)", "NEW top-15 G-NETFLIX"),
    "KXEIACRUDEW": ("API Tuesday inventory report + analyst consensus", "NEW top-15 G-EIA"),
    "KXSPACEXCOUNT": ("launch manifests (NextSpaceflight), FAA notices", "lead, annual"),
    "MORT1": ("Mortgage News Daily daily index; MBA weekly", "NEW top-15 G-MORT"),
    "MORT2": ("Mortgage News Daily daily index", "NEW top-15 G-MORT"),
    "FFDEC": ("CME FedWatch", "NEW top-15 G-FED; Q008 cross-venue running"),
    "FF": ("CME FedWatch", "NEW top-15 G-FED"),
    "UH*": ("NWS/NBM point forecasts (WU settlement)", "LIVE E5"),
    "UL*": ("same", "DEAD (Q007 lows)"),
    "IJC": ("consensus / continuing claims", "DEAD econ consensus-reject"),
    "FES": ("ES futures", "DEAD Q007 do-not-pursue"),
    "FNQ": ("NQ futures", "DEAD Q007 do-not-pursue"),
    "tc-temp": ("NWS/NBM forecasts", "F4 paper (PM weather)"),
    "rainc-rain": ("NWS hourly PoP; METAR rain codes", "new settlement (any METAR rain code), thin"),
    "raincli-rain": ("NWS PoP", "dead-adjacent R1"),
    "rotc-t": ("Rotten Tomatoes live score", "NEW top-15 G-RT (PM copy)"),
    "rdc-usfed": ("CME FedWatch", "NEW top-15 G-FED (PM copy)"),
    "ccrc-bilbrd": ("HitsDailyDouble projections", "NEW top-15 G-CHARTS (PM copy)"),
    "ccrc-ntflx": ("FlixPatrol", "NEW top-15 G-NETFLIX (PM copy)"),
}


def annotate(fam, venue, category):
    if fam in ANNOT:
        return ANNOT[fam]
    for k, v in ANNOT.items():
        if k.endswith("*") and fam.startswith(k[:-1]) and not (k == "KXAAAGASD*" and fam == "KXAAAGASD"):
            return v
    if (venue == "kalshi" and category == "Sports") or (venue == "polymarket_us" and category == "sports"):
        return ("Pinnacle / sharp-book close", "sports: market made off sharp books (FUND_BOOK 3d M2/S1-S3); not assessed per family")
    return ("", "not assessed")


def build():
    rows = []
    # ---- Kalshi
    series = {s["ticker"]: s for s in json.loads((RAW / "kalshi_series.json").read_text())}
    fam = defaultdict(lambda: {"events": 0, "markets": 0, "v24": 0.0, "oi": 0.0, "vol": 0.0, "spreads": [],
                               "two_sided": 0, "rules": "", "title": "", "close": []})
    for f in sorted(RAW.glob("kalshi_events_*.json")):
        for e in json.loads(f.read_text()).get("events") or []:
            k = fam[e["series_ticker"]]
            k["events"] += 1
            k["title"] = k["title"] or e.get("title", "")
            k["cat"] = e.get("category")
            k["src"] = e.get("settlement_sources")
            for m in e.get("markets") or []:
                if m.get("status") not in ("active", "open", None):
                    continue
                k["markets"] += 1
                k["v24"] += fnum(m.get("volume_24h_fp")) or 0
                k["oi"] += fnum(m.get("open_interest_fp")) or 0
                k["vol"] += fnum(m.get("volume_fp")) or 0
                b, a = fnum(m.get("yes_bid_dollars")), fnum(m.get("yes_ask_dollars"))
                if b and a and b > 0 and a < 1 and a > b:
                    k["spreads"].append(a - b)
                    k["two_sided"] += 1
                if not k["rules"]:
                    k["rules"] = (m.get("rules_primary") or "")[:300]
                k["close"].append(m.get("close_time", ""))
    for t, k in fam.items():
        s = series.get(t, {})
        srcs = s.get("settlement_sources") or k.get("src") or []
        rows.append({
            "venue": "kalshi", "family": t, "title": s.get("title") or k["title"], "category": s.get("category") or k.get("cat"),
            "tags": "|".join(s.get("tags") or []), "cadence": s.get("frequency", ""), "events_open": k["events"],
            "markets_open": k["markets"], "vol24h_contracts": round(k["v24"], 1), "open_interest": round(k["oi"], 1),
            "lifetime_vol_open_mkts": round(k["vol"], 1),
            "median_spread": round(median(k["spreads"]), 4) if k["spreads"] else "", "two_sided_mkts": k["two_sided"],
            "fee_type": f"{s.get('fee_type', '')} x{s.get('fee_multiplier', '')}",
            "settlement_source": "; ".join(x.get("name", "") for x in srcs)[:200],
            "rules_example": k["rules"].replace("\n", " "), "next_close": min([c for c in k["close"] if c] or [""]),
        })
    # ---- Polymarket US
    bbo = json.loads((RAW / "pm_bbo_sample.json").read_text())
    pf = defaultdict(lambda: {"markets": 0, "spreads": [], "two_sided": 0, "oi": [], "traded": [], "cat": set(),
                              "type": set(), "q": "", "desc": "", "fee": set(), "end": []})
    for m in pm_markets():
        k = pf[pm_family(m)]
        k["markets"] += 1
        k["cat"].add(m.get("category") or "")
        k["type"].add(m.get("sportsMarketTypeV2") or m.get("marketType") or "")
        k["q"] = k["q"] or m.get("question", "")
        k["desc"] = k["desc"] or (m.get("description") or "")[:300]
        k["fee"].add(str(m.get("feeCoefficient")))
        k["end"].append(m.get("endDate") or "")
        b = fnum((m.get("bestBidQuote") or {}).get("value"))
        a = fnum((m.get("bestAskQuote") or {}).get("value"))
        if b and a and b > 0 and a < 1 and a > b:
            k["spreads"].append(a - b)
            k["two_sided"] += 1
        d = bbo.get(m["slug"])
        if d:
            k["oi"].append(fnum(d.get("openInterest")) or 0)
            k["traded"].append(fnum(d.get("sharesTraded")) or 0)
    for t, k in pf.items():
        n_s = len(k["oi"])
        rows.append({
            "venue": "polymarket_us", "family": t, "title": k["q"], "category": "|".join(sorted(k["cat"])),
            "tags": "|".join(sorted(k["type"])), "cadence": "", "events_open": "", "markets_open": k["markets"],
            "vol24h_contracts": "",
            # extrapolated from the BBO sample: mean per sampled market x markets_open (flagged in col name)
            "open_interest": round(sum(k["oi"]) / n_s * k["markets"], 1) if n_s else "",
            "lifetime_vol_open_mkts": round(sum(k["traded"]) / n_s * k["markets"], 1) if n_s else "",
            "median_spread": round(median(k["spreads"]), 4) if k["spreads"] else "", "two_sided_mkts": k["two_sided"],
            "fee_type": f"taker {'/'.join(sorted(k['fee']))}*p(1-p); maker 0", "settlement_source": "",
            "rules_example": k["desc"].replace("\n", " "), "next_close": min([c for c in k["end"] if c] or [""]),
            "pm_bbo_sampled": n_s,
        })
    # ---- ForecastEx
    prods = json.loads((RAW / "fx_products.json").read_text())
    pfiles = sorted(RAW.glob("fx_prices_*.csv"))
    vol = defaultdict(float)
    ncon = defaultdict(int)
    fx_day = ""
    if pfiles:
        fx_day = pfiles[-1].stem.split("_")[-1]
        rd = csv.DictReader(io.StringIO(pfiles[-1].read_text()))
        for r in rd:
            code = (r.get("event_contract") or "").split("_")[0]
            if r.get("subtype", "YES") == "YES":
                ncon[code] += 1
                vol[code] += fnum(r.get("pair_quantity")) or 0
    for p in prods:
        pid = p.get("product_id")
        rows.append({
            "venue": "forecastex", "family": pid, "title": p.get("name"), "category": p.get("category"),
            "tags": p.get("calculation_method", ""), "cadence": p.get("frequency_type", ""), "events_open": "",
            "markets_open": ncon.get(pid, 0), "vol24h_contracts": round(vol.get(pid, 0), 1),
            "open_interest": p.get("open_interest"), "lifetime_vol_open_mkts": "", "median_spread": "",
            "two_sided_mkts": "", "fee_type": "$0.01/contract (IBKR $0 commission)",
            "settlement_source": f"{p.get('source_agency', '')} {p.get('source_agency_url', '')}"[:200],
            "rules_example": (p.get("description") or "")[:300].replace("\n", " "), "next_close": "",
            "fx_price_file_day": fx_day,
        })
    cols = ["venue", "family", "title", "category", "tags", "cadence", "events_open", "markets_open", "vol24h_contracts",
            "open_interest", "lifetime_vol_open_mkts", "median_spread", "two_sided_mkts", "fee_type", "settlement_source",
            "rules_example", "next_close", "pm_bbo_sampled", "fx_price_file_day", "reference_forecast", "fund_status"]
    for r in rows:
        r["reference_forecast"], r["fund_status"] = annotate(str(r["family"]), r["venue"], str(r.get("category") or ""))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print("wrote", OUT, len(rows))


def settled_counts(series_list):
    """Count settled events per Kalshi series (cadence / power inputs). Cached; no prices or outcomes are read."""
    p = RAW / "kalshi_settled_event_counts.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    for s in series_list:
        if s in out:
            continue
        n, cur, dates = 0, None, []
        for _ in range(60):
            prm = {"series_ticker": s, "status": "settled", "limit": 200}
            if cur:
                prm["cursor"] = cur
            r = get(f"{KAPI}/events", prm).json()
            es = r.get("events") or []
            n += len(es)
            dates += [e.get("strike_date") for e in es if e.get("strike_date")]
            cur = r.get("cursor")
            if not cur or not es:
                break
        out[s] = {"settled_events": n, "first_strike": min(dates) if dates else "",
                  "last_strike": max(dates) if dates else ""}
        print(s, out[s], flush=True)
        p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[1] == "settled":
        settled_counts(sys.argv[2:])
        sys.exit(0)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        which = sys.argv[2:] or ["kalshi", "pm", "fx"]
        for w in which:
            {"kalshi": fetch_kalshi, "pm": fetch_pm, "fx": fetch_fx}[w]()
    elif cmd == "build":
        build()
