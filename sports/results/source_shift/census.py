"""source_shift census: every settled market (live /markets status=settled + /historical/markets) for a series universe.
Read-only public Kalshi GETs. One JSON line per series -> census_markets.jsonl (resumable: done series are skipped).
Raw rules text is stored once per (series, template hash) in rules_text.jsonl so template changes can be diffed.
Universe: cached series list (count_nowcast/series_live.json, 16,768 series) minus Sports/Crypto/Elections (Elections dropped 06:20Z for speed) and minus Financials/Commodities
hourly/daily/15-min/custom/one_off price ladders; recurring-data categories first, Elections last.
Usage: python census.py [max_series]
"""
import hashlib, json, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

API = "https://api.elections.kalshi.com/trade-api/v2"
HERE = Path(__file__).resolve().parent
OUT, RULES, LOG = HERE / "census_markets.jsonl", HERE / "rules_text.jsonl", HERE / "census.log"
LOCK, RLOCK = threading.Lock(), threading.Lock()
GAP = 0.15  # ~6.5 req/s max shared by all threads (~06:27Z: lowered so other agents sharing the IP are not starved by 429s)
_last = [0.0]
MONTHS = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"


def get(path, params):
    for a in range(10):
        with RLOCK:
            w = GAP - (time.time() - _last[0])
            if w > 0:
                time.sleep(w)
            _last[0] = time.time()
        try:
            r = requests.get(API + path, params=params, timeout=60, headers={"User-Agent": "research-script/0.1"})
            if r.status_code == 429:
                time.sleep(min(3 * 2 ** a, 60)); continue
            if r.status_code >= 500:
                time.sleep(2 * (a + 1)); continue
            if r.status_code == 404:
                return {"markets": []}
            r.raise_for_status(); return r.json()
        except requests.RequestException:
            time.sleep(2 * (a + 1))
    return {"markets": [], "_error": "gave up"}


def template(text, subs):
    t = text or ""
    for s in subs:
        if s and len(s) > 1:
            t = t.replace(s, "<S>")
    t = re.sub(MONTHS, "<M>", t)
    t = re.sub(r"[-+]?\$?\d[\d,]*(\.\d+)?%?", "#", t)
    return re.sub(r"\s+", " ", t).strip()


def h(t):
    return hashlib.sha1(t.encode()).hexdigest()[:10]


def pull(series):
    rows, seen_tpl = [], {}
    for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
        cur = None
        while True:
            p = {"series_ticker": series, "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            d = get(path, p); mk = d.get("markets", []); cur = d.get("cursor")
            for m in mk:
                subs = [m.get("yes_sub_title"), m.get("no_sub_title"), m.get("subtitle"), m.get("custom_strike") and json.dumps(m.get("custom_strike"))]
                tp, ts_ = template(m.get("rules_primary"), subs), template(m.get("rules_secondary"), subs)
                hp, hs = h(tp), h(ts_)
                for k, raw, tt in ((hp, m.get("rules_primary"), tp), (hs, m.get("rules_secondary"), ts_)):
                    if k not in seen_tpl:
                        seen_tpl[k] = 1
                        with LOCK:
                            RULES.open("a").write(json.dumps({"series": series, "hash": k, "event": m["event_ticker"], "ticker": m["ticker"],
                                                              "created": m.get("created_time"), "template": tt, "raw": raw}) + "\n")
                rows.append({
                    "t": m["ticker"], "e": m["event_ticker"], "cr": m.get("created_time"), "op": m.get("open_time"), "cl": m.get("close_time"),
                    "ee": m.get("expected_expiration_time"), "le": m.get("latest_expiration_time"), "st": m.get("settlement_ts"),
                    "res": m.get("result"), "sv": m.get("settlement_value_dollars"), "ev": m.get("expiration_value"),
                    "fs": m.get("floor_strike"), "cs": m.get("cap_strike"), "sty": m.get("strike_type"), "last": m.get("last_price_dollars"),
                    "prev": m.get("previous_price_dollars"), "vol": m.get("volume_fp"), "ys": (m.get("yes_sub_title") or "")[:80],
                    "status": m.get("status"), "hp": hp, "hs": hs, "src": path[1:5]})
            if not cur or not mk:
                break
    with LOCK:
        OUT.open("a").write(json.dumps({"series": series, "n": len(rows), "markets": rows}) + "\n")
        LOG.open("a").write(f"{time.strftime('%H:%M:%S')} {series} {len(rows)}\n")


def universe():
    d = json.load(open(HERE.parent / "count_nowcast" / "series_live.json"))
    s = d["series"] if isinstance(d, dict) else d
    order = ["Economics", "Climate and Weather", "Science and Technology", "Entertainment", "Companies", "Health", "Transportation",
             "Social", "World", "Commodities", "Financials", "Mentions", "Exotics", "Culture", "Business", "Education", "Politics", "Elections"]
    keep = [x for x in s if x["category"] not in ("Sports", "Crypto", "Elections")
            and not (x["category"] in ("Financials", "Commodities") and x["frequency"] in ("hourly", "daily", "fifteen_min", "custom", "one_off"))]
    # 06:25Z: skip the big daily weather ladders (KXHIGH*/KXLOW*/KXTEMP*, thousands of markets each) -- weather holds are covered by
    # winners_transfer delays.parquet; KXHIGHNY/KXHIGHDEN kept for the 2026-08-14 NWS -> Weather Company switch
    keep = [x for x in keep if not re.match(r"^KX(HIGH|LOW|TEMP)", x["ticker"]) or x["ticker"] in ("KXHIGHNY", "KXHIGHDEN")]
    keep.sort(key=lambda x: (order.index(x["category"]) if x["category"] in order else 99, x["ticker"]))
    return [x["ticker"] for x in keep]


if __name__ == "__main__":
    U = universe()
    done = set()
    if OUT.exists():
        for line in OUT.open():
            try:
                done.add(json.loads(line)["series"])
            except Exception:
                pass
    todo = [x for x in U if x not in done]
    if len(sys.argv) > 1:
        todo = todo[: int(sys.argv[1])]
    print("universe", len(U), "done", len(done), "todo", len(todo), flush=True)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(pull, todo))
    print("finished", flush=True)
