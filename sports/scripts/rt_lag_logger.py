"""rt_review_lag FORWARD read-only logger (notes/rt_review_lag.md, 'Forward logger').

Measures what history cannot: how fast RT counts reviews vs when outlets publish them, and how the Kalshi book moves
in the hours after a review embargo. No orders, no keys, no logins.

Per tick (launchd every 300 s), for each open KXRT event closing within 12 days:
  - Kalshi markets of the event (1 GET via kx_throttle): bid/ask/sizes/volume -> book.jsonl
  - RT film page (criticsScore: scorePercent, likedCount, notLikedCount, reviewCount) -> rt.jsonl
      cadence: every tick while HOT (RT n == 0, or < 72 h since the first review we saw), else every 6th tick
  - Metacritic movie critic feed (robots allows; the endpoint metacritic.com calls) -> each NEW review once ->
    its outlet page fetched once for article:published_time (robots.txt honoured) -> mc_new.jsonl
Slugs: RT slug from q013's deployed state.json if present (q013 resolves them), else data/rt_lag/overrides.json;
RT /search is robots-disallowed, so this logger never calls it. MC slug = slugify(title)[-2026].
401/403 from any site: stop fetching that site for 24 h and log it. Never bypassed.

    python rt_lag_logger.py tick | report
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import requests  # noqa: E402

from kx_throttle import kget  # noqa: E402

BASE = Path(os.environ.get("RT_LAG_HOME", str(Path.home() / ".claude" / "quant-sports")))
D = BASE / "data" / "rt_lag"
D.mkdir(parents=True, exist_ok=True)
Q013_STATE = Path.home() / ".claude" / "quant-kalshi" / "data" / "q013" / "state.json"
UA = "Mozilla/5.0 (compatible; quant-research-readonly)"
S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept-Language": "en-US"})
RT = "https://www.rottentomatoes.com"
MC = "https://backend.metacritic.com/reviews/metacritic/critic/movies/{slug}/web"
HORIZON_D = 12
HOT_H = 72
COLD_EVERY = 6
PATS = [
    r'<meta[^>]+property="article:published_time"[^>]+content="([^"]+)"',
    r'<meta[^>]+content="([^"]+)"[^>]+property="article:published_time"',
    r'"datePublished"\s*:\s*"([^"]+)"',
    r'<time[^>]+datetime="([^"]+)"',
]


def now():
    return int(time.time())


def append(name, rows):
    if rows:
        with open(D / name, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def jload(p, default):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return default


def slugify(t):
    t = t.lower().replace("&", "and")
    t = re.sub(r"[’'\".:!?,()]", "", t)
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")


def film_title(m):
    t = m.get("title") or ""
    x = re.match(r"^If (.*?) has a Tomatometer", m.get("rules_primary") or "")
    return x.group(1).strip() if x else t


_robots = {}


def allowed(url):
    import urllib.robotparser
    from urllib.parse import urlparse
    u = urlparse(url)
    base = f"{u.scheme}://{u.netloc}"
    if base not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = S.get(base + "/robots.txt", timeout=15)
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        except requests.RequestException:
            rp.parse([])
        _robots[base] = rp
    return _robots[base].can_fetch(UA, url)


def blocked(st, site):
    return now() < st.get("blocked", {}).get(site, 0)


def block(st, site, code, errs):
    st.setdefault("blocked", {})[site] = now() + 86400
    errs.append(f"{site} returned {code}: paused 24 h (not bypassed)")


def rt_page(path):
    r = S.get(RT + path, timeout=30)
    if r.status_code != 200:
        return r.status_code, None
    m = re.search(r'<script[^>]*id="media-scorecard-json"[^>]*>(.*?)</script>', r.text, re.S)
    if not m:
        return "nojson", None
    c = (json.loads(m.group(1)).get("criticsScore") or {})
    sp = c.get("scorePercent") or c.get("score")
    score = float(str(sp).rstrip("%")) if sp not in (None, "") else None
    return 200, {"score": score, "n": c.get("reviewCount"), "liked": c.get("likedCount"),
                 "not_liked": c.get("notLikedCount")}


def tick():
    ts = now()
    st = jload(D / "state.json", {"events": {}, "tick": 0})
    st["tick"] = st.get("tick", 0) + 1
    ov = jload(D / "overrides.json", {})
    q13 = jload(Q013_STATE, {}).get("events", {})
    errs = []
    r = kget("/markets", {"series_ticker": "KXRT", "status": "open", "limit": 1000}) or {}
    mk = r.get("markets") or []
    evs = {}
    for m in mk:
        evs.setdefault(m["event_ticker"], []).append(m)
    book = []
    for ev, ms in evs.items():
        close = dt.datetime.fromisoformat(ms[0]["close_time"].replace("Z", "+00:00")).timestamp()
        if close - ts > HORIZON_D * 86400:
            continue
        for m in ms:
            book.append({"ts": ts, "event": ev, "ticker": m["ticker"], "floor": m.get("floor_strike"),
                         "close_ts": close, "yes_bid": m.get("yes_bid_dollars"), "yes_ask": m.get("yes_ask_dollars"),
                         "bid_size": m.get("yes_bid_size_fp"), "ask_size": m.get("yes_ask_size_fp"),
                         "volume": m.get("volume_fp"), "last": m.get("last_price_dollars")})
        e = st["events"].setdefault(ev, {"title": film_title(ms[0])})
        rt_path = ov.get(ev, {}).get("rt") or (q13.get(ev) or {}).get("ref")
        hot = e.get("first_review_ts") is None or ts - e["first_review_ts"] < HOT_H * 3600
        if rt_path and not blocked(st, "rt") and (hot or st["tick"] % COLD_EVERY == 0):
            try:
                code, sc = rt_page(rt_path)
                if code in (401, 403):
                    block(st, "rt", code, errs)
                elif sc:
                    append("rt.jsonl", [{"ts": ts, "event": ev, **sc}])
                    if (sc.get("n") or 0) > 0 and e.get("first_review_ts") is None:
                        e["first_review_ts"] = ts
                time.sleep(2)
            except Exception as x:  # noqa: BLE001
                errs.append(f"rt {ev}: {x}")
        if not blocked(st, "mc") and (hot or st["tick"] % COLD_EVERY == 0):
            slug = ov.get(ev, {}).get("mc") or e.get("mc_slug")
            seen = set(e.get("mc_seen", []))
            for cand in ([slug] if slug else [slugify(e["title"]) + "-2026", slugify(e["title"])]):
                try:
                    items, off = [], 0
                    while True:
                        rr = S.get(MC.format(slug=cand), params={"offset": off, "limit": 50}, timeout=30)
                        time.sleep(2)
                        if rr.status_code in (401, 403):
                            block(st, "mc", rr.status_code, errs)
                            break
                        if rr.status_code != 200:
                            break
                        dd = rr.json().get("data") or {}
                        got = dd.get("items") or []
                        items += got
                        off += len(got)
                        if not got or off >= dd.get("totalResults", 0):
                            break
                except Exception as x:  # noqa: BLE001
                    errs.append(f"mc {ev}: {x}")
                    continue
                if not items and not slug:
                    continue
                yr = str(dt.datetime.fromtimestamp(close, dt.UTC).year)
                if not slug and any((x.get("date") or yr)[:4] < str(int(yr) - 1) for x in items):
                    continue  # an older film with the same title
                e["mc_slug"] = cand
                new = []
                for x in items:
                    key = f"{x.get('publicationName')}|{x.get('url')}"
                    if key in seen:
                        continue
                    seen.add(key)
                    row = {"first_seen": ts, "event": ev, "publication": x.get("publicationName"),
                           "mc_score": x.get("score"), "mc_date": x.get("date"), "url": x.get("url"),
                           "published": None, "status": None, "backfill": e.get("mc_init") is None}
                    u = x.get("url") or ""
                    if u.startswith("http") and not row["backfill"]:
                        try:
                            if not allowed(u):
                                row["status"] = "robots_disallow"
                            else:
                                pr = S.get(u, timeout=30)
                                row["status"] = pr.status_code
                                if pr.status_code == 200:
                                    for p in PATS:
                                        mm = re.search(p, pr.text)
                                        if mm:
                                            row["published"] = mm.group(1)
                                            break
                            time.sleep(2)
                        except Exception as xx:  # noqa: BLE001
                            row["status"] = f"err {type(xx).__name__}"
                    new.append(row)
                append("mc_new.jsonl", new)
                e["mc_seen"] = sorted(seen)
                e["mc_init"] = e.get("mc_init") or ts
                break
    append("book.jsonl", book)
    st["last_tick"] = ts
    json.dump(st, open(D / "state.json", "w"))
    (D / "heartbeat").write_text(json.dumps({"ts": ts, "events": len(evs), "errors": errs}))
    if errs:
        append("errors.jsonl", [{"ts": ts, "errors": errs}])


def report():
    import collections
    rt = [json.loads(x) for x in open(D / "rt.jsonl")] if (D / "rt.jsonl").exists() else []
    mc = [json.loads(x) for x in open(D / "mc_new.jsonl")] if (D / "mc_new.jsonl").exists() else []
    print("rt rows", len(rt), "mc reviews", len(mc), "live (non-backfill)", sum(not r["backfill"] for r in mc))
    by = collections.Counter(r["event"] for r in mc if not r["backfill"])
    print("live MC reviews per event", dict(by))


if __name__ == "__main__":
    {"tick": tick, "report": report}[sys.argv[1]]()
