"""Rumor / news-flow signals vs Kalshi + Polymarket (user 2026-10-22). Free, credential-free sources only. No orders.

LIVE PIPELINE (first pass, LLM-free):
    python scripts/rumor_signals.py markets        # open Kalshi + Polymarket markets -> keyword index (data/rumor/markets.json)
    python scripts/rumor_signals.py ingest         # one tick: GDELT last 15 min + Bluesky search + Reddit RSS + Google News RSS
                                                   #   -> data/rumor/timeline.jsonl (dedup by id), each item tagged with matched markets
    python scripts/rumor_signals.py report [--hours 24]   # per-market hourly mention counts + spikes vs the item's own baseline

EVENT STUDY (dev only):
    python scripts/rumor_signals.py universe       # settled Kalshi rumor-type markets (politics/world/companies/economics personnel)
    python scripts/rumor_signals.py gdelt          # stream GDELT 2.0 mentions files for the DEV window, hourly keyword counts
    python scripts/rumor_signals.py candles        # hourly Kalshi candles for the universe, DEV window only
    python scripts/rumor_signals.py study [--ledger]

Sources and terms (details in notes/rumor_signals.md):
  GDELT 2.0 raw files  data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.mentions.CSV.zip -- free, open, 15-min. (The DOC 2.0 API
                       answered 429 "one request every 5 seconds" to every request from this machine on 10-22, incl. the first
                       after a 40 s pause -- likely the shared VPN exit -- so the study uses raw files.)
  Bluesky              api.bsky.app/xrpc/app.bsky.feed.searchPosts -- public AppView, no auth (public.api.bsky.app gave 403 10-22).
  Reddit               www.reddit.com/r/<sub>/new/.rss -- public RSS, polite rate (one request / 2 s), descriptive UA. No login.
  Google News          news.google.com/rss/search?q=... -- public RSS.
  NOT used: X (no free reads; needs the user's pay-per-use key), any login-walled page.

PRE-REGISTRATION (written 2026-10-22 BEFORE any GDELT count or Kalshi candle of the study universe was pulled)
  Question: do news-volume spikes (GDELT) PRECEDE Kalshi price moves on rumor-driven markets, at what lag, and does a simple
  enter-after-spike / exit-after-N-hours rule beat fees?
  Universe: settled Kalshi markets whose SERIES is in category Politics/World/Companies/Economics and whose series title matches
    RUMOR_RE (out as / leave / resign / nominate / confirm / pardon / CEO / fired / indict / successor / chair ...), with
    volume >= 20,000 contracts and a person-name keyword: last name of yes_sub_title if it is a 2-3 word capitalised name,
    else of the first such name in the market title. Keywords < 5 letters or in GENERIC are dropped.
  Windows: DEV = spike hours 2025-01-01 00Z .. 2025-12-31 23Z. SEALED = 2026-01-01 onward: no 2026 GDELT file is downloaded
    by this script and `study` refuses hours >= 2026-01-01 (scoring sealed needs the user's explicit yes + a new subcommand).
  News volume: GDELT mentions files at :00 and :30 of every hour (half sample, to keep downloads < 2 GB); unique URLs per
    file; a hit = keyword appears in the URL as a whole token (split on non-letters). Hourly count c_h, mean MentionDocTone.
    Keywords with 2025 mean c_h > 10 are too generic and dropped (decided now, applied before any price is looked at).
  Spike onset at hour h (bucket [h, h+1)): c_h >= 3 AND c_h >= mean(c_{h-168..h-1}) + 4*max(sd, 0.5) AND no onset in h-24..h-1.
  Availability: the :30 file is published ~h+0:45, so the signal is usable at h+1. Entry quote = hourly candle ending at
    h+1 (yes_bid/yes_ask close). mid(t) = (bid+ask)/2 of the candle ending at t. Bet only if spread at entry <= 10c and
    0.03 <= mid <= 0.97.
  Descriptive (not trials): placebo-matched |dmid| around spikes vs random hours of the same markets; lag of the first
    cumulative |dmid| >= 3c within [h-24, h+24] (negative = price moved first); pooled lead-lag correlation of log1p(c) vs
    |dmid| at lags -12..+12.
  Rules (4 trials, bar p < 0.05/4 = 0.0125, one-sided bootstrap over spike clusters = keyword x onset, 10,000 resamples):
    M6 / M24 : momentum. d = sign(mid(h+1) - mid(h-1)), need |move| >= 2c. d>0 buy YES at ask(h+1), d<0 buy NO at
               1-bid(h+1). Exit at h+1+N at the held side's bid (YES bid / 1-yes ask); if the market settled first, exit
               at settlement value (no exit fee).
    F6 / F24 : fade = the opposite side, same entry/exit conventions.
    Fees: taker 0.07*p*(1-p) per contract on entry and on a non-settlement exit (unrounded). ROI = sum pnl / sum cost.
    PASS = p < 0.0125 AND ROI > 0. Also reported: H1/H2-2025 split.
  Sports injuries are NOT in the study (GDELT URLs carry almost no injury news and Kalshi game markets move on play); the
  live matcher covers them (team name + injury word), so a forward log can test them later.

SCORED 2026-10-22 (dev, once, ledger trials rumor_M6/M24/F6/F24): all four lose (ROI -7% .. -13%, ~20 bets each, p 1.0);
  descriptively the price moves BEFORE the GDELT spike hour (68% of first >=3c moves). See notes/rumor_signals.md.

X FOLLOW-UP (pre-registered 2026-10-22, NOT run -- needs the user's X_BEARER_TOKEN in sports/.env and --confirm-spend):
    python scripts/rumor_signals.py xcounts [--confirm-spend]
    python scripts/rumor_signals.py study --source x [--ledger]
  Pulls X full-archive hourly post COUNTS (GET /2/tweets/counts/all, $0.010 per request, pay-per-use) for each study-universe
  person, query '"<full name>" -is:retweet', DEV window 2025 only. Without --confirm-spend it only prints the request plan.
  `study --source x` re-runs the IDENTICAL descriptive block and rules on X counts instead of GDELT counts
  (new trials XM6/XM24/XF6/XF24, bar p < 0.05/4, separate ledger sha). Spike rule unchanged (c>=3, mean+4 sd, 24 h gap);
  X buckets are exact hours, so the signal is usable at h+1.
  Primary question: share of first >=3c moves that come AFTER the X onset (GDELT: 0.32).
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests

from sports.http import fetch

D = ROOT / "data" / "rumor"
D.mkdir(parents=True, exist_ok=True)
LEDGER = ROOT / "lab" / "ledger.jsonl"
K = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
UA = {"User-Agent": "rumor-signals research script (personal, non-commercial; contact via github ttt5600)"}
DATASET = "rumor_gdelt_kalshi"
DEV_START, DEV_END = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")
CATS = {"Politics", "World", "Companies", "Economics"}
RUMOR_RE = re.compile(r"\b(out as|out\b|leave|leaves|resign|nomin|confirm|pardon|ceo|fired|fire |indict|successor|chair|"
                      r"next .* (secretary|director|commissioner|ambassador|justice)|depart|replace|step down|impeach)", re.I)
GENERIC = {"trump", "biden", "donald", "president", "senate", "house", "federal", "united", "states", "america", "american",
           "china", "canada", "mexico", "russia", "ukraine", "israel", "europe", "white", "court", "supreme", "congress",
           "republican", "democrat", "governor", "mayor", "chair", "secretary", "director", "general", "minister", "prime",
           "party", "national", "company", "reserve", "system", "security", "services", "chief", "advice", "bidens",
           "department", "board", "office", "homeland", "justice", "defense", "treasury", "commerce", "agency", "commission",
           "election", "elections", "island", "islands", "champion", "champions", "league", "award", "awards", "prize",
           "series", "world", "primary", "winner", "championship", "states", "kingdom", "republic", "city", "county",
           "council", "committee", "administration", "union", "bureau", "institute", "foundation", "group", "news", "tesla", "apple", "google", "openai", "microsoft", "before", "after", "first"}
STOP_LEAD = {"Will", "Who", "What", "When", "Which", "How", "The", "Trump", "Trump's", "Next", "New", "Before", "After", "In"}
NAME_RE = re.compile(r"\b([A-Z][a-zA-Z'\.\-]+(?:\s+[A-Z][a-zA-Z'\.\-]+){1,2})\b")
INJURY_RE = re.compile(r"\b(injur|ruled out|questionable|doubtful|out for|out vs|out tonight|day-to-day|acl|hamstring|ankle|"
                       r"concussion|sidelined|won't play|will not play|game-time decision)", re.I)
SUBREDDITS = ["politics", "PoliticalDiscussion", "Economics", "nba", "nfl", "fantasyfootball", "sportsbook", "Kalshi",
              "Polymarket", "news", "worldnews", "wallstreetbets", "business"]
SPORTS_SERIES = ["KXNBAGAME", "KXNFLGAME", "KXNHLGAME", "KXMLBGAME", "KXNCAAFGAME"]


# ---------------------------------------------------------------- keywords
def _name_key(name: str) -> str | None:
    w = [x for x in re.split(r"\s+", name.strip()) if x]
    if not 2 <= len(w) <= 3 or w[0] in STOP_LEAD:
        return None
    last = re.sub(r"[^a-z]", "", w[-1].lower())
    if len(last) < 5 or last in GENERIC or last in {"jr", "sr", "iii"}:
        return None
    return last


def keyword(title: str, sub: str) -> tuple[str | None, str | None]:
    """(keyword, full name) for a market: yes_sub_title if it looks like a person name, else the first name in the title."""
    for cand in [sub or ""] + NAME_RE.findall(title or ""):
        cand = cand.strip()
        if NAME_RE.fullmatch(cand):
            k = _name_key(cand)
            if k:
                return k, cand
    return None, None


def tokens(s: str) -> set[str]:
    return set(re.split(r"[^a-z]+", s.lower()))


# ---------------------------------------------------------------- live: markets
def cmd_markets() -> int:
    series = fetch(f"{K}/series", cache=True)["series"]
    cat = {s["ticker"]: s.get("category") for s in series}
    rows, cur = [], None
    while True:
        p = {"status": "open", "with_nested_markets": "true", "limit": 200}
        if cur:
            p["cursor"] = cur
        try:
            d = fetch(f"{K}/events", p, cache=False, retries=3)
        except Exception as e:  # noqa: BLE001 -- Kalshi live API returned 503 for hours on 10-22
            print(f"Kalshi /events unavailable ({e}); continuing with Polymarket only", flush=True)
            break
        for e in d.get("events", []):
            c = cat.get(e.get("series_ticker"))
            sport = e.get("series_ticker") in SPORTS_SERIES
            if c not in CATS | {"Elections", "Mentions"} and not sport:
                continue
            for m in e.get("markets", []):
                vol = float(m.get("volume_fp") or 0)
                if vol < 1000:
                    continue
                if sport:
                    team = (m.get("yes_sub_title") or "").strip()
                    k, full = (team.split()[-1].lower(), team) if team else (None, None)
                    kind = "sports_injury"
                else:
                    k, full = keyword(m.get("title", ""), m.get("yes_sub_title", ""))
                    kind = c.lower()
                if k:
                    rows.append({"venue": "kalshi", "id": m["ticker"], "title": m.get("title"), "sub": m.get("yes_sub_title"),
                                 "kw": k, "name": full, "kind": kind, "volume": vol,
                                 "mid": (float(m.get("yes_bid_dollars") or 0) + float(m.get("yes_ask_dollars") or 1)) / 2})
        cur = d.get("cursor")
        if not cur or not d.get("events"):
            break
    off = 0
    while off < 1000:
        evs = fetch(f"{GAMMA}/events", {"active": "true", "closed": "false", "order": "volume24hr", "ascending": "false",
                                        "limit": 100, "offset": off}, cache=False)
        if not evs:
            break
        for e in evs:
            for m in e.get("markets", []):
                k, full = keyword(m.get("question", ""), m.get("groupItemTitle", ""))
                if k:
                    try:
                        mid = (float(m.get("bestBid") or 0) + float(m.get("bestAsk") or 1)) / 2
                    except (TypeError, ValueError):
                        mid = None
                    rows.append({"venue": "polymarket", "id": m.get("slug"), "title": m.get("question"),
                                 "sub": m.get("groupItemTitle"), "kw": k, "name": full, "kind": "poly",
                                 "volume": float(m.get("volume") or 0), "mid": mid})
        off += 100
    (D / "markets.json").write_text(json.dumps(rows))
    kws = pd.DataFrame(rows).groupby("kw").volume.sum().sort_values(ascending=False)
    print(f"{len(rows)} markets ({sum(r['venue'] == 'kalshi' for r in rows)} Kalshi), {len(kws)} keywords; top: "
          + ", ".join(kws.index[:25]))
    return 0


# ---------------------------------------------------------------- live: ingest
def _get(url, params=None, timeout=30):
    for i in range(3):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 503):
                time.sleep(5 * (i + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(3)
    return None


def _rss(text: str, source: str) -> list[dict]:
    out = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return out
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for it in root.iter("item"):  # RSS 2.0 (Google News)
        t, link, pub = it.findtext("title") or "", it.findtext("link") or "", it.findtext("pubDate")
        try:
            ts = parsedate_to_datetime(pub).astimezone(timezone.utc).isoformat() if pub else None
        except (TypeError, ValueError):
            ts = None
        out.append({"source": source, "ts": ts, "text": t, "url": link})
    for it in root.findall("a:entry", ns):  # Atom (Reddit)
        t = it.findtext("a:title", default="", namespaces=ns)
        ln = it.find("a:link", ns)
        out.append({"source": source, "ts": it.findtext("a:published", default=None, namespaces=ns) or
                    it.findtext("a:updated", default=None, namespaces=ns), "text": t,
                    "url": ln.get("href") if ln is not None else ""})
    return out


def _gdelt_latest() -> list[dict]:
    r = _get("http://data.gdeltproject.org/gdeltv2/lastupdate.txt")
    if r is None:
        return []
    url = next((ln.split()[-1] for ln in r.text.splitlines() if ".mentions." in ln), None)
    z = _get(url, timeout=60) if url else None
    if z is None:
        return []
    out, seen = [], set()
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        for line in zf.read(zf.namelist()[0]).decode("utf-8", "replace").splitlines():
            f = line.split("\t")
            if len(f) < 14 or f[5] in seen:
                continue
            seen.add(f[5])
            ts = datetime.strptime(f[2], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat()
            out.append({"source": "gdelt", "ts": ts, "text": re.sub(r"[-_/]+", " ", f[5].split("://")[-1]), "url": f[5],
                        "tone": float(f[13]) if f[13] else None})
    return out


def _norm(s: str) -> str:
    return " " + " ".join(re.split(r"[^a-z]+", s.lower())).strip() + " "


def match(text: str, idx: dict[str, list[dict]]) -> list[str]:
    """Last-name token is the candidate filter; the FULL name (e.g. 'lebron james', not 'theo james') must then appear.
    Sports game markets additionally need an injury word in the same item."""
    tk = tokens(text)
    nt = _norm(text)
    hits = []
    for k in tk & idx.keys():
        for m in idx[k]:
            if m["kind"] == "sports_injury" and not INJURY_RE.search(text):
                continue
            if m.get("name") and len(m["name"].split()) > 1 and _norm(m["name"]) not in nt:
                continue
            hits.append(m["id"])
    return hits


def cmd_ingest() -> int:
    mk = json.loads((D / "markets.json").read_text())
    idx: dict[str, list[dict]] = {}
    for m in mk:
        idx.setdefault(m["kw"], []).append(m)
    top = (pd.DataFrame(mk).query("kind != 'sports_injury'").groupby("kw").volume.sum()
           .sort_values(ascending=False).index[:30].tolist())
    items = _gdelt_latest()
    for sub in SUBREDDITS:
        r = _get(f"https://www.reddit.com/r/{sub}/new/.rss", {"limit": 50})
        items += _rss(r.text, f"reddit/{sub}") if r is not None else []
        time.sleep(2)
    for k in top:
        name = next(m["name"] for m in idx[k] if m["name"])
        r = _get("https://news.google.com/rss/search", {"q": f'"{name}" when:1d', "hl": "en-US", "gl": "US", "ceid": "US:en"})
        items += _rss(r.text, "gnews") if r is not None else []
        r = _get("https://api.bsky.app/xrpc/app.bsky.feed.searchPosts", {"q": f'"{name}"', "sort": "latest", "limit": 50})
        if r is not None:
            for p in r.json().get("posts", []):
                items.append({"source": "bluesky", "ts": p.get("record", {}).get("createdAt") or p.get("indexedAt"),
                              "text": p.get("record", {}).get("text", ""), "url": p.get("uri"),
                              "likes": p.get("likeCount"), "reposts": p.get("repostCount")})
        time.sleep(1)
    tl = D / "timeline.jsonl"
    seen = set()
    if tl.exists():
        seen = {json.loads(ln)["id"] for ln in tl.read_text().splitlines() if ln.strip()}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = 0
    with tl.open("a") as f:
        for it in items:
            hits = match(it["text"], idx)
            if not hits:
                continue  # store only items that touch an open market (keeps the file small)
            it["id"] = hashlib.sha1((it["source"] + (it.get("url") or it["text"])).encode()).hexdigest()[:16]
            if it["id"] in seen:
                continue
            seen.add(it["id"])
            f.write(json.dumps({**it, "fetched": now, "markets": hits[:20]}) + "\n")
            new += 1
    by = pd.Series([it["source"].split("/")[0] for it in items]).value_counts().to_dict()
    print(f"{now} fetched {len(items)} items {by}; {new} new matched items -> {tl}")
    return 0


def cmd_report() -> int:
    hours = int(sys.argv[sys.argv.index("--hours") + 1]) if "--hours" in sys.argv else 24
    tl = D / "timeline.jsonl"
    if not tl.exists():
        print("no timeline yet; run ingest")
        return 1
    df = pd.DataFrame([json.loads(ln) for ln in tl.read_text().splitlines() if ln.strip()])
    df["ts"] = pd.to_datetime(df.ts, utc=True, errors="coerce", format="ISO8601")
    df = df.explode("markets").dropna(subset=["ts", "markets"])
    cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    recent = df[df.ts >= cut].groupby("markets").agg(n=("id", "size"), sources=("source", lambda s: ",".join(sorted(set(
        x.split("/")[0] for x in s))))).sort_values("n", ascending=False)
    base = df[df.ts < cut].groupby("markets").size() / max(1.0, (cut - df.ts.min()).total_seconds() / 3600 / hours)
    recent["baseline_per_window"] = base.reindex(recent.index).fillna(0).round(1)
    print(f"items per market in the last {hours} h (vs earlier windows; few ticks so far = no baseline yet):")
    print(recent.head(30).to_string())
    return 0


# ---------------------------------------------------------------- study: universe
def _settled(series: str) -> list[dict]:
    out = []
    for path, extra, cache in ((f"{K}/historical/markets", {}, True), (f"{K}/markets", {"status": "settled"}, False)):
        cur = None
        while True:
            p = {"series_ticker": series, "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            try:
                d = fetch(path, p, cache=cache, retries=4)
            except Exception as e:  # noqa: BLE001 -- the live endpoint was intermittently 503 on 10-22
                print(f"  {series} {path.rsplit('/', 1)[-1]}: {e}", flush=True)
                break
            out += d.get("markets", [])
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    return out


def cmd_universe() -> int:
    series = fetch(f"{K}/series", cache=True)["series"]
    pick = [s for s in series if s.get("category") in CATS and RUMOR_RE.search(s.get("title") or "")]
    print(f"{len(pick)} rumor-type series", flush=True)
    rows = []
    with ThreadPoolExecutor(4) as ex:
        for s, ms in zip(pick, ex.map(lambda s: _settled(s["ticker"]), pick)):
            for m in ms:
                if m.get("result") not in ("yes", "no") or float(m.get("volume_fp") or 0) < 20000:
                    continue
                k, full = keyword(m.get("title", ""), m.get("yes_sub_title", ""))
                if not k:
                    continue
                rows.append({"series": s["ticker"], "category": s.get("category"), "ticker": m["ticker"],
                             "event": m["event_ticker"], "title": m.get("title"), "sub": m.get("yes_sub_title"), "kw": k,
                             "name": full, "result": m["result"], "volume": float(m["volume_fp"]),
                             "open": m.get("open_time"), "close": m.get("close_time"),
                             "settle": m.get("settlement_ts") or m.get("close_time")})
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    o, c = pd.to_datetime(df.open, utc=True, format="ISO8601"), pd.to_datetime(df.close, utc=True, format="ISO8601")
    df = df[(o < DEV_END) & (c > DEV_START)]  # overlaps the DEV window
    df.to_parquet(D / "universe.parquet", index=False)
    print(f"universe: {len(df)} markets, {df.event.nunique()} events, {df.kw.nunique()} keywords, "
          f"{df.series.nunique()} series; by category {df.category.value_counts().to_dict()}")
    return 0


# ---------------------------------------------------------------- study: GDELT
def _gdelt_file(ts: pd.Timestamp, kws: frozenset) -> tuple[pd.Timestamp, int, dict]:
    url = f"http://data.gdeltproject.org/gdeltv2/{ts:%Y%m%d%H%M%S}.mentions.CSV.zip"
    r = _get(url, timeout=60)
    if r is None:
        return ts, -1, {}
    hits: dict[str, list[float]] = {}
    seen = set()
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            raw = zf.read(zf.namelist()[0]).decode("utf-8", "replace")
    except zipfile.BadZipFile:
        return ts, -1, {}
    for line in raw.splitlines():
        f = line.split("\t")
        if len(f) < 14 or f[5] in seen:
            continue
        seen.add(f[5])
        path = f[5].split("://", 1)[-1].split("/", 1)[-1]  # drop the domain: outlet names are not news
        for k in tokens(path) & kws:
            hits.setdefault(k, []).append(float(f[13]) if f[13] else np.nan)
    return ts, len(seen), hits


def cmd_gdelt() -> int:
    u = pd.read_parquet(D / "universe.parquet")
    # superset: universe keywords + person names in every rumor-type series title (so a universe rebuilt after a Kalshi
    # outage -- live /markets was 503 on 10-22 -- still has counts without re-downloading GDELT)
    series = fetch(f"{K}/series", cache=True)["series"]
    extra = {_name_key(n) for s in series if s.get("category") in CATS and RUMOR_RE.search(s.get("title") or "")
             for n in NAME_RE.findall(s.get("title") or "")}
    kws = frozenset(set(u.kw) | (extra - {None}))
    out = D / "gdelt_hourly.parquet"
    done = set(pd.read_parquet(out).file_ts) if out.exists() else set()
    stamps = [t for t in pd.date_range(DEV_START, DEV_END - pd.Timedelta(minutes=30), freq="30min") if t not in done]
    assert all(t < DEV_END for t in stamps), "sealed GDELT must not be pulled"
    print(f"{len(kws)} keywords, {len(stamps)} files to pull (:00 and :30)", flush=True)
    rows, n = [], 0
    with ThreadPoolExecutor(6) as ex:
        for ts, nurl, hits in ex.map(lambda t: _gdelt_file(t, kws), stamps):
            rows.append({"file_ts": ts, "kw": "__total__", "n": nurl, "tone": np.nan})
            for k, v in hits.items():
                rows.append({"file_ts": ts, "kw": k, "n": len(v), "tone": float(np.nanmean(v)) if len(v) else np.nan})
            n += 1
            if n % 2000 == 0 or n == len(stamps):
                new = pd.DataFrame(rows)
                old = pd.read_parquet(out) if out.exists() else None
                pd.concat([old, new] if old is not None else [new], ignore_index=True).to_parquet(out, index=False)
                rows = []
                print(f"  {n}/{len(stamps)} files ({ts:%Y-%m-%d %H:%M})", flush=True)
    return 0


# ---------------------------------------------------------------- study: candles
def _candles(series: str, tk: str, a: int, b: int) -> list[dict]:
    out = []
    for path in (f"{K}/historical/markets/{tk}/candlesticks", f"{K}/series/{series}/markets/{tk}/candlesticks"):
        t = a
        got = []
        while t < b:
            e = min(b, t + 4000 * 3600)
            try:
                d = fetch(path, {"start_ts": t, "end_ts": e, "period_interval": 60}, cache=True, retries=4)
            except Exception:  # noqa: BLE001
                d = {}
            got += d.get("candlesticks", [])
            t = e
        if got:
            out = got
            break
    rows = []
    for c in out:
        def g(side):
            v = c.get(side) or {}
            return pd.to_numeric(v.get("close_dollars", v.get("close")), errors="coerce")
        rows.append({"ticker": tk, "t": int(c["end_period_ts"]), "bid": g("yes_bid"), "ask": g("yes_ask"),
                     "vol": pd.to_numeric(c.get("volume_fp", c.get("volume")), errors="coerce")})
    return rows


def cmd_candles() -> int:
    u = pd.read_parquet(D / "universe.parquet")
    o = pd.to_datetime(u.open, utc=True, format="ISO8601").clip(lower=DEV_START)
    c = pd.to_datetime(u.close, utc=True, format="ISO8601").clip(upper=DEV_END + pd.Timedelta(days=2))
    jobs = [(s, tk, int(a.timestamp()), int(b.timestamp())) for s, tk, a, b in zip(u.series, u.ticker, o, c) if b > a]
    rows = []
    with ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(lambda j: _candles(*j), jobs)):
            rows += r
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} markets", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(D / "candles.parquet", index=False)
    print(f"candles: {len(df)} rows, {df.ticker.nunique()} of {len(jobs)} markets")
    return 0


# ---------------------------------------------------------------- study: scoring
def _fee(p):
    return 0.07 * p * (1 - p)


def spikes(g: pd.DataFrame) -> pd.DataFrame:
    """Hourly keyword counts -> onset hours (pre-registered rule)."""
    out = []
    for k, s in g.groupby("kw"):
        s = s.set_index("hour").n.reindex(HOURS, fill_value=0).astype(float)
        mu = s.shift(1).rolling(168, min_periods=72).mean()
        sd = s.shift(1).rolling(168, min_periods=72).std().clip(lower=0.5)
        on = (s >= 3) & (s >= mu + 4 * sd)
        last = None
        for h in s.index[on.values]:
            if last is None or h - last > pd.Timedelta(hours=24):
                out.append({"kw": k, "hour": h, "n": s[h], "base": mu[h]})
            last = h if (last is None or h - last > pd.Timedelta(hours=24)) else last
    return pd.DataFrame(out)


HOURS = pd.date_range(DEV_START, DEV_END - pd.Timedelta(hours=1), freq="h")


def boot(pnl: pd.Series, cost: pd.Series, cl: pd.Series, B=10000, seed=0) -> tuple[float, float, float]:
    g = pd.DataFrame({"p": pnl, "c": cost, "k": cl}).groupby("k").sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (B, len(g)))
    roi = g.p.values[idx].sum(1) / g.c.values[idx].sum(1)
    return float(np.quantile(roi, 0.025)), float(np.quantile(roi, 0.975)), float((roi <= 0).mean())


def cmd_study() -> int:
    u = pd.read_parquet(D / "universe.parquet")
    src = sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv else "gdelt"
    gd = pd.read_parquet(D / ("x_hourly.parquet" if src == "x" else "gdelt_hourly.parquet"))
    if src == "x":
        gd = pd.concat([gd, pd.DataFrame({"file_ts": [DEV_START], "kw": ["__total__"], "n": [0], "tone": [np.nan]})])
    gd["hour"] = gd.file_ts.dt.floor("h")
    assert gd.file_ts.max() < DEV_END, "sealed hours present"
    tot = gd[gd.kw == "__total__"]
    print(f"GDELT files {tot.file_ts.nunique()} ({(tot.n < 0).sum()} failed), unique URLs/file median {tot.n.median():.0f}")
    gd = gd[gd.kw != "__total__"]
    hourly = gd.groupby(["kw", "hour"]).agg(n=("n", "sum")).reset_index()
    mean_c = hourly.groupby("kw").n.sum() / len(HOURS)
    generic = set(mean_c[mean_c > 10].index)
    print(f"keywords with any hit {hourly.kw.nunique()}/{u.kw.nunique()}; dropped as generic (>10/h): {sorted(generic)}")
    sp = spikes(hourly[~hourly.kw.isin(generic)])
    print(f"spike onsets (DEV 2025): {len(sp)} across {sp.kw.nunique() if len(sp) else 0} keywords")
    cd = pd.read_parquet(D / "candles.parquet").drop_duplicates(["ticker", "t"])  # 4000-h request chunks share a boundary candle
    cd["mid"] = (cd.bid + cd.ask) / 2
    cd["spr"] = cd.ask - cd.bid
    cd["time"] = pd.to_datetime(cd.t, unit="s", utc=True)
    px = {tk: g.set_index("time")[["bid", "ask", "mid", "spr"]].sort_index() for tk, g in cd.groupby("ticker")}
    settle = dict(zip(u.ticker, pd.to_datetime(u.settle, utc=True, format="ISO8601")))
    res = dict(zip(u.ticker, (u.result == "yes").astype(float)))

    def at(tk, t):
        p = px.get(tk)
        if p is None:
            return None
        s = p.loc[:t]
        if not len(s) or t - s.index[-1] > pd.Timedelta(hours=2):
            return None
        return s.iloc[-1]

    ev, desc = [], []
    for _, s in sp.iterrows():
        h = s.hour
        for tk in u.loc[u.kw == s.kw, "ticker"]:
            q1, q0, qm = at(tk, h + pd.Timedelta(hours=1)), at(tk, h), at(tk, h - pd.Timedelta(hours=1))
            if q1 is None or q0 is None or qm is None or settle[tk] <= h + pd.Timedelta(hours=1):
                continue
            row = {"kw": s.kw, "hour": h, "ticker": tk, "mid1": q1.mid, "spr": q1.spr}
            for lag in range(-24, 25):
                q = at(tk, h + pd.Timedelta(hours=lag))
                row[f"m{lag}"] = q.mid if q is not None else np.nan
            desc.append(row)
            if q1.spr > 0.10 or not 0.03 <= q1.mid <= 0.97:
                continue
            mv = q1.mid - qm.mid
            if abs(mv) < 0.02 - 1e-9:
                continue
            for N in (6, 24):
                te = h + pd.Timedelta(hours=1 + N)
                for rule, d in (("M", np.sign(mv)), ("F", -np.sign(mv))):
                    cost = q1.ask if d > 0 else 1 - q1.bid
                    if settle[tk] <= te:
                        val = res[tk] if d > 0 else 1 - res[tk]
                        xfee = 0.0
                    else:
                        qe = at(tk, te)
                        if qe is None:
                            continue
                        val = qe.bid if d > 0 else 1 - qe.ask
                        xfee = _fee(val)
                    pnl = val - cost - _fee(cost) - xfee
                    ev.append({"rule": f"{rule}{N}", "kw": s.kw, "hour": h, "ticker": tk, "cost": cost, "pnl": pnl,
                               "move": mv})
    dd = pd.DataFrame(desc)
    print(f"\nspike x market pairs with prices: {len(dd)} ({dd.kw.nunique() if len(dd) else 0} keywords)")
    if len(dd):
        m = dd[[f"m{l}" for l in range(-24, 25)]].values
        ch = np.abs(np.diff(m, axis=1))  # |dmid| hour l-1 -> l, l = -23..24
        lags = np.arange(-23, 25)
        mean_abs = np.nanmean(ch, axis=0)
        print("mean |dmid| (cents) by hour relative to spike hour h (bucket [h,h+1) = lag +1):")
        print("  " + " ".join(f"{l:+d}:{v * 100:.2f}" for l, v in zip(lags, mean_abs) if l in (-12, -6, -3, -2, -1, 0, 1, 2, 3, 6, 12, 24)))
        # placebo: random hours of the same markets
        rng = np.random.default_rng(1)
        pl = []
        for tk in dd.ticker.unique():
            p = px[tk].mid
            p = p[(p.index >= DEV_START) & (p.index < DEV_END)]
            if len(p) > 50:
                pl.append(p.diff().abs().dropna().sample(min(200, len(p) - 1), random_state=int(rng.integers(1e9))))
        pl = pd.concat(pl) if pl else pd.Series(dtype=float)
        print(f"  placebo mean |dmid| same markets, random hours: {pl.mean() * 100:.2f}c (n={len(pl)})")
        # first cumulative >= 3c move from h-24
        first = []
        for r in m:
            base = r[0]
            hit = np.where(np.abs(r - base) >= 0.03 - 1e-9)[0]
            first.append(int(hit[0]) - 24 if len(hit) else np.nan)
        first = pd.Series(first).dropna()
        print(f"  first |cum move| >= 3c from h-24: n={len(first)} of {len(dd)}; lag quantiles (h) "
              f"{first.quantile([.1, .25, .5, .75, .9]).round(1).to_dict()}; share before spike (lag<=0) "
              f"{(first <= 0).mean():.2f}, at/after spike hour (lag>=1) {(first >= 1).mean():.2f}")
        pre = np.abs(dd["m0"] - dd["m-6"])
        post = np.abs(dd["m7"] - dd["m1"])
        both = pre.notna() & post.notna()
        print(f"  |move| 6h before spike (h-6..h) {pre[both].mean() * 100:.2f}c vs 6h after signal (h+1..h+7) "
              f"{post[both].mean() * 100:.2f}c; share post>pre {(post[both] > pre[both]).mean():.2f} (n={both.sum()})")
        lead = (post >= 0.03) & (pre < 0.03)
        print(f"  'news leads' (post>=3c & pre<3c): {lead[both].mean():.3f}; 'price leads' (pre>=3c & post<3c): "
              f"{((pre >= 0.03) & (post < 0.03))[both].mean():.3f}")
    # pooled lead-lag correlation: log1p(count) at hour h vs |dmid| over [h+L, h+L+1)
    cc = []
    hw = hourly.pivot_table(index="hour", columns="kw", values="n", aggfunc="sum").reindex(HOURS).fillna(0)
    for tk, kw in zip(u.ticker, u.kw):
        if tk not in px or kw not in hw or kw in generic:
            continue
        p = px[tk].mid.reindex(HOURS + pd.Timedelta(hours=1)).ffill(limit=2)
        dm = pd.Series(np.abs(p.diff().values), index=HOURS)  # |mid(h+1) - mid(h)| = move during bucket h
        cc.append((np.log1p(hw[kw]), dm))
    if cc:
        out = []
        for L in range(-12, 13):
            xs = pd.concat([c.shift(0) for c, _ in cc]); ys = pd.concat([d.shift(-L) for _, d in cc])
            ok = xs.notna() & ys.notna()
            out.append((L, np.corrcoef(xs[ok], ys[ok])[0, 1]))
        print("  pooled corr(log1p news_h, |dmid| in bucket h+L): " + " ".join(f"{L:+d}:{r:.3f}" for L, r in out
                                                                            if L in (-12, -6, -3, -2, -1, 0, 1, 2, 3, 6, 12)))
    E = pd.DataFrame(ev)
    print(f"\nTRIALS (bar p < 0.0125; one-sided bootstrap over keyword x onset clusters)")
    recs = []
    prior = [json.loads(ln) for ln in LEDGER.read_text().splitlines() if ln.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r.get("dataset") == DATASET}
    for rule in ("M6", "M24", "F6", "F24"):
        x = E[E.rule == rule] if len(E) else E
        rule_id = rule if src == "gdelt" else "X" + rule
        sha = hashlib.sha256(("rumor_signals:" + rule_id).encode()).hexdigest()[:12]
        if len(x) < 10:
            r = {"bets": int(len(x)), "roi": float("nan"), "p": 1.0}
            print(f"  {rule}: only {len(x)} bets -- not scorable")
        else:
            lo, hi, p = boot(x.pnl, x.cost, x.kw + x.hour.astype(str))
            h1 = x[x.hour < pd.Timestamp("2025-07-01", tz="UTC")]
            h2 = x[x.hour >= pd.Timestamp("2025-07-01", tz="UTC")]
            r = {"bets": int(len(x)), "clusters": int((x.kw + x.hour.astype(str)).nunique()), "staked": float(x.cost.sum()),
                 "units": float(x.pnl.sum()), "roi": float(x.pnl.sum() / x.cost.sum()), "roi_ci_lo": lo, "roi_ci_hi": hi,
                 "p": p, "win": float((x.pnl > 0).mean()),
                 "per_fold": {"2025H1": round(float(h1.pnl.sum()), 2), "2025H2": round(float(h2.pnl.sum()), 2)}}
            print(f"  {rule:4s} bets {r['bets']:4d} clusters {r['clusters']:4d} staked ${r['staked']:.1f} pnl {r['units']:+.2f} "
                  f"ROI {r['roi']:+.2%} CI [{lo:+.1%},{hi:+.1%}] p {p:.4f} win {r['win']:.2f} H1/H2 {r['per_fold']}")
        shas.add(sha)
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"rumor_{rule_id}", "sha": sha,
                     "dataset": DATASET, "hypothesis": f"pre-registered in scripts/rumor_signals.py ({rule_id})",
                     "source": "rumor signals GDELT vs Kalshi (user 2026-10-22)", "status": "ok",
                     "trial_number": len(shas), "passed": bool(r.get("p", 1) < 0.0125 and (r.get("roi") or 0) > 0), **r})
    if len(E):
        E.to_parquet(D / f"study_bets_{src}.parquet", index=False)
    if len(dd):
        dd.to_parquet(D / f"study_pairs_{src}.parquet", index=False)
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"appended {len(recs)} trials to {LEDGER}")
    return 0


def _env(key: str) -> str | None:
    f = ROOT / ".env"
    if f.exists():
        for ln in f.read_text().splitlines():
            if ln.strip().startswith(key + "="):
                return ln.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def cmd_xcounts() -> int:
    u = pd.read_parquet(D / "universe.parquet")
    names = u.dropna(subset=["name"]).drop_duplicates("kw").set_index("kw").name
    pages_est = 13  # ~31-day windows per year; page size UNVERIFIED, next_token is followed either way
    print(f"{len(names)} people; plan ~{len(names) * pages_est} requests x $0.010 = ~${len(names) * pages_est * 0.010:.2f} "
          f"(counts/all, hourly, {DEV_START:%Y-%m-%d}..{DEV_END:%Y-%m-%d})")
    tok = _env("X_BEARER_TOKEN")
    if not tok:
        print("no X_BEARER_TOKEN in sports/.env -- the user creates an X developer app and adds it there; nothing called")
        return 1
    if "--confirm-spend" not in sys.argv:
        print("dry run: add --confirm-spend to call the API")
        return 0
    out, nreq = [], 0
    for kw, name in names.items():
        start = DEV_START
        while start < DEV_END:
            end = min(DEV_END, start + pd.Timedelta(days=31))
            nxt = None
            while True:
                p = {"query": f'"{name}" -is:retweet', "granularity": "hour",
                     "start_time": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end_time": end.strftime("%Y-%m-%dT%H:%M:%SZ")}
                if nxt:
                    p["next_token"] = nxt
                r = requests.get("https://api.x.com/2/tweets/counts/all", params=p,
                                 headers={"Authorization": f"Bearer {tok}"}, timeout=60)
                nreq += 1
                if r.status_code == 429:
                    time.sleep(60)
                    continue
                if r.status_code != 200:
                    print(f"  {kw}: HTTP {r.status_code} {r.text[:200]}")
                    break
                j = r.json()
                for b in j.get("data", []):
                    out.append({"file_ts": pd.Timestamp(b["start"]), "kw": kw, "n": int(b["tweet_count"]), "tone": np.nan})
                nxt = j.get("meta", {}).get("next_token")
                time.sleep(1.1)
                if not nxt:
                    break
            start = end
    pd.DataFrame(out).to_parquet(D / "x_hourly.parquet", index=False)
    print(f"{nreq} requests (~${nreq * 0.010:.2f}); {len(out)} hourly buckets -> {D / 'x_hourly.parquet'}")
    return 0


def main() -> int:
    cmds = {"markets": cmd_markets, "ingest": cmd_ingest, "report": cmd_report, "universe": cmd_universe,
            "gdelt": cmd_gdelt, "candles": cmd_candles, "study": cmd_study, "xcounts": cmd_xcounts}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        return 2
    return cmds[sys.argv[1]]()


if __name__ == "__main__":
    raise SystemExit(main())
