"""Daily READ-ONLY Kalshi new-series watcher (notes/ops_series_watch.md). No keys, no orders: public GETs via kx_throttle.

    series_watch.py run   [--date YYYY-MM-DD]   snapshot GET /series + open events, diff vs the previous snapshot,
                                                score new / newly-active series, write the digest + queue lines
    series_watch.py seed  --census DIR --date D build a snapshot for date D from a raw census dir
                                                (data/census/raw_<date>/kalshi_series.json + kalshi_events_*.json)
    series_watch.py sync  [--worktree DIR]      copy digests + pending queue lines into the worktree (launchd cannot
                                                write ~/Desktop under TCC; run this from a session)

Why: our surviving edges (E4 weather tails, L39 monthly rain, TSA) sit in markets nobody else models. Kalshi lists new
series often; this replaces the manual census with a daily diff + an E4-pattern checklist score (0-10):
  S1 settlement source verifiable (public agency/data URL)          0-2
  S2 independent public forecast exists (domain map)                0-2
  S3 ladder/bracket structure (markets per open event)              0-2
  S4 far brackets likely stale (tail quotes 2-15c, zero 24h volume) 0-2
  S5 events per month (frequency / open events)                     0-2
  penalty -3: sports / politics / elections / mentions (crowded or not data-resolved)
PROMISING (pre-registered in the notes before the first run): score >= 6 AND >= 2 open markets AND no penalty.

Outputs (root = $SW_ROOT or the deployed ~/.claude/quant-sports, else the sports dir this file lives in):
  state/series_watch/snap_<date>.json.gz      compact per-series snapshot (series meta + open-event summary)
  digests/series_watch-<date>.md              daily digest (vault-indexed: one ### per promising series)
  notes/series_watch/<date>.md                same digest (worktree path; mirrored when writable)
  notes/series_watch/queue_pending.md         promising lines not yet appended to notes/RESEARCH_QUEUE.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import gzip
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path(os.environ.get("SW_ROOT", str(HERE.parent)))
WORKTREE = Path(os.environ.get("SW_WORKTREE",
                               str(Path.home() / "Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports")))
SNAPS = ROOT / "state" / "series_watch"
MAX_HEADINGS = 30

FREQ_PER_MONTH = {"hourly": 720, "fifteen_min": 2880, "daily": 30, "weekly": 4.3, "biweekly": 2.15, "monthly": 1,
                  "quarterly": 1 / 3, "annual": 1 / 12, "yearly": 1 / 12}

# settlement-source / title domain regexes -> (domain, verifiable public timestamped source, independent forecast exists)
DOMAINS = [
    ("weather", r"noaa|nws|weather\.gov|ncei|climate prediction|nhc|hurricane|wunderground|met office|meteo|snow|rain|"
                r"temperature|precip|tornado|heat index|wind", 2, 2),
    ("geo/space", r"usgs|earthquake|volcan|swpc|solar|nasa|nsidc|sea ice|aurora|geomagnetic|wildfire|nifc|airnow|"
                  r"air quality|epa", 2, 1),
    ("transport/energy", r"tsa|faa|bts|eia|aaa|gas price|gasbuddy|caiso|ercot|pjm|power grid|flights?|airline|"
                         r"portwatch|amtrak", 2, 2),
    ("macro", r"bls|bureau of labor|census\.gov|bea\b|bureau of economic|federal reserve|fred|treasury|cpi|payroll|"
              r"jobless|gdp|ism|conference board|university of michigan|mortgage|freddie mac|zillow|case-shiller", 2, 2),
    ("health", r"cdc|who\.int|covid|flu|measles|wastewater|hhs", 2, 1),
    ("charts/media", r"spotify|billboard|netflix|luminate|apple music|app store|box office|the-numbers|boxofficemojo|"
                     r"youtube|tiktok|steam|twitch|google trends|wikipedia|imdb|rotten tomatoes|nielsen|kworb", 2, 1),
    ("markets/crypto", r"cf benchmarks|coinbase|bitcoin|ethereum|crypto|nasdaq|nyse|s&p|cboe|cme|ice|"
                       r"wti|brent|gold|forex|treasury yield", 2, 2),
]
# Already-tested series (FUND_BOOK 3d + LEARNINGS_2026-11 + EDGES_AND_AGENTS, compiled 11-16 by ops_series_watch_fix).
# TESTED -> annotated in the digest and NEVER queued; RELATED -> a sibling was tested, queued with the note.
# Keep in sync when a new series is tested (the deployed job cannot read the worktree notes).
TESTED = [
    (r"KXHIGH.*", "E1/E2/E4 daily-high ladders (live/paper; EDGES_AND_AGENTS E1-E4)"),
    (r"KXLOWT.*", "X2 daily lows dead (STATE closed list)"),
    (r"KXRAIN.*", "L39 monthly rain / R1 / RD1-RD2 daily rain (l39_expansion.md, raind_daily_rain.md)"),
    (r"KXTEMP.*", "HT1 hourly temp dead + ND1 nyc_hourly_dir.md (index unobservable)"),
    (r"KXTSAW?", "TSA weekly maker sealed-weak (e1x_tsa.md)"),
    (r"KXAAAGAS.*", "Q016 dead + state gas: maker holds certain side (q016_commodities.md, state_gas.md)"),
    (r"KX(TX)?ERCOT.*", "ERCOT monthly dead (energy_monthly.md) / Q017 peak load (q017_ercot_load.md)"),
    (r"KX[A-Z0-9]*EVENT", "econ-event ladders: 1/1,349 tradable quotes (econ_event_ladders.md)"),
    (r"KXHURCAT.*", "NHC hurricane category dead (tropical_nhc.md)"),
    (r"KXALBUMEQUIV|KXTOPALBUM|KXTOPSONG|KXBBSINGLE", "album/Billboard: priced by forecast readers (billboard_top.md, "
                                                      "consensus_reject_census.md)"),
    (r"KXYTVIEWSW?|KXNETFLIXRANK.*", "E1X-M media dead (e1x_media.md)"),
    (r"KX(BTC|ETH|SOL)(D|E|Y|W)?|KX(BTC|ETH|SOL)(MAX|MIN)[A-Z]*", "crypto OT1 tails / CB1 barrier ladders "
                                                                 "(options_implied_tails.md, ot1_extension.md, crypto_barrier.md)"),
    (r"KXINXU?|KXNASDAQ100U?", "OT1 index arm failed (options_implied_tails.md)"),
    (r"KX(WTI|BRENT|GOLD|NATGAS)(D|W)?|KXTTFMON", "Q016 commodities dead (q016_commodities.md, energy_monthly.md)"),
    (r"KXBIGGESTQUAKE", "E1X-Q daily quake dead (e1x_quake.md)"),
    (r"KX(HORMUZ|PANAMA|SUEZ|BABELMANDEB)WEEKLY|KXEIACRUDEW?", "PortWatch / EIA crude dead (data_ladders_far.md)"),
    (r"KXSPACEXCOUNT|KXSPRLVL|KXTORNADO.*|KXGSTORM|KXSEAICE|KXCO2LEVEL|KXEVMKT|KXMEASLES|KXLAUNCH.*|KXPJM.*|"
     r"KXMISOIND", "physical-ladder census / SX1 / SP1 dead (phys_ladders.md)"),
    (r"KX30YMORTW", "mortgage-rate forward logger (EDGES_AND_AGENTS)"),
    (r"KXALTASNOW|KXMAMMOTHSNOW|KXASPSNOWM", "resort snow: too few settled events (consensus_reject_census.md)"),
]
RELATED = [
    (r"KXBIGGESTQUAKE.+", "daily KXBIGGESTQUAKE dead (e1x_quake.md)"),
    (r"KX(SILVER|COPPER|PLATINUM|PALLADIUM)(D|W)?", "Q016 metals/commodities dead (q016_commodities.md)"),
    (r"KX.*ICE(MAX|MIN).*", "KXSEAICE too few events (phys_ladders.md)"),
    (r"KX[A-Z]*SNOW[A-Z]*", "snow: resort/city monthly snow too few settled events, city monthly snow = L39 (consensus_reject_census.md)"),
    (r"KX(BTC|ETH|SOL).*", "crypto ladders tested (OT1/CB1)"),
]


def tested_note(t: str) -> tuple[str, str]:
    """('tested'|'related'|'', note) for a series ticker."""
    for rx, note in TESTED:
        if re.fullmatch(rx, t):
            return "tested", note
    for rx, note in RELATED:
        if re.fullmatch(rx, t):
            return "related", note
    return "", ""


CROWDED_CATS = {"sports", "politics", "elections"}
MENTION_RE = re.compile(r"\b(say|says|said|mention|mentions|tweet|post on x|truth social)\b", re.I)
NEWS_SRC_RE = re.compile(r"bloomberg|guardian|cnbc|msnbc|semafor|theinformation|axios|nytimes|apnews|ap\.org|reuters|politico|cnn|foxnews|washingtonpost|wsj|bbc|nbc|cbs|abc", re.I)


def fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------- snapshot building
def series_meta(slist) -> dict:
    out = {}
    for s in slist:
        out[s["ticker"]] = {
            "title": s.get("title") or "", "category": s.get("category") or "", "tags": s.get("tags") or [],
            "frequency": s.get("frequency") or "", "fee_type": s.get("fee_type") or "",
            "sources": [[x.get("name") or "", x.get("url") or ""] for x in (s.get("settlement_sources") or [])],
            "last_updated_ts": s.get("last_updated_ts") or ""}
    return out


def add_events(active: dict, events) -> None:
    for e in events:
        st = e.get("series_ticker") or ""
        mk = e.get("markets") or []
        a = active.setdefault(st, {"n_ev": 0, "n_mk": 0, "max_mk": 0, "vol24": 0.0, "vol": 0.0, "oi": 0.0,
                                   "n_ladder_mk": 0, "n_tail": 0, "n_tail_stale": 0, "n_quoted": 0,
                                   "first_created": "", "next_close": "", "rules_primary": "", "rules_secondary": "",
                                   "sample_market": "", "strike_type": "", "event_title": ""})
        a["n_ev"] += 1
        a["n_mk"] += len(mk)
        a["max_mk"] = max(a["max_mk"], len(mk))
        for m in mk:
            v24 = fnum(m.get("volume_24h_fp"))
            a["vol24"] += v24
            a["vol"] += fnum(m.get("volume_fp"))
            a["oi"] += fnum(m.get("open_interest_fp"))
            ask, bid = fnum(m.get("yes_ask_dollars")), fnum(m.get("yes_bid_dollars"))
            if 0 < ask < 1:
                a["n_quoted"] += 1
            if len(mk) >= 4:
                a["n_ladder_mk"] += 1
                if 0.02 <= ask <= 0.15:          # a far tail quoted at a few cents
                    a["n_tail"] += 1
                    if v24 == 0:
                        a["n_tail_stale"] += 1
            ct = m.get("created_time") or ""
            if ct and (not a["first_created"] or ct < a["first_created"]):
                a["first_created"] = ct
            cl = m.get("close_time") or ""
            if cl and (not a["next_close"] or cl < a["next_close"]):
                a["next_close"] = cl
        if mk and not a["sample_market"]:
            m = mk[0]
            a["sample_market"] = m.get("ticker") or ""
            a["rules_primary"] = (m.get("rules_primary") or "")[:700]
            a["rules_secondary"] = (m.get("rules_secondary") or "")[:400]
            a["strike_type"] = m.get("strike_type") or ""
            a["event_title"] = e.get("title") or ""


def fetch_snapshot() -> dict:
    from kx_throttle import kget
    d = kget("/series", {})
    slist = (d or {}).get("series") or []
    if len(slist) < 1000:
        raise SystemExit(f"GET /series returned {len(slist)} series -- refusing to snapshot a truncated list")
    active, cur, pages = {}, None, 0
    while True:
        prm = {"status": "open", "with_nested_markets": "true", "limit": 200}
        if cur:
            prm["cursor"] = cur
        d = kget("/events", prm)
        if d is None:
            raise SystemExit(f"GET /events failed at page {pages}")
        add_events(active, d.get("events") or [])
        pages += 1
        cur = d.get("cursor")
        if not cur or not d.get("events") or pages > 2000:
            break
    print(f"fetched series={len(slist)} open-event pages={pages} active series={len(active)}")
    return {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "source": "live",
            "series": series_meta(slist), "active": active}


def snapshot_from_census(cdir: Path) -> dict:
    slist = json.loads((cdir / "kalshi_series.json").read_text())
    active = {}
    for f in sorted(glob.glob(str(cdir / "kalshi_events_*.json"))):
        add_events(active, json.loads(Path(f).read_text()).get("events") or [])
    return {"ts": cdir.name, "source": f"census {cdir}", "series": series_meta(slist), "active": active}


def save(snap: dict, date: str) -> Path:
    SNAPS.mkdir(parents=True, exist_ok=True)
    p = SNAPS / f"snap_{date}.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump(snap, f)
    return p


def load_prev(date: str):
    snaps = sorted(p for p in SNAPS.glob("snap_*.json.gz") if p.name < f"snap_{date}")
    if not snaps:
        return None, None
    with gzip.open(snaps[-1], "rt") as f:
        return json.load(f), snaps[-1].name[5:15]


# ---------------------------------------------------------------- history (11-16 fix: KXTEMPNYCHS was flagged as
# newly active although it had traded since 09-10 -- the 10-29 seed simply had no open event for it). A candidate is
# NEW only if NO market of the series closed (settled/closed/finalized) at or before the previous snapshot day.
GAP_DAYS = 3                                     # last pre-snapshot close within 3 days of it -> it was trading then


def _ts(d: str) -> int:
    return int(dt.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())


def _last_close_before(t: str, ts: int):
    """close_time of the most recent market of series t closing <= ts; '' if none; None if the GET failed."""
    from kx_throttle import kget
    d = kget("/markets", {"series_ticker": t, "max_close_ts": ts, "limit": 1})
    if d is None:
        return None
    ms = d.get("markets") or []
    return (ms[0].get("close_time") or "?") if ms else ""


def _created_before(t: str, ts: int):
    """created_time of a market of series t created <= ts (it was listed/open at the prev snapshot even if it closes
    later -- e.g. KXCOINBASE markets created 08-04, closing after 10-29); '' if none; None if the GET failed."""
    from kx_throttle import kget
    d = kget("/markets", {"series_ticker": t, "max_created_ts": ts, "limit": 1})
    if d is None:
        return None
    ms = d.get("markets") or []
    return (ms[0].get("created_time") or "?") if ms else ""


def history(tickers, prev_date: str, date: str) -> dict:
    """t -> {pre, last_pre_close, since_before}. Cached per (date, prev_date) so rerender needs no network twice."""
    SNAPS.mkdir(parents=True, exist_ok=True)
    cf = SNAPS / f"hist_{date}_vs_{prev_date}.json"
    cache = json.loads(cf.read_text()) if cf.exists() else {}
    cut = _ts(prev_date)
    todo = [t for t in tickers if t not in cache or cache[t].get("pre") is None
            or (cache[t].get("pre") is False and cache[t].get("created_pre") in (None, "?"))]
    for i, t in enumerate(todo):
        lc = _last_close_before(t, cut)
        h = {"pre": None if lc is None else bool(lc), "last_pre_close": lc or "", "since_before": "",
             "created_pre": ""}
        if h["pre"] is False:                    # nothing closed yet: was any market already listed back then?
            cp = _created_before(t, cut)
            h["created_pre"] = "?" if cp is None else cp
        if h["pre"]:                             # coarse age: still trading 30 / 365 days before the prev snapshot?
            for days in (365, 30):
                if _last_close_before(t, cut - days * 86400):
                    h["since_before"] = (dt.date.fromisoformat(prev_date) - dt.timedelta(days=days)).isoformat()
                    break
        cache[t] = h
        if i % 50 == 49:
            cf.write_text(json.dumps(cache))
            print(f"history {i + 1}/{len(todo)}")
    cf.write_text(json.dumps(cache))
    return cache


def kind_of(t: str, h: dict, prev: dict, prev_date: str) -> str:
    """NEW = no market closed before the prev snapshot day and not listed then; 'newly active' = listed before, first
    events now; 'reactivated' = history ended > GAP_DAYS before the prev snapshot; 'old (snapshot gap)' = it was
    trading at the prev snapshot, which just had no open event for it (NOT a lead); 'unverified' = history GET failed."""
    if h is None or h.get("pre") is None or h.get("created_pre") == "?":
        return "unverified"
    if not h["pre"]:
        if h.get("created_pre"):
            return "old (open at prev snapshot)"
        return "newly active" if t in prev["series"] else "NEW"
    lc = h["last_pre_close"][:10]
    gap = (dt.date.fromisoformat(prev_date) - dt.date.fromisoformat(lc)).days if re.match(r"\d{4}-\d\d-\d\d", lc) else 0
    return "reactivated" if gap > GAP_DAYS else "old (snapshot gap)"


QUEUE_KINDS = {"NEW", "newly active", "reactivated", "unverified"}


# ---------------------------------------------------------------- classification + score
def classify(t: str, meta: dict, act: dict) -> dict:
    srcs = " ".join(f"{n} {u}" for n, u in meta.get("sources") or [])
    text = f"{meta.get('title','')} {act.get('event_title','')} {srcs} {act.get('rules_primary','')}".lower()
    cat = (meta.get("category") or "").lower()
    dom, s1, s2 = "other", 0, 0
    for name, rx, v1, v2 in DOMAINS:
        if re.search(rf"\b(?:{rx})\b", text):   # word-bounded: "ice" must not match "voice" (11-15 bug)
            dom, s1, s2 = name, v1, v2
            break
    if dom == "other" and srcs and not NEWS_SRC_RE.search(srcs):
        s1 = 1                                   # some named non-news source
    max_mk = act.get("max_mk", 0)
    s3 = 2 if max_mk >= 6 else 1 if max_mk >= 3 else 0
    lad = act.get("n_ladder_mk", 0)
    stale_frac = act.get("n_tail_stale", 0) / lad if lad else 0.0
    s4 = 2 if stale_frac >= 0.30 else 1 if stale_frac >= 0.10 else 0
    freq = (meta.get("frequency") or "").lower()
    epm = FREQ_PER_MONTH.get(freq)
    if epm is None:                              # custom / one_off: fall back on open events now
        epm = min(act.get("n_ev", 0), 30)
    s5 = 2 if epm >= 4 else 1 if epm >= 1 else 0
    pen, why = 0, []
    if cat in CROWDED_CATS:
        pen, why = -3, [f"category {cat}"]
    elif MENTION_RE.search(f"{meta.get('title','')} {act.get('rules_primary','')}") and dom == "other":
        pen, why = -3, ["mention/say market"]
    score = s1 + s2 + s3 + s4 + s5 + pen
    promising = score >= 6 and act.get("n_mk", 0) >= 2 and pen == 0
    return {"domain": dom, "S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5, "pen": pen, "pen_why": why,
            "score": score, "promising": promising, "stale_frac": round(stale_frac, 3), "epm": round(epm, 2),
            "physical_or_data": dom != "other"}


# ---------------------------------------------------------------- digest
def diff(prev: dict, cur: dict):
    ps, pa = prev["series"], prev["active"]
    cs, ca = cur["series"], cur["active"]
    new_listed = sorted(set(cs) - set(ps))
    new_active = sorted(t for t in ca if t not in pa and t in ps)      # listed before, no open events before
    new_both = sorted(t for t in ca if t not in ps)                    # new AND already has open events
    gone = sorted(set(ps) - set(cs))
    return new_listed, new_active, new_both, gone


def esc(s: str) -> str:
    return (s or "").replace("|", "/").replace("\n", " ").strip()


def collapse(items, cur):
    """families (e.g. 50 per-state KXAAAGASCD<ST>): same ticker minus the last 2 chars, >= 3 members; the member with
    the highest 24h volume represents the family. items = [(t, c)] -> [(t, c, members)]"""
    fam: dict[str, list[str]] = {}
    for t, _ in items:
        fam.setdefault(t[:-2], []).append(t)
    shown, seen_fam = [], set()
    for t, c in items:
        k = t[:-2]
        if len(fam[k]) >= 3:
            if k in seen_fam:
                continue
            seen_fam.add(k)
            t = max(fam[k], key=lambda x: cur["active"][x]["vol24"])
            c = classify(t, cur["series"].get(t, {}), cur["active"].get(t, {}))
        shown.append((t, c, fam[k] if len(fam[k]) >= 3 else [t]))
    return shown


def render(date: str, prev_date: str, prev: dict, cur: dict) -> tuple[str, list[str]]:
    new_listed, new_active, new_both, gone = diff(prev, cur)
    cands = []
    for t in sorted(set(new_both) | set(new_active)):
        c = classify(t, cur["series"].get(t, {}), cur["active"].get(t, {}))
        cands.append((t, c))
    cands.sort(key=lambda x: (-x[1]["score"], -cur["active"][x[0]]["vol24"]))
    hist = history([t for t, _ in cands], prev_date, date)
    kind = {t: kind_of(t, hist.get(t), prev, prev_date) for t, _ in cands}
    tst = {t: tested_note(t) for t, _ in cands}
    nk = {k: sum(1 for t in kind if kind[t] == k) for k in
          ("NEW", "newly active", "reactivated", "old (snapshot gap)", "old (open at prev snapshot)", "unverified")}
    prom = [x for x in cands if x[1]["promising"]]
    leads = [x for x in prom if kind[x[0]] in QUEUE_KINDS and tst[x[0]][0] != "tested"]
    rest = [x for x in prom if x not in leads]
    shown = collapse(leads, cur)
    L = [f"# Kalshi series watch -- {date}", "",
         f"Diff: snapshot {date} ({cur['source']}, {cur['ts']}) vs {prev_date} ({prev['source']}, {prev['ts']}).",
         f"Series listed: {len(prev['series'])} -> {len(cur['series'])} (+{len(new_listed)} new, -{len(gone)} removed). "
         f"Series with open events: {len(prev['active'])} -> {len(cur['active'])}.",
         f"Open now but not open at {prev_date}: {len(cands)}. History check (GET /markets max_close_ts <= {prev_date}): "
         + ", ".join(f"{k} {v}" for k, v in nk.items()) + ". "
         f"New but not yet open: {len(set(new_listed) - set(new_both))}.",
         f"PROMISING (score >= 6, >= 2 markets, no sports/politics/mentions penalty): {len(prom)}; of these "
         f"**{len(leads)} are leads** (NEW / newly active / reactivated AND not already tested) -> {len(shown)} after "
         f"collapsing families" + (f" (top {MAX_HEADINGS} get headings)" if len(shown) > MAX_HEADINGS else "") + ".", "",
         "Score = S1 source verifiable + S2 public forecast + S3 ladder + S4 stale far brackets + S5 events/month "
         "(each 0-2) - 3 penalty. NEW = no market of the series closed on/before the previous snapshot day. "
         "Read-only snapshot; nothing here is a tested edge.", "",
         "## Promising leads (untested, new or newly active)", ""]
    queue = []
    for t, c, members in shown[:MAX_HEADINGS]:
        m, a, h = cur["series"].get(t, {}), cur["active"][t], hist.get(t) or {}
        srcs = "; ".join(f"{n} <{u}>" for n, u in m.get("sources") or []) or "(none listed)"
        name = t if len(members) == 1 else f"{t[:-2]}* family ({len(members)} series, e.g. {t})"
        L += [f"### {name} -- {esc(m.get('title'))} (score {c['score']}/10, {kind[t]})", ""]
        if len(members) > 1:
            L += [f"- members: {', '.join(members)}"]
        if tst[t][0] == "related":
            L += [f"- RELATED already tested: {tst[t][1]}"]
        if kind[t] == "reactivated":
            L += [f"- history: last market closed {h.get('last_pre_close', '')[:10]} (before {prev_date}); "
                  f"trading since before {h.get('since_before') or prev_date}"]
        L += [
              f"- category {m.get('category')} / domain {c['domain']} / frequency {m.get('frequency') or '?'} "
              f"(~{c['epm']} events/month) / strike_type {a.get('strike_type') or '?'}",
              f"- open: {a['n_ev']} events, {a['n_mk']} markets (max {a['max_mk']}/event); 24h volume "
              f"{a['vol24']:.0f}, total volume {a['vol']:.0f}, OI {a['oi']:.0f}; first open market created "
              f"{a['first_created'][:16]}, next close {a['next_close'][:16]}",
              f"- checklist: S1 source {c['S1']}, S2 forecast {c['S2']}, S3 ladder {c['S3']}, S4 stale tails "
              f"{c['S4']} ({a['n_tail_stale']}/{a['n_ladder_mk']} ladder markets quoted 2-15c with 0 volume 24h), "
              f"S5 cadence {c['S5']}",
              f"- settlement sources: {esc(srcs)}",
              f"- rules ({a['sample_market']}): {esc(a['rules_primary'])[:500]}", ""]
        qt = t if len(members) == 1 else f"{t[:-2]}*"
        rel = f" [related tested: {tst[t][1]}]" if tst[t][0] == "related" else ""
        queue.append(f"- {date} **{qt}** {esc(m.get('title'))}"
                     f"{' (+%d siblings)' % (len(members) - 1) if len(members) > 1 else ''} -- {kind[t]}, score "
                     f"{c['score']}/10, {c['domain']}, {m.get('frequency') or '?'}, {a['n_mk']} mkts, vol24 "
                     f"{a['vol24']:.0f}{rel} (notes/series_watch/{date}.md)")
    if not shown:
        L += ["(none today)", ""]
    if rest:
        L += ["## Promising score but NOT a lead (already tested, or not actually new)", ""]
        for t, c, members in collapse(rest, cur):
            h = hist.get(t) or {}
            name = t if len(members) == 1 else f"{t[:-2]}* ({len(members)} series)"
            why = []
            if tst[t][0] == "tested":
                why.append(f"TESTED: {tst[t][1]}")
            if kind[t] == "old (open at prev snapshot)":
                why.append(f"{kind[t]}: first market created {h.get('created_pre', '')[:10]}")
            elif kind[t] not in QUEUE_KINDS:
                why.append(f"{kind[t]}: traded before {prev_date} (last close {h.get('last_pre_close', '')[:10]}, "
                           f"since before {h.get('since_before') or prev_date})")
            elif kind[t] == "reactivated":
                why.append(f"reactivated (last close {h.get('last_pre_close', '')[:10]})")
            L.append(f"- {name} -- {esc(cur['series'].get(t, {}).get('title'))} (score {c['score']}): "
                     + "; ".join(why))
        L.append("")
    L += ["## All series open now but not at the previous snapshot", "",
          "| series | title | cat | freq | kind | ev | mkts | vol24 | score | flags |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t, c in cands[:200]:
        m, a = cur["series"].get(t, {}), cur["active"][t]
        flags = ",".join(c["pen_why"]) or c["domain"]
        if tst[t][0]:
            flags += f",{tst[t][0]}"
        L.append(f"| {t} | {esc(m.get('title'))[:60]} | {m.get('category','')} | {m.get('frequency','')} | "
                 f"{kind[t]} | {a['n_ev']} | {a['n_mk']} | {a['vol24']:.0f} | {c['score']} | {flags} |")
    if len(cands) > 200:
        L.append(f"\n({len(cands) - 200} more not shown)")
    pend = sorted(set(new_listed) - set(new_both))
    if pend:
        L += ["", f"New series listed with no open events yet ({len(pend)}): " +
              ", ".join(f"{t} ({esc(cur['series'][t]['title'])[:40]})" for t in pend[:80])
              + (" ..." if len(pend) > 80 else "")]
    return "\n".join(L) + "\n", queue


# ---------------------------------------------------------------- outputs
def write_safe(p: Path, txt: str, mode: str = "w") -> bool:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, mode) as f:
            f.write(txt)
        return True
    except OSError as e:                         # TCC blocks ~/Desktop for launchd jobs
        print(f"cannot write {p}: {e}")
        return False


def queue_append(qfile: Path, lines: list[str]) -> list[str]:
    """Append lines (deduped by series ticker) under '## From series watch' in RESEARCH_QUEUE.md. Returns added."""
    txt = qfile.read_text()
    hdr = "## From series watch"
    if hdr not in txt:
        txt = txt.rstrip("\n") + f"\n\n{hdr} (auto, scripts/series_watch.py; read-only census leads, untested)\n"
    sect = txt[txt.index(hdr):]
    add = [l for l in lines if (m := re.search(r"\*\*(\S+)\*\*", l)) and f"**{m.group(1)}**" not in sect]
    if add:
        txt = txt.rstrip("\n") + "\n" + "\n".join(add) + "\n"
        qfile.write_text(txt)
    return add


def publish(date: str, digest: str, queue: list[str]) -> None:
    write_safe(ROOT / "digests" / f"series_watch-{date}.md", digest)          # vault-indexed (DIGESTS glob)
    write_safe(ROOT / "notes" / "series_watch" / f"{date}.md", digest)
    if queue:
        write_safe(ROOT / "notes" / "series_watch" / "queue_pending.md", "\n".join(queue) + "\n", "a")
    if os.environ.get("SW_NO_MIRROR") == "1":
        # under launchd an open() on ~/Desktop does not fail, it HANGS on the TCC prompt (seen 11-15: 15 min stuck in
        # open(), killed) -> never touch the worktree from the deployed job; a session runs `series_watch.py sync`
        print("SW_NO_MIRROR=1: worktree mirror skipped (run `series_watch.py sync` from a session)")
    elif ROOT.resolve() != WORKTREE.resolve():
        sync(WORKTREE)
    else:
        flush_pending(WORKTREE)


def flush_pending(wt: Path) -> None:
    pend = ROOT / "notes" / "series_watch" / "queue_pending.md"
    q = wt / "notes" / "RESEARCH_QUEUE.md"
    if not pend.exists():
        return
    try:
        added = queue_append(q, [l for l in pend.read_text().splitlines() if l.strip()])
        pend.unlink()
        print(f"RESEARCH_QUEUE.md: +{len(added)} series")
    except OSError as e:
        print(f"queue not flushed (pending kept in {pend}): {e}")


def sync(wt: Path) -> None:
    src = ROOT / "notes" / "series_watch"
    if src.resolve() != (wt / "notes" / "series_watch").resolve():
        for p in sorted(src.glob("20*.md")):
            dst = wt / "notes" / "series_watch" / p.name
            try:
                if not dst.exists() or dst.read_text() != p.read_text():
                    write_safe(dst, p.read_text())
            except OSError as e:
                print(f"sync skipped ({e})")
                return
    flush_pending(wt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "rerender", "seed", "sync"])
    ap.add_argument("--date", default=dt.datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--census")
    ap.add_argument("--worktree", default=str(WORKTREE))
    a = ap.parse_args()
    if a.cmd == "seed":
        snap = snapshot_from_census(Path(a.census))
        print(f"seed {a.date}: series={len(snap['series'])} active={len(snap['active'])} -> {save(snap, a.date)}")
        return
    if a.cmd == "sync":
        sync(Path(a.worktree))
        return
    prev, prev_date = load_prev(a.date)
    if a.cmd == "rerender":                      # re-score an existing snapshot (no network), e.g. after a rule fix
        with gzip.open(SNAPS / f"snap_{a.date}.json.gz", "rt") as f:
            cur = json.load(f)
    else:
        cur = fetch_snapshot()
        p = save(cur, a.date)
        print(f"saved {p} ({p.stat().st_size} bytes)")
    if prev is None:
        print("no previous snapshot: baseline only, no digest")
        return
    digest, queue = render(a.date, prev_date, prev, cur)
    publish(a.date, digest, queue)
    print(f"digest {a.date} vs {prev_date}: {len(queue)} promising")


if __name__ == "__main__":
    main()
