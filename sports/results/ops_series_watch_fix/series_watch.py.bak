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


def render(date: str, prev_date: str, prev: dict, cur: dict) -> tuple[str, list[str]]:
    new_listed, new_active, new_both, gone = diff(prev, cur)
    cands = []
    for t in sorted(set(new_both) | set(new_active)):
        c = classify(t, cur["series"].get(t, {}), cur["active"].get(t, {}))
        cands.append((t, c))
    cands.sort(key=lambda x: (-x[1]["score"], -cur["active"][x[0]]["vol24"]))
    prom = [x for x in cands if x[1]["promising"]]
    L = [f"# Kalshi series watch -- {date}", "",
         f"Diff: snapshot {date} ({cur['source']}, {cur['ts']}) vs {prev_date} ({prev['source']}, {prev['ts']}).",
         f"Series listed: {len(prev['series'])} -> {len(cur['series'])} (+{len(new_listed)} new, -{len(gone)} removed). "
         f"Series with open events: {len(prev['active'])} -> {len(cur['active'])}.",
         f"New series with open events: {len(new_both)}. Previously listed, newly active: {len(new_active)}. "
         f"New but not yet open: {len(set(new_listed) - set(new_both))}.",
         f"PROMISING (score >= 6, >= 2 markets, no sports/politics/mentions penalty): **{len(prom)}**"
         + (f" (top {MAX_HEADINGS} get headings)" if len(prom) > MAX_HEADINGS else "") + ".", "",
         "Score = S1 source verifiable + S2 public forecast + S3 ladder + S4 stale far brackets + S5 events/month "
         "(each 0-2) - 3 penalty. Read-only snapshot; nothing here is a tested edge.", ""]
    L.append("## Promising series")
    L.append("")
    queue = []
    # collapse families (e.g. 50 per-state KXAAAGASCD<ST> series) into one heading: same ticker minus the last
    # 2 chars, >= 3 promising members; the member with the highest 24h volume represents the family
    fam: dict[str, list[str]] = {}
    for t, _ in prom:
        fam.setdefault(t[:-2], []).append(t)
    shown, seen_fam = [], set()
    for t, c in prom:
        k = t[:-2]
        if len(fam[k]) >= 3:
            if k in seen_fam:
                continue
            seen_fam.add(k)
            t = max(fam[k], key=lambda x: cur["active"][x]["vol24"])
            c = classify(t, cur["series"].get(t, {}), cur["active"].get(t, {}))
        shown.append((t, c, fam[k] if len(fam[k]) >= 3 else [t]))
    L.insert(-3, f"Promising families after collapsing per-state/per-item siblings: {len(shown)}.")
    for t, c, members in shown[:MAX_HEADINGS]:
        m, a = cur["series"].get(t, {}), cur["active"][t]
        kind = "NEW" if t in new_both else "newly active"
        srcs = "; ".join(f"{n} <{u}>" for n, u in m.get("sources") or []) or "(none listed)"
        name = t if len(members) == 1 else f"{t[:-2]}* family ({len(members)} series, e.g. {t})"
        L += [f"### {name} -- {esc(m.get('title'))} (score {c['score']}/10, {kind})", ""]
        if len(members) > 1:
            L += [f"- members: {', '.join(members)}"]
        L += [
              f"- category {m.get('category')} / domain {c['domain']} / frequency {m.get('frequency') or '?'} "
              f"(~{c['epm']} events/month) / strike_type {a.get('strike_type') or '?'}",
              f"- open: {a['n_ev']} events, {a['n_mk']} markets (max {a['max_mk']}/event); 24h volume "
              f"{a['vol24']:.0f}, total volume {a['vol']:.0f}, OI {a['oi']:.0f}; first market created "
              f"{a['first_created'][:16]}, next close {a['next_close'][:16]}",
              f"- checklist: S1 source {c['S1']}, S2 forecast {c['S2']}, S3 ladder {c['S3']}, S4 stale tails "
              f"{c['S4']} ({a['n_tail_stale']}/{a['n_ladder_mk']} ladder markets quoted 2-15c with 0 volume 24h), "
              f"S5 cadence {c['S5']}",
              f"- settlement sources: {esc(srcs)}",
              f"- rules ({a['sample_market']}): {esc(a['rules_primary'])[:500]}", ""]
        qt = t if len(members) == 1 else f"{t[:-2]}*"
        queue.append(f"- {date} **{qt}** {esc(m.get('title'))}{' (+%d siblings)' % (len(members) - 1) if len(members) > 1 else ''} -- score {c['score']}/10, {c['domain']}, "
                     f"{m.get('frequency') or '?'}, {a['n_mk']} mkts, vol24 {a['vol24']:.0f} "
                     f"(notes/series_watch/{date}.md)")
    if not prom:
        L += ["(none today)", ""]
    L += ["## All new / newly-active series", "",
          "| series | title | cat | freq | kind | ev | mkts | vol24 | score | flags |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t, c in cands[:200]:
        m, a = cur["series"].get(t, {}), cur["active"][t]
        flags = ",".join(c["pen_why"]) or c["domain"]
        L.append(f"| {t} | {esc(m.get('title'))[:60]} | {m.get('category','')} | {m.get('frequency','')} | "
                 f"{'new' if t in new_both else 'react'} | {a['n_ev']} | {a['n_mk']} | {a['vol24']:.0f} | "
                 f"{c['score']} | {flags} |")
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
