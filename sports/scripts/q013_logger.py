"""Q013 forward paper logger: Kalshi Rotten Tomatoes (KXRT) / Metacritic (KXMC) brackets vs the live score.
NO ORDERS ARE EVER SENT. Public GET endpoints only, no API key.

    python scripts/q013_logger.py tick      # launchd every 15 min: books + live scores + frozen paper picks
    python scripts/q013_logger.py report    # settle picks on Kalshi results -> ROI (CI), calibration, CLV
    python scripts/q013_logger.py map       # print the event -> RT/MC page mapping (resolve + verify)

Pre-registration: notes/q013_rt_scores.md (frozen 2026-10-29 before the first tick). Everything below marked FROZEN
is the registered rule; do not edit it after the first pick without filing a new trial.

Files (data/q013/, our UTC clock is the point-in-time stamp for everything):
  snapshots.jsonl   every open KXRT/KXMC market each tick: yes bid/ask (+sizes), volume, OI
  scores.jsonl      every score fetch: score, review count, liked count (RT) / review mean+sd (MC)
  changes.jsonl     first time we see a new (score, n, liked) state for an event
  mc_reviews.jsonl  every Metacritic critic review the first time we see it (first_seen = our clock)
  picks.jsonl       one row per (decision, ticker) at the first tick inside [close - H, close - H + 60 min]
  state.json        event -> page mapping + last fetch time (cache: politeness)
  overrides.json    OPTIONAL hand mapping {event_ticker: "rt:/m/slug" | "mc:slug"}; wins over the resolver
"""
from __future__ import annotations

import html as _html
import json
import math
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from scipy.stats import betabinom, norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "q013"
API = "https://api.elections.kalshi.com/trade-api/v2"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/130.0 Safari/537.36")
RT = "https://www.rottentomatoes.com"
MC = "https://www.metacritic.com"
MC_API = "https://backend.metacritic.com/reviews/metacritic/critic/games/{slug}/web"
SERIES = ("KXRT", "KXMC")

# ---------------------------------------------------------------- FROZEN rule (notes/q013_rt_scores.md section 4)
DECISIONS = {"T24": 24 * 3600, "T6": 6 * 3600}   # decision time tau = close_time - H
WINDOW = 3600                                     # pick at the first tick inside [tau, tau + WINDOW)
REMAIN_FRAC = {"T24": 0.08, "T6": 0.04}           # remaining reviews m = max(M_MIN, ceil(frac * n))
M_MIN = 2
N_MIN = 10                                        # no pick with fewer reviews than this
PRIOR_A, PRIOR_B = 1.0, 1.0                       # Beta prior on P(fresh) of a remaining review (RT)
MC_SD_FALLBACK = 10.0                             # per-review sd if the MC review API fails
EDGE = 0.05                                       # model - price - fee >= 5c
FEE = 0.07                                        # Kalshi quadratic taker fee: 0.07 p (1-p) per contract
SEED, NBOOT = 1013, 10000

# politeness: re-fetch a score page at most every 15 min if close <= 8 days away, else hourly
FAST_DAYS, FAST_S, SLOW_S = 8, 14 * 60, 59 * 60


def now_ts() -> int:
    return int(time.time())


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def pts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def append(name: str, rows: list[dict]) -> None:
    if rows:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / name).open("a") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")


def read(name: str) -> list[dict]:
    p = OUT / name
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def load_json(name: str, default):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else default


def save_json(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(OUT / name)


# ---------------------------------------------------------------- HTTP
def kalshi(path: str, params=None, tries: int = 10) -> dict:
    """Kalshi public GET with backoff on 429 (the IP is shared with the other paper agents)."""
    last = None
    for i in range(tries):
        try:
            r = requests.get(f"{API}{path}", params=params, timeout=30, headers={"User-Agent": UA})
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code} {r.text[:120]}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(min(20.0, 2.0 * (i + 1)))   # <= ~110 s in total
    raise RuntimeError(f"kalshi {path} failed: {last}")


def web(url: str, params=None, tries: int = 3) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30, headers={"User-Agent": UA, "Accept-Language": "en-US"})
            if r.status_code in (200, 404):
                return r
            last = f"{r.status_code}"
            if r.status_code in (401, 403):   # blocked: never retry harder, never bypass
                break
        except requests.RequestException as e:
            last = str(e)
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def open_markets(series: str) -> list[dict]:
    out, cursor = [], None
    for _ in range(10):
        p = {"series_ticker": series, "status": "open", "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        d = kalshi("/markets", p)
        out += d.get("markets", [])
        cursor = d.get("cursor")
        if not cursor:
            break
    return out


# ---------------------------------------------------------------- title -> page resolution
def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", _html.unescape(s or "")).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", "and")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def event_title(m: dict) -> str:
    t = m.get("title") or ""
    return re.sub(r"\s+(Rotten Tomatoes|Metacritic) score\?\s*$", "", t).strip()


def rt_resolve(title: str, year: int) -> dict | None:
    """RT site search -> movie row whose title matches and release year is year or year-1."""
    r = web(f"{RT}/search", {"search": title})
    if r.status_code != 200:
        return None
    h = r.text
    i = h.find('type="movie"')
    if i < 0:
        return None
    j = h.find("<search-page-result", i + 10)
    block = h[i: j if j > 0 else len(h)]
    rows = re.findall(r'<search-page-media-row(.*?)</search-page-media-row>', block, re.S)
    cands = []
    for row in rows:
        y = re.search(r'release-year="(\d*)"', row)
        u = re.search(r'href="https://www\.rottentomatoes\.com(/m/[^"]+)"', row)
        a = re.search(r'alt="([^"]*)"', row)
        if not u:
            continue
        cands.append({"path": u.group(1), "title": _html.unescape(a.group(1)) if a else "",
                      "year": int(y.group(1)) if y and y.group(1) else None})
    nt = norm_title(title)
    for strict in (True, False):
        for c in cands:
            ok_year = c["year"] in (year, year - 1) if c["year"] else not strict
            ok_title = norm_title(c["title"]) == nt if strict else nt in norm_title(c["title"])
            if ok_year and ok_title:
                return c
    return None


def mc_slug(title: str) -> str:
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 -]+", "", s.replace("&", "and"))
    return re.sub(r"[\s-]+", "-", s).strip("-")


# ---------------------------------------------------------------- score pages
def rt_score(path: str) -> dict:
    r = web(f"{RT}{path}")
    if r.status_code != 200:
        raise RuntimeError(f"RT {path} -> {r.status_code}")
    h = r.text
    m = re.search(r'<script[^>]*id="media-scorecard-json"[^>]*>(.*?)</script>', h, re.S)
    if not m:
        raise RuntimeError(f"RT {path}: no media-scorecard-json (page layout changed?)")
    d = json.loads(m.group(1))
    c = d.get("criticsScore") or {}
    name = re.search(r'<script type="application/ld\+json">(.*?)</script>', h, re.S)
    nm = None
    if name:
        try:
            nm = json.loads(name.group(1)).get("name")
        except ValueError:
            pass
    liked, notl = num(c.get("likedCount")), num(c.get("notLikedCount"))
    n = num(c.get("reviewCount"))
    return {"page_title": nm, "score": num(c.get("score")), "n": n, "liked": liked, "not_liked": notl,
            "avg_rating": num(c.get("averageRating")), "certified": c.get("certified"),
            "score_calc": rt_round(liked, (liked or 0) + (notl or 0)) if liked is not None and notl is not None else None}


def rt_round(liked: float, n: float) -> int | None:
    """FROZEN: displayed Tomatometer = round-half-up(100 * liked / n). The logger records score vs score_calc so a
    rounding mismatch is visible (report prints the mismatch rate)."""
    if not n:
        return None
    return int(math.floor(100.0 * liked / n + 0.5))


def mc_score(slug: str, known: set) -> tuple[dict, list[dict]]:
    r = web(f"{MC}/game/{slug}/")
    if r.status_code != 200:
        raise RuntimeError(f"MC {slug} -> {r.status_code}")
    h = r.text
    nm = rel = None
    ms = n = None
    for m in re.finditer(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', h, re.S):
        try:
            d = json.loads(m.group(1))
        except ValueError:
            continue
        if d.get("@type") == "VideoGame":
            nm, rel = d.get("name"), d.get("datePublished")
            ar = d.get("aggregateRating") or {}
            ms, n = num(ar.get("ratingValue")), num(ar.get("reviewCount"))
    out = {"page_title": nm, "release": rel, "score": ms, "n": n}
    new = []
    if n:   # every critic review (score, publication, date): mean/sd + first_seen of each new one
        try:
            items, off = [], 0
            while off < 1000:
                d = web(MC_API.format(slug=slug), {"offset": off, "limit": 50, "filterBySentiment": "all",
                                                   "sort": "date", "componentName": "critic-reviews",
                                                   "componentDisplayName": "critic Reviews",
                                                   "componentType": "ReviewList"}).json()["data"]
                got = d.get("items") or []
                items += got
                off += len(got)   # the API pages ~10 items whatever `limit` says
                if not got or off >= (d.get("totalResults") or 0):
                    break
                time.sleep(1)
            sc = [num(x.get("score")) for x in items if num(x.get("score")) is not None]
            out.update({"api_n": len(sc), "api_mean": float(np.mean(sc)) if sc else None,
                        "api_sd": float(np.std(sc, ddof=1)) if len(sc) > 1 else None})
            for x in items:
                key = f"{x.get('publicationSlug')}|{x.get('url')}|{x.get('platform')}"
                if key not in known:
                    new.append({"key": key, "score": num(x.get("score")), "publication": x.get("publicationName"),
                                "date": x.get("date"), "platform": x.get("platform")})
        except Exception as e:  # noqa: BLE001
            out["api_error"] = str(e)[:200]
    return out, new


# ---------------------------------------------------------------- FROZEN model
def yes_set(m: dict):
    """Integer-score predicate for a market (rules: 'above X' is strict; 'between' is inclusive)."""
    st, f, c = m.get("strike_type"), num(m.get("floor_strike")), num(m.get("cap_strike"))
    if st == "greater" and f is not None:
        return lambda v: v > f
    if st == "greater_or_equal" and f is not None:
        return lambda v: v >= f
    if st == "less" and c is not None:
        return lambda v: v < c
    if st == "less_or_equal" and c is not None:
        return lambda v: v <= c
    if st == "between" and f is not None and c is not None:
        return lambda v: f <= v <= c
    return None


def remaining(n: float, decision: str) -> int:
    return max(M_MIN, int(math.ceil(REMAIN_FRAC[decision] * n)))


def pmf_rt(liked: int, n: int, decision: str) -> dict[int, float]:
    """Final Tomatometer pmf: m more reviews, K fresh ~ BetaBinom(m, liked + a, n - liked + b)."""
    m = remaining(n, decision)
    k = np.arange(m + 1)
    pk = betabinom.pmf(k, m, liked + PRIOR_A, n - liked + PRIOR_B)
    out: dict[int, float] = {}
    for kk, p in zip(k, pk):
        v = rt_round(liked + kk, n + m)
        out[v] = out.get(v, 0.0) + float(p)
    return out


def pmf_mc(score: float, n: int, sd_review: float | None, decision: str) -> dict[int, float]:
    """Final Metascore pmf: displayed score + m new reviews drawn around the current mean.
    delta sd = m/(n+m) * s * sqrt(1/m + 1/n) (predictive), plus the displayed-integer rounding (var 1/12)."""
    m = remaining(n, decision)
    s = sd_review if sd_review and sd_review > 0 else MC_SD_FALLBACK
    sd = math.sqrt((m / (n + m)) ** 2 * s * s * (1 / m + 1 / n) + 1 / 12)
    v = np.arange(0, 101)
    p = norm.cdf((v + 0.5 - score) / sd) - norm.cdf((v - 0.5 - score) / sd)
    p = p / p.sum()
    return {int(a): float(b) for a, b in zip(v, p) if b > 1e-12}


def p_yes(pmf: dict[int, float], pred) -> float:
    return float(sum(p for v, p in pmf.items() if pred(v)))


def fee(p: float) -> float:
    return FEE * p * (1 - p)


def decide(p: float, bid, ask) -> dict:
    """FROZEN bet rule: YES at the ask if p - ask - fee >= EDGE; else NO at 1 - bid if (1-p) - (1-bid) - fee >= EDGE."""
    if bid is None or ask is None or not (0 < bid < ask < 1):
        return {"side": None, "reason": "no two-sided quote"}
    ey = p - ask - fee(ask)
    en = (1 - p) - (1 - bid) - fee(1 - bid)
    side = "yes" if ey >= EDGE and ey >= en else ("no" if en >= EDGE else None)
    price = ask if side == "yes" else (1 - bid if side == "no" else None)
    return {"side": side, "price": price, "cost": price + fee(price) if side else None,
            "edge_yes": ey, "edge_no": en, "reason": None if side else "edge < 5c"}


# ---------------------------------------------------------------- tick
def resolve_event(ev: str, title: str, series: str, close_ts: int, st: dict, ov: dict) -> dict:
    rec = st["events"].get(ev)
    if ev in ov:
        src, ref = ov[ev].split(":", 1)
        if not rec or rec.get("ref") != ref:
            rec = {"source": src, "ref": ref, "query": title, "how": "override", "resolved_at": now_ts()}
    if rec and rec.get("ref"):
        return rec
    if rec and now_ts() - rec.get("resolved_at", 0) < 6 * 3600:   # unresolved: retry every 6 h, not every tick
        return rec
    year = datetime.fromtimestamp(close_ts, timezone.utc).year
    if series == "KXRT":
        c = rt_resolve(title, year)
        rec = {"source": "rt", "ref": c["path"] if c else None, "page": c, "query": title, "how": "search",
               "resolved_at": now_ts()}
    else:
        slug = mc_slug(title)
        r = web(f"{MC}/game/{slug}/")
        rec = {"source": "mc", "ref": slug if r.status_code == 200 else None, "query": title, "how": "slug",
               "resolved_at": now_ts()}
    st["events"][ev] = rec
    return rec


def tick() -> dict:
    ts = now_ts()
    st = load_json("state.json", {"events": {}, "first_tick": ts})
    ov = load_json("overrides.json", {})
    errors = []
    mk = []
    for s in SERIES:
        try:
            mk += [dict(x, _series=s) for x in open_markets(s)]
        except Exception as e:  # noqa: BLE001
            errors.append(f"{s}: {e}")
    snaps = [{"ts": ts, "series": m["_series"], "event": m["event_ticker"], "ticker": m["ticker"],
              "strike_type": m.get("strike_type"), "floor": num(m.get("floor_strike")), "cap": num(m.get("cap_strike")),
              "close_ts": pts(m["close_time"]), "yes_bid": num(m.get("yes_bid_dollars")),
              "yes_ask": num(m.get("yes_ask_dollars")), "bid_size": num(m.get("yes_bid_size_fp")),
              "ask_size": num(m.get("yes_ask_size_fp")), "volume": num(m.get("volume_fp")),
              "oi": num(m.get("open_interest_fp")), "status": m.get("status")} for m in mk]
    append("snapshots.jsonl", snaps)

    events: dict[str, dict] = {}
    for m in mk:
        e = events.setdefault(m["event_ticker"], {"series": m["_series"], "title": event_title(m),
                                                  "close_ts": pts(m["close_time"]), "markets": []})
        e["markets"].append(m)

    known_mc = {r["key"] for r in read("mc_reviews.jsonl")}
    last_state = {}
    for r in read("changes.jsonl"):
        last_state[r["event"]] = (r.get("score"), r.get("n"), r.get("liked"))
    scores_now: dict[str, dict] = {}
    score_rows, change_rows, review_rows = [], [], []
    for ev, e in sorted(events.items(), key=lambda kv: kv[1]["close_ts"]):
        try:
            rec = resolve_event(ev, e["title"], e["series"], e["close_ts"], st, ov)
        except Exception as ex:  # noqa: BLE001
            errors.append(f"resolve {ev}: {ex}")
            continue
        if not rec.get("ref"):
            continue
        due = FAST_S if e["close_ts"] - ts <= FAST_DAYS * 86400 else SLOW_S
        in_decision = any(e["close_ts"] - H <= ts < e["close_ts"] - H + WINDOW for H in DECISIONS.values())
        if ts - rec.get("last_fetch", 0) < due and not in_decision:
            continue
        try:
            if rec["source"] == "rt":
                sc = rt_score(rec["ref"])
            else:
                sc, new = mc_score(rec["ref"], known_mc)
                for x in new:
                    review_rows.append({"first_seen": ts, "event": ev, **x})
                    known_mc.add(x["key"])
            time.sleep(1)
        except Exception as ex:  # noqa: BLE001
            errors.append(f"score {ev}: {ex}")
            continue
        rec["last_fetch"] = ts
        rec["page_title"] = sc.get("page_title")
        row = {"ts": ts, "event": ev, "source": rec["source"], "ref": rec["ref"], **sc}
        score_rows.append(row)
        scores_now[ev] = row
        key = (sc.get("score"), sc.get("n"), sc.get("liked"))
        if last_state.get(ev) != key:
            change_rows.append({"first_seen": ts, "event": ev, "score": key[0], "n": key[1], "liked": key[2],
                                "prev": last_state.get(ev)})
    append("scores.jsonl", score_rows)
    append("changes.jsonl", change_rows)
    append("mc_reviews.jsonl", review_rows)

    # ---- frozen paper picks
    done = {(p["decision"], p["ticker"]) for p in read("picks.jsonl")}
    picks = []
    for ev, e in events.items():
        for dname, H in DECISIONS.items():
            tau = e["close_ts"] - H
            if not (tau <= ts < tau + WINDOW):
                continue
            sc = scores_now.get(ev)
            for m in e["markets"]:
                if (dname, m["ticker"]) in done:
                    continue
                base = {"ts": ts, "decision": dname, "series": e["series"], "event": ev, "ticker": m["ticker"],
                        "close_ts": e["close_ts"], "tau": tau, "yes_bid": num(m.get("yes_bid_dollars")),
                        "yes_ask": num(m.get("yes_ask_dollars")), "strike_type": m.get("strike_type"),
                        "floor": num(m.get("floor_strike")), "cap": num(m.get("cap_strike"))}
                pred = yes_set(m)
                if sc is None or pred is None or not sc.get("n") or sc["n"] < N_MIN or sc.get("score") is None:
                    picks.append({**base, "side": None, "reason": "no score / n < N_MIN / unknown strike",
                                  "n": sc.get("n") if sc else None})
                    continue
                n = int(sc["n"])
                if e["series"] == "KXRT":
                    if sc.get("liked") is None:
                        picks.append({**base, "side": None, "reason": "no liked count"})
                        continue
                    pm = pmf_rt(int(sc["liked"]), n, dname)
                else:
                    pm = pmf_mc(float(sc["score"]), n, sc.get("api_sd"), dname)
                p = p_yes(pm, pred)
                picks.append({**base, "score": sc["score"], "n": n, "liked": sc.get("liked"),
                              "m_remaining": remaining(n, dname), "p_yes": p, **decide(p, base["yes_bid"], base["yes_ask"])})
    append("picks.jsonl", picks)
    save_json("state.json", st)
    out = {"ts": iso(ts), "markets": len(snaps), "events": len(events), "scores_fetched": len(score_rows),
           "changes": len(change_rows), "new_mc_reviews": len(review_rows), "picks_logged": len(picks),
           "bets": sum(1 for p in picks if p.get("side")), "errors": errors}
    return out


# ---------------------------------------------------------------- report
def results_for(tickers: set) -> dict:
    cache = load_json("results.json", {})
    for t in sorted(tickers - set(cache)):
        try:
            m = kalshi(f"/markets/{t}")["market"]
        except Exception:  # noqa: BLE001
            continue
        if m.get("result") in ("yes", "no"):
            cache[t] = {"result": m["result"], "expiration_value": m.get("expiration_value")}
        time.sleep(0.3)
    save_json("results.json", cache)
    return cache


def boot_roi(rows: list[dict], seed: int = SEED, n: int = NBOOT):
    ev = {}
    for r in rows:
        a = ev.setdefault(r["event"], [0.0, 0.0])
        a[0] += r["pnl"]
        a[1] += r["cost"]
    P = np.array([v[0] for v in ev.values()])
    C = np.array([v[1] for v in ev.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(P), size=(n, len(P)))
    b = P[idx].sum(1) / C[idx].sum(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float((b <= 0).mean())


def report_rows(picks: list[dict], res: dict, snaps: list[dict]) -> dict:
    """Shared by both loggers: ROI (event-clustered CI), calibration (model vs mid Brier), CLV vs last pre-close mid."""
    last_mid = {}
    for s in snaps:
        if s.get("yes_bid") is not None and s.get("yes_ask") is not None and 0 < s["yes_bid"] < s["yes_ask"] < 1 \
                and s["ts"] < s["close_ts"]:
            if s["ticker"] not in last_mid or s["ts"] >= last_mid[s["ticker"]][0]:
                last_mid[s["ticker"]] = (s["ts"], (s["yes_bid"] + s["yes_ask"]) / 2)
    out = {}
    for dname in sorted({p["decision"] for p in picks}) + ["ALL"]:
        rows = [p for p in picks if (dname == "ALL" or p["decision"] == dname) and p["ticker"] in res]
        cal = [p for p in rows if p.get("p_yes") is not None]
        bets = [dict(p) for p in rows if p.get("side")]
        for b in bets:
            y = res[b["ticker"]]["result"]
            b["pnl"] = (1.0 if b["side"] == y else 0.0) - b["cost"]
            lm = last_mid.get(b["ticker"])
            b["clv"] = None if lm is None else ((lm[1] if b["side"] == "yes" else 1 - lm[1]) - b["price"])
        d = {"decision_rows_settled": len(rows), "bets_settled": len(bets),
             "events_with_bets": len({b["event"] for b in bets})}
        if bets:
            cost = sum(b["cost"] for b in bets)
            d["roi"] = sum(b["pnl"] for b in bets) / cost
            if d["events_with_bets"] >= 5:
                lo, hi, p = boot_roi(bets)
                d.update({"roi_lo": lo, "roi_hi": hi, "p_one_sided": p})
            cl = [b["clv"] for b in bets if b["clv"] is not None]
            if cl:
                d.update({"clv_mean": float(np.mean(cl)), "clv_pos_frac": float(np.mean([c > 0 for c in cl])),
                          "clv_n": len(cl)})
        if cal:
            yv = np.array([1.0 if res[p["ticker"]]["result"] == "yes" else 0.0 for p in cal])
            pm = np.array([p["p_yes"] for p in cal])
            mids = np.array([(p["yes_bid"] + p["yes_ask"]) / 2 if p.get("yes_bid") is not None and p.get("yes_ask") is not None
                             else np.nan for p in cal])
            ok = ~np.isnan(mids)
            d["brier_model"] = float(np.mean((pm[ok] - yv[ok]) ** 2)) if ok.any() else None
            d["brier_mid"] = float(np.mean((mids[ok] - yv[ok]) ** 2)) if ok.any() else None
            d["brier_n"] = int(ok.sum())
            bins = []
            for lo_, hi_ in [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.0001)]:
                k = (pm >= lo_) & (pm < hi_)
                if k.any():
                    bins.append({"bin": f"{lo_:.1f}-{min(hi_, 1):.1f}", "n": int(k.sum()),
                                 "mean_p": float(pm[k].mean()), "freq_yes": float(yv[k].mean())})
            d["calibration"] = bins
        out[dname] = d
    return out


def report() -> int:
    picks = read("picks.jsonl")
    bets = [p for p in picks if p.get("side")]
    print(f"decision rows {len(picks)}, bets {len(bets)}, events with bets {len({b['event'] for b in bets})}")
    sc = read("scores.jsonl")
    rt = [r for r in sc if r.get("source") == "rt" and r.get("score") is not None and r.get("score_calc") is not None]
    if rt:
        mis = sum(1 for r in rt if int(r["score"]) != int(r["score_calc"]))
        print(f"RT rounding check: displayed != round-half-up(100*liked/n) on {mis} of {len(rt)} fetches")
    if not picks:
        return 0
    res = results_for({p["ticker"] for p in picks})
    out = report_rows(picks, res, read("snapshots.jsonl"))
    print(json.dumps(out, indent=1))
    return 0


def cmd_map() -> int:
    st = load_json("state.json", {"events": {}})
    for ev, r in sorted(st["events"].items()):
        print(f"{ev:16s} {r.get('source')} {str(r.get('ref')):45s} query={r.get('query')!r} page={r.get('page_title')!r}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tick"
    if cmd == "tick":
        o = tick()
        # wrapper logs only non-empty output: print only errors, picks, or an hourly heartbeat
        if o["errors"] or o["picks_logged"] or datetime.now(timezone.utc).minute < 15:
            print(json.dumps(o))
    elif cmd == "tick-verbose":
        print(json.dumps(tick()))
    elif cmd == "report":
        report()
    elif cmd == "map":
        cmd_map()
    else:
        print(__doc__)
