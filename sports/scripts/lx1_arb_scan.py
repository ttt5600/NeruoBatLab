"""LX1: Kalshi-internal logical-dependency arbitrage scanner. READ-ONLY (public GETs, no auth, no orders).

    python scripts/lx1_arb_scan.py scan            # full scan -> results/lx1_logical_arb/{violations,scans}.jsonl
    python scripts/lx1_arb_scan.py persist [--hours 24]   # 1-min candles for logged violations -> persistence.csv
    python scripts/lx1_arb_scan.py report          # distinct violations vs the pre-registered bar

Pre-registration: sports/notes/lx1_logical_arb.md. A VIOLATION is a leg set bought at executable asks (walking the
book) whose minimum payoff over every logically possible outcome beats cost + taker fees by >= MIN_NET (2c) per set
at a size of >= MIN_SIZE (10) contracts on every leg.

Constraint families (all deterministic):
  ladder_strike   one event, strike_type greater/greater_or_equal/less/less_or_equal, same proposition (title with
                  digits stripped): P(X>k) non-increasing in k. Buy YES(looser) + NO(tighter); min payoff $1.
  ladder_subtitle one event, sub-titles "Before <date>", "before round N", "N+", "at least N": cumulative.
  ladder_date     one series, several events whose market titles differ only in a "before/by <date>" deadline and
                  share the yes_sub_title: P(by later) >= P(by earlier).
  excl_no         mutually exclusive event: NO on any m legs pays >= m-1.
  yes_dutch       mutually exclusive event: YES on every leg pays 1 ONLY IF exhaustive -> reported, never counted.
  fed             KXFED-<mtg> (upper bound > k) vs KXFEDDECISION-<mtg> for the NEXT meeting only, using the current
                  upper bound from FRED DFEDTARU: Above(R)=H25|H26, Above(R+.25)=H26, Above(R-.25)=not(C25|C26),
                  Above(R-.5)=not C26.
Taker fee per leg = ceil_cent(0.07 * fee_multiplier(series) * sum_levels c*p*(1-p)).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

K = "https://api.elections.kalshi.com/trade-api/v2"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lx1_logical_arb"
VIOL = OUT / "violations.jsonl"
CANDS = OUT / "candidates.jsonl"
SCANS = OUT / "scans.jsonl"
MIN_NET = 0.02
MIN_SIZE = 10
SCREEN_NET = 0.0          # top-of-book screen (1 contract, unrounded fee) before fetching books
S = requests.Session()
S.headers["User-Agent"] = "lx1-readonly-scanner"


MIN_GAP = 0.25          # <= 4 req/s: the live bots share this IP; scan 2 + candle retries drew 429s on 11-12
_last = [0.0]


def get(path, params=None, tries=6):
    for i in range(tries):
        try:
            wait = _last[0] + MIN_GAP - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
            r = S.get(K + path, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(1.5 * (i + 1)); continue
            if 400 <= r.status_code < 500:
                return None                 # e.g. candle window before the market existed: not transient, no retry
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def paged(path, key, params):
    out, cur = [], None
    while True:
        p = dict(params)
        if cur:
            p["cursor"] = cur
        r = get(path, p)
        for back in (10, 30, 60):           # scan 3 (11-12) died on a None page (429s exhausted): back off, retry
            if r is not None:
                break
            time.sleep(back)
            r = get(path, p)
        if r is None:
            raise RuntimeError(f"{path} page failed after retries (cursor {cur})")
        out += r.get(key) or []
        cur = r.get("cursor")
        if not cur or not r.get(key):
            return out


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def series_of(m):
    return m["event_ticker"].split("-")[0]


# ---------------------------------------------------------------- fees and books
FEE_MULT: dict[str, float] = {}


def load_fees():
    for s in get("/series", {"limit": 5000}).get("series", []):
        FEE_MULT[s["ticker"]] = float(s.get("fee_multiplier") if s.get("fee_multiplier") is not None else 1)


def fee_rate(series):
    return 0.07 * FEE_MULT.get(series, 1.0)


def top(m, side):
    """(ask price, size at ask) for buying `side` from the /markets snapshot."""
    if side == "yes":
        return f(m.get("yes_ask_dollars")), f(m.get("yes_ask_size_fp"))
    return 1 - f(m.get("yes_bid_dollars")), f(m.get("yes_bid_size_fp"))


BOOKS: dict[str, dict] = {}


def ask_levels(ticker, side):
    """Ascending ask levels [(price, size)] for buying `side`: YES asks = 1 - NO bids, NO asks = 1 - YES bids."""
    if ticker not in BOOKS:
        BOOKS[ticker] = (get(f"/markets/{ticker}/orderbook") or {}).get("orderbook_fp") or {}
    ob = BOOKS[ticker]
    opp = ob.get("no_dollars" if side == "yes" else "yes_dollars") or []
    lv = [(round(1 - f(p), 4), f(q)) for p, q in opp if f(q) > 0]
    return sorted(lv)


def leg_cost(levels, n, rate):
    """Cost (incl. fee rounded up to the cent) of buying n contracts walking `levels`; None if not enough depth."""
    left, cash, fee = n, 0.0, 0.0
    for p, q in levels:
        take = min(left, q)
        cash += take * p
        fee += rate * take * p * (1 - p)
        left -= take
        if left <= 1e-9:
            return cash + math.ceil(round(fee * 100, 6)) / 100
    return None


def evaluate(legs, minpay):
    """legs: [(market, side)]. Walk books; return dict with net/set at MIN_SIZE and the max size with net >= MIN_NET."""
    lvls = [ask_levels(m["ticker"], s) for m, s in legs]
    rates = [fee_rate(series_of(m)) for m, _ in legs]
    depth = min(sum(q for _, q in lv) for lv in lvls) if lvls else 0

    def net_at(n):
        tot = 0.0
        for lv, r in zip(lvls, rates):
            c = leg_cost(lv, n, r)
            if c is None:
                return None
            tot += c
        return minpay - tot / n

    n10 = net_at(MIN_SIZE) if depth >= MIN_SIZE else None
    # largest size on a grid with net >= MIN_NET (no early break: cent-rounded fees make tiny n look worse than n=10)
    best_n, best_net = 0, None
    n = 1
    while n <= depth:
        v = net_at(n)
        if v is None:
            break
        if v >= MIN_NET:
            best_n, best_net = n, v
        n = n + 1 if n < 20 else int(n * 1.15) + 1
    return {"net_at_min_size": None if n10 is None else round(n10, 4), "depth_min_leg": depth,
            "max_size_at_min_net": best_n, "net_at_max_size": None if best_net is None else round(best_net, 4),
            "legs_book": [{"ticker": m["ticker"], "side": s, "asks": lv[:5]} for (m, s), lv in zip(legs, lvls)]}


def screen(legs, minpay):
    tot = 0.0
    for m, s in legs:
        p, q = top(m, s)
        if p <= 0 or p >= 1 or q <= 0:
            return None
        tot += p + fee_rate(series_of(m)) * p * (1 - p)
    return minpay - tot


# ---------------------------------------------------------------- constraint generators
NUM = re.compile(r"[-+]?\$?\d[\d,]*\.?\d*%?")
MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
DATE_PH = re.compile(rf"\b(before|by)\s+(the\s+end\s+of\s+|end\s+of\s+)?((?:{MONTHS})[a-z]*\.?\s+\d{{1,2}},?\s+\d{{4}}|(?:{MONTHS})[a-z]*\.?\s+\d{{4}}|\d{{4}})", re.I)
BAD_WINDOW = re.compile(r"\bbetween\b|\bafter (?!issuance)|\bduring\b|\bin (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)


def parse_deadline(m):
    """(datetime value, title with deadline removed) for a cumulative 'before/by <date>' proposition, else None."""
    t = m.get("title") or ""
    hits = list(DATE_PH.finditer(t))
    if len(hits) != 1 or BAD_WINDOW.search(t):
        return None
    h = hits[0]
    txt = h.group(3).replace(",", "").replace(".", "")
    d = None
    for fmt in ("%b %d %Y", "%B %d %Y", "%b %Y", "%B %Y", "%Y"):
        try:
            d = datetime.strptime(txt, fmt); break
        except ValueError:
            continue
    if d is None:
        return None
    end_of = bool(h.group(2))
    if fmt == "%Y":                       # "before 2027" = before Jan 1 2027; "by end of 2027" = before Jan 1 2028
        d = d.replace(year=d.year + 1) if (end_of or h.group(1).lower() == "by") else d
    elif fmt in ("%b %Y", "%B %Y"):       # "before Mar 2027" = before Mar 1; "by (end of) Mar 2027" = before Apr 1
        if end_of or h.group(1).lower() == "by":
            d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    elif h.group(1).lower() == "by":      # "by Dec 31 2026" = before Jan 1 2027
        d = d + timedelta(days=1)
    return d, (t[:h.start()] + t[h.end():]).strip().lower()


SUB_RULES = [
    (re.compile(r"^before\s+(.+)$", re.I), "date"),
    (re.compile(r"before round (\d+)", re.I), "inc"),
    (re.compile(r"^(?:at least|over)\s+(\d+(?:\.\d+)?)", re.I), "dec"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:\+|or more|or above)", re.I), "dec"),
]


def sub_value(st):
    st = re.sub(r"(?<=\d),(?=\d{3})", "", (st or "").replace("$", ""))   # "$2,769.99" -> "2769.99" (scan-1 bug)
    for rx, kind in SUB_RULES:
        mm = rx.search(st)
        if not mm:
            continue
        if kind == "date":
            for fmt in ("%b %d %Y", "%B %d %Y", "%Y"):
                try:
                    return ("date", datetime.strptime(mm.group(1).replace(",", "").strip(), fmt).timestamp())
                except ValueError:
                    pass
            return None
        v = float(mm.group(1))
        return (kind, v if kind == "inc" else -v)
    return None


RNUM = re.compile(r"\d[\d,]*\.?\d*")


def one_number_apart(a, b):
    """True when the two markets' rules_primary differ only in ONE number (the strike). Rejects two-dimension pairs
    such as USCLIMATE-2025 (<=4909.9 by 2025) vs USCLIMATE-2030 (<=3317.5 by 2030)."""
    ra, rb = a.get("rules_primary") or "", b.get("rules_primary") or ""
    if RNUM.sub("#", ra) != RNUM.sub("#", rb):
        return False
    na, nb = RNUM.findall(ra), RNUM.findall(rb)
    return len(na) == len(nb) and sum(x != y for x, y in zip(na, nb)) == 1


def ladder_pairs(group, check_rules=False):
    """group: [(value, market)] where P(YES) is non-decreasing in value. Yield (legs, minpay) for screening."""
    g = sorted(group, key=lambda x: x[0])
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            if g[j][0] > g[i][0]:
                if check_rules and not one_number_apart(g[j][1], g[i][1]):
                    continue
                yield [(g[j][1], "yes"), (g[i][1], "no")], 1.0


def gen_constraints(ms, events, fed_upper):
    by_ev = defaultdict(list)
    for m in ms:
        by_ev[m["event_ticker"]].append(m)
    # 1. strike ladders
    grp = defaultdict(list)
    for m in ms:
        st, k = m.get("strike_type"), m.get("floor_strike") if m.get("strike_type") in ("greater", "greater_or_equal") else m.get("cap_strike")
        # guard: Kalshi tags "exactly k" buckets as strike_type less with floor == cap (KXSTARSHIPSPACE-26, 11-12)
        lo, hi = m.get("floor_strike"), m.get("cap_strike")
        one_sided = (hi is None) if (st or "").startswith("greater") else (lo is None)
        if re.search(r"\bexactly\b", (m.get("title") or "") + (m.get("rules_primary") or ""), re.I):
            one_sided = False
        if st in ("greater", "greater_or_equal", "less", "less_or_equal") and k is not None and one_sided:
            key = (m["event_ticker"], st, NUM.sub("#", m.get("title") or ""), NUM.sub("#", m.get("yes_sub_title") or ""))
            grp[key].append((-float(k) if st.startswith("greater") else float(k), m))
    for key, g in grp.items():
        if len(g) > 1:
            for legs, mp in ladder_pairs(g, check_rules=True):
                yield "ladder_strike", key[0], legs, mp
    # 2. sub-title ladders inside an event
    grp = defaultdict(list)
    for m in ms:
        if m.get("strike_type") in ("greater", "greater_or_equal", "less", "less_or_equal", "between"):
            continue                       # numeric strikes are handled by family 1 from floor/cap_strike
        v = sub_value(m.get("yes_sub_title"))
        if v:
            # sub-title with digits stripped must match: "India (2+ times)" vs "Club Wingstop (3+ times)" are unrelated
            grp[(m["event_ticker"], v[0], NUM.sub("#", m.get("title") or ""), NUM.sub("#", m.get("yes_sub_title") or ""))].append((v[1], m))
    for key, g in grp.items():
        if len(g) > 1:
            for legs, mp in ladder_pairs(g, check_rules=True):
                yield "ladder_subtitle", key[0], legs, mp
    # 3. cross-event date ladders in a series
    grp = defaultdict(list)
    for m in ms:
        pdl = parse_deadline(m)
        if pdl:
            grp[(series_of(m), pdl[1], (m.get("yes_sub_title") or "").lower(), m.get("floor_strike"))].append((pdl[0].timestamp(), m))
    for key, g in grp.items():
        if len({m["event_ticker"] for _, m in g}) > 1:
            for legs, mp in ladder_pairs(g):
                if legs[0][0]["event_ticker"] != legs[1][0]["event_ticker"]:
                    yield "ladder_date", key[0], legs, mp
    # 4/5. mutually exclusive events
    for e in events:
        if not e.get("mutually_exclusive"):
            continue
        legs = by_ev.get(e["event_ticker"], [])
        if len(legs) < 2:
            continue
        # NO subset: keep legs whose YES bid exceeds the NO-leg fee (each adds 1 - no_ask - fee > 0 to the set value)
        sub = [m for m in legs if f(m.get("yes_bid_dollars")) > 0 and f(m.get("yes_bid_size_fp")) > 0]
        sub = [m for m in sub if f(m["yes_bid_dollars"]) - fee_rate(series_of(m)) * f(m["yes_bid_dollars"]) * (1 - f(m["yes_bid_dollars"])) > 0]
        if len(sub) >= 2:
            yield "excl_no", e["event_ticker"], [(m, "no") for m in sub], float(len(sub) - 1)
        if all(f(m.get("yes_ask_dollars")) > 0 for m in legs):
            yield "yes_dutch", e["event_ticker"], [(m, "yes") for m in legs], 1.0
    # 6. Fed: next meeting only
    if fed_upper is not None:
        fed = defaultdict(dict)
        for m in ms:
            s = series_of(m)
            if s in ("KXFED", "KXFEDDECISION"):
                fed[m["event_ticker"].split("-")[1]][m["ticker"].split("-")[-1]] = m
        if fed:
            mtg = min(fed, key=lambda x: min(m["close_time"] for m in fed[x].values()))
            d = fed[mtg]
            R = fed_upper
            eq = [(R, ["H25", "H26"], False), (R + .25, ["H26"], False), (R - .25, ["C25", "C26"], True), (R - .5, ["C26"], True)]
            for k, dec, negated in eq:
                a = d.get(f"T{k:.2f}")
                ds = [d.get(x) for x in dec]
                if a is None or any(x is None for x in ds):
                    continue
                # A == union(dec) (or A == not union(dec)). Direction 1: YES A + NO each dec (or YES each dec if negated)
                if not negated:
                    yield "fed", f"KXFED-{mtg}", [(a, "no")] + [(x, "yes") for x in ds], 1.0
                    yield "fed", f"KXFED-{mtg}", [(a, "yes")] + [(x, "no") for x in ds], float(len(ds))
                else:
                    yield "fed", f"KXFED-{mtg}", [(a, "yes")] + [(x, "yes") for x in ds], 1.0
                    yield "fed", f"KXFED-{mtg}", [(a, "no")] + [(x, "no") for x in ds], float(len(ds))


def fed_upper_bound():
    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                         params={"id": "DFEDTARU", "cosd": (datetime.now(timezone.utc) - timedelta(days=20)).date().isoformat()}, timeout=30)
        rows = [l.split(",") for l in r.text.strip().splitlines()[1:] if l.split(",")[1] not in (".", "")]
        return float(rows[-1][1])
    except Exception:
        return None


# ---------------------------------------------------------------- commands
def scan():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    load_fees()
    ms = paged("/markets", "markets", {"status": "open", "limit": 1000, "mve_filter": "exclude"})
    ms = [m for m in ms if m.get("status") == "active"]
    events = paged("/events", "events", {"status": "open", "limit": 200})
    R = fed_upper_bound()
    counts, cands = defaultdict(int), []
    for kind, ev, legs, minpay in gen_constraints(ms, events, R):
        counts[kind] += 1
        v = screen(legs, minpay)
        if v is not None and v > SCREEN_NET:
            cands.append((v, kind, ev, legs, minpay))
    # counted families first; YES-dutch (unverified exhaustiveness, never counted) capped at 20 book walks per scan
    cands.sort(key=lambda x: (x[1] == "yes_dutch", -x[0]))
    nd = [c for c in cands if c[1] != "yes_dutch"]
    cands = nd + [c for c in cands if c[1] == "yes_dutch"][:20]
    by_kind = defaultdict(int)
    for c in cands:
        by_kind[c[1]] += 1
    print("screen candidates by kind:", dict(by_kind), flush=True)
    n_viol = 0
    with open(CANDS, "a") as fc, open(VIOL, "a") as fv_real, open(OUT / "yes_dutch_unverified.jsonl", "a") as fv_dutch:
        for v, kind, ev, legs, minpay in cands[:400]:
            ev_res = evaluate(legs, minpay)
            rec = {"ts": ts, "kind": kind, "event": ev, "minpay": minpay, "screen_net_top": round(v, 4),
                   "legs": [{"ticker": m["ticker"], "side": s, "title": m.get("title"), "sub": m.get("yes_sub_title"),
                             "top_ask": top(m, s)[0], "top_size": top(m, s)[1], "close": m.get("close_time")} for m, s in legs],
                   **ev_res}
            rec["violation"] = bool(kind != "yes_dutch" and ev_res["net_at_min_size"] is not None and ev_res["net_at_min_size"] >= MIN_NET)
            rec["key"] = kind + "|" + "|".join(sorted(f"{m['ticker']}:{s}" for m, s in legs))
            fc.write(json.dumps(rec) + "\n")
            if rec["violation"]:
                fv_real.write(json.dumps(rec) + "\n")
                n_viol += 1
            elif kind == "yes_dutch" and (ev_res["net_at_min_size"] or -1) >= MIN_NET:
                fv_dutch.write(json.dumps(rec) + "\n")     # named-candidate fields are not exhaustive: never counted
                n_viol += rec["violation"]
    summ = {"ts": ts, "secs": round(time.time() - t0), "markets": len(ms), "events": len(events), "fed_upper": R,
            "constraints": dict(counts), "screen_candidates": len(cands), "violations": n_viol}
    with open(SCANS, "a") as fs:
        fs.write(json.dumps(summ) + "\n")
    print(json.dumps(summ))
    for v, kind, ev, legs, minpay in cands[:15]:
        print(f"  {kind:15s} top-net {v:+.3f}  " + " + ".join(f"{s.upper()} {m['ticker']}@{top(m, s)[0]:.2f}x{top(m, s)[1]:.0f}" for m, s in legs)[:220])


def candles(series, ticker, start, end):
    cs, s, end = [], int(start), int(end)
    while s < end:                      # 1-min candles: the API rejects (400) a 7-day range; 48 h chunks work
        e = min(int(end), s + 48 * 3600)
        cs += (get(f"/series/{series}/markets/{ticker}/candlesticks",
                   {"start_ts": s, "end_ts": e, "period_interval": 1}) or {}).get("candlesticks", [])
        s = e
    out = {}
    for c in cs:
        ya, yb = c.get("yes_ask") or {}, c.get("yes_bid") or {}
        a = f(ya.get("close_dollars") or ya.get("close"))
        b = f(yb.get("close_dollars") or yb.get("close"))
        if a > 1.5:   # cents
            a, b = a / 100, b / 100
        out[int(c["end_period_ts"])] = (a, b)
    return out


def timeline(legs, minpay, series_c, end):
    """legs: [(ticker, side)]. Forward-fill quote-change candles onto a 1-min grid (from when every leg has a quote);
    return (minutes, flags) where flag = top-of-book net (1 contract, unrounded fee) >= MIN_NET. No depth in candles."""
    if not series_c or any(not series_c.get(t) for t, _ in legs):
        return 0, []
    t0 = max(min(series_c[t]) for t, _ in legs)
    mins = list(range(t0, int(end) // 60 * 60 + 1, 60))
    filled = {}
    for tk, _ in legs:
        c = series_c[tk]
        ks, last, out, i = sorted(c), None, [], 0
        for t in mins:
            while i < len(ks) and ks[i] <= t:
                last = c[ks[i]]; i += 1
            out.append(last)
        filled[tk] = out
    flags = []
    for j in range(len(mins)):
        tot = 0.0
        for tk, side in legs:
            a, b = filled[tk][j]
            p = a if side == "yes" else 1 - b
            if p <= 0 or p >= 1:
                tot = 99; break
            tot += p + fee_rate(tk.split("-")[0]) * p * (1 - p)
        flags.append(minpay - tot >= MIN_NET)
    return len(mins), flags


def runs_of(flags):
    runs, cur = [], 0
    for x in flags:
        if x:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    return runs


def persist(hours):
    """Minute candles (top of book only, no depth) for each distinct logged violation: minutes in violation, run lengths."""
    import csv
    recs = [json.loads(l) for l in open(VIOL)] if VIOL.exists() else []
    seen, rows = set(), []
    end = time.time()
    start = end - hours * 3600
    load_fees()
    for r in recs:
        if r["key"] in seen or not r["violation"]:
            continue
        seen.add(r["key"])
        legs = [(lg["ticker"], lg["side"]) for lg in r["legs"]]
        series_c = {t: candles(t.split("-")[0], t, start, end) for t, _ in legs}
        n, flags = timeline(legs, r["minpay"], series_c, end)
        runs = runs_of(flags)
        rows.append({"key": r["key"], "kind": r["kind"], "minutes_with_candles": n, "minutes_in_violation": sum(flags),
                     "n_runs": len(runs), "median_run_min": sorted(runs)[len(runs) // 2] if runs else 0,
                     "max_run_min": max(runs) if runs else 0})
    with open(OUT / "persistence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["key"])
        w.writeheader(); w.writerows(rows)
    for x in rows:
        print(x)


def history(hours, kinds):
    """Backward look: for every CURRENT constraint of the given kinds, replay `hours` of 1-min candles (top of book,
    no depth -> an UPPER bound on violations) and count distinct leg sets that were ever >= MIN_NET."""
    import csv
    load_fees()
    ms = [m for m in paged("/markets", "markets", {"status": "open", "limit": 1000, "mve_filter": "exclude"})
          if m.get("status") == "active"]
    R = fed_upper_bound()
    cons = [(k, ev, legs, mp) for k, ev, legs, mp in gen_constraints(ms, [], R) if k in kinds]
    tick = sorted({m["ticker"] for _, _, legs, _ in cons for m, _ in legs})
    print(f"history: {len(cons)} constraints over {len(tick)} markets, {hours} h", flush=True)
    end = time.time()
    cc = {t: candles(t.split("-")[0], t, end - hours * 3600, end) for t in tick}
    rows = []
    for k, ev, legs, mp in cons:
        lg = [(m["ticker"], s) for m, s in legs]
        n, flags = timeline(lg, mp, cc, end)
        runs = runs_of(flags)
        if runs:
            rows.append({"kind": k, "legs": " + ".join(f"{s.upper()} {t}" for t, s in lg), "minutes": n,
                         "minutes_in_violation": sum(flags), "n_runs": len(runs),
                         "median_run_min": sorted(runs)[len(runs) // 2], "max_run_min": max(runs)})
    with open(OUT / "history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "legs", "minutes", "minutes_in_violation", "n_runs", "median_run_min", "max_run_min"])
        w.writeheader(); w.writerows(rows)
    print(f"distinct leg sets ever in top-of-book violation (>= {MIN_NET:.2f} net, 1 contract): {len(rows)}")
    for x in sorted(rows, key=lambda r: -r["minutes_in_violation"])[:25]:
        print(x)


def report():
    recs = [json.loads(l) for l in open(VIOL)] if VIOL.exists() else []
    v = [r for r in recs if r["violation"]]
    keys = defaultdict(list)
    for r in v:
        keys[r["key"]].append(r)
    first = min((r["ts"] for r in recs), default=None)
    print(f"scans: {sum(1 for _ in open(SCANS)) if SCANS.exists() else 0}; first violation-log row {first}")
    print(f"violation rows {len(v)}; DISTINCT leg sets {len(keys)} (bar: >= 5 in 7 days)")
    for k, rs in keys.items():
        r0 = rs[-1]   # latest sighting (scan-2 rows predate the size-search fix)
        print(f"  {r0['kind']:14s} seen {len(rs)}x {rs[0]['ts']}..{r0['ts']}  net@10 {r0['net_at_min_size']:+.3f}  "
              f"max n {r0['max_size_at_min_net']}  {k[:150]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "persist", "history", "report"])
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--kinds", default="ladder_date,ladder_subtitle,fed")
    a = ap.parse_args()
    {"scan": scan, "persist": lambda: persist(a.hours), "report": report,
     "history": lambda: history(a.hours, set(a.kinds.split(",")))}[a.cmd]()
