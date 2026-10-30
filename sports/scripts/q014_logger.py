"""Q014 forward paper logger: Kalshi Truth Social post-count brackets (KXTRUTHSOCIAL weekly, KXTRUTHSOCIALD daily)
vs the running count. NO ORDERS ARE EVER SENT. Public GET endpoints only, no API key.

    python scripts/q014_logger.py tick      # launchd every 15 min: books + post archive + frozen paper picks
    python scripts/q014_logger.py report    # settle picks on Kalshi results -> ROI (CI), calibration, CLV
    python scripts/q014_logger.py model     # print the frozen calibration model (or calibration progress)

Pre-registration: notes/q014_truth_social.md (frozen 2026-10-29 before the first tick).

Count source = the settlement source itself: Roll Call Factba.se's public JSON
(rollcall.com/wp-json/factbase/v1/twitter?platform=truth social, the endpoint its own archive page calls). Cross-check:
trumpstruth.org RSS (same post ids). truthsocial.com's own API is Cloudflare-blocked (403) and is NOT used or bypassed.

CALIBRATION (said plainly): the per-hour rate model is fitted ONCE on the posts dated in the logger's own first 7 days
[first_tick, first_tick + 7 d), then frozen in data/q014/model.json and never refitted. No picks are made during
calibration. The rate fit uses no Kalshi prices.

Files (data/q014/, our UTC clock = point-in-time stamp):
  snapshots.jsonl  every open market each tick: yes bid/ask (+sizes), volume, OI
  posts.jsonl      every Roll Call post the first time we see it (first_seen) + its post timestamp + deleted flag
  post_updates.jsonl  deleted-flag flips and posts that vanished from a full sweep (first time seen)
  counts.jsonl     per open event per tick: window, running count (RC, excl. deleted), incl. deleted, RSS cross-check
  picks.jsonl      frozen paper rule, one row per (slot, ticker)
"""
from __future__ import annotations

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests
from scipy.stats import nbinom, poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q013_logger as C  # noqa: E402  (shared: kalshi GET with backoff, jsonl io, fee/decide, report_rows)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "q014"
C.OUT = OUT          # route the shared jsonl helpers to data/q014
RC = "https://rollcall.com/wp-json/factbase/v1/twitter"
TT_RSS = "https://www.trumpstruth.org/feed"
SERIES = {"KXTRUTHSOCIAL": "W", "KXTRUTHSOCIALD": "D"}
NY = ZoneInfo("America/New_York")

# ---------------------------------------------------------------- FROZEN rule (notes/q014_truth_social.md section 4)
CALIB_S = 7 * 86400
SLOTS = {"W": [12], "D": [12, 18]}   # ET hour of each decision; weekly: at 12:00 ET on every day of the window
WINDOW = 3600
EDGE = C.EDGE                       # 5c after fee (same decide() as Q013)
FULL_SWEEP_S = 3600                 # re-read the whole needed range hourly (deletions); otherwise incremental
MAX_PAGES = 20                      # 50 posts per page


def rc_page(page: int) -> dict:
    r = C.web(RC, {"platform": "truth social", "sort": "date", "sort_order": "desc", "page": page, "format": "json"})
    if r.status_code != 200:
        raise RuntimeError(f"rollcall page {page} -> {r.status_code}")
    return r.json()


def post_ts(s: str) -> int:
    return int(datetime.fromisoformat(s).timestamp())


def sweep(since_ts: int, known: dict, full: bool) -> tuple[list[dict], set, int]:
    """Read RC newest-first until posts are older than since_ts (full) or until a page holds only known posts."""
    seen, rows, pages = set(), [], 0
    for page in range(1, MAX_PAGES + 1):
        d = rc_page(page)
        pages += 1
        data = d.get("data") or []
        for x in data:
            pid = str(x.get("id"))
            seen.add(pid)
            soc = x.get("social") or {}
            rows.append({"id": pid, "date": x.get("date"), "post_ts": post_ts(x["date"]),
                         "deleted": bool(x.get("deleted_flag")), "deleted_date": soc.get("deleted_date"),
                         "repost": bool(soc.get("repost_flag")), "quote": bool(soc.get("quote_flag")),
                         "reply_to": soc.get("in_reply_to_screen_name"), "media_type": x.get("media_type")})
        if not data:
            break
        oldest = min(post_ts(x["date"]) for x in data)
        if oldest < since_ts:
            break
        if not full and all(str(x.get("id")) in known for x in data):
            break
        time.sleep(1)
    return rows, seen, pages


def rss_ids() -> dict[str, int]:
    r = C.web(TT_RSS)
    root = ET.fromstring(r.content)
    ns = {"truth": "https://truthsocial.com/ns"}
    out = {}
    for it in root.iter("item"):
        oid = it.findtext("truth:originalId", default=None, namespaces=ns)
        pd_ = it.findtext("pubDate")
        if oid and pd_:
            out[oid] = int(datetime.strptime(pd_, "%a, %d %b %Y %H:%M:%S %z").timestamp())
    return out


# ---------------------------------------------------------------- windows
_WIN = re.compile(r"begins at 12:00:00 AM ET on (\w{3} \d{1,2}, \d{4}) and ends at 11:59:59 PM ET on (\w{3} \d{1,2}, \d{4})")


def window(m: dict) -> tuple[int, int] | None:
    g = _WIN.search(m.get("rules_secondary") or "")
    if not g:
        return None
    d0 = datetime.strptime(g.group(1), "%b %d, %Y").replace(tzinfo=NY)
    d1 = datetime.strptime(g.group(2), "%b %d, %Y").replace(tzinfo=NY) + timedelta(days=1)
    return int(d0.timestamp()), int(d1.timestamp())    # [ws, we): we = next ET midnight


def slots_for(kind: str, ws: int, we: int) -> list[tuple[str, int]]:
    out = []
    day = datetime.fromtimestamp(ws, NY).date()
    while True:
        base = datetime(day.year, day.month, day.day, tzinfo=NY)
        if int(base.timestamp()) >= we:
            break
        for h in SLOTS[kind]:
            t = int(base.replace(hour=h).timestamp())
            if ws <= t < we:
                out.append((f"{kind}_{day.isoformat()}_{h:02d}", t))
        day += timedelta(days=1)
    return out


# ---------------------------------------------------------------- FROZEN model
def fit(posts: list[dict], t0: int) -> dict:
    """Hour-of-day (ET) rate profile + day-level NB dispersion from the 7 x 24 h blocks after t0."""
    t1 = t0 + CALIB_S
    ps = [p for p in posts if t0 <= p["post_ts"] < t1 and not p["deleted"]]
    hours = np.zeros(24)
    nh = np.zeros(24)
    for h in range(7 * 24):    # how many times each ET hour-of-day occurs in the block (DST-aware)
        nh[datetime.fromtimestamp(t0 + h * 3600 + 1800, NY).hour] += 1
    for p in ps:
        hours[datetime.fromtimestamp(p["post_ts"], NY).hour] += 1
    rate = hours / np.maximum(nh, 1)
    days = np.array([sum(1 for p in ps if t0 + i * 86400 <= p["post_ts"] < t0 + (i + 1) * 86400) for i in range(7)])
    mu, s2 = float(days.mean()), float(days.var(ddof=1))
    k = mu * mu / (s2 - mu) if s2 > mu else None
    return {"t0": t0, "t1": t1, "fitted_at": C.now_ts(), "n_posts": len(ps), "rate_per_hour_et": rate.tolist(),
            "daily_counts": days.tolist(), "daily_mean": mu, "daily_var": s2, "nb_k": k}


def remaining_dist(model: dict, t: int, we: int):
    """Remaining posts in [t, we): sum over ET days of NB(mu_j, k) moment-matched to one NB (Poisson if k is None)."""
    rate = model["rate_per_hour_et"]
    k = model["nb_k"]
    per_day: dict[str, float] = {}
    cur = t
    while cur < we:
        dt = datetime.fromtimestamp(cur, NY)
        nxt = min(we, int((dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)).timestamp()))
        key = dt.date().isoformat()
        per_day[key] = per_day.get(key, 0.0) + rate[dt.hour] * (nxt - cur) / 3600
        cur = nxt
    M = sum(per_day.values())
    V = sum(mu + (mu * mu / k if k else 0.0) for mu in per_day.values())
    return M, V


def p_yes_count(count: int, M: float, V: float, pred, nb: bool = True) -> float:
    hi = int(M + 12 * np.sqrt(max(V, 1.0)) + 50)
    r = np.arange(0, hi + 1)
    if nb and V > M > 0:
        n_ = M * M / (V - M)
        pr = nbinom.pmf(r, n_, n_ / (n_ + M))
    elif M > 0:
        pr = poisson.pmf(r, M)
    else:
        pr = (r == 0).astype(float)
    return float(sum(p for x, p in zip(r, pr) if pred(count + x)))


# ---------------------------------------------------------------- tick
def tick() -> dict:
    ts = C.now_ts()
    st = C.load_json("state.json", {})
    st.setdefault("first_tick", ts)
    errors = []
    mk = []
    for s in SERIES:
        try:
            mk += [dict(x, _series=s) for x in C.open_markets(s)]
        except Exception as e:  # noqa: BLE001
            errors.append(f"{s}: {e}")
    C.append("snapshots.jsonl", [{"ts": ts, "series": m["_series"], "event": m["event_ticker"], "ticker": m["ticker"],
                                  "strike_type": m.get("strike_type"), "floor": C.num(m.get("floor_strike")),
                                  "cap": C.num(m.get("cap_strike")), "close_ts": C.pts(m["close_time"]),
                                  "yes_bid": C.num(m.get("yes_bid_dollars")), "yes_ask": C.num(m.get("yes_ask_dollars")),
                                  "bid_size": C.num(m.get("yes_bid_size_fp")), "ask_size": C.num(m.get("yes_ask_size_fp")),
                                  "volume": C.num(m.get("volume_fp")), "oi": C.num(m.get("open_interest_fp"))}
                                 for m in mk])
    events: dict[str, dict] = {}
    for m in mk:
        w = window(m)
        e = events.setdefault(m["event_ticker"], {"series": m["_series"], "kind": SERIES[m["_series"]], "win": w,
                                                  "markets": []})
        e["markets"].append(m)

    # ---- post archive
    known = {}
    for p in C.read("posts.jsonl"):
        known[p["id"]] = p
    for u in C.read("post_updates.jsonl"):
        if u["id"] in known and "deleted" in u:
            known[u["id"]]["deleted"] = u["deleted"]
    starts = [e["win"][0] for e in events.values() if e["win"]]
    if C.load_json("model.json", None) is None:
        starts.append(st["first_tick"])          # calibration week must be fully archived
    need = min(starts + [ts - 86400])
    full = ts - st.get("last_full", 0) >= FULL_SWEEP_S
    new_rows, upd, pages, rc_ok = [], [], 0, True
    try:
        rows, seen, pages = sweep(need - 3600, known, full)
        for r in rows:
            if r["id"] not in known:
                r["first_seen"] = ts
                new_rows.append(r)
                known[r["id"]] = r
            elif r["deleted"] != known[r["id"]]["deleted"]:
                upd.append({"ts": ts, "id": r["id"], "deleted": r["deleted"], "deleted_date": r["deleted_date"]})
                known[r["id"]]["deleted"] = r["deleted"]
        if full:
            st["last_full"] = ts
            lo = min((r["post_ts"] for r in rows), default=ts)
            vanished = st.setdefault("vanished", [])
            for pid in [v for v in vanished if v in seen]:      # page-boundary miss, not a removal
                vanished.remove(pid)
                upd.append({"ts": ts, "id": pid, "vanished": False})
            for pid, p in known.items():
                if lo <= p["post_ts"] and pid not in seen and pid not in vanished:
                    vanished.append(pid)
                    upd.append({"ts": ts, "id": pid, "vanished": True})
    except Exception as e:  # noqa: BLE001
        rc_ok = False
        errors.append(f"rollcall: {e}")
    C.append("posts.jsonl", new_rows)
    C.append("post_updates.jsonl", upd)
    vanished = set(st.get("vanished", []))
    try:
        rss = rss_ids()
    except Exception as e:  # noqa: BLE001
        rss = {}
        errors.append(f"rss: {e}")
    rss_lo = min(rss.values()) if rss else None

    # ---- running counts
    counts = {}
    crow = []
    for ev, e in events.items():
        if not e["win"]:
            errors.append(f"{ev}: window not parsed from rules")
            continue
        ws, we = e["win"]
        inwin = [p for p in known.values() if ws <= p["post_ts"] < min(we, ts + 1)]
        c = sum(1 for p in inwin if not p["deleted"] and p["id"] not in vanished)
        counts[ev] = c
        cov = rss_lo is not None and rss_lo <= ws
        crow.append({"ts": ts, "event": ev, "kind": e["kind"], "ws": ws, "we": we, "count": c,
                     "count_incl_deleted": len(inwin), "rc_ok": rc_ok,
                     "rss_count": sum(1 for t in rss.values() if ws <= t < we) if rss else None,
                     "rss_covers_window": cov,
                     "rc_count_since_rss_lo": sum(1 for p in inwin if rss_lo and p["post_ts"] >= rss_lo and not p["deleted"]) if rss else None})
    C.append("counts.jsonl", crow)

    # ---- calibration -> frozen model
    model = C.load_json("model.json", None)
    if model is None and ts >= st["first_tick"] + CALIB_S and rc_ok:
        model = fit(list(known.values()), st["first_tick"])
        C.save_json("model.json", model)   # written once; never overwritten (see the guard above)

    # ---- frozen paper picks
    picks = []
    if model is not None and rc_ok:
        done = {(p["slot"], p["ticker"]) for p in C.read("picks.jsonl")}
        for ev, e in events.items():
            if not e["win"] or ev not in counts:
                continue
            ws, we = e["win"]
            for slot, tau in slots_for(e["kind"], ws, we):
                if not (tau <= ts < tau + WINDOW):
                    continue
                M, V = remaining_dist(model, ts, we)
                for m in e["markets"]:
                    if (slot, m["ticker"]) in done:
                        continue
                    b, a = C.num(m.get("yes_bid_dollars")), C.num(m.get("yes_ask_dollars"))
                    base = {"ts": ts, "decision": e["kind"], "slot": slot, "series": e["series"], "event": ev,
                            "ticker": m["ticker"], "close_ts": C.pts(m["close_time"]), "tau": tau, "yes_bid": b,
                            "yes_ask": a, "strike_type": m.get("strike_type"), "floor": C.num(m.get("floor_strike")),
                            "cap": C.num(m.get("cap_strike")), "count": counts[ev], "rem_mean": M, "rem_var": V}
                    pred = C.yes_set(m)
                    if pred is None:
                        picks.append({**base, "side": None, "reason": "unknown strike"})
                        continue
                    p = p_yes_count(counts[ev], M, V, pred, nb=True)
                    pp = p_yes_count(counts[ev], M, V, pred, nb=False)
                    picks.append({**base, "p_yes": p, "p_yes_poisson": pp, **C.decide(p, b, a)})
    C.append("picks.jsonl", picks)
    C.save_json("state.json", st)
    return {"ts": C.iso(ts), "markets": len(mk), "events": len(events), "rc_pages": pages, "full_sweep": full,
            "new_posts": len(new_rows), "post_updates": len(upd), "counts": {k: v for k, v in counts.items()},
            "model": "frozen" if model else f"calibrating until {C.iso(st['first_tick'] + CALIB_S)}",
            "picks_logged": len(picks), "bets": sum(1 for p in picks if p.get("side")), "errors": errors}


def report() -> int:
    picks = C.read("picks.jsonl")
    bets = [p for p in picks if p.get("side")]
    print(f"decision rows {len(picks)}, bets {len(bets)}, events with bets {len({b['event'] for b in bets})}")
    cr = C.read("counts.jsonl")
    cc = [r for r in cr if r.get("rss_count") is not None and r.get("rc_count_since_rss_lo") is not None]
    if cc:
        diff = [r["rss_count"] - r["rc_count_since_rss_lo"] for r in cc]
        print(f"RSS vs RC cross-check (posts since the RSS horizon): mean diff {np.mean(diff):+.2f}, "
              f"|diff|>2 on {sum(abs(d) > 2 for d in diff)} of {len(diff)} rows")
    if not picks:
        return 0
    res = C.results_for({p["ticker"] for p in picks})
    out = C.report_rows(picks, res, C.read("snapshots.jsonl"))
    print(json.dumps(out, indent=1))
    # shadow: same rule with the Poisson probability (baseline for overdispersion; NOT the registered test)
    sh = []
    for p in picks:
        if p.get("p_yes_poisson") is not None:
            sh.append({**p, "p_yes": p["p_yes_poisson"], **C.decide(p["p_yes_poisson"], p["yes_bid"], p["yes_ask"])})
    if sh:
        print("SHADOW Poisson:", json.dumps(C.report_rows(sh, res, []).get("ALL"), indent=1))
    return 0


def cmd_model() -> int:
    m = C.load_json("model.json", None)
    st = C.load_json("state.json", {})
    print(json.dumps(m, indent=1) if m else f"calibrating: first_tick {C.iso(st['first_tick']) if st else 'none'}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tick"
    if cmd == "tick":
        o = tick()
        if o["errors"] or o["picks_logged"] or datetime.now(timezone.utc).minute < 15:
            print(json.dumps(o))
    elif cmd == "tick-verbose":
        print(json.dumps(tick()))
    elif cmd == "report":
        report()
    elif cmd == "model":
        cmd_model()
    else:
        print(__doc__)
