"""rt_review_lag M4: outlet-review nowcast at embargo + 30 min vs the Kalshi tape (pre-reg in notes/rt_review_lag.md).

    python scripts/rt_m4.py collect     # markets (live or /historical), MC slug, outlet times, trades near E -> films.json
    python scripts/rt_m4.py score       # LOFO map, nowcast, Brier vs tape -> m4_rows.json + summary

Read-only. Kalshi via kx_throttle; MC feed + outlet pages: 2 s between requests, robots.txt respected, no logins.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_outlet_times as ro  # noqa: E402
from kx_throttle import kget  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "rt_review_lag"
FILMS = OUT / "m4_films.json"
DEV_CLOSE_MAX = "2026-11-16T23:59:59Z"
EXCLUDE = {"KXRT-EBE"}


def pts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def parse_pub(s):
    """Outlet timestamp -> epoch seconds, or None if there is no time-of-day."""
    if not s or not re.search(r"\d{1,2}:\d{2}", s):
        return None
    s = s.strip()
    for fmt in (None, "%a, %d %b %Y %H:%M %Z"):
        try:
            if fmt is None:
                x = s.replace("Z", "+00:00")
                x = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", x)
                d = dt.datetime.fromisoformat(x)
            else:
                from email.utils import parsedate_to_datetime
                d = parsedate_to_datetime(s)
            if d.tzinfo is None:
                return None  # ambiguous zone -> unusable for minute-level timing
            return d.timestamp()
        except ValueError:
            continue
    return None


def slugify(t):
    t = t.lower().replace("&", "and")
    t = re.sub(r"[’'\".:!?,()]", "", t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t


def film_title(ev):
    t = ev.get("title", "")
    m = re.match(r'^"?(.*?)"?\s*(?:·|:)?\s*Rotten Tomatoes score\??$', t)
    return (m.group(1) if m else t).strip().strip('"').strip()


def mc_resolve(title, close_ts):
    for slug in (slugify(title) + "-2026", slugify(title)):
        try:
            r = ro.S.get(ro.MC.format(slug=slug), params={"offset": 0, "limit": 50}, timeout=30)
        except Exception:  # noqa: BLE001
            continue
        time.sleep(2)
        if r.status_code != 200:
            continue
        d = r.json().get("data") or {}
        items = d.get("items") or []
        dates = [x["date"] for x in items if x.get("date")]
        if not dates:
            continue
        lo = dt.datetime.fromtimestamp(close_ts - 120 * 86400, dt.UTC).strftime("%Y-%m-%d")
        hi = dt.datetime.fromtimestamp(close_ts, dt.UTC).strftime("%Y-%m-%d")
        if all(lo <= x <= hi for x in dates):
            return slug
    return None


def collect():
    evs = json.load(open(OUT / "kxrt_events.json"))
    done = json.load(open(FILMS)) if FILMS.exists() else {}
    for ev in evs:
        tk = ev["event_ticker"]
        if tk in done or tk in EXCLUDE:
            continue
        ms = ev.get("markets") or []
        if not ms:
            r = kget("/historical/markets", {"event_ticker": tk}) or {}
            ms = r.get("markets") or []
        ms = [m for m in ms if m.get("result") in ("yes", "no")]
        if not ms:
            continue
        close = max(m["close_time"] for m in ms)
        if close > DEV_CLOSE_MAX:
            continue
        close_ts = pts(close)
        title = film_title(ev)
        rec = {"event": tk, "title": title, "close": close, "hist": not ev.get("markets"),
               "markets": [{"ticker": m["ticker"], "floor": m.get("floor_strike"), "strike_type": m.get("strike_type"),
                            "result": m["result"]} for m in ms]}
        slug = mc_resolve(title, close_ts)
        rec["mc_slug"] = slug
        if slug:
            items = ro.mc_reviews(slug)
            rec["metascore"] = ((items[0].get("reviewedProduct") or {}).get("criticScoreSummary") or {}).get("score") if items else None
            dates = sorted(x["date"] for x in items if x.get("date"))
            d0 = dates[0] if dates else None
            d1 = (dt.date.fromisoformat(d0) + dt.timedelta(days=1)).isoformat() if d0 else None
            revs = []
            for x in items:
                row = {"publication": x.get("publicationName"), "mc_score": x.get("score"), "mc_date": x.get("date"),
                       "url": x.get("url"), "published": None, "status": None}
                if d0 and x.get("date") and x["date"] <= d1 and (x.get("url") or "").startswith("http"):
                    if not ro.allowed(x["url"]):
                        row["status"] = "robots_disallow"
                    else:
                        try:
                            r = ro.S.get(x["url"], timeout=30)
                            row["status"] = r.status_code
                            if r.status_code == 200:
                                for p in ro.PATS:
                                    mm = re.search(p, r.text)
                                    if mm:
                                        row["published"] = mm.group(1)
                                        break
                        except Exception as e:  # noqa: BLE001
                            row["status"] = f"err {type(e).__name__}"
                        time.sleep(2)
                revs.append(row)
            rec["reviews"] = revs
            # embargo E
            ts = sorted(t for t in (parse_pub(r["published"]) for r in revs) if t)
            E = next((t for i, t in enumerate(ts) if sum(1 for u in ts[i:] if u - t <= 600) >= 3), None)
            rec["E"] = E
            if E:
                trades = []
                for m in rec["markets"]:
                    for path in ("/markets/trades", "/historical/trades"):
                        cur = None
                        got = []
                        while True:
                            p = {"ticker": m["ticker"], "min_ts": int(E - 3600), "max_ts": int(E + 4 * 3600), "limit": 1000}
                            if cur:
                                p["cursor"] = cur
                            r = kget(path, p) or {}
                            got += r.get("trades") or []
                            cur = r.get("cursor")
                            if not cur or not r.get("trades"):
                                break
                        if got:
                            trades += got
                            break
                rec["trades"] = [{"t": x["ticker"], "ts": pts(x["created_time"]), "p": float(x["yes_price_dollars"]),
                                  "c": float(x["count_fp"])} for x in trades]
        done[tk] = rec
        json.dump(done, open(FILMS, "w"))
        print(tk, title, "slug", slug, "E", rec.get("E") and dt.datetime.fromtimestamp(rec["E"], dt.UTC).isoformat(),
              "revs", len(rec.get("reviews", [])), "trades", len(rec.get("trades", [])), flush=True)


def rt_final(rec):
    yes = [m["floor"] for m in rec["markets"] if m["strike_type"] == "greater" and m["result"] == "yes" and m["floor"] is not None]
    no = [m["floor"] for m in rec["markets"] if m["strike_type"] == "greater" and m["result"] == "no" and m["floor"] is not None]
    lo = max(yes) if yes else 0.0
    hi = min(no) if no else 100.0
    return (lo + 0.5 + hi) / 2 if hi > lo else None   # score in (lo, hi]


def score(offset_min=30, tag="t30"):
    import numpy as np
    films = json.load(open(FILMS))
    usable = []
    for f in films.values():
        if not f.get("E") or f.get("metascore") is None:
            continue
        rf = rt_final(f)
        if rf is None:
            continue
        usable.append((f, rf))
    print("films collected", len(films), "with E+metascore+bracket", len(usable))
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    rng = np.random.default_rng(4)
    rows = []
    for i, (f, rf) in enumerate(usable):
        X = np.array([(g["metascore"] - 50) / 10 for j, (g, _) in enumerate(usable) if j != i])
        Y = np.array([logit(np.clip(r / 100, 0.01, 0.99)) for j, (_, r) in enumerate(usable) if j != i])
        A = np.vstack([np.ones_like(X), X]).T
        (a, b), *_ = np.linalg.lstsq(A, Y, rcond=None)
        sig = float(np.std(Y - A @ np.array([a, b]), ddof=2))
        tstar = f["E"] + offset_min * 60
        sc = [r["mc_score"] for r in f["reviews"] if r["mc_score"] is not None and (parse_pub(r["published"]) or 9e18) <= tstar]
        if len(sc) < 3:
            continue
        s, k = float(np.mean(sc)), len(sc)
        var_s = float(np.var(sc, ddof=1)) if k > 1 else 225.0
        mu = a + b * (s - 50) / 10
        sd = np.sqrt(sig ** 2 + b ** 2 * var_s / (100 * k))
        draws = np.floor(100 / (1 + np.exp(-(mu + sd * rng.standard_normal(20000)))) + 0.5)
        for m in f["markets"]:
            if m["strike_type"] != "greater" or m["floor"] is None:
                continue
            tr = [x for x in f.get("trades", []) if x["t"] == m["ticker"] and abs(x["ts"] - tstar) <= 600]
            if not tr:
                continue
            vw = sum(x["p"] * x["c"] for x in tr) / sum(x["c"] for x in tr)
            if not (0.03 <= vw <= 0.97):
                continue
            p = float((draws > m["floor"]).mean())
            y = 1.0 if m["result"] == "yes" else 0.0
            rows.append({"event": f["event"], "ticker": m["ticker"], "floor": m["floor"], "k": k, "s": s, "p_now": p,
                         "vwap": vw, "y": y, "rt_final": rf, "metascore": f["metascore"],
                         "d": (p - y) ** 2 - (vw - y) ** 2})
    json.dump(rows, open(OUT / f"m4_rows_{tag}.json", "w"), indent=0)
    ev = sorted({r["event"] for r in rows})
    per = np.array([np.mean([r["d"] for r in rows if r["event"] == e]) for e in ev])
    print(tag, "films", len(ev), "strikes", len(rows))
    if not len(ev):
        return
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(10000)]
    print(f"{tag} mean film Brier diff (nowcast - tape) {per.mean():+.4f}  CI [{np.percentile(bs, 2.5):+.4f}, "
          f"{np.percentile(bs, 97.5):+.4f}]  films nowcast better {int((per < 0).sum())}/{len(per)}")
    bn = np.mean([(r["p_now"] - r["y"]) ** 2 for r in rows]); bk = np.mean([(r["vwap"] - r["y"]) ** 2 for r in rows])
    print(f"{tag} pooled Brier nowcast {bn:.4f} tape {bk:.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect()
    elif sys.argv[1] == "score":
        score()
        score(180, "t180_descriptive")
