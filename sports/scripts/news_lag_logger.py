"""news_lag forward logger (READ-ONLY): official availability news timestamps + Kalshi top-of-book around them.

    nohup ../.venv_quant/bin/python scripts/news_lag_logger.py --until 2026-11-16T08:00:00Z > results/news_lag/forward/logger.out 2>&1 &

What it records (results/news_lag/forward/, one JSON object per line, `t` = local fetch time UTC epoch seconds):
  kalshi_<day>.jsonl  top of book for every open KXNFLGAME / KXNBAGAME market. One /markets?series_ticker= call per
                      league per poll (covers all games). Every 5 s while any game of that league is in its hot window,
                      else every 120 s. All Kalshi GETs go through scripts/kx_throttle.py (shared cross-process limit).
  news_<day>.jsonl    change events from free timestamped sources, polled only in hot windows:
     espn_inj    ESPN game summary `injuries` per team (status + ESPN's own `date`), logged on any change
     espn_roster ESPN core per-game competitor roster (NFL: HTTP status + didNotPlay/starter sets; first 200 and
                 any change logged) -- the game-day roster is the closest free proxy for the inactive list
     espn_news   ESPN league news feed: new article ids with `published` + headline
     nba_pdf     official NBA injury report (15-min PDFs): first time each label returns 200, Last-Modified, rows
                 for today's games
Hot windows: NFL KO-130 min .. KO+5 min; NBA tip-90 min .. tip+5 min.
No orders, no keys. Exits at --until.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kx_throttle import kget  # noqa: E402

OUT = ROOT / "results" / "news_lag" / "forward"
OUT.mkdir(parents=True, exist_ok=True)
ESPN = {"nfl": "football/nfl", "nba": "basketball/nba"}
SERIES = {"nfl": "KXNFLGAME", "nba": "KXNBAGAME"}
HOT = {"nfl": (130 * 60, 5 * 60), "nba": (90 * 60, 5 * 60)}
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (research; read-only logger)"
_last_espn = [0.0]


def now() -> float:
    return time.time()


def wl(name: str, rec: dict) -> None:
    day = pd.Timestamp.utcnow().strftime("%Y%m%d")
    with open(OUT / f"{name}_{day}.jsonl", "a") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def eget(url: str, timeout: int = 20):
    """ESPN/NBA GET with a local 0.25 s gap. Returns Response or None."""
    w = _last_espn[0] + 0.25 - now()
    if w > 0:
        time.sleep(w)
    _last_espn[0] = now()
    try:
        return S.get(url, timeout=timeout)
    except requests.RequestException:
        return None


def schedule() -> list[dict]:
    games = []
    today = pd.Timestamp.utcnow().tz_convert("America/New_York")
    for lg, path in ESPN.items():
        for d in {today.strftime("%Y%m%d"), (today + pd.Timedelta(days=1)).strftime("%Y%m%d")}:
            r = eget(f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard?dates={d}")
            if r is None or not r.ok:
                continue
            for e in r.json().get("events", []):
                comp = e["competitions"][0]
                teams = {c["homeAway"]: (c["team"]["id"], c["team"]["abbreviation"]) for c in comp["competitors"]}
                ko = pd.Timestamp(e["date"]).timestamp()
                games.append({"lg": lg, "id": e["id"], "ko": ko, "name": e["shortName"],
                              "home": teams["home"], "away": teams["away"]})
    seen, out = set(), []
    for g in games:
        if g["id"] not in seen:
            seen.add(g["id"])
            out.append(g)
    return out


def hot(g: dict, t: float) -> bool:
    pre, post = HOT[g["lg"]]
    return g["ko"] - pre <= t <= g["ko"] + post


def poll_kalshi(lg: str, t: float) -> None:
    j = kget("/markets", {"series_ticker": SERIES[lg], "status": "open", "limit": 1000})
    if not j:
        wl("kalshi", {"t": t, "lg": lg, "err": "none"})
        return
    rows = []
    for m in j.get("markets", []):
        rows.append([m["ticker"], m.get("yes_bid_dollars"), m.get("yes_ask_dollars"), m.get("yes_bid_size_fp"),
                     m.get("yes_ask_size_fp"), m.get("last_price_dollars"), m.get("volume_fp", m.get("volume")),
                     m.get("updated_time")])
    wl("kalshi", {"t": t, "t_done": now(), "lg": lg, "cols": "tk,yb,ya,ybs,yas,last,vol,upd", "m": rows})


STATE: dict = {}


def changed(key: str, val) -> bool:
    h = hashlib.sha1(json.dumps(val, sort_keys=True, default=str).encode()).hexdigest()
    if STATE.get(key) != h:
        STATE[key] = h
        return True
    return False


def poll_inj(g: dict, t: float) -> None:
    r = eget(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN[g['lg']]}/summary?event={g['id']}")
    if r is None or not r.ok:
        return
    js = r.json()
    inj = {}
    for tm in js.get("injuries", []) or []:
        ab = tm.get("team", {}).get("abbreviation")
        inj[ab] = sorted([[i.get("athlete", {}).get("displayName"), i.get("athlete", {}).get("position", {}).get("abbreviation") if isinstance(i.get("athlete", {}).get("position"), dict) else None,
                           i.get("status"), i.get("date"), (i.get("details") or {}).get("fantasyStatus", {}).get("abbreviation") if isinstance(i.get("details"), dict) else None]
                          for i in tm.get("injuries", [])], key=lambda x: str(x[0]))
    if changed(f"inj:{g['id']}", inj):
        wl("news", {"t": t, "src": "espn_inj", "lg": g["lg"], "game": g["id"], "name": g["name"], "ko": g["ko"], "inj": inj})
    # boxscore players (NBA starters appear here near tip)
    box = js.get("boxscore", {}).get("players") or []
    starters = {}
    for tm in box:
        ab = tm.get("team", {}).get("abbreviation")
        st = []
        for sg in tm.get("statistics", []):
            for a in sg.get("athletes", []):
                st.append([a.get("athlete", {}).get("displayName"), a.get("starter"), a.get("didNotPlay"), a.get("reason")])
        starters[ab] = sorted(st, key=lambda x: str(x[0]))
    if starters and changed(f"box:{g['id']}", starters):
        wl("news", {"t": t, "src": "espn_box", "lg": g["lg"], "game": g["id"], "name": g["name"], "ko": g["ko"], "box": starters})


def poll_roster(g: dict, t: float) -> None:
    lg_path = "football/leagues/nfl" if g["lg"] == "nfl" else "basketball/leagues/nba"
    for side in ("home", "away"):
        tid, ab = g[side]
        u = f"https://sports.core.api.espn.com/v2/sports/{lg_path}/events/{g['id']}/competitions/{g['id']}/competitors/{tid}/roster"
        r = eget(u)
        if r is None:
            continue
        val = {"http": r.status_code}
        if r.ok:
            ents = r.json().get("entries", [])
            val["n"] = len(ents)
            val["dnp"] = sorted(str(e.get("displayName")) + ":" + str(e.get("playerId")) for e in ents if e.get("didNotPlay"))
            val["starter"] = sorted(str(e.get("displayName")) + ":" + str(e.get("playerId")) for e in ents if e.get("starter"))
            val["active_n"] = sum(bool(e.get("active")) for e in ents)
        if changed(f"ros:{g['id']}:{tid}", val):
            wl("news", {"t": t, "src": "espn_roster", "lg": g["lg"], "game": g["id"], "team": ab, "ko": g["ko"], **val})


def poll_news(lg: str, t: float) -> None:
    r = eget(f"https://site.api.espn.com/apis/site/v2/sports/{ESPN[lg]}/news?limit=50")
    if r is None or not r.ok:
        return
    for a in r.json().get("articles", []):
        k = f"news:{lg}:{a.get('id') or a.get('headline')}"
        if k not in STATE:
            STATE[k] = 1
            wl("news", {"t": t, "src": "espn_news", "lg": lg, "id": a.get("id"), "published": a.get("published"),
                        "lastModified": a.get("lastModified"), "type": a.get("type"), "headline": a.get("headline"),
                        "desc": (a.get("description") or "")[:300], "first_poll": STATE.get(f"newsinit:{lg}") is None})
    STATE[f"newsinit:{lg}"] = 1


def nba_label(ts: pd.Timestamp) -> str:
    et = ts.tz_convert("America/New_York")
    m = et.minute // 15 * 15
    h = et.hour
    hh, ap = (h - 12 if h > 12 else (12 if h == 0 else h)), ("PM" if h >= 12 else "AM")
    return f"{et.strftime('%Y-%m-%d')}_{hh:02d}_{m:02d}{ap}"


def poll_nba_pdf(t: float) -> None:
    lab = nba_label(pd.Timestamp(t, unit="s", tz="UTC"))
    if STATE.get(f"pdf:{lab}"):
        return
    r = eget(f"https://ak-static.cms.nba.com/referee/injury/Injury-Report_{lab}.pdf", timeout=30)
    if r is None or r.status_code != 200:
        return
    STATE[f"pdf:{lab}"] = 1
    rec = {"t": t, "src": "nba_pdf", "lg": "nba", "label": lab, "last_modified": r.headers.get("Last-Modified"), "bytes": len(r.content)}
    try:
        import q011_reports as Q
        p = Q.parse(r.content)
        rec["header"] = p["header"]
        rec["rows"] = p["rows"]
        rec["notsub"] = p["notsub"]
    except Exception as e:  # keep the timestamp even if parsing fails
        rec["parse_error"] = repr(e)[:300]
    wl("news", rec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="2026-11-16T08:00:00Z")
    a = ap.parse_args()
    until = pd.Timestamp(a.until).timestamp()
    games, sched_t = [], 0.0
    last = {}
    print(f"start {pd.Timestamp.utcnow()} until {a.until}", flush=True)
    while now() < until:
        t = now()
        try:
            if t - sched_t > 1800 or not games:
                games = schedule()
                sched_t = t
                wl("news", {"t": t, "src": "schedule", "games": games})
                print(f"{pd.Timestamp.utcnow()} schedule {len(games)} games", flush=True)
            hot_g = [g for g in games if hot(g, t)]
            for lg in ("nfl", "nba"):
                lg_hot = any(g["lg"] == lg for g in hot_g)
                gap = 5 if lg_hot else 120
                if t - last.get(f"k:{lg}", 0) >= gap:
                    last[f"k:{lg}"] = t
                    poll_kalshi(lg, t)
                if lg_hot and t - last.get(f"n:{lg}", 0) >= 30:
                    last[f"n:{lg}"] = t
                    poll_news(lg, t)
            for g in hot_g:
                if t - last.get(f"i:{g['id']}", 0) >= 20:
                    last[f"i:{g['id']}"] = t
                    poll_inj(g, now())
                if t - last.get(f"r:{g['id']}", 0) >= 15:
                    last[f"r:{g['id']}"] = t
                    poll_roster(g, now())
            et_hour = pd.Timestamp(t, unit="s", tz="UTC").tz_convert("America/New_York").hour
            nba_today = any(g["lg"] == "nba" and g["ko"] > t - 3600 and g["ko"] - t < 14 * 3600 for g in games)
            if nba_today and 11 <= et_hour <= 23 and t - last.get("pdf", 0) >= 30:
                last["pdf"] = t
                poll_nba_pdf(t)
            if t - last.get("hb", 0) >= 600:
                last["hb"] = t
                print(f"{pd.Timestamp.utcnow()} alive hot={[g['name'] for g in hot_g]}", flush=True)
        except Exception:
            print(traceback.format_exc(), flush=True)
            time.sleep(5)
        time.sleep(1.0)
    print(f"done {pd.Timestamp.utcnow()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
