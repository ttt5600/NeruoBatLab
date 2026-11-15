"""NHL starting-goalie CONFIRMATION timestamp logger + Kalshi NHL ladder snapshots (READ-ONLY; notes/ops_nhl_goalie.md).

    python scripts/nhl_goalie_logger.py tick                    # launchd every 120 s (idle exit when no game is near)
    python scripts/nhl_goalie_logger.py dfo-archive 2026-10-07 2026-11-14   # DailyFaceoff date pages, 1 per 15 s
    python scripts/nhl_goalie_logger.py status

Sources (free, read-only, low rate):
  dfo   DailyFaceoff /starting-goalies/<ET date> HTML page (robots.txt: Allow / except /api/ /cms/; we read the page's
        __NEXT_DATA__ JSON only): per side goalie, newsStrengthName (Confirmed / Likely / ...), newsCreatedAt, source.
        At most one fetch per ET date every 10 min, only while a game is within 14 h.
  espn  ESPN scoreboard competitors[].probables (probableStartingGoalie; no confirmed flag), every 10 min.
  nhl   api-web.nhle.com gamecenter/<id>/boxscore goalies[].starter (official lineup), every tick in
        [tip - 75 min, tip + 20 min] until both starters are seen.
  kx    Kalshi KXNHLTOTAL / KXNHLGAME / KXNHLSPREAD top of book for every open market of games with tip in the next
        14 h, every tick (3 GETs via scripts/kx_throttle.py).
Output (~/.claude/quant-sports/data/nhl_goalie, override NHLG_DIR; append-only):
  events.jsonl         one row per NEW observation (source, gameId, side, goalie, status, news_created_at, first_seen_at)
  kx/<UTC date>.jsonl.gz  one row per market per tick: t, tk, bid, ask, last, vol, oi (gzip members appended)
  games.jsonl          schedule rows (gameId, tip, teams) the first time a game enters the window
  state.json           poll clocks + dedupe keys + schedule cache
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kx_throttle import kget  # noqa: E402

OUT = Path(os.environ.get("NHLG_DIR", str(Path.home() / ".claude" / "quant-sports" / "data" / "nhl_goalie")))
ET = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (quant-sports read-only research logger; low rate)"}
WINDOW_H = float(os.environ.get("NHLG_WINDOW_H", "14"))   # Kalshi snapshots: tip within 14 h
SRC_H = float(os.environ.get("NHLG_SRC_H", "36"))         # goalie sources: tip within 36 h (some starters named the day before)
DFO_EVERY, ESPN_EVERY, SCHED_EVERY = 600, 600, 3 * 3600
KX_CODE = {"SJS": "SJ", "LAK": "LA", "TBL": "TB", "NJD": "NJ"}            # NHL -> Kalshi
TO_NHL = {"SJ": "SJS", "LA": "LAK", "TB": "TBL", "NJ": "NJD", "UTAH": "UTA", "WAS": "WSH", "MON": "MTL"}  # ESPN/DFO -> NHL


def now_utc():
    return datetime.now(timezone.utc)


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def pt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def nhl(code: str) -> str:
    return TO_NHL.get(code, code)


def http_json(url, params=None):
    for i in range(3):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.json()
            if 400 <= r.status_code < 500:
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 * (i + 1))
    return None


def load_state():
    p = OUT / "state.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"seen": [], "polls": {}, "sched": {}, "games_logged": [], "nhl_done": []}


def save_state(s):
    s["seen"] = s["seen"][-20000:]
    tmp = OUT / "state.json.tmp"
    tmp.write_text(json.dumps(s))
    tmp.replace(OUT / "state.json")


def append(name, rows):
    if not rows:
        return
    with open(OUT / name, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def schedule(st, now):
    """Games (regular + playoffs) with tip in [now - 1 day, now + 2 days]; NHL API weekly schedule cached 3 h."""
    if st["sched"].get("t", 0) > time.time() - SCHED_EVERY and st["sched"].get("games"):
        return st["sched"]["games"]
    games = []
    for d in {(now - timedelta(days=1)).astimezone(ET).date(), now.astimezone(ET).date()}:
        j = http_json(f"https://api-web.nhle.com/v1/schedule/{d}") or {}
        for wk in j.get("gameWeek", []):
            for g in wk.get("games", []):
                if g.get("gameType") not in (2, 3):
                    continue
                games.append(dict(gameId=g["id"], date=wk["date"], tip=g["startTimeUTC"], home=g["homeTeam"]["abbrev"],
                                  away=g["awayTeam"]["abbrev"]))
    if games:
        games = list({g["gameId"]: g for g in games}.values())
        st["sched"] = {"t": time.time(), "games": games}
    return st["sched"].get("games", games)


def kx_event_suffix(g):
    d = pt(g["tip"]).astimezone(ET).strftime("%y%b%d").upper()
    return f"{d}{KX_CODE.get(g['away'], g['away'])}{KX_CODE.get(g['home'], g['home'])}"


def obs(st, out, key, row):
    k = json.dumps(key)
    if k in st["seen_set"]:
        return
    st["seen_set"].add(k)
    st["seen"].append(k)
    out.append(row)


def dfo_rows(date: str):
    try:
        r = requests.get(f"https://www.dailyfaceoff.com/starting-goalies/{date}", headers=UA, timeout=30)
    except requests.RequestException as e:
        return None, f"dfo error {type(e).__name__}"
    if r.status_code != 200:
        return None, f"dfo http {r.status_code}"
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
    if not m:
        return None, "dfo: no __NEXT_DATA__"
    try:
        return json.loads(m.group(1))["props"]["pageProps"].get("data") or [], None
    except (ValueError, KeyError) as e:
        return None, f"dfo parse {e}"


def dfo_side(g, side):
    p = side
    return dict(goalie=g.get(f"{p}GoalieName"), dfo_goalie_id=g.get(f"{p}GoalieId"), status=g.get(f"{p}NewsStrengthName"),
                news_created_at=g.get(f"{p}NewsCreatedAt"), news_source=g.get(f"{p}NewsSourceName"),
                news_url=g.get(f"{p}NewsSourceUrl"), news_details=g.get(f"{p}NewsDetails"),
                team=nhl(g.get(f"{p}TeamAbbreviation") or ""))


def tick():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kx").mkdir(exist_ok=True)
    now = now_utc()
    st = load_state()
    st["seen_set"] = set(st["seen"])
    games = schedule(st, now)
    near = [g for g in games if now - timedelta(minutes=20) <= pt(g["tip"]) <= now + timedelta(hours=SRC_H)]
    (OUT / "heartbeat").write_text(iso(now))
    if not near:
        st.pop("seen_set")
        save_state(st)
        print(f"{iso(now)} idle: no NHL game within {SRC_H} h (schedule {len(games)} games)")
        return
    byid = {g["gameId"]: g for g in near}
    bykey = {(g["date"], nhl(g["away"]), nhl(g["home"])): g for g in near}
    new_games = [dict(g, first_in_window=iso(now)) for g in near if g["gameId"] not in st["games_logged"]]
    st["games_logged"] = (st["games_logged"] + [g["gameId"] for g in new_games])[-3000:]
    append("games.jsonl", new_games)
    ev, msg = [], []
    pre = [g for g in near if pt(g["tip"]) > now]
    # ---- DailyFaceoff (<= 1 fetch per ET date per 10 min)
    for d in sorted({g["date"] for g in pre}):
        if st["polls"].get(f"dfo_{d}", 0) > time.time() - DFO_EVERY:
            continue
        st["polls"][f"dfo_{d}"] = time.time()
        rows, err = dfo_rows(d)
        if err:
            msg.append(err); continue
        n0 = len(ev)
        for g in rows:
            mg = bykey.get((g.get("date"), nhl(g.get("awayTeamAbbreviation") or ""), nhl(g.get("homeTeamAbbreviation") or "")))
            for side in ("home", "away"):
                s = dfo_side(g, side)
                obs(st, ev, ["dfo", g.get("date"), g.get("awayTeamAbbreviation"), g.get("homeTeamAbbreviation"), side,
                             s["goalie"], s["status"], s["news_created_at"]],
                    dict(source="dfo", first_seen_at=iso(now), gameId=mg["gameId"] if mg else None, date=g.get("date"),
                         tip=mg["tip"] if mg else g.get("dateGmt"), side=side, **s))
        msg.append(f"dfo {d}: {len(rows)} games, +{len(ev) - n0} obs")
    # ---- ESPN probables (every 10 min)
    if st["polls"].get("espn", 0) <= time.time() - ESPN_EVERY:
        st["polls"]["espn"] = time.time()
        n0 = len(ev)
        for d in sorted({g["date"] for g in pre}):
            j = http_json("https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
                          {"dates": d.replace("-", "")}) or {}
            for e in j.get("events", []):
                comp = (e.get("competitions") or [{}])[0]
                cs = {c.get("homeAway"): c for c in comp.get("competitors", [])}
                if set(cs) != {"home", "away"}:
                    continue
                mg = bykey.get((d, nhl(cs["away"]["team"]["abbreviation"]), nhl(cs["home"]["team"]["abbreviation"])))
                for side, c in cs.items():
                    for p in c.get("probables") or []:
                        a = p.get("athlete") or {}
                        obs(st, ev, ["espn", e.get("id"), side, a.get("id")],
                            dict(source="espn", first_seen_at=iso(now), gameId=mg["gameId"] if mg else None, date=d,
                                 tip=mg["tip"] if mg else e.get("date"), side=side, team=nhl(c["team"]["abbreviation"]),
                                 goalie=a.get("fullName"), espn_goalie_id=a.get("id"), status="probable",
                                 espn_event=e.get("id")))
        msg.append(f"espn: +{len(ev) - n0} obs")
    # ---- NHL API official starters near puck drop
    for g in near:
        tip = pt(g["tip"])
        if g["gameId"] in st["nhl_done"] or not (tip - timedelta(minutes=75) <= now <= tip + timedelta(minutes=20)):
            continue
        b = http_json(f"https://api-web.nhle.com/v1/gamecenter/{g['gameId']}/boxscore") or {}
        pb = b.get("playerByGameStats") or {}
        found = 0
        for side in ("home", "away"):
            for x in (pb.get(f"{side}Team") or {}).get("goalies", []):
                if x.get("starter"):
                    found += 1
                    obs(st, ev, ["nhl", g["gameId"], side, x.get("playerId")],
                        dict(source="nhl", first_seen_at=iso(now), gameId=g["gameId"], date=g["date"], tip=g["tip"],
                             side=side, team=g[side], goalie=(x.get("name") or {}).get("default"),
                             nhl_goalie_id=x.get("playerId"), status="starter", game_state=b.get("gameState")))
        if found == 2:
            st["nhl_done"] = (st["nhl_done"] + [g["gameId"]])[-3000:]
        msg.append(f"nhl {g['gameId']}: starters {found}")
    append("events.jsonl", ev)
    # ---- Kalshi snapshots for pre-tip games
    suf = {kx_event_suffix(g): g for g in pre if pt(g["tip"]) <= now + timedelta(hours=WINDOW_H)}
    if not suf:
        msg.append(f"kalshi: no game within {WINDOW_H} h")
    snap = []
    for ser in ("KXNHLTOTAL", "KXNHLGAME", "KXNHLSPREAD") if suf else ():
        cur = None
        for _ in range(5):
            p = {"series_ticker": ser, "status": "open", "limit": 1000}
            if cur:
                p["cursor"] = cur
            j = kget("/markets", p) or {}
            t = iso(now_utc())
            for m in j.get("markets", []):
                if m.get("event_ticker", "").split("-")[-1] not in suf:
                    continue
                snap.append(dict(t=t, tk=m["ticker"], bid=m.get("yes_bid_dollars"), ask=m.get("yes_ask_dollars"),
                                 last=m.get("last_price_dollars"), vol=m.get("volume_fp", m.get("volume")),
                                 oi=m.get("open_interest_fp", m.get("open_interest")), fs=m.get("floor_strike")))
            cur = j.get("cursor")
            if not cur or not j.get("markets"):
                break
    if snap:
        with gzip.open(OUT / "kx" / f"{now.strftime('%Y-%m-%d')}.jsonl.gz", "at") as f:
            for r in snap:
                f.write(json.dumps(r) + "\n")
    st.pop("seen_set")
    save_state(st)
    print(f"{iso(now)} games near {len(near)} (pre-tip {len(pre)}, kalshi events {len({r['tk'].rsplit('-', 1)[0] for r in snap})}); "
          f"+{len(ev)} goalie obs ({sum(r['source'] == 'dfo' and r.get('status') == 'Confirmed' for r in ev)} dfo Confirmed); "
          f"{len(snap)} kalshi rows; " + "; ".join(msg))


def dfo_archive(d0: str, d1: str):
    """DailyFaceoff date pages (one request per 15 s) -> dfo_archive.jsonl: the final per-side row of each past game.
    Caveat: a past page shows each side's LATEST news item; newsCreatedAt of a Confirmed row = time of that item,
    which may postdate an earlier confirmation (e.g. a later update) -- an upper bound on the confirmation time."""
    OUT.mkdir(parents=True, exist_ok=True)
    have = set()
    p = OUT / "dfo_archive.jsonl"
    if p.exists():
        have = {json.loads(l)["page_date"] for l in p.read_text().splitlines() if l.strip()}
    d, end = datetime.fromisoformat(d0).date(), datetime.fromisoformat(d1).date()
    while d <= end:
        if str(d) not in have:
            rows, err = dfo_rows(str(d))
            out = []
            for g in rows or []:
                for side in ("home", "away"):
                    out.append(dict(page_date=str(d), fetched_at=iso(now_utc()), date=g.get("date"), tip=g.get("dateGmt"),
                                    away=nhl(g.get("awayTeamAbbreviation") or ""), home=nhl(g.get("homeTeamAbbreviation") or ""),
                                    side=side, **dfo_side(g, side)))
            if err:
                out = [dict(page_date=str(d), fetched_at=iso(now_utc()), error=err)]
            append("dfo_archive.jsonl", out)
            print(d, err or f"{len(rows)} games", flush=True)
            time.sleep(15)
        d += timedelta(days=1)


def status():
    e = [json.loads(l) for l in (OUT / "events.jsonl").read_text().splitlines()] if (OUT / "events.jsonl").exists() else []
    from collections import Counter
    print("events", len(e), dict(Counter((r["source"], r.get("status")) for r in e)))
    for f in sorted((OUT / "kx").glob("*.jsonl.gz"))[-3:]:
        with gzip.open(f, "rt") as fh:
            n = sum(1 for _ in fh)
        print(f.name, n, "rows", f.stat().st_size, "bytes")


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "tick":
        tick()
    elif a == "dfo-archive":
        dfo_archive(sys.argv[2], sys.argv[3])
    else:
        status()
