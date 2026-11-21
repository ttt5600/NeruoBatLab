"""Forward, read-only logger for one UFC event (no orders). Measures what history cannot:
 (1) publication latency of the UFC live JSON (first time WE see fight_over / results / unofficial_winner_* vs the
     action's own Timestamp), (2) Kalshi KXUFCFIGHT order-book depth in the horn -> announcement window, (3) the Kalshi
     book at the moment the JSON first shows the winner.
Usage: fwd_logger.py EVENT_FMID KALSHI_DATECODE START_UTC STOP_UTC  e.g. 1348 26NOV21 2026-11-21T14:45:00Z 2026-11-21T23:30:00Z
Polls the CDN JSON every 3 s; Kalshi orderbooks (via kx_throttle) every ~2 s only while a fight is in its final round or
between fight_over and the winner + 90 s; plus a light 60-s snapshot otherwise. Writes JSONL to fwd/<fmid>.jsonl."""
import json, sys, time, datetime as dt
from pathlib import Path
import requests
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1] / "scripts"))
from kx_throttle import kget
fmid, dcode = sys.argv[1], sys.argv[2]
start, stop = (dt.datetime.fromisoformat(a.replace("Z", "+00:00")) for a in sys.argv[3:5])
while dt.datetime.now(dt.timezone.utc) < start: time.sleep(30)
(OUT / "fwd").mkdir(exist_ok=True)
LOG = open(OUT / "fwd" / f"{fmid}.jsonl", "a")
S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0"
def now(): return dt.datetime.now(dt.timezone.utc)
def log(kind, **k): LOG.write(json.dumps({"ts": now().isoformat(), "kind": kind, **k}) + "\n"); LOG.flush()
def kalshi_map():
    ms, cur = [], None
    while True:
        p = {"series_ticker": "KXUFCFIGHT", "status": "open", "limit": 1000}
        if cur: p["cursor"] = cur
        r = kget("/markets", p) or {}; ms += r.get("markets", []); cur = r.get("cursor")
        if not cur or not r.get("markets"): break
    return [m for m in ms if f"-{dcode}" in m["event_ticker"]]
seen, kmap, last_snap, mk = set(), {}, 0.0, kalshi_map()
log("kalshi_markets", tickers=[(m["ticker"], m["yes_sub_title"]) for m in mk])
def tickers_for(fight):
    last = {(x["Name"]["LastName"] or "").lower() for x in fight["Fighters"]}
    return [m["ticker"] for m in mk if (m["yes_sub_title"] or "").split()[-1].lower() in last]
hot = {}   # fight id -> until (epoch)
first = True   # backlog on the first poll is logged but never opens a hot window
while now() < stop:
    t0 = time.time()
    try:
        r = S.get(f"https://d29dxerjsp82wz.cloudfront.net/api/v3/event/live/{fmid}.json", params={"_": int(t0)}, timeout=10)
        d = r.json()["LiveEventDetail"]
        log_hdr = {"age": r.headers.get("Age"), "xcache": r.headers.get("X-Cache"), "lastmod": r.headers.get("Last-Modified")}
        for f in d["FightCard"]:
            fid = f["FightId"]
            for a in f.get("FightNightTracking") or []:
                key = (fid, a["ActionId"])
                if key in seen: continue
                seen.add(key)
                log("action", fight=fid, type=a["Type"], round=a.get("RoundNumber"), action_ts=a["Timestamp"], hdr=log_hdr, backlog=first)
                if first: continue
                if a["Type"] == "round_start" and a.get("RoundNumber") == (f.get("RuleSet") or {}).get("PossibleRounds"):
                    hot[fid] = t0 + 15 * 60
                if a["Type"] == "fight_over": hot[fid] = t0 + 10 * 60
                if a["Type"].startswith("unofficial_winner"): hot[fid] = min(hot.get(fid, t0), t0 + 90)
            res = f.get("Result") or {}
            k2 = (fid, "result", res.get("Method"), bool(res.get("FightScores")), bool(res.get("RoundScores")))
            if res.get("Method") and k2 not in seen:
                seen.add(k2); log("result_seen", fight=fid, method=res.get("Method"), has_scores=bool(res.get("FightScores")), result=res)
        live = d.get("LiveFightId")
        snap = time.time() - last_snap > 60
        for f in d["FightCard"]:
            fid = f["FightId"]
            if hot.get(fid, 0) > t0 or (snap and fid == live):
                for tk in tickers_for(f):
                    ob = kget(f"/markets/{tk}/orderbook", {"depth": 10})
                    log("orderbook", fight=fid, ticker=tk, ob=ob, hot=hot.get(fid, 0) > t0)
        if snap: last_snap = time.time()
        first = False
        if d.get("Status") == "Final" and not any(v > t0 for v in hot.values()): log("event_final"); break
    except Exception as e:
        log("error", err=str(e)[:300])
    gap = 1.5 if any(v > t0 for v in hot.values()) else 5.0
    time.sleep(max(0.0, gap - (time.time() - t0)))
log("stop")
