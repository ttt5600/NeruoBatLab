"""NHL in-game FORWARD paper logger (no orders): Poisson + empty-net model vs live KXNHLGAME quotes.

    python scripts/nhl_ingame_logger.py tick          # one pass (launchd: com.jonathanwang.quant-sports-nhl-ingame, 60 s)
    python scripts/nhl_ingame_logger.py loop [--sec 60]
    python scripts/nhl_ingame_logger.py report        # counts per day / games / settled (no outcomes joined)

Why forward-only: Kalshi NHL markets start 2025-04; NHL 2025-26 is SEALED by the frozen MANIFEST
(notes/nhl_simple_model.md), so the only non-sealed history is 83 games (Apr-Jun 2025) -- too thin for a trial.
edgelab trial: dataset kalshi_nhl_ingame (forward from 2026-11-08), rule edgelab/rules/nhl_ingame.py. The RULE decides
bets from these rows; the `signal` field written here is an informational paper flag with the same threshold.

Sources (each tick): api-web.nhle.com/v1/score/now (state, clock secondsRemaining, scores) and Kalshi
/markets?series_ticker=KXNHLGAME (both team markets per game). Model = scripts/kalshi_ingame_lowscore.py nhl_probs
(2023-24 pbp prior: base goal rate + last-3-min empty-net multipliers; OT win prob = pregame strength share).
Pregame anchor = home market two-sided mid (spread <= 5c) seen while the game was still FUT/PRE; if the logger was not
running before puck drop, the last 1-min Kalshi candle ending <= scheduled start (two-sided, spread <= 5c).

Cheap when idle: the day's start times are cached in the state file (refreshed every 3 h); with no game in
[start - 75 min, start + 5 h] the tick exits without any network call.

Output (ROOT/data/live/):
  nhl_ingame_paper.jsonl    one row per live game per tick: decided_at (UTC, after both fetches), as_of {nhl, kalshi,
                            anchor}, NHL state, minutes_left (regulation; null in OT/SO), home/away bid/ask, p_model
                            (P home wins incl. OT/SO; null without anchor or past regulation), signal. Plus one
                            `final` row per game the first tick it is seen FINAL/OFF.
  nhl_ingame_results.jsonl  Kalshi settlement per ticker (result yes/no), fetched after the game is final.
  nhl_ingame_paper_state.json  anchors, schedule cache, pending settlements.
Errors in one source are logged and the tick exits 1 (the launchd wrapper records the exit code); state stays valid.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

KAPI = "https://api.elections.kalshi.com/trade-api/v2"
NHL_NOW = "https://api-web.nhle.com/v1/score/now"
LIVE = ROOT / "data" / "live"
OUT = LIVE / "nhl_ingame_paper.jsonl"
RESULTS = LIVE / "nhl_ingame_results.jsonl"
STATE = LIVE / "nhl_ingame_paper_state.json"
# NHL api abbrev -> Kalshi ticker code (Kalshi list checked 2026-11-06: LA, NJ, SJ, TB; the rest identical)
KCODE = {"LAK": "LA", "NJD": "NJ", "SJS": "SJ", "TBL": "TB"}
EDGE, TAKER = 0.03, 0.07          # paper flag only; the registered edgelab rule is the decision of record
PRE_WIN, POST_WIN = 75 * 60, 5 * 3600
LIVE_STATES = {"LIVE", "CRIT"}
FINAL_STATES = {"FINAL", "OFF"}
MONTHS = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()


def _get(url, params=None, timeout=15):
    import requests
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


def epoch(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def event_ticker(game: dict) -> str:
    y, m, d = game["gameDate"].split("-")
    k = lambda t: KCODE.get(t["abbrev"], t["abbrev"])  # noqa: E731
    return f"KXNHLGAME-{y[2:]}{MONTHS[int(m) - 1]}{d}{k(game['awayTeam'])}{k(game['homeTeam'])}"


def minutes_left(game: dict) -> float | None:
    """Regulation minutes remaining; None once past regulation (OT / SO) or unknown."""
    pd_ = game.get("periodDescriptor") or {}
    per = int(game.get("period") or pd_.get("number") or 0)
    if per < 1 or per > 3 or pd_.get("periodType", "REG") != "REG":
        return None
    clk = game.get("clock") or {}
    if clk.get("inIntermission"):
        left_in = 0.0                        # intermission AFTER period `per` (clock there = intermission time)
    elif clk.get("secondsRemaining") is not None:
        left_in = float(clk["secondsRemaining"]) / 60
    else:
        mm, ss = (str(clk.get("timeRemaining", "20:00")).split(":") + ["0"])[:2]
        left_in = int(mm) + float(ss) / 60
    return round((3 - per) * 20 + left_in, 4)


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text()) if STATE.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}                            # a torn state file must not stop logging (anchors rebuild from candles)


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st))
    os.replace(tmp, STATE)


def append(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def active(st: dict, now: datetime) -> bool | None:
    """True/False from the cached schedule; None = cache stale (caller refreshes)."""
    sc = st.get("sched")
    if not sc or now.timestamp() - sc.get("fetched", 0) > 3 * 3600:
        return None
    t = now.timestamp()
    return any(s - PRE_WIN <= t <= s + POST_WIN for s in sc.get("starts", [])) or bool(st.get("pending"))


def candle_anchor(home_ticker: str, start: int, get=_get) -> dict | None:
    j = get(f"{KAPI}/series/KXNHLGAME/markets/{home_ticker}/candlesticks",
            {"start_ts": start - 3600, "end_ts": start, "period_interval": 1})
    best = None
    for c in j.get("candlesticks") or []:
        e = int(c["end_period_ts"])
        b = float((c.get("yes_bid") or {}).get("close_dollars") or 0)
        a = float((c.get("yes_ask") or {}).get("close_dollars") or 0)
        if e <= start and 0 < b < a < 1 and a - b <= 0.05:
            if best is None or e > best["t"]:
                best = {"t": e, "pm": (a + b) / 2}
    if best is None:
        return None
    return {"pm": best["pm"], "as_of": iso(datetime.fromtimestamp(best["t"], timezone.utc)), "src": "candle"}


def strength(pm: float, prior: dict):
    import numpy as np
    from kalshi_ingame_lowscore import nhl_probs
    tot = 2 * prior["base_per_team_min"]
    lo, hi = -3.0, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        r = 1 / (1 + np.exp(-mid))
        if nhl_probs(tot * r, tot * (1 - r), 0, 60.0, prior, r) > pm:
            hi = mid
        else:
            lo = mid
    r = float(1 / (1 + np.exp(-(lo + hi) / 2)))
    return tot, r


def model_p(pm: float, margin: int, left: float, prior: dict) -> float:
    from kalshi_ingame_lowscore import nhl_probs
    tot, r = strength(pm, prior)
    return float(nhl_probs(tot * r, tot * (1 - r), margin, left, prior, r))


def signal(p_home, hb, ha, ab, aa):
    """Paper flag: best (team, venue side) with model - price - taker fee >= EDGE. Informational only."""
    if p_home is None:
        return None
    opts = []
    for team, p, b, a in (("home", p_home, hb, ha), ("away", 1 - p_home, ab, aa)):
        if b is None or a is None or not (0 < b < a < 1):
            continue
        opts.append((p - a - TAKER * a * (1 - a), team, "yes", a))           # back `team` via its YES
        opts.append(((1 - p) - (1 - b) - TAKER * (1 - b) * b, "away" if team == "home" else "home", "no", 1 - b))
    if not opts:
        return None
    e, team, side, price = max(opts)
    return {"back": team, "via": side, "price": round(price, 4), "edge": round(e, 4)} if e >= EDGE else None


def settle(st: dict, now: datetime, get=_get, max_n: int = 20) -> int:
    """Fetch Kalshi results for pending tickers whose game was seen final (or is > 6 h past its last live tick)."""
    pend = st.setdefault("pending", {})
    done = 0
    rows = []
    for tk, info in list(pend.items())[:max_n]:
        if not info.get("final") and now.timestamp() - info.get("last", 0) < 6 * 3600:
            continue
        m = get(f"{KAPI}/markets/{tk}").get("market", {})
        res = m.get("result")
        if m.get("status") in ("settled", "finalized", "determined") and res in ("yes", "no"):
            rows.append({"ticker": tk, "result": res, "status": m.get("status"),
                         "settlement_ts": m.get("settlement_ts"), "fetched_at": iso(now)})
            pend.pop(tk)
            done += 1
        elif now.timestamp() - info.get("last", 0) > 7 * 86400:
            pend.pop(tk)                     # voided / never settled: give up after a week, say so
            print(f"settle: gave up on {tk} (status {m.get('status')}, result {res!r})")
    append(RESULTS, rows)
    return done


def tick(get=_get, now: datetime | None = None, clock=now_utc) -> int:
    """`clock` stamps each fetch (decided_at / as_of); tests pass a fixed clock."""
    now = now or clock()
    st = load_state()
    act = active(st, now)
    if act is False:
        print(f"{iso(now)} idle: no NHL game window")
        return 0
    try:
        sb = get(NHL_NOW)
        t_nhl = clock()
    except Exception as ex:  # noqa: BLE001
        print(f"{iso(now)} nhl api error: {type(ex).__name__}: {ex}")
        return 1
    games = sb.get("games") or []
    st["sched"] = {"fetched": now.timestamp(), "date": sb.get("currentDate"),
                   "starts": sorted({epoch(g["startTimeUTC"]) for g in games if g.get("startTimeUTC")})}
    if not active(st, now):
        save_state(st)
        print(f"{iso(now)} idle: {len(games)} NHL games on {sb.get('currentDate')}, none in window")
        return 0
    rc = 0
    try:
        mk = get(f"{KAPI}/markets", {"series_ticker": "KXNHLGAME", "status": "open", "limit": 1000}).get("markets", [])
        t_k = clock()
    except Exception as ex:  # noqa: BLE001
        print(f"{iso(now)} kalshi api error: {type(ex).__name__}: {ex}")
        mk, t_k, rc = None, None, 1
    prior = json.loads((ROOT / "data" / "ingame_lowscore_nhl_prior.json").read_text())
    anchors = st.setdefault("anchor", {})
    pend = st.setdefault("pending", {})
    finals = st.setdefault("final_seen", {})
    by_ticker = {m["ticker"]: m for m in (mk or [])}
    rows = []
    for g in games:
        gid, gs = str(g["id"]), g.get("gameState")
        et = event_ticker(g)
        h, a = g["homeTeam"], g["awayTeam"]
        htk = f"{et}-{KCODE.get(h['abbrev'], h['abbrev'])}"
        atk = f"{et}-{KCODE.get(a['abbrev'], a['abbrev'])}"
        if gs in FINAL_STATES:
            for tk in (htk, atk):
                if tk in pend:
                    pend[tk]["final"] = True
            if gid not in finals and gid in anchors:
                finals[gid] = iso(t_nhl)
                rows.append({"kind": "final", "decided_at": iso(t_nhl), "gid": gid, "event_ticker": et,
                             "game_date": g["gameDate"], "home": h["abbrev"], "away": a["abbrev"],
                             "home_score": h.get("score"), "away_score": a.get("score"),
                             "period_type": (g.get("periodDescriptor") or {}).get("periodType")})
            continue
        if mk is None:
            continue
        hm, am = by_ticker.get(htk), by_ticker.get(atk)
        if hm is None or am is None:
            if gs in LIVE_STATES:
                print(f"  no Kalshi pair for {gid} {et} (have home={hm is not None} away={am is not None})")
            continue
        f = lambda m, k: float(m.get(k) or 0)  # noqa: E731
        hb, ha, ab, aa = f(hm, "yes_bid_dollars"), f(hm, "yes_ask_dollars"), f(am, "yes_bid_dollars"), f(am, "yes_ask_dollars")
        start = epoch(g["startTimeUTC"])
        if gs in ("FUT", "PRE"):
            if 0 < hb < ha < 1 and ha - hb <= 0.05 and now.timestamp() >= start - PRE_WIN:
                anchors[gid] = {"pm": (hb + ha) / 2, "as_of": iso(t_k), "src": "live_pre"}
            continue
        if gs not in LIVE_STATES:
            continue
        if gid not in anchors:
            try:
                an = candle_anchor(htk, start, get)
            except Exception as ex:  # noqa: BLE001
                an, rc = None, 1
                print(f"  candle anchor error {htk}: {type(ex).__name__}: {ex}")
            if an:
                anchors[gid] = an
        an = anchors.get(gid)
        margin = int(h.get("score") or 0) - int(a.get("score") or 0)
        left = minutes_left(g)
        p = None
        if an and left is not None:
            p = model_p(an["pm"], margin, left, prior)
        pdsc = g.get("periodDescriptor") or {}
        clk = g.get("clock") or {}
        decided = max(t_nhl, t_k)
        rows.append({
            "kind": "tick", "decided_at": iso(decided),
            "as_of": {"nhl": iso(t_nhl), "kalshi": iso(t_k), "anchor": an["as_of"] if an else None},
            "gid": gid, "game_date": g["gameDate"], "event_ticker": et, "home": h["abbrev"], "away": a["abbrev"],
            "home_ticker": htk, "away_ticker": atk, "start_utc": g["startTimeUTC"], "state": gs,
            "period": g.get("period"), "period_type": pdsc.get("periodType"),
            "sec_remaining": clk.get("secondsRemaining"), "running": clk.get("running"),
            "intermission": clk.get("inIntermission"), "minutes_left": left,
            "home_score": h.get("score"), "away_score": a.get("score"), "margin": margin,
            "p_pre": an["pm"] if an else None, "anchor_src": an["src"] if an else None,
            "p_model": None if p is None else round(p, 5),
            "home_bid": hb, "home_ask": ha, "away_bid": ab, "away_ask": aa,
            "home_bid_size": hm.get("yes_bid_size_fp"), "home_ask_size": hm.get("yes_ask_size_fp"),
            "away_bid_size": am.get("yes_bid_size_fp"), "away_ask_size": am.get("yes_ask_size_fp"),
            "signal": signal(p, hb, ha, ab, aa)})
        for tk in (htk, atk):
            pend.setdefault(tk, {})["last"] = now.timestamp()
    append(OUT, rows)
    try:
        n_set = settle(st, now, get)
    except Exception as ex:  # noqa: BLE001
        n_set, rc = 0, 1
        print(f"  settle error: {type(ex).__name__}: {ex}")
    # forget anchors of games more than 2 days old
    keep = {str(g["id"]) for g in games}
    for k in [k for k in anchors if k not in keep and k not in {r.get("gid") for r in rows}]:
        if now.timestamp() - datetime.fromisoformat(anchors[k]["as_of"]).timestamp() > 2 * 86400:
            anchors.pop(k)
            finals.pop(k, None)
    save_state(st)
    tk_rows = [r for r in rows if r["kind"] == "tick"]
    live = [f"{r['away']}@{r['home']} P{r['period']} {r['minutes_left']} m{r['margin']:+d} p={r['p_model']} "
            f"hb/ha={r['home_bid']:.2f}/{r['home_ask']:.2f}" for r in tk_rows]
    print(f"{iso(now)} logged {len(tk_rows)} live games, {sum(r['signal'] is not None for r in tk_rows)} paper "
          f"signals, {sum(r['kind'] == 'final' for r in rows)} finals, {n_set} settled"
          + ("; " + " | ".join(live) if live else ""))
    return rc


def report() -> int:
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()] if OUT.exists() else []
    res = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()] if RESULTS.exists() else []
    from collections import Counter
    c = Counter((r.get("game_date"), r.get("kind")) for r in rows)
    for k in sorted(c, key=str):
        print(k, c[k])
    print(f"games {len({r['gid'] for r in rows})}, settled tickers {len(res)}")
    return 0


def main() -> int:
    a = sys.argv[1:]
    if a[:1] == ["loop"]:
        sec = int(a[a.index("--sec") + 1]) if "--sec" in a else 60
        while True:
            try:
                tick()
            except Exception as ex:  # noqa: BLE001
                print("tick error", ex)
            time.sleep(sec)
    if a[:1] == ["report"]:
        return report()
    return tick()


if __name__ == "__main__":
    sys.exit(main())
