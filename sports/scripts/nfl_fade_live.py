"""NFL post-big-play maker fade -- LIVE paper logger (F1L, a real-time PROXY of F1). Paper only; hands signals to the
deployed executor (ingame_exec.py, strategy NFLF), which is DRY unless .env arms LEAD_LIVE_NFLF=1.

    python scripts/nfl_fade_live.py tick      # launchd quant-sports-nfl-fade every 60 s: 15 s sub-ticks for ~50 s
    python scripts/nfl_fade_live.py grade     # F1 candle mechanics on signals >= 10 min old (also run by tick)
    python scripts/nfl_fade_live.py report

Why a proxy: F1 (scripts/nfl_ingame_fade.py, pre-registered 10-20, forward 10-22..11-30, scored once after 12-02) is
RETROSPECTIVE -- big plays come from nflverse vegas_home_wp, which exists only after the nightly pbp rebuild, and
fills from 1-min candles. Nothing can mirror it in real time. This logger applies the same rule to what is
observable live:
  big play : |ESPN homeWinPercentage(play i) - (play i-1)| >= 0.08 (ESPN summary `winprobability`, post-play WP;
             nflverse's vegas_home_wp(next) - (this) is the same play's effect), t = ESPN `wallclock` of play i.
  side     : the team whose WP fell. Market KXNFLGAME-<yyMMMdd ET><AWAY><HOME>-<SIDE> (ESPN JAX->JAC, WSH->WAS).
  entry    : first sub-tick with t+60 <= now <= t+120: rest a YES bid at the live YES bid if spread <= 5c and
             0.03 <= bid <= 0.97 (else `no_quote`); seen only after t+120 -> `late` (ESPN lag; logged, not traded).
  window   : the executor rests post-only until t+240 (exchange expiration_time), exits at the bid at t+480.
Paper record (`grade` rows): F1's own candle rule on each signal -- entry bid of the first candle ending >= t+60,
fill iff a candle ending in (entry, t+240] has yes_ask low <= bid - 1c, exit at the bid of the first candle ending
>= t+480, P&L (x - 0.07x(1-x)) - (b + 0.0175b(1-b)); also the same fill test at the live signal price.
NOT the pre-registered F1 and NOT an edgelab trial: agreement with F1 (same plays? same prices?) must be checked
against `nfl_ingame_fade.py interim` before trusting it.

Output (ROOT/data/live/): nfl_fade_live.jsonl (kinds big_play / signal / no_quote / late / grade),
nfl_fade_live_state.json (schedule cache, seen plays).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

KAPI = "https://api.elections.kalshi.com/trade-api/v2"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
LIVE = ROOT / "data" / "live"
OUT = LIVE / "nfl_fade_live.jsonl"
STATE = LIVE / "nfl_fade_live_state.json"
ET = ZoneInfo("America/New_York")
KCODE = {"JAX": "JAC", "WSH": "WAS"}          # ESPN -> Kalshi (checked 2026-11-07 against open KXNFLGAME markets)
BIG, ENTRY_LO, ENTRY_HI, SPREAD, PMIN, PMAX = 0.08, 60, 120, 0.05, 0.03, 0.97
FILL_S, EXIT_S = 240, 480
PRE_WIN, POST_WIN = 15 * 60, 5 * 3600
SUB_S, RUN_S = 15, 50


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


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text()) if STATE.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st))
    os.replace(tmp, STATE)


def append(rows: list[dict]) -> None:
    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def games(sb: dict) -> list[dict]:
    out = []
    for e in sb.get("events") or []:
        c = (e.get("competitions") or [{}])[0]
        teams = {x["homeAway"]: x["team"]["abbreviation"] for x in c.get("competitors") or []}
        if set(teams) != {"home", "away"}:
            continue
        start = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
        d = start.astimezone(ET)
        k = lambda t: KCODE.get(t, t)  # noqa: E731
        ev = f"KXNFLGAME-{d:%y}{d.strftime('%b').upper()}{d:%d}{k(teams['away'])}{k(teams['home'])}"
        out.append({"id": str(e["id"]), "home": teams["home"], "away": teams["away"], "start": int(start.timestamp()),
                    "state": ((e.get("status") or {}).get("type") or {}).get("state"), "event_ticker": ev,
                    "game_day": d.date().isoformat(),
                    "tickers": {"home": f"{ev}-{k(teams['home'])}", "away": f"{ev}-{k(teams['away'])}"}})
    return out


def big_plays(summary: dict) -> list[dict]:
    """Consecutive ESPN WP pairs with |delta| >= BIG; t = wallclock of the later play (the play that moved WP)."""
    plays = {}
    dr = summary.get("drives") or {}
    for d in (dr.get("previous") or []) + ([dr["current"]] if isinstance(dr.get("current"), dict) else []):
        for p in d.get("plays") or []:
            plays[str(p.get("id"))] = p
    wp = summary.get("winprobability") or []
    out = []
    for a, b in zip(wp, wp[1:]):
        d = float(b["homeWinPercentage"]) - float(a["homeWinPercentage"])
        p = plays.get(str(b.get("playId")))
        if abs(d) >= BIG and p and p.get("wallclock"):
            out.append({"play_id": str(b["playId"]), "t": epoch(p["wallclock"]), "wp_delta": round(d, 4),
                        "wp_after": float(b["homeWinPercentage"]), "side": "away" if d > 0 else "home",
                        "text": str(p.get("text", ""))[:120]})
    return out


def subtick(st: dict, get, clock) -> tuple[list[dict], list[dict], int]:
    """One pass: new big plays -> pending; pending plays in [t+60, t+120] -> signal (live YES bid)."""
    rows, sigs, rc = [], [], 0
    try:
        gs = games(get(f"{ESPN}/scoreboard"))
    except Exception as ex:  # noqa: BLE001
        print(f"  espn scoreboard error: {type(ex).__name__}: {ex}")
        return rows, sigs, 1
    seen = st.setdefault("seen", {})
    for g in gs:
        if g["state"] != "in":
            continue
        try:
            sm = get(f"{ESPN}/summary", {"event": g["id"]})
            t_espn = clock()
        except Exception as ex:  # noqa: BLE001
            print(f"  espn summary error {g['id']}: {type(ex).__name__}: {ex}")
            rc = 1
            continue
        for bp in big_plays(sm):
            key = f"NFLF|{g['id']}|{bp['play_id']}"
            if key in seen:
                continue
            seen[key] = {"t": bp["t"], "side": bp["side"], "status": "pending", "gid": g["id"],
                         "ticker": g["tickers"][bp["side"]], "game_day": g["game_day"], "wp_delta": bp["wp_delta"]}
            rows.append({"kind": "big_play", "key": key, "seen_at": iso(t_espn), "gid": g["id"], "home": g["home"],
                         "away": g["away"], "t_play": bp["t"], "detect_lag_s": int(t_espn.timestamp()) - bp["t"],
                         **{k: bp[k] for k in ("play_id", "wp_delta", "wp_after", "side", "text")},
                         "ticker": g["tickers"][bp["side"]]})
    for key, s in seen.items():
        if s["status"] != "pending":
            continue
        el = clock().timestamp() - s["t"]
        if el < ENTRY_LO:
            continue
        if el > ENTRY_HI:
            s["status"] = "late"
            rows.append({"kind": "late", "key": key, "at": iso(clock()), "elapsed_s": round(el, 1)})
            continue
        try:
            m = get(f"{KAPI}/markets/{s['ticker']}").get("market") or {}
            t_k = clock()
        except Exception as ex:  # noqa: BLE001
            print(f"  kalshi error {s['ticker']}: {type(ex).__name__}: {ex}")
            rc = 1
            continue
        b, a = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
        base = {"key": key, "decided_at": iso(t_k), "ticker": s["ticker"], "bid": b, "ask": a,
                "elapsed_s": round(t_k.timestamp() - s["t"], 1), "t_play": s["t"]}
        if m.get("status") not in ("active", "open") or not (a - b <= SPREAD + 1e-9 and PMIN <= b <= PMAX and b < a):
            s["status"] = "no_quote"
            rows.append({"kind": "no_quote", **base, "market_status": m.get("status")})
            continue
        s["status"] = "signal"
        sig = {"strategy": "NFLF", "key": key, "ticker": s["ticker"], "side": "yes", "price": b,
               "decided_at": iso(t_k), "t_play": s["t"], "play_id": key.rsplit("|", 1)[1], "gid": s["gid"],
               "wp_delta": s["wp_delta"], "game_day": s["game_day"], "source": "nfl_fade_live.jsonl", "source_row": key}
        rows.append({"kind": "signal", **base, "wp_delta": s["wp_delta"], "game_day": s["game_day"]})
        sigs.append(sig)
    return rows, sigs, rc


def candles(tk: str, s: int, e: int, get=_get) -> list[dict]:
    j = get(f"{KAPI}/series/KXNFLGAME/markets/{tk}/candlesticks", {"start_ts": s, "end_ts": e, "period_interval": 1})
    f = lambda c, k, f2: float((c.get(k) or {}).get(f2 + "_dollars") or 0)  # noqa: E731
    return sorted(({"end": int(c["end_period_ts"]), "bid": f(c, "yes_bid", "close"), "ask": f(c, "yes_ask", "close"),
                    "ask_low": f(c, "yes_ask", "low")} for c in j.get("candlesticks") or []), key=lambda c: c["end"])


def grade_one(sig: dict, cs: list[dict]) -> dict:
    """F1's candle rule (nfl_ingame_fade.bets) on one live signal, plus the same fill test at the live price."""
    t = int(sig["t_play"])
    i = next((c for c in cs if c["end"] >= t + 60), None)
    j = next((c for c in cs if c["end"] >= t + EXIT_S), None)
    out = {"kind": "grade", "key": sig["key"], "ticker": sig["ticker"], "live_price": sig["bid"]}
    if i is None or j is None:
        return {**out, "f1": "no_candles"}
    win = [c for c in cs if i["end"] < c["end"] <= t + FILL_S]
    b, x = i["bid"], j["bid"]
    ok = i["ask"] - i["bid"] <= SPREAD + 1e-9 and PMIN <= b <= PMAX
    fill = ok and any(c["ask_low"] and c["ask_low"] <= b - 0.01 + 1e-9 for c in win)
    lp = sig["bid"]
    live_fill = any(c["ask_low"] and c["ask_low"] <= lp - 0.01 + 1e-9 and c["end"] <= t + FILL_S
                    for c in cs if c["end"] > int(datetime.fromisoformat(sig["decided_at"]).timestamp()))
    pnl = lambda e: (x - 0.07 * x * (1 - x)) - (e + 0.0175 * e * (1 - e))  # noqa: E731
    return {**out, "f1": "fill" if fill else ("no_fill" if ok else "no_quote"), "f1_entry": b, "exit": x,
            "f1_pnl": round(pnl(b), 4) if fill else None, "live_fill": live_fill,
            "live_pnl": round(pnl(lp), 4) if live_fill else None}


def grade(st: dict, now: datetime, get=_get) -> int:
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()] if OUT.exists() else []
    done = {r["key"] for r in rows if r.get("kind") == "grade"}
    out = []
    for s in (r for r in rows if r.get("kind") == "signal" and r["key"] not in done):
        if now.timestamp() < s["t_play"] + EXIT_S + 300:
            continue
        try:
            out.append({**grade_one(s, candles(s["ticker"], s["t_play"] - 120, s["t_play"] + EXIT_S + 180, get)),
                        "graded_at": iso(now)})
        except Exception as ex:  # noqa: BLE001
            print(f"  grade error {s['key']}: {type(ex).__name__}: {ex}")
        time.sleep(0.4 if get is _get else 0)            # candlesticks 429 at ~5 req/s (seen 11-07)
    append(out)
    return len(out)


def active(st: dict, now: datetime) -> bool | None:
    sc = st.get("sched")
    if not sc or now.timestamp() - sc.get("fetched", 0) > 3 * 3600:
        return None
    t = now.timestamp()
    return any(s - PRE_WIN <= t <= s + POST_WIN for s in sc.get("starts", []))


def tick(get=_get, now: datetime | None = None, clock=now_utc, sleep=time.sleep, hook=None) -> int:
    from nhl_ingame_logger import exec_hook
    hook = hook or exec_hook
    now = now or clock()
    st = load_state()
    act = active(st, now)
    if act is None:
        try:
            gs = games(get(f"{ESPN}/scoreboard"))
        except Exception as ex:  # noqa: BLE001
            print(f"{iso(now)} espn scoreboard error: {type(ex).__name__}: {ex}")
            return 1
        st["sched"] = {"fetched": now.timestamp(), "starts": sorted({g["start"] for g in gs})}
        act = active(st, now)
    if not act and not any(s["status"] == "pending" for s in st.get("seen", {}).values()):
        save_state(st)
        print(f"{iso(now)} idle: no NFL game window")
        return 0
    rc, n_rows, n_sig, t0 = 0, 0, 0, clock()
    while True:
        rows, sigs, r = subtick(st, get, clock)
        rc = max(rc, r)
        append(rows)
        save_state(st)
        rc = max(rc, hook(sigs))                 # every sub-tick: entries now, cancels / exits for older ones
        n_rows, n_sig = n_rows + len(rows), n_sig + len(sigs)
        if (clock() - t0).total_seconds() + SUB_S > RUN_S:
            break
        sleep(SUB_S)
    n_g = grade(st, clock(), get)
    # forget plays older than 2 days
    cut = clock().timestamp() - 2 * 86400
    st["seen"] = {k: v for k, v in st.get("seen", {}).items() if v["t"] >= cut}
    save_state(st)
    print(f"{iso(now)} rows {n_rows}, signals {n_sig}, graded {n_g}")
    return rc


def report() -> int:
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()] if OUT.exists() else []
    from collections import Counter
    print(Counter(r["kind"] for r in rows))
    g = [r for r in rows if r.get("kind") == "grade"]
    f1 = [r["f1_pnl"] for r in g if r.get("f1_pnl") is not None]
    lv = [r["live_pnl"] for r in g if r.get("live_pnl") is not None]
    print(f"graded {len(g)}: F1-mechanics fills {len(f1)} P&L/contract {sum(f1):+.3f}; live-price fills {len(lv)} "
          f"{sum(lv):+.3f}")
    lags = [r["detect_lag_s"] for r in rows if r.get("kind") == "big_play"]
    if lags:
        lags.sort()
        print(f"ESPN detect lag s: median {lags[len(lags) // 2]}, max {lags[-1]} (n {len(lags)})")
    return 0


def main() -> int:
    a = sys.argv[1:]
    if a[:1] == ["grade"]:
        st = load_state()
        n = grade(st, now_utc())
        print(f"graded {n}")
        return 0
    if a[:1] == ["report"]:
        return report()
    return tick()


if __name__ == "__main__":
    sys.exit(main())
