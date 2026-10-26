"""L28 NBA big-lead FORWARD PAPER logger. Paper only: NO orders are ever placed by this script.

    python scripts/nba_bigleads_logger.py tick      # launchd every 120 s (cheap exit when no NBA game is live)
    python scripts/nba_bigleads_logger.py report    # n, wins, losses, P&L/contract after fee, vs history & break-even

PRE-REGISTRATION (frozen 2026-10-25, written before the first tick; any change = a new trial with a new name)
---------------------------------------------------------------------------------------------------------------
Rule (L28): at a tick, a live NBA game QUALIFIES when the leader is
    (a) up >= 25 points in Q3 (any clock in Q3), or
    (b) up >= 20 points in Q4 with game clock > 6:00 (strictly more than 360 s) left.
    OT never qualifies. Game state = ESPN live scoreboard (period, status.clock, scores) at the tick.
Entry: BUY the leader's KXNBAGAME YES (KXNBAGAME-<ET date><AWAY><HOME>-<LEADER>) at the Kalshi YES ask read at that
    tick (market endpoint; depth from the orderbook), if and only if 0 < ask <= 0.96 and the market is active. ONE entry per game: the first
    qualifying tick with ask <= 0.96. Flat 1 contract. Hold to settlement. No exits, no adds.
Fee (primary): Kalshi taker 0.07 * p * (1-p) per contract, unrounded (prod fills showed ~exact fees).
    Sensitivity: cent-rounded-up fee on a 1-lot order, ceil(100 * 0.07 p (1-p)) / 100 (= $0.01 at every ask here).
Settlement: from ESPN final (status completed, competitor winner); Kalshi market `result` recorded when available as a
    cross-check. P&L/contract = (1 if leader won else 0) - ask - fee.
Recorded at entry: ask, bid, depth (contracts at the best YES ask), fee, period/clock/scores/lead, ESPN in-game home WP,
    pregame spread/total/moneylines (ESPN pickcenter, first provider) when available.
Also logged (measurement, not bets): EVERY qualifying-state tick where the ask is > 0.96 or missing (no quote / no
    market), to measure how often the price is too high to buy; and qualifying ticks after the game's entry are not logged.
Report: n entries, wins, losses, total and mean P&L per contract after fee (primary and rounded-fee sensitivity),
    realised win rate vs mean break-even (ask + fee) and vs the empirical historical rate for these states, 99.3-99.9%
    (notes/nba_big_lead.md; unconditional on team strength), exact binomial P(>= observed losses | p = break-even win
    rate). Too-high ticks: count, distinct games, distinct game-qualifications without an entry.
Scoring: descriptive forward paper. No significance claim is possible until n is in the hundreds (a 1-pt edge at 95c
    needs ~3,700 bets); the first read-out is a sanity check of fill availability and P&L sign, not a pass/fail.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("BIGLEADS_DATA", ROOT / "data"))
ENTRIES = DATA / "nba_bigleads_entries.jsonl"
TOOHIGH = DATA / "nba_bigleads_toohigh.jsonl"
SETTLED = DATA / "nba_bigleads_settled.jsonl"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
ET = ZoneInfo("America/New_York")
ESPN2K = {"SA": "SAS", "GS": "GSW", "NY": "NYK", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
MAX_ASK = 0.96
HIST = (0.993, 0.999)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def fee(p: float) -> float:
    return 0.07 * p * (1 - p)


def fee_rounded(p: float) -> float:
    return math.ceil(round(100 * fee(p), 9)) / 100


def qualifies(period: int, clock_s: float, lead: int) -> bool:
    return (period == 3 and lead >= 25) or (period == 4 and lead >= 20 and clock_s > 360)


def get(url: str, **params) -> dict:
    r = requests.get(url, params=params or None, timeout=15)
    r.raise_for_status()
    return r.json()


def game_state(e: dict) -> dict:
    c = e["competitions"][0]
    st = c["status"]
    clock = st.get("clock")
    if clock is None:  # fall back to displayClock "m:ss" or "ss.s"
        d = st.get("displayClock", "0:00")
        mm, ss = d.split(":") if ":" in d else ("0", d)
        clock = float(mm) * 60 + float(ss)
    t = {x["homeAway"]: x for x in c["competitors"]}
    code = lambda s: ESPN2K.get(t[s]["team"]["abbreviation"], t[s]["team"]["abbreviation"])
    hs, as_ = int(t["home"].get("score") or 0), int(t["away"].get("score") or 0)
    start = datetime.fromisoformat(e["date"].replace("Z", "+00:00")).astimezone(ET)
    return {"eid": e["id"], "game": e.get("shortName"), "state": st["type"]["state"],
            "completed": bool(st["type"].get("completed")), "period": int(st.get("period") or 0),
            "clock_s": float(clock), "home": code("home"), "away": code("away"), "hs": hs, "as": as_,
            "winner": next((code(s) for s in ("home", "away") if t[s].get("winner")), None),
            "event": f"KXNBAGAME-{start.strftime('%y%b%d').upper()}{code('away')}{code('home')}"}


def kalshi_quote(ticker: str) -> dict:
    """Leader's YES ask/bid from the market endpoint + contracts at the best YES ask (= best NO bid level)."""
    try:
        m = get(f"{KALSHI}/markets/{ticker}")["market"]
    except Exception as ex:  # noqa: BLE001
        return {"ticker": ticker, "ask": None, "err": f"market: {ex}"[:200]}
    f = lambda k: float(m[k]) if m.get(k) not in (None, "") else None
    q = {"ticker": ticker, "ask": f("yes_ask_dollars"), "bid": f("yes_bid_dollars"), "status": m.get("status"),
         "result": m.get("result") or None}
    try:
        ob = get(f"{KALSHI}/markets/{ticker}/orderbook")["orderbook_fp"]
        no = [(float(p), float(n)) for p, n in (ob.get("no_dollars") or [])]
        if no:
            bp, bn = max(no)
            q["ob_ask"], q["depth_at_ask"] = round(1 - bp, 4), bn
            q["depth_within_1c"] = sum(n for p, n in no if p >= bp - 0.01 + 1e-9)
    except Exception as ex:  # noqa: BLE001
        q["ob_err"] = str(ex)[:200]
    return q


def pregame(eid: str) -> dict:
    try:
        s = get(f"{ESPN}/summary", event=eid)
    except Exception as ex:  # noqa: BLE001
        return {"pregame_err": str(ex)[:200]}
    out = {}
    pc = (s.get("pickcenter") or [{}])[0]
    if pc:
        out = {"spread_details": pc.get("details"), "home_spread": pc.get("spread"), "total": pc.get("overUnder"),
               "ml_home": (pc.get("homeTeamOdds") or {}).get("moneyLine"),
               "ml_away": (pc.get("awayTeamOdds") or {}).get("moneyLine"),
               "provider": (pc.get("provider") or {}).get("name")}
    wp = s.get("winprobability") or []
    if wp:
        out["espn_home_wp"] = wp[-1].get("homeWinPercentage")
    return out


def tick(mock: str | None = None) -> int:
    entries = jl(ENTRIES)
    settled = {r["eid"] for r in jl(SETTLED)}
    open_ = [r for r in entries if r["eid"] not in settled]
    h = datetime.now(ET).hour
    if not mock and not open_ and 3 <= h < 11:  # cheap exit: no NBA game is live 03:00-11:00 ET
        return 0
    sb = json.loads(Path(mock).read_text()) if mock else get(f"{ESPN}/scoreboard")
    games = {g["eid"]: g for g in map(game_state, sb.get("events", []))}
    entered = {r["eid"] for r in entries}
    msgs = []
    for g in games.values():
        if g["state"] != "in" or g["eid"] in entered:
            continue
        lead = abs(g["hs"] - g["as"])
        if not qualifies(g["period"], g["clock_s"], lead):
            continue
        leader = g["home"] if g["hs"] > g["as"] else g["away"]
        q = kalshi_quote(f"{g['event']}-{leader}")
        row = {"ts": now(), "eid": g["eid"], "game": g["game"], "period": g["period"], "clock_s": g["clock_s"],
               "home": g["home"], "away": g["away"], "hs": g["hs"], "as": g["as"], "lead": lead, "leader": leader,
               "leader_home": leader == g["home"], **q}
        ask = q.get("ask")
        if ask is not None and 0 < ask <= MAX_ASK and q.get("status") in (None, "active"):
            row.update({"fee": round(fee(ask), 5), "fee_rounded": fee_rounded(ask),
                        "breakeven": round(ask + fee(ask), 5), "contracts": 1, **pregame(g["eid"])})
            append(ENTRIES, row)
            entered.add(g["eid"])
            msgs.append(f"ENTRY {g['game']} Q{g['period']} {g['clock_s']:.0f}s lead {lead} {leader} ask {ask:.2f} "
                        f"depth {q.get('depth_at_ask')}")
        else:
            append(TOOHIGH, row)
            msgs.append(f"too-high {g['game']} Q{g['period']} {g['clock_s']:.0f}s lead {lead} {leader} ask {ask}")
    # settle open entries from ESPN finals (scoreboard first, summary for games no longer on it)
    for r in entries:
        if r["eid"] in settled:
            continue
        g = games.get(r["eid"])
        if g is None:
            try:
                s = get(f"{ESPN}/summary", event=r["eid"])
                hc = s["header"]["competitions"][0]
                g = game_state({"id": r["eid"], "date": hc["date"], "shortName": r["game"], "competitions": [hc]})
            except Exception as ex:  # noqa: BLE001
                msgs.append(f"settle lookup failed {r['eid']}: {ex}")
                continue
        if not g["completed"] or g["winner"] is None:
            continue
        won = g["winner"] == r["leader"]
        kq = kalshi_quote(r["ticker"])
        pnl = (1.0 if won else 0.0) - r["ask"] - r["fee"]
        append(SETTLED, {"ts": now(), "eid": r["eid"], "game": r["game"], "leader": r["leader"], "won": won,
                         "final_home": g["hs"], "final_away": g["as"], "winner": g["winner"],
                         "kalshi_result": kq.get("result"), "pnl": round(pnl, 5),
                         "pnl_rounded_fee": round((1.0 if won else 0.0) - r["ask"] - r["fee_rounded"], 5)})
        settled.add(r["eid"])
        msgs.append(f"SETTLED {r['game']} {r['leader']} {'WON' if won else 'LOST'} pnl {pnl:+.4f}")
    if msgs:
        print("\n".join(msgs))
    return 0


def binom_tail_ge(k: int, n: int, p_loss: float) -> float:
    return sum(math.comb(n, i) * p_loss**i * (1 - p_loss) ** (n - i) for i in range(k, n + 1))


def report() -> int:
    entries = {r["eid"]: r for r in jl(ENTRIES)}
    st = {r["eid"]: r for r in jl(SETTLED)}
    th = jl(TOOHIGH)
    done = [(entries[e], st[e]) for e in st if e in entries]
    n, w = len(done), sum(s["won"] for _, s in done)
    print(f"L28 NBA big-lead forward paper (rule frozen 2026-10-25; ask <= {MAX_ASK}, 1 contract)")
    print(f"entries {len(entries)} | settled {n} | open {len(entries) - n}")
    if n:
        pnl = sum(s["pnl"] for _, s in done)
        pnl_r = sum(s["pnl_rounded_fee"] for _, s in done)
        be = sum(e["breakeven"] for e, _ in done) / n
        ask = sum(e["ask"] for e, _ in done) / n
        # P(>= observed losses) if every bet's true win prob equalled its break-even (mean BE approx)
        p = binom_tail_ge(n - w, n, 1 - be)
        print(f"wins {w} | losses {n - w} | win rate {w / n:.4f} | mean ask {ask:.4f} | mean break-even {be:.4f}")
        print(f"P&L after fee: total ${pnl:+.4f}, per contract ${pnl / n:+.4f}  "
              f"(cent-rounded fee: total ${pnl_r:+.4f}, per contract ${pnl_r / n:+.4f})")
        print(f"vs historical {HIST[0]:.3f}-{HIST[1]:.3f}: realised - break-even {100 * (w / n - be):+.2f} pts; "
              f"historical - break-even {100 * (HIST[0] - be):+.2f}..{100 * (HIST[1] - be):+.2f} pts")
        print(f"exact P(losses >= {n - w} | win p = break-even) = {p:.3f}  (descriptive; n far below ~3,700 needed)")
        for e, s in done:
            print(f"  {e['ts'][:16]} {e['game']:<12} Q{e['period']} {e['clock_s']:>5.0f}s lead {e['lead']:>2} "
                  f"{e['leader']} ask {e['ask']:.2f} depth {e.get('depth_at_ask')} -> {'W' if s['won'] else 'L'} "
                  f"{s['pnl']:+.4f}")
    g_th = {r["eid"] for r in th}
    never = g_th - set(entries)
    asks = [r["ask"] for r in th if r.get("ask") is not None]
    print(f"too-high/no-quote qualifying ticks: {len(th)} over {len(g_th)} games; "
          f"{len(never)} qualifying games never got an entry (ask stayed > {MAX_ASK} or missing); "
          f"no-quote ticks {sum(r.get('ask') is None for r in th)}"
          + (f"; median too-high ask {sorted(asks)[len(asks) // 2]:.2f}" if asks else ""))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "tick":
        mock = sys.argv[sys.argv.index("--mock") + 1] if "--mock" in sys.argv else None
        raise SystemExit(tick(mock))
    raise SystemExit(report())
