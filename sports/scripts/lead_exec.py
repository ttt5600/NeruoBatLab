"""Kalshi executor for the TIER B judgment-size leads in notes/ACT_LIST.md (B1 L39, B2 DUL, B4 TSA). DRY BY DEFAULT.

    python scripts/lead_exec.py run            # dry: decide every fresh signal against the live book, log only
    python scripts/lead_exec.py run --live     # orders ONLY if .env has EXEC_MODE=live AND LEAD_EXEC_MODE=live
    python scripts/lead_exec.py status         # per-strategy spend vs caps, open, W/L, P&L, CLV, kill state
    python scripts/lead_exec.py settle         # settle filled rows from market results (+ DUL closing-line value)
    python scripts/lead_exec.py kill [reason]  # write data/LEAD_KILL, cancel this executor's resting orders

These are LEADS, not edgelab passes: each one is a forward test at fixed small stakes with its kill rule written in
ACT_LIST.md (11-05) before the first bet. The table below is FROZEN from that file; change it there first.

  lead_L39  B1 monthly rain/snow "unreachable" NO, MAKER. Mirror each NEW pick in kalshi_wx_monthly_picks.jsonl
            (deployed L39 logger) as a post-only NO bid at the pick's maker_q, resting to market close, no taker
            fallback. 1-3 contracts, <= $3/order, <= $15 per calendar month (month of the pick's decision day d).
            Kill: 2 losses within the first 100 settled fills.
  lead_TSA  B4 KXTSAW weekly MAKER (tsa_t2_maker paper rule). Mirror each NEW paper post in e1x/tsa_paper.jsonl as a
            real post-only bid on the SAME side at the same q (the logger posts NO above its band and YES below it).
            contracts = max(1, floor($1 / (q + fee))), <= $8 per KXTSAW event (= week). Kill: 2 losses in first 30 fills.
  lead_DUL  B2 NFL division-game UNDER, 2026 REG week >= 10. Kalshi series KXNFLTOTAL, event
            KXNFLTOTAL-<yyMMMdd ET><AWAY><HOME> (Kalshi codes: JAX->JAC, LA->LAR), markets "over X.5" (greater).
            90..30 min before kickoff: strike whose YES mid is closest to 0.50 -> BUY NO (= under) taker IOC at the
            NO ask, only if NO ask <= 0.56. $2 stake (contracts = floor(2 / (ask + taker fee))), one bet per game,
            <= $60 for the season. Kill: once >= 40 settled, ROI < 0 or mean CLV < 0 (CLV = last NO trade price
            before kickoff - our NO price).
  global    lead open exposure <= $40; Kalshi shard-0 cash after the order must stay >= $80 (weather executor room).

Safety: mode = kalshi_exec.resolve_mode() AND .env LEAD_EXEC_MODE=live (so arming the weather executor never arms
this one). data/LEAD_KILL or kalshi_exec's KILL stops everything; data/LEAD_DISABLE_<L39|TSA|DUL> stops one strategy
(written automatically by its kill rule in demo/live; in dry the rule only prints). One decision per (mode, ticker)
ever, written to data/lead_exec.jsonl BEFORE the POST; client_order_id = uuid5(lead_exec/strategy/mode/ticker), so a
re-send is a 409, never a second order; the exchange is asked for existing orders/positions on the ticker first.
Maker fills more than 2c from q -> LEAD_KILL (price convention check). Reuses kalshi_exec's client, .env, fees,
notify. kalshi_exec does not write bet.py's bets.jsonl, so neither does this (bet.py's B+C caps do NOT see these).
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kalshi_exec as kx  # noqa: E402

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------- FROZEN (notes/ACT_LIST.md tier B, 11-05)
STRATS = {
    "L39": {"tag": "lead_L39", "act": "B1", "style": "maker", "max_order": 3.00, "max_contracts": 3,
            "period": "month", "period_cap": 15.00, "max_age_h": 3.0, "kill_first": 100, "kill_losses": 2},
    "TSA": {"tag": "lead_TSA", "act": "B4", "style": "maker", "max_order": 1.00, "max_contracts": None,
            "period": "week", "period_cap": 8.00, "max_age_h": 24.0, "kill_first": 30, "kill_losses": 2},
    "DUL": {"tag": "lead_DUL", "act": "B2", "style": "taker", "max_order": 2.00, "max_contracts": None,
            "period": "season", "period_cap": 60.00, "season": 2026, "min_week": 10, "max_no_ask": 0.56,
            "win_lo_min": 30, "win_hi_min": 90, "judge_after": 40},
}
GLOBAL_OPEN_CAP = 40.00
CASH_RESERVE = 80.00
DEPTH_FRAC = 0.5                 # DUL taker: <= half the NO-ask depth within 1c (as kalshi_exec)
CONVENTION_TOL = 0.02
IMPROVE_TOL = 0.10               # paying well under the limit is normal; this far under suggests a YES/NO price mix-up
KICK_TOL_MIN = 20                # Kalshi expected_expiration_time - 3h must match the schedule kickoff
NFL_SERIES = "KXNFLTOTAL"
KALSHI_TEAM = {"JAX": "JAC", "LA": "LAR"}  # nflverse -> Kalshi codes (others identical; checked 11-05 week 9 events)
ZERO = ("no_fill", "rejected", "rejected_post_only", "not_placed")
# In-game strategies executed by scripts/ingame_exec.py (11-07). Their rows share data/lead_exec.jsonl (keyed by `key`,
# not ticker: two NHL sides / several NFL fades can share one market) but ingame_exec.py reconciles, settles, caps and
# kills them; the functions below skip them so the L39/TSA/DUL accounting is unchanged.
INGAME = ("NHL", "NFLF")

# sources (module globals so tests can point them elsewhere)
L39_PICKS = Path(os.environ.get("LEAD_L39_PICKS") or kx.DEPLOY / "data" / "kalshi_wx_monthly_picks.jsonl")
TSA_PAPER = Path(os.environ.get("LEAD_TSA_PAPER")
                 or Path.home() / ".claude" / "quant-sports" / "data" / "e1x" / "tsa_paper.jsonl")
SCHEDULE = Path(os.environ["LEAD_NFL_SCHEDULE"]) if os.environ.get("LEAD_NFL_SCHEDULE") else None


def log_path() -> Path:
    return kx.DATA / "lead_exec.jsonl"


def lead_kill_path() -> Path:
    return kx.DATA / "LEAD_KILL"


def disable_path(s: str) -> Path:
    return kx.DATA / f"LEAD_DISABLE_{s}"


def schedule_path() -> Path:
    if SCHEDULE is not None:
        return SCHEDULE
    p = kx.DATA / "nfl_schedule.csv"
    return p if p.exists() else kx.DEPLOY / "data" / "nfl_schedule.csv"


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def ts(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def journal() -> list[dict]:
    return jl(log_path())


def append(rec: dict) -> dict:
    p = log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def lead_mode(e: dict, live_flag: bool) -> str:
    m = kx.resolve_mode(e, live_flag)
    return m if m != "dry" and e.get("LEAD_EXEC_MODE") == "live" else "dry"


def coid(strategy: str, mode: str, ticker: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"lead_exec/{strategy}/{mode}/{ticker}"))


# ---------------------------------------------------------------- journal folding / accounting
def fold(hist: list[dict], mode: str) -> dict:
    """ticker -> the ONE decision for that ticker in `mode`, with its updates and settle merged in."""
    out = {}
    for r in hist:
        if r.get("mode") != mode or "ticker" not in r:
            continue
        ev, t = r.get("event", "order"), r.get("key") or r["ticker"]
        if ev == "order" and (t not in out or (out[t].get("strategy") == "DUL"
                                               and out[t].get("status", "").startswith("skip"))):
            out[t] = dict(r)  # a DUL skip does not block a later decision on the same game inside its window
        elif ev in ("update", "settle") and t in out:
            out[t].update({k: v for k, v in r.items() if k not in ("event", "ts", "mode", "ticker")})
            if ev == "settle":
                out[t].update(settled=True, settled_ts=r["ts"])
    return out


def committed(o: dict) -> float:
    s = o.get("status", "")
    return 0.0 if s.startswith("skip") or s in ZERO else float(o.get("cost") or 0)


def held(o: dict) -> float:
    return float(o.get("filled") or 0)


def period_key(strategy: str, o: dict) -> str:
    if strategy == "L39":
        return str(o.get("pick_d") or o["ts"])[:7]
    if strategy == "TSA":
        return str(o.get("event_ticker") or o["ticker"].rsplit("-", 1)[0])
    return str(STRATS["DUL"]["season"])


def spent_in(state: dict, strategy: str, key: str) -> float:
    return sum(committed(o) for o in state.values() if o.get("strategy") == strategy and period_key(strategy, o) == key)


def open_exposure(state: dict) -> float:
    return sum(committed(o) for o in state.values() if not o.get("settled") and o.get("strategy") not in INGAME)


def settled_rows(state: dict, strategy: str) -> list[dict]:
    rows = [o for o in state.values() if o.get("strategy") == strategy and o.get("settled") and held(o) > 0]
    return sorted(rows, key=lambda o: o.get("first_fill_ts") or o["ts"])


def kill_verdict(state: dict, strategy: str) -> str | None:
    """The pre-set ACT_LIST kill rule on this mode's settled fills; None = keep going."""
    s, rows = STRATS[strategy], settled_rows(state, strategy)
    if strategy in ("L39", "TSA"):
        losses = sum(not o.get("won") for o in rows[: s["kill_first"]])
        if losses >= s["kill_losses"]:
            return f"{losses} losses within the first {s['kill_first']} settled fills"
        return None
    if len(rows) < s["judge_after"]:
        return None
    pnl, cost = sum(o["pnl"] for o in rows), sum(committed(o) for o in rows)
    clv = [o["clv"] for o in rows if o.get("clv") is not None]
    roi, mclv = (pnl / cost if cost else 0.0), (sum(clv) / len(clv) if clv else 0.0)
    if roi < 0 or mclv < 0:
        return f"after {len(rows)} settled: ROI {roi:+.2%}, mean CLV {mclv:+.4f} ({len(clv)} with CLV)"
    return None


# ---------------------------------------------------------------- exchange helpers
def shard0_cash(client: kx.Kalshi) -> float:
    """Shard-0 cash (weather, NFL and TSA markets all have exchange_index 0); falls back to balance_dollars."""
    out = client._ok(*client.request("GET", "/portfolio/balance"), "balance")
    for x in out.get("balance_breakdown") or []:
        if int(x.get("exchange_index", -1)) == 0:
            return float(x["balance"])
    return float(out.get("balance_dollars", 0))


def mkt_open(m: dict | None, now: datetime) -> bool:
    if not m or m.get("status") not in ("active", "open"):
        return False
    c = m.get("close_time")
    return c is None or ts(c) > now + timedelta(minutes=5)


# ---------------------------------------------------------------- signals
def l39_signals(now: datetime) -> list[dict]:
    out = []
    for p in jl(L39_PICKS):
        if p.get("result") is not None or p.get("off_schedule") or p.get("maker_q") is None:
            continue
        out.append({"strategy": "L39", "ticker": p["ticker"], "side": "no", "q": float(p["maker_q"]),
                    "source_ts": p["ts"], "pick_d": p.get("d"), "event_ticker": p.get("event"),
                    "src_age_h": round((now - ts(p["ts"])).total_seconds() / 3600, 2)})
    return out


def tsa_signals(now: datetime) -> list[dict]:
    out = []
    for p in jl(TSA_PAPER):
        if p.get("kind") != "post":
            continue
        out.append({"strategy": "TSA", "ticker": p["ticker"], "side": p["side"], "q": float(p["q"]),
                    "source_ts": p["ts"], "event_ticker": p.get("event"),
                    "src_age_h": round((now - ts(p["ts"])).total_seconds() / 3600, 2)})
    return out


def dul_games(now: datetime) -> list[dict]:
    """2026 REG division games, week >= 10, kicking off 30..90 min from now (schedule gametime is ET)."""
    s = STRATS["DUL"]
    p = schedule_path()
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for r in csv.DictReader(f):
            if (r.get("season") != str(s["season"]) or r.get("game_type") != "REG" or r.get("div_game") != "1"
                    or int(r.get("week") or 0) < s["min_week"] or not r.get("gametime")):
                continue
            kick = datetime.strptime(f"{r['gameday']} {r['gametime']}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
            mins = (kick - now).total_seconds() / 60
            if s["win_lo_min"] <= mins <= s["win_hi_min"]:
                out.append({"game_id": r["game_id"], "week": int(r["week"]), "away": r["away_team"],
                            "home": r["home_team"], "kickoff": iso(kick), "mins_to_kick": round(mins, 1)})
    return out


def nfl_event_ticker(g: dict) -> str:
    d = ts(g["kickoff"]).astimezone(ET)
    return (f"{NFL_SERIES}-{d:%y}{d.strftime('%b').upper()}{d:%d}"
            f"{KALSHI_TEAM.get(g['away'], g['away'])}{KALSHI_TEAM.get(g['home'], g['home'])}")


def nfl_markets(client: kx.Kalshi, event: str) -> list[dict]:
    code, out = client.request("GET", "/markets", params={"event_ticker": event, "limit": 200}, auth=False)
    return out.get("markets", []) if code == 200 else []


def pick_strike(ms: list[dict]) -> dict | None:
    """Two-sided active market whose YES mid is closest to 0.50 (ties: tighter spread, then lower strike)."""
    best = None
    for m in ms:
        yb, ya = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
        if m.get("status") not in ("active", "open") or yb <= 0 or ya >= 1 or ya < yb:
            continue
        key = (round(abs((yb + ya) / 2 - 0.5), 6), round(ya - yb, 6), float(m.get("floor_strike") or 0))
        if best is None or key < best[0]:
            best = (key, {**m, "_mid": round((yb + ya) / 2, 4)})
    return best[1] if best else None


# ---------------------------------------------------------------- decisions
def _base(sig: dict, mode: str, now: datetime) -> dict:
    s = STRATS[sig["strategy"]]
    return {"event": "order", "ts": iso(now), "decided_at": iso(now), "mode": mode, "strategy": sig["strategy"],
            "tag": s["tag"], "act_list": s["act"], **{k: v for k, v in sig.items() if k != "strategy"},
            "count": 0, "filled": 0, "cost": 0.0}


def limits(sig: dict, state: dict, cash: float | None) -> dict:
    """Dollars still allowed by each cap for this order: per order, period (month/week/season), lead open, cash."""
    s = STRATS[sig["strategy"]]
    key = period_key(sig["strategy"], sig | {"ts": sig.get("ts", "")})
    return {"order": s["max_order"], "period": s["period_cap"] - spent_in(state, sig["strategy"], key),
            "open": GLOBAL_OPEN_CAP - open_exposure(state),
            "cash": (cash - CASH_RESERVE) if cash is not None else math.inf}


def budget_left(sig: dict, state: dict, cash: float | None) -> tuple[float, str]:
    lim = limits(sig, state, cash)
    why = min(lim, key=lim.get)
    return lim[why], why


def decide_maker(sig: dict, client: kx.Kalshi, mode: str, now: datetime, state: dict, cash: float | None) -> dict:
    s = STRATS[sig["strategy"]]
    rec = _base(sig, mode, now)
    if sig["src_age_h"] > s["max_age_h"]:
        return {**rec, "status": "skip_stale_signal"}
    m = client.market(sig["ticker"])
    if not mkt_open(m, now):
        return {**rec, "status": "skip_market_not_active", "market_status": (m or {}).get("status")}
    no_ask, depth, no_bid = client.book(sig["ticker"])
    q, side = sig["q"], sig["side"]
    rec.update(price_ts=iso(now), no_ask_now=no_ask, no_bid_now=no_bid, close_time=m.get("close_time"))
    # post_only would cross (Kalshi would reject it): NO bid at q crosses when NO ask <= q; YES bid when YES ask <= q
    cross = (no_ask == no_ask and no_ask <= q + 1e-9) if side == "no" else (no_bid > 0 and 1 - no_bid <= q + 1e-9)
    if cross:
        return {**rec, "status": "skip_would_cross"}
    if not 0 < q < 1:
        return {**rec, "status": "skip_bad_price"}
    fee_type = client.series_fee_type(sig["ticker"].split("-")[0])
    unit = q + kx.maker_fee_unit(q, fee_type)
    cost = lambda n: round(n * q + (kx.fee_total(n, q) if kx.maker_fee_unit(q, fee_type) else 0.0), 4)  # noqa: E731
    left, why = budget_left(sig, state, cash)
    if sig["strategy"] == "TSA":
        n = max(1, int(s["max_order"] // unit))                      # ACT_LIST B4: $1 per order, min 1 contract
        lim = {k: v for k, v in limits(sig, state, cash).items() if k != "order"}
        bad = [k for k, v in lim.items() if cost(n) > v + 1e-9]
        if bad:
            return {**rec, "fee_type": fee_type, "status": f"skip_{bad[0]}"}
    else:
        n = min(s["max_contracts"], int(max(left, 0) // unit))
        while n > 0 and cost(n) > left + 1e-9:
            n -= 1
        if n < 1:
            return {**rec, "fee_type": fee_type, "status": f"skip_{why}"}
    return {**rec, "fee_type": fee_type, "maker_q": q, "count": n, "cost": cost(n), "avg_price": q,
            "expiration_time": int(ts(m["close_time"]).timestamp()), "status": "go"}


def decide_dul(g: dict, client: kx.Kalshi, mode: str, now: datetime, state: dict, cash: float | None) -> dict:
    s = STRATS["DUL"]
    ev = nfl_event_ticker(g)
    sig = {"strategy": "DUL", "ticker": ev, "side": "no", "event_ticker": ev, **g}
    rec = _base(sig, mode, now)
    ms = nfl_markets(client, ev)
    if not ms:
        return {**rec, "status": "skip_no_kalshi_event"}
    exp = ms[0].get("expected_expiration_time")
    if exp and abs((ts(exp) - timedelta(hours=3) - ts(g["kickoff"])).total_seconds()) > KICK_TOL_MIN * 60:
        return {**rec, "status": "skip_kickoff_mismatch", "kalshi_expected_expiration": exp}
    m = pick_strike(ms)
    if m is None:
        return {**rec, "status": "skip_no_two_sided_strike"}
    t = m["ticker"]
    ask, depth, _ = client.book(t)
    rec.update(ticker=t, strike=float(m.get("floor_strike") or 0), yes_mid=m["_mid"], price_ts=iso(now),
               no_ask_now=ask, depth_1c=depth, n_strikes=len(ms), source_ts=m.get("updated_time"))
    if not (ask == ask) or ask >= 1:
        return {**rec, "status": "skip_no_book"}
    if ask > s["max_no_ask"] + 1e-9:
        return {**rec, "status": "skip_price"}
    left, why = budget_left(sig, state, cash)
    n = int(max(left, 0) // (ask + kx.FEE * ask * (1 - ask)))
    n = min(n, int(depth * DEPTH_FRAC))
    while n > 0 and kx.order_cost(n, ask) > left + 1e-9:
        n -= 1
    if n < 1:
        return {**rec, "status": "skip_depth" if left >= kx.order_cost(1, ask) else f"skip_{why}"}
    return {**rec, "count": n, "cost": kx.order_cost(n, ask), "fee": kx.fee_total(n, ask), "avg_price": ask,
            "status": "go"}


# ---------------------------------------------------------------- placement
def order_body(rec: dict, cid: str) -> dict:
    t, n = rec["ticker"], str(rec["count"])
    if rec["strategy"] == "DUL":   # buy NO at the ask == sell YES at 1 - ask, IOC
        return {"ticker": t, "side": "ask", "count": n, "price": f"{1 - rec['avg_price']:.4f}",
                "time_in_force": "immediate_or_cancel", "self_trade_prevention_type": "taker_at_cross",
                "client_order_id": cid}
    q = rec["maker_q"]
    side, price = ("ask", 1 - q) if rec["side"] == "no" else ("bid", q)
    return {"ticker": t, "side": side, "count": n, "price": f"{price:.4f}", "time_in_force": "good_till_canceled",
            "post_only": True, "expiration_time": rec["expiration_time"],
            "self_trade_prevention_type": "taker_at_cross", "client_order_id": cid}


def own_price(o: dict, side: str) -> float | None:
    """Average price paid on OUR side from an exchange Order object (cost / filled; cost is our side's dollars)."""
    f = kx.fills_from_order(o)
    return f["avg_no"] if f["filled"] else None  # kalshi_exec names it avg_no; it is cost/filled for either side


def write_lead_kill(reason: str, cfg: dict) -> None:
    lead_kill_path().parent.mkdir(parents=True, exist_ok=True)
    lead_kill_path().write_text(f"{iso(datetime.now(timezone.utc))} {reason}\n")
    append({"event": "kill", "ts": iso(datetime.now(timezone.utc)), "reason": reason})
    kx.notify("Kalshi LEAD exec KILLED", reason, cfg)


def fill_price_bad(paid: float, ref: float) -> bool:
    """Convention check: above the limit is impossible on a real fill; a few cents below it is price improvement."""
    return paid > ref + CONVENTION_TOL or paid < ref - IMPROVE_TOL


def apply_fills(rec: dict, o: dict, cfg: dict, now: datetime) -> dict:
    """Exchange Order -> update row; maker remainder stays committed at q; convention check -> LEAD_KILL."""
    u = kx.fills_from_order(o)
    u["avg_price"] = u.pop("avg_no")
    if u["filled"] and rec.get("first_fill_ts") is None:
        u["first_fill_ts"] = iso(now)
    if rec.get("maker_q") is not None and u["status"] == "resting":
        rest = max(float(rec["count"]) - u["filled"], 0.0)
        u["cost"] = round(u["cost"] + rest * (rec["maker_q"] + kx.maker_fee_unit(rec["maker_q"], rec.get("fee_type"))), 4)
    ref = rec.get("maker_q", rec.get("avg_price"))
    if u["filled"] and u["avg_price"] is not None and ref is not None and fill_price_bad(u["avg_price"], ref):
        u["cost"] = round(u["filled"] * 1.0 + u["fee"], 4)
        append({"event": "update", "ts": iso(now), "mode": rec["mode"], "ticker": rec["ticker"], **u})
        write_lead_kill(f"fill price check failed on {rec['ticker']}: paid {u['avg_price']} vs {ref}", cfg)
        return {**rec, **u}
    append({"event": "update", "ts": iso(now), "mode": rec["mode"], "ticker": rec["ticker"], **u})
    return {**rec, **u}


def place(client: kx.Kalshi, rec: dict, cfg: dict, now: datetime) -> dict:
    mode, t = rec["mode"], rec["ticker"]
    cid = coid(rec["strategy"], mode, t)
    existing = client.orders(ticker=t)
    pos = [p for p in client.positions(ticker=t) if float(p.get("position_fp") or 0) != 0]
    if existing or pos:
        return append({**rec, "status": "skip_exchange_has_order", "count": 0, "cost": 0.0, "filled": 0})
    append({**rec, "status": "submitting", "client_order_id": cid, "filled": 0})   # write-ahead
    upd = {"event": "update", "ts": iso(now), "mode": mode, "ticker": t}
    maker = rec.get("maker_q") is not None
    try:
        code, out = client.create_order(order_body(rec, cid))
    except kx.TransportError as ex:
        append({**upd, "status": "unknown", "error": str(ex)[:300]})
        kx.notify(f"Kalshi lead {mode} order UNKNOWN", f"{t}: {ex}", cfg)
        return {**rec, "status": "unknown"}
    if code == 409 or code in (200, 201):
        match = [x for x in client.orders(ticker=t) if x.get("client_order_id") == cid]
        if match:
            r = apply_fills(rec, match[0], cfg, now)
            if r.get("filled"):
                kx.notify(f"Kalshi lead {mode} fill", f"{rec['tag']} {t}: {r['filled']:g} {rec['side'].upper()} "
                          f"@ {r['avg_price']}", cfg)
            elif r.get("status") == "resting":
                kx.notify(f"Kalshi lead {mode} bid resting", f"{rec['tag']} {t}: {rec['count']} @ {rec.get('maker_q')}", cfg)
            return r
        if maker and kx._is_post_only_reject(out):
            append({**upd, "status": "rejected_post_only", "http": code, "cost": 0.0})
            return {**rec, "status": "rejected_post_only", "cost": 0.0}
        if code in (200, 201) and not maker and not float(out.get("fill_count") or 0):
            append({**upd, "status": "no_fill", "http": code, "cost": 0.0})
            return {**rec, "status": "no_fill", "cost": 0.0}
        append({**upd, "status": "unknown", "http": code, "response": out})   # reconcile next run
        return {**rec, "status": "unknown"}
    status = "rejected_post_only" if maker and kx._is_post_only_reject(out) else "rejected"
    append({**upd, "status": status, "http": code, "response": out, "cost": 0.0})
    kx.notify(f"Kalshi lead {mode} order {status}", f"{t}: HTTP {code} {str(out)[:120]}", cfg)
    return {**rec, "status": status, "cost": 0.0}


def reconcile(client: kx.Kalshi, mode: str, cfg: dict, now: datetime) -> None:
    """Resolve submitting/unknown/resting rows from the exchange; cancel our maker bids whose market has closed."""
    for t, o in fold(journal(), mode).items():
        if o.get("status") not in ("submitting", "unknown", "resting") or o.get("settled") or o.get("strategy") in INGAME:
            continue
        match = [x for x in client.orders(ticker=t) if x.get("client_order_id") == o.get("client_order_id")]
        if match:
            r = apply_fills(o, match[0], cfg, now)
            if r.get("status") == "resting" and not mkt_open(client.market(t), now):
                code, _ = client.cancel(match[0]["order_id"], t)
                append({"event": "cancel", "ts": iso(now), "mode": mode, "ticker_cancelled": t, "http": code,
                        "why": "market_closed"})
            if lead_kill_path().exists():
                return
        elif now - ts(o["ts"]) > timedelta(minutes=30):
            append({"event": "update", "ts": iso(now), "mode": mode, "ticker": t, "status": "not_placed",
                    "filled": 0, "cost": 0.0})


# ---------------------------------------------------------------- settle (+ DUL CLV)
def closing_no(client: kx.Kalshi, ticker: str, kickoff: str) -> float | None:
    k = int(ts(kickoff).timestamp())
    code, out = client.request("GET", "/markets/trades", params={"ticker": ticker, "max_ts": k, "min_ts": k - 6 * 3600,
                                                                 "limit": 100}, auth=False)
    tr = out.get("trades", []) if code == 200 else []
    if not tr:
        return None
    last = max(tr, key=lambda x: x["created_time"])
    v = last.get("no_price_dollars")
    return float(v) if v is not None else round(1 - float(last["yes_price_dollars"]), 4)


def settle(client: kx.Kalshi, now: datetime) -> int:
    hist, n = journal(), 0
    for mode in sorted({r.get("mode") for r in hist if r.get("mode")}):
        for t, o in fold(hist, mode).items():
            if (o.get("settled") or o.get("status") in ("resting", "submitting", "unknown") or committed(o) <= 0
                    or o.get("strategy") in INGAME):
                continue
            if mode == "dry" and o.get("status") == "dry":
                o = {**o, "filled": o.get("count", 0)}  # dry assumes the whole order filled (maker fills unknown)
            if held(o) <= 0:
                continue
            m = client.market(t) or {}
            r = m.get("result")
            if r not in ("yes", "no"):
                continue
            won = r == o["side"]
            cost = committed(o)
            row = {"event": "settle", "ts": iso(now), "mode": mode, "ticker": t, "result": r, "won": won,
                   "held": held(o), "pnl": round(held(o) - cost if won else -cost, 4)}
            if o.get("strategy") == "DUL" and o.get("kickoff"):
                c = closing_no(client, t, o["kickoff"])
                row.update(close_no=c, clv=round(c - float(o["avg_price"]), 4) if c is not None else None)
            append(row)
            n += 1
    return n


def apply_kill_rules(mode: str, cfg: dict, now: datetime) -> list[str]:
    st, out = fold(journal(), mode), []
    for s in STRATS:
        why = kill_verdict(st, s)
        if not why or disable_path(s).exists():
            continue
        if mode == "dry":
            out.append(f"[dry] {STRATS[s]['tag']} kill rule met ({why}) -- would write {disable_path(s).name}")
            continue
        disable_path(s).write_text(f"{iso(now)} {why}\n")
        append({"event": "disable", "ts": iso(now), "mode": mode, "strategy": s, "reason": why})
        kx.notify(f"Kalshi {STRATS[s]['tag']} DISABLED", why, cfg)
        out.append(f"{STRATS[s]['tag']} DISABLED: {why}")
    return out


# ---------------------------------------------------------------- commands
def cmd_run(argv: list[str] | None = None, client: kx.Kalshi | None = None, now: datetime | None = None) -> int:
    argv = sys.argv[2:] if argv is None else argv
    now = now or datetime.now(timezone.utc)
    e = kx.env()
    cfg = kx.config(e)
    mode = lead_mode(e, "--live" in argv)
    for k in (lead_kill_path(), kx.killed()):
        if k and Path(k).exists():
            print(f"KILL file present ({k}) -> nothing done")
            return 0
    if "--live" in argv and mode == "dry":
        print("--live ignored: .env needs EXEC_MODE=live AND LEAD_EXEC_MODE=live -> dry run")
    client = client or kx.Kalshi(e, cfg["env"])
    cash = None
    if mode != "dry":
        if not client.has_key():
            print("no Kalshi key in .env -> refusing to trade")
            return 2
        reconcile(client, mode, cfg, now)
        if lead_kill_path().exists():
            print("LEAD_KILL written during reconcile -> stop")
            return 0
    settle(client, now)
    for line in apply_kill_rules(mode, cfg, now):
        print(line)
    if mode != "dry" or client.has_key():
        try:
            cash = shard0_cash(client)
        except Exception as ex:  # noqa: BLE001
            if mode != "dry":
                print(f"cannot read balance ({str(ex)[:120]}) -> refusing to trade")
                return 2
    state = fold(journal(), mode)
    print(f"[{mode}] {now:%Y-%m-%d %H:%M}Z  lead open ${open_exposure(state):.2f}/${GLOBAL_OPEN_CAP:.0f}  "
          f"shard-0 cash {'$%.2f' % cash if cash is not None else 'unknown (no key in dry)'} (reserve ${CASH_RESERVE:.0f})")
    decided = {t for t, o in state.items() if not (o.get("strategy") == "DUL" and o.get("status", "").startswith("skip"))}
    dul_done = {o.get("game_id") for o in state.values()
                if o.get("strategy") == "DUL" and not o.get("status", "").startswith("skip")}
    todo = []
    for s, sigs in (("L39", l39_signals(now)), ("TSA", tsa_signals(now))):
        fresh = [x for x in sigs if x["ticker"] not in decided]
        print(f"  {STRATS[s]['tag']}: {len(sigs)} source rows, {len(fresh)} not yet decided"
              + (" (DISABLED)" if disable_path(s).exists() else ""))
        if not disable_path(s).exists():
            todo += fresh
    games = [g for g in dul_games(now) if g["game_id"] not in dul_done]
    print(f"  lead_DUL: {len(games)} division game(s) in the 30-90 min window"
          + (" (DISABLED)" if disable_path("DUL").exists() else ""))
    if disable_path("DUL").exists():
        games = []
    for item in todo + games:
        if "strategy" in item:
            rec = decide_maker(item, client, mode, now, state, cash)
        else:
            rec = decide_dul(item, client, mode, now, state, cash)
        if rec["status"] != "go":
            append(rec)
        elif mode == "dry":
            rec = append({**rec, "status": "dry", "filled": rec["count"]})
        else:
            rec = place(client, rec, cfg, now)
            if lead_kill_path().exists():
                print("LEAD_KILL written after a fill -> stop")
                break
        c = committed(rec)
        state[rec["ticker"]] = rec
        if cash is not None:
            cash -= c
        extra = (f" strike {rec.get('strike')} yes_mid {rec.get('yes_mid')} kick {rec.get('kickoff')}"
                 if rec["strategy"] == "DUL" else f" q {rec.get('q')} src age {rec.get('src_age_h')}h")
        print(f"  {mode} {rec['tag']:9s} {rec['ticker']:34s} {rec.get('side', '').upper():3s} -> {rec['status']} "
              f"{rec.get('count', 0)} ct ${c:.2f}{extra}")
    return 0


def cmd_settle(client: kx.Kalshi | None = None, now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    e = kx.env()
    client = client or kx.Kalshi(e, kx.config(e)["env"])
    print(f"settled {settle(client, now)} row(s)")
    return 0


def cmd_status(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    e = kx.env()
    hist = journal()
    print(f"lead_exec: launchd --live would be {lead_mode(e, True).upper()}; LEAD_KILL "
          f"{'PRESENT' if lead_kill_path().exists() else 'none'}; kalshi_exec KILL {kx.killed() or 'none'}")
    for mode in ("dry", "demo", "live"):
        st = fold(hist, mode)
        if not st:
            continue
        print(f"\n[{mode}] lead open ${open_exposure(st):.2f} / ${GLOBAL_OPEN_CAP:.0f}")
        for s, cfg in STRATS.items():
            rows = [o for o in st.values() if o.get("strategy") == s]
            done = settled_rows(st, s)
            last = max(rows, key=lambda o: o["ts"]) if rows else None
            key = {"L39": f"{now:%Y-%m}", "TSA": period_key(s, last) if last else "-",
                   "DUL": str(cfg.get("season"))}[s]
            skips = {}
            for o in rows:
                if o.get("status", "").startswith("skip"):
                    skips[o["status"]] = skips.get(o["status"], 0) + 1
            clv = [o["clv"] for o in done if o.get("clv") is not None]
            print(f"  {cfg['tag']:9s} {len(rows)} decisions, {sum(committed(o) > 0 for o in rows)} with money, "
                  f"{len(done)} settled ({sum(bool(o.get('won')) for o in done)} W) P&L ${sum(o['pnl'] for o in done):+.2f}"
                  f"; {cfg['period']} {key} spent ${spent_in(st, s, key):.2f}/${cfg['period_cap']:.0f}"
                  + (f"; mean CLV {sum(clv) / len(clv):+.4f} (n {len(clv)})" if clv else "")
                  + (f"; DISABLED ({disable_path(s).read_text().strip()})" if disable_path(s).exists() else "")
                  + (f"; kill rule MET: {kill_verdict(st, s)}" if kill_verdict(st, s) else ""))
            if skips:
                print("            skips: " + ", ".join(f"{k} {v}" for k, v in sorted(skips.items())))
    return 0


def cmd_kill(argv: list[str] | None = None, client: kx.Kalshi | None = None) -> int:
    argv = sys.argv[2:] if argv is None else argv
    e = kx.env()
    cfg = kx.config(e)
    write_lead_kill(" ".join(argv) or "manual kill", cfg)
    client = client or kx.Kalshi(e, cfg["env"])
    if client.has_key():
        ours = {o.get("client_order_id") for m in ("demo", "live") for o in fold(journal(), m).values()}
        for x in client.orders(status="resting"):
            if x.get("client_order_id") in ours:
                code, _ = client.cancel(x["order_id"], x.get("ticker"))
                print(f"cancelled {x.get('ticker')} -> HTTP {code}")
    print(f"LEAD_KILL written: {lead_kill_path()}; remove it to resume")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    fns = {"run": cmd_run, "status": cmd_status, "settle": cmd_settle, "kill": cmd_kill}
    if cmd not in fns:
        print(__doc__)
        return 1
    try:
        return fns[cmd]()
    except Exception as ex:  # noqa: BLE001
        msg = f"{type(ex).__name__}: {str(ex)[:300]}"
        try:
            append({"event": "error", "ts": iso(datetime.now(timezone.utc)), "where": cmd, "error": msg})
            kx.notify(f"Kalshi lead exec ERROR ({cmd})", msg, kx.config(kx.env()))
        except Exception:  # noqa: BLE001
            pass
        print(f"ERROR {msg}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
