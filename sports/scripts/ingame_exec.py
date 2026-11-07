"""In-game Kalshi executor, SMALL real money (user decision 11-07): NHL late-game taker + NFL post-big-play maker fade.
DRY unless armed. Never run it from the worktree against the deployed .env; the loggers call the deployed copy.

    python scripts/ingame_exec.py tick < signals.jsonl   # decide new signals (first, for latency), then housekeep
    python scripts/ingame_exec.py tick < /dev/null       # housekeep only: NFLF cancel/exit, NHL CLV marks, settle, kills
    python scripts/ingame_exec.py status                 # per-strategy mode, spend vs caps, W/L, P&L, CLV, kill state
    python scripts/ingame_exec.py kill [reason]          # = lead_exec.py kill (LEAD_KILL + cancel our resting orders)

Who calls it: the deployed paper loggers, in the SAME tick that produced the signal, as a subprocess of the
quant-kalshi venv (keys / .env stay in ~/.claude/quant-kalshi; the paper process never reads them):
  NHL   scripts/nhl_ingame_logger.py (launchd quant-sports-nhl-ingame, 60 s): frozen edgelab rule
        nhl_ingame.t1_late_taker_edge3 evaluated live on the game's ticks so far -> rows decided THIS tick.
  NFLF  scripts/nfl_fade_live.py (launchd quant-sports-nfl-fade, 60 s with 15 s sub-ticks): live F1 proxy.
Why a call from the logger rather than polling a signal file: the signal is ~1 s old when the executor starts; a
lead_exec launchd poll (900 s) or even a 15 s poller adds 0-15 s of a 20 s budget for nothing and a second process
racing on the journal.

Gate (per strategy S in NHL / NFLF): orders only when .env EXEC_MODE=live AND LEAD_EXEC_MODE=live (lead_exec.lead_mode
with --live) AND .env LEAD_LIVE_<S>=1. Anything else -> dry (decisions logged as status "dry", nothing sent, no
authenticated call). EXEC_ENV=demo -> demo exchange. KILL (kalshi_exec), data/LEAD_KILL, data/LEAD_DISABLE_<S> stop it.
Journal: data/lead_exec.jsonl (shared with lead_exec.py; rows carry `key`, lead_exec.py skips NHL/NFLF rows).

  NHL   buy the rule's side (YES at the ask / NO at 1 - bid) IOC, limit = signal price + 1c. contracts =
        floor($1 / (limit + taker fee)) (0 -> skip). Caps: <= $1.00/bet, <= $5/day (ET game date), <= $40 season,
        one per (game, backed team) = key NHL|gid|backed. Hold to settlement. CLV = same market mid at the first
        housekeep in [t+300, t+420] s else [t+60, t+180] s (the trial's definition) minus our price.
        Kill: >= 10 losses within the first 30 settled; after >= 100 settled: ROI 95% CI (cluster bootstrap by date)
        entirely < 0, or mean CLV < 0 with one-sided p < .05 -> data/LEAD_DISABLE_NHL.
  NFLF  post-only YES bid on the side whose win prob fell, at the signal's bid q, contracts = floor($2 / (q + fee)),
        expiration_time = t_play + 240 s (exchange-side cancel; we also cancel). Exit: at t_play + 480 s sell the
        filled contracts IOC at the YES bid (retries each housekeep until t_play + 900 s, then hold to settlement).
        Caps: <= $2/order, <= $10 per game day (ET date), <= $40 total. One order per big play (key NFLF|gid|play).
        Kill: real P&L <= -$15, or >= 8 losses within the first 20 filled trades -> data/LEAD_DISABLE_NFLF.
  both  refuse if the signal is > 20 s old or the price we would pay/post moved > 2c from the signal price;
        Kalshi shard-0 cash after the order must stay >= $80 (weather executor room); market must be active.
        client_order_id = uuid5(ingame_exec/S/mode/key); write-ahead row before the POST; the exchange is asked for
        our order first (re-send = 409, never a second order). Fill priced > 2c from the limit -> LEAD_KILL.
Dry NFLF is simulated from the book at each housekeep: filled iff the YES ask <= q - 1c before t+240 (F1's
trade-through test, polled), exit at the YES bid at the first housekeep >= t+480.
"""
from __future__ import annotations

import fcntl
import json
import math
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead_exec as lx  # noqa: E402

kx = lx.kx
STRATS = {
    "NHL": {"tag": "live_NHL", "max_order": 1.00, "day_cap": 5.00, "season_cap": 40.00, "slip": 0.01,
            "early_window": 30, "early_losses": 10, "judge_after": 100},
    "NFLF": {"tag": "live_NFLF", "max_order": 2.00, "day_cap": 10.00, "season_cap": 40.00,
             "fill_s": 240, "exit_s": 480, "give_up_s": 900, "pnl_stop": -15.00, "early_window": 20, "early_losses": 8},
}
MAX_AGE_S = 20
MAX_MOVE = 0.02
CONVENTION_TOL = 0.02
MAKER_FEE = 0.0175          # KXNFLGAME fee_type quadratic_with_maker_fees (checked 11-07); sizing uses the taker 0.07
ZERO = lx.ZERO


def iso(t: datetime) -> str:
    return lx.iso(t)


def ts(s) -> datetime:
    return lx.ts(s)


def strat_mode(e: dict, s: str) -> str:
    m = lx.lead_mode(e, True)
    return m if m != "dry" and e.get(f"LEAD_LIVE_{s}") == "1" else "dry"


def coid(s: str, mode: str, key: str, suffix: str = "") -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ingame_exec/{s}/{mode}/{key}{suffix}"))


def state(mode: str) -> dict:
    return {k: o for k, o in lx.fold(lx.journal(), mode).items() if o.get("strategy") in STRATS}


def upd(o: dict, now: datetime, **kw) -> dict:
    """Append an update row for decision `o` and return the merged row."""
    lx.append({"event": "update", "ts": iso(now), "mode": o["mode"], "key": o["key"], "ticker": o["ticker"], **kw})
    return {**o, **kw}


def et_date(t: datetime) -> str:
    return t.astimezone(lx.ET).date().isoformat()


def spent(st: dict, s: str, day: str | None = None) -> float:
    return sum(lx.committed(o) for o in st.values()
               if o.get("strategy") == s and (day is None or o.get("game_day") == day))


def yes_quotes(client, ticker: str) -> tuple[float, float]:
    """(YES bid, YES ask) from the book; nan when a side is empty."""
    no_ask, _, no_bid = client.book(ticker)
    yb = round(1 - no_ask, 4) if no_ask == no_ask else math.nan
    ya = round(1 - no_bid, 4) if no_bid > 0 else math.nan
    return yb, ya


def taker_unit(p: float) -> float:
    return p + kx.FEE * p * (1 - p)


# ---------------------------------------------------------------- decisions
def decide(sig: dict, client, mode: str, now: datetime, st: dict, cash: float | None) -> dict:
    s = sig["strategy"]
    S = STRATS[s]
    dec = ts(sig["decided_at"])
    rec = {"event": "order", "ts": iso(now), "mode": mode, "strategy": s, "tag": S["tag"], "key": sig["key"],
           "ticker": sig["ticker"], "side": sig["side"], "signal_price": float(sig["price"]),
           "decided_at": sig["decided_at"], "exec_at": iso(now), "signal_age_s": round((now - dec).total_seconds(), 1),
           "source": sig.get("source"), "source_row": sig.get("source_row"), "game_day": sig.get("game_day") or et_date(dec),
           **{k: sig[k] for k in ("gid", "backed", "edge", "p_side", "t_play", "wp_delta", "play_id") if k in sig},
           "count": 0, "filled": 0, "cost": 0.0}
    if rec["signal_age_s"] > MAX_AGE_S:
        return {**rec, "status": "skip_stale_signal"}
    m = client.market(sig["ticker"])
    if not m or m.get("status") not in ("active", "open"):
        return {**rec, "status": "skip_market_not_active", "market_status": (m or {}).get("status")}
    yb, ya = yes_quotes(client, sig["ticker"])
    rec.update(yes_bid_now=yb, yes_ask_now=ya, price_ts=iso(now))
    q = float(sig["price"])
    if s == "NHL":
        now_px = ya if sig["side"] == "yes" else (round(1 - yb, 4) if yb == yb else math.nan)
    else:
        now_px = yb
    rec["price_now"] = now_px
    if not now_px == now_px:
        return {**rec, "status": "skip_no_book"}
    if abs(now_px - q) > MAX_MOVE + 1e-9:
        return {**rec, "status": "skip_price_moved"}
    if s == "NHL":
        px = round(min(q + S["slip"], 0.99), 2)
        n = int(S["max_order"] // taker_unit(px))
        cost = lambda k: kx.order_cost(k, px)  # noqa: E731
    else:
        if ya == ya and ya <= q + 1e-9:
            return {**rec, "status": "skip_would_cross"}
        px = q
        n = int(S["max_order"] // taker_unit(px))           # conservative: taker fee formula for the maker fee
        cost = lambda k: kx.order_cost(k, px)  # noqa: E731
        rec.update(maker_q=q, fill_until=int(sig["t_play"]) + S["fill_s"], exit_at=int(sig["t_play"]) + S["exit_s"])
    if not 0 < px < 1 or n < 1:
        return {**rec, "status": "skip_size"}
    lim = {"order": S["max_order"], "day": S["day_cap"] - spent(st, s, rec["game_day"]),
           "season": S["season_cap"] - spent(st, s), "cash": (cash - lx.CASH_RESERVE) if cash is not None else math.inf}
    while n > 0 and cost(n) > min(lim.values()) + 1e-9:
        n -= 1
    if n < 1:
        why = min(lim, key=lim.get)
        return {**rec, "status": f"skip_{why}", "limits": {k: round(v, 2) for k, v in lim.items() if v != math.inf}}
    return {**rec, "count": n, "limit_price": px, "cost": cost(n), "status": "go"}


def order_body(rec: dict, cid: str) -> dict:
    n, px = str(rec["count"]), rec["limit_price"]
    if rec["strategy"] == "NFLF":   # rest a YES bid; the exchange expires it at t_play + 240 s
        return {"ticker": rec["ticker"], "side": "bid", "count": n, "price": f"{px:.4f}",
                "time_in_force": "good_till_canceled", "post_only": True, "expiration_time": int(rec["fill_until"]),
                "self_trade_prevention_type": "taker_at_cross", "client_order_id": cid}
    side, yes_px = ("bid", px) if rec["side"] == "yes" else ("ask", 1 - px)   # buy NO at px == sell YES at 1 - px
    return {"ticker": rec["ticker"], "side": side, "count": n, "price": f"{yes_px:.4f}",
            "time_in_force": "immediate_or_cancel", "self_trade_prevention_type": "taker_at_cross",
            "client_order_id": cid}


def apply_fills(o: dict, x: dict, cfg: dict, now: datetime) -> dict:
    u = kx.fills_from_order(x)
    u["avg_price"] = u.pop("avg_no")            # cost / filled on OUR side (kalshi_exec's name says NO; either side)
    if u["filled"] and o.get("first_fill_ts") is None:
        u["first_fill_ts"] = iso(now)
    if o["strategy"] == "NFLF" and u["status"] == "resting":
        rest = max(float(o["count"]) - u["filled"], 0.0)
        u["cost"] = round(u["cost"] + rest * taker_unit(o["limit_price"]), 4)   # remainder stays committed
    r = upd(o, now, **u)
    if u["filled"] and u["avg_price"] is not None and abs(u["avg_price"] - o["limit_price"]) > CONVENTION_TOL:
        lx.write_lead_kill(f"ingame fill price check failed on {o['ticker']}: paid {u['avg_price']} vs limit "
                           f"{o['limit_price']}", cfg)
    return r


def place(client, rec: dict, cfg: dict, now: datetime) -> dict:
    mode, t, key = rec["mode"], rec["ticker"], rec["key"]
    cid = coid(rec["strategy"], mode, key)
    on_ticker = client.orders(ticker=t)
    mine = [x for x in on_ticker if x.get("client_order_id") == cid]
    if mine:   # already sent (crash between POST and journal): adopt it, never send twice
        lx.append({**rec, "status": "submitting", "client_order_id": cid})
        return apply_fills({**rec, "client_order_id": cid}, mine[0], cfg, now)
    mine_all = {o.get("client_order_id") for o in state(mode).values()}
    if any(x.get("status") == "resting" and x.get("client_order_id") not in mine_all for x in on_ticker):
        return lx.append({**rec, "status": "skip_exchange_has_order", "count": 0, "cost": 0.0})   # not ours
    ours = {o["ticker"] for o in state(mode).values() if lx.committed(o) > 0}
    pos = [p for p in client.positions(ticker=t) if float(p.get("position_fp") or 0) != 0]
    if pos and t not in ours:   # someone else's position on this market (manual bet): do not mix
        return lx.append({**rec, "status": "skip_exchange_has_position", "count": 0, "cost": 0.0})
    rec = lx.append({**rec, "status": "submitting", "client_order_id": cid})       # write-ahead
    try:
        code, out = client.create_order(order_body(rec, cid))
    except kx.TransportError as ex:
        kx.notify(f"Kalshi {rec['tag']} order UNKNOWN", f"{t}: {ex}", cfg)
        return upd(rec, now, status="unknown", error=str(ex)[:300])
    if code in (200, 201, 409):
        match = [x for x in client.orders(ticker=t) if x.get("client_order_id") == cid]
        if match:
            r = apply_fills(rec, match[0], cfg, now)
            what = "fill" if r.get("filled") else ("bid resting" if r.get("status") == "resting" else r.get("status"))
            kx.notify(f"Kalshi {rec['tag']} {mode} {what}", f"{t} {rec['side'].upper()} {r.get('filled', 0):g}/"
                      f"{rec['count']} @ {r.get('avg_price') or rec['limit_price']}", cfg)
            return r
        if rec["strategy"] == "NFLF" and kx._is_post_only_reject(out):
            return upd(rec, now, status="rejected_post_only", http=code, cost=0.0)
        if code in (200, 201) and rec["strategy"] == "NHL" and not float(out.get("fill_count") or 0):
            return upd(rec, now, status="no_fill", http=code, cost=0.0)
        return upd(rec, now, status="unknown", http=code, response=out)
    st_ = "rejected_post_only" if rec["strategy"] == "NFLF" and kx._is_post_only_reject(out) else "rejected"
    kx.notify(f"Kalshi {rec['tag']} order {st_}", f"{t}: HTTP {code} {str(out)[:120]}", cfg)
    return upd(rec, now, status=st_, http=code, response=out, cost=0.0)


# ---------------------------------------------------------------- housekeeping
def find_order(client, o: dict, cid: str | None = None) -> dict | None:
    cid = cid or o.get("client_order_id")
    m = [x for x in client.orders(ticker=o["ticker"]) if x.get("client_order_id") == cid]
    return m[0] if m else None


def mid_mark(client, o: dict, now: datetime) -> dict:
    """NHL CLV marks: first housekeep in [t+60, t+180] -> mark60, in [t+300, t+420] -> mark300 (same market mid)."""
    el = (now - ts(o["decided_at"])).total_seconds()
    for lo, hi, k in ((60, 180, "mark60"), (300, 420, "mark300")):
        if lo <= el <= hi and o.get(k) is None:
            yb, ya = yes_quotes(client, o["ticker"])
            if yb == yb and ya == ya and 0 < yb < ya < 1:
                mid = round((yb + ya) / 2, 4)
                o = upd(o, now, **{k: mid if o["side"] == "yes" else round(1 - mid, 4), f"{k}_ts": iso(now)})
    return o


def nflf_manage(client, o: dict, cfg: dict, now: datetime) -> dict:
    t = now.timestamp()
    S = STRATS["NFLF"]
    if o["mode"] == "dry":
        if o.get("status") == "dry_resting":
            yb, ya = yes_quotes(client, o["ticker"])
            if t < o["fill_until"] and ya == ya and ya <= o["maker_q"] - 0.01 + 1e-9:
                o = upd(o, now, status="dry_filled", filled=o["count"], first_fill_ts=iso(now), fill_ask=ya)
            elif t >= o["fill_until"]:
                o = upd(o, now, status="dry_no_fill", filled=0, cost=0.0)
        if o.get("status") == "dry_filled" and t >= o["exit_at"] and not o.get("settled"):
            yb, _ = yes_quotes(client, o["ticker"])
            if yb == yb and yb > 0:
                n = float(o["filled"])
                proceeds = round(n * yb - kx.fee_total(n, yb), 4)
                o = upd(o, now, exit_filled=n, exit_price=yb, exit_proceeds=proceeds)
                o = close_row(o, now, proceeds - o["cost"], "exited")
            elif t >= int(o["t_play"]) + S["give_up_s"]:
                o = upd(o, now, exit_gave_up=True)
        return o
    if o.get("status") in ("submitting", "unknown", "resting"):
        x = find_order(client, o)
        if x is None:
            if now - ts(o["ts"]) > timedelta(minutes=10):
                o = upd(o, now, status="not_placed", filled=0, cost=0.0)
            return o
        o = apply_fills(o, x, cfg, now)
        if o.get("status") == "resting" and t >= o["fill_until"]:
            code, _ = client.cancel(x["order_id"], o["ticker"])
            lx.append({"event": "cancel", "ts": iso(now), "mode": o["mode"], "key": o["key"],
                       "ticker_cancelled": o["ticker"], "http": code, "why": "fill_window_over"})
            x = find_order(client, o)
            if x is not None:
                o = apply_fills(o, x, cfg, now)
                if o.get("status") == "resting":    # cancel not reflected yet: count only what filled
                    o = upd(o, now, status="partial" if o["filled"] else "no_fill",
                            cost=round(float(o["filled"]) * taker_unit(o["limit_price"]), 4) if o["filled"] else 0.0)
    left = float(o.get("filled") or 0) - float(o.get("exit_filled") or 0)
    if left > 0 and t >= o["exit_at"] and o.get("status") != "resting" and not o.get("exit_gave_up") \
            and not o.get("settled"):
        if t >= int(o["t_play"]) + S["give_up_s"]:
            kx.notify("Kalshi live_NFLF exit gave up", f"{o['ticker']}: holding {left:g} to settlement", cfg)
            return upd(o, now, exit_gave_up=True)
        yb, _ = yes_quotes(client, o["ticker"])
        if not (yb == yb and yb > 0):
            return o
        k = int(o.get("exit_attempts") or 0) + 1
        cid = coid("NFLF", o["mode"], o["key"], f"/exit/{k}")
        o = upd(o, now, exit_attempts=k, exit_coid=cid, exit_limit=yb)              # write-ahead
        body = {"ticker": o["ticker"], "side": "ask", "count": str(int(left)), "price": f"{yb:.4f}",
                "time_in_force": "immediate_or_cancel", "self_trade_prevention_type": "taker_at_cross",
                "client_order_id": cid}
        try:
            client.create_order(body)
        except kx.TransportError as ex:
            return upd(o, now, exit_error=str(ex)[:200])
        x = find_order(client, o, cid)
        f = float((x or {}).get("fill_count_fp") or 0)
        if f > 0:
            fee = float((x or {}).get("taker_fees_dollars") or 0) or kx.fee_total(f, yb)
            proceeds = round(float(o.get("exit_proceeds") or 0) + f * yb - fee, 4)   # lower bound: IOC sells >= bid
            o = upd(o, now, exit_filled=float(o.get("exit_filled") or 0) + f, exit_proceeds=proceeds, exit_price=yb,
                    exit_raw={k_: (x or {}).get(k_) for k_ in ("taker_fill_cost_dollars", "maker_fill_cost_dollars",
                                                               "outcome_side", "status")})
            if float(o["exit_filled"]) >= float(o["filled"]) - 1e-9:
                o = close_row(o, now, proceeds - lx.committed(o), "exited")
                kx.notify(f"Kalshi live_NFLF exit {o['mode']}", f"{o['ticker']} sold {o['exit_filled']:g} @ {yb} "
                          f"P&L ${o['pnl']:+.2f}", cfg)
    return o


def close_row(o: dict, now: datetime, pnl: float, result: str) -> dict:
    row = {"event": "settle", "ts": iso(now), "mode": o["mode"], "key": o["key"], "ticker": o["ticker"],
           "result": result, "won": pnl > 0, "pnl": round(pnl, 4)}
    if o["strategy"] == "NHL":
        mk = o.get("mark300", o.get("mark60"))
        px = o.get("avg_price") or o.get("limit_price")
        row.update(clv=round(mk - px, 4) if mk is not None and px is not None else None)
    lx.append(row)
    return {**o, **{k: v for k, v in row.items() if k not in ("event", "ts")}, "settled": True}


def settle_one(client, o: dict, now: datetime) -> dict:
    if o.get("settled") or now - ts(o["decided_at"]) < timedelta(minutes=30):
        return o
    held = float(o.get("filled") or 0) - float(o.get("exit_filled") or 0)
    if o.get("status") in ("resting", "submitting", "unknown", "dry_resting") or held <= 0 or lx.committed(o) <= 0:
        return o
    if o["strategy"] == "NFLF" and not o.get("exit_gave_up"):
        return o            # the exit path closes NFLF rows; settlement only for contracts we could not sell
    r = (client.market(o["ticker"]) or {}).get("result")
    if r not in ("yes", "no"):
        return o
    pnl = float(o.get("exit_proceeds") or 0) + (held if r == o["side"] else 0.0) - lx.committed(o)
    return close_row(o, now, pnl, r)


def housekeep(client, mode_of: dict, cfg: dict, now: datetime) -> list[str]:
    out = []
    for mode in sorted(set(mode_of.values())):
        for o in state(mode).values():
            if o.get("settled") or mode_of.get(o["strategy"]) != mode:
                continue
            try:
                if o["strategy"] == "NFLF":
                    o = nflf_manage(client, o, cfg, now)
                elif lx.committed(o) > 0:
                    o = mid_mark(client, o, now)
                    if o.get("status") in ("submitting", "unknown") and mode != "dry":
                        x = find_order(client, o)
                        if x is not None:
                            o = apply_fills(o, x, cfg, now)
                        elif now - ts(o["ts"]) > timedelta(minutes=10):
                            o = upd(o, now, status="not_placed", filled=0, cost=0.0)
                settle_one(client, o, now)
            except kx.TransportError as ex:
                out.append(f"  housekeep {o['key']}: {ex}")
            if lx.lead_kill_path().exists():
                return out + ["LEAD_KILL written during housekeeping -> stop"]
    return out


# ---------------------------------------------------------------- kill rules
def _boot(clusters: dict, B: int = 2000, seed: int = 7) -> list[float]:
    """Cluster bootstrap of sum(num)/sum(den); clusters = {key: (num, den)}."""
    ks = list(clusters)
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        pick = [clusters[ks[rng.randrange(len(ks))]] for _ in ks]
        den = sum(d for _, d in pick)
        out.append(sum(n for n, _ in pick) / den if den else 0.0)
    return sorted(out)


def closed_rows(st: dict, s: str) -> list[dict]:
    rows = [o for o in st.values() if o.get("strategy") == s and o.get("settled") and "pnl" in o]
    return sorted(rows, key=lambda o: o.get("first_fill_ts") or o["ts"])


def kill_verdict(st: dict, s: str) -> str | None:
    S, rows = STRATS[s], closed_rows(st, s)
    first = rows[: S["early_window"]]
    losses = sum(o["pnl"] < 0 for o in first)
    if losses >= S["early_losses"]:
        return f"{losses} losses within the first {S['early_window']} settled"
    if s == "NFLF":
        pnl = sum(o["pnl"] for o in rows)
        return f"real P&L ${pnl:+.2f} <= ${S['pnl_stop']:.0f}" if pnl <= S["pnl_stop"] + 1e-9 else None
    if len(rows) < S["judge_after"]:
        return None
    cl: dict = {}
    for o in rows:
        a = cl.setdefault(o["game_day"], [0.0, 0.0])
        a[0] += o["pnl"]
        a[1] += lx.committed(o)
    bs = _boot({k: tuple(v) for k, v in cl.items()})
    hi = bs[int(0.975 * (len(bs) - 1))]
    if hi < 0:
        return f"after {len(rows)} settled: ROI 95% CI upper {hi:+.2%} < 0"
    cc: dict = {}
    for o in rows:
        if o.get("clv") is not None:
            a = cc.setdefault(o["game_day"], [0.0, 0.0])
            a[0] += o["clv"]
            a[1] += 1
    if cc:
        bc = _boot({k: tuple(v) for k, v in cc.items()}, seed=8)
        p_neg = (1 + sum(b >= 0 for b in bc)) / (len(bc) + 1)
        mean = sum(v[0] for v in cc.values()) / sum(v[1] for v in cc.values())
        if mean < 0 and p_neg < 0.05:
            return f"after {len(rows)} settled: mean CLV {mean:+.4f}, one-sided p {p_neg:.3f}"
    return None


def apply_kills(mode_of: dict, cfg: dict, now: datetime) -> list[str]:
    out = []
    for s, mode in mode_of.items():
        why = kill_verdict(state(mode), s)
        if not why or lx.disable_path(s).exists():
            continue
        if mode == "dry":
            out.append(f"[dry] {STRATS[s]['tag']} kill rule met ({why}) -- would write {lx.disable_path(s).name}")
            continue
        lx.disable_path(s).parent.mkdir(parents=True, exist_ok=True)
        lx.disable_path(s).write_text(f"{iso(now)} {why}\n")
        lx.append({"event": "disable", "ts": iso(now), "mode": mode, "strategy": s, "reason": why})
        kx.notify(f"Kalshi {STRATS[s]['tag']} DISABLED", why, cfg)
        out.append(f"{STRATS[s]['tag']} DISABLED: {why}")
    return out


# ---------------------------------------------------------------- commands
class _Lock:
    def __init__(self, wait_s: float = 15.0):
        self.p, self.wait_s, self.f = kx.DATA / "ingame_exec.lock", wait_s, None

    def __enter__(self):
        self.p.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.p.open("w")
        t0 = time.monotonic()
        while True:
            try:
                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.monotonic() - t0 > self.wait_s:
                    raise TimeoutError("ingame_exec lock busy")
                time.sleep(0.2)

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def cmd_tick(signals: list[dict], client=None, clock=None) -> int:
    clock = clock or (lambda: datetime.now(timezone.utc))
    e = kx.env()
    cfg = kx.config(e)
    mode_of = {s: strat_mode(e, s) for s in STRATS}
    for k in (lx.lead_kill_path(), kx.killed()):
        if k and Path(k).exists():
            print(f"KILL file present ({k}) -> nothing done ({len(signals)} signal(s) dropped)")
            return 0
    client = client or kx.Kalshi(e, cfg["env"])
    live_any = any(m != "dry" for m in mode_of.values())
    if live_any and not client.has_key():
        print("no Kalshi key in .env -> refusing to trade")
        return 2
    with _Lock():
        now = clock()
        cash = None
        if signals and any(mode_of[x["strategy"]] != "dry" for x in signals if x.get("strategy") in STRATS):
            try:
                cash = lx.shard0_cash(client)
            except Exception as ex:  # noqa: BLE001
                print(f"cannot read balance ({str(ex)[:120]}) -> refusing to trade")
                return 2
        for sig in signals:
            s = sig.get("strategy")
            if s not in STRATS:
                print(f"  unknown strategy in signal: {s!r}")
                continue
            mode = mode_of[s]
            st = state(mode)
            if sig["key"] in st:
                print(f"  {mode} {STRATS[s]['tag']} {sig['key']} already decided -> {st[sig['key']].get('status')}")
                continue
            if lx.disable_path(s).exists():
                lx.append({"event": "order", "ts": iso(now), "mode": mode, "strategy": s, "key": sig["key"],
                           "ticker": sig["ticker"], "status": "skip_disabled", "count": 0, "cost": 0.0,
                           "decided_at": sig["decided_at"], "source_row": sig.get("source_row")})
                print(f"  {mode} {STRATS[s]['tag']} DISABLED -> skip {sig['key']}")
                continue
            now = clock()
            rec = decide(sig, client, mode, now, st, cash if mode != "dry" else None)
            if rec["status"] != "go":
                lx.append(rec)
            elif mode == "dry":
                rec = lx.append({**rec, "status": "dry" if s == "NHL" else "dry_resting",
                                 "filled": rec["count"] if s == "NHL" else 0, "avg_price": rec["limit_price"]})
            else:
                rec = place(client, rec, cfg, now)
                if cash is not None:
                    cash -= lx.committed(rec)
            print(f"  {mode} {rec['tag']} {rec['key']} {rec['ticker']} {rec['side'].upper()} sig {rec['signal_price']} "
                  f"now {rec.get('price_now')} age {rec['signal_age_s']}s -> {rec['status']} {rec.get('count', 0)} ct "
                  f"${lx.committed(rec):.2f}")
            if lx.lead_kill_path().exists():
                print("LEAD_KILL written -> stop")
                return 0
        for line in housekeep(client, mode_of, cfg, clock()) + apply_kills(mode_of, cfg, clock()):
            print(line)
    return 0


def cmd_status(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    e = kx.env()
    print("ingame_exec: " + ", ".join(f"{s} -> {strat_mode(e, s).upper()}" for s in STRATS)
          + f"; LEAD_KILL {'PRESENT' if lx.lead_kill_path().exists() else 'none'}; KILL {kx.killed() or 'none'}")
    for mode in ("dry", "demo", "live"):
        st = state(mode)
        if not st:
            continue
        print(f"[{mode}]")
        for s, S in STRATS.items():
            rows = [o for o in st.values() if o.get("strategy") == s]
            if not rows:
                continue
            done = closed_rows(st, s)
            skips: dict = {}
            for o in rows:
                if o.get("status", "").startswith("skip"):
                    skips[o["status"]] = skips.get(o["status"], 0) + 1
            clv = [o["clv"] for o in done if o.get("clv") is not None]
            day = et_date(now)
            print(f"  {S['tag']:9s} {len(rows)} decisions, {sum(lx.committed(o) > 0 for o in rows)} with money, "
                  f"{len(done)} closed ({sum(o['pnl'] > 0 for o in done)} W) P&L ${sum(o['pnl'] for o in done):+.2f}; "
                  f"today ${spent(st, s, day):.2f}/${S['day_cap']:.0f}, total ${spent(st, s):.2f}/${S['season_cap']:.0f}"
                  + (f"; mean CLV {sum(clv) / len(clv):+.4f} (n {len(clv)})" if clv else "")
                  + (f"; DISABLED ({lx.disable_path(s).read_text().strip()})" if lx.disable_path(s).exists() else "")
                  + (f"; kill rule MET: {kill_verdict(st, s)}" if kill_verdict(st, s) else ""))
            if skips:
                print("            skips: " + ", ".join(f"{k} {v}" for k, v in sorted(skips.items())))
    return 0


def read_signals(text: str) -> list[dict]:
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    try:
        if cmd == "tick":
            return cmd_tick(read_signals("" if sys.stdin.isatty() else sys.stdin.read()))
        if cmd == "status":
            return cmd_status()
        if cmd == "kill":
            return lx.cmd_kill(argv[2:])
    except Exception as ex:  # noqa: BLE001
        msg = f"{type(ex).__name__}: {str(ex)[:300]}"
        try:
            lx.append({"event": "error", "ts": iso(datetime.now(timezone.utc)), "where": f"ingame_exec {cmd}",
                       "error": msg})
            kx.notify(f"Kalshi ingame exec ERROR ({cmd})", msg, kx.config(kx.env()))
        except Exception:  # noqa: BLE001
            pass
        print(f"ERROR {msg}")
        return 1
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
