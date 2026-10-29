"""L40b Kalshi Liquidity Incentive Program (LIP) PAPER test on NON-weather markets. NO ORDERS ARE PLACED, EVER.

    python scripts/lip_paper_b.py sample      # once, before T0: draw + freeze the sample -> data/lip_b_sample.json
    python scripts/lip_paper_b.py tick        # launchd, every 60 s: refresh programs (hourly), poll books + tape, accrue
    python scripts/lip_paper_b.py report      # per policy x fill model x stratum: LIP, fills, fees, net, vs the bars
    python scripts/lip_paper_b.py programs    # force a program refresh for the sampled tickers and print a summary

Pre-registration: notes/kalshi_lip_b.md (filed before the sample was drawn and before any snapshot). Window
T0 = 2026-10-30 00:00Z -> T1 = 2026-11-09 00:00Z. Scoring, book parsing, tape and market pulls are imported from
scripts/lip_paper.py (L40) so both tests use the same filed algorithm and the same two fill models (phi, fifo).

Differences from L40 (all in the note):
  * Universe = a frozen seed-4040 random sample of non-weather market tickers (<= 20% per series), split by
    close_time into PRIMARY (<= T0 + 30 d, scored against the bars) and SECONDARY (long-dated, LIP + mark-to-mid only).
  * Per-program R / Target / DF and period length; P2 size = T, P3 size = 100 x T / 1000.
  * Orders keyed per (policy, ticker, episode); every live program on the ticker accrues from the same order.
  * We quote only inside [T0, T1), so a program's payout to us = R x sum(share x dt) / (2 x full program length).
  * Maker fee on series whose fee_type is not 'quadratic' (0.0175 x mult x n x p(1-p), ceil to cent per print).
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lip_paper as L  # noqa: E402  (reuse: get, side_score, parse_book, fetch_books, fetch_markets, fetch_trades, ...)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SAMPLE = DATA / "lip_b_sample.json"
STATE = DATA / "lip_b_state.json"
TICKS = DATA / "lip_b_ticks.jsonl"
EVENTS = DATA / "lip_b_events.jsonl"
LOCK = DATA / "lip_b.lock"

T0 = datetime(2026, 10, 30, tzinfo=timezone.utc).timestamp()
T1 = datetime(2026, 11, 9, tzinfo=timezone.utc).timestamp()
LONG_DATED = T0 + 30 * 86400
SEED = 4040
QUOTA = {"primary": 240, "secondary": 60}
SERIES_CAP = 0.20
MIN_PERIOD_S = 0.9 * 86400
WEATHER_CATEGORY = "Climate and Weather"
W_CAP = L.W_CAP
POLICIES = {"P2": {"phi": 1.0, "replenish": True}, "P3": {"phi": 0.25, "replenish": True}}
assert all(L.POLICIES[p]["phi"] == POLICIES[p]["phi"] for p in POLICIES)  # L.apply_trade reads phi from L.POLICIES
MODELS = L.MODELS
ALPHA = 0.05 / 2
MAKER_FEE = 0.0175
KILL_CHANGE_FRAC = 0.20

iso, pts, now_ts, get = L.iso, L.pts, L.now_ts, L.get


def day_of(ts: float) -> str:
    return iso(ts)[:10]


def log_event(kind: str, **kw):
    DATA.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a") as f:
        f.write(json.dumps({"ts": iso(now_ts()), "kind": kind, **kw}) + "\n")


def series_of(ticker: str) -> str:
    return ticker.split("-")[0]


# ---------------------------------------------------------------- pulls (programs for ALL tickers; L.fetch_programs filters weather)
def prog_rec(p: dict, status: str) -> dict:
    return {"ticker": p.get("market_ticker") or "", "start": pts(p["start_date"]), "end": pts(p["end_date"]),
            "R": p["period_reward"] / 10000.0, "target": float(p["target_size_fp"]),
            "df": p["discount_factor_bps"] / 10000.0, "status": status, "desc": p.get("incentive_description")}


def fetch_all_programs() -> dict:
    progs = {}
    for status in ("active", "upcoming"):
        cur = ""
        while True:
            params = {"status": status, "type": "liquidity", "limit": 10000}
            if cur:
                params["cursor"] = cur
            d = get("/incentive_programs", params)
            ps = d.get("incentive_programs") or []
            for p in ps:
                if p.get("market_ticker"):
                    progs[p["id"]] = prog_rec(p, status)
            cur = d.get("next_cursor") or ""
            if not cur or not ps:
                break
    return progs


def fetch_series() -> dict:
    out, cur = {}, ""
    while True:
        params = {"limit": 10000}
        if cur:
            params["cursor"] = cur
        d = get("/series", params)
        ss = d.get("series") or []
        for s in ss:
            out[s["ticker"]] = {"category": s.get("category"), "fee_type": s.get("fee_type"),
                                "fee_multiplier": float(s.get("fee_multiplier") or 1)}
        cur = d.get("cursor") or ""
        if not cur or not ss:
            break
    return out


# ---------------------------------------------------------------- sample (pure part is testable)
def eligible_programs(progs: dict, series: dict, t0=T0, t1=T1) -> dict:
    out = {}
    for pid, p in progs.items():
        s = series.get(series_of(p["ticker"]))
        if s is None or s.get("category") == WEATHER_CATEGORY:
            continue
        if p["end"] - p["start"] < MIN_PERIOD_S or not (p["start"] < t1 and p["end"] > t0 + 3600):
            continue
        out[pid] = p
    return out


def draw_sample(tickers_by_stratum: dict, quota: dict, seed=SEED, cap=SERIES_CAP) -> dict:
    """Outcome-free stratified draw: sorted -> seeded shuffle -> take unless the series is at its cap."""
    rng = random.Random(seed)
    out = {}
    for stratum in sorted(quota):
        pool = sorted(tickers_by_stratum.get(stratum, []))
        rng.shuffle(pool)
        lim = max(1, math.floor(cap * quota[stratum]))
        cnt, take = {}, []
        for tk in pool:
            if len(take) >= quota[stratum]:
                break
            s = series_of(tk)
            if cnt.get(s, 0) >= lim:
                continue
            cnt[s] = cnt.get(s, 0) + 1
            take.append(tk)
        out[stratum] = sorted(take)
    return out


def cmd_sample(path: Path = SAMPLE):
    if path.exists():
        sys.exit(f"{path} exists: the sample is frozen and is never redrawn")
    t = now_ts()
    if t >= T0:
        log_event("sample_after_T0", note="sample drawn after T0")
    progs = fetch_all_programs()
    series = fetch_series()
    el = eligible_programs(progs, series)
    tickers = sorted({p["ticker"] for p in el.values()})
    mk = L.fetch_markets(tickers)
    by = {"primary": [], "secondary": []}
    dropped = {"no_market": 0, "closed": 0}
    for tk in tickers:
        m = mk.get(tk)
        if m is None:
            dropped["no_market"] += 1
            continue
        if m.get("status") not in ("active", "initialized", "open") or m["close"] <= T0:
            dropped["closed"] += 1
            continue
        by["primary" if m["close"] <= LONG_DATED else "secondary"].append(tk)
    pick = draw_sample(by, QUOTA)
    tk_info = {}
    for stratum, tks in pick.items():
        for tk in tks:
            s = series[series_of(tk)]
            tk_info[tk] = {"stratum": stratum, "close": mk[tk]["close"], "series": series_of(tk),
                           "category": s["category"], "fee_type": s["fee_type"], "fee_multiplier": s["fee_multiplier"]}
    sp = {pid: p for pid, p in el.items() if p["ticker"] in tk_info}
    mix = {}
    for tk, v in tk_info.items():
        mix.setdefault(v["stratum"], {}).setdefault(v["series"], 0)
        mix[v["stratum"]][v["series"]] += 1
    out = {"drawn_at": iso(t), "seed": SEED, "T0": iso(T0), "T1": iso(T1), "long_dated_after": iso(LONG_DATED),
           "quota": QUOTA, "series_cap": SERIES_CAP, "n_programs_listed": len(progs), "n_eligible_programs": len(el),
           "n_eligible_tickers": {k: len(v) for k, v in by.items()}, "dropped": dropped,
           "tickers": tk_info, "programs_at_draw": sp,
           "series_mix": {k: dict(sorted(v.items(), key=lambda x: -x[1])) for k, v in mix.items()}}
    DATA.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    log_event("sample", n={k: len(v) for k, v in pick.items()}, eligible=out["n_eligible_tickers"])
    print(json.dumps({k: out[k] for k in ("drawn_at", "n_programs_listed", "n_eligible_programs", "n_eligible_tickers",
                                          "dropped")}))
    for k, v in out["series_mix"].items():
        print(f"{k}: {sum(v.values())} tickers, {len(v)} series; top {list(v.items())[:8]}")


def load_sample() -> dict:
    if not SAMPLE.exists():
        sys.exit("no frozen sample: run `lip_paper_b.py sample` first")
    return json.loads(SAMPLE.read_text())


# ---------------------------------------------------------------- state
def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"programs": {}, "prog_refreshed": 0.0, "markets": {}, "orders": {}, "pstats": {}, "last_poll": {},
            "trades_since": None, "seen_trade_ids": [], "judged": {}, "episode": {}, "was_live": {}, "mid": {},
            "changed_tickers": [], "last_tick": None, "created": iso(now_ts())}


def save_state(st: dict):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, separators=(",", ":")))
    os.replace(tmp, STATE)


def merge_programs(st: dict, fresh: dict, sampled: set, t: float) -> list:
    """Fold a program pull into state for the sampled tickers. Returns the change events (also logged)."""
    evs = []
    old = st["programs"]
    for pid, p in fresh.items():
        if p["ticker"] not in sampled:
            continue
        if pid not in old:
            old[pid] = dict(p)
            if st["prog_refreshed"]:
                evs.append({"kind": "program_new", "pid": pid, "ticker": p["ticker"], "R": p["R"],
                            "target": p["target"], "df": p["df"], "start": iso(p["start"]), "end": iso(p["end"])})
            continue
        o = old[pid]
        diff = {k: [o[k], p[k]] for k in ("R", "target", "df", "start", "end") if abs(o[k] - p[k]) > 1e-9}
        if diff:  # kill-relevant only if R / Target / DF moved; a start/end move alone is logged as program_retimed
            kind = "program_change" if set(diff) & {"R", "target", "df"} else "program_retimed"
            evs.append({"kind": kind, "pid": pid, "ticker": p["ticker"], "diff": diff})
        o.update(p)
        o.pop("missing", None)
    for pid, o in old.items():
        if pid not in fresh and not o.get("missing") and t < o["end"] - 60:
            o["missing"] = t
            evs.append({"kind": "program_change", "pid": pid, "ticker": o["ticker"], "diff": "removed_before_end",
                        "end": iso(o["end"])})
    for e in evs:
        if e["kind"] == "program_change" and e["ticker"] not in st["changed_tickers"]:
            st["changed_tickers"].append(e["ticker"])
        log_event(**e)
    return evs


def refresh(st: dict, sample: dict, force=False):
    t = now_ts()
    if not force and t - st["prog_refreshed"] < 3600:
        return
    if not st["programs"]:  # seed with the programs seen at draw time, so a removal before the first tick is caught
        st["programs"] = {pid: dict(p) for pid, p in sample["programs_at_draw"].items()}
        st["prog_refreshed"] = 0.0
    fresh = fetch_all_programs()
    sampled = set(sample["tickers"])
    merge_programs(st, fresh, sampled, t)
    st["markets"].update(L.fetch_markets(sorted(sampled)))
    st["prog_refreshed"] = t
    live = [p for p in st["programs"].values() if p["ticker"] in sampled and p["start"] <= t < p["end"]
            and not p.get("missing")]
    st["prog_summary"] = {"ts": iso(t), "listed_total": len(fresh), "sampled_programs": len(st["programs"]),
                          "live_on_sample": len(live), "tickers_live": len({p["ticker"] for p in live}),
                          "combos": sorted({(p["R"], p["target"], p["df"]) for p in live})[:20]}


# ---------------------------------------------------------------- orders, fills, accrual (pure functions, tested)
def new_order(pol, ticker, ep, side, price, size, t, book, info, extra=None):
    q_ahead = book[side].get(price, 0.0)
    o = {"pol": pol, "ticker": ticker, "ep": ep, "side": side, "price": price, "size": float(size), "placed": t,
         "end": None, "close": info.get("close", 1e18), "stratum": info["stratum"], "fee_type": info.get("fee_type"),
         "fee_mult": info.get("fee_multiplier", 1.0),
         "v": {m: {"resting": float(size), "q_ahead": q_ahead, "filled": 0.0, "cost": 0.0, "fee": 0.0,
                   "score": {}, "fills_day": {}, "w": 0.0, "share_w": 0.0, "share_wt": 0.0, "cap_w": 0.0,
                   "replenished": 0} for m in MODELS}}
    if extra:
        o.update(extra)
    log_event("place", pol=pol, ticker=ticker, ep=ep, side=side, price=price, size=size, q_ahead=q_ahead,
              stratum=info["stratum"], **(extra or {}))
    return o


def fee_for(n: float, price_c: int, fee_type, mult=1.0) -> float:
    if n <= 0 or fee_type == "quadratic":
        return 0.0
    p = price_c / 100.0
    return math.ceil(round(MAKER_FEE * mult * n * p * (1 - p) * 100, 6)) / 100


def apply_print(o: dict, tr: dict):
    """L.apply_trade (same fill models) + per-day fill/cost/fee bookkeeping."""
    before = {m: (o["v"][m]["filled"], o["v"][m]["cost"]) for m in MODELS}
    L.apply_trade(o, tr)
    d = day_of(tr["ts"])
    for m in MODELS:
        v = o["v"][m]
        df_ = v["filled"] - before[m][0]
        if df_ > 0:
            fee = fee_for(df_, o["price"], o["fee_type"], o["fee_mult"])
            v["fee"] += fee
            row = v["fills_day"].setdefault(d, [0.0, 0.0, 0.0])
            row[0] += df_
            row[1] += v["cost"] - before[m][1]
            row[2] += fee


def accrue(o: dict, book: dict, progs_live: dict, ws: dict, t: float, w_order: float, is_open: bool):
    """Score the order against every live program on its ticker, each with its own T/DF and its own interval
    ws[pid]; book side share (vs the largest-T program), capital, then replenish (after scoring, as in L40)."""
    d = day_of(t)
    pmax = max(progs_live.values(), key=lambda p: p["target"]) if progs_live else None
    for m in MODELS:
        v = o["v"][m]
        r = v["resting"] if is_open else 0.0
        add = (o["price"], r) if r > 0 else None
        share_max = None
        for pid, p in progs_live.items():
            sy = L.side_score(book["yes"], p["target"], p["df"], add if o["side"] == "yes" else None)
            sn = L.side_score(book["no"], p["target"], p["df"], add if o["side"] == "no" else None)
            mine = sy if o["side"] == "yes" else sn
            share = mine[2] / mine[1] if mine[0] and mine[1] > 0 else 0.0
            w = ws.get(pid, 0.0)
            if is_open and sy[0] and sn[0] and w > 0 and share > 0:
                sc = v["score"].setdefault(pid, {})
                sc[d] = sc.get(d, 0.0) + w * share
            if p is pmax and mine[0]:
                share_max = share
        v["w"] += w_order
        if r > 0 and share_max is not None:
            v["share_w"] += w_order * share_max
            v["share_wt"] += w_order
        v["cap_w"] += w_order * (r * o["price"] / 100.0 + v["cost"])
        if POLICIES[o["pol"]]["replenish"] and is_open and v["resting"] < o["size"]:
            v["resting"] = o["size"]
            v["q_ahead"] = book[o["side"]].get(o["price"], 0.0)
            v["replenished"] += 1


def program_lip(p: dict, score_by_day: dict) -> tuple[float, float, dict]:
    """-> (floored payout, raw payout, payout allocated to days pro rata to score)."""
    win_s = p["end"] - p["start"]
    tot = sum(score_by_day.values())
    raw = p["R"] * tot / (2 * win_s) if win_s > 0 else 0.0
    pay = L.lip_round(raw)
    alloc = {d: pay * s / tot for d, s in score_by_day.items()} if tot > 0 and pay > 0 else {}
    return pay, raw, alloc


def program_done(p: dict, t: float) -> bool:
    return t >= min(p["end"], T1) + 120


# ---------------------------------------------------------------- tick
def cmd_tick():
    DATA.mkdir(parents=True, exist_ok=True)
    lockf = LOCK.open("w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another tick is running")
        return
    sample = load_sample()
    info = sample["tickers"]
    st = load_state()
    t_start = now_ts()
    refresh(st, sample)
    tickers = sorted(info)
    books = L.fetch_books(tickers)
    t = now_ts()
    in_window = T0 <= t < T1
    progs = st["programs"]
    live = {pid: p for pid, p in progs.items() if p["start"] <= t < p["end"] and not p.get("missing")}
    live_by_tk = {}
    for pid, p in live.items():
        live_by_tk.setdefault(p["ticker"], {})[pid] = p
    # ---- fills since the last tick (only orders that were resting)
    open_orders = [o for o in st["orders"].values() if o["end"] is None]
    order_tickers = {o["ticker"] for o in open_orders}
    since = st["trades_since"] or t
    if st["trades_since"] is None or not order_tickers:
        trades, tape_mode = [], "skip"
    else:
        trades, tape_mode = L.fetch_trades(since, order_tickers, order_tickers)
    seen = set(st["seen_trade_ids"])
    fresh = sorted((tr for tr in trades if tr["id"] not in seen), key=lambda x: x["ts"])
    by_tk = {}
    for o in open_orders:
        by_tk.setdefault(o["ticker"], []).append(o)
    nfill = 0
    for tr in fresh:
        for o in by_tk.get(tr["ticker"], []):
            b = sum(o["v"][m]["filled"] for m in MODELS)
            apply_print(o, tr)
            nfill += sum(o["v"][m]["filled"] for m in MODELS) > b
    st["trades_since"] = t - 15
    st["seen_trade_ids"] = [tr["id"] for tr in trades if tr["ts"] >= t - 20]
    # ---- per ticker: program stats, accrue, cancel, place
    n_valid0 = n_scored = n_placed = n_cancel = 0
    for tk in tickers:
        bk = books.get(tk)
        mk = st["markets"].get(tk, {})
        close = mk.get("close", info[tk]["close"])
        is_open = t < close and mk.get("status", "active") in ("active", "open", "initialized")
        pl = live_by_tk.get(tk, {})
        if bk is not None and bk["yes"] and bk["no"]:
            st["mid"][tk] = (max(bk["yes"]) + 100 - max(bk["no"])) / 2.0
        ws = {}
        for pid, p in pl.items():
            prev = st["last_poll"].get(pid)
            lo = max(prev if prev is not None else t, p["start"], T0)
            hi = min(t, p["end"], T1)
            ws[pid] = max(0.0, min(hi - lo, W_CAP))
            if bk is not None:
                v0 = is_open and L.side_score(bk["yes"], p["target"], p["df"])[0] and \
                    L.side_score(bk["no"], p["target"], p["df"])[0]
                ps = st["pstats"].setdefault(pid, {"w": 0.0, "valid0_w": 0.0, "polls": 0})
                ps["w"] += ws[pid]
                ps["valid0_w"] += ws[pid] * v0
                ps["polls"] += 1
                n_valid0 += v0
            st["last_poll"][pid] = t
        w_tk = max(ws.values()) if ws else 0.0
        for o in [o for o in open_orders if o["ticker"] == tk]:
            if bk is not None and pl:
                accrue(o, bk, pl, ws, t, w_tk, is_open)
                n_scored += 1
            if not pl or not is_open or t >= T1:
                o["end"] = min(t, close, T1)
                o["close"] = o["end"]
                for m in MODELS:
                    o["v"][m]["resting"] = 0.0
                log_event("cancel", pol=o["pol"], ticker=tk, ep=o["ep"],
                          why="no_live_program" if not pl else "market_closed" if not is_open else "T1")
                n_cancel += 1
        # episode bookkeeping: a new episode starts when the ticker becomes live again
        was = st["was_live"].get(tk, False)
        if pl and not was:
            st["episode"][tk] = st["episode"].get(tk, 0) + 1
        st["was_live"][tk] = bool(pl)
        if not (in_window and pl and is_open and bk is not None):
            continue
        ep = st["episode"].get(tk, 1)
        T = max(p["target"] for p in pl.values())
        pmax = max(pl.values(), key=lambda p: p["target"])
        valid0 = L.side_score(bk["yes"], T, pmax["df"])[0] and L.side_score(bk["no"], T, pmax["df"])[0]
        jd = st["judged"].setdefault(tk, {})
        if jd.get("P2") != ep and bk["yes"] and bk["no"]:
            jd["P2"] = ep
            if max(bk["yes"]) <= 2:
                st["orders"][f"P2|{tk}|{ep}"] = new_order("P2", tk, ep, "yes", 1, T, t, bk, info[tk],
                                                          {"yes_best_at_place": max(bk["yes"]), "target": T})
                n_placed += 1
        if jd.get("P3") != ep and valid0:
            jd["P3"] = ep
            yb, nb = max(bk["yes"]), max(bk["no"])
            side, price = ("yes", yb) if yb < nb else ("no", nb)
            st["orders"][f"P3|{tk}|{ep}"] = new_order("P3", tk, ep, side, price, round(100 * T / 1000), t, bk,
                                                      info[tk], {"yes_best": yb, "no_best": nb, "target": T})
            n_placed += 1
    settled = finalize(st, t)
    for pid in [k for k, p in progs.items() if p["end"] < t - 3 * 86400]:
        st["last_poll"].pop(pid, None)
    st["last_tick"] = iso(t)
    save_state(st)
    rec = {"ts": iso(t), "phase": "pre" if t < T0 else "window" if t < T1 else "post", "sampled": len(tickers),
           "live_programs": len([p for p in live.values() if p["ticker"] in info]), "books": len(books),
           "valid0": n_valid0, "orders_scored": n_scored, "placed": n_placed, "cancelled": n_cancel,
           "prints_new": len(fresh), "fill_prints": nfill, "tape": tape_mode, "settled": settled,
           "n_orders": len(st["orders"]), "reqs": L._nreq[0], "secs": round(now_ts() - t_start, 1)}
    with TICKS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    if n_placed or settled or tape_mode.startswith("per_ticker") or rec["live_programs"] == 0:
        print(json.dumps(rec))


def finalize(st: dict, t: float) -> int:
    todo = [o for o in st["orders"].values() if o["end"] is not None and "result" not in o
            and any(o["v"][m]["filled"] > 0 for m in MODELS) and o["close"] <= t]
    # settle only once the market itself has a result (close_time of the market, not the order end)
    n = 0
    if todo:
        mk = L.fetch_markets(sorted({o["ticker"] for o in todo}))
        for o in todo:
            res = (mk.get(o["ticker"]) or {}).get("result")
            if res not in ("yes", "no"):
                continue
            o["result"] = res
            n += 1
            log_event("settle", pol=o["pol"], ticker=o["ticker"], ep=o["ep"], result=res,
                      filled={m: o["v"][m]["filled"] for m in MODELS})
    return n


def settle_by_day(o: dict, m: str):
    """-> per-day settlement P&L incl. fees, or None if filled and unsettled."""
    v = o["v"][m]
    if v["filled"] <= 0:
        return {}
    if "result" not in o:
        return None
    win = 1.0 if o["side"] == o["result"] else 0.0
    return {d: r[0] * win - r[1] - r[2] for d, r in v["fills_day"].items()}


# ---------------------------------------------------------------- report
def summarize(orders: list, progs: dict, m: str, t: float, mids=None) -> dict:
    lip, raw_l, lips, day_net = 0.0, [], [], {}
    pend_prog = 0
    for o in orders:
        for pid, sc in o["v"][m]["score"].items():
            p = progs.get(pid)
            if p is None:
                continue
            if not program_done(p, t):
                pend_prog += 1
                continue
            pay, raw, alloc = program_lip(p, sc)
            lip += pay
            lips.append(pay)
            raw_l.append(raw)
            for d, x in alloc.items():
                day_net[d] = day_net.get(d, 0.0) + x
    sp = fees = fills = cost = mtm = 0.0
    pending_fill = 0
    for o in orders:
        v = o["v"][m]
        fills += v["filled"]
        cost += v["cost"]
        fees += v["fee"]
        sd = settle_by_day(o, m)
        if sd is None:
            pending_fill += 1
            if mids is not None and o["ticker"] in mids:
                mid = mids[o["ticker"]] / 100.0
                val = mid if o["side"] == "yes" else 1 - mid
                mtm += v["filled"] * val - v["cost"] - v["fee"]
            continue
        for d, x in sd.items():
            sp += x
            day_net[d] = day_net.get(d, 0.0) + x
    days = {d: x for d, x in day_net.items() if iso(T0)[:10] <= d < iso(T1)[:10]}
    for d0 in range(int((T1 - T0) // 86400)):  # zero-net days still count as clusters once the window has passed them
        d = day_of(T0 + d0 * 86400)
        if T0 + (d0 + 1) * 86400 <= t:
            days.setdefault(d, 0.0)
    lb, p = L.boot(days)
    capd = sum(o["v"][m]["cap_w"] for o in orders) / 86400
    shares = [o["v"][m]["share_w"] / o["v"][m]["share_wt"] for o in orders if o["v"][m]["share_wt"] > 0]
    return {"lip": lip, "programs_done": len(lips), "programs_running": pend_prog, "median_lip": L.med(lips),
            "median_raw": L.med(raw_l), "fills": fills, "cost": cost, "fees": fees, "settle": sp,
            "pending_fill_orders": pending_fill, "mtm_pending": mtm, "net": lip + sp, "days": len(days),
            "lb": lb, "p": p, "capd": capd, "median_share": L.med(shares)}


def verdict(s: dict) -> str:
    ok = (s["days"] > 0 and s["net"] > 0 and s["lb"] is not None and s["lb"] > 0 and s["p"] is not None
          and s["p"] < ALPHA and s["median_lip"] is not None and s["median_lip"] >= 1.0)
    kill = s["days"] > 0 and (s["net"] <= 0 or (s["lb"] is not None and s["lb"] <= 0))
    return "PASS" if ok else "KILL" if kill else "open"


def cmd_report():
    sample = load_sample()
    st = load_state()
    t = now_ts()
    prim = [tk for tk, v in sample["tickers"].items() if v["stratum"] == "primary"]
    sec = [tk for tk, v in sample["tickers"].items() if v["stratum"] == "secondary"]
    print(f"L40b LIP paper (non-weather)  sample drawn {sample['drawn_at']} seed {sample['seed']}: "
          f"{len(prim)} primary + {len(sec)} secondary tickers; window {sample['T0']} -> {sample['T1']}")
    print(f"last tick {st.get('last_tick')}  programs {st.get('prog_summary')}")
    ticks = TICKS.read_text().splitlines() if TICKS.exists() else []
    if ticks:
        print(f"ticks logged {len(ticks)}; last: {ticks[-1]}")
    phase = "pre-test (before T0; nothing scored)" if t < T0 else "running" if t < T1 else "collecting/settling"
    print(f"phase: {phase}")
    progs = st["programs"]
    for stratum in ("primary", "secondary"):
        print(f"\n== {stratum.upper()} {'(scored against the bars)' if stratum == 'primary' else '(descriptive; fills MTM at mid)'}")
        for pol in POLICIES:
            os_ = [o for o in st["orders"].values() if o["pol"] == pol and o["stratum"] == stratum]
            if not os_:
                print(f"  {pol}: no orders yet")
                continue
            print(f"  {pol}  orders {len(os_)} ({sum(o['end'] is None for o in os_)} resting, "
                  f"{sum('result' in o for o in os_)} settled)")
            res = {}
            for m in MODELS:
                s = summarize(os_, progs, m, t, st.get("mid") if stratum == "secondary" else None)
                print(f"    [{m:4s}] LIP ${s['lip']:.2f} on {s['programs_done']} finished programs "
                      f"({s['programs_running']} running) | fills {s['fills']:.0f} cost ${s['cost']:.2f} fees "
                      f"${s['fees']:.2f} | settle P&L ${s['settle']:+.2f} ({s['pending_fill_orders']} filled orders "
                      f"unsettled{', MTM $%+.2f' % s['mtm_pending'] if stratum == 'secondary' else ''}) | NET "
                      f"${s['net']:+.2f} over {s['days']} days | $/$-day "
                      f"{s['net'] / s['capd'] if s['capd'] else float('nan'):+.4f} | LB95 {L.fmt(s['lb'])} p "
                      f"{L.fmt(s['p'], 4)} | median LIP/program ${L.fmt(s['median_lip'])} (pre-$1 "
                      f"${L.fmt(s['median_raw'])}) | median share {L.fmt(s['median_share'], 3)}")
                res[m] = verdict(s)
            if stratum == "primary":
                v = "PASS" if all(r == "PASS" for r in res.values()) else \
                    "KILL" if any(r == "KILL" for r in res.values()) else "open"
                final = t >= T1 and all("result" in o or all(o["v"][m]["filled"] <= 0 for m in MODELS) for o in os_)
                print(f"    verdict ({'FINAL' if final else 'INTERIM -- bars apply only after T1 + settlement'}): "
                      f"{v} {res}")
    ch = [tk for tk in st.get("changed_tickers", []) if tk in prim]
    frac = len(ch) / len(prim) if prim else 0.0
    print(f"\nKILL WATCH: program_change on {len(ch)}/{len(prim)} primary tickers ({frac:.1%}; kill at "
          f">= {KILL_CHANGE_FRAC:.0%}){' -> KILL' if frac >= KILL_CHANGE_FRAC else ''}")
    cov = [ps["w"] / (min(progs[pid]["end"], T1) - max(progs[pid]["start"], T0)) for pid, ps in st["pstats"].items()
           if pid in progs and min(progs[pid]["end"], T1) - max(progs[pid]["start"], T0) > 0
           and min(progs[pid]["end"], T1) < t]
    if cov:
        print(f"observation coverage of finished in-window program spans: median {L.med(cov):.2f} "
              f"min {min(cov):.2f} (n={len(cov)})")
    vf = [ps["valid0_w"] / ps["w"] for ps in st["pstats"].values() if ps["w"] > 0]
    if vf:
        print(f"valid fraction without us (in-window, per program): median {L.med(vf):.2f} (n={len(vf)})")


def cmd_programs():
    sample = load_sample()
    st = load_state()
    refresh(st, sample, force=True)
    save_state(st)
    print(json.dumps(st["prog_summary"]))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tick"
    if cmd == "tick":
        cmd_tick()
    elif cmd == "report":
        cmd_report()
    elif cmd == "programs":
        cmd_programs()
    elif cmd == "sample":
        cmd_sample()
    else:
        sys.exit(__doc__)
