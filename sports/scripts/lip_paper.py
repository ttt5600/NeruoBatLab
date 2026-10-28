"""L40 Kalshi Liquidity Incentive Program (LIP) PAPER test on daily-temp weather brackets. NO ORDERS ARE PLACED.

    python scripts/lip_paper.py tick        # launchd, every 60 s: refresh programs (hourly), poll books + tape, accrue
    python scripts/lip_paper.py report      # per policy x fill model: LIP, fills, settlement, net, vs the bars
    python scripts/lip_paper.py programs    # force a program-list refresh and print what is incentivised

Pre-registration: notes/kalshi_lip.md section 6 (10 days, 2026-10-29 00:00Z -> 2026-11-08 00:00Z). Everything below
is that plan; the "OPEN DETAILS" block records the choices this script makes where the plan was silent. All of it
was written down before the first logged snapshot (first tick 2026-10-28, a pre-test shakedown that is NOT scored).

SCORING (filed algorithm, CFTC amendment 2026-10-14, Appendix B; notes/kalshi_lip.md section 1)
  * Snapshot excluded if the market is closed or EITHER side has < Target Size resting (whole book, us included).
  * Per side (a YES ask is a NO bid; the API book is YES bids + NO bids): walk down from the best bid adding whole
    price levels until cumulative size >= Target -> qualifying bids. Reference price = first level where cumulative
    size >= Target/5. Score = size x DF^max(ref - price in 1c ticks, 0), DF = 0.5. Normalise per side to sum 1.
    Our snapshot score = our share of our side (we quote one side per order).
  * Payout = R x (sum over snapshots of our score) / (2 x all snapshots), floored to the cent; < $1.00 per program
    -> $0. Excluded snapshots stay in the denominator. Pro rata, both sides must hold Target or nobody is paid.
  * Our hypothetical order is INSERTED into the real book before scoring (it can move the reference price, truncate
    the qualifying set and complete a short side).

CADENCE: Kalshi samples once per second. We sample every ~60 s (launchd StartInterval 60). Scoring at that cadence is
an APPROXIMATION: each poll stands in for the ~60 s before it; orders that rest only between polls are invisible.
Basic-tier rate limit is 20 reads/s; one tick costs ~6 batched orderbook calls (100 tickers each) + ~10 pages of the
global trade tape + 1 markets call, paced at <= 8 req/s.

POLICIES (frozen in the note)
  P1 "E1 piggyback": every E1/E2 paper pick (data/kalshi_paper.jsonl, logged ~22:00 local D-1) -> NO x100 at the E4
     M3 price q = min(floor_cent(1 - (yes_bid + yes_ask)/2), no_ask - 1c), using the pick's logged quote; skip if
     q <= 0 or no_ask - q > 5c (the sealed M3 rule, notes/kalshi_temp_maker.md). Rests until close. phi 0.25.
     Baseline = the same fills with no LIP (reported as p1_fills_only).
  P2 "YES completer at 1c": YES x1,000 at 1c on every daily-temp bracket whose best YES bid is <= 2c. Replenish
     after fills. phi 1.0.
  P3 "join cheaper side x100": on every valid bracket, x100 at the best bid of the cheaper side. Replenish. phi 0.25.

FILLS: public trade tape (GET /markets/trades). A resting YES bid at y is hit by a taker buying NO (taker_side 'no')
at yes_price <= y; a resting NO bid at q by a taker buying YES at no_price <= q. Only prints after placement and at
or before close count. TWO fill models run side by side, each with its own order state and LIP accrual:
  * "phi" (the note's pre-registered model): fill += phi x print size for every print at or through our price.
  * "fifo" (queue model): we join the BACK of the queue at placement (queue_ahead = resting size at our price on our
    side in that poll's book, us excluded). A print strictly through our price fills our whole remainder; a print at
    our price first eats queue_ahead, then us. Cancellations ahead of us are NOT credited (conservative).
Filled contracts settle at the market result (YES bid: +(1-y) if YES else -y; NO bid: +(1-q) if NO else -q). Maker
fee 0 (weather series are fee_type quadratic: no maker fee).

SUCCESS (per policy; Bonferroni over 3 -> one-sided p < 0.0167):
  10-day net > 0 AND day-clustered bootstrap 95% lower bound > 0 AND bootstrap one-sided p < 0.0167 AND median
  per-program LIP >= $1 AND (P2 only) time-weighted valid fraction with us minus without us >= 0.30 (all hours).
KILL: net <= 0, or LB <= 0, or median time-weighted side share of a x100 order (P1, P3) < 5%, or Kalshi changes,
  pauses or removes the weather LIP (checked at every hourly program refresh; logged as a program_change event).

OPEN DETAILS -> CONSERVATIVE CHOICES (fixed before the first snapshot)
  1. Test cohort = programs whose start_date is in [2026-10-29 00:00Z, 2026-11-08 00:00Z): the 10 daily periods that
     start 10-29 14:00Z ... 11-07 14:00Z (events 10-30 ... 11-08). Programs that started before T0 are logged as a
     shakedown and never scored. The logger keeps running until the last cohort program ends and settles.
  2. Note's LIP formula "R x summed share / (2 N) x (valid/N)" is read as the filed formula (scores are 0 on
     excluded snapshots, so multiplying by valid/N again would double-count the exclusion). Snapshots we did NOT
     observe (Mac asleep, failed tick) are counted as excluded / zero for us: payout = R x sum(score x dt) /
     (2 x program window seconds). An observed-time-only estimate is reported alongside, never used for the bars.
  3. Each poll represents the interval since the previous poll of that market, capped at 120 s; an order placed at
     a poll earns nothing until the next poll; the score at a poll uses the size resting AFTER that poll's fills,
     and replenishment happens after scoring (so a partly-filled interval is credited at the smaller size).
  4. P2 eligibility is judged once, at the first poll at/after program start where the book has at least one YES
     and one NO bid (markets open at program start with empty books; an empty YES side is not "best bid <= 2c").
  5. P3 is placed once, at the first poll where the book is valid without us; the price is the cheaper side's best
     bid at that poll (lower price; tie -> NO) and it is never re-priced. Replenish keeps that price.
  6. P1 is NOT replenished (the note defines it as the x100 pick with a x100 baseline); LIP accrues on what still
     rests. P2 and P3 replenish to full size at the next poll after a fill (logger rule), the whole order re-joining
     the back of the queue in the fifo model.
  7. Bars must hold under BOTH fill models (pass = the worse of the two passes). Bootstrap LB = 2.5th percentile of
     10,000 day-resamples of the cohort's per-day net (stricter than a one-sided 5th percentile).
  8. Unsettled programs at report time are shown as pending and excluded from net (report says how many).
  9. Market list: GET /incentive_programs?status={active,upcoming}&type=liquidity (public), tickers starting
     KXHIGH or KXLOWT, refreshed hourly and within the 14:00-14:10Z window when new periods start.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://api.elections.kalshi.com/trade-api/v2"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATE = DATA / "lip_paper_state.json"
TICKS = DATA / "lip_paper_ticks.jsonl"
EVENTS = DATA / "lip_paper_events.jsonl"
PICKS = DATA / "kalshi_paper.jsonl"
LOCK = DATA / "lip_paper.lock"

T0 = datetime(2026, 10, 29, tzinfo=timezone.utc).timestamp()
T1 = datetime(2026, 11, 8, tzinfo=timezone.utc).timestamp()
CADENCE_S = 60
W_CAP = 2 * CADENCE_S
PREFIXES = ("KXHIGH", "KXLOWT")
EXPECTED = {"period_reward_usd": 20.0, "target": 1000.0, "df": 0.5}  # what the note found; any change -> event
REQ_GAP = 0.125  # <= 8 req/s, under the Basic tier's 20 reads/s
POLICIES = {
    "P1": {"size": 100, "replenish": False, "phi": 0.25},
    "P2": {"size": 1000, "replenish": True, "phi": 1.0, "side": "yes", "price": 1},
    "P3": {"size": 100, "replenish": True, "phi": 0.25},
}
MODELS = ("phi", "fifo")
ALPHA = 0.05 / 3
H = {"User-Agent": "Mozilla/5.0 (lip_paper; read-only)"}
_last_req = [0.0]
_nreq = [0]


def now_ts() -> float:
    return time.time()


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def pts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def get(path: str, params=None, tries: int = 5):
    for a in range(tries):
        wait = REQ_GAP - (time.time() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()
        _nreq[0] += 1
        try:
            r = requests.get(f"{API}{path}", params=params, headers=H, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2.0 * (a + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if a == tries - 1:
                raise
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"GET {path} failed after {tries} tries")


def log_event(kind: str, **kw):
    with EVENTS.open("a") as f:
        f.write(json.dumps({"ts": iso(now_ts()), "kind": kind, **kw}) + "\n")


# ---------------------------------------------------------------- scoring (filed algorithm)
def side_score(levels: dict, target: float, df: float, add=None):
    """levels {price_cents: size}; add (price_cents, size) = our order. -> (valid, W_total, our_weighted)."""
    lv = dict(levels)
    if add and add[1] > 0:
        lv[add[0]] = lv.get(add[0], 0.0) + add[1]
    cum, ref, qual = 0.0, None, []
    for p in sorted(lv, reverse=True):
        cum += lv[p]
        qual.append(p)
        if ref is None and cum >= target / 5:
            ref = p
        if cum >= target:
            break
    if cum < target:
        return False, 0.0, 0.0
    W = sum(lv[p] * df ** max(ref - p, 0) for p in qual)
    ours = add[1] * df ** max(ref - add[0], 0) if add and add[1] > 0 and add[0] in qual else 0.0
    return True, W, ours


def parse_book(ob: dict):
    o = ob.get("orderbook_fp") or ob.get("orderbook") or {}
    out = {}
    for side in ("yes", "no"):
        lv = {}
        for p, s in o.get(f"{side}_dollars") or []:
            c = round(float(p) * 100)
            lv[c] = lv.get(c, 0.0) + float(s)
        out[side] = lv
    return out


# ---------------------------------------------------------------- data pulls
def fetch_programs() -> dict:
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
                t = p.get("market_ticker") or ""
                if not t.startswith(PREFIXES):
                    continue
                progs[p["id"]] = {"ticker": t, "start": pts(p["start_date"]), "end": pts(p["end_date"]),
                                  "R": p["period_reward"] / 10000.0, "target": float(p["target_size_fp"]),
                                  "df": p["discount_factor_bps"] / 10000.0, "status": status,
                                  "desc": p.get("incentive_description")}
            cur = d.get("next_cursor") or ""
            if not cur or not ps:
                break
    return progs


def fetch_books(tickers: list[str]) -> dict:
    books = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        d = get("/markets/orderbooks", [("tickers", t) for t in chunk])
        for ob in d.get("orderbooks") or []:
            books[ob["ticker"]] = parse_book(ob)
    return books


def fetch_markets(tickers: list[str]) -> dict:
    out = {}
    for i in range(0, len(tickers), 100):
        d = get("/markets", {"tickers": ",".join(tickers[i:i + 100]), "limit": 1000})
        for m in d.get("markets") or []:
            out[m["ticker"]] = {"close": pts(m["close_time"]), "status": m.get("status"), "result": m.get("result")}
    return out


def _trade(t: dict):
    return {"id": t["trade_id"], "ticker": t["ticker"], "ts": pts(t["created_time"]), "taker": t["taker_side"],
            "yes": round(float(t["yes_price_dollars"]) * 100), "n": float(t["count_fp"])}


def fetch_trades(since: float, watch: set, order_tickers: set) -> tuple[list, str]:
    """Prints since `since` on watched tickers. Global tape for normal gaps; per-ticker for long gaps (asleep)."""
    out = []
    if now_ts() - since <= 20 * 60:
        cur, pages = "", 0
        while True:
            params = {"limit": 1000, "min_ts": int(since) - 1}
            if cur:
                params["cursor"] = cur
            d = get("/markets/trades", params)
            tr = d.get("trades") or []
            out += [_trade(t) for t in tr if t["ticker"] in watch]
            pages += 1
            cur = d.get("cursor") or ""
            if not cur or not tr or pages >= 200:
                break
        return out, f"global:{pages}p"
    for tk in sorted(order_tickers):
        cur = ""
        while True:
            params = {"ticker": tk, "limit": 1000, "min_ts": int(since) - 1}
            if cur:
                params["cursor"] = cur
            d = get("/markets/trades", params)
            tr = d.get("trades") or []
            out += [_trade(t) for t in tr]
            cur = d.get("cursor") or ""
            if not cur or not tr:
                break
    return out, f"per_ticker:{len(order_tickers)}"


# ---------------------------------------------------------------- state
def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"programs": {}, "prog_refreshed": 0.0, "markets": {}, "orders": {}, "pstats": {}, "last_tick": None,
            "last_poll": {}, "trades_since": None, "seen_trade_ids": [], "p1_seen_picks": [], "p2_judged": [],
            "p3_judged": [], "created": iso(now_ts())}


def save_state(st: dict):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, separators=(",", ":")))
    os.replace(tmp, STATE)


def refresh_programs(st: dict, force=False):
    t = now_ts()
    hh = datetime.now(timezone.utc)
    in_start_window = hh.hour == 14 and t - st["prog_refreshed"] > 5 * 60
    if not force and t - st["prog_refreshed"] < 3600 and not in_start_window:
        return
    progs = fetch_programs()
    live = [p for p in progs.values() if p["status"] == "active"]
    combos = sorted({(p["R"], p["target"], p["df"]) for p in live})
    changed = (not live) or combos != [(EXPECTED["period_reward_usd"], EXPECTED["target"], EXPECTED["df"])]
    if changed:
        log_event("program_change", n_active=len(live), combos=combos)
    old = st["programs"]
    for pid, p in progs.items():
        if pid in old:
            old[pid].update(p)
        else:
            old[pid] = {**p, "cohort": T0 <= p["start"] < T1}
    st["prog_refreshed"] = t
    st["prog_summary"] = {"ts": iso(t), "active": len(live), "upcoming": len(progs) - len(live), "combos": combos,
                          "changed": changed}


def order_key(pol, pid):
    return f"{pol}|{pid}"


def new_order(pol, pid, prog, side, price, size, t, book, extra=None):
    q_ahead = book[side].get(price, 0.0)
    o = {"pol": pol, "pid": pid, "ticker": prog["ticker"], "side": side, "price": price, "size": size,
         "placed": t, "cohort": prog.get("cohort", False), "day": iso(prog["start"])[:10],
         "v": {m: {"resting": float(size), "q_ahead": q_ahead, "filled": 0.0, "cost": 0.0, "score_w": 0.0,
                   "w": 0.0, "valid1_w": 0.0, "valid0_w": 0.0, "share_w": 0.0, "share_wt": 0.0, "cap_w": 0.0,
                   "replenished": 0} for m in MODELS}}
    if extra:
        o.update(extra)
    log_event("place", pol=pol, pid=pid, ticker=o["ticker"], side=side, price=price, size=size,
              q_ahead=q_ahead, cohort=o["cohort"], **(extra or {}))
    return o


def apply_trade(o: dict, tr: dict):
    if tr["ts"] <= o["placed"] or tr["ts"] > o.get("close", 1e18):
        return
    if o["side"] == "yes":
        if tr["taker"] != "no" or tr["yes"] > o["price"]:
            return
        through = tr["yes"] < o["price"]
    else:
        nop = 100 - tr["yes"]
        if tr["taker"] != "yes" or nop > o["price"]:
            return
        through = nop < o["price"]
    phi = POLICIES[o["pol"]]["phi"]
    for m in MODELS:
        v = o["v"][m]
        if v["resting"] <= 0:
            continue
        if m == "phi":
            f = min(v["resting"], phi * tr["n"])
        elif through:
            f = v["resting"]
        else:
            eat = min(v["q_ahead"], tr["n"])
            v["q_ahead"] -= eat
            f = min(v["resting"], tr["n"] - eat)
        if f > 0:
            v["resting"] -= f
            v["filled"] += f
            v["cost"] += f * o["price"] / 100.0


def m3_price(pick):
    yb, ya = float(pick["yes_bid"]), float(pick["yes_ask"])
    no_ask = 1 - yb
    q = min(math.floor(round((1 - (yb + ya) / 2) * 100, 6)) / 100, round(no_ask - 0.01, 2))
    if q <= 0 or no_ask - q > 0.05 + 1e-9:
        return None
    return round(q * 100)


def cmd_tick():
    DATA.mkdir(parents=True, exist_ok=True)
    lockf = LOCK.open("w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another tick is running")
        return
    st = load_state()
    t_start = now_ts()
    refresh_programs(st)
    t = now_ts()
    progs = st["programs"]
    live = {pid: p for pid, p in progs.items() if p["start"] <= t < p["end"] + 60}
    by_ticker = {}
    for pid, p in live.items():
        by_ticker.setdefault(p["ticker"], []).append(pid)
    tickers = sorted(by_ticker)
    # market close times (once per ticker)
    need = [tk for tk in tickers if tk not in st["markets"]]
    if need:
        st["markets"].update(fetch_markets(need))
    books = fetch_books(tickers)
    t = now_ts()  # snapshot time
    # ---- fills from the tape since the last tick
    open_orders = [o for o in st["orders"].values() if any(o["v"][m]["resting"] > 0 for m in MODELS)
                   and t - 600 < o.get("close", 1e18) + 600]
    order_tickers = {o["ticker"] for o in open_orders}
    since = st["trades_since"] or t
    trades, tape_mode = ([], "first") if st["trades_since"] is None else fetch_trades(since, set(tickers) | order_tickers,
                                                                                         order_tickers)
    seen = set(st["seen_trade_ids"])
    fresh = [tr for tr in trades if tr["id"] not in seen]
    fresh.sort(key=lambda x: x["ts"])
    by_tk = {}
    for o in open_orders:
        by_tk.setdefault(o["ticker"], []).append(o)
    nfill_prints = 0
    for tr in fresh:
        for o in by_tk.get(tr["ticker"], []):
            before = o["v"]["phi"]["filled"] + o["v"]["fifo"]["filled"]
            apply_trade(o, tr)
            nfill_prints += (o["v"]["phi"]["filled"] + o["v"]["fifo"]["filled"]) > before
    # next fetch starts 15 s before this snapshot (overlap absorbs late-stamped prints; ids dedupe the overlap)
    st["trades_since"] = t - 15
    st["seen_trade_ids"] = [tr["id"] for tr in trades if tr["ts"] >= t - 20]
    # ---- score every live program; accrue existing orders; then replenish; then place
    picks = []
    if PICKS.exists():
        for line in PICKS.read_text().splitlines():
            try:
                picks.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    pick_by_ticker = {}
    seen_p1 = set(st["p1_seen_picks"])
    for pk in picks:
        if pk.get("ticker") and f"{pk['ticker']}|{pk.get('logged_at')}" not in seen_p1:
            pick_by_ticker.setdefault(pk["ticker"], []).append(pk)
    judged2, judged3 = set(st["p2_judged"]), set(st["p3_judged"])
    n_valid0 = n_scored = n_placed = 0
    for pid, p in live.items():
        tk = p["ticker"]
        bk = books.get(tk)
        mk = st["markets"].get(tk, {})
        close = min(mk.get("close", p["end"]), p["end"])
        if bk is None:
            continue
        T, df = p["target"], p["df"]
        is_open = t < close
        vy0 = side_score(bk["yes"], T, df)[0]
        vn0 = side_score(bk["no"], T, df)[0]
        valid0 = is_open and vy0 and vn0
        n_valid0 += valid0
        prev = st["last_poll"].get(pid)
        lo = max(prev if prev is not None else t, p["start"])
        hi = min(t, p["end"])
        w = max(0.0, min(hi - lo, W_CAP))
        ps = st["pstats"].setdefault(pid, {"w": 0.0, "valid0_w": 0.0, "polls": 0})
        ps["w"] += w
        ps["valid0_w"] += w * valid0
        ps["polls"] += 1
        for pol in POLICIES:
            o = st["orders"].get(order_key(pol, pid))
            if o is None:
                continue
            o["close"] = close
            lo_o = max(lo, o["placed"])
            w_o = max(0.0, min(hi - lo_o, W_CAP)) if prev is not None or o["placed"] < t else 0.0
            for m in MODELS:
                v = o["v"][m]
                r = v["resting"] if is_open else 0.0
                add = (o["price"], r) if r > 0 else None
                sy = side_score(bk["yes"], T, df, add if o["side"] == "yes" else None)
                sn = side_score(bk["no"], T, df, add if o["side"] == "no" else None)
                valid1 = is_open and sy[0] and sn[0]
                mine = sy if o["side"] == "yes" else sn
                share = mine[2] / mine[1] if mine[0] and mine[1] > 0 else 0.0
                v["score_w"] += w_o * (share if valid1 else 0.0)
                v["w"] += w_o
                v["valid1_w"] += w_o * valid1
                v["valid0_w"] += w_o * valid0
                if mine[0] and r > 0:
                    v["share_w"] += w_o * share
                    v["share_wt"] += w_o
                v["cap_w"] += w_o * (r * o["price"] / 100.0 + v["cost"])
                # replenish after scoring (P2, P3)
                if POLICIES[pol]["replenish"] and is_open and v["resting"] < o["size"]:
                    v["resting"] = float(o["size"])
                    v["q_ahead"] = bk[o["side"]].get(o["price"], 0.0)
                    v["replenished"] += 1
            n_scored += 1
        if is_open:
            # P1: E1/E2 paper picks on this ticker
            for pk in pick_by_ticker.get(tk, []):
                key = f"{tk}|{pk.get('logged_at')}"
                seen_p1.add(key)
                if order_key("P1", pid) in st["orders"]:
                    continue
                q = m3_price(pk) if pk.get("yes_bid") is not None and pk.get("yes_ask") is not None else None
                if q is None:
                    log_event("p1_skip", ticker=tk, pick=pk.get("logged_at"), why="m3_rule")
                    continue
                o = new_order("P1", pid, p, "no", q, POLICIES["P1"]["size"], t, bk,
                              {"pick_logged_at": pk.get("logged_at"), "cohort_pick": pk.get("cohort")})
                o["close"] = close
                st["orders"][order_key("P1", pid)] = o
                n_placed += 1
            # P2: judged once, at the first poll with both sides quoted
            if pid not in judged2 and bk["yes"] and bk["no"]:
                judged2.add(pid)
                if max(bk["yes"]) <= 2:
                    o = new_order("P2", pid, p, "yes", 1, POLICIES["P2"]["size"], t, bk,
                                  {"yes_best_at_place": max(bk["yes"])})
                    o["close"] = close
                    st["orders"][order_key("P2", pid)] = o
                    n_placed += 1
            # P3: placed once, at the first valid poll
            if pid not in judged3 and valid0:
                judged3.add(pid)
                yb, nb = max(bk["yes"]), max(bk["no"])
                side, price = ("yes", yb) if yb < nb else ("no", nb)
                o = new_order("P3", pid, p, side, price, POLICIES["P3"]["size"], t, bk, {"yes_best": yb, "no_best": nb})
                o["close"] = close
                st["orders"][order_key("P3", pid)] = o
                n_placed += 1
        st["last_poll"][pid] = t
    st["p1_seen_picks"] = sorted(seen_p1)
    st["p2_judged"], st["p3_judged"] = sorted(judged2), sorted(judged3)
    # ---- finalize LIP for ended programs; settle filled contracts
    settled = finalize(st, t)
    # prune bookkeeping for programs long gone (orders kept)
    for pid in [k for k, p in progs.items() if p["end"] < t - 3 * 86400]:
        st["last_poll"].pop(pid, None)
        if not progs[pid].get("cohort"):
            st["pstats"].pop(pid, None)
    st["last_tick"] = iso(t)
    save_state(st)
    rec = {"ts": iso(t), "live_programs": len(live), "books": len(books), "valid0": n_valid0,
           "orders_scored": n_scored, "placed": n_placed, "prints_new": len(fresh), "fill_prints": nfill_prints,
           "tape": tape_mode, "settled": settled, "n_orders": len(st["orders"]), "reqs": _nreq[0],
           "secs": round(now_ts() - t_start, 1)}
    with TICKS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    if n_placed or settled or tape_mode.startswith("per_ticker"):
        print(json.dumps(rec))


def finalize(st: dict, t: float) -> int:
    progs = st["programs"]
    todo = [o for o in st["orders"].values() if "result" not in o and o.get("close", 1e18) < t]
    n = 0
    if todo:
        mk = fetch_markets(sorted({o["ticker"] for o in todo}))
        for o in todo:
            m = mk.get(o["ticker"]) or {}
            res = m.get("result")
            if res not in ("yes", "no"):
                continue
            o["result"] = res
            win = (o["side"] == res)
            for mdl in MODELS:
                v = o["v"][mdl]
                v["settle_pnl"] = round(v["filled"] * (1.0 if win else 0.0) - v["cost"], 4)
            n += 1
            log_event("settle", pol=o["pol"], ticker=o["ticker"], result=res,
                      filled={m_: o["v"][m_]["filled"] for m_ in MODELS})
    for o in st["orders"].values():
        p = progs.get(o["pid"])
        if p is None or "lip" in o or t < p["end"] + 120:
            continue
        win_s = p["end"] - p["start"]
        ps = st["pstats"].get(o["pid"], {})
        o["lip"], o["lip_obs"] = {}, {}
        for mdl in MODELS:
            v = o["v"][mdl]
            raw = p["R"] * v["score_w"] / (2 * win_s)
            o["lip"][mdl] = lip_round(raw)
            obs = ps.get("w") or 0
            o["lip_obs"][mdl] = lip_round(p["R"] * v["score_w"] / (2 * obs)) if obs else 0.0
        o["lip_raw"] = {mdl: round(p["R"] * o["v"][mdl]["score_w"] / (2 * win_s), 4) for mdl in MODELS}
        o["coverage"] = round((ps.get("w") or 0) / win_s, 3)
    return n


def lip_round(x: float) -> float:
    x = math.floor(x * 100 + 1e-9) / 100
    return x if x >= 1.0 else 0.0


# ---------------------------------------------------------------- report
def boot(day_net: dict, n=10000, seed=0):
    days = sorted(day_net)
    if len(days) < 2:
        return None, None
    rng = random.Random(seed)
    vals = [day_net[d] for d in days]
    sums = sorted(sum(rng.choice(vals) for _ in days) for _ in range(n))
    lb = sums[int(0.025 * n)]
    p = sum(s <= 0 for s in sums) / n
    return lb, p


def med(xs):
    xs = sorted(xs)
    if not xs:
        return None
    k = len(xs)
    return xs[k // 2] if k % 2 else (xs[k // 2 - 1] + xs[k // 2]) / 2


def cmd_report(include_shakedown=False):
    st = load_state()
    t = now_ts()
    print(f"L40 LIP paper  last tick {st.get('last_tick')}  programs {st.get('prog_summary')}")
    ticks = TICKS.read_text().splitlines() if TICKS.exists() else []
    if ticks:
        last = json.loads(ticks[-1])
        print(f"ticks logged {len(ticks)}; last: {last}")
    cohort_label = "SHAKEDOWN (pre-T0, never scored)" if include_shakedown else "COHORT 2026-10-29..11-07 starts"
    print(f"\n== {cohort_label}; phase: {'pre-test' if t < T0 else 'running' if t < T1 else 'collecting/settling'}")
    sel = [o for o in st["orders"].values() if (o.get("cohort") != include_shakedown)]
    pstats = st.get("pstats", {})
    verdicts = {}
    for pol in POLICIES:
        os_ = [o for o in sel if o["pol"] == pol]
        if not os_:
            print(f"\n{pol}: no orders yet")
            continue
        print(f"\n{pol}  orders {len(os_)}  ({sum('lip' in o for o in os_)} programs ended, "
              f"{sum('result' in o for o in os_)} settled)")
        res = {}
        for mdl in MODELS:
            done = [o for o in os_ if "lip" in o and "result" in o]
            lip = sum(o["lip"][mdl] for o in done)
            lip_obs = sum(o["lip_obs"][mdl] for o in done)
            fill_n = sum(o["v"][mdl]["filled"] for o in done)
            cost = sum(o["v"][mdl]["cost"] for o in done)
            sp = sum(o["v"][mdl]["settle_pnl"] for o in done)
            net = lip + sp
            day_net = {}
            for o in done:
                day_net[o["day"]] = day_net.get(o["day"], 0.0) + o["lip"][mdl] + o["v"][mdl]["settle_pnl"]
            nd = len(day_net)
            capd = sum(o["v"][mdl]["cap_w"] for o in done) / 86400
            lb, p = boot(day_net)
            mlip = med([o["lip"][mdl] for o in done])
            mraw = med([o["lip_raw"][mdl] for o in done])
            shares = [o["v"][mdl]["share_w"] / o["v"][mdl]["share_wt"] for o in done if o["v"][mdl]["share_wt"] > 0]
            msh = med(shares)
            w = sum(o["v"][mdl]["w"] for o in done)
            upl = (sum(o["v"][mdl]["valid1_w"] for o in done) - sum(o["v"][mdl]["valid0_w"] for o in done)) / w if w else None
            # running (not yet ended) accrual estimate
            run = [o for o in os_ if "lip" not in o]
            print(f"  [{mdl:4s}] done {len(done)}: LIP ${lip:.2f} (obs-time est ${lip_obs:.2f}) | fills {fill_n:.0f} "
                  f"cost ${cost:.2f} | settle P&L ${sp:+.2f} | NET ${net:+.2f} over {nd} days = ${net / nd if nd else 0:+.2f}/day | "
                  f"$/$-day {net / capd if capd else float('nan'):+.4f} | LB95 {fmt(lb)} p {fmt(p, 4)} | "
                  f"median LIP/program ${fmt(mlip)} (pre-$1-min ${fmt(mraw)}) | median share {fmt(msh, 3)} | "
                  f"valid uplift {fmt(upl, 3)} | running {len(run)}")
            if pol == "P1":
                print(f"         P1 baseline (fills only, no LIP): ${sp:+.2f}; LIP adds ${lip:+.2f}")
            ok = (nd > 0 and net > 0 and lb is not None and lb > 0 and p is not None and p < ALPHA
                  and mlip is not None and mlip >= 1.0 and (pol != "P2" or (upl is not None and upl >= 0.30)))
            kill = (nd > 0 and (net <= 0 or (lb is not None and lb <= 0)
                                or (pol in ("P1", "P3") and msh is not None and msh < 0.05)))
            res[mdl] = "PASS" if ok else "KILL" if kill else "open"
        final = t >= T1 and all("result" in o and "lip" in o for o in os_)
        v = "PASS" if all(r == "PASS" for r in res.values()) else "KILL" if any(r == "KILL" for r in res.values()) else "open"
        verdicts[pol] = v
        print(f"  verdict ({'FINAL' if final else 'INTERIM -- bars apply only after 11-08 + settlement'}): {v} {res}")
    chg = [json.loads(l) for l in (EVENTS.read_text().splitlines() if EVENTS.exists() else [])]
    chg = [e for e in chg if e["kind"] == "program_change"]
    if chg:
        print(f"\nKILL WATCH: {len(chg)} program_change events; last {chg[-1]}")
    cov = [ps["w"] / (st["programs"][pid]["end"] - st["programs"][pid]["start"])
           for pid, ps in pstats.items() if pid in st["programs"] and st["programs"][pid].get("cohort") != include_shakedown
           and st["programs"][pid]["end"] < t]
    if cov:
        print(f"observation coverage of ended programs: median {med(cov):.2f} min {min(cov):.2f} (n={len(cov)})")


def fmt(x, nd=2):
    return "na" if x is None else f"{x:.{nd}f}"


def cmd_programs():
    st = load_state()
    refresh_programs(st, force=True)
    save_state(st)
    print(json.dumps(st["prog_summary"]))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tick"
    if cmd == "tick":
        cmd_tick()
    elif cmd == "report":
        cmd_report(include_shakedown="--shakedown" in sys.argv)
    elif cmd == "programs":
        cmd_programs()
    else:
        sys.exit(__doc__)
