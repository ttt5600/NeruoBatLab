"""E5 EXECUTOR: maker orders on ForecastEx (IBKR) for the FX1-fwd weather consensus-reject picks.

    python scripts/forecastex_exec.py run            # DRY-RUN: prints exactly the orders it WOULD place; sends nothing
    python scripts/forecastex_exec.py run --live     # place them (all gates below must also hold)
    python scripts/forecastex_exec.py status         # open orders, fills, positions, P&L, caps used, gates
    python scripts/forecastex_exec.py cancel-all     # cancel every open order this executor placed (needs a session)
    python scripts/forecastex_exec.py reconcile      # fills/positions from the gateway -> journals (read-only calls)
    python scripts/forecastex_exec.py settle         # settle filled orders from the ForecastEx results

NO NEW SELECTION LOGIC. The picks are the `pick` rows the FX1-fwd paper agent (forecastex_paper.py, frozen E5 rule
from forecastex_wx.py) logs at 22:00 station-local on D-1. This script only executes them:
  * due = FX1-fwd pick, logged <= FRESH_H hours ago, no exec decision yet for that ticker, before the order cutoff.
  * side = the pick's side (NO usually; YES if the rule picked YES). Book re-read from the session snapshot.
  * price = that side's CURRENT BEST BID (maker). No bid -> skip. Bid >= ask (would cross) -> skip. bid + fee >= 1
    -> skip (no positive payoff, the paper log's "cost < 1" filter).
  * size = largest whole contracts with: <= FX_MAX_CONTRACTS (10), cost <= FX_MAX_ORDER_USD ($5), cost <= tier-B 0.5u
    ($25), cost <= the FX_MAX_DAILY_USD ($25) left today (client journal, UTC day). Cost = n x (price + $0.01 fee).
    At a 0.94 bid that is 5 contracts / $4.75, so the $25 daily cap covers ~5 picks per night, in log order.
  * order: BUY LMT, GTC, post-only (client re-reads the ask just before sending). IBKR has no GTD on this API, so the
    expiry is enforced HERE: every run cancels any of our orders past cutoff = contract last_trade_time - CUTOFF_MIN.
  * one decision per ticker (tag E5X-<ticker>; the client also refuses a reused tag).
GATES (any failing -> no order): --live; KILL file; client gating (session, IBKR_MODE/account, FX caps); bet.py tier-B
  caps + B/C loss stops (bet.check_caps); executor daily loss stop: realized E5 P&L settled in the trailing 24 h
  <= -FX_EXEC_LOSS_STOP_USD ($10) halts new orders.
VERIFY: each placed order is re-read from the gateway (side BUY, conid of the bought side, limit price, size). Any
  mismatch -> cancel-all + touch KILL + stop.
JOURNALS: data/forecastex_exec.jsonl (decision / order / verify / fill / cancel / settle rows), bet.py's data/bets.jsonl
  (venue forecastex, tier B, strategy E5: bet -> update -> settle), plus the client's wire journal
  data/forecastex_orders.jsonl. Dry-run writes nothing.
SETTLEMENT: the bought side's outcome from the FX1-fwd paper log's `yes_won` (settled hourly by the paper agent from
  the forecastex.com daily price files); pnl = filled x (1 if won) - filled x price - commission (gateway commission if
  reported, else $0.01/contract).
Data home: FX_EXEC_HOME (default ~/.claude/quant-kalshi).
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from venues.base import TradingDisabled, VenueError  # noqa: E402
from venues.forecastex_client import DEPLOY, FEE, ForecastEx  # noqa: E402

HOME = Path(os.environ.get("FX_EXEC_HOME") or DEPLOY)
TRIAL = "FX1-fwd"
TIER, STRATEGY, VENUE = "B", "E5", "forecastex"
UNIT = 50.0
TIER_B_USD = 0.5 * UNIT
FRESH_H = 6
CUTOFF_MIN = 15
TAG = "E5X-"
DONE = {"filled", "cancelled", "canceled", "inactive", "expired", "rejected"}


def paths(home: Path | None = None) -> dict:
    h = Path(home or HOME) / "data"
    return {"picks": h / "forecastex_paper.jsonl", "log": h / "forecastex_exec.jsonl", "kill": Path(home or HOME) / "KILL"}


def _rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class Exec:
    def __init__(self, live: bool = False, client: ForecastEx | None = None, home: Path | None = None,
                 bet_mod=None, out=print):
        self.live = live
        self.p = paths(home)
        self.c = client or ForecastEx(live=live)
        self.out = out
        self.bet = bet_mod
        if self.bet is None and live:
            import bet as _bet  # unified router journal + tier caps (imports kalshi_exec; deploy copy has it)
            _bet.set_root(Path(home or HOME))
            self.bet = _bet
        self.loss_stop = float(self.c.env.get("FX_EXEC_LOSS_STOP_USD", 10.0))

    # ------------------------------------------------------------------ journal
    def log(self, rec: dict) -> dict:
        self.p["log"].parent.mkdir(parents=True, exist_ok=True)
        with self.p["log"].open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def rows(self) -> list[dict]:
        return _rows(self.p["log"])

    def orders(self) -> dict[str, dict]:
        """order_id -> folded order state (order row + verify/fill/cancel/settle rows)."""
        out: dict[str, dict] = {}
        for r in self.rows():
            oid = r.get("order_id")
            if r.get("kind") == "order" and oid:
                out[oid] = dict(r, filled=0.0, status="submitted", settled=False)
            elif oid in out and r.get("kind") in ("verify", "fill", "cancel", "status"):
                out[oid].update({k: v for k, v in r.items() if k in ("filled", "status", "avg_price", "commission")})
            elif oid in out and r.get("kind") == "settle":
                out[oid].update(settled=True, result=r["result"], pnl_usd=r["pnl_usd"], settled_ts=r["ts"])
        return out

    def _bet(self, rec: dict) -> None:
        if self.bet is not None:
            self.bet.append(rec)

    # ------------------------------------------------------------------ gates
    def killed(self) -> Path | None:
        k = self.c.killed()
        return k or (self.p["kill"] if self.p["kill"].exists() else None)

    def realized_24h(self, now: datetime) -> float:
        return sum(o["pnl_usd"] for o in self.orders().values()
                   if o.get("settled") and now - _ts(o["settled_ts"]) <= timedelta(hours=24))

    def loss_stopped(self, now: datetime) -> str | None:
        r = self.realized_24h(now)
        return f"daily loss stop: realized E5 P&L last 24 h {r:+.2f} <= -{self.loss_stop:g}" if r <= -self.loss_stop + 1e-9 else None

    # ------------------------------------------------------------------ selection (execution only)
    def due_picks(self, now: datetime) -> list[dict]:
        decided = {r["ticker"] for r in self.rows() if r.get("kind") == "decision"}
        out = []
        for r in _rows(self.p["picks"]):
            if r.get("kind") != "pick" or r.get("trial") != TRIAL or r["ticker"] in decided:
                continue
            if now - _ts(r["ts"]) > timedelta(hours=FRESH_H) or r["ts"] > _iso(now + timedelta(minutes=1)):
                continue
            out.append(r)
        return out

    @staticmethod
    def cutoff(contract: dict) -> datetime | None:
        ltt = contract.get("last_trade_time") if contract else None
        if not ltt:
            return None
        return datetime.fromtimestamp(int(ltt), timezone.utc) - timedelta(minutes=CUTOFF_MIN)

    def size(self, price: float, spent_today: float) -> tuple[int, float]:
        cost1 = price + FEE
        room = min(self.c.cap("FX_MAX_ORDER_USD"), TIER_B_USD, self.c.cap("FX_MAX_DAILY_USD") - spent_today)
        n = int(min(self.c.cap("FX_MAX_CONTRACTS"), math.floor(round(room / cost1, 6)))) if room > 0 else 0
        return max(n, 0), round(n * cost1, 2)

    def plan(self, pick: dict, now: datetime, spent_today: float) -> dict:
        t, side = pick["ticker"], pick["side"].lower()
        base = {"ticker": t, "code": pick.get("code"), "side": side, "pick_ts": pick["ts"],
                "event_date": pick.get("event_date")}
        c = self.c.contract(t)
        if c is None:
            return {**base, "action": "skip", "reason": "contract not listed"}
        cut = self.cutoff(c)
        if cut is None or now >= cut:
            return {**base, "action": "skip", "reason": f"past order cutoff {cut}"}
        b = self.c.book(t)
        bid, ask = b.best(side) if b else (None, None)
        base.update(bid=bid, ask=ask, cutoff=_iso(cut), conid=int(c["conid_yes"] if side == "yes" else c["conid_no"]))
        if bid is None:
            return {**base, "action": "skip", "reason": "no bid on the side"}
        price = round(float(bid), 2)
        if ask is not None and price >= ask - 1e-9:
            return {**base, "action": "skip", "reason": f"bid {price} >= ask {ask} (would cross)"}
        if price + FEE >= 1 - 1e-9:
            return {**base, "action": "skip", "reason": f"cost {price + FEE:.2f} >= 1"}
        n, cost = self.size(price, spent_today)
        if n < 1:
            return {**base, "action": "skip", "reason": f"caps leave room for 0 contracts (spent today ${spent_today:.2f})"}
        return {**base, "action": "place", "price": price, "qty": n, "cost": cost}

    # ------------------------------------------------------------------ run
    def run(self, now: datetime | None = None) -> int:
        now = now or datetime.now(timezone.utc)
        k = self.killed()
        if k:
            self.out(f"KILL file present ({k}); nothing sent")
            return 0
        if self.live:
            try:
                self.c._require_trading()
            except TradingDisabled as e:
                self.out(f"NOT TRADING: {e}")
                return 1
            self.reconcile(now)
            self.expire(now)
            self.settle(now)
        stop = self.loss_stopped(now)
        if stop:
            self.out(f"HALT: {stop}")
            return 0
        picks = self.due_picks(now)
        spent = self.c.spent_today(now)
        self.out(f"{'LIVE' if self.live else 'DRY-RUN'} {_iso(now)}: {len(picks)} due picks; spent today ${spent:.2f} "
                 f"of ${self.c.cap('FX_MAX_DAILY_USD'):g}")
        for pick in picks:
            try:
                pl = self.plan(pick, now, spent)
            except (VenueError, TradingDisabled) as e:
                pl = {"ticker": pick["ticker"], "side": pick["side"].lower(), "action": "skip", "reason": f"error {e}"}
            if pl["action"] != "place":
                self.out(f"  SKIP  {pl['ticker']:<18} {pl['side'].upper():<3} bid {pl.get('bid')} ask {pl.get('ask')}: {pl['reason']}")
                if self.live:
                    self.log({"kind": "decision", "ts": _iso(now), **pl})
                continue
            line = (f"BUY {pl['side'].upper():<3} {pl['qty']:>2} x {pl['ticker']:<18} @ {pl['price']:.2f} "
                    f"(bid {pl['bid']}, ask {pl['ask']}) ${pl['cost']:.2f} GTC post-only, cancel at {pl['cutoff']}")
            if not self.live:
                self.out("  WOULD " + line)
                spent += pl["cost"]
                continue
            if not self.place(pl, now):
                return 2
            spent = self.c.spent_today(now)
        return 0

    def place(self, pl: dict, now: datetime) -> bool:
        """Live path for one planned order. Returns False iff the run must stop (panic)."""
        bs = list(self.bet.bets().values())
        why = self.bet.check_caps(TIER, pl["cost"], bs, now)
        if why:
            self.log({"kind": "decision", "ts": _iso(now), **pl, "action": "refused", "reason": "; ".join(why)})
            self.out(f"  REFUSED {pl['ticker']}: {'; '.join(why)}")
            return True
        tag = TAG + pl["ticker"]
        res = self.c.place_limit(pl["ticker"], pl["side"], pl["price"], pl["qty"], post_only=True,
                                 expire=_ts(pl["cutoff"]), tag=tag)
        self.log({"kind": "decision", "ts": _iso(now), **pl, "action": res.status, "order_id": res.order_id,
                  "detail": (res.detail or "")[:300]})
        if res.status == "ambiguous":
            self.out(f"  AMBIGUOUS {pl['ticker']}: {res.detail}; panic")
            self.panic(f"ambiguous order POST on {pl['ticker']}")
            return False
        if res.status != "placed":
            self.out(f"  REJECTED {pl['ticker']}: {res.detail}")
            return True
        oid = res.order_id
        self.log({"kind": "order", "ts": _iso(now), "order_id": oid, "tag": tag, **{k: pl[k] for k in
                  ("ticker", "code", "side", "conid", "price", "qty", "cost", "cutoff", "event_date")}})
        self._bet({"event": "bet", "id": f"fx-{oid}", "ts": _iso(now), "venue": VENUE, "ticker": pl["ticker"],
                   "market": pl["ticker"], "side": pl["side"], "price": pl["price"], "contracts": pl["qty"],
                   "stake_usd": pl["cost"], "units": round(pl["cost"] / UNIT, 4), "tier": TIER, "strategy": STRATEGY,
                   "taker": False, "expire": pl["cutoff"], "order_id": oid, "client_order_id": tag, "status": "placed"})
        bad = self.verify(oid, pl, now)
        self.out(f"  PLACED order {oid}: BUY {pl['side'].upper()} {pl['qty']} x {pl['ticker']} @ {pl['price']:.2f} "
                 f"${pl['cost']:.2f}" + (f" -- VERIFY FAILED: {bad}" if bad else " -- verified"))
        if bad:
            self.panic(f"verify failed on {oid}: {bad}")
            return False
        return True

    # ------------------------------------------------------------------ gateway reads
    def order_status(self, oid: str) -> dict:
        """Normalised {status, side, conid, price, size, filled, avg_price} from the gateway."""
        code, out = self.c._call("GET", f"/iserver/account/order/status/{oid}")
        if code != 200 or not isinstance(out, dict):
            raise VenueError(f"order status {oid} -> HTTP {code}: {str(out)[:200]}")
        g = lambda *ks: next((out[k] for k in ks if out.get(k) not in (None, "")), None)  # noqa: E731
        num = lambda v: None if v is None else float(str(v).replace(",", ""))  # noqa: E731
        return {"status": str(g("order_status", "status") or "").lower(), "side": str(g("side") or "").upper(),
                "conid": int(g("conid") or 0), "price": num(g("limit_price", "price")),
                "size": num(g("size", "total_size", "totalSize")), "filled": num(g("cum_fill", "filledQuantity")) or 0.0,
                "avg_price": num(g("average_price", "avgPrice"))}

    def verify(self, oid: str, pl: dict, now: datetime) -> str | None:
        try:
            s = self.order_status(oid)
        except VenueError as e:
            return str(e)
        self.log({"kind": "verify", "ts": _iso(now), "order_id": oid, **s})
        bad = []
        if s["side"] not in ("BUY", "B"):
            bad.append(f"side {s['side']}")
        if s["conid"] != pl["conid"]:
            bad.append(f"conid {s['conid']} != {pl['conid']}")
        if s["price"] is None or abs(s["price"] - pl["price"]) > 1e-6:
            bad.append(f"price {s['price']} != {pl['price']}")
        if s["size"] is not None and abs(s["size"] - pl["qty"]) > 1e-6:
            bad.append(f"size {s['size']} != {pl['qty']}")
        if s["filled"] > 0:
            bad.append(f"filled {s['filled']} immediately (post-only expected to rest)")
        if s["status"] in ("cancelled", "canceled", "inactive", "rejected"):
            bad.append(f"status {s['status']}")
        return "; ".join(bad) or None

    def trades_by_tag(self) -> dict[str, dict]:
        """Executions from /iserver/account/trades summed per order_ref (= our cOID tag)."""
        try:
            rows = self.c.fills(limit=1000)
        except (VenueError, TradingDisabled):
            return {}
        out: dict[str, dict] = {}
        for r in rows:
            ref = r.get("order_ref") or ""
            if not ref.startswith(TAG):
                continue
            a = out.setdefault(ref, {"filled": 0.0, "notional": 0.0, "commission": 0.0})
            q = float(str(r.get("size", 0)).replace(",", ""))
            a["filled"] += q
            a["notional"] += q * float(r.get("price", 0))
            a["commission"] += float(r.get("commission") or 0)
        return out

    def reconcile(self, now: datetime | None = None) -> int:
        """Order state + fills from the gateway into both journals. Read-only gateway calls."""
        now = now or datetime.now(timezone.utc)
        trades = self.trades_by_tag()
        n = 0
        for oid, o in self.orders().items():
            if o["status"] in DONE or o.get("settled"):
                continue
            try:
                s = self.order_status(oid)
            except VenueError as e:
                self.out(f"  reconcile {oid}: {e}")
                continue
            t = trades.get(o["tag"], {})
            filled = max(s["filled"], t.get("filled", 0.0))
            comm = t.get("commission") or None
            if filled != o["filled"] or s["status"] != o["status"]:
                self.log({"kind": "fill" if filled != o["filled"] else "status", "ts": _iso(now), "order_id": oid,
                          "status": s["status"], "filled": filled, "avg_price": s["avg_price"], "commission": comm})
                if filled > o["filled"]:
                    self.out(f"  FILL {oid} {o['ticker']} {o['side'].upper()} {filled:g}/{o['qty']} @ {s['avg_price'] or o['price']}")
                if filled > o["qty"] + 1e-9 or (s["avg_price"] and s["avg_price"] > o["price"] + 1e-6):
                    self.panic(f"unexpected fill on {oid}: {filled:g} @ {s['avg_price']} vs order {o['qty']} @ {o['price']}")
                    return n
                if s["status"] in DONE:  # terminal: risk = what actually filled
                    self._bet({"event": "update", "id": f"fx-{oid}", "ts": _iso(now), "filled": filled,
                               "stake_usd": round(filled * (o["price"] + FEE), 2), "order_status": s["status"]})
                n += 1
        return n

    def expire(self, now: datetime) -> int:
        n = 0
        for oid, o in self.orders().items():
            if o["status"] not in DONE and not o.get("settled") and now >= _ts(o["cutoff"]):
                self.cancel(oid, o, now, why="cutoff")
                n += 1
        return n

    def cancel(self, oid: str, o: dict, now: datetime, why: str) -> bool:
        # cancelling only removes risk, so it needs a session + account but NOT --live / no-KILL (panic writes KILL)
        try:
            self.c._require_session()
            code, out = self.c._call("DELETE", f"/iserver/account/{self.c.account}/order/{oid}")
        except Exception as e:  # noqa: BLE001 -- a failed cancel must be reported, never raised past cancel-all
            self.out(f"  cancel {oid} FAILED: {e}")
            return False
        ok = code == 200 and isinstance(out, dict) and "error" not in out
        self.log({"kind": "cancel", "ts": _iso(now), "order_id": oid, "why": why, "ok": ok, "http": code,
                  "out": str(out)[:200]})
        self.out(f"  CANCEL {oid} {o['ticker']} ({why}): {'ok' if ok else 'not confirmed'}")
        return ok

    def cancel_all(self, now: datetime | None = None, why: str = "cancel-all") -> int:
        now = now or datetime.now(timezone.utc)
        ours = {oid: o for oid, o in self.orders().items() if o["status"] not in DONE and not o.get("settled")}
        try:  # also any open gateway order carrying our tag that the journal lost
            for r in self.c.open_orders():
                oid = str(r.get("orderId"))
                if str(r.get("order_ref", "")).startswith(TAG) and oid not in ours and \
                        str(r.get("status", "")).lower() not in DONE:
                    ours[oid] = {"ticker": r.get("ticker") or r.get("order_ref")[len(TAG):]}
        except (VenueError, TradingDisabled) as e:
            self.out(f"  open_orders read failed: {e}")
        n = sum(self.cancel(oid, o, now, why) for oid, o in ours.items())
        self.out(f"cancel-all: {n}/{len(ours)} cancelled")
        return n

    def panic(self, why: str) -> None:
        self.out(f"PANIC: {why} -> cancel-all + KILL")
        self.cancel_all(why=f"panic: {why}")
        self.p["kill"].parent.mkdir(parents=True, exist_ok=True)
        self.p["kill"].write_text(f"{_iso(datetime.now(timezone.utc))} forecastex_exec panic: {why}\n")
        self.log({"kind": "panic", "ts": _iso(datetime.now(timezone.utc)), "why": why})

    # ------------------------------------------------------------------ settlement
    def settle(self, now: datetime | None = None) -> int:
        now = now or datetime.now(timezone.utc)
        res = {r["ticker"]: r.get("yes_won") for r in _rows(self.p["picks"])
               if r.get("kind") == "pick" and r.get("yes_won") is not None}
        n = 0
        for oid, o in self.orders().items():
            if o.get("settled"):
                continue
            filled = float(o.get("filled") or 0)
            yw = res.get(o["ticker"])
            if filled <= 0:
                if o["status"] in DONE or yw is not None:
                    if o["status"] not in DONE:
                        continue  # still resting on a settled market: let reconcile/expire close it first
                    rec = {"result": "void-unfilled", "pnl_usd": 0.0, "filled": 0}
                else:
                    continue
            else:
                if yw is None:
                    continue
                won = (yw == 1.0) if o["side"] == "yes" else (yw == 0.0)
                comm = o.get("commission")
                fee = float(comm) if comm else FEE * filled
                pnl = round((filled if won else 0.0) - filled * o["price"] - fee, 2)
                rec = {"result": "win" if won else "loss", "pnl_usd": pnl, "filled": filled}
            self.log({"kind": "settle", "ts": _iso(now), "order_id": oid, **rec})
            self._bet({"event": "settle", "id": f"fx-{oid}", "ts": _iso(now), **rec})
            self.out(f"  SETTLE {oid} {o['ticker']} {o['side'].upper()} {rec['result']} {rec['pnl_usd']:+.2f}")
            n += 1
        return n

    # ------------------------------------------------------------------ status
    def status(self, now: datetime | None = None) -> int:
        now = now or datetime.now(timezone.utc)
        os_ = self.orders()
        k = self.killed()
        sess = self.c.session_ok()
        self.out(f"gateway session {'AUTHENTICATED' if sess else 'DOWN'}; mode {self.c.mode}; account "
                 f"{(self.c.account or 'unset')[:2]}***; KILL {k or 'none'}")
        spent = self.c.spent_today(now)
        self.out(f"caps: today ${spent:.2f} / ${self.c.cap('FX_MAX_DAILY_USD'):g} (UTC day); per order "
                 f"${min(self.c.cap('FX_MAX_ORDER_USD'), TIER_B_USD):g}, {self.c.cap('FX_MAX_CONTRACTS'):g} contracts; "
                 f"loss stop -${self.loss_stop:g}/24 h, realized 24 h {self.realized_24h(now):+.2f}"
                 + (" HALTED" if self.loss_stopped(now) else ""))
        opn = [o for o in os_.values() if o["status"] not in DONE and not o.get("settled")]
        self.out(f"orders {len(os_)}; open {len(opn)}; filled contracts {sum(float(o.get('filled') or 0) for o in os_.values()):g}")
        for oid, o in os_.items():
            self.out(f"  {oid:<12} {o['ts'][:16]} {o['ticker']:<18} {o['side'].upper():<3} {o['qty']:>2} @ {o['price']:.2f} "
                     f"filled {float(o.get('filled') or 0):g} {o['status']:<10} "
                     + (f"{o['result']} {o['pnl_usd']:+.2f}" if o.get("settled") else f"cancel at {o['cutoff'][:16]}"))
        st = [o for o in os_.values() if o.get("settled")]
        self.out(f"P&L settled {len(st)}: {sum(o['pnl_usd'] for o in st):+.2f} USD; open risk "
                 f"${sum(o['cost'] for o in opn):.2f}")
        if sess and self.c.account:
            try:
                pos = self.c.positions()
                self.out(f"gateway ForecastEx positions: {len(pos)}")
                for r in pos:
                    self.out(f"  conid {r.get('conid')} {r.get('contractDesc')} pos {r.get('position')} "
                             f"mkt {r.get('mktValue')} unrl {r.get('unrealizedPnl')}")
            except (VenueError, TradingDisabled) as e:
                self.out(f"positions read failed: {e}")
        return 0


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    cmd = args[0] if args else ""
    live = "--live" in argv
    if cmd == "run":
        return Exec(live=live).run()
    if cmd == "status":
        return Exec().status()
    if cmd == "reconcile":
        e = Exec()
        e.bet = _bet_mod()
        n = e.reconcile()
        print(f"reconciled {n} order changes")
        return 0
    if cmd == "settle":
        e = Exec()
        e.bet = _bet_mod()
        print(f"settled {e.settle()}")
        return 0
    if cmd == "cancel-all":
        Exec().cancel_all()
        return 0
    print(__doc__)
    return 1


def _bet_mod():
    import bet as _bet
    _bet.set_root(HOME)
    return _bet


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
