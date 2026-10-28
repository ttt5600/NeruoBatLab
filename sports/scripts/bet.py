"""Unified bet router (base setup, user 2026-10-27). Every bet -- API or phone -- goes through here and into one journal.

    bet.py place TICKER yes|no (--units U | --contracts N) [--price P] [--tier A|B|C] [--strategy NAME]
                [--taker] [--expire ISO] [--dry]
    bet.py log-external --venue robinhood|kalshi-app|unknown --market "POR ML" --side yes|no --price 0.55
                (--units U | --contracts N) --tier C --strategy judgment [--result win|loss|void]
    bet.py settle [--id BET_ID --result win|loss|void]     # API bets: from order fills + market result
    bet.py pnl | orders | cancel ORDER_ID|--all | shards
    bet.py allocate 0=70 3=30 [--confirm]                  # target_balance_allocation; DRY-RUN unless --confirm

Sizing (notes: STATE.md base setup): BANKROLL $2,000, 1 unit = $50. Units are DOLLARS AT RISK (contracts x price + fee).
  A proven (sealed pass)     : <= 1u per night (ET date) total, incl. kalshi_exec.py live spend that night
  B paper lead w/ fwd evidence: <= 0.5u per bet
  C judgment                 : <= 0.25u per bet; monthly C loss cap 2u (realized this ET month - open C risk - new)
  Loss stops on manual tiers B+C (worst case: realized P&L in window - all open B+C risk - this bet):
  daily (ET day) -3u, weekly (trailing 7 ET days) -6u. KILL file (kalshi_exec) stops everything.
Shards: Kalshi keeps cash per exchange shard (0 default, 1 combos, 2 crypto/commodities, 3 tennis/baseball/basketball);
an order only sees cash on its market's shard. Shard = market.exchange_index (authoritative), else SHARD_TABLE.
Default order: post-only maker, GTC, expiring at game START (ESPN scoreboard; non-sports: market close).
Journal: <deploy>/data/bets.jsonl (append-only events). Legacy manual_bets.jsonl rows are read as tier L (pre-router
$25 bonus; excluded from caps). Uses kalshi_exec's signed client + .env; never prints the key.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kalshi_exec as K  # noqa: E402

ET = ZoneInfo("America/New_York")
BANKROLL = 2000.0
UNIT = 50.0
TIER_BET_CAP_U = {"B": 0.5, "C": 0.25}
TIER_A_NIGHT_U = 1.0
C_MONTH_LOSS_U = 2.0
DAILY_STOP_U = 3.0
WEEKLY_STOP_U = 6.0
MANUAL_TIERS = ("B", "C")
SHARD_NAMES = {0: "default", 1: "combos", 2: "crypto/commodities", 3: "tennis/baseball/basketball"}
# fallback only -- market.exchange_index wins whenever the API returns it
SHARD_TABLE = [(r"^KX(NBA|WNBA|NCAAMB|NCAAWB|NCAAB|MLB|ATP|WTA|EUROLEAGUE|BBL)", 3), (r"^KX(BTC|ETH|SOL|XRP|DOGE|GOLD|SILVER|WTI|OIL|NATGAS|COPPER)", 2),
               (r"^KXMVE", 1)]
ESPN_LEAGUE = {"KXNBA": "basketball/nba", "KXWNBA": "basketball/wnba", "KXNFL": "football/nfl",
               "KXNCAAF": "football/college-football", "KXNHL": "hockey/nhl", "KXMLB": "baseball/mlb",
               "KXNCAAMB": "basketball/mens-college-basketball"}
SPORTS_PREFIX = r"^KX(NBA|WNBA|NFL|NCAA|NHL|MLB|ATP|WTA|MLS|EPL|UFC|PGA|F1|NASCAR)"

DEPLOY = K.DEPLOY
ROOT = Path(os.environ.get("BET_HOME") or DEPLOY)
JOURNAL = ROOT / "data" / "bets.jsonl"
LEGACY = ROOT / "data" / "manual_bets.jsonl"


def set_root(root: Path) -> None:
    global ROOT, JOURNAL, LEGACY
    ROOT = Path(root)
    JOURNAL, LEGACY = ROOT / "data" / "bets.jsonl", ROOT / "data" / "manual_bets.jsonl"
    K.set_root(ROOT, deploy=ROOT)


set_root(ROOT)


# --------------------------------------------------------------------------------------------- journal
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def append(rec: dict) -> dict:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def bets() -> dict[str, dict]:
    """bet id -> folded state (bet row + updates + settle). Legacy manual_bets rows come in as tier L."""
    out: dict[str, dict] = {}
    for r in _read(LEGACY):
        if r.get("status") != "placed":
            continue
        bid = "L-" + r["client_order_id"][-8:]
        out[bid] = {"id": bid, "ts": r["ts"], "venue": "kalshi", "ticker": r["ticker"], "market": r["ticker"],
                    "side": r["side"], "price": r["price"], "contracts": r["count"], "stake_usd": r["max_cost"],
                    "units": round(r["max_cost"] / UNIT, 4), "tier": "L", "strategy": "legacy-manual",
                    "order_id": r.get("order_id"), "status": "placed", "taker": r.get("taker", False)}
    for r in _read(JOURNAL):
        ev, bid = r.get("event"), r.get("id")
        if ev == "bet":
            out[bid] = dict(r)
        elif bid in out and ev == "update":
            out[bid].update({k: v for k, v in r.items() if k not in ("event", "ts", "id")})
        elif bid in out and ev == "settle":
            out[bid].update(settled=True, result=r["result"], pnl_usd=r["pnl_usd"], settled_ts=r["ts"],
                            filled=r.get("filled", out[bid].get("filled")))
    return out


def live(b: dict) -> bool:
    """Counts toward exposure: placed on the exchange or logged external; not refused/rejected/cancelled-unfilled."""
    return b.get("status") in ("placed", "external")


def risk(b: dict) -> float:
    """Open dollars at risk on an unsettled bet."""
    return 0.0 if b.get("settled") or not live(b) else float(b.get("stake_usd") or 0)


def et_day(t: datetime) -> str:
    return t.astimezone(ET).date().isoformat()


# --------------------------------------------------------------------------------------------- risk rules
def realized(bs, tiers, since: datetime) -> float:
    return sum(b["pnl_usd"] for b in bs if b.get("settled") and b.get("tier") in tiers and _ts(b["settled_ts"]) >= since)


def exec_night_spend(now: datetime) -> float:
    """kalshi_exec.py live spend on this ET night (tier A runs there too)."""
    st = K.fold(K.journal(), "live")
    return sum(K.committed(o) for o in st.values() if et_day(_ts(o["ts"])) == et_day(now))


def check_caps(tier: str, cost: float, bs: list[dict], now: datetime, exec_spend: float = 0.0) -> list[str]:
    """Reasons to refuse (empty = OK). cost = max dollars at risk on the new bet."""
    why = []
    u = cost / UNIT
    if tier in TIER_BET_CAP_U and u > TIER_BET_CAP_U[tier] + 1e-9:
        why.append(f"tier {tier} per-bet cap {TIER_BET_CAP_U[tier]}u (${TIER_BET_CAP_U[tier] * UNIT:.2f}); this bet "
                   f"{u:.3f}u (${cost:.2f})")
    if tier == "A":
        night = sum(float(b.get("stake_usd") or 0) for b in bs if b.get("tier") == "A" and live(b)
                    and et_day(_ts(b["ts"])) == et_day(now)) + exec_spend
        if night + cost > TIER_A_NIGHT_U * UNIT + 1e-9:
            why.append(f"tier A night cap {TIER_A_NIGHT_U}u: ${night:.2f} already (incl. kalshi_exec ${exec_spend:.2f}) "
                       f"+ ${cost:.2f} > ${TIER_A_NIGHT_U * UNIT:.2f}")
    if tier in MANUAL_TIERS:
        start_day = datetime.combine(now.astimezone(ET).date(), datetime.min.time(), ET)
        open_risk = sum(risk(b) for b in bs if b.get("tier") in MANUAL_TIERS)
        for name, since, stop in (("daily", start_day, DAILY_STOP_U), ("weekly", start_day - timedelta(days=6), WEEKLY_STOP_U)):
            worst = realized(bs, MANUAL_TIERS, since) - open_risk - cost
            if worst < -stop * UNIT - 1e-9:
                why.append(f"{name} loss stop -{stop}u: realized {realized(bs, MANUAL_TIERS, since) / UNIT:+.2f}u, "
                           f"open B+C risk {open_risk / UNIT:.2f}u, this bet {u:.2f}u -> worst {worst / UNIT:+.2f}u")
    if tier == "C":
        month = datetime(now.astimezone(ET).year, now.astimezone(ET).month, 1, tzinfo=ET)
        open_c = sum(risk(b) for b in bs if b.get("tier") == "C")
        worst = realized(bs, ("C",), month) - open_c - cost
        if worst < -C_MONTH_LOSS_U * UNIT - 1e-9:
            why.append(f"monthly tier-C loss cap -{C_MONTH_LOSS_U}u: realized {realized(bs, ('C',), month) / UNIT:+.2f}u, "
                       f"open C {open_c / UNIT:.2f}u, this bet {u:.2f}u -> worst {worst / UNIT:+.2f}u")
    return why


def fee_unit(price: float, taker: bool, fee_type: str | None) -> float:
    return K.FEE * price * (1 - price) if taker else K.maker_fee_unit(price, fee_type)


def size(units: float | None, contracts: int | None, price: float, taker: bool, fee_type: str | None) -> tuple[int, float]:
    """(contracts, max cost incl. fee rounded up to the cent). Units -> the most contracts whose cost fits units x $50."""
    fu = fee_unit(price, taker, fee_type)
    if contracts is None:
        contracts = int(math.floor(round(units * UNIT / (price + fu), 6)))
    fee = math.ceil(round(contracts * fu * 100, 6)) / 100 if contracts > 0 and fu > 0 else 0.0
    return contracts, round(contracts * price + fee, 2)


def shard_of(market: dict | None, ticker: str) -> tuple[int, str]:
    if market and market.get("exchange_index") is not None:
        return int(market["exchange_index"]), "api"
    for pat, s in SHARD_TABLE:
        if re.match(pat, ticker):
            return s, "table"
    return 0, "table"


def shard_cash(bal: dict) -> dict[int, float]:
    """GET /portfolio/balance -> {exchange_index: dollars}. balance_breakdown[].balance is a dollar string."""
    return {int(x["exchange_index"]): float(x["balance"]) for x in bal.get("balance_breakdown") or []}


# --------------------------------------------------------------------------------------------- game start
def parse_event(ticker: str) -> tuple[str, str, str] | None:
    """KXNBAGAME-26OCT27PORLAL-POR -> ('KXNBAGAME', '2026-10-27', 'PORLAL'); MLB tickers carry HHMM after the date."""
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    m = re.match(r"^(\d{2})([A-Z]{3})(\d{2})(\d{4})?([A-Z]+)$", parts[1])
    if not m:
        return None
    try:
        d = datetime.strptime(f"20{m[1]}{m[2]}{m[3]}", "%Y%b%d").date()
    except ValueError:
        return None
    return parts[0], d.isoformat(), m[5]


def espn_start(ticker: str, get=None) -> datetime | None:
    pe = parse_event(ticker)
    if not pe:
        return None
    series, day, teams = pe
    league = next((v for k, v in ESPN_LEAGUE.items() if series.startswith(k)), None)
    if not league:
        return None
    if get is None:
        import requests
        get = lambda url: requests.get(url, timeout=10).json()  # noqa: E731
    base = f"https://site.api.espn.com/apis/site/v2/sports/{league}/scoreboard"
    evs = []
    for url in (f"{base}?dates={day.replace('-', '')}", base):  # dated call can come back empty on game day
        try:
            evs = get(url).get("events", [])
        except Exception:
            evs = []
        if evs:
            break
    alias = {"GS": ["GSW"], "NY": ["NYK"], "SA": ["SAS"], "NO": ["NOP"], "UTAH": ["UTA"], "WSH": ["WAS"]}
    for e in evs:
        cs = e["competitions"][0]["competitors"]
        side = {c["homeAway"]: c["team"]["abbreviation"].upper() for c in cs}
        codes = lambda x: [x] + alias.get(x, [])  # noqa: E731
        if any(a + h == teams for a in codes(side.get("away", "")) for h in codes(side.get("home", ""))) \
                and et_day(_ts(e["date"])) == day:
            return _ts(e["date"])
    return None


def default_expire(ticker: str, market: dict, now: datetime, get=None) -> tuple[datetime | None, str]:
    st = espn_start(ticker, get)
    if st:
        return st, "espn start"
    if re.match(SPORTS_PREFIX, ticker) or market.get("exchange_index") == 3:
        return None, "sports market but no ESPN start found -- pass --expire ISO (game start)"
    return _ts(market["close_time"]), "market close"


# --------------------------------------------------------------------------------------------- client
def client():
    e = K.env()
    cfg = K.config(e)
    return K.Kalshi(e, cfg["env"])


def best(market: dict, side: str) -> tuple[float | None, float | None]:
    f = lambda k: float(market[k]) if market.get(k) not in (None, "") else None  # noqa: E731
    return f(f"{side}_bid_dollars"), f(f"{side}_ask_dollars")


# --------------------------------------------------------------------------------------------- commands
def cmd_place(a, c=None, now: datetime | None = None, get=None) -> int:
    now = now or _now()
    t, side, tier = a.ticker.upper(), a.side, a.tier
    def refuse(msg: str) -> int:
        print("REFUSED:", msg)
        if a.dry:
            return 1
        append({"event": "bet", "id": "R-" + uuid.uuid4().hex[:8], "ts": _iso(now), "venue": "kalshi", "ticker": t,
                "market": t, "side": side, "tier": tier, "strategy": a.strategy, "status": "refused", "reason": msg})
        return 1
    if (a.units is None) == (a.contracts is None):
        print("give exactly one of --units / --contracts")
        return 2
    k = K.killed()
    if k:
        return refuse(f"KILL file present ({k})")
    c = c or client()
    m = c.market(t)
    if not m or m.get("status") not in ("active", "open"):
        return refuse(f"market not active: {m and m.get('status')}")
    bid, ask = best(m, side)
    price = a.price if a.price is not None else (ask if a.taker else bid)
    if not price or not 0 < price < 1:
        return refuse(f"no usable {'ask' if a.taker else 'bid'} for {side}; pass --price")
    if not a.taker and ask is not None and price >= ask - 1e-9:
        return refuse(f"maker price {price} would cross the {side} ask {ask} (post-only); lower it or use --taker")
    fee_type = c.series_fee_type(t.split("-")[0])
    n, cost = size(a.units, a.contracts, price, a.taker, fee_type)
    if n < 1:
        return refuse(f"{a.units}u buys 0 contracts at {price}")
    bs = list(bets().values())
    why = check_caps(tier, cost, bs, now, exec_night_spend(now) if tier == "A" else 0.0)
    if why:
        return refuse("; ".join(why))
    shard, src = shard_of(m, t)
    code, bal = c.request("GET", "/portfolio/balance")
    cash = shard_cash(bal if code == 200 else {})
    if cash.get(shard, 0.0) + 1e-9 < cost:
        return refuse(f"shard {shard} ({SHARD_NAMES.get(shard, '?')}, from {src}) has ${cash.get(shard, 0.0):.2f} < "
                      f"${cost:.2f} needed. Fund it: `bet.py allocate ... --confirm` (auto-rebalance) or a shard transfer")
    exp = None
    if not a.taker:
        if a.expire:
            exp, why_exp = _ts(a.expire), "--expire"
        else:
            exp, why_exp = default_expire(t, m, now, get)
        if exp is None:
            return refuse(why_exp)
        if exp <= now + timedelta(minutes=1):
            return refuse(f"expiry {exp.isoformat()} ({why_exp}) is not in the future -- game started?")
    yes_price = price if side == "yes" else round(1 - price, 2)
    bid_id = uuid.uuid4().hex[:10]
    coid = f"bet-{bid_id}"
    body = {"ticker": t, "side": "bid" if side == "yes" else "ask", "count": str(n), "price": f"{yes_price:.2f}",
            "self_trade_prevention_type": "taker_at_cross", "client_order_id": coid}
    if a.taker:
        body["time_in_force"] = "immediate_or_cancel"
    else:
        body.update(time_in_force="good_till_canceled", post_only=True, expiration_time=int(exp.timestamp()))
    rec = {"event": "bet", "id": bid_id, "ts": _iso(now), "venue": "kalshi", "ticker": t, "market": m.get("title") or t,
           "side": side, "price": price, "contracts": n, "stake_usd": cost, "units": round(cost / UNIT, 4), "tier": tier,
           "strategy": a.strategy, "taker": a.taker, "shard": shard, "expire": exp and _iso(exp),
           "client_order_id": coid}
    if a.dry:
        print(f"DRY: would send {side.upper()} {n} @ {price} {t} | ${cost:.2f} = {cost / UNIT:.2f}u tier {tier} | "
              f"shard {shard} | {'IOC taker' if a.taker else 'post-only, expires ' + _iso(exp)}")
        return 0
    code, out = c.create_order(body)
    o = (out or {}).get("order", out) or {}
    rec.update(http=code, order_id=o.get("order_id"), status="placed" if code in (200, 201) else "rejected",
               response=None if code in (200, 201) else str(out)[:300])
    append(rec)
    print(f"HTTP {code} {rec['status']} bet {bid_id} order {rec['order_id']} | {side.upper()} {n} @ {price} {t} | "
          f"${cost:.2f} ({cost / UNIT:.2f}u) tier {tier} {a.strategy} | shard {shard}"
          + ("" if a.taker else f" | expires {_iso(exp)}"))
    return 0 if rec["status"] == "placed" else 1


def ext_pnl(b: dict, result: str) -> float:
    stake = float(b["stake_usd"])
    if result == "win":
        n = b.get("contracts") or stake / b["price"]
        return round(n * 1.0 - stake, 2)
    return -stake if result == "loss" else 0.0


def cmd_log_external(a, now: datetime | None = None) -> int:
    now = now or _now()
    if (a.units is None) == (a.contracts is None):
        print("give exactly one of --units / --contracts")
        return 2
    stake = round(a.units * UNIT if a.units is not None else a.contracts * a.price, 2)
    bs = list(bets().values())
    warn = check_caps(a.tier, stake, bs, now)
    rec = append({"event": "bet", "id": uuid.uuid4().hex[:10], "ts": _iso(now), "venue": a.venue, "ticker": None,
                  "market": a.market, "side": a.side, "price": a.price, "contracts": a.contracts, "stake_usd": stake,
                  "units": round(stake / UNIT, 4), "tier": a.tier, "strategy": a.strategy, "status": "external",
                  "cap_breach": warn or None})
    print(f"logged external bet {rec['id']}: {a.venue} | {a.market} {a.side.upper()} @ {a.price} | ${stake:.2f} "
          f"({stake / UNIT:.2f}u) tier {a.tier} {a.strategy}")
    for w in warn:
        print("  CAP BREACH (logged anyway -- already placed):", w)
    if a.result:
        _settle_one(rec["id"], a.result, now)
    return 0


def _settle_one(bid: str, result: str, now: datetime) -> int:
    b = bets().get(bid)
    if not b:
        print("no such bet", bid)
        return 1
    if b.get("settled"):
        print(f"bet {bid} already settled {b['result']} {b['pnl_usd']:+.2f}")
        return 1
    pnl = ext_pnl(b, result)
    append({"event": "settle", "id": bid, "ts": _iso(now), "result": result, "pnl_usd": pnl})
    print(f"settled {bid} {b.get('market')} {result} {pnl:+.2f} ({pnl / UNIT:+.2f}u)")
    return 0


def cmd_settle(a, c=None, now: datetime | None = None) -> int:
    now = now or _now()
    if a.id:
        return _settle_one(a.id, a.result, now)
    c = c or client()
    n = 0
    for b in bets().values():
        if b.get("settled") or b.get("venue") != "kalshi" or b.get("status") != "placed" or not b.get("order_id"):
            continue
        code, out = c.request("GET", f"/portfolio/orders/{b['order_id']}")
        o = (out or {}).get("order") or {}
        if code != 200 or not o:
            continue
        filled = float(o.get("fill_count_fp") or o.get("fill_count") or 0)
        m = c.market(b["ticker"]) or {}
        res = m.get("result")
        if filled <= 0:
            if o.get("status") in ("canceled", "cancelled", "expired") or res in ("yes", "no"):
                append({"event": "settle", "id": b["id"], "ts": _iso(now), "result": "void-unfilled", "pnl_usd": 0.0,
                        "filled": 0})
                n += 1
            continue
        if res not in ("yes", "no"):
            continue
        # cost = fills x OUR limit price (exact for a post-only maker fill; an upper bound for a taker IOC that
        # filled better) + the fees Kalshi reports on the order
        cost = filled * b["price"] + sum(float(o.get(k) or 0) for k in ("taker_fees_dollars", "maker_fees_dollars"))
        pnl = round((filled if res == b["side"] else 0.0) - cost, 2)
        append({"event": "settle", "id": b["id"], "ts": _iso(now), "result": "win" if res == b["side"] else "loss",
                "pnl_usd": pnl, "filled": filled})
        print(f"settled {b['id']} {b['ticker']} {res} filled {filled:g} pnl {pnl:+.2f}")
        n += 1
    open_ext = [b for b in bets().values() if b.get("status") == "external" and not b.get("settled")]
    print(f"{n} settled from the API; {len(open_ext)} external bets still open "
          f"(settle with: bet.py settle --id ID --result win|loss|void)")
    for b in open_ext:
        print(f"  {b['id']}  {b['ts'][:10]}  {b['venue']}  {b['market']}  {b['side']} @ {b['price']}  {b['units']}u")
    return 0


def summarize(bs: list[dict], now: datetime) -> list[str]:
    out = []
    for key in ("tier", "strategy", "venue"):
        g = defaultdict(lambda: [0, 0, 0, 0.0, 0.0, 0.0])  # n, wins, losses, staked, pnl, open risk
        for b in bs:
            if not live(b):
                continue
            r = g[b.get(key) or "?"]
            r[0] += 1
            r[3] += float(b.get("stake_usd") or 0)
            if b.get("settled"):
                r[1] += b["result"] == "win"
                r[2] += b["result"] == "loss"
                r[4] += b["pnl_usd"]
            else:
                r[5] += risk(b)
        out.append(f"{'by ' + key + ':':33s} {'bets':>4} {'W-L':>6} {'staked':>9} {'P&L $':>8} {'P&L u':>7} {'open $':>8}")
        for k, r in sorted(g.items()):
            out.append(f"         {k:24.24s} {r[0]:>4} {f'{r[1]}-{r[2]}':>6} {r[3]:>9.2f} {r[4]:>+8.2f} "
                       f"{r[4] / UNIT:>+7.2f} {r[5]:>8.2f}")
    start_day = datetime.combine(now.astimezone(ET).date(), datetime.min.time(), ET)
    month = datetime(now.astimezone(ET).year, now.astimezone(ET).month, 1, tzinfo=ET)
    open_bc = sum(risk(b) for b in bs if b.get("tier") in MANUAL_TIERS)
    open_c = sum(risk(b) for b in bs if b.get("tier") == "C")
    a_night = sum(float(b.get("stake_usd") or 0) for b in bs if b.get("tier") == "A" and live(b)
                  and et_day(_ts(b["ts"])) == et_day(now))
    out.append(f"caps (1u = ${UNIT:.0f}, bankroll ${BANKROLL:,.0f}):")
    out.append(f"  daily  B+C realized {realized(bs, MANUAL_TIERS, start_day) / UNIT:+.2f}u, open {open_bc / UNIT:.2f}u"
               f"  vs stop -{DAILY_STOP_U}u")
    out.append(f"  weekly B+C realized {realized(bs, MANUAL_TIERS, start_day - timedelta(days=6)) / UNIT:+.2f}u, "
               f"open {open_bc / UNIT:.2f}u  vs stop -{WEEKLY_STOP_U}u")
    out.append(f"  month  C realized {realized(bs, ('C',), month) / UNIT:+.2f}u, open {open_c / UNIT:.2f}u  vs cap "
               f"-{C_MONTH_LOSS_U}u")
    out.append(f"  tonight tier A via bet.py ${a_night:.2f} (+ kalshi_exec spend) vs {TIER_A_NIGHT_U}u; "
               f"per-bet B {TIER_BET_CAP_U['B']}u, C {TIER_BET_CAP_U['C']}u")
    breaches = [b for b in bs if b.get("cap_breach")]
    for b in breaches:
        out.append(f"  ! {b['id']} {b['market']} logged over cap: {b['cap_breach'][0]}")
    return out


def cmd_pnl(now: datetime | None = None) -> int:
    print("\n".join(summarize(list(bets().values()), now or _now())))
    return 0


def cmd_orders(c=None) -> int:
    c = c or client()
    bs = bets()
    by_order = {b.get("order_id"): b for b in bs.values() if b.get("order_id")}
    for o in c.orders(status="resting"):
        b = by_order.get(o.get("order_id"))
        if b:
            print(f"RESTING {b['id']} tier {b['tier']} {o.get('ticker')} {b['side']} @ {b['price']} filled "
                  f"{o.get('fill_count_fp') or o.get('fill_count')} remaining {o.get('remaining_count_fp') or o.get('remaining_count')}"
                  f" exp {b.get('expire')} order {o.get('order_id')}")
    tickers = {b.get("ticker") for b in bs.values() if b.get("ticker")}
    for p in c.positions():
        if p.get("ticker") in tickers:
            print("POSITION", p.get("ticker"), p.get("position_fp") or p.get("position"), "exposure",
                  p.get("market_exposure_dollars"))
    for b in bs.values():
        if live(b) and not b.get("settled") and b.get("venue") != "kalshi":
            print(f"EXTERNAL {b['id']} {b['venue']} {b['market']} {b['side']} @ {b['price']} {b['units']}u tier {b['tier']}")
    return 0


def cmd_cancel(target: str, c=None, now: datetime | None = None) -> int:
    c = c or client()
    bs = [b for b in bets().values() if b.get("order_id") and b.get("status") == "placed" and not b.get("settled")]
    pick = bs if target == "--all" else [b for b in bs if target in (b["order_id"], b["id"])]
    if not pick:
        print("no matching open bet/order")
        return 1
    for b in pick:
        code, out = c.cancel(b["order_id"], b["ticker"])
        print(f"cancel {b['id']} {b['ticker']} -> HTTP {code}")
        if code == 200 and b["id"][:2] != "L-":
            append({"event": "update", "id": b["id"], "ts": _iso(now or _now()), "cancel_requested": True})
    return 0


def cmd_shards(c=None) -> int:
    c = c or client()
    code, bal = c.request("GET", "/portfolio/balance")
    if code != 200:
        print("balance HTTP", code)
        return 1
    cash = shard_cash(bal)
    for s in sorted(cash):
        print(f"shard {s} {SHARD_NAMES.get(s, '?'):28s} ${cash[s]:>9.2f}")
    print(f"total cash ${sum(cash.values()):.2f}")
    code, alloc = c.request("GET", "/portfolio/target_balance_allocation")
    if code == 200:
        al = alloc.get("allocations") or []
        print("auto-rebalance target:", ", ".join(f"{x['exchange_index']}={x['percent']}%" for x in al) if al else "off")
    return 0


def parse_alloc(specs: list[str]) -> list[dict]:
    al = []
    for s in specs:
        k, v = s.split("=")
        al.append({"exchange_index": int(k), "percent": int(v)})
    if sum(x["percent"] for x in al) != 100 or len({x["exchange_index"] for x in al}) != len(al):
        raise ValueError("percents must sum to 100 with distinct shard indexes")
    return al


def cmd_allocate(specs: list[str], confirm: bool, c=None) -> int:
    try:
        body = {"allocations": parse_alloc(specs)} if specs != ["off"] else {"allocations": []}
    except ValueError as ex:
        print("bad allocation:", ex)
        return 2
    c = c or client()
    code, cur = c.request("GET", "/portfolio/target_balance_allocation")
    print("current:", cur if code == 200 else f"HTTP {code}")
    print("POST /portfolio/target_balance_allocation", json.dumps(body))
    if not confirm:
        print("DRY-RUN (nothing sent). Add --confirm to apply.")
        return 0
    code, out = c.request("POST", "/portfolio/target_balance_allocation", body=body)
    print("HTTP", code, out or "")
    return 0 if code == 200 else 1


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    pl = sp.add_parser("place")
    pl.add_argument("ticker")
    pl.add_argument("side", choices=["yes", "no"])
    pl.add_argument("--units", type=float)
    pl.add_argument("--contracts", type=int)
    pl.add_argument("--price", type=float)
    pl.add_argument("--tier", choices=["A", "B", "C"], default="C")
    pl.add_argument("--strategy", default="judgment")
    pl.add_argument("--taker", action="store_true")
    pl.add_argument("--expire")
    pl.add_argument("--dry", action="store_true", help="size + check everything, send nothing")
    ex = sp.add_parser("log-external")
    ex.add_argument("--venue", required=True)
    ex.add_argument("--market", required=True)
    ex.add_argument("--side", required=True)
    ex.add_argument("--price", type=float, required=True)
    ex.add_argument("--units", type=float)
    ex.add_argument("--contracts", type=int)
    ex.add_argument("--tier", choices=["A", "B", "C"], default="C")
    ex.add_argument("--strategy", default="judgment")
    ex.add_argument("--result", choices=["win", "loss", "void"])
    st = sp.add_parser("settle")
    st.add_argument("--id")
    st.add_argument("--result", choices=["win", "loss", "void"])
    sp.add_parser("pnl")
    sp.add_parser("orders")
    cn = sp.add_parser("cancel")
    cn.add_argument("target", help="order id, bet id, or --all")
    sp.add_parser("shards")
    al = sp.add_parser("allocate")
    al.add_argument("specs", nargs="+", help="e.g. 0=70 3=30 (or 'off')")
    al.add_argument("--confirm", action="store_true")
    if argv[:1] == ["cancel"] and argv[1:] == ["--all"]:
        return cmd_cancel("--all")
    a = p.parse_args(argv)
    if a.cmd == "place":
        return cmd_place(a)
    if a.cmd == "log-external":
        return cmd_log_external(a)
    if a.cmd == "settle":
        if bool(a.id) != bool(a.result):
            print("--id and --result go together")
            return 2
        return cmd_settle(a)
    if a.cmd == "pnl":
        return cmd_pnl()
    if a.cmd == "orders":
        return cmd_orders()
    if a.cmd == "cancel":
        return cmd_cancel(a.target)
    if a.cmd == "shards":
        return cmd_shards()
    if a.cmd == "allocate":
        return cmd_allocate(a.specs, a.confirm)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
