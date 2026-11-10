"""Post-trade review: READ-ONLY audit of the last N days of live trading (learning_loop, 2026-11-10).

    python scripts/post_trade_review.py [--days 14] [--offline] [--no-ledger]

Exchange truth (Kalshi GETs only: /portfolio/fills, /orders, /positions, /settlements) is compared with the bots'
journals in ~/.claude/quant-kalshi/data (kalshi_exec.jsonl = weather E4, lead_exec.jsonl = lead + in-game NHL/NFLF,
bets.jsonl, forecastex_exec.jsonl + forecastex_orders.jsonl, kalshi_paper.jsonl = E4's paper twin).
Nothing here POSTs, cancels, or writes to the deployed data dir.

Checks (each one is a lesson in sports/notes/lessons.jsonl):
  C1 sell_past_held / opposite_side  replay every exchange fill per ticker (book_side bid = +YES, ask = -YES; NOT
                                     action/side, a weather maker NO is action=sell side=yes): an exit that moves the
                                     position through zero, or any position on the side opposite the journal's intent.
  C2 multi_cid                       > 1 client_order_id per intended action (lead_exec key entry / exit_coid), and
                                     exchange bursts (>= 3 distinct cids on one ticker+book side within 10 min).
  C3 price_band                      fill price on OUR side vs the journal limit (traps.classify_fill: only 'worse' is
                                     an anomaly; price improvement is counted, never flagged -- the 11-08 false
                                     LEAD_KILL); exchange fills > journal intended count; orders unknown to any journal.
  C4 pnl_divergence                  journal settle P&L vs exchange settlements per ticker; E4 live vs paper twin.
  C5 no_terminal                     journal orders still submitting/unknown after 1 h, resting past expiry; exchange
                                     resting orders past expiry; ForecastEx orders with no terminal row after cutoff.
  C6 position_reconcile              exchange open positions vs journal-attributed tickers (phantom / orphan risk).
Output: sports/notes/post_trade_review_<UTC date>.md; raw exchange snapshot in sports/results/post_trade_review/<date>/.
New lesson TYPES (a check that fires and has no ledger row yet) are appended to sports/notes/lessons.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPORTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.path.expanduser("~/agent-harness/lib"))
import kalshi_exec as kx  # noqa: E402
from traps import classify_fill  # noqa: E402

DATA = kx.DEPLOY / "data"
LEDGER = SPORTS / "notes" / "lessons.jsonl"
TERMINAL_EX = ("executed", "canceled")
JOURNAL_PENDING = ("submitting", "unknown")
BURST_WINDOW_S, BURST_CIDS = 600, 3


# ------------------------------------------------------------------------------------------------ helpers
def ts(x) -> datetime | None:
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)):
        return datetime.fromtimestamp(float(x), timezone.utc)
    try:
        d = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def jl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def f(x, d=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


# ------------------------------------------------------------------------------------------------ inputs
def fetch_exchange(since: datetime, snap_dir: Path | None) -> dict:
    """Read-only Kalshi GETs. Fills/orders from since-30d so window-start positions can be reconstructed."""
    kx.set_root(kx.DEPLOY)
    e = kx.env()
    c = kx.Kalshi(e, kx.config(e)["env"])
    lo = int((since - timedelta(days=30)).timestamp())
    ex = {"fills": c._pages("/portfolio/fills", "fills", {"min_ts": lo}),
          "orders": c._pages("/portfolio/orders", "orders", {"min_ts": lo}),
          "positions": c.positions(),
          "settlements": c._pages("/portfolio/settlements", "settlements", {"min_ts": int(since.timestamp())})}
    if snap_dir:
        snap_dir.mkdir(parents=True, exist_ok=True)
        for k, v in ex.items():
            (snap_dir / f"{k}.json").write_text(json.dumps(v))
    return ex


def journal_intents() -> tuple[dict, dict, list]:
    """cid -> intent {strategy, ticker, role, side, limit (own-side price), count, ts}; ticker -> {intended sides}."""
    cid, tick = {}, defaultdict(set)
    for r in jl(DATA / "kalshi_exec.jsonl"):                        # weather E4: always NO
        if r.get("mode") == "live" and r.get("event") == "order" and r.get("client_order_id"):
            lim = 1 - f(r["limit_yes_price"]) if r.get("limit_yes_price") else r.get("maker_q")
            cid[r["client_order_id"]] = dict(strategy="E4_weather", ticker=r["ticker"], role="entry", side="no",
                                             limit=lim, count=f(r.get("count")), ts=r["ts"])
        if r.get("mode") == "live" and r.get("ticker") and r.get("event") == "order" and f(r.get("count")) > 0:
            tick[r["ticker"]].add("no")
    lead_keys = {}
    for r in jl(DATA / "lead_exec.jsonl"):
        if r.get("mode") != "live":
            continue
        if r.get("event") == "order":
            lead_keys[r.get("key") or r["ticker"]] = r
            if f(r.get("count")) > 0 or r.get("client_order_id"):
                tick[r["ticker"]].add(r.get("side"))
            if r.get("client_order_id"):
                cid[r["client_order_id"]] = dict(strategy=r.get("strategy"), ticker=r["ticker"], role="entry",
                                                 side=r.get("side"), limit=r.get("limit_price", r.get("maker_q")),
                                                 count=f(r.get("count")), ts=r["ts"], key=r.get("key"))
        elif r.get("event") == "update" and r.get("exit_coid"):
            o = lead_keys.get(r.get("key"), {})
            cid.setdefault(r["exit_coid"], dict(strategy=o.get("strategy"), ticker=r["ticker"], role="exit",
                                                side=o.get("side"), limit=r.get("exit_limit"), count=None,
                                                ts=r["ts"], key=r.get("key")))
    for r in jl(DATA / "bets.jsonl"):                               # router (bet.py) -- Kalshi + ForecastEx
        if r.get("event") == "bet" and r.get("venue") == "kalshi" and r.get("client_order_id"):
            cid[r["client_order_id"]] = dict(strategy=r.get("strategy"), ticker=r["ticker"], role="entry",
                                             side=r.get("side"), limit=r.get("price"), count=f(r.get("contracts")),
                                             ts=r["ts"])
            tick[r["ticker"]].add(r.get("side"))
    needle = jl(DATA / "nyt_needle_house_log.jsonl")
    for r in needle:
        if r.get("action") == "live" and r.get("side"):
            tick[r["ticker"]].add(str(r["side"]).lower())
    return cid, tick, needle


def classify_cid(c: str | None, intents: dict) -> dict:
    if c in intents:
        return intents[c]
    if c and c.startswith("needle-exit-"):
        return dict(strategy="needle", role="exit")
    if c and c.startswith("needle"):
        return dict(strategy="needle", role="entry")
    return {}


# ------------------------------------------------------------------------------------------------ checks
def c1_replay(ex: dict, intents: dict, tick_side: dict, since: datetime) -> tuple[list, list]:
    by_oid = {o["order_id"]: o for o in ex["orders"]}
    flags, info = [], []
    per = defaultdict(list)
    for x in ex["fills"]:
        per[x["ticker"]].append(x)
    cur = {p["ticker"]: f(p.get("position_fp")) for p in ex["positions"]}
    for t, rows in per.items():
        rows.sort(key=lambda r: (r["ts"], r.get("created_time", "")))
        pos, bad = 0.0, []
        for x in rows:
            n = f(x.get("count_fp") or x.get("count"))
            d = n if x.get("book_side") == "bid" else -n
            o = by_oid.get(x.get("order_id"), {})
            it = classify_cid(o.get("client_order_id"), intents)
            before, pos = pos, pos + d
            if ts(x["created_time"]) < since:
                continue
            want = tick_side.get(t, set()) - {None}
            why = []
            if it.get("role") == "exit" and (abs(pos) > abs(before) + 1e-9 or before * pos < 0):
                why.append("sell_past_held")
            if want == {"yes"} and pos < -1e-9:
                why.append("opposite_side")
            if want == {"no"} and pos > 1e-9:
                why.append("opposite_side")
            if not want and before * pos < 0:
                why.append("sign_flip_unknown_intent")
            if why:
                bad.append(dict(time=x["created_time"], why=why, delta=d, before=before, after=pos,
                                yes_px=x.get("yes_price_dollars"), cid=o.get("client_order_id"),
                                strategy=it.get("strategy"), role=it.get("role")))
        if bad:
            worst = max(bad, key=lambda b: abs(b["after"]))
            flags.append(dict(ticker=t, n_fills=len(bad), kinds=sorted({w for b in bad for w in b["why"]}),
                              intended=sorted(tick_side.get(t, set()) - {None}), first=bad[0]["time"],
                              last=bad[-1]["time"], worst_after=worst["after"], now=cur.get(t),
                              strategy=bad[0]["strategy"], sample=bad[:3]))
        if t in cur and abs(cur[t] - pos) > 1e-6:
            info.append(f"replay of {t} ends at {pos:+g} but exchange shows {cur[t]:+g} (fills before the lookback?)")
    return flags, info


def c2_multi_cid(ex: dict, since: datetime) -> tuple[list, list]:
    jflags = []
    keys = defaultdict(lambda: {"entry": set(), "exit": set(), "ticker": None, "strategy": None})
    strat = {}
    for r in jl(DATA / "lead_exec.jsonl"):
        if r.get("mode") != "live" or not r.get("key") or (ts(r["ts"]) or since) < since:
            continue
        k = keys[r["key"]]
        k["ticker"] = r.get("ticker")
        if r.get("event") == "order":
            strat[r["key"]] = r.get("strategy")
        if r.get("client_order_id"):
            k["entry"].add(r["client_order_id"])
        if r.get("exit_coid"):
            k["exit"].add(r["exit_coid"])
    for key, k in keys.items():
        for role in ("entry", "exit"):
            if len(k[role]) > 1:
                jflags.append(dict(key=key, ticker=k["ticker"], strategy=strat.get(key), role=role,
                                   n_cids=len(k[role])))
    xflags = []
    grp = defaultdict(list)
    for o in ex["orders"]:
        if (ts(o["created_time"]) or since) >= since:
            grp[(o["ticker"], o.get("book_side"))].append(o)
    for (t, side), os_ in grp.items():
        os_.sort(key=lambda o: o["created_time"])
        best = 0
        for i, o in enumerate(os_):
            t0 = ts(o["created_time"])
            w = {p.get("client_order_id") for p in os_[i:] if (ts(p["created_time"]) - t0).total_seconds() <= BURST_WINDOW_S}
            best = max(best, len(w))
        if best >= BURST_CIDS:
            filled = sum(f(o.get("fill_count_fp")) for o in os_)
            xflags.append(dict(ticker=t, book_side=side, max_cids_10min=best, orders=len(os_), filled=filled))
    return jflags, xflags


def c3_price(ex: dict, intents: dict, since: datetime) -> tuple[list, dict, list, list]:
    by_oid = {o["order_id"]: o for o in ex["orders"]}
    worse, counts = [], defaultdict(int)
    filled_by_cid = defaultdict(float)
    for x in ex["fills"]:
        if ts(x["created_time"]) < since:
            continue
        o = by_oid.get(x.get("order_id"), {})
        c = o.get("client_order_id")
        filled_by_cid[c] += f(x.get("count_fp"))
        it = intents.get(c)
        if not it or it.get("limit") in (None, "") or it.get("side") not in ("yes", "no"):
            counts["no_journal_limit"] += 1
            continue
        own = f(x["yes_price_dollars"]) if it["side"] == "yes" else f(x["no_price_dollars"])
        side = "buy" if it["role"] == "entry" else "sell"
        k = classify_fill(side, own, f(it["limit"]))
        counts[k] += 1
        if k == "worse":
            worse.append(dict(ticker=x["ticker"], strategy=it["strategy"], role=it["role"], own_price=own,
                              limit=it["limit"], time=x["created_time"], cid=c))
    over = [dict(cid=c, ticker=intents[c]["ticker"], strategy=intents[c]["strategy"], intended=intents[c]["count"],
                 filled=n) for c, n in filled_by_cid.items()
            if c in intents and intents[c].get("count") and n > intents[c]["count"] + 1e-6]
    unknown = defaultdict(lambda: [0, 0.0])
    for o in ex["orders"]:
        if (ts(o["created_time"]) or since) < since:
            continue
        if not classify_cid(o.get("client_order_id"), intents):
            u = unknown[o["ticker"]]
            u[0] += 1
            u[1] += f(o.get("fill_count_fp"))
    unk = [dict(ticker=t, orders=n, filled=fl) for t, (n, fl) in sorted(unknown.items())]
    return worse, dict(counts), over, unk


def c4_pnl(ex: dict, since: datetime) -> tuple[list, dict]:
    jp = defaultdict(float)
    for name in ("kalshi_exec.jsonl", "lead_exec.jsonl"):
        for r in jl(DATA / name):
            if r.get("event") == "settle" and r.get("mode") == "live" and (ts(r["ts"]) or since) >= since:
                jp[r["ticker"]] += f(r.get("pnl"))
    # Exchange P&L per settled ticker from FILLS in YES terms: cash = -yes_px x delta - fee, plus final position x
    # YES payout. (Settlement revenue - total costs is wrong once anything was sold or netted before settlement:
    # HOUSEAZ2-26-R shows yes 28 / no 28 / revenue 0.)
    cash, pos = defaultdict(float), defaultdict(float)
    for x in ex["fills"]:
        n = f(x.get("count_fp") or x.get("count"))
        d = n if x.get("book_side") == "bid" else -n
        cash[x["ticker"]] += -f(x.get("yes_price_dollars")) * d - f(x.get("fee_cost"))
        pos[x["ticker"]] += d
    xp = {}
    for s in ex["settlements"]:
        t = s["ticker"]
        if t not in cash:
            continue                                  # fills older than the lookback: cannot rebuild
        pay = f(s.get("value"), 100.0 if s.get("market_result") == "yes" else 0.0) / 100.0
        if s.get("market_result") == "no":
            pay = 0.0
        xp[t] = round(cash[t] + pos[t] * pay, 4)
    rows = []
    for t in sorted(set(jp) | set(xp)):
        j, x = jp.get(t), xp.get(t)
        if j is None and x is not None:
            rows.append(dict(ticker=t, journal=None, exchange=x, why="exchange P&L no live journal books"))
        elif x is None:
            continue                                  # not settled on the exchange yet (or fills out of range)
        elif abs(j - x) > 0.05:
            rows.append(dict(ticker=t, journal=round(j, 4), exchange=x, why="journal and exchange disagree"))
    # E4 live vs paper twin (same pick row): per-contract P&L difference
    paper = {}
    for p in jl(DATA / "kalshi_paper.jsonl"):
        paper[(p.get("ticker"), p.get("logged_at"))] = p
    picks = {}
    for r in jl(DATA / "kalshi_exec.jsonl"):
        if r.get("mode") == "live" and r.get("event") == "order" and f(r.get("count")) > 0:
            picks[r["ticker"]] = r
    diffs, disagree = [], []
    for r in jl(DATA / "kalshi_exec.jsonl"):
        if r.get("event") != "settle" or r.get("mode") != "live" or (ts(r["ts"]) or since) < since:
            continue
        o = picks.get(r["ticker"])
        p = paper.get((r["ticker"], (o or {}).get("pick_logged_at")))
        if not o or not p or f(r.get("held")) <= 0:
            continue
        live_pc = f(r["pnl"]) / f(r["held"])
        pr = p.get("result") or r["result"]
        npx, fee = f(p.get("no_price")), f(p.get("fee_per_contract"))
        paper_pc = (1 - npx - fee) if pr == "no" else -(npx + fee)
        diffs.append(live_pc - paper_pc)
        if p.get("result") and p["result"] != r["result"]:
            disagree.append(r["ticker"])
    summ = {"n": len(diffs)}
    if diffs:
        import random
        random.seed(0)
        m = sum(diffs) / len(diffs)
        bs = sorted(sum(random.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(2000))
        summ.update(mean_live_minus_paper_per_contract=round(m, 4), ci95=[round(bs[49], 4), round(bs[1949], 4)],
                    result_disagreements=disagree)
    return rows, summ


def c5_terminal(ex: dict, now: datetime, since: datetime) -> list:
    out = []
    by_cid = {o.get("client_order_id"): o for o in ex["orders"]} if ex else {}
    for name in ("kalshi_exec.jsonl", "lead_exec.jsonl"):
        state = {}
        for r in jl(DATA / name):
            if r.get("mode") != "live" or not r.get("ticker"):
                continue
            k = r.get("key") or r["ticker"]
            if r.get("event") == "order":
                state[k] = dict(r)
            elif r.get("event") in ("update", "settle") and k in state:
                state[k].update({a: b for a, b in r.items() if a not in ("event", "ts")}, settled=r.get("event") == "settle")
        for k, o in state.items():
            t0 = ts(o["ts"])
            if t0 is None or t0 < since or not o.get("client_order_id"):
                continue
            st = o.get("status")
            exp = ts(o.get("expiration_time"))
            xo = by_cid.get(o["client_order_id"])
            why = None
            if st in JOURNAL_PENDING and now - t0 > timedelta(hours=1):
                why = f"journal status {st!r} {(now - t0).total_seconds() / 3600:.0f} h after the order"
            elif st == "resting" and exp and now > exp + timedelta(hours=1) and not o.get("settled"):
                why = "journal resting past expiration, never settled"
            if xo is not None and st in ("resting",) + JOURNAL_PENDING and xo["status"] in TERMINAL_EX:
                why = (why + "; " if why else "") + f"exchange says {xo['status']} (fill {xo.get('fill_count_fp')})"
            if why:
                out.append(dict(journal=name, key=k, ticker=o["ticker"], status=st, why=why,
                                exchange=None if xo is None else xo["status"]))
    for o in (ex or {}).get("orders", []):
        exp = ts(o.get("expiration_time"))
        if o["status"] == "resting" and exp and now > exp + timedelta(hours=1):
            out.append(dict(journal="exchange", key=o.get("client_order_id"), ticker=o["ticker"], status="resting",
                            why="exchange order resting past its expiration_time", exchange="resting"))
    # ForecastEx: every order id needs a terminal row (status filled/cancelled, fill, cancel, settle) after cutoff
    fx = jl(DATA / "forecastex_exec.jsonl")
    last = {}
    orders = {}
    for r in fx:
        oid = r.get("order_id")
        if not oid:
            continue
        if r.get("kind") == "order":
            orders[oid] = r
        if r.get("kind") in ("fill", "cancel", "settle") or (r.get("kind") == "status" and str(r.get("status")).lower()
                                                             in ("filled", "cancelled", "canceled", "inactive",
                                                                 "expired_not_found")):
            last[oid] = r
    for oid, r in orders.items():
        cut = ts(r.get("cutoff"))
        if (ts(r["ts"]) or since) >= since and oid not in last and cut and now > cut + timedelta(hours=12):
            out.append(dict(journal="forecastex_exec.jsonl", key=oid, ticker=r.get("ticker"), status="open",
                            why="no terminal row 12 h after cutoff", exchange=None))
    resp = {r.get("tag") for r in jl(DATA / "forecastex_orders.jsonl") if r.get("event") == "response"}
    for r in jl(DATA / "forecastex_orders.jsonl"):
        if r.get("event") == "intent" and r.get("mode") == "live" and r.get("tag") not in resp \
                and (ts(r["ts"]) or since) >= since:
            out.append(dict(journal="forecastex_orders.jsonl", key=r.get("tag"), ticker=r.get("market"),
                            status="intent", why="intent with no response row", exchange=None))
    return out


def c6_positions(ex: dict, tick_side: dict) -> list:
    out = []
    for p in ex["positions"]:
        pos = f(p.get("position_fp"))
        if abs(pos) < 1e-9:
            continue
        want = tick_side.get(p["ticker"], set()) - {None}
        side = "yes" if pos > 0 else "no"
        if not want:
            out.append(dict(ticker=p["ticker"], position=pos, exposure=p.get("market_exposure_dollars"),
                            why="no journal claims this ticker (manual / needle / untracked bot)"))
        elif side not in want:
            out.append(dict(ticker=p["ticker"], position=pos, exposure=p.get("market_exposure_dollars"),
                            why=f"held {side}, journal intended {sorted(want)}"))
    return out


# ------------------------------------------------------------------------------------------------ report
LESSON_FOR = {"sell_past_held": "L-nflf-runaway-exit", "opposite_side": "L-nflf-runaway-exit",
              "multi_cid": "L-nflf-new-cid-per-retry", "price_worse": "L-fill-worse-than-limit",
              "over_fill": "L-filled-more-than-intended", "unjournaled": "L-unjournaled-orders",
              "pnl_divergence": "L-journal-vs-exchange-pnl", "no_terminal": "L-orders-without-terminal-state",
              "orphan_position": "L-unjournaled-orders"}


def md_table(rows: list[dict], cols: list[str], cap: int = 40) -> str:
    if not rows:
        return "_none_\n"
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows[:cap]:
        s += "| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |\n"
    if len(rows) > cap:
        s += f"\n... {len(rows) - cap} more\n"
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=14)
    ap.add_argument("--offline", action="store_true", help="journals only; skip the Kalshi GETs")
    ap.add_argument("--no-ledger", action="store_true", help="do not append new lessons to lessons.jsonl")
    ap.add_argument("--out", default=None, help="report path (default sports/notes/post_trade_review_<date>.md)")
    ap.add_argument("--ledger", default=None, help="lessons.jsonl path (launchd copy cannot write ~/Desktop: TCC)")
    ap.add_argument("--snap-root", default=None, help="dir for raw exchange snapshots (default sports/results/post_trade_review)")
    a = ap.parse_args(argv)
    global LEDGER
    if a.ledger:
        LEDGER = Path(a.ledger)
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=a.days)
    day = now.date().isoformat()
    snap = (Path(a.snap_root) if a.snap_root else SPORTS / "results" / "post_trade_review") / day
    intents, tick_side, _needle = journal_intents()
    ex = None if a.offline else fetch_exchange(since, snap)
    empty = {"fills": [], "orders": [], "positions": [], "settlements": []}
    exd = ex or empty

    c1, c1info = c1_replay(exd, intents, tick_side, since)
    c2j, c2x = c2_multi_cid(exd, since)
    c3w, c3counts, c3over, c3unk = c3_price(exd, intents, since)
    c4rows, c4paper = c4_pnl(exd, since)
    c5 = c5_terminal(ex, now, since)
    c6 = c6_positions(exd, tick_side)

    fired = set()
    if any("sell_past_held" in r["kinds"] for r in c1):
        fired.add("sell_past_held")
    if any("opposite_side" in r["kinds"] for r in c1):
        fired.add("opposite_side")
    if c2j or c2x:
        fired.add("multi_cid")
    if c3w:
        fired.add("price_worse")
    if c3over:
        fired.add("over_fill")
    if c3unk:
        fired.add("unjournaled")
    if c4rows:
        fired.add("pnl_divergence")
    if c5:
        fired.add("no_terminal")
    if c6:
        fired.add("orphan_position")

    L = []
    L.append(f"# Post-trade review {day} (last {a.days:g} days, since {since.isoformat(timespec='minutes')})\n")
    L.append(f"Generated by `sports/scripts/post_trade_review.py` at {now.isoformat(timespec='seconds')}. READ-ONLY. "
             f"Exchange: {'OFFLINE (journals only)' if ex is None else f'{len(exd['fills'])} fills / {len(exd['orders'])} orders since window-30d, {len(exd['positions'])} positions, {len(exd['settlements'])} settlements; raw in sports/results/post_trade_review/{day}/'}.\n")
    L.append("## Summary\n")
    summ = [dict(check="C1 sell past held / opposite side (exchange replay)", flags=len(c1)),
            dict(check="C2 >1 cid per action (journal)", flags=len(c2j)),
            dict(check="C2 cid bursts >=3 in 10 min (exchange)", flags=len(c2x)),
            dict(check="C3 fills worse than journal limit", flags=len(c3w)),
            dict(check="C3 filled more than journal count", flags=len(c3over)),
            dict(check="C3 exchange orders no journal claims (tickers)", flags=len(c3unk)),
            dict(check="C4 journal vs exchange settle P&L (> $0.05)", flags=len(c4rows)),
            dict(check="C5 orders with no terminal state", flags=len(c5)),
            dict(check="C6 open positions not matching journal intent", flags=len(c6))]
    L.append(md_table(summ, ["check", "flags"]))
    L.append(f"\nFill price vs limit, all fills in window (price improvement is NOT an anomaly): `{c3counts}`\n")
    L.append("\n## C1 sells past held / positions opposite intent\n")
    L.append(md_table(c1, ["ticker", "strategy", "intended", "kinds", "n_fills", "first", "last", "worst_after", "now"]))
    for r in c1[:10]:
        L.append(f"- {r['ticker']} sample: " + "; ".join(f"{b['time'][:19]} {b['before']:+g}->{b['after']:+g} "
                                                         f"yes@{b['yes_px']} {b['role']}" for b in r["sample"]))
    if c1info:
        L.append("\nReplay notes: " + "; ".join(c1info[:15]))
    L.append("\n\n## C2 more than one client_order_id per intended action\n")
    L.append("Journal (lead_exec key):\n\n" + md_table(c2j, ["key", "ticker", "strategy", "role", "n_cids"]))
    L.append("\nExchange bursts:\n\n" + md_table(c2x, ["ticker", "book_side", "max_cids_10min", "orders", "filled"]))
    L.append("\n## C3 price band / over-fill / unjournaled orders\n")
    L.append("Worse than limit:\n\n" + md_table(c3w, ["time", "ticker", "strategy", "role", "own_price", "limit"]))
    L.append("\nFilled more than intended:\n\n" + md_table(c3over, ["ticker", "strategy", "intended", "filled", "cid"]))
    L.append("\nExchange orders no journal claims (needle bot without cid prefix, manual, or a journal gap):\n\n"
             + md_table(c3unk, ["ticker", "orders", "filled"]))
    L.append("\n## C4 P&L: journal vs exchange settlements; E4 live vs paper\n")
    L.append(md_table(c4rows, ["ticker", "journal", "exchange", "why"], cap=60))
    L.append(f"\nE4 live minus paper twin, $/contract (bootstrap 95% CI): `{c4paper}`\n")
    L.append("\n## C5 orders with no terminal state\n")
    L.append(md_table(c5, ["journal", "key", "ticker", "status", "exchange", "why"]))
    L.append("\n## C6 open positions vs journal intent\n")
    L.append(md_table(c6, ["ticker", "position", "exposure", "why"]))

    added = []
    if not a.no_ledger:
        have = {r.get("id") for r in jl(LEDGER)}
        for k in sorted(fired):
            lid = LESSON_FOR[k]
            if lid in have or lid in {x["id"] for x in added}:
                continue
            added.append(dict(id=lid, date=day, source="post_trade_review", incident=f"post_trade_review {day} check {k} fired",
                              mechanism="TBD by the lead after reading the flagged rows", check=f"post_trade_review.py {k}",
                              check_location="sports/scripts/post_trade_review.py", status="needs_triage",
                              evidence=f"sports/notes/post_trade_review_{day}.md"))
        if added:
            with LEDGER.open("a") as fh:
                for r in added:
                    fh.write(json.dumps(r) + "\n")
    L.append("\n## Ledger\n")
    L.append(("New lesson rows appended to sports/notes/lessons.jsonl: " + ", ".join(r["id"] for r in added))
             if added else "No new lesson types (every fired check already has a ledger row).")
    out = Path(a.out) if a.out else SPORTS / "notes" / f"post_trade_review_{day}.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}; fired: {sorted(fired) or 'none'}; ledger +{len(added)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
