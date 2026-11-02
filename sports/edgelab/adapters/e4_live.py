"""E4 maker (orig-7 = e4_maker_e1, X1 = e4_maker_x1): forward bets = the LIVE executor's real maker orders.

Source (read-only): ~/.claude/quant-kalshi/data/kalshi_exec.jsonl, written by quant-kalshi scripts/kalshi_exec.py.
  `order`  rows (mode live, exec_style maker, cohort orig7|x1, status submitting = posted; skip_* = never posted)
  `update` rows (POST response with order_id, then hourly `reconcile` snapshots: filled, avg_no, fee, status)
  `settle` rows (per ticker: result, held, pnl -- the exchange settlement as the executor journaled it)
There is no separate fills file: the executor journals fills through these updates.

Bet row (one per posted order that filled > 0 contracts and settled):
  decided_at   = the order row's ts (post time; the executor posts within seconds of it)
  inputs_as_of = pick_logged_at (when kalshi_paper logged the frozen pick: forecasts + quote at T)
  price        = avg_no (the real average NO fill price); price_source maker_fill; contracts = filled
  price_time   = first journal snapshot showing filled > 0 (an UPPER bound on the fill time: reconcile is hourly)
  fill_deadline= the order's expiration_time (Kalshi cannot fill after it)
  event_start  = local midnight of the ticker's observation day at the station (wx.CITIES + kalshi_paper.X1_CITIES tz)
  close_price  = edgelab's E4 CLV close: last public print with taker_side 'no' in (post, post + 10 h] (no_price)
  fee          = the fee schedule (maker 0 on fee_type 'quadratic'); the executor's journaled fee and settle pnl are
                 cross-checked by the evaluator (venue_pnl); any order with fee_type != quadratic fails loudly.
Outcome (hidden): settle `result` ('no' -> our NO wins). Unsettled fills are pending (counted, not scored).
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ._common import QK_DATA, AdapterError, date_range, empty_bets, imp_scripts, read_jsonl, tape_last_no, with_report

OUTCOME = {"kind": "result_side", "key": "bet_id"}
HIDDEN = ["result", "venue_pnl"]
EXEC_LOG = QK_DATA / "kalshi_exec.jsonl"
CLOSE_H = 10


def _tz_map() -> dict:
    imp_scripts()
    from sports import wx
    import kalshi_paper as kp
    return {s: tz for s, (_st, tz) in {**wx.CITIES, **kp.X1_CITIES}.items()}


def _event_day(ticker: str) -> pd.Timestamp:
    return pd.to_datetime(ticker.split("-")[1], format="%y%b%d")


def load(params: dict):
    cohort = params["cohort"]
    ev = read_jsonl(EXEC_LOG)
    if not ev:
        raise AdapterError(f"{EXEC_LOG} missing or empty")
    tzs = _tz_map()
    orders = [e for e in ev if e.get("event") == "order" and e.get("mode") == "live" and e.get("exec_style") == "maker"
              and e.get("cohort") == cohort]
    posted = [o for o in orders if o.get("status") == "submitting"]
    skipped = {}
    for o in orders:
        if o.get("status") != "submitting":
            skipped[o["status"]] = skipped.get(o["status"], 0) + 1
    upd = [e for e in ev if e.get("event") == "update" and e.get("mode") == "live"]
    settles = [e for e in ev if e.get("event") == "settle" and e.get("mode") == "live"]
    out, rejected = [], 0
    for o in posted:
        tk, t0 = o["ticker"], o["ts"]
        if o.get("fee_type") != "quadratic":
            raise AdapterError(f"{tk}: fee_type {o.get('fee_type')!r}; the maker_fill schedule assumes 'quadratic' (fee 0)")
        u = sorted([x for x in upd if x.get("ticker") == tk and x["ts"] >= t0], key=lambda x: x["ts"])
        post = [x for x in u if "http" in x]
        if not post:
            raise AdapterError(f"{tk} posted {t0} but no POST response journaled")
        if post[0]["http"] != 201:
            rejected += 1
            continue
        oid = post[0]["order_id"]
        mine = [x for x in u if x.get("order_id") == oid]
        seen = [x for x in mine if (x.get("filled") or 0) > 0]
        last = mine[-1]
        st = [s for s in settles if s["ticker"] == tk and s["ts"] >= t0]
        if len(st) > 1:
            raise AdapterError(f"{tk}: {len(st)} settle rows after {t0}")
        filled = float(last.get("filled") or 0)
        if st and filled > 0 and abs(float(st[0]["held"]) - filled) > 1e-9:
            raise AdapterError(f"{tk}: settled held {st[0]['held']} != journaled fill {filled}")
        series = tk.split("-")[0]
        if series not in tzs:
            raise AdapterError(f"{tk}: no station timezone for series {series}")
        day = _event_day(tk)
        midnight = datetime.combine(day.date(), dtime(0), ZoneInfo(tzs[series])).astimezone(timezone.utc)
        exp = o.get("expiration_time")
        out.append({
            "bet_id": f"{tk}|{oid}", "ticker": tk, "order_id": oid, "cohort": cohort, "date": day,
            "decided_at": pd.Timestamp(t0), "inputs_as_of": pd.Timestamp(o.get("pick_logged_at") or t0),
            "event_start": pd.Timestamp(midnight), "q": float(o["maker_q"]), "count": float(o["count"]),
            "pick_no_price": o.get("pick_no_price"), "no_ask_now": o.get("no_ask_now"),
            "fill_price": float(last["avg_no"]) if last.get("avg_no") is not None else np.nan,
            "contracts": filled, "fee_journal": float(last.get("fee") or 0), "status": last.get("status"),
            "fill_seen_at": pd.Timestamp(seen[0]["ts"]) if seen else pd.NaT,
            "fill_deadline": pd.Timestamp(int(exp), unit="s", tz="UTC") if exp else pd.NaT,
            "settled": bool(st), "result": st[0]["result"] if st else None,
            "venue_pnl": float(st[0]["pnl"]) if st else np.nan,
        })
    df = pd.DataFrame(out)
    if df.empty:
        df = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "bet_id": pd.Series(dtype=str)})
    # CLV close from the public tape, only for filled orders whose (post, post+10h] window is over
    closes = []
    for r in df.itertuples() if len(df) and "contracts" in df else []:
        if r.contracts > 0:
            t0 = int(r.decided_at.timestamp())
            try:
                closes.append(tape_last_no(r.ticker, t0, t0 + CLOSE_H * 3600))
            except Exception as e:  # noqa: BLE001 -- recorded; CLV coverage shows it
                closes.append(np.nan)
                df.attrs.setdefault("tape_errors", []).append(f"{r.ticker}: {type(e).__name__}")
        else:
            closes.append(np.nan)
    if len(df) and "contracts" in df:
        df["close_no"] = closes
    df.attrs["loader_report"] = {"source": str(EXEC_LOG), "orders_seen": len(orders), "posted": len(posted),
                                 "not_posted_by_executor": skipped, "post_rejected": rejected,
                                 "first_maker_post": min((o["ts"] for o in posted), default=None)}
    return df, [str(EXEC_LOG)]


def rows(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    rep = dict(frame.attrs.get("loader_report") or {})
    if frame.empty or "contracts" not in frame:
        return with_report(empty_bets(), {**rep, "rows_in_window": 0, "bets": 0})
    f = frame
    filled = f["contracts"] > 0
    ok = filled & f["settled"]
    rep.update({"rows_in_window": int(len(f)), "dates": date_range(f["date"]), "filled": int(filled.sum()),
                "no_fill_or_open": int((~filled).sum()), "filled_unsettled_pending": int((filled & ~f["settled"]).sum()),
                "bets": int(ok.sum()), "bet_dates": date_range(f.loc[ok, "date"]),
                "fill_price_ne_q": int((ok & ((f["fill_price"] - f["q"]).abs() > 1e-9)).sum()),
                "fee_journal_nonzero": int((ok & (f["fee_journal"].abs() > 1e-12)).sum()),
                "partial_fills": int((ok & (f["contracts"] < f["count"] - 1e-9)).sum()),
                "tape_errors": frame.attrs.get("tape_errors", [])})
    g = f[ok]
    b = pd.DataFrame({
        "bet_id": g["bet_id"].to_numpy(), "cluster": g["date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "event_start": g["event_start"].to_numpy(), "decided_at": g["decided_at"].to_numpy(),
        "inputs_as_of": g["inputs_as_of"].to_numpy(), "price": g["fill_price"].to_numpy(float),
        "price_source": "maker_fill", "price_time": g["fill_seen_at"].to_numpy(), "side": "no",
        "contracts": g["contracts"].to_numpy(float), "stake": (g["contracts"] * g["fill_price"]).to_numpy(float),
        "close_price": g["close_no"].to_numpy(float), "fill_deadline": g["fill_deadline"].to_numpy(),
    })
    return with_report(b, rep)
