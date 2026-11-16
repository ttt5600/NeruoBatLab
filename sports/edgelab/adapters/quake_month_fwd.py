"""QM1 / QM2 KXBIGGESTQUAKEMONTH forward paper (qm1_quake_month_taker, qm2_quake_month_maker; logger
scripts/quake_monthly_logger.py, prereg notes/quake_monthly.md, ops notes there under 'Ops').

Source (read-only): ~/.claude/quant-sports/data/kx_forward/quake_monthly.jsonl (launchd job quake-monthly; repo data/ as
fallback). params: {rule: QM1 | QM2}.
  `pick`   QM1, first fire per (ticker, side) at a DECISION snapshot (first run in [15:00, 15:45) UTC); status bet.
  `post`   QM2, one per (ticker, side, decision day): hypothetical bid q, valid until next day 15:00 UTC / close.
  `fill`   one per post once its window ended: filled iff a public print in (decided_at, valid_until] with the opposite
           taker side at S-price <= q (fill_lt = strictly below q, reported).
  `settle` one per logged ticker once Kalshi settled it: result yes|no.
Bet row:
  decided_at = end of that market's book read; inputs_as_of = USGS fetch time (before every Kalshi GET of the snapshot)
  QM1: price = ask_S (taker), price_time = book read; QM2: price = q (maker_tape), price_time = fill_ts,
       fill_deadline = valid_until; contracts 1, stake = price
  event_start = month end (UTC 1st of next month 00:00) -- IN-PLAY by construction (the month window is running; edgelab
  in-play convention: event_start = when the outcome is fixed at the latest). cluster = 'quake_month|YYYY-MM' (all rungs,
  sides and days of a month share one outcome = the prereg's unit of inference).
Outcome (hidden): `result`. Unsettled (or unfilled-window-open) are pending, counted not scored.
"""
from __future__ import annotations

import pandas as pd

from ._common import AdapterError, date_range, empty_bets, qs_or_repo, read_jsonl, with_report

OUTCOME = {"kind": "result_side", "key": "bet_id"}
HIDDEN = ["result"]


def load(params: dict):
    rule = params.get("rule")
    if rule not in ("QM1", "QM2"):
        raise AdapterError(f"unknown quake rule {rule!r}")
    f = qs_or_repo("kx_forward/quake_monthly.jsonl")
    rs = read_jsonl(f)
    kind = "pick" if rule == "QM1" else "post"
    xs = [r for r in rs if r.get("kind") == kind and r.get("rule") == rule]
    rep = {"source": str(f), "rule": rule, "rows": len(xs), "days": sum(1 for r in rs if r.get("kind") == "day"),
           "decision_days": len({r["date"] for r in rs if r.get("kind") == "day" and r.get("decision")})}
    if not xs:
        e = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "bet_id": pd.Series(dtype=str)})
        e.attrs["loader_report"] = rep
        return e, [str(f)]
    res = {}
    for r in rs:
        if r.get("kind") == "settle":
            if r["ticker"] in res and res[r["ticker"]] != r["result"]:
                raise AdapterError(f"{f}: {r['ticker']} settled twice with different results")
            res[r["ticker"]] = r["result"]
    d = pd.DataFrame(xs)
    d["bet_id"] = (d["ticker"] + "|" + d["side"]) if rule == "QM1" else d["post_id"]
    if d["bet_id"].duplicated().any():
        raise AdapterError(f"{f}: duplicate {kind} rows: {d.loc[d['bet_id'].duplicated(), 'bet_id'].tolist()[:3]}")
    if rule == "QM2":
        fl = {r["post_id"]: r for r in rs if r.get("kind") == "fill"}
        d["fill_known"] = d["post_id"].isin(fl)
        d["filled"] = d["post_id"].map(lambda p: bool(fl.get(p, {}).get("filled")))
        d["fill_lt"] = d["post_id"].map(lambda p: bool(fl.get(p, {}).get("fill_lt")))
        d["fill_ts"] = d["post_id"].map(lambda p: fl.get(p, {}).get("fill_ts"))
    d["date"] = pd.to_datetime(d["date"])                       # UTC decision date = the vault split column
    d["result"] = d["ticker"].map(res)
    d["settled"] = d["result"].isin(["yes", "no"])
    d.attrs["loader_report"] = rep
    return d, [str(f)]


def rows(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    rep = dict(frame.attrs.get("loader_report") or {})
    if frame.empty or "side" not in frame:
        return with_report(empty_bets(), {**rep, "rows_in_window": 0, "bets": 0})
    f = frame
    if not f["side"].isin(["yes", "no"]).all():
        raise AdapterError("rows with a side other than yes/no")
    rep.update({"rows_in_window": int(len(f)), "dates": date_range(f["date"])})
    if rep["rule"] == "QM1":
        live = f["status"] == "bet"
        g = f[live & f["settled"]]
        rep.update({"bets_logged": int(live.sum()), "bets_unsettled_pending": int((live & ~f["settled"]).sum())})
        price, src, pt, dl = g["ask"].astype(float), "taker", pd.to_datetime(g["quote_time"], utc=True), pd.NaT
    else:
        fil = f["fill_known"] & f["filled"]
        g = f[fil & f["settled"]]
        # adverse-selection check (prereg): loss rate filled vs unfilled among settled posts
        s = f[f["settled"] & f["fill_known"]]
        lost = s["result"] != s["side"]
        rep.update({"posts": int(len(f)), "fill_window_open": int((~f["fill_known"]).sum()), "filled": int(fil.sum()),
                    "filled_strict_lt": int((f["fill_known"] & f["fill_lt"]).sum()),
                    "filled_unsettled_pending": int((fil & ~f["settled"]).sum()),
                    "loss_rate_filled": float(lost[s["filled"]].mean()) if s["filled"].any() else None,
                    "loss_rate_unfilled": float(lost[~s["filled"]].mean()) if (~s["filled"]).any() else None})
        price, src = g["q"].astype(float), "maker_tape"
        pt, dl = pd.to_datetime(g["fill_ts"], utc=True), pd.to_datetime(g["valid_until"], utc=True).to_numpy()
    rep.update({"bets": int(len(g)), "clusters": int(g["cluster"].nunique()) if len(g) else 0})
    b = pd.DataFrame({"bet_id": g["bet_id"].to_numpy(), "cluster": g["cluster"].to_numpy(),
                      "event_start": pd.to_datetime(g["month_end"], utc=True).to_numpy(),
                      "decided_at": pd.to_datetime(g["decided_at"], utc=True).to_numpy(),
                      "inputs_as_of": pd.to_datetime(g["inputs_as_of"], utc=True).to_numpy(),
                      "price": price.to_numpy(), "price_source": src, "price_time": pt.to_numpy(),
                      "side": g["side"].to_numpy(), "contracts": 1.0, "stake": price.to_numpy(),
                      "close_price": float("nan"), "fill_deadline": dl})
    return with_report(b, rep)
