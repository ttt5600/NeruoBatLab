"""Snow markets forward paper (tg1_snow_thanksgiving_taker, sd1_snowdays_climo_taker, sd2_snowdays_conditional_taker):
bets from scripts/snow_markets.py (prereg sports/notes/snow_markets.md; ops notes there under 'Ops').

Source (read-only): ~/.claude/quant-sports/results/snow_markets/snow_log.jsonl (launchd job snow-markets; the repo copy
sports/results/snow_markets/ is the fallback and holds the first snapshot 2026-11-16 11:07Z, seeded into the deployed log)
  snapshot rows (no `kind`): ts, rule (TG1|SD1|SD2), ticker, yb, ya, fire (yes|no|None), cost (ya + exact fee | 1-yb + fee).
  snow_settle.jsonl: one row per fired ticker once Kalshi says settled/finalized: result, settlement_value_dollars,
  expiration_value (the count for KXSNOWDAYS), custom_strike (scalar_floor / scalar_cap).
params: {rule: TG1 | SD1 | SD2}.
Bets = FIRST fire per (rule, ticker, side), recomputed here from the raw snapshot rows (the logger's first_fire flag is
informational). Only PRE-EVENT decisions count for TG1/SD1 (decided_at < event_start); snapshots at/after it (the
logger also runs on 11-26 and on 12-01) are dropped BEFORE first fires are taken and are counted in the report.
  decided_at = inputs_as_of = price_time = snapshot ts (one Kalshi GET + ACIS/Open-Meteo pulls at that tick)
  price      = ya (YES) or 1 - yb (NO), taker; contracts 1; fee exact .07 p(1-p) (spec fee.rounding exact)
  event_start: TG1 = 2026-11-26 00:00 LST at the station (UTC); SD1 = 2026-12-01 00:00 EST = 05:00Z;
               SD2 = market close 2027-04-02 04:59Z (in-play count market: the count is final from close on -- same
               convention as rules/spacex_count.py; SD2 is in-play by design, it conditions on the observed count).
  cluster    = the event (TG1: 2026-11-26; SD: the season) -> ONE cluster each: the cities share one weather draw.
  close_price= same-side mid of the LAST logged snapshot of that ticker with ts < close/event_start (NaN if none).
Outcome (hidden):
  TG1 binary: result 'yes'/'no' from Kalshi.
  SD scalar: YES value v = settlement_value_dollars (Kalshi), cross-checked against floor_cent(clip((N - floor)/(cap -
  floor))) with N = expiration_value when both exist (disagreement -> AdapterError, never scored silently).
  v in {0, 1} -> result 'yes'/'no' (exact under any scoring version). 0 < v < 1 -> result 'scalar' + settle_value = v,
  which edgelab scores at the true value ONLY under SCORING_VERSION >= 2; under v1 the vault would score both sides 0,
  so rows() raises instead (fail closed). true_value_pnl() below scores the true value independently of edgelab.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..paths import QS_DATA, SPORTS
from ._common import AdapterError, date_range, empty_bets, first_existing, read_jsonl, with_report

OUTCOME = {"kind": "result_side", "key": "bet_id", "value_col": "settle_value"}
HIDDEN = ["result", "settle_value"]
RULES = ("TG1", "SD1", "SD2")
REL = ("results", "snow_markets")
QS_ROOT = QS_DATA.parent
LST_OFF = {"E": -5, "C": -6, "M": -7, "P": -8}
ZONE = {**{s: "E" for s in "BOS PWM PVD NYC EWR TTN PHL PIT BWI DCA CLE CMH DTW IND ATL CLT CVG".split()},
        **{s: "C" for s in "ORD MKE MSN MSP DSM AUS DFW HOU MCI BNA".split()},
        "DEN": "M", "SEA": "P", "PDX": "P"}
SD1_START = pd.Timestamp("2026-12-01 05:00", tz="UTC")
SD_CLOSE = pd.Timestamp("2027-04-02 04:59", tz="UTC")


def paths():
    log = first_existing(QS_ROOT.joinpath(*REL, "snow_log.jsonl"), SPORTS.joinpath(*REL, "snow_log.jsonl"))
    return log, log.parent / "snow_settle.jsonl"


def floor_cent(x: float) -> float:
    return math.floor(round(x * 100, 6)) / 100


def event_start(rule: str, stn: str) -> pd.Timestamp:
    if rule == "TG1":
        return pd.Timestamp(datetime(2026, 11, 26, tzinfo=timezone.utc) - timedelta(hours=LST_OFF[ZONE[stn]]))
    return SD1_START if rule == "SD1" else SD_CLOSE


def yes_value(s: dict, ticker: str) -> float | None:
    """Settled YES value per contract from a snow_settle row (scalar or binary). Loud on inconsistency."""
    res = s.get("result")
    sv = s.get("settlement_value_dollars")
    sv = float(sv) if sv not in (None, "") else None
    if "KXSNOWDAYS" not in ticker:
        if res not in ("yes", "no"):
            raise AdapterError(f"{ticker}: binary market settled {res!r}")
        return 1.0 if res == "yes" else 0.0
    cs = s.get("custom_strike") or {}
    own = None
    if s.get("expiration_value") not in (None, "") and "scalar_floor" in cs:
        lo, hi, n = float(cs["scalar_floor"]), float(cs["scalar_cap"]), float(s["expiration_value"])
        own = floor_cent(min(max((n - lo) / (hi - lo), 0.0), 1.0))
    v = sv if sv is not None else (1.0 if res == "yes" else 0.0 if res == "no" else own)
    if v is None:
        raise AdapterError(f"{ticker}: settled {res!r} with no settlement value and no expiration_value")
    if own is not None and abs(own - v) > 0.005:
        raise AdapterError(f"{ticker}: Kalshi value {v} != payout(expiration_value={s.get('expiration_value')}) {own}")
    if not 0 <= v <= 1:
        raise AdapterError(f"{ticker}: settlement value {v} outside [0, 1]")
    return v


def first_fires(snaps: pd.DataFrame, rule: str) -> tuple[pd.DataFrame, int]:
    """-> (first fire per ticker-side among pre-event snapshots of `rule`, # fire rows dropped as at/after event_start)."""
    d = snaps[(snaps["rule"] == rule) & snaps["fire"].isin(["yes", "no"])].copy()
    if d.empty:
        return d, 0
    d["decided_at"] = pd.to_datetime(d["ts"], utc=True, format="ISO8601")
    d["event_start"] = [event_start(rule, s) for s in d["stn"]]
    late = d["decided_at"] >= d["event_start"]
    d = d[~late].sort_values("decided_at")
    return d.drop_duplicates(["ticker", "fire"], keep="first"), int(late.sum())


def _close_mid(snaps: pd.DataFrame, ticker: str, side: str, before: pd.Timestamp) -> float:
    s = snaps[(snaps["ticker"] == ticker) & (snaps["_ts"] < before)].dropna(subset=["yb", "ya"])
    if s.empty:
        return np.nan
    m = (float(s.iloc[-1]["yb"]) + float(s.iloc[-1]["ya"])) / 2
    return m if side == "yes" else 1 - m


def load(params: dict):
    rule = (params or {}).get("rule")
    if rule not in RULES:
        raise AdapterError(f"rule must be one of {RULES}, got {rule!r}")
    log, sf = paths()
    snaps = pd.DataFrame([r for r in read_jsonl(log) if "kind" not in r])
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "bet_id": pd.Series(dtype=str)})
    if snaps.empty:
        empty.attrs["loader_report"] = {"source": str(log), "snapshot_rows": 0}
        return empty, [str(log)]
    snaps["_ts"] = pd.to_datetime(snaps["ts"], utc=True, format="ISO8601")
    snaps = snaps.sort_values("_ts")
    st = read_jsonl(sf)
    sd = {r["ticker"]: r for r in st}
    if len(sd) != len(st):
        raise AdapterError(f"{sf}: a ticker was settled twice")
    d, late = first_fires(snaps, rule)
    rep = {"source": str(log), "settle_source": str(sf), "rule": rule, "snapshot_rows": int(len(snaps)),
           "snapshots_rule": int((snaps["rule"] == rule).sum()),
           "snapshot_days": int(snaps.loc[snaps["rule"] == rule, "_ts"].dt.date.nunique()),
           "fire_rows_at_or_after_event_start_dropped": late, "settle_rows": len(st)}
    if d.empty:
        empty.attrs["loader_report"] = {**rep, "first_fires": 0}
        return empty, [str(log), str(sf)]
    d["side"] = d["fire"]
    d["bet_id"] = rule + "|" + d["ticker"] + "|" + d["side"]
    d["date"] = d["decided_at"].dt.tz_localize(None).dt.normalize()
    d["price"] = np.where(d["side"] == "yes", d["ya"].astype(float), 1 - d["yb"].astype(float))
    vals = [yes_value(sd[t], t) if t in sd else None for t in d["ticker"]]
    d["yes_value"] = vals
    d["settled"] = [v is not None for v in vals]

    def res(v, t):
        if v is None:
            return None
        if "KXSNOWDAYS" not in t or v in (0.0, 1.0):
            return "yes" if v == 1.0 else "no"
        return "scalar"
    d["result"] = [res(v, t) for v, t in zip(vals, d["ticker"])]
    d["settle_value"] = vals
    d["close_mid"] = [_close_mid(snaps, t, s, min(e, SD_CLOSE)) for t, s, e in zip(d["ticker"], d["side"], d["event_start"])]
    d.attrs["loader_report"] = {**rep, "first_fires": int(len(d)), "sides": d["side"].value_counts().to_dict(),
                                "scalar_settled": int((d["result"] == "scalar").sum())}   # hidden cols are stripped
    # before rows() sees the frame, so the v1 fail-closed check reads this count from the report
    return d.reset_index(drop=True), [str(log), str(sf)]


def rows(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    rep = dict(frame.attrs.get("loader_report") or {})
    if frame.empty or "side" not in frame:
        return with_report(empty_bets(), {**rep, "rows_in_window": 0, "bets": 0})
    f = frame
    g = f[f["settled"]]
    from .. import SCORING_VERSION
    if SCORING_VERSION < 2 and rep.get("scalar_settled", 0) > 0:
        raise AdapterError(f"{rep['scalar_settled']} bets settled at a fractional scalar value; edgelab "
                           "SCORING_VERSION 1 would score them 0 for both sides. Needs SCORING_VERSION >= 2 (user's "
                           "switch); true-value P&L: edgelab.adapters.snow_fwd.true_value_pnl()")
    rep.update({"rows_in_window": int(len(f)), "dates": date_range(f["date"]), "first_fires": int(len(f)),
                "bets_unsettled_pending": int((~f["settled"]).sum()), "bets": int(len(g)),
                "close_missing": int(pd.to_numeric(g["close_mid"], errors="coerce").isna().sum())})
    cl = g["ticker"].str.rsplit("-", n=1).str[0]                    # event ticker = one weather draw
    b = pd.DataFrame({"bet_id": g["bet_id"].to_numpy(), "cluster": cl.to_numpy(),
                      "event_start": g["event_start"].to_numpy(), "decided_at": g["decided_at"].to_numpy(),
                      "inputs_as_of": g["decided_at"].to_numpy(), "price": g["price"].to_numpy(float),
                      "price_source": "taker", "price_time": g["decided_at"].to_numpy(), "side": g["side"].to_numpy(),
                      "contracts": 1.0, "stake": g["price"].to_numpy(float),
                      "close_price": pd.to_numeric(g["close_mid"], errors="coerce").to_numpy(float)})
    return with_report(b, rep)


def true_value_pnl(rule: str) -> dict:
    """Independent of edgelab scoring: per-bet P&L at the TRUE settlement value (YES v, NO 1 - v), cost = price + exact
    fee .07 p(1-p), 1 contract. -> {'bets': [...], 'n_fired', 'n_settled', 'pnl', 'cost', 'units'}."""
    d, _ = load({"rule": rule})
    if d.empty:
        return {"rule": rule, "bets": [], "n_fired": 0, "n_settled": 0, "pnl": 0.0, "cost": 0.0, "units": math.nan,
                "report": d.attrs.get("loader_report")}
    out = []
    for r in d.itertuples():
        cost = r.price + 0.07 * r.price * (1 - r.price)
        pay = None if (r.yes_value is None or pd.isna(r.yes_value)) else (r.yes_value if r.side == "yes" else 1 - r.yes_value)
        out.append({"bet_id": r.bet_id, "ticker": r.ticker, "side": r.side, "decided_at": str(r.decided_at),
                    "price": r.price, "cost": cost, "close_mid": r.close_mid, "payout": pay,
                    "pnl": None if pay is None else pay - cost})
    s = [o for o in out if o["pnl"] is not None]
    pnl, cost = sum(o["pnl"] for o in s), sum(o["cost"] for o in s)
    return {"rule": rule, "bets": out, "n_fired": len(out), "n_settled": len(s), "pnl": pnl, "cost": cost,
            "units": pnl / cost if cost else math.nan, "report": d.attrs.get("loader_report")}
