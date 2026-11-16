"""ot1_extension: OT1 (options-implied tail fade, taker) on OTHER Kalshi crypto ladders.
Prereg: notes/ot1_extension.md (written before any quote / outcome of these markets was read).
Reuses scripts/options_implied_tails.py (smile, p_above_crypto, p_yes, tail_candidates, deribit_window) UNCHANGED.

    python scripts/ot1_extension.py census     # market lists (result -> hidden column), outcome-free counts
    python scripts/ot1_extension.py deribit    # extra Deribit windows (SOL via USDC; none needed for BTC/ETH dev days)
    python scripts/ot1_extension.py implied    # options-implied P(YES) per (market, decision time)
    python scripts/ot1_extension.py candles    # hourly candles for tail candidates
    python scripts/ot1_extension.py frame      # edgelab frames -> results/ot1_extension/
Read-only GETs only. No orders.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import options_implied_tails as O  # noqa: E402
from kx_throttle import kget  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "ot1_extension"
R.mkdir(parents=True, exist_ok=True)
MK = R / "mkts"
MK.mkdir(exist_ok=True)
ET = "America/New_York"
DEV_END = pd.Timestamp("2026-04-10")
DAILY = ["KXBTC", "KXETH", "KXSOLD", "KXSOLE", "KXBTCD", "KXETHD"]
CUR = {"KXBTC": "BTC", "KXBTCD": "BTC", "KXETH": "ETH", "KXETHD": "ETH", "KXSOLD": "SOL", "KXSOLE": "SOL"}
KEEP = ("ticker", "event_ticker", "strike_type", "floor_strike", "cap_strike", "open_time", "close_time", "result",
        "status")


def _tag(d):
    return d.strftime("%y") + d.strftime("%b").upper() + d.strftime("%d")


def event_markets(job):
    s, d = job
    tag = _tag(d)
    if s in ("KXBTCD", "KXETHD"):  # reuse OT1's cache when present
        f0 = O.D / "mkts" / f"{s}_{tag}.json"
        if f0.exists():
            return s, d, json.loads(f0.read_text())
    f = MK / f"{s}_{tag}.json"
    if f.exists():
        return s, d, json.loads(f.read_text())
    try:
        ms = (kget("/historical/markets", {"event_ticker": f"{s}-{tag}17", "limit": 1000}) or {}).get("markets", [])
    except Exception as e:  # noqa: BLE001
        print("ERR", s, tag, e, flush=True)
        return s, d, []
    ms = [{k: m.get(k) for k in KEEP} for m in ms]
    f.write_text(json.dumps(ms))
    return s, d, ms


def cmd_census():
    jobs = [(s, d) for s in DAILY for d in pd.date_range("2024-10-24", DEV_END)]
    print(len(jobs), "event lookups", flush=True)
    rows = []
    with ThreadPoolExecutor(4) as ex:
        for s, d, ms in ex.map(event_markets, jobs):
            for m in ms:
                ct = pd.Timestamp(m["close_time"]).tz_convert(ET)
                if not (ct.hour == 17 and ct.minute == 0):
                    continue
                rows.append({"series": s, "cur": CUR[s], "ticker": m["ticker"], "event_ticker": m["event_ticker"],
                             "date": pd.Timestamp(ct.date()), "strike_type": m.get("strike_type"),
                             "floor": O._num(m.get("floor_strike")), "cap": O._num(m.get("cap_strike")),
                             "open_ts": int(pd.Timestamp(m["open_time"]).timestamp()),
                             "close_ts": int(pd.Timestamp(m["close_time"]).timestamp()),
                             "result": m.get("result"), "status": m.get("status")})
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df = df[df.date <= DEV_END]
    df.to_parquet(R / "markets.parquet")
    ev = df.groupby("event_ticker").agg(series=("series", "first"), date=("date", "first"), n=("ticker", "size"),
                                         open_ts=("open_ts", "min"), close_ts=("close_ts", "first"))
    ev["listed_h"] = (ev.close_ts - ev.open_ts) / 3600
    ev["weekly"] = ev.listed_h > 48
    print(ev.groupby(["series", "weekly"]).agg(ev=("n", "size"), mk=("n", "sum"), d0=("date", "min"),
                                                d1=("date", "max"), med_listed_h=("listed_h", "median")))


# ------------------------------------------------------------------ decisions
def decisions(m: pd.DataFrame) -> pd.DataFrame:
    """(event, decision T) pairs per prereg: X1/X3 T = 11:00 ET close date; X2 = 11:00 ET on each of the 4 days
    before a weekly-listed event's close date (only while the event is open at T - 1 h)."""
    ev = m.groupby("event_ticker").agg(series=("series", "first"), cur=("cur", "first"), date=("date", "first"),
                                        open_ts=("open_ts", "min"), close_ts=("close_ts", "first")).reset_index()
    out = []
    for r in ev.itertuples():
        T0 = O.decision_ts(r.date)
        if r.series in ("KXBTC", "KXETH", "KXSOLD", "KXSOLE"):
            out.append((r.event_ticker, "same_day", T0, r.date))
        if (r.close_ts - r.open_ts) > 48 * 3600 and r.cur in ("BTC", "ETH"):
            for k in (4, 3, 2, 1):
                dd = r.date - pd.Timedelta(days=k)
                T = O.decision_ts(dd)
                if T - 3600 >= r.open_ts:
                    out.append((r.event_ticker, f"ahead{k}", T, dd))
    return pd.DataFrame(out, columns=["event_ticker", "arm", "T", "ddate"])


def cmd_deribit():
    """SOL windows (USDC options, SOL_USDC-* only) for days that need them; BTC/ETH reuse OT1's cache."""
    m = pd.read_parquet(R / "markets.parquet")
    dz = decisions(m).merge(m.groupby("event_ticker").cur.first().reset_index(), on="event_ticker")
    cache = R / "deribit"
    cache.mkdir(exist_ok=True)
    need = sorted({(c, int(T), str(pd.Timestamp(dd).date())) for c, T, dd in zip(dz.cur, dz["T"], dz.ddate)})
    jobs = []
    for c, T, ds in need:
        if c in ("BTC", "ETH") and (O.D / "deribit" / f"{c}_{ds}.json").exists():
            continue
        f = cache / f"{c}_{ds}.json"
        if not f.exists():
            jobs.append((c, T, f))
    print(len(jobs), "Deribit windows to fetch", flush=True)

    def go(j):
        c, T, f = j
        tr = O.deribit_window("USDC" if c == "SOL" else c, (T - 3600) * 1000, T * 1000)
        if c == "SOL":
            tr = [x for x in tr if x["instrument_name"].startswith("SOL_USDC-")]
            for x in tr:  # SOL_USDC-13JUN25-150-C -> SOL-13JUN25-150-C so O._expiry / O.smile parse it
                x["instrument_name"] = "SOL" + x["instrument_name"][len("SOL_USDC"):]
                # linear USDC options: mark_price is in USD already; O.smile multiplies by index -> pre-divide
                x["mark_price"] = float(x["mark_price"]) / float(x["index_price"])
        f.write_text(json.dumps([{k: x.get(k) for k in ("timestamp", "instrument_name", "price", "mark_price", "iv",
                                                         "index_price", "amount")} for x in tr]))
        return len(tr)

    with ThreadPoolExecutor(3) as ex:
        for i, n in enumerate(ex.map(go, jobs)):
            if i % 100 == 0:
                print(i, n, flush=True)


def _deribit_file(cur, ds):
    f = O.D / "deribit" / f"{cur}_{ds}.json"
    return f if (cur in ("BTC", "ETH") and f.exists()) else R / "deribit" / f"{cur}_{ds}.json"


def cmd_implied():
    m = pd.read_parquet(R / "markets.parquet")
    dz = decisions(m)
    rows = []
    mg = {k: g for k, g in m.groupby("event_ticker")}
    for r in dz.itertuples():
        g = mg[r.event_ticker]
        cur = g.cur.iloc[0]
        ds = str(pd.Timestamp(r.ddate).date())
        f = _deribit_file(cur, ds)
        close = int(g.close_ts.iloc[0])
        sm = O.smile(json.loads(f.read_text()), close, int(r.T)) if f.exists() else None
        tauK = (close - r.T) / (365 * 86400)
        for q in g.itertuples():
            row = q._asdict()
            p = O.p_yes(row, lambda k: O.p_above_crypto(k, sm, tauK)) if sm is not None else np.nan
            rows.append({"ticker": q.ticker, "arm": r.arm, "T": int(r.T), "ddate": pd.Timestamp(r.ddate),
                         "p_yes": p, "spot": sm[0] if sm else np.nan, "tauD_h": sm[1] * 8760 if sm else np.nan,
                         "inputs_as_of": int(r.T) - 1})
    out = pd.DataFrame(rows)
    out.to_parquet(R / "implied.parquet")
    x = out.merge(m[["ticker", "series"]], on="ticker")
    print(x.groupby(["series", "arm"]).agg(n=("p_yes", "size"), ok=("p_yes", lambda s: s.notna().sum())))


def _cands():
    m = pd.read_parquet(R / "markets.parquet")
    imp = pd.read_parquet(R / "implied.parquet")
    x = m.merge(imp, on="ticker")
    parts = []
    for (arm, T), g in x.groupby(["arm", "T"]):
        c = O.tail_candidates(g.reset_index(drop=True))  # amendment-2 rule, per (event, decision)
        parts.append(c)
    return pd.concat(parts, ignore_index=True)


def cmd_candles():
    c = _cands()
    u = c.drop_duplicates("ticker")
    cache = R / "candles"
    cache.mkdir(exist_ok=True)
    todo = [r for r in u.itertuples() if not (cache / f"{r.ticker}.json").exists()
            and not (O.D / "candles" / f"{r.ticker}.json").exists()]
    print(len(c), "candidate rows;", len(u), "markets;", len(todo), "to fetch", flush=True)

    def go(r):
        d = kget(f"/historical/markets/{r.ticker}/candlesticks",
                 {"start_ts": int(r.open_ts), "end_ts": int(r.close_ts), "period_interval": 60}) or {}
        (cache / f"{r.ticker}.json").write_text(json.dumps(d.get("candlesticks", [])))

    with ThreadPoolExecutor(4) as ex:
        for i, _ in enumerate(ex.map(go, todo)):
            if i % 500 == 0:
                print(i, flush=True)


def cmd_frame():
    c = _cands()
    rows = []
    for r in c.itertuples():
        f = R / "candles" / f"{r.ticker}.json"
        if not f.exists():
            f = O.D / "candles" / f"{r.ticker}.json"
        cs = json.loads(f.read_text()) if f.exists() else []
        pre = [x for x in cs if x["end_period_ts"] <= r.T]
        mk = [x for x in cs if x["end_period_ts"] <= r.close_ts - 3600]
        rows.append({"yes_bid": O._px(pre[-1], "yes_bid", "close") if pre else np.nan,
                     "yes_ask": O._px(pre[-1], "yes_ask", "close") if pre else np.nan,
                     "quote_ts": pre[-1]["end_period_ts"] if pre else -1,
                     "mark_yes_bid": O._px(mk[-1], "yes_bid", "close") if mk else np.nan,
                     "mark_yes_ask": O._px(mk[-1], "yes_ask", "close") if mk else np.nan})
    fr = pd.concat([c.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    fr["asset_class"] = "crypto"
    fr["date"] = pd.to_datetime(fr["date"])
    fr.to_parquet(R / "frame.parquet")
    print(len(fr), "frame rows;", fr.yes_bid.notna().sum(), "quoted")
    print(fr.groupby(["series", "arm"]).agg(n=("ticker", "size"), q=("yes_bid", lambda s: s.notna().sum())))


if __name__ == "__main__":
    {"census": cmd_census, "deribit": cmd_deribit, "implied": cmd_implied, "candles": cmd_candles,
     "frame": cmd_frame}[sys.argv[1]]()
