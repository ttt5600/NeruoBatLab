"""Q009 step 1: download SPY EOD option chains (bid/ask) from DoltHub post-no-preference/options.

One SQL query per calendar weekday (date-range scans time out on the public API; equality on `date`
is indexed). <= ~1.5 req/s. Writes one parquet per month to data/options/dolthub/ (gitignored).
Rows are untrusted data: parsed as JSON -> pandas only, nothing executed.

  ../.venv_quant/bin/python -I scripts/q009_fetch_dolthub.py --start 2019-09-01 --end 2026-10-30
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
OUT = Path(__file__).resolve().parents[1] / "data" / "options" / "dolthub"


def fetch(day: str, sym: str, tries: int = 4):
    q = f"SELECT * FROM option_chain WHERE date='{day}' AND act_symbol='{sym}'"
    for k in range(tries):
        try:
            r = requests.get(API, params={"q": q}, timeout=150)
            d = json.loads(r.text)
            if d.get("query_execution_status") == "Success":
                return d["rows"]
        except Exception:
            pass
        time.sleep(2 + 3 * k)
    return None


def first_snapshot_of_week(monday, sym):
    """Try Mon..Fri in order; return (date, rows, log) for the first non-empty day (the prereg entry day)."""
    log = []
    for k in range(5):
        ds = (monday + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        got = fetch(ds, sym)
        log.append({"date": ds, "n": -1 if got is None else len(got)})
        if got:
            return ds, got, log
    return None, [], log


def main_weekly(a):
    from concurrent.futures import ThreadPoolExecutor
    mondays = pd.date_range(a.start, a.end, freq="W-MON")
    todo = [m for m in mondays if not (OUT / f"{a.sym}_wk_{m:%Y-%m-%d}.parquet").exists()]
    logf = OUT / f"fetch_log_weekly_{a.sym}.csv"
    with ThreadPoolExecutor(max_workers=2) as ex:
        for m, (ds, rows, log) in zip(todo, ex.map(lambda m: first_snapshot_of_week(m, a.sym), todo)):
            pd.DataFrame(rows).to_parquet(OUT / f"{a.sym}_wk_{m:%Y-%m-%d}.parquet", index=False)
            lg = pd.DataFrame(log).assign(week=f"{m:%Y-%m-%d}")
            lg.to_csv(logf, mode="a", header=not logf.exists(), index=False)
            print(f"{m:%Y-%m-%d}", ds, len(rows), flush=True)


def fix_holidays(a):
    """Amendment (pre-P&L, notes/q009_options_vrp.md): a snapshot dated on an NYSE holiday (not a Yahoo SPY trading
    day) is a stale copy and not tradeable; replace it with the next non-empty TRADING-day snapshot of that week."""
    px = pd.read_parquet(OUT.parent / "yahoo_spy" / "spy_daily.parquet")
    tdays = set(pd.to_datetime(px.index).strftime("%Y-%m-%d"))
    logf = OUT / f"fetch_log_weekly_{a.sym}.csv"
    lg = pd.read_csv(logf)
    first = lg[lg.n > 0].groupby("week").date.min()
    for wk, ds in first.items():
        if ds in tdays:
            continue
        d0 = pd.Timestamp(ds)
        new = None
        for k in range(1, 5):
            d = d0 + pd.Timedelta(days=k)
            if d.weekday() > 4:
                break
            dss = d.strftime("%Y-%m-%d")
            if dss not in tdays:
                continue
            got = fetch(dss, a.sym)
            pd.DataFrame([{"date": dss, "n": -1 if got is None else len(got), "week": wk}]).to_csv(
                logf, mode="a", header=False, index=False)
            if got:
                new = (dss, got)
                break
        hol = OUT / f"holiday_{a.sym}_{ds}.parquet"
        (OUT / f"{a.sym}_wk_{wk}.parquet").rename(hol)
        if new:
            pd.DataFrame(new[1]).to_parquet(OUT / f"{a.sym}_wk_{wk}.parquet", index=False)
        else:
            pd.DataFrame().to_parquet(OUT / f"{a.sym}_wk_{wk}.parquet", index=False)
        pd.DataFrame([{"date": ds, "n": 0, "week": wk, "note": "holiday"}]).to_csv(OUT / "holiday_log.csv", mode="a",
                     header=not (OUT / "holiday_log.csv").exists(), index=False)
        print("holiday", wk, ds, "->", new[0] if new else None, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sym", default="SPY")
    ap.add_argument("--weekly", action="store_true", help="only the first snapshot of each ISO week (prereg entry day)")
    ap.add_argument("--fix-holidays", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.weekly:
        return main_weekly(a)
    if a.fix_holidays:
        return fix_holidays(a)
    days = pd.bdate_range(a.start, a.end)
    log = []
    for month, grp in pd.Series(days).groupby(days.to_period("M")):
        f = OUT / f"{a.sym}_{month}.parquet"
        if f.exists():
            continue
        rows = []
        for day in grp:
            ds = day.strftime("%Y-%m-%d")
            got = fetch(ds, a.sym)
            log.append({"date": ds, "n": -1 if got is None else len(got)})
            if got:
                rows.extend(got)
            time.sleep(0.7)
        df = pd.DataFrame(rows)
        df.to_parquet(f, index=False)
        print(month, len(df), flush=True)
        pd.DataFrame(log).to_csv(OUT / f"fetch_log_{a.sym}.csv", index=False)


if __name__ == "__main__":
    main()
