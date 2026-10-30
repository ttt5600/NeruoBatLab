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
            r = requests.get(API, params={"q": q}, timeout=60)
            d = json.loads(r.text)
            if d.get("query_execution_status") == "Success":
                return d["rows"]
        except Exception:
            pass
        time.sleep(2 + 3 * k)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sym", default="SPY")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
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
