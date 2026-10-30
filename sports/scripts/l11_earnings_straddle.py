"""L11 earnings run-up straddle (pre-registered in notes/l11_earnings_straddle.md).

  ../.venv_quant/bin/python -I scripts/l11_earnings_straddle.py events      # yfinance + EDGAR -> event table (no prices)
  ../.venv_quant/bin/python -I scripts/l11_earnings_straddle.py fetch dev   # DoltHub chains for dev snapshot dates only
  ../.venv_quant/bin/python -I scripts/l11_earnings_straddle.py dev [--ledger]
  ../.venv_quant/bin/python -I scripts/l11_earnings_straddle.py fetch sealed; ... sealed [--ledger]   # only on a dev pass
"""
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "l11"
CHAINS = DATA / "chains"
RES = ROOT / "results" / "l11_earnings_straddle"
LEDGER = ROOT / "lab" / "ledger.jsonl"
API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"

UNIVERSE = ("AAPL MSFT NVDA AMZN META GOOGL TSLA NFLX AMD JPM BAC WMT DIS INTC BA CSCO ORCL CRM ADBE PYPL UBER "
            "COST PFE KO MU QCOM V GS").split()
START, END = "2020-01-06", "2026-10-28"     # DoltHub single-name coverage window (probed)
DEV_FRAC = 0.70
FEE = 0.03 / 100                            # $ per share-equivalent per contract per side
NBOOT = 10_000
K_TRIALS = 2
BAR = 0.05 / K_TRIALS


# ------------------------------------------------------------------------------------------------ calendar
def trading_days():
    """NYSE trading days from Yahoo SPY daily bars (reuses the q009 file; extended if needed)."""
    f = ROOT / "data" / "options" / "yahoo_spy" / "spy_daily.parquet"
    px = pd.read_parquet(f)
    return pd.DatetimeIndex(pd.to_datetime(px.index).tz_localize(None).normalize()).unique().sort_values()


# ------------------------------------------------------------------------------------------------ events
def edgar_202():
    """8-K item 2.02 acceptance times straight from data.sec.gov submissions JSON (raw acceptanceDateTime is true
    UTC: AAPL 20:30Z = 16:30 ET, JPM 10:30Z = 06:30 ET; the company_graph copy is shifted, so it is not reused)."""
    import requests
    f = DATA / "edgar_202.parquet"
    if f.exists():
        return pd.read_parquet(f)
    H = {"User-Agent": "quant-research-bot admin@quantresearch.local"}
    tick = json.loads((ROOT / "data" / "company_graph" / "company_tickers_2026-10-22.json").read_text())
    t2c = {v["ticker"]: int(v["cik_str"]) for v in tick.values()}
    rows = []
    for t in UNIVERSE:
        j = requests.get(f"https://data.sec.gov/submissions/CIK{t2c[t]:010d}.json", headers=H, timeout=30).json()
        time.sleep(0.5)
        blocks = [j["filings"]["recent"]]
        for ff in j["filings"].get("files", []):
            if ff.get("filingTo", "9999") < "2019-06-01":
                continue
            blocks.append(requests.get(f"https://data.sec.gov/submissions/{ff['name']}", headers=H, timeout=30).json())
            time.sleep(0.5)
        for b in blocks:
            for form, acc, items in zip(b["form"], b["acceptanceDateTime"], b["items"]):
                if form == "8-K" and "2.02" in str(items):
                    rows.append((t, acc))
    e = pd.DataFrame(rows, columns=["ticker", "accepted"]).drop_duplicates()
    e["ed_ts"] = pd.to_datetime(e.accepted, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    e.to_parquet(f)
    return e


def cmd_events():
    import yfinance as yf
    DATA.mkdir(parents=True, exist_ok=True)
    yf_rows = []
    for t in ([] if (DATA / "yf_earnings_dates.parquet").exists() else UNIVERSE):
        d = yf.Ticker(t).get_earnings_dates(limit=60)
        time.sleep(0.6)
        d = d.reset_index()
        d.columns = ["ts", "eps_est", "eps_rep", "surprise"][: len(d.columns)] if len(d.columns) == 4 else d.columns
        for _, r in d.iterrows():
            ts = pd.Timestamp(r["ts"]).tz_convert("America/New_York")
            yf_rows.append({"ticker": t, "yf_ts": ts.tz_localize(None), "eps_rep": r.get("eps_rep")})
    if yf_rows:
        pd.DataFrame(yf_rows).to_parquet(DATA / "yf_earnings_dates.parquet")
    y = pd.read_parquet(DATA / "yf_earnings_dates.parquet")

    e = edgar_202()

    td = trading_days()
    rows = []
    y = y[(y.yf_ts >= START) & (y.yf_ts <= END) & y.eps_rep.notna()]
    for _, r in y.iterrows():
        cand = e[(e.ticker == r.ticker)]
        dd = (cand.ed_ts.dt.normalize() - r.yf_ts.normalize()).dt.days
        ok = cand[(dd >= -3) & (dd <= 3)]
        ed = ok.ed_ts.min() if len(ok) else pd.NaT
        rows.append({"ticker": r.ticker, "yf_ts": r.yf_ts, "edgar_ts": ed})
    ev = pd.DataFrame(rows)

    def amc(ts):
        return ts.hour >= 16

    def bmo(ts):
        return ts.hour < 9 or (ts.hour == 9 and ts.minute < 30)

    out = []
    for _, r in ev.iterrows():
        rec = r.to_dict()
        if pd.isna(r.edgar_ts):
            rec["dropped"] = "no_edgar_match"
            out.append(rec)
            continue
        # latest allowed exit close date = the last close strictly before the earliest possible release.
        lim = []
        for ts, src in ((r.yf_ts, "yf"), (r.edgar_ts, "edgar")):
            dday = ts.normalize()
            if src == "yf" and ts.hour == 0 and ts.minute == 0:     # no time given -> unknown -> treat as BMO
                lim.append(dday - pd.Timedelta(days=1))
            elif amc(ts):
                lim.append(dday)                                    # release after D's close -> D's close is OK
            else:
                lim.append(dday - pd.Timedelta(days=1))             # BMO or intraday -> must exit by D-1 close
        xmax = min(lim)
        tmax = td[td <= xmax]
        if not len(tmax):
            rec["dropped"] = "calendar"
            out.append(rec)
            continue
        t0 = tmax[-1]                                               # last trading close before the release
        i0 = td.get_loc(t0)
        e_session = td[i0 + 1] if i0 + 1 < len(td) else t0 + pd.Timedelta(days=1)  # first session after release
        rec.update(t0=t0, e_session=e_session, timing="AMC" if t0 == min(r.yf_ts, r.edgar_ts).normalize() else "BMO",
                   entry_max=td[i0 + 1 - 5],                         # <= E - 5 trading days
                   disagree_days=abs((r.yf_ts.normalize() - r.edgar_ts.normalize()).days), dropped="")
        out.append(rec)
    ev = pd.DataFrame(out).sort_values(["e_session", "ticker"]).reset_index(drop=True)
    ev.to_parquet(DATA / "events.parquet")
    print(ev.dropped.value_counts().to_dict())
    k = ev[ev.dropped == ""]
    print("usable events", len(k), "tickers", k.ticker.nunique(), k.e_session.min(), k.e_session.max())
    print("timing", k.timing.value_counts().to_dict(), "yf/edgar date disagree>0:", int((k.disagree_days > 0).sum()))
    print(k.groupby("ticker").size().to_dict())


if __name__ == "__main__":
    globals()["cmd_" + sys.argv[1]](*sys.argv[2:3])
