"""Inference-vs-training thesis tracker (user thesis 2026-10-22; report: notes/inference_thesis.md).

    python scripts/thesis_tracker.py rules            # print the frozen basket rules + their sha256
    python scripts/thesis_tracker.py history          # realized relative performance since each stated start date
    python scripts/thesis_tracker.py tick             # append today's basket levels to data/thesis/tracker.jsonl
    python scripts/thesis_tracker.py report           # forward record since the freeze date (2026-10-22)

The baskets below were written from the mechanism map in notes/inference_thesis.md (step 2) BEFORE any
basket return was computed. Do not edit them after the fact: if a basket must change, add a NEW basket
version with its own freeze date and keep the old one in the log, so the forward record stays honest.

Rules (fixed with the baskets):
  * US listings only (ADRs ok), prices from yfinance auto_adjust=True (split/dividend adjusted), USD.
  * Equal weight, rebalanced daily (so one name cannot swamp the basket after a big run).
    A name enters on its first full trading day; until then the basket is equal weight over names
    that exist. This is disclosed survivorship/selection bias: the baskets were chosen in 2026 by
    someone who knows which companies became AI names, so the HISTORY is a description, not a backtest.
  * Spread = daily return(inference) - daily return(training), compounded; also vs SMH and SPY.
  * The forward record starts at FREEZE; only returns after FREEZE are out-of-sample.
This is research, not personalized financial advice.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "thesis"
LOG = OUT / "tracker.jsonl"
FREEZE = "2026-10-22"
BASE_DATES = ["2023-01-03", "2025-01-02", "2026-01-02"]  # stated before looking: post-ChatGPT, 2025, YTD
BENCH = ["SMH", "SPY"]

# ---- v1 baskets, frozen 2026-10-22 from the step-2 mechanism map (labels + reasons in notes/inference_thesis.md)
BASKETS = {
    "inference_v1": [],   # filled below
    "training_v1": [],
}
# (see bottom of file for the frozen lists; kept in one place so the sha covers them)


def rules_blob() -> str:
    return json.dumps({"freeze": FREEZE, "baskets": BASKETS, "base_dates": BASE_DATES, "bench": BENCH,
                       "weighting": "equal, daily rebalance, enter on first trading day"}, sort_keys=True)


def rules_sha() -> str:
    return hashlib.sha256(rules_blob().encode()).hexdigest()


def fetch(tickers, start):
    import yfinance as yf
    yf.set_tz_cache_location(os.environ.get("YF_CACHE", str(OUT / ".yfcache")))
    OUT.mkdir(parents=True, exist_ok=True)
    d = yf.download(sorted(set(tickers)), start=start, auto_adjust=True, progress=False, threads=False)
    c = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]].rename(columns={"Close": tickers[0]})
    c = c.dropna(how="all")
    missing = [t for t in tickers if t not in c.columns or c[t].notna().sum() == 0]
    if missing:
        sys.exit(f"no prices for {missing}; refusing to log a partial basket")
    return c


def basket_returns(close: pd.DataFrame, names) -> pd.Series:
    r = close[names].pct_change(fill_method=None)
    # a name counts from its 2nd priced day (first full return); NaN otherwise -> excluded that day
    return r.mean(axis=1, skipna=True)


def summarize(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    return {"total": eq.iloc[-1] - 1, "cagr": eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan,
            "vol": r.std() * np.sqrt(252), "maxdd": (eq / eq.cummax() - 1).min(), "days": len(r)}


def cmd_rules():
    print(rules_blob())
    print("sha256", rules_sha())


def cmd_history():
    names = sorted({t for v in BASKETS.values() for t in v} | set(BENCH))
    close = fetch(names, BASE_DATES[0])
    rets = {k: basket_returns(close, v) for k, v in BASKETS.items()}
    rets.update({b: close[b].pct_change(fill_method=None) for b in BENCH})
    df = pd.DataFrame(rets).iloc[1:]
    inf, trn = list(BASKETS)
    df["spread"] = df[inf] - df[trn]
    print(f"prices through {close.index.max().date()}  rules sha {rules_sha()[:12]}")
    rows = []
    for b in BASE_DATES:
        d = df[df.index > b]
        for col in [inf, trn, "SMH", "SPY", "spread"]:
            s = summarize(d[col])
            rows.append({"since": b, "series": col, **{k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}})
        # how often the spread is positive by calendar month (descriptive only)
        m = (1 + d["spread"]).groupby(d.index.to_period("M")).prod() - 1
        rows.append({"since": b, "series": "spread_months_pos", "total": round((m > 0).mean(), 3), "days": len(m)})
        # correlation of the two baskets: if ~1, the 'tilt' is mostly the same bet
        rows.append({"since": b, "series": "corr_inf_trn", "total": round(d[inf].corr(d[trn]), 3), "days": len(d)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    # per-name contribution since the latest base date, so one name's run is visible
    b = BASE_DATES[-1]
    per = {}
    for k, v in BASKETS.items():
        for t in v:
            s = close[t][close.index >= b].dropna()
            per[(k, t)] = s.iloc[-1] / s.iloc[0] - 1 if len(s) > 1 else np.nan
    print(f"\nper-name total return since {b}:")
    print(pd.Series(per).round(3).to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "history_daily.parquet")


def cmd_tick():
    names = sorted({t for v in BASKETS.values() for t in v} | set(BENCH))
    close = fetch(names, FREEZE)
    close = close[close.index >= FREEZE]
    # never log an intraday print as a close: drop today's row until 16:15 New York time
    from zoneinfo import ZoneInfo
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    if now_ny.hour * 60 + now_ny.minute < 16 * 60 + 15:
        close = close[close.index.date < now_ny.date()]
    if len(close) < 1:
        print("no trading day since freeze yet"); return
    rets = {k: basket_returns(close, v) for k, v in BASKETS.items()}
    rets.update({b: close[b].pct_change(fill_method=None) for b in BENCH})
    lv = (1 + pd.DataFrame(rets).fillna(0)).cumprod()
    last = lv.index.max()
    row = {"logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "date": str(last.date()),
           "rules_sha": rules_sha()[:12], **{k: round(float(lv.loc[last, k]), 6) for k in lv.columns}}
    inf, trn = list(BASKETS)
    row["rel_inf_over_trn"] = round(row[inf] / row[trn], 6)
    seen = set()
    if LOG.exists():
        seen = {json.loads(x)["date"] for x in LOG.read_text().splitlines() if x.strip()}
    if row["date"] in seen:
        print(f"{row['date']} already logged"); return
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


def cmd_report():
    if not LOG.exists():
        print("no forward log yet; run tick"); return
    d = pd.DataFrame([json.loads(x) for x in LOG.read_text().splitlines() if x.strip()])
    print(d.tail(15).to_string(index=False))
    inf, trn = list(BASKETS)
    last = d.iloc[-1]
    print(f"\nforward since {FREEZE}: {inf} {last[inf]-1:+.2%}  {trn} {last[trn]-1:+.2%}  "
          f"rel {last['rel_inf_over_trn']-1:+.2%}  SMH {last['SMH']-1:+.2%}  SPY {last['SPY']-1:+.2%}  "
          f"({len(d)} logged days; weeks of data say nothing -- judge on the leading indicators, not this)")


# ---------------------------------------------------------------- FROZEN v1 LISTS (2026-10-22)
# inference_v1: US-listed names labelled "inference-levered" in step 2 (CBRS, SNDK, INTC, QCOM, NET) plus the
#   custom-ASIC / scale-up / Arm-CPU designers labelled "both, inference-tilted" (AVGO: TPU v8i + MTIA; MRVL: custom XPU;
#   ALAB: scale-up switches; ARM: agent-orchestration CPUs). Kioxia (285A.T) excluded: not US-listed.
# training_v1: the leg the user wants to trim -- "pure training GPUs and hyperscalers" -- NVDA + the four hyperscalers
#   + the scale-out optics/switching names labelled "both, training-tilted" (ANET, COHR, LITE).
BASKETS["inference_v1"] = ["CBRS", "SNDK", "INTC", "QCOM", "NET", "AVGO", "MRVL", "ALAB", "ARM"]
BASKETS["training_v1"] = ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "ANET", "COHR", "LITE"]

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if not BASKETS["inference_v1"] or not BASKETS["training_v1"]:
        sys.exit("baskets not frozen yet")
    {"rules": cmd_rules, "history": cmd_history, "tick": cmd_tick, "report": cmd_report}[cmd]()
