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


# ================================================================ v2: "HPC for AI agents" (user thesis update 2026-10-22)
# Report: notes/thesis_v2.md. Four sleeves: MEMORY, COMPUTE-INFERENCE, ENERGY-GEN, ENERGY-EQUIP. Networking/connectivity
# and training-specific names deliberately left out (user: de-emphasize). v1 above is untouched (its sha must stay b745e0d3aa0e).
# Rules v2 (frozen with the lists, before any v2 return was computed):
#   * equal weight, rebalanced MONTHLY on the first trading day of each month (weights drift in between);
#   * a name enters at the first monthly rebalance on/after max(first priced day, ENTRY_V2 date) -- ENTRY_V2 skips the
#     pre-merger SPAC shell prices yfinance carries for OKLO / SMR / VRT;
#   * a name with a missing price mid-month is carried at its last price (no look-ahead);
#   * benchmarks SPY, QQQ, SMH; history windows 2019-01-02 and 2023-01-03; forward record from FREEZE_V2.
#   * "v2_all" = the four sleeves at 25% each, rebalanced monthly (sleeve-level, so it is NOT equal weight by name).
FREEZE_V2 = "2026-10-22"
BASE_DATES_V2 = ["2019-01-02", "2023-01-03"]
BENCH_V2 = ["SPY", "QQQ", "SMH"]
LOG_V2 = OUT / "tracker_v2.jsonl"
BASKETS_V2 = {
    # MEMORY: HBM/DRAM/NAND makers, HDD bulk tier, enterprise storage systems, memory-interface/SSD-controller chips
    "memory_v2": ["MU", "SNDK", "WDC", "STX", "P", "NTAP", "SIMO", "RMBS"],
    # COMPUTE-INFERENCE: server CPUs, custom inference ASIC designers, inference chips, foundry + advanced packaging
    "compute_inf_v2": ["AMD", "INTC", "ARM", "AVGO", "MRVL", "QCOM", "CBRS", "TSM", "AMKR", "ASX"],
    # ENERGY-GEN: IPPs / nuclear fleet owners, utilities with large datacenter load pipelines, SMR developers, uranium
    "energy_gen_v2": ["CEG", "VST", "TLN", "NRG", "D", "AEP", "SO", "ETR", "OKLO", "SMR", "CCJ"],
    # ENERGY-EQUIP: gas turbines + grid, power/cooling infrastructure, fuel cells, switchgear, grid construction, nuclear components
    "energy_equip_v2": ["GEV", "VRT", "ETN", "BE", "PWR", "HUBB", "MOD", "NVT", "BWXT", "POWL"],
}
ENTRY_V2 = {"OKLO": "2024-05-10", "SMR": "2022-05-03", "VRT": "2020-02-10"}  # de-SPAC dates (operating company starts)


def rules_blob_v2() -> str:
    return json.dumps({"freeze": FREEZE_V2, "baskets": BASKETS_V2, "entry": ENTRY_V2, "base_dates": BASE_DATES_V2,
                       "bench": BENCH_V2, "weighting": "equal by name within sleeve, monthly rebalance on first trading "
                       "day of month, enter at first rebalance after first price/entry date; v2_all = 25% per sleeve "
                       "monthly"}, sort_keys=True)


def rules_sha_v2() -> str:
    return hashlib.sha256(rules_blob_v2().encode()).hexdigest()


def _rebalance_days(idx: pd.DatetimeIndex, start) -> list:
    idx = idx[idx >= pd.Timestamp(start)]
    if len(idx) == 0:
        return []
    m = idx.to_period("M")
    firsts = [idx[0]] + [d for d, prev, cur in zip(idx[1:], m[:-1], m[1:]) if cur != prev]
    return firsts


def monthly_rebalanced(close: pd.DataFrame, weights_fn, start) -> tuple[pd.Series, pd.DataFrame]:
    """Daily returns of a portfolio reset to weights_fn(rebalance_date) on the first trading day of each month.
    Returns (daily return series, per-name daily dollar P&L on a $1 start) -- P&L sums to the equity gain, so
    contribution shares are exact."""
    px = close.ffill()
    days = _rebalance_days(px.index, start)
    out_r, out_c = [], []
    v = 1.0
    for i, d0 in enumerate(days):
        d1 = days[i + 1] if i + 1 < len(days) else None
        w = weights_fn(d0)  # Series name -> weight, sums to 1, only names priced on d0
        seg = px.loc[d0:d1, w.index]  # d1 (next rebalance day) included as this segment's last close
        rel = seg / seg.iloc[0]
        val = (rel * w).sum(axis=1)                       # value of $1 placed at d0
        pos = rel.mul(w, axis=1)                          # per-name value
        dpos = pos.diff().iloc[1:] * v                    # per-name dollar P&L, scaled by equity at d0
        r = (val / val.shift(1) - 1).iloc[1:]
        out_r.append(r); out_c.append(dpos)
        v *= float(val.iloc[-1])
    r = pd.concat(out_r)
    c = pd.concat(out_c).fillna(0.0)
    return r[~r.index.duplicated()], c.groupby(level=0).sum()


def sleeve_weights(close: pd.DataFrame, names):
    first = {t: max(close[t].first_valid_index(), pd.Timestamp(ENTRY_V2.get(t, "1900-01-01"))) for t in names}

    def fn(d0):
        live = [t for t in names if first[t] < d0 and pd.notna(close[t].ffill().loc[d0])]
        return pd.Series(1.0 / len(live), index=live)
    return fn


def v2_series(close: pd.DataFrame, start) -> tuple[pd.DataFrame, dict]:
    rets, contrib = {}, {}
    for k, names in BASKETS_V2.items():
        r, c = monthly_rebalanced(close, sleeve_weights(close, names), start)
        rets[k], contrib[k] = r, c
    df = pd.DataFrame(rets)
    # v2_all: 25% per sleeve, monthly -- built from sleeve "prices"
    lv = (1 + df.fillna(0)).cumprod()
    lv.loc[lv.index[0] - pd.Timedelta(days=1)] = 1.0
    lv = lv.sort_index()
    r_all, _ = monthly_rebalanced(lv, lambda d0: pd.Series(0.25, index=list(BASKETS_V2)), lv.index[0])
    df["v2_all"] = r_all
    for b in BENCH_V2:
        df[b] = close[b].pct_change(fill_method=None)
    return df.dropna(subset=list(BASKETS_V2)), contrib


def cmd_rules_v2():
    print(rules_blob_v2())
    print("sha256", rules_sha_v2())


def cmd_history_v2():
    names = sorted({t for v in BASKETS_V2.values() for t in v} | set(BENCH_V2))
    close = fetch(names, "2018-12-01")
    close = close[close.index <= pd.Timestamp(FREEZE_V2) - pd.Timedelta(days=1)]  # through the day before freeze
    print(f"prices through {close.index.max().date()}  v2 rules sha {rules_sha_v2()[:12]}  (HINDSIGHT-BIASED description)")
    res = {}
    for b in BASE_DATES_V2:
        df, contrib = v2_series(close, b)
        df = df[df.index > pd.Timestamp(b)]
        rows = []
        for col in df.columns:
            s = summarize(df[col])
            rows.append({"series": col, "total": s["total"], "cagr": s["cagr"], "vol": s["vol"], "maxdd": s["maxdd"],
                         "corr_SMH": df[col].corr(df["SMH"]), "corr_SPY": df[col].corr(df["SPY"])})
        print(f"\n=== since {b}")
        print(pd.DataFrame(rows).set_index("series").round(3).to_string())
        print("\ndaily-return correlation:")
        print(df[list(BASKETS_V2) + ["SMH", "QQQ", "SPY"]].corr().round(2).to_string())
        print("\nconcentration (exact $-P&L attribution on $1, monthly rebalance):")
        crow = []
        for k, c in contrib.items():
            c = c[c.index > pd.Timestamp(b)]
            tot = c.sum().sort_values(ascending=False)
            gain = tot.sum()
            top2 = tot.head(2)
            # ex-top-2 counterfactual: same rules without those two names
            rest = [t for t in BASKETS_V2[k] if t not in top2.index]
            r_ex, _ = monthly_rebalanced(close, sleeve_weights(close, rest), b)
            r_ex = r_ex[r_ex.index > pd.Timestamp(b)]
            crow.append({"sleeve": k, "gain_on_$1": round(gain, 3), "top2": ",".join(top2.index),
                         "top2_share_of_gain": round(top2.sum() / gain, 3) if gain else np.nan,
                         "sleeve_total": round(summarize(df[k])["total"], 3),
                         "ex_top2_total": round(summarize(r_ex)["total"], 3),
                         "per_name": " ".join(f"{t}:{v:+.2f}" for t, v in tot.items())})
        cdf = pd.DataFrame(crow)
        print(cdf.drop(columns="per_name").to_string(index=False))
        for _, x in cdf.iterrows():
            print(f"  {x['sleeve']}: {x['per_name']}")
        res[b] = {"table": rows, "concentration": crow}
        df.to_parquet(OUT / f"history_v2_since_{b}.parquet")
    (OUT / "history_v2.json").write_text(json.dumps(res, default=float, indent=1))


def tick_v2():
    names = sorted({t for v in BASKETS_V2.values() for t in v} | set(BENCH_V2))
    close = fetch(names, "2018-12-01")  # long history so entry dates resolve; only post-freeze returns are logged
    from zoneinfo import ZoneInfo
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    if now_ny.hour * 60 + now_ny.minute < 16 * 60 + 15:
        close = close[close.index.date < now_ny.date()]
    if (close.index >= pd.Timestamp(FREEZE_V2)).sum() < 2:
        print("v2: no full trading day since freeze yet"); return
    df, _ = v2_series(close, FREEZE_V2)
    df = df[df.index > pd.Timestamp(FREEZE_V2)]
    lv = (1 + df.fillna(0)).cumprod()
    last = lv.index.max()
    row = {"logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "date": str(last.date()),
           "rules_sha_v2": rules_sha_v2()[:12], **{k: round(float(lv.loc[last, k]), 6) for k in lv.columns}}
    seen = {json.loads(x)["date"] for x in LOG_V2.read_text().splitlines() if x.strip()} if LOG_V2.exists() else set()
    if row["date"] in seen:
        print(f"v2: {row['date']} already logged"); return
    with LOG_V2.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


def report_v2():
    if not LOG_V2.exists():
        print("v2: no forward log yet; run tick"); return
    d = pd.DataFrame([json.loads(x) for x in LOG_V2.read_text().splitlines() if x.strip()])
    print(d.tail(15).to_string(index=False))
    last = d.iloc[-1]
    cols = list(BASKETS_V2) + ["v2_all"] + BENCH_V2
    print(f"\nv2 forward since {FREEZE_V2}: " + "  ".join(f"{c} {last[c]-1:+.2%}" for c in cols) +
          f"  ({len(d)} logged days; judge on the leading indicators in notes/thesis_v2.md, not weeks of prices)")


def cmd_tick_all():
    cmd_tick()   # v1, unchanged log + schema
    tick_v2()


def cmd_report_all():
    cmd_report()
    print()
    report_v2()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if not BASKETS["inference_v1"] or not BASKETS["training_v1"]:
        sys.exit("baskets not frozen yet")
    {"rules": cmd_rules, "history": cmd_history, "tick": cmd_tick_all, "report": cmd_report_all,
     "rules-v2": cmd_rules_v2, "history-v2": cmd_history_v2}[cmd]()
