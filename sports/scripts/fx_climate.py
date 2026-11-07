"""Builder for edgelab dataset fx_climate (trials FC1 fc1_gtm_nowcast, FC2 fc2_macd_nowcast; rule edgelab/rules/fx_climate.py).

    python scripts/fx_climate.py pull     # forecastex.com daily price files -> data/fx_climate/raw/ (GTM_/MACD_ rows only)
    python scripts/fx_climate.py build    # -> data/fx_climate/frame.parquet + days.parquet (+ inputs.json: c, sd fits)

Everything about the rule, the split and the pricing is in the rule module docstring and the specs; this script only
assembles inputs. It prints NO prices or outcomes (edgelab evaluates; the frame's `result` is hidden by the vault).
Inputs:
  ERA5 daily global 2 m anomaly (1991-2020): Copernicus Climate Pulse era5_daily_series_2t_global.csv
  NOAA NCEI global land+ocean monthly departure (1901-2000): climate-at-a-glance data.csv  (FIT WINDOW ONLY, <= 2024-06)
  NOAA GML Mauna Loa daily + monthly CO2: co2_daily_mlo.txt, co2_mm_mlo.txt (monthly used for the FIT WINDOW ONLY)
"""
from __future__ import annotations

import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "fx_climate"
RAW = D / "raw"
H = {"User-Agent": "Mozilla/5.0"}
FIRST, LAST = date(2024, 7, 25), date(2026, 11, 5)
PRODUCTS = ("GTM_", "MACD_")
FIT = ("2015-01", "2024-06")
CT = "America/Chicago"
ERA5 = "https://sites.ecmwf.int/data/climatepulse/data/series/era5_daily_series_2t_global.csv"
NOAA = ("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/1/0/"
        "2010-2024/data.csv")
MLO_D = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_daily_mlo.txt"
MLO_M = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"


def _get(url, **kw):
    r = requests.get(url, headers=H, timeout=180, **kw)
    r.raise_for_status()
    return r.text


def pull_one(d: date) -> str:
    f = RAW / f"p_{d:%Y%m%d}.csv"
    if f.exists() and d < date.today() - timedelta(days=2):
        return "cached"
    try:
        txt = _get("https://forecastex.com/api/download", params={"type": "prices", "date": f"{d:%Y%m%d}"})
    except requests.HTTPError as e:
        return f"http {e}"
    if not txt.strip():
        return "empty"
    t = pd.read_csv(io.StringIO(txt))
    t = t[t["event_contract"].astype(str).str.startswith(PRODUCTS)]
    t.to_csv(f, index=False)
    return f"{len(t)} rows"


def cmd_pull() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    ds = [FIRST + timedelta(days=i) for i in range((LAST - FIRST).days + 1)]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(pull_one, ds))
    bad = [(d, r) for d, r in zip(ds, res) if r.startswith(("http", "empty"))]
    print(f"{len(ds)} dates; cached {sum(r == 'cached' for r in res)}; missing {len(bad)} {bad[:5]}")
    return 0


def era5_monthly() -> pd.DataFrame:
    t = pd.read_csv(io.StringIO(_get(ERA5)), comment="#")
    t["date"] = pd.to_datetime(t["date"])
    t["m"] = t["date"].dt.to_period("M")
    g = t.groupby("m")["ano_91-20"]
    return pd.DataFrame({"pred_raw": g.mean(), "n_days": g.count()})


def noaa_monthly() -> pd.Series:
    t = pd.read_csv(io.StringIO(_get(NOAA)), comment="#")
    t.columns = ["Date", "v"]
    return pd.Series(t["v"].to_numpy(float), index=pd.PeriodIndex(t["Date"].astype(str), freq="M"))


def mlo_monthly() -> tuple[pd.DataFrame, pd.Series]:
    d = pd.read_csv(io.StringIO(_get(MLO_D)), comment="#", sep=r"\s+", header=None,
                    names=["y", "mo", "d", "dec", "v"])
    d = d[d["v"] > 0]
    d["m"] = pd.PeriodIndex.from_fields(year=d["y"], month=d["mo"], freq="M")
    g = d.groupby("m")["v"]
    m = pd.read_csv(io.StringIO(_get(MLO_M)), comment="#", sep=r"\s+", header=None,
                    names=["y", "mo", "dec", "avg", "deseas", "ndays", "sdev", "unc"])
    off = pd.Series(m["avg"].to_numpy(float), index=pd.PeriodIndex.from_fields(year=m["y"], month=m["mo"], freq="M"))
    return pd.DataFrame({"pred_raw": g.mean(), "n_days": g.count()}), off[off > 0]


def fit(pred: pd.Series, off: pd.Series) -> tuple[float, float, int]:
    w = pd.period_range(FIT[0], FIT[1], freq="M")
    r = (off.reindex(w) - pred.reindex(w)).dropna()
    return float(r.mean()), float(r.std(ddof=1)), int(len(r))


def cmd_build() -> int:
    fr = [pd.read_csv(f) for f in sorted(RAW.glob("p_*.csv"))]
    raw = pd.concat([x for x in fr if len(x)], ignore_index=True)
    raw["ticker"] = raw["event_contract"].astype(str)
    p = raw["ticker"].str.extract(r"^(GTM|MACD)_(\d{2})(\d{2})_(-?\d+(?:\.\d+)?)$")
    raw["product"], raw["X"] = p[0], pd.to_numeric(p[3], errors="coerce")
    raw["month"] = pd.to_datetime("20" + p[2] + "-" + p[1] + "-01", errors="coerce")
    raw = raw[raw["product"].notna() & raw["month"].notna()]
    raw["fdate"] = pd.to_datetime(raw["date"])
    exp = pd.to_datetime(raw.groupby("ticker")["expiration_date"].first(), utc=True)
    # outcome: YES settlement in the latest row dated >= expiration CT date, accepted only if exactly 0/1
    k = raw.drop_duplicates("ticker").set_index("ticker")[["product", "X", "month"]]
    k["exp_ts"] = exp.reindex(k.index).astype("int64") // 10**9
    k["exp_date"] = exp.reindex(k.index).dt.tz_convert(CT).dt.tz_localize(None).dt.normalize()
    y = raw[raw["subtype"] == "YES"].join(k["exp_date"], on="ticker")
    y = y[y["fdate"] >= y["exp_date"]].sort_values("fdate")
    fin = y.groupby("ticker").tail(1).set_index("ticker")["settlement_price"]
    k["settlement_price"] = fin.reindex(k.index)
    settled = k["settlement_price"].isin([0.0, 1.0])
    k["result"] = np.where(k["settlement_price"] == 1.0, "yes", "no")
    print(f"contracts {len(k)}; settled 0/1 {int(settled.sum())}; dropped (unsettled/non-binary) {int((~settled).sum())}")
    k = k[settled]
    # inputs
    e5 = era5_monthly()
    c_g, sd_g, n_g = fit(e5["pred_raw"], noaa_monthly())
    ml, mm = mlo_monthly()
    c_m, sd_m, n_m = fit(ml["pred_raw"], mm)
    inputs = {"GTM": {"c": c_g, "sd": sd_g, "fit_months": n_g, "fit": FIT, "as_of_days_after_month_end": 6},
              "MACD": {"c": c_m, "sd": sd_m, "fit_months": n_m, "fit": FIT, "as_of_days_after_month_end": 4}}
    print(json.dumps(inputs))
    rows = []
    for tk, r in k.iterrows():
        per = pd.Period(r["month"], freq="M")
        src, cfg = (e5, inputs["GTM"]) if r["product"] == "GTM" else (ml, inputs["MACD"])
        pr = src["pred_raw"].get(per, np.nan)
        nd = int(src["n_days"].get(per, 0))
        asof = (per.end_time.normalize() + pd.Timedelta(days=cfg["as_of_days_after_month_end"])).tz_localize("UTC")
        rows.append({"ticker": tk, "product": r["product"], "month": r["month"], "X": r["X"], "exp_ts": int(r["exp_ts"]),
                     "exp_date": r["exp_date"], "date": r["exp_date"], "pred": pr + cfg["c"], "sd": cfg["sd"],
                     "n_days": nd, "pred_as_of": int(asof.timestamp()), "result": r["result"],
                     "settlement_price": r["settlement_price"]})
    frame = pd.DataFrame(rows)
    days = raw[(raw["pair_quantity"] > 0) & raw["ticker"].isin(frame["ticker"])]
    days = pd.DataFrame({"ticker": days["ticker"], "file_date": days["fdate"].dt.strftime("%Y-%m-%d"),
                         "side": days["subtype"].str.lower(), "high": days["high_price"].astype(float),
                         "end": days["end_price"].astype(float), "qty": days["pair_quantity"].astype(float)})
    D.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(D / "frame.parquet", index=False)
    days.to_parquet(D / "days.parquet", index=False)
    (D / "inputs.json").write_text(json.dumps(inputs, indent=1))
    print(f"frame {len(frame)} contracts ({frame['product'].value_counts().to_dict()}), "
          f"{frame.groupby('product')['month'].nunique().to_dict()} months, exp {frame['date'].min():%Y-%m-%d}.."
          f"{frame['date'].max():%Y-%m-%d}; traded instrument-days {len(days)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"pull": cmd_pull, "build": cmd_build}.get(cmd, lambda: print(__doc__) or 1)())
