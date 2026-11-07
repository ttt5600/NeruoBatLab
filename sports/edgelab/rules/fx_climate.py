"""FC1 / FC2: ForecastEx monthly PHYSICAL ladders, nowcast-reject (data built by scripts/fx_climate.py).

Contracts (forecastex.com /api/products, read 2026-11-06 before any price file of these products was pulled):
  GTM_<MMYY>_<X>   "Will the Average Global Temperature in <month> be greater than X degrees Celsius above the 20th
                    century average?"  NOAA NCEI monthly global report (NOAAGlobalTemp land+ocean departure, 1901-2000
                    base). Expires on the report day (~10th-13th of M+1).
  MACD_<MMYY>_<X>  "Will Monthly Average Atmospheric Carbon Dioxide be greater than X ppm in <month>?"  NOAA GML trends
                    page; expiry ~5th of M+1 = Mauna Loa monthly mean (co2_mm_mlo; the global mean lags ~2 months, and
                    the listed strikes bracket the Mauna Loa level, not the global one).
  YES pays $1 iff value > X (strict). YES and NO are separate instruments, $0.01/contract fee.

Frame (loader fx_climate): one row per settled contract.
  ticker, product (GTM|MACD), month (first day of M), X, exp_ts (UTC epoch s of the contract expiration),
  exp_date (expiration date in America/Chicago = vault split column `date`),
  pred      GTM: mean over the days of M of Copernicus/ECMWF Climate Pulse ERA5 daily global 2 m anomaly (1991-2020)
                 + c, c = mean(NOAA - ERA5 monthly) over 2015-01..2024-06 (fit window ends before the first dev event).
            MACD: mean of NOAA GML Mauna Loa daily CO2 values of M + c, c = mean(co2_mm_mlo - mean of daily) over
                 2015-01..2024-06.
  sd        residual sd of (official - pred) over the same fit window (one constant per product).
  n_days    days of M with a predictor value;  pred_as_of = UTC epoch s when the month's predictor was public:
            GTM M_end + 6 d 00:00 UTC (CDS 5-day delay), MACD M_end + 4 d 00:00 UTC.
attrs['days'] = every daily-file row of these contracts with pair_quantity > 0:
  ticker, file_date (ForecastEx trade date, YYYY-MM-DD, taken to be a Central-time day), side (yes|no),
  high (that instrument's highest trade that day), end (its last trade that day), qty.

RULE (frozen 2026-11-06, before any GTM/MACD price or outcome was pulled):
  usable: GTM n_days == days in M; MACD n_days >= 20.
  m = K * sd, K = 3.0.   NO  if pred + m <= X   (value cannot plausibly exceed X)
                          YES if pred - m >  X   (value cannot plausibly stay <= X)
  decision day F = the EARLIEST file_date whose CT day STARTS at/after pred_as_of (every trade of F is after the input
     was public) and F <= exp_date - GAP days (GAP = 2)
     on which the picked instrument traded (qty > 0) with 0 < high and high + 0.01 < 1.  No such day -> no bet.
  price = that day's HIGH trade of the bought instrument (daily files have no bid/ask: this is the worst price any
     buyer paid that day, a proxy for the ask, not a quote) ; fee $0.01/contract (edgelab schedule 'forecastex').
  decided_at = price_time = end of CT day F (next CT midnight - 1 s); inputs_as_of = pred_as_of;
  event_start = exp_date 00:00 America/Chicago (the release day; GAP keeps every decision >= 1 day before it).
  close_price (CLV) = bought instrument's last trade (`end`) on the LATEST later file_date in (F, exp_date - 1]
     where it traded; NaN if none.
  cluster = product|month (all strikes of one release share the outcome). One bet per contract, stake 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K = 3.0
GAP = 2
FEE = 0.01
CT = "America/Chicago"
MIN_DAYS = {"GTM": None, "MACD": 20}


def _ct_end(d: pd.Timestamp) -> pd.Timestamp:
    """End of Central-time day d (next CT midnight - 1 s) as UTC."""
    return (pd.Timestamp(d.date()) + pd.Timedelta(days=1)).tz_localize(CT).tz_convert("UTC") - pd.Timedelta(seconds=1)


def select(frame: pd.DataFrame, k: float = K) -> pd.DataFrame:
    """Input-only selection: rows with a side."""
    f = frame.copy()
    m = k * f["sd"].astype(float)
    pred, X = f["pred"].astype(float), f["X"].astype(float)
    month = pd.to_datetime(f["month"])
    full = month.dt.days_in_month
    need = np.where(f["product"] == "GTM", full, MIN_DAYS["MACD"])
    ok = np.isfinite(pred) & np.isfinite(m) & (f["n_days"].astype(float) >= need)
    f["side"] = np.where(ok & (pred + m <= X), "no", np.where(ok & (pred - m > X), "yes", ""))
    return f[f["side"] != ""]


def bets(frame: pd.DataFrame, params: dict | None = None, product: str | None = None) -> pd.DataFrame:
    params = params or {}
    s = select(frame, float(params.get("k", K)))
    if product:
        s = s[s["product"] == product]
    gap = int(params.get("gap_days", GAP))
    days = frame.attrs.get("days")
    g = {key: v for key, v in days.groupby(["ticker", "side"])} if days is not None and len(days) else {}
    out = []
    for r in s.itertuples(index=False):
        d = g.get((r.ticker, r.side))
        if d is None:
            continue
        fd = pd.to_datetime(d["file_date"])
        exp_d = pd.Timestamp(r.exp_date)
        asof = pd.Timestamp(int(r.pred_as_of), unit="s", tz="UTC")
        first = pd.Timestamp(asof.tz_convert(CT).date())      # first CT day that STARTS at/after pred_as_of
        if pd.Timestamp(first.date()).tz_localize(CT) < asof:
            first += pd.Timedelta(days=1)
        hi = d["high"].to_numpy(float)
        win = (fd >= first) & (fd <= exp_d - pd.Timedelta(days=gap)) & (hi > 0) & (hi + FEE < 1) \
            & (d["qty"].to_numpy(float) > 0)
        if not win.any():
            continue
        i = np.flatnonzero(win)[np.argmin(fd[win].to_numpy())]
        F = fd.iloc[i]
        t = _ct_end(F)
        later = (fd > F) & (fd <= exp_d - pd.Timedelta(days=1)) & (d["qty"].to_numpy(float) > 0) \
            & np.isfinite(d["end"].to_numpy(float))
        close = float(d["end"].to_numpy(float)[np.flatnonzero(later)[np.argmax(fd[later].to_numpy())]]) \
            if later.any() else np.nan
        out.append({"bet_id": r.ticker, "cluster": f"{r.product}|{pd.Timestamp(r.month):%Y-%m}",
                    "event_start": pd.Timestamp(exp_d.date()).tz_localize(CT).tz_convert("UTC"),
                    "decided_at": t, "inputs_as_of": asof, "side": r.side, "stake": 1.0,
                    "price": float(hi[i]), "price_source": "taker", "price_time": t, "close_price": close})
    cols = ["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "side", "stake", "price", "price_source",
            "price_time", "close_price"]
    return pd.DataFrame(out, columns=cols)


def gtm(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return bets(frame, params, "GTM")


def macd(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return bets(frame, params, "MACD")
