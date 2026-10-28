"""E1/E2 consensus-reject at EARLIER decision times on the day before (DEV only). Does deciding earlier buy NO cheaper?

    python scripts/wx_timing.py forecasts # forecasts only (Open-Meteo previous-runs lead-1 + lead-2), candidate tickers
    python scripts/wx_timing.py prices    # hourly candles (open .. 22:00 D-1) + public trade tapes for candidates
    python scripts/wx_timing.py dev [--ledger]

QUESTION (user 2026-10-27): E1/E2 buy NO at ~22:00 local the night before, when NO already costs ~94-99c. Kalshi daily-high
markets open at 10:00 ET on D-1 (checked: open_time 14:00Z/15:00Z on 2026 X1 rows). Deciding earlier on D-1 might buy
the same rejects cheaper, at the cost of staler forecasts.

ALREADY DEAD, NOT RE-TESTED HERE: 06:00 day-of with 00Z runs (N1, wx_dayof.py: D1 +0.19% p .34), 17:00 day-of with
observed max (L26, wx_lockin.py), hourly-temp series (N3, too thin). This file only tests D-1 decision times BEFORE 22:00.

PRE-REGISTERED 2026-10-27, written before any price, candle or trade tape was pulled for this study
------------------------------------------------------------------------------------------------------
Universe (DEV only; sealed rows are dropped on load, before anything is computed):
  E1 orig 7 cities : data/wx_<CITY>.parquet rows dated 2024-03-01 .. 2025-06-30 (inside the kalshi_temp DEV quarters
                     2021Q3..2025Q2; the 3-model rule cannot fire before 2024-03 when only GFS archives exist).
  X1 17 cities     : data/wx_expand.parquet kind == high, date < 2026-03-01.
  Drop result_mismatch rows (as the frozen rule's loaders do).
Decision times T (station-local clock, on D-1): 11:00, 14:00, 18:00; reference = 22:00 (the live E1/E2 time).
  (10:00 is not used: the book opens at 10:00 ET, so there is no candle ending at or before it in ET cities.)
Forecast available at T (leak-free generalisation of the frozen fcsafe definition): per model in {gfs_seamless,
  ecmwf_ifs025, icon_seamless, gem_seamless, ncep_nbm_conus}, value at hour h of day D = temperature_2m_previous_day1
  if h <= T - 4 (issued >= 24 h before h, i.e. by T - 4 on D-1: the same 4 h publication margin the frozen rule uses,
  hours <= 18 at T = 22), else temperature_2m_previous_day2 (issued >= 48 h before h). Daily value = max over hours
  0..18 of D, needs >= 15 non-null hours. At T = 22 this is exactly the frozen fcsafe_* (checked as a reproduction test).
Rule (frozen logic, margin varied): reject iff >= 3 models available and even the extreme model misses the bracket
  by > MARGIN F (greater: hi + M < floor + 1; less: lo - M > cap - 1; between: hi + M < floor or lo - M > cap).
  MARGIN in {3, 5}.
Price at T: last hourly candle ending at or before T (same as wx.prices_at_decision); tradeable quote only:
  yes_bid > 0, yes_ask < 1, spread <= 10c, yes_ask > 0.01.
Execution:
  taker : buy NO at the ask 1 - yes_bid, fee 0.07 * p * (1 - p) unrounded (exact direct-member fee).
  maker : post-only NO bid at the NO best bid q = 1 - yes_ask (fee 0, KXHIGH fee_type quadratic); filled iff >= 10
          contracts print at no_price <= q in (T, close] on the public tape (the E4 N=10 queue proxy). Unfilled = no bet.
Stake: 1 unit of cost per bet; pnl = 1/cost - 1 if NO wins else -1 (as every prior temp study).
TRIALS (12): T in {11, 14, 18} x MARGIN in {3, 5} x {taker, maker}. Bar p < 0.05 / 12 = 0.00417.
PRIMARY statistic per trial: delta units = sum pnl(rule at T) - sum pnl(same MARGIN + same execution at 22:00), over
  the same pooled dev station-days (each rule picks its own brackets; a rule with no tradeable pick that day adds 0).
  One-sided cluster bootstrap over calendar dates (all cities on a date resample together), 4000 draws, seed 0,
  p = P(delta <= 0). PASS iff delta > 0, p < bar, delta > 0 in both halves (split at the median dev date), and the
  T rule's own units > 0.
Reported per decision time (not extra trials): n, loss rate + Wilson 95% CI, break-even loss rate, mean NO price,
  mean profit per contract (c) and per WIN (c), ROI, units, both halves, picks/month, capacity proxy (displayed book
  depth is NOT available historically: report candle volume in the decision hour for taker, and contracts printed at
  <= q after T for maker), overlap with the 22:00 picks.
Ledger: lab/ledger.jsonl, dataset wx_timing, 12 rows.
Caveats stated up front: previous_day1/2 are Open-Meteo's 24 h/48 h-shifted archives, not run-by-run; the 4 h
  publication margin is inherited from the frozen rule, not re-verified per model. Candle close quotes are not a live
  book. Maker fill = prints at/through q, queue position unknown.
AMENDMENT 2026-10-27 (lead's instruction, after quotes were pulled, BEFORE any P&L was computed; rules unchanged):
  the trade-tape pull was stopped at 0 of 3,562 tickers. The 6 MAKER trials need tapes for their primary metric ->
  "NOT SCORED: data incomplete" (they stay in the trial count, bar stays 0.05/12; ledger status not_scored_data_incomplete).
  Their entry price (NO bid at T) is reported descriptively only. The 6 TAKER trials are scored as pre-registered.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports import wx
from sports.http import fetch, fetch_many
from kalshi_paper import PAPER_CITIES
import kalshi_temp_maker as KM

DATA = ROOT / "data"
FC = DATA / "wx_timing_fc.parquet"
PX = DATA / "wx_timing_quotes.parquet"
TR = DATA / "wx_timing_trades.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "wx_timing"
MODELS = {"gfs_seamless": "2021-03-01", "ecmwf_ifs025": "2024-03-01", "icon_seamless": "2024-03-01",
          "gem_seamless": "2024-03-01", "ncep_nbm_conus": "2025-03-01"}
TIMES = [11, 14, 18, 22]
MARGINS = [3.0, 5.0]
E1_START, E1_END, X1_END = pd.Timestamp("2024-03-01"), pd.Timestamp("2025-06-30"), pd.Timestamp("2026-03-01")
FEE = 0.07
N_FILL = 10


# --------------------------------------------------------------------------------------------- universe
def universe() -> pd.DataFrame:
    keep = ["ticker", "event_ticker", "strike_type", "floor_strike", "cap_strike", "expiration_value", "result",
            "close_time", "date", "result_mismatch"] + [f"fcsafe_{m}" for m in MODELS]
    fr = []
    for s in wx.CITIES:
        d = pd.read_parquet(DATA / f"wx_{s.replace('KXHIGH', '')}.parquet")
        d["date"] = pd.to_datetime(d["date"])
        d = d[(d.date >= E1_START) & (d.date <= E1_END)]  # sealed rows dropped before anything else
        fr.append(d[keep].assign(series=s, cohort="E1"))
    x = pd.read_parquet(DATA / "wx_expand.parquet")
    x["date"] = pd.to_datetime(x["date"])
    x = x[(x.kind == "high") & (x.date < X1_END)]
    fr.append(x[keep + ["series"]].assign(cohort="X1"))
    u = pd.concat(fr, ignore_index=True)
    u = u[~u.result_mismatch.astype(bool)].drop_duplicates("ticker").reset_index(drop=True)
    return u


# --------------------------------------------------------------------------------------------- forecasts
def model_hourly(model, lat, lon, tz, start, end) -> pd.DataFrame:
    fr, s = [], pd.Timestamp(start)
    while s <= pd.Timestamp(end):
        e = min(s + pd.Timedelta(days=364), pd.Timestamp(end))
        d = fetch(wx.PREV_RUNS, {"latitude": lat, "longitude": lon, "models": model,
                                 "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2",
                                 "temperature_unit": "fahrenheit", "timezone": tz,
                                 "start_date": s.date().isoformat(), "end_date": e.date().isoformat()}, cache=True)
        h = pd.DataFrame(d.get("hourly", {}))
        if not h.empty:
            fr.append(h)
        s = e + pd.Timedelta(days=1)
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def fc_at(h: pd.DataFrame, T: int) -> pd.Series:
    t = pd.to_datetime(h["time"])
    hr = t.dt.hour
    v = np.where(hr <= T - 4, h["temperature_2m_previous_day1"], h["temperature_2m_previous_day2"]).astype(float)
    g = pd.Series(v, index=t.dt.normalize())[(hr <= 18).to_numpy()]
    grp = g.groupby(level=0)
    return grp.max().where(grp.count() >= 15)


def cmd_fc() -> int:
    u = universe()
    rows = []
    for s, g in u.groupby("series"):
        st, tz = PAPER_CITIES[s]
        lat, lon = wx.station_coords(st)
        days = pd.DataFrame(index=sorted(g.date.unique()))
        for m, start in MODELS.items():
            lo = max(pd.Timestamp(start), g.date.min())
            if lo > g.date.max():
                continue
            h = model_hourly(m, lat, lon, tz, lo.date(), g.date.max().date())
            if h.empty:
                continue
            for T in TIMES:
                days = days.join(fc_at(h, T).rename(f"fc{T}_{m}"))
        days["series"] = s
        rows.append(days.rename_axis("date").reset_index())
        print(f"{s}: {len(days)} days", flush=True)
    fc = pd.concat(rows, ignore_index=True)
    fc.to_parquet(FC, index=False)
    # reproduction check: T=22 must equal the frozen fcsafe_* columns
    j = u.merge(fc, on=["series", "date"], how="left")
    for m in MODELS:
        a, b = j[f"fcsafe_{m}"], j[f"fc22_{m}"]
        both = a.notna() & b.notna()
        print(f"  repro {m}: rows {int(both.sum())}  |diff|>0.05: {int(((a - b).abs() > 0.05)[both].sum())}  "
              f"frozen-only {int((a.notna() & b.isna()).sum())}  new-only {int((a.isna() & b.notna()).sum())}")
    c = cands(j)
    print(f"candidate tickers (rejected at margin 3 at any T, before prices): {c.ticker.nunique()}")
    return 0


def rejected(d: pd.DataFrame, T: int, M: float) -> np.ndarray:
    cols = [f"fc{T}_{m}" for m in MODELS if f"fc{T}_{m}" in d.columns]
    n = d[cols].notna().sum(axis=1).to_numpy()
    hi, lo = d[cols].max(axis=1).to_numpy(float), d[cols].min(axis=1).to_numpy(float)
    f, c, st = d.floor_strike.to_numpy(float), d.cap_strike.to_numpy(float), d.strike_type.to_numpy()
    r = np.where(st == "greater", hi + M < f + 1, np.where(st == "less", lo - M > c - 1, (hi + M < f) | (lo - M > c)))
    return r & (n >= 3)


def cands(j: pd.DataFrame) -> pd.DataFrame:
    m = np.zeros(len(j), bool)
    for T in TIMES:
        m |= rejected(j, T, 3.0)
    return j[m]


# --------------------------------------------------------------------------------------------- prices + tapes
def t_local(date: pd.Timestamp, tz: str, hour: int) -> int:
    d = date - pd.Timedelta(days=1)
    return int(datetime(d.year, d.month, d.day, hour, tzinfo=ZoneInfo(tz)).timestamp())


def _candles(tk, t0, t1):
    s = tk.split("-")[0]
    for url in (f"{wx.API}/historical/markets/{tk}/candlesticks", f"{wx.API}/series/{s}/markets/{tk}/candlesticks"):
        try:
            d = fetch(f"{url}?start_ts={t0}&end_ts={t1}&period_interval=60", cache=True)
        except Exception as e:  # noqa: BLE001 -- 404 on the wrong endpoint
            if "404" in str(e) or "400" in str(e):
                continue
            raise
        if d.get("candlesticks"):
            return d["candlesticks"]
    return []


def cmd_prices() -> int:
    u = universe().merge(pd.read_parquet(FC), on=["series", "date"], how="left")
    c = cands(u).copy()
    c["tz"] = c.series.map(lambda s: PAPER_CITIES[s][1])
    for T in TIMES:
        c[f"T{T}"] = [t_local(d, tz, T) for d, tz in zip(c.date, c.tz)]
    c["close_ts"] = pd.to_datetime(c.close_time, utc=True).map(lambda x: int(x.timestamp()))
    print(f"{len(c)} candidate tickers", flush=True)
    out = []

    def one(r):
        cs = _candles(r.ticker, r.T11 - 4 * 3600, r.T22)
        rec = {"ticker": r.ticker}
        for T in TIMES:
            x = wx._row(r.ticker, r.event_ticker, cs, getattr(r, f"T{T}"))
            if x:
                last = [k for k in cs if k["end_period_ts"] <= getattr(r, f"T{T}")][-1]
                vol = float(last.get("volume_fp", last.get("volume", 0)) or 0)
                rec.update({f"yes_bid{T}": x["yes_bid"], f"yes_ask{T}": x["yes_ask"], f"pts{T}": x["price_ts"], f"vol{T}": vol})
        return rec

    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, rec in enumerate(ex.map(one, c.itertuples(index=False))):
            out.append(rec)
            if i % 500 == 0:
                print(f"  candles {i}/{len(c)}", flush=True)
    px = pd.DataFrame(out)
    px.to_parquet(PX, index=False)
    print(f"quotes: {px.yes_bid22.notna().sum() if 'yes_bid22' in px else 0} at 22, "
          f"{px.yes_bid11.notna().sum() if 'yes_bid11' in px else 0} at 11", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, r in enumerate(ex.map(lambda a: KM._tape(*a), zip(c.ticker, c.T11, c.close_ts))):
            rows += r
            if i % 500 == 0:
                print(f"  tapes {i}/{len(c)}", flush=True)
    tr = pd.DataFrame(rows, columns=["ticker", "ts", "no_price", "count", "taker_side"])
    tr.to_parquet(TR, index=False)
    print(f"trades {len(tr)} on {tr.ticker.nunique()} tickers")
    return 0


# --------------------------------------------------------------------------------------------- scoring
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def frame() -> pd.DataFrame:
    u = universe().merge(pd.read_parquet(FC), on=["series", "date"], how="left")
    u = u.merge(pd.read_parquet(PX), on="ticker", how="left")
    u["tz"] = u.series.map(lambda s: PAPER_CITIES[s][1])
    u["close_ts"] = pd.to_datetime(u.close_time, utc=True).map(lambda x: int(x.timestamp()))
    for T in TIMES:
        u[f"T{T}"] = [t_local(d, tz, T) for d, tz in zip(u.date, u.tz)]
    u["won"] = (u.result == "no").astype(int)
    return u


def bets(u: pd.DataFrame, tr_g: dict, T: int, M: float, style: str) -> pd.DataFrame:
    b, a = u[f"yes_bid{T}"], u[f"yes_ask{T}"]
    ok = rejected(u, T, M) & (b > 0).to_numpy() & (a < 1).to_numpy() & ((a - b) <= 0.10).to_numpy() & (a > 0.01).to_numpy()
    s = u[ok].copy()
    if style == "taker":
        p = 1 - s[f"yes_bid{T}"].to_numpy(float)
        s["price"], s["cost"] = p, p + FEE * p * (1 - p)
        s["cap"] = s[f"vol{T}"]
        s["bet"] = True
    else:
        q = np.round(1 - s[f"yes_ask{T}"].to_numpy(float), 4)
        filled, printed = np.zeros(len(s), bool), np.zeros(len(s))
        for i, (tk, t0, t1) in enumerate(zip(s.ticker, s[f"T{T}"], s.close_ts)):
            d = tr_g.get(tk)
            if d is None:
                continue
            m = (d.ts.to_numpy() > t0) & (d.ts.to_numpy() <= t1) & (d.no_price.to_numpy() <= q[i] + 1e-9)
            printed[i] = d["count"].to_numpy()[m].sum()
            filled[i] = printed[i] >= N_FILL
        s["price"], s["cost"], s["cap"], s["bet"] = q, q, printed, filled
    s = s[s.bet].copy()
    s["pnl"] = np.where(s.won == 1, 1 / s.cost - 1, -1.0)
    s["ppc"] = np.where(s.won == 1, 1 - s.cost, -s.cost)  # $ per contract
    return s


def day_sums(s: pd.DataFrame, days: pd.Index) -> np.ndarray:
    return s.groupby(s.date.dt.strftime("%Y-%m-%d"))["pnl"].sum().reindex(days, fill_value=0.0).to_numpy()


def describe(s: pd.DataFrame, half: pd.Timestamp, months: float) -> dict:
    n, k = len(s), int((s.won == 0).sum())
    lo, hi = wilson(k, n)
    w = s[s.won == 1]
    return {"bets": n, "losses": k, "loss_rate": k / n if n else float("nan"), "loss_ci": [lo, hi],
            "breakeven_loss_rate": float((1 - s.cost).mean()) if n else float("nan"),
            "mean_no_price": float(s.price.mean()) if n else float("nan"),
            "profit_per_contract_c": float(100 * s.ppc.mean()) if n else float("nan"),
            "profit_per_win_c": float(100 * (1 - w.cost).mean()) if len(w) else float("nan"),
            "units": float(s.pnl.sum()), "roi": float(s.pnl.mean()) if n else 0.0,
            "units_h1": float(s[s.date < half].pnl.sum()), "units_h2": float(s[s.date >= half].pnl.sum()),
            "bets_per_month": n / months, "cap_median": float(s.cap.median()) if n else float("nan"),
            "cap_total": float(s.cap.sum()), "cap_dollars": float((s.cap * s.price).sum())}


def cmd_dev() -> int:
    u = frame()
    tr_g = {k: v for k, v in pd.read_parquet(TR).groupby("ticker")} if TR.exists() else None
    styles = ("taker", "maker") if tr_g is not None else ("taker",)
    days = pd.Index(sorted(u.date.dt.strftime("%Y-%m-%d").unique()))
    half = pd.Timestamp(days[len(days) // 2])
    months = max(1.0, (u.date.max() - u.date.min()).days / 30.4)
    print(f"DEV {u.series.nunique()} series, {u.event_ticker.nunique()} station-days, {len(days)} dates "
          f"{days[0]}..{days[-1]}, half split {half.date()}; candidate quotes at 22: {int(u.yes_bid22.notna().sum())}")
    rng_seed = 0
    ref, recs, table = {}, [], []
    for M in MARGINS:
        for style in styles:
            ref[(M, style)] = bets(u, tr_g, 22, M, style)
            r = describe(ref[(M, style)], half, months)
            table.append({"T": 22, "M": M, "style": style, **r, "delta": 0.0, "p": float("nan"), "verdict": "reference"})
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    for T in TIMES[:-1]:
        for M in MARGINS:
            if tr_g is None:  # maker trials: not scored (amendment)
                name = f"T{T}_m{int(M)}_maker"
                recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"wx_timing_{name}",
                             "sha": hashlib.sha256(("wx_timing:" + name).encode()).hexdigest()[:12], "dataset": DATASET,
                             "trial": name, "hypothesis": f"pre-registered in scripts/wx_timing.py ({name})",
                             "status": "not_scored_data_incomplete", "study_trials": 12, "bonferroni_bar": 0.05 / 12})
            for style in styles:
                s, b0 = bets(u, tr_g, T, M, style), ref[(M, style)]
                d = day_sums(s, days) - day_sums(b0, days)
                rng = np.random.default_rng(rng_seed)
                idx = rng.integers(0, len(days), size=(4000, len(days)))
                bd = d[idx].sum(1)
                p = float(np.mean(bd <= 0))
                hmask = np.array([pd.Timestamp(x) < half for x in days])
                dh1, dh2 = float(d[hmask].sum()), float(d[~hmask].sum())
                r = describe(s, half, months)
                ov = len(set(s.ticker) & set(b0.ticker))
                ok = d.sum() > 0 and p < 0.05 / 12 and dh1 > 0 and dh2 > 0 and r["units"] > 0
                row = {"T": T, "M": M, "style": style, **r, "delta": float(d.sum()),
                       "delta_ci": [float(np.quantile(bd, .025)), float(np.quantile(bd, .975))], "p": p,
                       "delta_h1": dh1, "delta_h2": dh2, "overlap_with_22": ov, "verdict": "PASS" if ok else "fail"}
                table.append(row)
                name = f"T{T}_m{int(M)}_{style}"
                recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"wx_timing_{name}",
                             "sha": hashlib.sha256(("wx_timing:" + name).encode()).hexdigest()[:12], "dataset": DATASET,
                             "trial": name, "hypothesis": f"pre-registered in scripts/wx_timing.py ({name})",
                             "source": "Kalshi KXHIGH hourly candles + public tape, Open-Meteo previous-runs lead1/lead2",
                             "status": "ok", "study_trials": 12, "bonferroni_bar": 0.05 / 12, **row})
    cols = ["T", "M", "style", "bets", "losses", "loss_rate", "loss_ci", "breakeven_loss_rate", "mean_no_price",
            "profit_per_contract_c", "profit_per_win_c", "units", "roi", "units_h1", "units_h2", "bets_per_month",
            "cap_median", "cap_dollars", "delta", "delta_ci", "p", "delta_h1", "delta_h2", "overlap_with_22", "verdict"]
    t = pd.DataFrame(table)[cols].sort_values(["M", "style", "T"])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    t.to_json(DATA / "wx_timing_results.json", orient="records", indent=1, default_handler=str)
    for T in TIMES:  # descriptive: NO bid (maker entry) on the rule's tradeable rejects, no fill model
        for M in MARGINS:
            b, a = u[f"yes_bid{T}"], u[f"yes_ask{T}"]
            ok = rejected(u, T, M) & (b > 0).to_numpy() & (a < 1).to_numpy() & ((a - b) <= 0.10).to_numpy() & (a > 0.01).to_numpy()
            x = u[ok]
            print(f"T{T} m{int(M)}: mean NO ask {(1 - x[f'yes_bid{T}']).mean():.4f}  NO bid {(1 - x[f'yes_ask{T}']).mean():.4f}  "
                  f"spread {(x[f'yes_ask{T}'] - x[f'yes_bid{T}']).mean():.4f}  n {len(x)}  median candle vol {x[f'vol{T}'].median():.0f}  "
                  f"loss rate {1 - x.won.mean():.4f}")
    # cohort split (descriptive)
    for T in TIMES:
        s = bets(u, tr_g, T, 3.0, "taker")
        print(f"T{T} m3 taker by cohort:", {c: f"{g.pnl.sum():+.2f}/{len(g)} NO {g.price.mean():.3f}" for c, g in s.groupby("cohort")})
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"appended {len(recs)} rows to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"forecasts": cmd_fc, "prices": cmd_prices, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
