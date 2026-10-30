"""Q016 -- Kalshi AAA gas weekly + commodity ladders (WTI, Brent, gold, nat gas) vs a public futures-based fair value.

Pre-registration: notes/q016_commodities.md (written before any price was joined to an outcome).

    python scripts/q016_commodities.py markets   # settled markets -> data/q016/markets.parquet + outcomes.parquet
    python scripts/q016_commodities.py ref       # Yahoo futures (daily + hourly) + EIA weekly retail gas, frozen
    python scripts/q016_commodities.py fitgas    # gas pass-through OLS on EIA 2006-2022 -> gas_model.json
    python scripts/q016_commodities.py decide    # decision time + model P(YES) per market (past-only inputs)
    python scripts/q016_commodities.py sample    # seeded outcome-free draw <= 3,000 markets (rate-limit cap)
    python scripts/q016_commodities.py quotes    # Kalshi hourly candles at the decision time
    python scripts/q016_commodities.py avail     # quote availability by model distance (NO outcomes)
    python scripts/q016_commodities.py score dev|sealed [--ledger]
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from sports.http import fetch, fetch_many

API = "https://api.elections.kalshi.com/trade-api/v2"
D = ROOT / "data" / "q016"
D.mkdir(parents=True, exist_ok=True)
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "q016_commodities"
ET = ZoneInfo("America/New_York")
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
TAKER = 0.07
K_TRIALS = 4
BAR = 0.05 / K_TRIALS
HOURLY_START = pd.Timestamp("2024-06-10", tz="UTC")
YF = {"WTI": "CL=F", "BRENT": "BZ=F", "GOLD": "GC=F", "NATGAS": "NG=F"}
TRIALS = {
    "T1_GAS_W": {"series": {"KXAAAGASW": "GAS"}},
    "T2_WTI_D_FADE": {"series": {"KXWTI": "WTI"}},
    "T3_PYTH_D_FADE": {"series": {"KXGOLDD": "GOLD", "KXNATGASD": "NATGAS", "KXBRENTD": "BRENT"}},
    "T4_WEEKLY_FADE": {"series": {"KXWTIW": "WTI", "KXGOLDW": "GOLD", "KXNATGASW": "NATGAS", "KXBRENTW": "BRENT"}},
}
ALL_SERIES = {s: a for t in TRIALS.values() for s, a in t["series"].items()}


def ts_int(x) -> np.ndarray:
    x = pd.to_datetime(x, utc=True, format="ISO8601")
    return ((x - EPOCH) // pd.Timedelta(seconds=1)).astype("int64")


def et_dt(sec: int) -> datetime:
    return datetime.fromtimestamp(int(sec), tz=timezone.utc).astimezone(ET)


def et_ts(d, hour: int, minute: int = 0) -> int:
    return int(datetime(d.year, d.month, d.day, hour, minute, tzinfo=ET).timestamp())


# ------------------------------------------------------------------------------------------------ markets
def settled(series: str) -> pd.DataFrame:
    rows = []
    for path, extra, cache in ((f"{API}/historical/markets", {}, True), (f"{API}/markets", {"status": "settled"}, False)):
        cur = None
        while True:
            p = {"series_ticker": series, "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            d = fetch(path, p, cache=cache)
            rows += d.get("markets", [])
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    return pd.DataFrame(rows).drop_duplicates("ticker")


def cmd_markets() -> int:
    mk, oc = [], []
    for s in ALL_SERIES:
        df = settled(s)
        df = df[df.result.isin(["yes", "no"])].copy()
        if "strike_type" not in df:
            df["strike_type"] = "greater"
        df["strike_type"] = df.strike_type.fillna("greater")
        df = df[df.strike_type.isin(["greater", "less", "between"])]
        for c in ("floor_strike", "cap_strike"):
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
        df["series"] = s
        df["open_ts"] = ts_int(df.open_time)
        df["close_ts"] = ts_int(df.close_time)
        df["close_ts"] = df.groupby("event_ticker").close_ts.transform("max")
        df["settle_date"] = pd.to_datetime(df.event_ticker.str.split("-").str[1].str[:7], format="%y%b%d", errors="coerce")
        df = df[df.settle_date.notna()]
        mk.append(df[["series", "ticker", "event_ticker", "strike_type", "floor_strike", "cap_strike", "open_ts", "close_ts", "settle_date"]])
        oc.append(df[["ticker", "event_ticker", "result", "expiration_value"]])
        print(f"{s}: {len(df)} markets, {df.event_ticker.nunique()} events {df.settle_date.min().date()}..{df.settle_date.max().date()}", flush=True)
    pd.concat(mk).to_parquet(D / "markets.parquet", index=False)
    o = pd.concat(oc)
    o["expiration_value"] = pd.to_numeric(o.expiration_value, errors="coerce")
    o.to_parquet(D / "outcomes.parquet", index=False)
    return 0


# ------------------------------------------------------------------------------------------------ reference data
def cmd_ref() -> int:
    import yfinance as yf
    import requests
    for a, tk in YF.items():
        h = yf.download(tk, period="730d", interval="1h", progress=False, auto_adjust=False)
        h.columns = [c[0] if isinstance(c, tuple) else c for c in h.columns]
        h = h.reset_index().rename(columns={"Datetime": "start"})
        h["start_ts"] = ts_int(h.start)
        h[["start_ts", "Close", "Volume"]].rename(columns={"Close": "close", "Volume": "volume"}).to_parquet(D / f"yf_1h_{a}.parquet", index=False)
        print(a, tk, "hourly", len(h), h.start.min(), h.start.max())
    rb = yf.download("RB=F", period="max", interval="1d", progress=False, auto_adjust=False)
    rb.columns = [c[0] if isinstance(c, tuple) else c for c in rb.columns]
    rb = rb.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "rb"})
    rb.to_parquet(D / "yf_1d_RB.parquet", index=False)
    print("RB daily", len(rb), rb.date.min(), rb.date.max())
    r = requests.get("https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_NUS_DPGw.xls", headers={"User-Agent": "Mozilla/5.0 (research)"}, timeout=60)
    (D / "eia_gas_w.xls").write_bytes(r.content)
    x = pd.read_excel(D / "eia_gas_w.xls", sheet_name="Data 1", skiprows=2)
    x.columns = ["date", "eia"]
    x.to_parquet(D / "eia_gas_w.parquet", index=False)
    print("EIA weekly", len(x), x.date.min(), x.date.max())
    return 0


# ------------------------------------------------------------------------------------------------ gas model
def gas_features(eia: pd.DataFrame, rb: pd.DataFrame) -> pd.DataFrame:
    rbd = rb.set_index("date").rb.sort_index()

    def rb_before(day):
        s = rbd[rbd.index < day]
        return float(s.iloc[-1]) if len(s) else np.nan

    e = eia.sort_values("date").reset_index(drop=True).copy()
    e["rb0"] = [rb_before(d) for d in e.date]
    e["rb1"] = [rb_before(d - pd.Timedelta(days=7)) for d in e.date]
    e["rb2"] = [rb_before(d - pd.Timedelta(days=14)) for d in e.date]
    e["margin"] = e.eia - e.rb0
    e["margin_mean52"] = e.margin.shift(1).rolling(52, min_periods=40).mean()
    e["next"] = e.eia.shift(-1)
    e["next_date"] = e.date.shift(-1)
    return e


def gas_x(rw, rb0, rb1, rb2, mm52):
    return np.array([1.0, rb0 - rb1, rb1 - rb2, (rw - rb0) - mm52])


def cmd_fitgas() -> int:
    eia = pd.read_parquet(D / "eia_gas_w.parquet")
    rb = pd.read_parquet(D / "yf_1d_RB.parquet")
    e = gas_features(eia, rb)
    fit = e[(e.date >= "2006-01-02") & (e.date <= "2022-12-26")].dropna(subset=["rb0", "rb1", "rb2", "margin_mean52", "next"])
    fit = fit[(fit.next_date - fit.date) == pd.Timedelta(days=7)]
    X = np.vstack([gas_x(r.eia, r.rb0, r.rb1, r.rb2, r.margin_mean52) for r in fit.itertuples()])
    y = (fit.next - fit.eia).to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    r2 = 1 - res.var() / y.var()
    json.dump({"beta": beta.tolist(), "n": int(len(y)), "r2": float(r2), "resid_sd": float(res.std(ddof=4)),
               "fit": "EIA Mondays 2006-01-02..2022-12-26"}, open(D / "gas_model.json", "w"), indent=1)
    print(f"gas OLS n={len(y)} beta(const, dRB_1w, dRB_2w, margin_dev)={np.round(beta, 4).tolist()} R2={r2:.3f} resid sd={res.std(ddof=4):.4f}")
    # direction hit-rate in-sample (descriptive)
    print(f"in-sample sign agreement {np.mean(np.sign(X @ beta) == np.sign(y)):.3f}")
    return 0


# ------------------------------------------------------------------------------------------------ decisions
def hourly(asset: str) -> pd.DataFrame:
    h = pd.read_parquet(D / f"yf_1h_{asset}.parquet").sort_values("start_ts")
    h = h[h.close.notna() & (h.close > 0)]
    h["lr"] = np.log(h.close).diff()
    return h.reset_index(drop=True)


def ref_at(h: pd.DataFrame, t: int):
    """Close of the last bar usable at t (start + 3600 <= t), its start, and sigma^2 per hour."""
    i = np.searchsorted(h.start_ts.to_numpy(), t - 3600, side="right") - 1
    if i < 0:
        return None
    last = h.iloc[i]
    if t - (int(last.start_ts) + 3600) > 4 * 3600:
        return None
    w = h.iloc[: i + 1]
    w = w[w.volume > 0].lr.dropna().tail(300)
    if len(w) < 200:
        return None
    return float(last.close), int(last.start_ts), float(np.mean(w.to_numpy() ** 2))


def bar_before(h: pd.DataFrame, close_ts: int):
    """Close of the bar containing the last minute before close_ts."""
    i = np.searchsorted(h.start_ts.to_numpy(), close_ts - 60, side="right") - 1
    if i < 0 or close_ts - 60 - int(h.start_ts.iloc[i]) >= 3600:
        return np.nan
    return float(h.close.iloc[i])


def trading_hours(t: int, close: int) -> float:
    hrs = (close - t) / 3600
    d = et_dt(t).date()
    end = et_dt(close)
    while True:
        b = datetime(d.year, d.month, d.day, 17, tzinfo=ET)
        if b >= end:
            break
        if b > et_dt(t):
            hrs -= 1
            if b.weekday() == 4:
                hrs -= 48
        d += timedelta(days=1)
    return max(hrs, 0.25)


T4DF = 4.0
T4SCALE = math.sqrt((T4DF - 2) / T4DF)


def p_above(x, mu_log, sd):
    """P(S > x) with log S = mu_log + sd * Z, Z = unit-variance t4."""
    z = (np.log(x) - mu_log) / sd
    return stats.t.sf(z / T4SCALE, T4DF)


def p_yes_market(r, mu_log, sd):
    if r.strike_type == "greater":
        return float(p_above(r.floor_strike, mu_log, sd))
    if r.strike_type == "less":
        return float(1 - p_above(r.cap_strike, mu_log, sd))
    return float(p_above(r.floor_strike, mu_log, sd) - p_above(r.cap_strike + 0.01, mu_log, sd))


def decide_futures(mk: pd.DataFrame, oc: pd.DataFrame, trial: str) -> pd.DataFrame:
    out = []
    ev_val = oc.groupby("event_ticker").expiration_value.max()
    for s, asset in TRIALS[trial]["series"].items():
        h = hourly(asset)
        m = mk[(mk.series == s)]
        evs = m.groupby("event_ticker").agg(close_ts=("close_ts", "max"), open_ts=("open_ts", "min"), settle_date=("settle_date", "first")).sort_values("close_ts")
        evs = evs[pd.to_datetime(evs.close_ts, unit="s", utc=True) >= HOURLY_START]
        evs["val"] = evs.index.map(ev_val)
        evs["yf_at_close"] = [bar_before(h, int(c)) for c in evs.close_ts]
        evs["basis"] = np.log(evs.val) - np.log(evs.yf_at_close)
        n_skip = {"guard": 0, "ref": 0, "time": 0}
        for e, r in evs.iterrows():
            cl = et_dt(r.close_ts)
            if trial == "T4_WEEKLY_FADE":
                mon = cl.date() - timedelta(days=cl.weekday())
                t = et_ts(mon, 10)
            else:
                t = et_ts(cl.date(), 10)
            if not (r.open_ts <= t < r.close_ts - 1800):
                n_skip["time"] += 1
                continue
            prior = evs[(evs.close_ts <= t - 3600) & evs.basis.notna()].tail(5)
            if len(prior) < 3 or abs(prior.basis.iloc[-1] - prior.basis.median()) > 0.005:
                n_skip["guard"] += 1
                continue
            ra = ref_at(h, t)
            if ra is None:
                n_skip["ref"] += 1
                continue
            C, bar_start, s2 = ra
            hrs = trading_hours(t, int(r.close_ts))
            mu = math.log(C) + float(prior.basis.median())
            sd = math.sqrt(s2 * hrs)
            for mr in m[m.event_ticker == e].itertuples():
                out.append({"trial": trial, "series": s, "asset": asset, "ticker": mr.ticker, "event_ticker": e,
                            "strike_type": mr.strike_type, "floor_strike": mr.floor_strike, "cap_strike": mr.cap_strike,
                            "open_ts": mr.open_ts, "close_ts": int(r.close_ts), "settle_date": r.settle_date, "t": t,
                            "ref_bar_start": bar_start, "ref_close": C, "basis": float(prior.basis.median()), "hours": hrs,
                            "center": math.exp(mu), "sd_log": sd, "p_yes": p_yes_market(mr, mu, sd)})
        print(f"  {trial} {s}: events {len(evs)}, skipped {n_skip}", flush=True)
    return pd.DataFrame(out)


def decide_gas(mk: pd.DataFrame, oc: pd.DataFrame) -> pd.DataFrame:
    g = json.load(open(D / "gas_model.json"))
    beta = np.array(g["beta"])
    eia = pd.read_parquet(D / "eia_gas_w.parquet")
    rb = pd.read_parquet(D / "yf_1d_RB.parquet")
    e = gas_features(eia, rb).set_index("date")
    # out-of-sample (frozen-coef) residuals on EIA, known from the day after next_date
    e["pred"] = [float(gas_x(r.eia, r.rb0, r.rb1, r.rb2, r.margin_mean52) @ beta) if pd.notna(r.margin_mean52) and pd.notna(r.rb2) else np.nan
                 for r in e.itertuples()]
    e["resid"] = (e.next - e.eia) - e.pred
    m = mk[mk.series == "KXAAAGASW"]
    ev_val = oc.groupby("event_ticker").expiration_value.max()
    evs = m.groupby("event_ticker").agg(close_ts=("close_ts", "max"), open_ts=("open_ts", "min"), settle_date=("settle_date", "first")).sort_values("close_ts")
    evs["val"] = evs.index.map(ev_val)
    out, skip = [], {"time": 0, "prev": 0, "offset": 0, "feat": 0}
    for ev, r in evs.iterrows():
        lst = et_dt(r.open_ts) + timedelta(minutes=60)
        noon = datetime(lst.year, lst.month, lst.day, 12, tzinfo=ET)
        t0 = max(lst, noon)
        t = int(math.ceil(t0.timestamp() / 3600) * 3600)
        if et_dt(t).date() != et_dt(r.open_ts).date() or t >= r.close_ts:
            skip["time"] += 1
            continue
        prev = evs[(evs.close_ts <= t - 3600)]
        if not len(prev) or abs((r.settle_date - prev.settle_date.iloc[-1]).days - 7) > 0 or pd.isna(prev.val.iloc[-1]):
            skip["prev"] += 1
            continue
        A = float(prev.val.iloc[-1])
        mon = prev.settle_date.iloc[-1]  # Monday whose AAA = A
        both = [(float(v), float(e.eia.get(d))) for d, v in zip(prev.settle_date, prev.val)
                if d < mon and d in e.index and pd.notna(v)]
        both = both[-8:]
        if len(both) < 4:
            skip["offset"] += 1
            continue
        off = float(np.median([a - b for a, b in both]))
        if mon not in e.index or pd.isna(e.loc[mon, "rb2"]) or pd.isna(e.loc[mon, "margin_mean52"]):
            skip["feat"] += 1
            continue
        f = e.loc[mon]
        yhat = float(gas_x(A - off, f.rb0, f.rb1, f.rb2, f.margin_mean52) @ beta)
        known = e[(e.next_date < mon) & e.resid.notna()].resid.tail(26)
        s = max(float(known.std(ddof=1)), 0.01)
        for mr in m[m.event_ticker == ev].itertuples():
            p = float(1 - stats.norm.cdf((mr.floor_strike - A - yhat) / s))
            out.append({"trial": "T1_GAS_W", "series": "KXAAAGASW", "asset": "GAS", "ticker": mr.ticker, "event_ticker": ev,
                        "strike_type": mr.strike_type, "floor_strike": mr.floor_strike, "cap_strike": mr.cap_strike,
                        "open_ts": mr.open_ts, "close_ts": int(r.close_ts), "settle_date": r.settle_date, "t": t, "aaa_now": A,
                        "offset": off, "yhat": yhat, "sd": s, "center": A + yhat, "p_yes": p, "n_rungs": int((m.event_ticker == ev).sum())})
    print(f"  T1_GAS_W: events {len(evs)}, skipped {skip}", flush=True)
    return pd.DataFrame(out)


def cmd_decide() -> int:
    mk = pd.read_parquet(D / "markets.parquet")
    oc = pd.read_parquet(D / "outcomes.parquet")[["ticker", "event_ticker", "expiration_value"]]  # inputs: past values only
    frames = [decide_gas(mk, oc)] + [decide_futures(mk, oc, t) for t in ("T2_WTI_D_FADE", "T3_PYTH_D_FADE", "T4_WEEKLY_FADE")]
    d = pd.concat(frames, ignore_index=True)
    # lookahead asserts
    assert (d.t < d.close_ts).all()
    if "ref_bar_start" in d:
        x = d[d.ref_bar_start.notna()]
        assert (x.ref_bar_start + 3600 <= x.t).all()
    # per-trial 70/30 split by distinct settlement date (outcome-free)
    d["split"] = ""
    for tr, g in d.groupby("trial"):
        dates = np.sort(g.settle_date.unique())
        cut = dates[int(math.floor(0.7 * len(dates)))]
        d.loc[g.index, "split"] = np.where(g.settle_date < cut, "dev", "sealed")
        print(f"{tr}: {g.event_ticker.nunique()} events, {len(g)} markets, dates {pd.Timestamp(dates[0]).date()}..{pd.Timestamp(dates[-1]).date()}; "
              f"SEALED starts {pd.Timestamp(cut).date()}")
    d.to_parquet(D / "decisions.parquet", index=False)
    return 0


# ------------------------------------------------------------------------------------------------ quotes
HIST_CUTOFF = None


def _q(c, side):
    v = c.get(side) or {}
    x = v.get("close_dollars", v.get("close"))
    return float(x) if x is not None else None


SAMPLE_SEED = 1616
SAMPLE_PER_TRIAL = 750  # 4 trials -> 3,000 markets (coordinator cap 10-29: sustained 429s on our IP)
GAP = 1.0               # 1 req/s single thread (429s persisted at 3 req/s on the shared IP)


def cmd_sample() -> int:
    """Outcome-free, quote-free seeded draw. T1: any market; T2-T4: only markets the rule could ever bet
    (model longshot prob <= 0.10). Per trial 750, split across series by largest remainder of eligible counts."""
    d = pd.read_parquet(D / "decisions.parquet").drop_duplicates("ticker")
    d["p_long"] = np.minimum(d.p_yes, 1 - d.p_yes)
    elig = d[(d.trial == "T1_GAS_W") | (d.p_long <= 0.10)]
    rng = np.random.default_rng(SAMPLE_SEED)
    picks = []
    for tr in TRIALS:
        g = elig[elig.trial == tr]
        cnt = g.series.value_counts().sort_index()
        n = min(SAMPLE_PER_TRIAL, len(g))
        raw = cnt / cnt.sum() * n
        q = np.floor(raw).astype(int)
        for s in (raw - q).sort_values(ascending=False).index[: n - q.sum()]:
            q[s] += 1
        for s, k in q.items():
            gs = g[g.series == s].sort_values("ticker")
            picks.append(gs.iloc[np.sort(rng.choice(len(gs), int(min(k, len(gs))), replace=False))])
        print(f"{tr}: eligible {len(g)} -> sampled {n} {q.to_dict()}")
    smp = pd.concat(picks)
    smp[["trial", "series", "ticker"]].to_parquet(D / "sample.parquet", index=False)
    print(f"sample {len(smp)} markets")
    return 0


def _get(url: str):
    import time
    import requests
    from sports.http import CACHE_DIR, HEADERS, _key
    path = CACHE_DIR / (_key(url, None) + ".json")
    if path.exists():
        return json.loads(path.read_text()), True
    for a in range(10):
        time.sleep(GAP)
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
        except requests.RequestException:
            time.sleep(min(5 * 2 ** a, 60))
            continue
        if r.status_code in (400, 404):
            return None, False
        if r.status_code == 429 or r.status_code >= 500:
            w = min(5 * 2 ** a, 60)
            print(f"  http {r.status_code}, backoff {w}s", flush=True)
            time.sleep(w)
            continue
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
        return r.json(), False
    return None, False


def cmd_quotes() -> int:
    smp = pd.read_parquet(D / "sample.parquet")
    d = pd.read_parquet(D / "decisions.parquet").drop_duplicates("ticker")
    d = d[d.ticker.isin(smp.ticker)].reset_index(drop=True)
    late_cut = int(pd.Timestamp("2026-08-25", tz="UTC").timestamp())
    rows, hits, n429 = [], 0, 0
    for i, r in enumerate(d.itertuples()):
        st = max(int(r.open_ts), int(r.t) - 72 * 3600)
        q = f"?start_ts={st}&end_ts={int(r.t)}&period_interval=60"
        uh = f"{API}/historical/markets/{r.ticker}/candlesticks{q}"
        ul = f"{API}/series/{r.series}/markets/{r.ticker}/candlesticks{q}"
        res = {}
        if st >= int(r.t):  # rung listed after the decision time: it did not exist -> no quote
            rows.append({"ticker": r.ticker, "q_ts": np.nan, "yes_bid": np.nan, "yes_ask": np.nan})
            continue
        for u in ((ul, uh) if r.close_ts >= late_cut else (uh, ul)):
            x, hit = _get(u)
            hits += hit
            if x is not None:
                res = x
                break
        cs = [c for c in res.get("candlesticks", []) if c["end_period_ts"] <= r.t]
        if cs:
            last = cs[-1]
            rows.append({"ticker": r.ticker, "q_ts": int(last["end_period_ts"]), "yes_bid": _q(last, "yes_bid"), "yes_ask": _q(last, "yes_ask")})
        else:
            rows.append({"ticker": r.ticker, "q_ts": np.nan, "yes_bid": np.nan, "yes_ask": np.nan})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(d)} (cache hits {hits})", flush=True)
    qq = pd.DataFrame(rows)
    qq.to_parquet(D / "quotes.parquet", index=False)
    print(f"quotes: {len(qq)} markets, with a candle {qq.q_ts.notna().mean():.3f}, cache hits {hits}")
    return 0


# ------------------------------------------------------------------------------------------------ bets
def fee(p):
    return TAKER * p * (1 - p)


def joined() -> pd.DataFrame:
    d = pd.read_parquet(D / "decisions.parquet").merge(pd.read_parquet(D / "quotes.parquet"), on="ticker", how="left")
    assert (d.q_ts.isna() | (d.q_ts <= d.t)).all()
    d["spread"] = d.yes_ask - d.yes_bid
    d["no_ask"] = 1 - d.yes_bid
    return d


def bets(d: pd.DataFrame, trial: str, dist: str = "t4") -> pd.DataFrame:
    g = d[d.trial == trial].copy()
    g = g[g.yes_bid.notna() & g.yes_ask.notna()]
    py = g.p_yes.to_numpy()
    ok_yes = (g.yes_ask >= 0.01) & (g.yes_ask <= 0.99)
    ok_no = (g.no_ask >= 0.01) & (g.no_ask <= 0.99)
    e_yes = py - g.yes_ask - fee(g.yes_ask)
    e_no = (1 - py) - g.no_ask - fee(g.no_ask)
    if trial == "T1_GAS_W":
        sp = g.spread <= 0.05
        by = ok_yes & sp & (e_yes >= 0.05) & (e_yes >= e_no)
        bn = ok_no & sp & (e_no >= 0.05) & ~by
    else:
        sp = g.spread <= 0.05
        by = ok_yes & sp & (1 - py <= 0.10) & (e_yes >= 0.03)   # NO is the longshot -> buy YES (favourite)
        bn = ok_no & sp & (py <= 0.10) & (e_no >= 0.03)         # YES is the longshot -> buy NO
    a = g[by].assign(side="yes", ask=g.yes_ask[by], edge=e_yes[by], p_side=py[by.to_numpy()])
    b = g[bn].assign(side="no", ask=g.no_ask[bn], edge=e_no[bn], p_side=1 - py[bn.to_numpy()])
    out = pd.concat([a, b])
    out["cost"] = out.ask + fee(out.ask)
    return out


def boot(b: pd.DataFrame, cl: str, n=10_000, seed=16016):
    b = b.assign(pnl=np.where(b.won, 1 / b.cost - 1, -1.0))
    agg = b.groupby(cl).agg(p=("pnl", "sum"), s=("pnl", "size"))
    P, S = agg.p.to_numpy(), agg.s.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(P), (n, len(P)))
    tp, ts = P[idx].sum(1), S[idx].sum(1)
    roi = tp / ts
    return {"bets": int(len(b)), "clusters": int(len(P)), "units": float(b.pnl.sum()), "roi": float(b.pnl.mean()),
            "ci_lo": float(np.quantile(roi, 0.025)), "ci_hi": float(np.quantile(roi, 0.975)), "p": float(np.mean(tp <= 0)),
            "win_rate": float(b.won.mean()), "avg_cost": float(b.cost.mean())}


def cmd_avail() -> int:
    d = joined()
    d["p_long"] = np.minimum(d.p_yes, 1 - d.p_yes)
    d["band"] = pd.cut(d.p_long, [-1, 0.001, 0.01, 0.05, 0.10, 0.25, 0.51], labels=["<.001", ".001-.01", ".01-.05", ".05-.10", ".10-.25", ".25-.50"])
    d["has"] = d.yes_ask.notna()
    d["two"] = d.has & (d.yes_bid > 0) & (d.yes_ask < 1)
    d["tight"] = d.two & (d.spread <= 0.05)
    for tr, g in d.groupby("trial"):
        print(f"\n== {tr} ({g.event_ticker.nunique()} events, {len(g)} markets) -- quote availability at decision time, by model longshot prob")
        t = g.groupby("band", observed=False).agg(markets=("ticker", "size"), candle=("has", "mean"), two_sided=("two", "mean"),
                                                  spread_le_5c=("tight", "mean"), med_spread=("spread", "median"),
                                                  med_longshot_bid=("yes_bid", "median"))
        print(t.round(3).to_string())
        for sp in ("dev", "sealed"):
            n = len(bets(d[d.split == sp], tr))
            print(f"  candidate bets ({sp}, rule applied, no outcomes): {n}")
    return 0


# ------------------------------------------------------------------------------------------------ score
def cmd_score() -> int:
    which = sys.argv[2]
    assert which in ("dev", "sealed")
    d = joined()
    d = d[d.split == which]
    oc = pd.read_parquet(D / "outcomes.parquet")[["ticker", "result", "expiration_value"]]
    res = []
    for tr in TRIALS:
        b = bets(d, tr).merge(oc, on="ticker", how="left")
        b["won"] = b.result == b.side
        b["week"] = pd.to_datetime(b.settle_date).dt.to_period("W").astype(str)
        cl = "week" if tr == "T4_WEEKLY_FADE" else "settle_date"
        b.to_csv(D / f"bets_{which}_{tr}.csv", index=False)
        if not len(b):
            r = {"bets": 0, "roi": 0.0, "p": 1.0}
            halves = (np.nan, np.nan)
        else:
            r = boot(b, cl)
            dates = np.sort(b.settle_date.unique())
            mid = dates[len(dates) // 2]
            b["pnl"] = np.where(b.won, 1 / b.cost - 1, -1.0)
            halves = (float(b[b.settle_date < mid].pnl.mean()), float(b[b.settle_date >= mid].pnl.mean()))
        if which == "dev":
            passed = bool(r["bets"] >= 100 and r["p"] < BAR and r["roi"] > 0 and all(h > 0 for h in halves))
        else:
            passed = bool(r["bets"] > 0 and r["roi"] > 0 and r["p"] < 0.05)
        r.update({"trial": tr, "halves_roi": halves, "passed": passed})
        res.append(r)
        print(f"{which} {tr}: bets {r['bets']}, ROI {r['roi']:+.4f}, CI [{r.get('ci_lo', np.nan):+.4f}, {r.get('ci_hi', np.nan):+.4f}], "
              f"p {r['p']:.4f}, clusters {r.get('clusters')}, win {r.get('win_rate', np.nan):.3f}, avg cost {r.get('avg_cost', np.nan):.3f}, "
              f"halves {np.round(halves, 4).tolist()} -> {'PASS' if passed else 'FAIL'}", flush=True)
        # descriptive, not a trial: Brier of model vs Kalshi mid on every two-sided quoted market
        a = d[(d.trial == tr) & (d.yes_bid > 0) & (d.yes_ask < 1)].merge(oc, on="ticker", how="left")
        if len(a):
            y = (a.result == "yes").astype(float)
            a["dm"] = (a.p_yes - y) ** 2 - ((a.yes_bid + a.yes_ask) / 2 - y) ** 2
            cl2 = pd.to_datetime(a.settle_date).dt.to_period("W").astype(str) if tr == "T4_WEEKLY_FADE" else a.settle_date
            per = a.groupby(cl2).dm.agg(["sum", "size"])
            rng = np.random.default_rng(1)
            ix = rng.integers(0, len(per), (5000, len(per)))
            bd = per["sum"].to_numpy()[ix].sum(1) / per["size"].to_numpy()[ix].sum(1)
            print(f"   Brier model - mid on {len(a)} two-sided markets: {a.dm.mean():+.5f} [{np.quantile(bd, .025):+.5f}, {np.quantile(bd, .975):+.5f}] "
                  f"(model {((a.p_yes - y) ** 2).mean():.5f}, mid {(((a.yes_bid + a.yes_ask) / 2 - y) ** 2).mean():.5f}; negative = model better)")
        if len(b):
            print("   by series/side:", b.groupby(["series", "side"]).agg(n=("pnl", "size"), roi=("pnl", "mean"), win=("won", "mean"), cost=("cost", "mean")).round(3).to_dict("index"))
    if "--ledger" in sys.argv:
        prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
        n0 = sum(1 for x in prior if x.get("dataset") == DATASET and x.get("status") == "ok")
        with LEDGER.open("a") as f:
            for i, r in enumerate(res):
                sha = hashlib.sha256(f"q016:{r['trial']}".encode()).hexdigest()[:12]
                f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"q016_{r['trial']}_{which}",
                                    "sha": sha, "dataset": DATASET, "hypothesis": "pre-registered in notes/q016_commodities.md",
                                    "source": "Q016 brief 10-29", "status": "ok", "split": which.upper(), "trial_number": n0 + i + 1,
                                    "study_trials": K_TRIALS, "bonferroni_bar": BAR, **r}, default=str) + "\n")
        print("appended to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    f = {"markets": cmd_markets, "ref": cmd_ref, "fitgas": cmd_fitgas, "decide": cmd_decide, "sample": cmd_sample, "quotes": cmd_quotes,
         "avail": cmd_avail, "score": cmd_score}.get(cmd)
    raise SystemExit(f() if f else (print(__doc__) or 1))
