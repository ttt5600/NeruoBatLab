"""L19 K-FAV: Kalshi favourite side (88-97c) in NON-SPORTS daily categories, taker (F1) and maker (F2). DEV only.

    python scripts/kalshi_fav_maker.py census          # settled markets for every candidate series (no outcomes printed)
    python scripts/kalshi_fav_maker.py build           # hourly candles close-6h..close, DEV markets only (cached)
    python scripts/kalshi_fav_maker.py describe        # counts/liquidity by category at decision time, NO outcomes
    python scripts/kalshi_fav_maker.py dev [--ledger]  # score F1/F2 on DEV

Source: Burgi, Deng & Whelan (2025) -- makers out-earn takers on Kalshi; contracts > 70c have small positive
post-fee returns (notes/prior_research_2026-10-22.md idea #1). Our own symmetric ladder-maker test died on adverse
selection (EDGES_AND_AGENTS.md DEAD), so F2 reports conditional-on-fill outcomes explicitly.

PRE-REGISTRATION (written 2026-10-22 before any price or outcome of these markets was looked at)
------------------------------------------------------------------------------------------------
Universe. Kalshi series with frequency == "daily" in every category except Sports, MINUS all daily-HIGH temperature
  series (KXHIGH*, HIGH*, *HIGH e.g. KXDENHIGH/KXHOUHIGH/KXDVHIGH, HIGHUS, HIGHNY0, KXCITIESWEATHER -- already traded
  live) and test series (ticker contains TEST). Settled markets with result yes/no.
  One event per series per ET calendar day: series whose events close at several times a day (e.g. hourly crypto
  KXBTCD) keep only events whose close time-of-day (America/New_York) equals that series' modal close time-of-day
  computed over the series' DEV markets; ties broken by the LATEST time-of-day. The same rule drops markets that
  closed early (close_time-of-day != the series' scheduled one) -> no look-ahead in when we decide.
Categories reported: Financials, Commodities, Economics, Crypto, Weather (Climate and Weather + World rain/other
  weather), Other (Entertainment, Politics, Transportation, Health, Science, World non-weather).
Decision time T = close_time - 3 h. Quote = the last hourly candle with end_period_ts <= T (yes_bid, yes_ask close).
  Eligible quote: 0 < yes_bid < yes_ask < 1 and yes_ask - yes_bid <= 0.05.
Favoured side: YES if yes_ask in [0.88, 0.97]; NO if no_ask = 1 - yes_bid in [0.88, 0.97] (cannot both hold under
  the spread filter). Every qualifying market counts (rungs of one ladder are clustered in the bootstrap).
F1 taker: buy 1 contract of the favoured side at its ask a; cost a + taker fee 0.07*a*(1-a) (formula, i.e. a large
  order; the 1-contract ceil-to-cent fee is reported as a sensitivity, not scored).
F2 maker: rest a bid at b = a - 0.01 for the favoured side at T, good until close.
  FILL RULE: filled iff some trade in a candle ending in (T, close] printed at or below b on that side
  (YES side: min YES trade price <= b; NO side: max YES trade price >= 1 - b). Unfilled = no bet.
  Cost b + maker fee 0.0175*b*(1-b) applied to ALL dates (conservative; Kalshi maker fees were not general before
  2025, a zero-fee variant is reported). Sensitivity (not scored): strict trade-through (< b instead of <= b).
  Adverse selection: report win rate and ROI of F2 conditional on fill vs the same markets' unconditional F1-at-b
  outcome (would-have-won rate among unfilled).
Payoff 1 if the favoured side settles true. Unit = 1 contract; ROI = sum P&L / sum cost.
Split: DEV = close_time < 2025-07-01 UTC; SEALED = close_time >= 2025-07-01 (never built nor scored here).
Inference: one-sided bootstrap (3000 draws), clusters = ET calendar DAY of close (all series/rungs that day together;
  coarser than event, covers cross-series correlation on market-wide days). p = share of draws with P&L <= 0.
Trials (dataset kalshi_fav_maker, new): F1_taker_fav8897, F2_maker_fav8897 -> 2 trials, bar p < 0.05/2 = 0.025
  (Bonferroni over the dataset's trial count at scoring time). Per-category numbers are DESCRIPTIVE ONLY; a category
  selected from them would be a new post-hoc trial.
PASS (dev) requires ALL of: ROI > 0; p < bar; worst calendar quarter P&L >= -3 x mean quarterly P&L (a 90c buy loses
  ~9x what it wins, so one bad quarter must not erase > 3 average quarters); >= 60% of quarters positive.
AMENDMENT 1 (2026-10-22, after the outcome-blind census showed Kalshi files its main crypto daily and index
  above/below contracts under frequency "hourly"; still before any price or P&L was looked at): also include
  KXBTCD, KXETHD, KXBTC, KXETH (Crypto; only the 17:00 ET daily-close event) and KXINXU, KXNASDAQ100U (Financials;
  only the 16:00 ET event). Fixed time-of-day replaces the modal rule for these six.
AMENDMENT 2 (2026-10-22, build-cost only, before any decision-time price or P&L was examined): archived markets
  need one request each (~136k at ~4 req/s = 9 h). (a) Drop the crypto RANGE ladders KXBTC, KXETH, KXSHIBA: same
  underlying and same close as their above/below twins KXBTCD, KXETHD, KXSHIBAD (one instrument per underlying-day).
  (b) Above/below ladders with > 16 numeric strikes (KXBTCD, KXETHD, KXSHIBAD, KXINXU, KXNASDAQ100U): YES price is
  monotone in strike, so locate the rungs with decision-time mid in (0.02, 0.98) by bisection on DECISION-TIME
  quotes only, and fetch every rung in that interval (+1 each side). If any probed rung lacks a decision quote, the
  whole event is fetched. Rungs outside the interval cannot be in the 88-97c band except through non-monotone
  quotes (reported).
If a trial passes: ask the user for one sealed score (2025-07-01..), same rule, frozen, bar 0.05/#promoted.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sports.http import fetch as _fetch, fetch_many as _fm
import requests


def fetch(url, params=None, **kw):
    """sports.http.fetch, but a 404 (market id unknown to the archive) returns {} instead of raising."""
    try:
        return _fetch(url, params, **kw)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return {}
        raise


def fetch_many(urls, workers=4, **kw):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda u: fetch(u, **kw), urls))

API = "https://api.elections.kalshi.com/trade-api/v2"
MKTS = ROOT / "data" / "kalshi_fav_markets.parquet"
CANDLES = ROOT / "data" / "kalshi_fav_candles_dev.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "kalshi_fav_maker"
SPLIT = pd.Timestamp("2025-07-01", tz="UTC")
TAKER, MAKER = 0.07, 0.0175
LO, HI = 0.88, 0.97
DEC_H = 3
ET = "America/New_York"


def _num(v):
    return pd.to_numeric(v, errors="coerce")


def is_high_temp(tk: str) -> bool:
    return (tk.startswith("KXHIGH") or tk.startswith("HIGH") or tk.endswith("HIGH")
            or tk in ("KXCITIESWEATHER",))


def bucket(cat: str, title: str) -> str:
    t = title.lower()
    if cat == "Climate and Weather" or (cat == "World" and any(w in t for w in ("rain", "temperature", "pollution"))):
        return "Weather"
    if cat in ("Financials", "Commodities", "Economics", "Crypto"):
        return cat
    return "Other"


EXTRA_TOD = {"KXBTCD": "17:00", "KXETHD": "17:00", "KXBTC": "17:00", "KXETH": "17:00",
             "KXINXU": "16:00", "KXNASDAQ100U": "16:00"}  # amendment 1


def candidate_series() -> pd.DataFrame:
    d = fetch(f"{API}/series", {}, cache=False)
    rows = [{"series": s["ticker"], "category": s.get("category"), "title": s.get("title", "")}
            for s in d["series"] if ((s.get("frequency") == "daily" and s.get("category") != "Sports"
            and not is_high_temp(s["ticker"]) and "TEST" not in s["ticker"]) or s["ticker"] in EXTRA_TOD)]
    df = pd.DataFrame(rows)
    df["bucket"] = [bucket(c, t) for c, t in zip(df.category, df.title)]
    return df


def settled(series: str) -> list[dict]:
    rows = []
    for path, extra, cache in ((f"{API}/historical/markets", {}, True), (f"{API}/markets", {"status": "settled"}, False)):
        cursor = None
        while True:
            p = {"series_ticker": series, "limit": 1000, **extra}
            if cursor:
                p["cursor"] = cursor
            d = fetch(path, p, cache=cache)
            rows += d.get("markets", [])
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                break
    return rows


DROP_RANGE = {"KXBTC", "KXETH", "KXSHIBA"}  # amendment 2a
BISECT = {"KXBTCD", "KXETHD", "KXSHIBAD", "KXINXU", "KXNASDAQ100U"}  # amendment 2b
EXTRA_FMT = {"17:00": "%y%b%d17", "16:00": "%y%b%dH1600"}


def extra_markets(ser: pd.DataFrame) -> pd.DataFrame:
    """Daily-close events of the six hourly-listed series, DEV window only (they start 2024-11 / 2025-03)."""
    rows = []
    meta = ser.set_index("series")
    for sname, tod in EXTRA_TOD.items():
        days = pd.date_range("2024-06-01", SPLIT.tz_localize(None) - pd.Timedelta(days=1), freq="D")
        if tod == "16:00":
            days = days[days.weekday < 5]
        evs = [f"{sname}-" + d.strftime(EXTRA_FMT[tod]).upper() for d in days]
        urls = [f"{API}/historical/markets?event_ticker={e}&limit=1000" for e in evs]
        n = 0
        for d in fetch_many(urls):
            for x in d.get("markets", []):
                if x.get("result") in ("yes", "no") and x.get("close_time"):
                    rows.append({"series": sname, "category": meta.loc[sname, "category"], "bucket": meta.loc[sname, "bucket"],
                                 "ticker": x["ticker"], "event_ticker": x["event_ticker"],
                                 "close_time": pd.to_datetime(x["close_time"], utc=True, format="ISO8601"),
                                 "volume": _num(x.get("volume_fp", x.get("volume"))), "result": x["result"]})
                    n += 1
        print(f"  {sname}: {n} daily-close markets", flush=True)
    return pd.DataFrame(rows)


def cmd_census() -> int:
    ser = candidate_series()
    print(f"{len(ser)} candidate daily non-sports series (daily-high temp excluded)")
    frames, have = [], set()
    if MKTS.exists() and "--fresh" not in sys.argv:
        old = pd.read_parquet(MKTS)
        frames.append(old)
        have = set(old.series)
    for i, r in enumerate(ser.itertuples()):
        if r.series in EXTRA_TOD:  # hourly-listed series: fetch only the daily-close events by ticker (amendment 1)
            continue
        if r.series in have:
            continue
        for attempt in range(5):
            try:
                m = settled(r.series)
                break
            except Exception as e:  # 429 bursts: back off and retry the series
                print(f"  {r.series}: {e}; retry", flush=True)
                import time
                time.sleep(60)
        else:
            raise RuntimeError(f"census failed on {r.series}")
        if not m:
            continue
        df = pd.DataFrame(m).drop_duplicates("ticker")
        df = df[df["result"].isin(["yes", "no"]) & df["close_time"].notna()]
        if df.empty:
            continue
        vol = _num(df["volume_fp"]) if "volume_fp" in df else _num(df.get("volume"))
        frames.append(pd.DataFrame({"series": r.series, "category": r.category, "bucket": r.bucket,
                                    "ticker": df.ticker, "event_ticker": df.event_ticker,
                                    "close_time": pd.to_datetime(df.close_time, utc=True, format="ISO8601"),
                                    "volume": vol, "result": df.result}))
        if i % 20 == 0:
            print(f"  {i}/{len(ser)} {r.series}: {len(df)}", flush=True)
    frames = [f[~f.series.isin(list(EXTRA_TOD))] for f in frames]
    frames.append(extra_markets(ser))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(MKTS, index=False)
    out["period"] = np.where(out.close_time < SPLIT, "dev", "sealed")
    print(out.groupby(["bucket", "period"]).agg(series=("series", "nunique"), events=("event_ticker", "nunique"),
                                                markets=("ticker", "size"), med_vol=("volume", "median")).to_string())
    return 0


def daily_events(m: pd.DataFrame) -> pd.DataFrame:
    """Apply the one-event-per-series-per-day / scheduled-close rule (uses DEV markets only for the mode)."""
    m = m.copy()
    et = m.close_time.dt.tz_convert(ET)
    m["tod"] = et.dt.strftime("%H:%M")
    m["et_day"] = et.dt.date
    dev = m[m.close_time < SPLIT]
    cnt = dev.drop_duplicates("event_ticker").groupby(["series", "tod"]).size().reset_index(name="n")
    cnt = cnt.sort_values(["series", "n", "tod"], ascending=[True, False, False])
    mode = cnt.drop_duplicates("series").set_index("series")["tod"]
    mode = pd.concat([mode.drop([k for k in EXTRA_TOD if k in mode.index]), pd.Series(EXTRA_TOD)])
    return m[m.tod == m.series.map(mode)]


def _parse(tk, candles, close_ts):
    t_dec = close_ts - DEC_H * 3600
    pre = [c for c in candles if c["end_period_ts"] <= t_dec]
    win = [c for c in candles if t_dec < c["end_period_ts"] <= close_ts]
    if not pre:
        return None
    last = pre[-1]
    g = lambda c, side, f: _num((c.get(side) or {}).get(f + "_dollars", (c.get(side) or {}).get(f)))
    lows = [x for x in (g(c, "price", "low") for c in win) if pd.notna(x)]
    highs = [x for x in (g(c, "price", "high") for c in win) if pd.notna(x)]
    vp = lambda c: _num(c.get("volume_fp", c.get("volume")))
    return {"ticker": tk, "yes_bid": g(last, "yes_bid", "close"), "yes_ask": g(last, "yes_ask", "close"),
            "vol_pre": float(np.nansum([vp(c) for c in pre])), "vol_win": float(np.nansum([vp(c) for c in win])),
            "min_trade_after": min(lows) if lows else np.nan, "max_trade_after": max(highs) if highs else np.nan}


def _bisect_event(ev: pd.DataFrame, get) -> list[dict]:
    """Amendment 2b. Band candidates need yes_ask <= 0.97 (YES side) or yes_bid >= 0.03 (NO side), i.e. all lie in
    the strike interval between 'ask >= 0.98' rungs and 'bid <= 0.02' rungs. Bisect on decision-time quotes only."""
    ev = ev.assign(strike=pd.to_numeric(ev.ticker.str.extract(r"-T([0-9.]+)$")[0], errors="coerce"))
    if ev.strike.isna().any() or len(ev) <= 16:
        return [r for r in (get(tk, t) for tk, t in zip(ev.ticker, ev.close_ts)) if r]
    ev = ev.sort_values("strike")
    tks, ts = list(ev.ticker), list(ev.close_ts)
    n = len(tks)
    cache: dict[int, dict | None] = {}

    def q(i):
        if i not in cache:
            cache[i] = get(tks[i], ts[i])
        r = cache[i]
        return None if (r is None or pd.isna(r["yes_bid"]) or pd.isna(r["yes_ask"])) else r

    def near(i):  # nearest quoted rung within +-3 (decision-time quote only)
        for d in (0, 1, -1, 2, -2, 3, -3):
            if 0 <= i + d < n and q(i + d) is not None:
                return q(i + d)
        raise KeyError

    try:
        sign = 1 if near(0)["yes_ask"] >= near(n - 1)["yes_ask"] else -1  # +1: YES price falls as strike rises
        pos = (lambda i: i) if sign == 1 else (lambda i: n - 1 - i)
        lo, hi = 0, n  # first (oriented) idx with ask < 0.98
        while lo < hi:
            k = (lo + hi) // 2
            lo, hi = (k + 1, hi) if near(pos(k))["yes_ask"] >= 0.98 else (lo, k)
        a = lo
        lo, hi = a, n  # first idx >= a with bid <= 0.02
        while lo < hi:
            k = (lo + hi) // 2
            lo, hi = (k + 1, hi) if near(pos(k))["yes_bid"] > 0.02 else (lo, k)
        for k in range(max(a - 1, 0), min(lo + 1, n - 1) + 1):
            q(pos(k))
    except KeyError:
        for i in range(n):
            q(i)
    return [r for r in cache.values() if r]


def cmd_build() -> int:
    m = pd.read_parquet(MKTS)
    m = daily_events(m[m.close_time < SPLIT])  # DEV only; sealed candles are never fetched here
    m["close_ts"] = ((m.close_time - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype(int)
    print(f"DEV daily-event markets {len(m)} / events {m.event_ticker.nunique()} / series {m.series.nunique()}")
    # All DEV markets are archived: the event candlestick endpoint returns nothing for them and the batch
    # /markets/candlesticks endpoint does not serve history, so go straight to per-market historical candles.
    import sports.http as H
    H.MIN_INTERVAL = 0.15
    url = lambda tk, t: (f"{API}/historical/markets/{tk}/candlesticks?start_ts={t - 6 * 3600}&end_ts={t}"
                         f"&period_interval=60")
    get = lambda tk, t: _parse(tk, fetch(url(tk, t), retries=12).get("candlesticks", []), t)
    m = m[~m.series.isin(DROP_RANGE)]
    rows = []
    for series, mk in m.groupby("series"):
        n0 = len(rows)
        if series not in BISECT:
            ds = fetch_many([url(tk, t) for tk, t in zip(mk.ticker, mk.close_ts)], workers=6, retries=12)
            rows += [r for r in (_parse(tk, d.get("candlesticks", []), t)
                                 for (tk, t), d in zip(zip(mk.ticker, mk.close_ts), ds)) if r]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(6) as ex:
                for res in ex.map(lambda e: _bisect_event(e[1], get), mk.groupby("event_ticker")):
                    rows += res
        print(f"{series}: {len(mk)} markets, {len(rows) - n0} with a decision-time candle", flush=True)
    c = pd.DataFrame(rows).drop_duplicates("ticker")
    out = m.merge(c, on="ticker", how="inner")
    out.to_parquet(CANDLES, index=False)
    print(f"-> {CANDLES} ({len(out)} markets)")
    return 0


def frame() -> pd.DataFrame:
    d = pd.read_parquet(CANDLES)
    d = d[d.close_time < SPLIT]
    ok = d.yes_bid.notna() & d.yes_ask.notna() & (d.yes_bid > 0) & (d.yes_ask < 1) & (d.yes_ask > d.yes_bid) \
        & (d.yes_ask - d.yes_bid <= 0.05 + 1e-9)
    d = d[ok].copy()
    yes = d.yes_ask.between(LO - 1e-9, HI + 1e-9)
    no = (1 - d.yes_bid).between(LO - 1e-9, HI + 1e-9)
    d = d[yes | no].copy()
    d["side"] = np.where(yes[yes | no], "yes", "no")
    d["ask"] = np.where(d.side == "yes", d.yes_ask, 1 - d.yes_bid).round(4)
    d["bid"] = (d.ask - 0.01).round(4)
    # fills (prices are YES trade prices)
    d["fill"] = np.where(d.side == "yes", d.min_trade_after <= d.bid + 1e-9, d.max_trade_after >= 1 - d.bid - 1e-9)
    d["fill_strict"] = np.where(d.side == "yes", d.min_trade_after < d.bid - 1e-9, d.max_trade_after > 1 - d.bid + 1e-9)
    return d


def cmd_describe() -> int:
    m = pd.read_parquet(CANDLES)
    m = m[m.close_time < SPLIT]
    print("DEV markets with a decision-time candle, by bucket (no outcomes):")
    m["quoted"] = m.yes_bid.notna() & m.yes_ask.notna() & (m.yes_bid > 0) & (m.yes_ask < 1)
    m["spread"] = m.yes_ask - m.yes_bid
    print(m.groupby("bucket").agg(series=("series", "nunique"), events=("event_ticker", "nunique"), markets=("ticker", "size"),
                                  quoted=("quoted", "mean"), med_spread=("spread", "median"),
                                  med_vol_pre6h=("vol_pre", "median"), med_vol_win3h=("vol_win", "median"),
                                  first=("close_time", "min"), last=("close_time", "max")).to_string())
    d = frame()
    print("\nEligible favoured-side (88-97c ask, spread<=5c) markets, by bucket (no outcomes):")
    print(d.groupby("bucket").agg(markets=("ticker", "size"), days=("et_day", "nunique"), series=("series", "nunique"),
                                  share_no=("side", lambda s: (s == "no").mean()), mean_ask=("ask", "mean"),
                                  fill=("fill", "mean"), fill_strict=("fill_strict", "mean"),
                                  med_vol_win=("vol_win", "median")).to_string())
    print("\nTop series by eligible count:")
    print(d.groupby(["bucket", "series"]).size().sort_values(ascending=False).head(25).to_string())
    return 0


def score(t: pd.DataFrame, rng) -> dict:
    cl = t.groupby("cluster").agg(pnl=("pnl", "sum"))
    idx = rng.integers(0, len(cl), size=(3000, len(cl)))
    boots = cl.pnl.to_numpy()[idx].sum(axis=1)
    q = pd.PeriodIndex(pd.to_datetime(t.date), freq="Q").astype(str)
    per = t.groupby(q)["pnl"].sum()
    mean_q = per.mean()
    worst = per.min()
    roi = float(t.pnl.sum() / t.stake.sum()) if len(t) else 0.0
    p = float(np.mean(boots <= 0))
    return {"bets": int(len(t)), "clusters_days": int(len(cl)), "staked": float(t.stake.sum()), "units": float(t.pnl.sum()),
            "roi": roi, "ci_lo": float(np.quantile(boots, 0.025)), "ci_hi": float(np.quantile(boots, 0.975)), "p": p,
            "win_rate": float(t.won.mean()), "mean_cost": float(t.stake.mean()),
            "folds_positive": int((per > 0).sum()), "n_quarters": int(len(per)), "worst_quarter": float(worst),
            "mean_quarter": float(mean_q), "per_fold": per.round(2).to_dict()}


def trial(d: pd.DataFrame, name: str, maker_fee: float = MAKER, fill_col: str = "fill", one_contract: bool = False):
    won = (d.result == d.side).to_numpy()
    if name.startswith("F1"):
        s, px = d, d.ask.to_numpy()
        fee = np.ceil(TAKER * px * (1 - px) * 100 - 1e-9) / 100 if one_contract else TAKER * px * (1 - px)
    else:
        s = d[d[fill_col]]
        won = won[d[fill_col].to_numpy()]
        px = s.bid.to_numpy()
        fee = (np.ceil(maker_fee * px * (1 - px) * 100 - 1e-9) / 100 if one_contract else maker_fee * px * (1 - px))
    cost = px + fee
    return pd.DataFrame({"cluster": s.et_day.astype(str).to_numpy(), "bucket": s.bucket.to_numpy(), "series": s.series.to_numpy(),
                         "date": pd.to_datetime(s.et_day).to_numpy(), "pnl": np.where(won, 1 - cost, -cost), "stake": cost,
                         "won": won})


def verdict(r, bar):
    return (r["roi"] > 0 and r["p"] < bar and r["worst_quarter"] >= -3 * r["mean_quarter"]
            and r["folds_positive"] >= 0.6 * r["n_quarters"])


def cmd_dev() -> int:
    d = frame()
    print(f"DEV eligible {len(d)} markets, {d.et_day.nunique()} days, {d.close_time.min():%Y-%m-%d}..{d.close_time.max():%Y-%m-%d}")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    names = ["F1_taker_fav8897", "F2_maker_fav8897"]
    for n in names:
        shas.add(hashlib.sha256(("kalshi_fav_maker:" + n).encode()).hexdigest()[:12])
    bar = 0.05 / len(shas)
    recs = []
    for name in names:
        t = trial(d, name)
        r = score(t, np.random.default_rng(0))
        ok = verdict(r, bar)
        print(f"\n{name}: bets {r['bets']} days {r['clusters_days']} units {r['units']:+.2f} ROI {r['roi']:+.2%} "
              f"95%CI [{r['ci_lo']:+.1f},{r['ci_hi']:+.1f}] p {r['p']:.4f} win {r['win_rate']:.3f} cost {r['mean_cost']:.3f} "
              f"q+ {r['folds_positive']}/{r['n_quarters']} worstQ {r['worst_quarter']:+.2f} meanQ {r['mean_quarter']:+.2f} "
              f"-> {'PASS' if ok else 'FAIL'} (bar {bar:.4f})")
        print("  quarters:", r["per_fold"])
        by = t.groupby("bucket").apply(lambda g: f"{g.pnl.sum() / g.stake.sum():+.2%}/{len(g)} (win {g.won.mean():.3f}, "
                                                 f"cost {g.stake.mean():.3f})", include_groups=False)
        for k, v in by.items():
            print(f"    {k:12s} {v}")
        sha = hashlib.sha256(("kalshi_fav_maker:" + name).encode()).hexdigest()[:12]
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"kalshi_{name}", "sha": sha,
                     "dataset": DATASET, "hypothesis": f"pre-registered in scripts/kalshi_fav_maker.py ({name})",
                     "source": "Burgi, Deng & Whelan 2025 (L19 K-FAV)", "status": "ok", "trial_number": len(shas),
                     "bonferroni_bar": bar, "passed": bool(ok),
                     "by_bucket": {k: v for k, v in by.items()}, **r})
    # sensitivities (not trials)
    print("\nSensitivities (not trials):")
    for lab, kw, nm in (("F1 1-contract ceil fee", {"one_contract": True}, "F1"),
                        ("F2 zero maker fee", {"maker_fee": 0.0}, "F2"),
                        ("F2 strict trade-through fill", {"fill_col": "fill_strict"}, "F2"),
                        ("F2 1-contract ceil fee", {"one_contract": True}, "F2")):
        t = trial(d, nm, **kw)
        print(f"  {lab:32s} bets {len(t):6d} ROI {t.pnl.sum() / t.stake.sum():+.2%} units {t.pnl.sum():+.2f}")
    # adverse selection for F2
    won = d.result == d.side
    print("\nF2 adverse selection (favoured side wins):")
    print(f"  all eligible {won.mean():.4f} (n {len(d)}); filled {won[d.fill].mean():.4f} (n {int(d.fill.sum())}); "
          f"unfilled {won[~d.fill].mean():.4f} (n {int((~d.fill).sum())})")
    print(f"  mean bid among filled {d.bid[d.fill].mean():.4f}; breakeven win rate incl fee "
          f"{(d.bid[d.fill] + MAKER * d.bid[d.fill] * (1 - d.bid[d.fill])).mean():.4f}")
    for b, g in d.groupby("bucket"):
        w = g.result == g.side
        print(f"    {b:12s} filled win {w[g.fill].mean():.4f} (n {int(g.fill.sum())}) vs unfilled {w[~g.fill].mean():.4f} "
              f"(n {int((~g.fill).sum())})")
    print(f"\n{DATASET} trials {len(shas)} -> bar p < {bar:.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"appended {len(recs)} trials to {LEDGER}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"census": cmd_census, "build": cmd_build, "describe": cmd_describe,
                      "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
