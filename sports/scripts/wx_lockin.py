"""L26: Kalshi daily-high LATE-DAY LOCK-IN from the station's OBSERVED max so far. DEV only.

    python scripts/wx_lockin.py obs              # IEM ASOS METAR archive for the 24 stations, dev days only (no prices)
    python scripts/wx_lockin.py diag             # settlement-definition audit: observed max vs settled value (no prices)
    python scripts/wx_lockin.py build            # 17:00 hourly quotes + public trade tapes for candidate brackets
    python scripts/wx_lockin.py dev [--ledger]   # score the 6 trials below

Idea (lead L26): KXHIGH* brackets settle on the NWS climate report (CLI) daily max. By late afternoon the station has
usually peaked; its observed max so far M is (nearly) a lower bound on the official max. Brackets whose every YES
value is below M are (nearly) dead, so NO should trade at ~$1. Question: does the late-day price lag, and how often
does the rule lose (= the official max comes in BELOW a temperature the station already reported)?

PRE-REGISTERED 2026-10-25, written before any late-day price, tape or observation was pulled
------------------------------------------------------------------------------------------------
Cities : the 24 paper cities (kalshi_paper.PAPER_CITIES = orig 7 wx.CITIES + X1 17), station = the CLI station in the
         contract rules (KNYC Central Park, KMDW, ... as in those dicts). IEM ASOS id = station without the leading K.
Dev    : orig 7: event dates 2021-08-01..2025-06-30 (kalshi_temp dev quarters 2021Q3..2025Q2);
         X1 17 : event dates < 2026-03-01 (wx_expand SPLIT). Rows on or after those dates are dropped on load and
         their observations are never requested. Sealed for L26 = orig7 2025-07-01+ and X1 2026-03-01+, one shot,
         user's explicit yes only.
Climate day: NWS CLI day = local STANDARD time midnight..midnight. In DST that is 01:00..00:59 local clock time.
Decision T : 17:00 local CLOCK time on event day D (DST-aware; = 16:00 LST in summer). Price = close yes_bid/yes_ask
         of the last hourly candle ending <= T (window T-3h..T).
Observed max M: METARs (IEM report_type 3 routine + 4 specials) with valid time in [D 00:00 LST, T - 10 min]
         (10 min = real-time publication lag). Each ob -> whole F: from the RMK T-group (tenths C) as
         round_half_up(C*1.8+32); if no T-group, from the body whole-C temp as the conservative LOW end
         round_half_up((c-0.5)*1.8+32). M = max over obs. Need >= 12 obs in the window, else the station-day is skipped.
Bracket YES range (integers): between [floor, cap]; less: v <= cap-1; greater: v >= floor+1 (checked on data: values
         are integers, less cap=88 & value 88 -> NO, between 88-89 & value 88 -> YES).
Rules (each bracket judged independently, one bet = 1 unit of cost):
  BELOW_m : top YES value u <= M - 1 - m  (m=0: the bracket needs the official max below an already-observed temp;
            m=1: one more degree of slack). Between and less brackets only.
  ABOVE_k : bottom YES value l >= M + 1 + k with k=2 (needs a further rise of >= 3F after 17:00). Between/greater only.
Execution:
  TAKER : buy NO at the NO ask = 1 - yes_bid; requires yes_bid > 0 (a NO offer < $1 exists) and yes_ask - yes_bid <= 0.10.
          Fee = sports.live.kalshi_fee(p) = 0.07 p (1-p), unrounded (direct-member $0.0001 precision, as E1/E4).
          Sensitivity (not a trial): per-order cent round-up at 1 contract (fee = 1c whenever > 0).
  MAKER : post-only NO bid at the NO best bid q = 1 - yes_ask at T (requires yes_ask < 1 and q >= 0.01; q < NO ask);
          filled iff >= 10 contracts print at no_price <= q in (T, close] on the public tape (/historical/trades,
          fallback /markets/trades) -- the E4 N=10 queue proxy. Maker fee 0 (series fee_type quadratic, checked 10-22);
          sensitivity 0.0175 q(1-q). Unfilled = no bet.
Trials (6): BELOW_0_taker BELOW_1_taker ABOVE_2_taker BELOW_0_maker BELOW_1_maker ABOVE_2_maker; bar p < 0.05/6
         (dataset wx_lockin; the bar grows if more trials are ever added).
PRIMARY per trial: sum of pnl over bets (pnl = 1/cost - 1 if NO else -1, cost incl. fee); one-sided cluster
         bootstrap over calendar dates (all cities on a date together), 4000 draws, p = P(sum <= 0).
         PASS = p < bar AND sum > 0 AND both date-halves of dev (split at the median bet date) have sum > 0.
TRUTH  : Kalshi's `result` field (what actually paid). Rows whose result disagrees with expiration_value are KEPT
         (they are exactly the settlement risk) and reported.
Reported (not trials): n, losses, loss rate + Clopper-Pearson 95% CI vs break-even loss rate (1 - mean cost),
         ROI, by cohort/city/half; settlement audit (diag): distribution of official max - M_final and - M_17,
         count of days the official max < M_17 and why (no T-group, window, station, correction);
         capacity = contracts printed at no_price <= entry in (T, close] (a proxy: historical displayed depth is not
         available from any public endpoint), as contracts/day and $/day.

AMENDMENT 2026-10-25 (lead instruction, BEFORE any trial was scored; rules unchanged): the public trade-tape pull ran
at ~1,000 tickers per 30 min (4-5 h left) and was stopped after ~2,000 of 11,176 tickers. `tapes-cached` assembles
the tapes only for tickers whose every tape page is already in the HTTP cache (no network). The three MAKER trials
need complete tapes for their primary metric -> reported "NOT SCORED: data incomplete" (kept in the trial count, bar
stays 0.05/6; not logged as ok). Capacity and maker fill rates on the cached subset are DESCRIPTIVE only.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports import http as H
from sports import wx
from sports.http import fetch, fetch_many
from sports.live import kalshi_fee
from kalshi_paper import PAPER_CITIES
import wx_expand as X

DATA = ROOT / "data"
OBS = DATA / "wx_lockin_obs.parquet"
MK = DATA / "wx_lockin_markets.parquet"
QUOTES = DATA / "wx_lockin_quotes.parquet"
TAPES = DATA / "wx_lockin_trades.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "wx_lockin"
ORIG_START, ORIG_END = pd.Timestamp("2021-08-01"), pd.Timestamp("2025-07-01")  # dev = [start, end)
X1_END = X.SPLIT  # 2026-03-01
DEC_HOUR, LAG_MIN, MIN_OBS = 17, 10, 12
TRIALS = ["BELOW_0_taker", "BELOW_1_taker", "ABOVE_2_taker", "BELOW_0_maker", "BELOW_1_maker", "ABOVE_2_maker"]
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def _rhu(x):
    return np.floor(np.asarray(x, float) + 0.5)


def std_offset(tz: str, day: pd.Timestamp) -> timedelta:
    z = ZoneInfo(tz)
    dt = datetime(day.year, day.month, day.day, 12, tzinfo=z)
    return dt.utcoffset() - (dt.dst() or timedelta(0))


# ------------------------------------------------------------------------------------------------ markets (dev only)
def dev_markets() -> pd.DataFrame:
    orig = {s.replace("KXHIGH", "") for s in wx.CITIES}
    rows = []
    for f in sorted(DATA.glob("wx_*.parquet")):
        c = f.stem.replace("wx_", "")
        if c not in orig:
            continue
        d = pd.read_parquet(f, columns=["ticker", "event_ticker", "strike_type", "floor_strike", "cap_strike",
                                        "expiration_value", "result", "close_time", "date", "result_mismatch"])
        d["date"] = pd.to_datetime(d.date)
        d = d[(d.date >= ORIG_START) & (d.date < ORIG_END)]  # sealed rows dropped before anything else
        d["series"], d["cohort"] = "KXHIGH" + c, "E1"
        rows.append(d)
    x = pd.read_parquet(X.OUT, columns=["ticker", "event_ticker", "strike_type", "floor_strike", "cap_strike",
                                        "expiration_value", "result", "close_time", "date", "result_mismatch", "series", "kind"])
    x["date"] = pd.to_datetime(x.date)
    x = x[(x.date < X1_END) & (x.kind == "high") & x.series.isin(PAPER_CITIES)].drop(columns="kind")
    x["cohort"] = "X1"
    m = pd.concat(rows + [x], ignore_index=True).drop_duplicates("ticker")
    assert (m[m.cohort == "E1"].date < ORIG_END).all() and (m[m.cohort == "X1"].date < X1_END).all()
    st = m.strike_type
    m["u"] = np.where(st == "between", np.floor(m.cap_strike), np.where(st == "less", np.ceil(m.cap_strike) - 1, np.nan))
    m["l"] = np.where(st == "between", np.ceil(m.floor_strike), np.where(st == "greater", np.floor(m.floor_strike) + 1, np.nan))
    m["station"] = m.series.map(lambda s: PAPER_CITIES[s][0])
    m["tz"] = m.series.map(lambda s: PAPER_CITIES[s][1])
    return m


# ------------------------------------------------------------------------------------------------ observations
_TG = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")
_BODY = re.compile(r"\s(M?\d{2})/(M?\d{2})?\s")


def ob_f(metar: str) -> tuple[float, bool]:
    """-> (whole F, from_tgroup). NaN if no temperature."""
    g = _TG.search(metar or "")
    if g:
        c = (-1 if g.group(1) == "1" else 1) * int(g.group(2)) / 10
        return float(_rhu(c * 1.8 + 32)), True
    b = _BODY.search(" " + (metar or "") + " ")
    if b:
        t = b.group(1)
        c = -int(t[1:]) if t.startswith("M") else int(t)
        return float(_rhu((c - 0.5) * 1.8 + 32)), False
    return float("nan"), False


def cmd_obs() -> int:
    m = dev_markets()
    out = []
    for (station, tz), g in m.groupby(["station", "tz"]):
        d0, d1 = g.date.min() - pd.Timedelta(days=1), g.date.max() + pd.Timedelta(days=2)
        frames, s = [], d0
        while s < d1:
            e = min(s + pd.Timedelta(days=365), d1)
            raw = fetch(IEM, {"station": station[1:], "data": "metar", "year1": s.year, "month1": s.month, "day1": s.day,
                              "year2": e.year, "month2": e.month, "day2": e.day, "tz": "Etc/UTC", "format": "onlycomma",
                              "latlon": "no", "report_type": [3, 4]}, binary=True, cache=True)
            frames.append(pd.read_csv(io.BytesIO(raw)))
            s = e
        o = pd.concat(frames).drop_duplicates(["valid", "metar"])
        o["valid"] = pd.to_datetime(o.valid).dt.tz_localize("UTC")
        fv = o.metar.map(ob_f)
        o["tf"], o["tgroup"] = [x[0] for x in fv], [x[1] for x in fv]
        o["ghcnh"] = o.metar.str.contains("IEM_GHCNH", na=False)
        o["station"], o["tz"] = station, tz
        out.append(o[["station", "tz", "valid", "tf", "tgroup", "ghcnh", "metar"]].dropna(subset=["tf"]))
        print(f"{station}: {len(out[-1])} obs {o.valid.min().date()}..{o.valid.max().date()} "
              f"T-group {out[-1].tgroup.mean():.1%}", flush=True)
    pd.concat(out, ignore_index=True).to_parquet(OBS, index=False)
    return 0


def day_obs(m: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Per station-day: M_17 (obs to T-10min), M_final (full LST day), the local-midnight variant, counts, T."""
    days = m.drop_duplicates("event_ticker")[["event_ticker", "series", "cohort", "station", "tz", "date",
                                              "expiration_value", "close_time"]].copy()
    g = {k: v.sort_values("valid") for k, v in obs.groupby("station")}
    rows = []
    for r in days.itertuples(index=False):
        so = std_offset(r.tz, r.date)
        lst0 = pd.Timestamp(r.date).tz_localize("UTC") - so
        lst1 = lst0 + pd.Timedelta(days=1)
        T = pd.Timestamp(datetime(r.date.year, r.date.month, r.date.day, DEC_HOUR, tzinfo=ZoneInfo(r.tz)))
        loc0 = pd.Timestamp(datetime(r.date.year, r.date.month, r.date.day, tzinfo=ZoneInfo(r.tz)))
        o = g.get(r.station)
        if o is None:
            continue
        v = o.valid
        a = o[(v >= lst0) & (v <= T - pd.Timedelta(minutes=LAG_MIN))]
        f = o[(v >= lst0) & (v < lst1)]
        lm = o[(v >= loc0) & (v <= T - pd.Timedelta(minutes=LAG_MIN))]
        a_tg = a[a.tgroup]
        rows.append({"event_ticker": r.event_ticker, "T": int(T.timestamp()), "n17": len(a),
                     "M17": a.tf.max() if len(a) else np.nan, "M17_tg_only": a_tg.tf.max() if len(a_tg) else np.nan,
                     "M17_localmid": lm.tf.max() if len(lm) else np.nan,
                     "M17_from_nontg": bool(len(a) and (a.tf.max() > (a_tg.tf.max() if len(a_tg) else -999))),
                     "nfin": len(f), "Mfin": f.tf.max() if len(f) else np.nan,
                     "late_rise": (f[f.valid > T - pd.Timedelta(minutes=LAG_MIN)].tf.max() - a.tf.max()) if len(a) and len(f) else np.nan})
    return days.merge(pd.DataFrame(rows), on="event_ticker", how="inner")


def cmd_diag() -> int:
    m = dev_markets()
    obs = pd.read_parquet(OBS)
    d = day_obs(m, obs)
    d = d[d.n17 >= MIN_OBS]
    d["gap17"] = d.expiration_value - d.M17
    d["gapfin"] = d.expiration_value - d.Mfin
    print(f"DEV station-days with >= {MIN_OBS} obs by 17:00: {len(d)} "
          f"({d.cohort.value_counts().to_dict()}), {d.date.min().date()}..{d.date.max().date()}")
    print("official max - M_final (full LST day METAR max):", d.gapfin.value_counts().sort_index().to_dict())
    print("official max - M_17                            :", d.gap17.value_counts().sort_index().to_dict())
    below = d[d.gap17 < 0]
    print(f"days official < M_17: {len(below)} ({len(below) / len(d):.3%}); of which M_17 came from a non-T-group ob: "
          f"{int(below.M17_from_nontg.sum())}; official < M_17 by >= 2F: {int((d.gap17 <= -2).sum())}")
    lm = d.expiration_value < d.M17_localmid
    print(f"if the window started at LOCAL midnight instead of LST: official < M: {int(lm.sum())} days "
          f"(extra vs LST window {int((lm & ~(d.gap17 < 0)).sum())})")
    print("late rise after 17:00 (M_final - M_17) F:", d.late_rise.value_counts().sort_index().to_dict())
    print("by station, official<M17 / days:", {s: f"{int((g.gap17 < 0).sum())}/{len(g)}" for s, g in d.groupby("station")})
    mm = m[m.event_ticker.isin(d.event_ticker)]
    print(f"brackets whose Kalshi result disagrees with expiration_value: {int(mm.result_mismatch.sum())} of {len(mm)}")
    if len(below):
        cols = ["event_ticker", "station", "date", "expiration_value", "M17", "M17_tg_only", "Mfin", "M17_from_nontg"]
        print(below.sort_values("gap17")[cols].head(40).to_string(index=False))
    d.to_parquet(DATA / "wx_lockin_days.parquet", index=False)
    return 0


# ------------------------------------------------------------------------------------------------ candidates + prices
def candidates(m: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    c = m.merge(d[["event_ticker", "M17", "n17", "T", "late_rise"]], on="event_ticker", how="inner")
    c = c[c.n17 >= MIN_OBS]
    c["below_gap"] = c.M17 - 1 - c.u  # rule BELOW_m fires iff below_gap >= m
    c["above_gap"] = c.l - (c.M17 + 1)  # rule ABOVE_k fires iff above_gap >= k
    return c[(c.below_gap >= 0) | (c.above_gap >= 2)].copy()


def _patient(urls):
    import time
    out = []
    for u in urls:
        for i in range(30):
            try:
                out.append(fetch(u))
                break
            except Exception as ex:  # noqa: BLE001
                if "429" not in str(ex) and i > 3:
                    raise
                time.sleep(30)
    return out


def cmd_build() -> int:
    H.MIN_INTERVAL = 0.1  # Kalshi basic tier allows 20 reads/s; 10/s
    m = dev_markets()
    d = pd.read_parquet(DATA / "wx_lockin_days.parquet")
    c = candidates(m, d)
    print(f"candidate brackets: {len(c)} (BELOW_0 {int((c.below_gap >= 0).sum())}, ABOVE_2 {int((c.above_gap >= 2).sum())})", flush=True)
    rows = []
    # event endpoint first (one call per day), per-market /historical fallback for archived events
    ev = c.groupby(["series", "event_ticker"])["T"].first().reset_index()
    urls = [f"{wx.API}/series/{s}/events/{e}/candlesticks?start_ts={t - 3 * 3600}&end_ts={t}&period_interval=60"
            for s, e, t in ev.itertuples(index=False)]
    want, missing = set(c.ticker), []
    res = fetch_many(urls)
    for (s, e, t), dd in zip(ev.itertuples(index=False), res):
        tks = dd.get("market_tickers", [])
        if not tks:
            missing.append(e)
        for tk, cs in zip(tks, dd.get("market_candlesticks", [])):
            if tk in want:
                r = wx._row(tk, e, cs, t)
                if r:
                    rows.append(r)
    leg = c[c.event_ticker.isin(missing)]
    print(f"event endpoint: {len(rows)} quotes; per-market fallback for {len(leg)} brackets", flush=True)
    urls = [f"{wx.API}/historical/markets/{tk}/candlesticks?start_ts={t - 3 * 3600}&end_ts={t}&period_interval=60"
            for tk, t in zip(leg.ticker, leg["T"])]
    for i in range(0, len(urls), 2000):
        for (tk, e, t), dd in zip(list(zip(leg.ticker, leg.event_ticker, leg["T"]))[i:i + 2000], fetch_many(urls[i:i + 2000])):
            r = wx._row(tk, e, dd.get("candlesticks", []), t)
            if r:
                rows.append(r)
        print(f"  fallback {min(i + 2000, len(urls))}/{len(urls)}", flush=True)
    q = pd.DataFrame(rows)
    q.to_parquet(QUOTES, index=False)
    cq = c.merge(q[["ticker", "price_ts", "yes_bid", "yes_ask"]], on="ticker", how="inner")
    cq = cq[(cq.yes_bid > 0) | (cq.yes_ask < 1)]
    print(f"quotes: {len(q)}; with a NO bid or offer < $1: {len(cq)} -> tapes", flush=True)
    from kalshi_temp_maker import _tape
    close = pd.to_datetime(cq.close_time, utc=True).map(lambda x: int(x.timestamp()))
    tr = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, r in enumerate(ex.map(lambda a: _tape(*a), zip(cq.ticker, cq["T"], close))):
            tr += r
            if i % 1000 == 0:
                print(f"  tapes {i}/{len(cq)}", flush=True)
    pd.DataFrame(tr, columns=["ticker", "ts", "no_price", "count", "taker_side"]).to_parquet(TAPES, index=False)
    print(f"trades: {len(tr)}")
    return 0


def cmd_tapes_cached() -> int:
    """Assemble tapes from the HTTP cache only; a ticker counts only if all its pages are cached."""
    m = dev_markets()
    d = pd.read_parquet(DATA / "wx_lockin_days.parquet")
    c = candidates(m, d).merge(pd.read_parquet(QUOTES)[["ticker", "yes_bid", "yes_ask"]], on="ticker", how="inner")
    c = c[(c.yes_bid > 0) | (c.yes_ask < 1)]
    close = pd.to_datetime(c.close_time, utc=True).map(lambda x: int(x.timestamp()))
    rows, done = [], []
    for tk, t0, t1 in zip(c.ticker, c["T"], close):
        got, ok = [], False
        for path in ("historical/trades", "markets/trades"):
            cursor, pages, complete = None, [], True
            while True:
                prm = {"ticker": tk, "min_ts": t0, "max_ts": t1, "limit": 1000}
                if cursor:
                    prm["cursor"] = cursor
                f = H.CACHE_DIR / (H._key(f"{wx.API}/{path}", prm) + ".json")
                if not f.exists():
                    complete = False
                    break
                dd = json.loads(f.read_text())
                pages += dd.get("trades", [])
                cursor = dd.get("cursor")
                if not cursor or not dd.get("trades"):
                    break
            if not complete:
                break
            if pages:
                got, ok = pages, True
                break
            ok = True  # historical complete and empty -> _tape would try markets/trades next
        if not ok:
            continue
        done.append(tk)
        for r in got:
            npx = r.get("no_price_dollars")
            npx = float(npx) if npx is not None else float(r.get("no_price")) / 100
            cnt = r.get("count_fp")
            cnt = float(cnt) if cnt is not None else float(r.get("count", 0))
            rows.append({"ticker": tk, "ts": int(pd.Timestamp(r["created_time"]).timestamp()), "no_price": npx,
                         "count": cnt, "taker_side": r.get("taker_side")})
    pd.DataFrame(rows, columns=["ticker", "ts", "no_price", "count", "taker_side"]).to_parquet(TAPES, index=False)
    pd.Series(done, name="ticker").to_frame().to_parquet(DATA / "wx_lockin_tape_tickers.parquet", index=False)
    print(f"cached complete tapes: {len(done)} of {len(c)} tickers, {len(rows)} trades")
    return 0


# ------------------------------------------------------------------------------------------------ scoring
def cp_ci(k, n, a=0.05):
    from scipy.stats import beta
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def boot_p(dates, x, rng, n=4000):
    u, inv = np.unique(dates, return_inverse=True)
    s = np.bincount(inv, weights=x, minlength=len(u))
    b = s[rng.integers(0, len(u), size=(n, len(u)))].sum(1)
    return float(np.mean(b <= 0)), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def trial_bets(c: pd.DataFrame, tr: pd.DataFrame, name: str, maker_fee: float = 0.0, cent_fee: bool = False) -> pd.DataFrame:
    rule, k, ex = name.split("_")
    k = int(k)
    s = c[(c.below_gap >= k) if rule == "BELOW" else (c.above_gap >= k)].copy()
    won = (s.result == "no").to_numpy()
    if ex == "taker":
        s = s[(s.yes_bid > 0) & ((s.yes_ask - s.yes_bid) <= 0.10)].copy()
        p = 1 - s.yes_bid.to_numpy(float)
        fee = np.array([kalshi_fee(x) for x in p])
        if cent_fee:
            fee = np.where(fee > 0, np.ceil(fee * 100 - 1e-9) / 100, 0.0)
        s["entry"], s["cost"] = p, p + fee
    else:
        s = s[(s.yes_ask < 1)].copy()
        q = np.round(1 - s.yes_ask.to_numpy(float), 4)
        s = s[(q >= 0.01) & (q < 1 - s.yes_bid.to_numpy(float) - 1e-9)].copy()
        q = np.round(1 - s.yes_ask.to_numpy(float), 4)
        g = {kk: v for kk, v in tr.groupby("ticker")}
        close = pd.to_datetime(s.close_time, utc=True).map(lambda x: int(x.timestamp())).to_numpy()
        filled = np.zeros(len(s), bool)
        for i, (tk, t0) in enumerate(zip(s.ticker, s["T"])):
            dd = g.get(tk)
            if dd is None:
                continue
            mm = (dd.ts.to_numpy() > t0) & (dd.ts.to_numpy() <= close[i]) & (dd.no_price.to_numpy() <= q[i] + 1e-9)
            filled[i] = dd["count"].to_numpy()[mm].sum() >= 10
        s["entry"], s["cost"] = q, q + maker_fee * q * (1 - q)
        s = s[filled].copy()
    s["won"] = (s.result == "no").astype(int)
    s["pnl"] = np.where(s.won == 1, 1 / s.cost - 1, -1.0)
    return s


def capacity(s: pd.DataFrame, tr: pd.DataFrame) -> tuple[float, float]:
    """contracts printed at no_price <= entry in (T, close], summed per bet -> per dev day with >=1 bet."""
    if s.empty:
        return 0.0, 0.0
    g = {k: v for k, v in tr.groupby("ticker")}
    close = pd.to_datetime(s.close_time, utc=True).map(lambda x: int(x.timestamp())).to_numpy()
    vol = []
    for i, (tk, t0, e) in enumerate(zip(s.ticker, s["T"], s.entry)):
        dd = g.get(tk)
        if dd is None:
            vol.append(0.0)
            continue
        mm = (dd.ts.to_numpy() > t0) & (dd.ts.to_numpy() <= close[i]) & (dd.no_price.to_numpy() <= e + 1e-9)
        vol.append(float(dd["count"].to_numpy()[mm].sum()))
    s = s.assign(vol=vol)
    per_day = s.groupby(s.date.dt.date).apply(lambda x: (x.vol * x.entry).sum())
    return float(s.vol.sum() / s.date.dt.date.nunique()), float(per_day.mean())


def summarize(s: pd.DataFrame, rng) -> dict:
    if s.empty:
        return {"bets": 0}
    dates = s.date.dt.strftime("%Y-%m-%d").to_numpy()
    p, lo, hi = boot_p(dates, s.pnl.to_numpy(), rng)
    k, n = int((s.won == 0).sum()), len(s)
    med = np.median(s.date.values.astype("datetime64[D]").astype(int))
    h1 = s[s.date.values.astype("datetime64[D]").astype(int) <= med]
    h2 = s[s.date.values.astype("datetime64[D]").astype(int) > med]
    cl, ch = cp_ci(k, n)
    return {"bets": n, "losses": k, "loss_rate": k / n, "loss_ci_lo": cl, "loss_ci_hi": ch,
            "breakeven_loss_rate": float(1 - s.cost.mean()), "avg_entry": float(s.entry.mean()),
            "units": float(s.pnl.sum()), "roi": float(s.pnl.mean()), "ci_lo": lo, "ci_hi": hi, "p": p,
            "half1_units": float(h1.pnl.sum()), "half1_bets": len(h1), "half2_units": float(h2.pnl.sum()), "half2_bets": len(h2),
            "days": int(s.date.dt.date.nunique()),
            "by_cohort": {c: f"{g.pnl.sum():+.2f}/{len(g)} L{int((g.won == 0).sum())}" for c, g in s.groupby("cohort")}}


def cmd_dev() -> int:
    m = dev_markets()
    d = pd.read_parquet(DATA / "wx_lockin_days.parquet")
    c = candidates(m, d).merge(pd.read_parquet(QUOTES)[["ticker", "price_ts", "yes_bid", "yes_ask"]], on="ticker", how="inner")
    tr = pd.read_parquet(TAPES)
    assert (c[c.cohort == "E1"].date < ORIG_END).all() and (c[c.cohort == "X1"].date < X1_END).all()
    print(f"DEV candidates with a 17:00 quote: {len(c)} on {c.event_ticker.nunique()} station-days; "
          f"tapes on {c.ticker.isin(tr.ticker).mean():.1%} of them")
    for nm, msk in (("BELOW_0", c.below_gap >= 0), ("ABOVE_2", c.above_gap >= 2)):
        x = c[msk]
        print(f"  {nm}: {len(x)} brackets; quote at 17:00: yes_bid>0 {(x.yes_bid > 0).mean():.1%}, "
              f"yes_ask<1 {(x.yes_ask < 1).mean():.1%}; yes_bid dist {x.yes_bid.round(2).value_counts().sort_index().head(8).to_dict()}; "
              f"Kalshi result=yes (rule would lose) {int((x.result == 'yes').sum())}")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    shas |= {hashlib.sha256(("wx_lockin:" + n).encode()).hexdigest()[:12] for n in TRIALS}
    bar = 0.05 / len(shas)
    recs = []
    have = set(pd.read_parquet(DATA / "wx_lockin_tape_tickers.parquet").ticker)
    print(f"tape subset (complete cached tapes): {len(have)} tickers -> capacity / maker fills DESCRIPTIVE only")
    for name in TRIALS:
        if name.endswith("maker"):
            sub = c[c.ticker.isin(have)]
            s = trial_bets(sub, tr, name)
            rule, k, _ = name.split("_")
            elig = sub[((sub.below_gap >= int(k)) if rule == "BELOW" else (sub.above_gap >= int(k))) & (sub.yes_ask < 1)]
            print(f"{name:14s} NOT SCORED: data incomplete (tapes on {len(have)}/11,176 tickers). Descriptive on subset: "
                  f"{len(elig)} eligible, {len(s)} filled at N=10, losses {int((s.won == 0).sum()) if len(s) else 0}, "
                  f"units {s.pnl.sum() if len(s) else 0:+.2f}, avg entry {s.entry.mean() if len(s) else float('nan'):.3f}")
            recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"wx_lockin_{name}",
                         "sha": hashlib.sha256(("wx_lockin:" + name).encode()).hexdigest()[:12], "dataset": DATASET,
                         "status": "not_scored_data_incomplete", "trial": name, "trial_number": TRIALS.index(name) + 1,
                         "study_trials": len(TRIALS), "bonferroni_bar": bar, "source": "lead L26",
                         "descriptive_subset": {"tickers_with_tape": len(have), "eligible": int(len(elig)), "filled": int(len(s))}})
            continue
        s = trial_bets(c, tr, name)
        r = summarize(s, np.random.default_rng(0))
        if r["bets"]:
            st = s[s.ticker.isin(have)]
            r["capacity_subset_bets"] = int(len(st))
            r["contracts_per_day"], r["dollars_per_day"] = capacity(st, tr) if len(st) else (0.0, 0.0)
            r["passed"] = bool(r["p"] < bar and r["units"] > 0 and r["half1_units"] > 0 and r["half2_units"] > 0)
            if name.endswith("taker"):
                r["units_cent_fee"] = float(trial_bets(c, tr, name, cent_fee=True).pnl.sum())
            else:
                r["units_maker_fee"] = float(trial_bets(c, tr, name, maker_fee=0.0175).pnl.sum())
            print(f"{name:14s} bets {r['bets']:5d} losses {r['losses']:3d} loss {r['loss_rate']:.2%} "
                  f"[{r['loss_ci_lo']:.2%},{r['loss_ci_hi']:.2%}] BE {r['breakeven_loss_rate']:.2%} entry {r['avg_entry']:.3f} "
                  f"units {r['units']:+.2f} ROI {r['roi']:+.2%} [{r['ci_lo']:+.1f},{r['ci_hi']:+.1f}] p {r['p']:.4f} "
                  f"halves {r['half1_units']:+.2f}/{r['half2_units']:+.2f} cap(tape subset n={r['capacity_subset_bets']}) {r['contracts_per_day']:.0f} ct/day "
                  f"${r['dollars_per_day']:.0f}/day {'PASS' if r['passed'] else ''}")
            print(f"               {r['by_cohort']}  sens: {r.get('units_cent_fee', r.get('units_maker_fee')):+.2f}")
            if r["losses"]:
                print(s[s.won == 0][["ticker", "date", "M17", "u", "l", "expiration_value", "result", "entry", "late_rise"]].to_string(index=False))
        else:
            r["passed"] = False
            print(f"{name:14s} no bets")
        sha = hashlib.sha256(("wx_lockin:" + name).encode()).hexdigest()[:12]
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"wx_lockin_{name}", "sha": sha,
                     "dataset": DATASET, "hypothesis": f"pre-registered in scripts/wx_lockin.py ({name}); late-day 17:00 "
                     "observed-max lock-in on KXHIGH brackets, dev only", "source": "lead L26", "status": "ok",
                     "trial_number": TRIALS.index(name) + 1, "study_trials": len(TRIALS), "bonferroni_bar": bar, "trial": name, **r})
    print(f"{DATASET}: {len(shas)} trials -> bar p < {bar:.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        print("appended to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"obs": cmd_obs, "diag": cmd_diag, "build": cmd_build, "tapes-cached": cmd_tapes_cached, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
