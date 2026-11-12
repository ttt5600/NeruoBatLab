"""KALSHI FIRST LISTING: how stale is the first tradeable Kalshi sports price, across ALL sports series? (agent
kalshi_first_listing, 2026-11-12; notes/kalshi_first_listing.md has the pre-registration). Read-only GETs, no orders.

    python scripts/kalshi_first_listing.py universe   # settled sports markets closing in the census window
    python scripts/kalshi_first_listing.py candles    # hourly candles for the sampled markets
    python scripts/kalshi_first_listing.py census     # per-series census table
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, os.path.expanduser("~/agent-harness/lib"))

import numpy as np
import pandas as pd

from sports.http import fetch
from kalshi_longshot import API
from traps import check_epoch_seconds, epoch_seconds

OUT = ROOT / "results" / "kalshi_first_listing"
OUT.mkdir(parents=True, exist_ok=True)
UNIV = OUT / "universe.parquet"
WIN_LO, WIN_HI = "2026-10-20", "2026-11-11"   # census window on close_time (UTC)


def series_list() -> pd.DataFrame:
    s = fetch(f"{API}/series", {"category": "Sports"}, cache=True)["series"]
    return pd.DataFrame([{"series": x["ticker"], "title_s": x.get("title"), "freq": x.get("frequency"),
                          "tags": ",".join(x.get("tags") or [])} for x in s])


def cmd_universe() -> int:
    lo = int(pd.Timestamp(WIN_LO, tz="UTC").timestamp())
    hi = int(pd.Timestamp(WIN_HI, tz="UTC").timestamp())
    check_epoch_seconds([lo, hi], "window")
    # phase A (selection only): replay the global settled-market pages already cached (close 11-07..11-11, all
    # categories; the full global pull was ~90 min) to rank Sports series by volume. Phase B: per-series pull of the
    # whole window for the top series.
    from sports.http import CACHE_DIR, _key
    A, cursor = [], None
    while True:
        p = {"status": "settled", "min_close_ts": lo, "max_close_ts": hi, "limit": 1000, "mve_filter": "exclude"}
        if cursor:
            p["cursor"] = cursor
        if not (CACHE_DIR / (_key(f"{API}/markets", p) + ".json")).exists():
            break
        d = fetch(f"{API}/markets", p, cache=True)
        A += [(m["event_ticker"].split("-")[0], m["event_ticker"], f(m.get("volume_fp"))) for m in d.get("markets", [])]
        cursor = d.get("cursor")
        if not cursor:
            break
    a = pd.DataFrame(A, columns=["series", "event_ticker", "volume"])
    sl = series_list()
    a = a[a.series.isin(set(sl.series))]
    rank = a.groupby("series").agg(ev=("event_ticker", "nunique"), vol=("volume", "sum")).sort_values("vol", ascending=False)
    rank.to_csv(OUT / "phaseA_rank.csv")
    top = rank[rank.ev >= 5].head(TOP_SERIES + 15).index.tolist()
    print(f"phase A: {len(A)} markets cached; top series: {top}", flush=True)
    rows = []
    for s in top:
        cursor = None
        while True:
            p = {"series_ticker": s, "status": "settled", "min_close_ts": lo, "max_close_ts": hi, "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            d = fetch(f"{API}/markets", p, cache=True)
            for m in d.get("markets", []):
                rows.append({k: m.get(k) for k in ("ticker", "event_ticker", "open_time", "close_time",
                                                   "occurrence_datetime", "expected_expiration_time", "result",
                                                   "volume_fp", "title", "yes_sub_title", "strike_type",
                                                   "floor_strike", "cap_strike")})
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                break
    u =pd.DataFrame(rows).drop_duplicates("ticker")
    u["volume"] = pd.to_numeric(u.volume_fp, errors="coerce")
    u["series"] = u.event_ticker.str.split("-").str[0]
    sl = series_list()
    u = u[u.series.isin(set(sl.series))].merge(sl, on="series", how="left")
    u.to_parquet(UNIV, index=False)
    g = u.groupby("series").agg(markets=("ticker", "size"), events=("event_ticker", "nunique"), vol=("volume", "sum"))
    print(g.sort_values("vol", ascending=False).head(70).to_string())
    print(f"{len(u)} sports markets, {u.series.nunique()} series -> {UNIV}")
    return 0


CAND = OUT / "candles.parquet"
SAMPLE = OUT / "sample_events.parquet"
MIN_EVENTS, TOP_SERIES, PER_SERIES, SEED = 30, 40, 40, 0
MAXSPR, TAKER, MAKER = 0.10, 0.07, 0.0175


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def candles(series: str, ticker: str, s: int, e: int) -> list[dict]:
    out, a = [], s
    while a < e:
        b = min(e, a + 3600 * 4000)
        j = fetch(f"{API}/series/{series}/markets/{ticker}/candlesticks",
                  {"start_ts": a, "end_ts": b, "period_interval": 60}, cache=True)
        for c in j.get("candlesticks", []):
            p, ya, yb = c.get("price") or {}, c.get("yes_ask") or {}, c.get("yes_bid") or {}
            out.append(dict(ticker=ticker, end=int(c["end_period_ts"]), bid=f(yb.get("close_dollars")),
                            ask=f(ya.get("close_dollars")), tlow=f(p.get("low_dollars")), thigh=f(p.get("high_dollars")),
                            vol=f(c.get("volume_fp"))))
        a = b
    return out


def sample() -> pd.DataFrame:
    u = pd.read_parquet(UNIV)
    g = u.groupby("series").agg(events=("event_ticker", "nunique"), vol=("volume", "sum"))
    keep = g[g.events >= MIN_EVENTS].sort_values("vol", ascending=False).head(TOP_SERIES).index
    u = u[u.series.isin(keep)].copy()
    rng = np.random.default_rng(SEED)
    ev = []
    for s, grp in u.groupby("series"):
        e = sorted(grp.event_ticker.unique())
        ev += list(rng.choice(e, size=min(PER_SERIES, len(e)), replace=False))
    u = u[u.event_ticker.isin(set(ev))].copy()
    # cap ladders at the middle 5 markets by strike (or ticker order) -- chosen from metadata, no prices
    u["sk"] = pd.to_numeric(u.floor_strike, errors="coerce").fillna(pd.to_numeric(u.cap_strike, errors="coerce"))
    u = u.sort_values(["event_ticker", "sk", "ticker"])
    u["rk"] = u.groupby("event_ticker").cumcount()
    u["nk"] = u.groupby("event_ticker").ticker.transform("size")
    u = u[u.rk == (u.nk - 1) // 2].copy()  # ONE market per event (rate limits, 11-12 10:05)
    for c in ("open_time", "close_time", "expected_expiration_time"):
        u[c + "_s"] = epoch_seconds(pd.to_datetime(u[c], utc=True, format="ISO8601"))
    u["tstart"] = u.expected_expiration_time_s - 3 * 3600
    u["tref"] = u.tstart - 1800
    check_epoch_seconds(u.tref, "tref")
    return u


def cmd_candles() -> int:
    u = sample()
    u.to_parquet(SAMPLE, index=False)
    done = set(pd.read_parquet(CAND).ticker) if CAND.exists() else set()
    rows = [pd.read_parquet(CAND)] if CAND.exists() else []
    todo = u[~u.ticker.isin(done) & (u.tref > u.open_time_s)]
    print(f"{u.series.nunique()} series, {u.event_ticker.nunique()} events, {len(u)} markets; to fetch {len(todo)}", flush=True)
    buf = []
    for i, r in enumerate(todo.itertuples()):
        e = int(min(r.tref + 3600, r.close_time_s))
        try:
            buf += candles(r.series, r.ticker, int(r.open_time_s) - 3600, e)
        except Exception as ex:  # noqa: BLE001
            print("ERR", r.ticker, ex, flush=True)
        if (i + 1) % 500 == 0:
            rows.append(pd.DataFrame(buf)); buf = []
            pd.concat(rows, ignore_index=True).to_parquet(CAND, index=False)
            print(i + 1, flush=True)
    rows.append(pd.DataFrame(buf))
    pd.concat(rows, ignore_index=True).drop_duplicates(["ticker", "end"]).to_parquet(CAND, index=False)
    return 0


def spearman(x, y):
    from scipy.stats import spearmanr
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan, np.nan
    r = spearmanr(x[m], y[m])
    return float(r.statistic), float(r.pvalue)


def per_market(u: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    c = c.sort_values(["ticker", "end"])
    check_epoch_seconds(c.end, "candle end")
    meta = u.set_index("ticker")
    out = []
    for t, g in c.groupby("ticker", sort=False):
        m = meta.loc[t]
        g = g[g.end <= m.tref]
        q = g[(g.bid > 0) & (g.ask < 1) & (g.ask - g.bid <= MAXSPR)]
        if q.empty:
            continue
        f0 = q.iloc[0]
        mid0 = (f0.bid + f0.ask) / 2
        last = q.iloc[-1]
        mid_ref = (last.bid + last.ask) / 2
        q2 = q[q.end <= f0.end + 2 * 3600]
        mid2 = (q2.iloc[-1].bid + q2.iloc[-1].ask) / 2
        later = g[g.end > f0.end]
        ing = g[(g.end > m.tref - 2 * 3600)]
        out.append(dict(series=m.series, event_ticker=m.event_ticker, ticker=t, close_s=m.close_time_s,
                        lead_h=(m.tstart - m.open_time_s) / 3600, delay_h=(f0.end - m.open_time_s) / 3600,
                        hours_left=(m.tref - f0.end) / 3600, first_bid=f0.bid, first_ask=f0.ask, spread=f0.ask - f0.bid,
                        mid0=mid0, mid2=mid2, mid_ref=mid_ref, ref_age_h=(m.tref - last.end) / 3600,
                        vol_pre=g.vol.sum(), yes=float(m.result == "yes"),
                        ingame=bool(((ing.thigh - ing.tlow) >= 0.25).any()),
                        fill_yes=bool((later.tlow <= f0.bid - 0.01).any()),
                        fill_no=bool((later.thigh >= f0.ask + 0.01).any())))
    d = pd.DataFrame(out)
    d["D"] = d.mid_ref - d.mid0
    d["d1"] = d.mid2 - d.mid0
    d["r"] = d.mid_ref - d.mid2
    # one market per event: first mid closest to 0.5 (chosen from first quotes only)
    d["k"] = (d.mid0 - 0.5).abs()
    return d.sort_values(["event_ticker", "k", "ticker"]).drop_duplicates("event_ticker")


def cmd_census() -> int:
    u = pd.read_parquet(SAMPLE)
    c = pd.read_parquet(CAND)
    d = per_market(u, c)
    d.to_parquet(OUT / "per_event.parquet", index=False)
    rows = []
    for s, g in d.groupby("series"):
        g = g.sort_values("close_s")
        h = len(g) // 2
        p1 = spearman(g.d1.to_numpy(), g.r.to_numpy())
        p2 = spearman((g.mid0 - 0.5).to_numpy(), g.D.to_numpy())
        p1a, p1b = spearman(g.d1.to_numpy()[:h], g.r.to_numpy()[:h]), spearman(g.d1.to_numpy()[h:], g.r.to_numpy()[h:])
        p2a = spearman((g.mid0 - 0.5).to_numpy()[:h], g.D.to_numpy()[:h])
        p2b = spearman((g.mid0 - 0.5).to_numpy()[h:], g.D.to_numpy()[h:])
        cy = g.first_ask + TAKER * g.first_ask * (1 - g.first_ask)
        cn = (1 - g.first_bid) + TAKER * g.first_bid * (1 - g.first_bid)
        fy, fn = g[g.fill_yes], g[g.fill_no]
        by = fy.first_bid + MAKER * fy.first_bid * (1 - fy.first_bid)
        bn = (1 - fn.first_ask) + MAKER * fn.first_ask * (1 - fn.first_ask)
        sig1 = p1[1] < 0.001 and np.sign(p1a[0]) == np.sign(p1b[0]) == np.sign(p1[0])
        sig2 = p2[1] < 0.001 and np.sign(p2a[0]) == np.sign(p2b[0]) == np.sign(p2[0])
        row = dict(series=s, n=len(g), lead_h=g.lead_h.median(), delay_h=g.delay_h.median(),
                   spread=g.spread.median(), absD_med=g.D.abs().median(), absD_mean=g.D.abs().mean(),
                   D_mean=g.D.mean(), vol_pre=g.vol_pre.median(), ingame=g.ingame.mean(),
                   p1_rho=p1[0], p1_p=p1[1], p2_rho=p2[0], p2_p=p2[1],
                   tk_yes_roi=(g.yes - cy).sum() / cy.sum(), tk_no_roi=((1 - g.yes) - cn).sum() / cn.sum(),
                   mk_yes_fill=g.fill_yes.mean(), mk_yes_roi=(fy.yes - by).sum() / by.sum() if len(fy) else np.nan,
                   yes_rate_filled=fy.yes.mean(), yes_rate_unfilled=g[~g.fill_yes].yes.mean(),
                   mk_no_fill=g.fill_no.mean(), mk_no_roi=((1 - fn.yes) - bn).sum() / bn.sum() if len(fn) else np.nan)
        row["PASS"] = bool(row["n"] >= MIN_EVENTS and row["absD_med"] >= 0.03 and row["spread"] <= 0.05
                           and row["ingame"] <= 0.20 and (sig1 or sig2))
        rows.append(row)
    t = pd.DataFrame(rows).sort_values("absD_med", ascending=False)
    t.to_csv(OUT / "census.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "census"
    sys.exit({"universe": cmd_universe, "candles": cmd_candles, "census": cmd_census}[cmd]())
