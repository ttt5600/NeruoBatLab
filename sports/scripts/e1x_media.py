"""E1X-M: Kalshi weekly MEDIA series far-bracket survey / consensus-reject (YouTube, Netflix, charts).
Prereg + results: notes/e1x_media.md.  Stages: survey -> markets -> ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_family_scan import get  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "e1x_media"
SERIES = ["KXYTVIEWSW", "KXNETFLIXRANKSHOW", "KXNETFLIXRANKSHOWRUNNERUP", "KXNETFLIXRANKSHOWGLOBAL",
          "KXALBUMEQUIV", "KXBBSINGLE", "KXTOPSONG", "KXTOPALBUM"]


def cmd_survey() -> int:
    """Series metadata, settled event count, rules text, open ladder shape. No prices or outcomes joined."""
    for s in SERIES:
        se = get(f"/series/{s}").get("series", {})
        print("=====", s, se.get("frequency"), se.get("fee_type"),
              [x.get("name") for x in (se.get("settlement_sources") or [])])
        ev, cur = [], None
        while True:
            p = {"series_ticker": s, "status": "settled", "limit": 200}
            if cur:
                p["cursor"] = cur
            r = get("/events", p)
            ev += r.get("events", [])
            cur = r.get("cursor")
            if not cur or not r.get("events"):
                break
        print(" settled events", len(ev), [e["event_ticker"] for e in ev[:4]])
        op = get("/markets", {"series_ticker": s, "status": "open", "limit": 200}).get("markets", [])
        if op:
            m = op[0]
            print(" RULES:", m["rules_primary"][:500].replace("\n", " "))
            print(" open", len(op), [(x["ticker"], x.get("yes_sub_title"), x.get("strike_type"), x.get("floor_strike"),
                                      x.get("open_time"), x.get("close_time")) for x in op[:5]])
    return 0


def _all_markets(s: str) -> list[dict]:
    out = []
    for ep in ("/markets", "/historical/markets"):
        cur = None
        while True:
            p = {"series_ticker": s, "limit": 1000}
            if cur:
                p["cursor"] = cur
            r = get(ep, p)
            ms = r.get("markets", [])
            for m in ms:
                m["_src"] = ep
            out += ms
            cur = r.get("cursor")
            if not cur or not ms:
                break
    return out


def cmd_markets() -> int:
    D.mkdir(parents=True, exist_ok=True)
    for s in SERIES:
        ms = _all_markets(s)
        if not ms:
            print(s, "no markets")
            continue
        df = pd.DataFrame(ms).drop_duplicates("ticker")
        df.to_parquet(D / f"{s}_markets.parquet")
        st = df[df.status.isin(["settled", "finalized", "determined"])] if "status" in df else df
        print(s, "markets", len(df), "settled-ish", len(st), "events", df.event_ticker.nunique(),
              "close range", df.close_time.min(), df.close_time.max(),
              "vol sum", pd.to_numeric(df.get("volume_fp", df.get("volume")), errors="coerce").sum())
    return 0


# ------------------------------------------------------------------ KXYTVIEWSW (trial ytv_m1_maker)
YT = "KXYTVIEWSW"
M = 0.02          # consensus margin (rung spacing ~2% near the money)
MIN_RESID = 20


def yt_markets() -> pd.DataFrame:
    df = pd.read_parquet(D / f"{YT}_markets.parquet")
    df = df[df.event_ticker.str.match(r"^KXYTVIEWSW-[A-Z]+\d\d[A-Z]{3}\d\d$")].copy()
    df["wend"] = pd.to_datetime(df.event_ticker.str[-7:], format="%y%b%d")
    df["K"] = df.floor_strike.astype(float)
    df["open"] = pd.to_datetime(df.open_time, utc=True)
    df["close"] = pd.to_datetime(df.close_time, utc=True)
    return df[df.K.notna()]


def decision_time(wend: pd.Timestamp) -> pd.Timestamp:
    """Sunday (last day of the Mon..Sun week) 16:00 UTC: data through Friday/Saturday published, Sunday not."""
    return pd.Timestamp(wend).tz_localize("UTC") + pd.Timedelta(hours=16)


def _upper_open(g: pd.DataFrame, T: pd.Timestamp) -> tuple[float, float]:
    """U = lowest rung listed at T and still trading (not early-closed by T); L = highest rung early-closed by T.
    Early close (close <= T, i.e. before the scheduled Tuesday close) is public at close time = YES lock."""
    ex = g[g.open <= T]
    alive = ex[ex.close > T]
    locked = ex[ex.close <= T]
    return (alive.K.min() if len(alive) else np.nan), (locked.K.max() if len(locked) else 0.0)


def weekly_max(g: pd.DataFrame) -> tuple[float, bool]:
    # Amendment A1 (2026-11-06, before any price join): numeric expiration_value can sit BELOW a YES rung (audit 6/105),
    # so the weekly max is at least the highest YES rung -> M = max(numeric, highest YES K).
    v = pd.to_numeric(g.expiration_value, errors="coerce").dropna()
    ky = g.loc[g.result == "yes", "K"]
    if len(v) or len(ky):
        m = max(float(v.max()) if len(v) else 0.0, float(ky.max()) if len(ky) else 0.0)
        return m, bool((g.result == "yes").all())
    return np.nan, False


def cmd_yt_audit() -> int:
    """Weekly max (max numeric expiration_value) must agree with every rung's result (yes iff M > K)."""
    df = yt_markets()
    df = df[df.result.isin(["yes", "no"])]
    ok = n = 0
    bad = []
    for ev, g in df.groupby("event_ticker"):
        m, lb = weekly_max(g)
        if np.isnan(m) or lb:
            continue
        n += 1
        good = all((m > k) == (r == "yes") for k, r in zip(g.K, g.result))
        ok += good
        if not good:
            bad.append((ev, m, sorted(g[g.result == "yes"].K)[-1:], sorted(g[g.result == "no"].K)[:1]))
    print(f"consistent {ok}/{n} = {ok / max(n, 1):.3f}")
    for b in bad[:10]:
        print("  ", b)
    return 0


def cmd_yt_build() -> int:
    df = yt_markets()
    ev = []
    for e, g in df.groupby("event_ticker"):
        T = decision_time(g.wend.iloc[0])
        U, L = _upper_open(g, T)
        m, lb = weekly_max(g[g.result.isin(["yes", "no"])]) if g.result.isin(["yes", "no"]).all() else (np.nan, False)
        ev.append(dict(event=e, wend=g.wend.iloc[0], T=T, U=U, L=L, Mw=m, M_lower_bound=lb))
    ev = pd.DataFrame(ev)
    ev["r"] = np.log(ev.Mw / ev.U)
    rows = []
    for _, x in ev.iterrows():
        past = ev[(ev.wend <= x.wend - pd.Timedelta(days=7)) & ev.r.notna()]
        rows.append(dict(event=x.event, n_resid=len(past), rmax=past.r.max() if len(past) else np.nan,
                         n_lb=int(past.M_lower_bound.sum())))
    b = ev.merge(pd.DataFrame(rows), on="event")
    b["hi"] = b.U * np.exp(b.rmax + M)
    b.drop(columns=["Mw", "r", "M_lower_bound"]).to_parquet(D / "ytv_build.parquet")   # no outcome info kept
    print("events", len(b), "with U", b.U.notna().sum(), "n_resid>=20", (b.n_resid >= MIN_RESID).sum())
    print(b[["event", "T", "U", "L", "n_resid", "rmax", "hi"]].tail(8).to_string())
    return 0


def yt_rungs() -> pd.DataFrame:
    b = pd.read_parquet(D / "ytv_build.parquet")
    df = yt_markets()
    x = df.merge(b.drop(columns="wend"), left_on="event_ticker", right_on="event")
    x = x[(x.n_resid >= MIN_RESID) & x.U.notna() & (x.open <= x["T"]) & (x.close > x["T"])]
    x = x[x.K >= x.hi].copy()
    x["side"] = "no"
    return x


def cmd_yt_candles() -> int:
    x = yt_rungs()
    cache = D / "ytv_candles.json"
    out = json.loads(cache.read_text()) if cache.exists() else {}
    for _, m in x.iterrows():
        if m.ticker in out:
            continue
        ts = int(m["T"].timestamp())
        es = ts + 8 * 3600   # through event_start (Monday 00:00Z) for the CLV mark
        p = {"start_ts": ts - 12 * 3600, "end_ts": es, "period_interval": 60}
        ep = "/historical/markets/{}/candlesticks" if m["_src"] == "/historical/markets" else \
            "/series/" + YT + "/markets/{}/candlesticks"
        out[m.ticker] = get(ep.format(m.ticker), p).get("candlesticks", [])
    cache.write_text(json.dumps(out))
    print("rejected rungs", len(x), "events", x.event_ticker.nunique(), "with candles",
          sum(bool(out.get(t)) for t in x.ticker))
    return 0


def cmd_yt_trades() -> int:
    x = yt_rungs()
    cache = D / "ytv_trades.json"
    out = json.loads(cache.read_text()) if cache.exists() else {}
    for _, m in x.iterrows():
        if m.ticker in out:
            continue
        ep = "/historical/trades" if m["_src"] == "/historical/markets" else "/markets/trades"
        tr, cur = [], None
        while True:
            p = {"ticker": m.ticker, "min_ts": int(m["T"].timestamp()) - 12 * 3600, "limit": 1000}
            if cur:
                p["cursor"] = cur
            r = get(ep, p)
            tr += r.get("trades", [])
            cur = r.get("cursor")
            if not cur or not r.get("trades"):
                break
        out[m.ticker] = tr
        if len(out) % 20 == 0:
            cache.write_text(json.dumps(out))
            print("progress", len(out), flush=True)
    cache.write_text(json.dumps(out))
    print("tickers", len(out), "trades", sum(len(v) for v in out.values()))
    return 0


def cmd_yt_capacity() -> int:
    """Contracts that printed THROUGH our maker NO price after T (upper bound on what a resting bid could fill), per week.
    Uses quotes + tape only (no outcomes)."""
    sys.path.insert(0, str(ROOT))
    from edgelab.rules import media_yt as R
    x = yt_rungs()
    x.attrs["candles"] = json.loads((D / "ytv_candles.json").read_text())
    x.attrs["trades"] = json.loads((D / "ytv_trades.json").read_text())
    ts = R._unix(pd.to_datetime(x["T"], utc=True))
    yb, ya, _ = R._quotes(x, ts)
    rows = []
    for (tk, wk, T, b, a) in zip(x.ticker, x.wend, x["T"], yb, ya):
        if np.isnan(b) or np.isnan(a):
            rows.append(dict(wk=wk, quoted=0, q=np.nan, n=0.0)); continue
        q = np.floor(round(min(1 - (b + a) / 2, 1 - b - 0.01) * 100, 6)) / 100
        n = 0.0
        if 0.5 <= q <= 0.99:
            for k in x.attrs["trades"].get(tk, []):
                if pd.Timestamp(k["created_time"]) > T and k["taker_side"] == "yes" and R._yp(k) > 1 - q + 1e-9:
                    n += float(k.get("count_fp", k.get("count", 0)))
        rows.append(dict(wk=wk, quoted=1, q=q, n=n))
    c = pd.DataFrame(rows)
    c["usd"] = c.n * c.q.fillna(0)
    print("rungs", len(c), "quoted", int(c.quoted.sum()), "q in [.5,.99]", int(c.q.between(.5, .99).sum()),
          "mean q", round(c.q.mean(), 3))
    w = c.groupby("wk").agg(rungs=("n", "size"), fillable=("n", lambda v: int((v > 0).sum())), contracts=("n", "sum"),
                            usd=("usd", "sum"))
    print(w.to_string()); print("median per week contracts", w.contracts.median(), "usd", round(w.usd.median(), 2))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    a = ap.parse_args()
    sys.exit(dict(survey=cmd_survey, markets=cmd_markets, yt_audit=cmd_yt_audit, yt_build=cmd_yt_build,
                  yt_candles=cmd_yt_candles, yt_trades=cmd_yt_trades,
                  yt_capacity=cmd_yt_capacity)[a.stage]())
