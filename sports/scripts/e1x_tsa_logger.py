"""E1X-T forward PAPER logger: KXTSAW weekly TSA consensus-reject, maker 1c inside (trial T2 of notes/e1x_tsa.md).
No orders are ever placed. Read-only public Kalshi API, paced (~1.6 req/s, backoff on 429).

Daily TSA numbers: history from data/e1x/tsa_daily_<year>.csv (2023-01-01..2026-10-29); newer days from Kalshi KXTSAD
settled expiration values (first-published TSA numbers; matched the CSV exactly on 2026-10-16..10-29). tsa.gov itself
returns 403 to scripts, so it is not called.

Usage:
  python scripts/e1x_tsa_logger.py tick     # run Mondays 12:00-20:00 ET (after KXTSAW lists ~10:00 ET); idempotent
  python scripts/e1x_tsa_logger.py fills    # daily: mark paper fills from the trade tape, then settle
  python scripts/e1x_tsa_logger.py report   # judge rule: >= 60 settled fills; KILL if ROI <= 0 or >= 3 losses in first 30
Log: data/e1x/tsa_paper.jsonl (one row per posted paper order; fills/settles appended as update rows).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_family_scan import get  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "e1x"
LOG = D / "tsa_paper.jsonl"
M = 0.01
ET = "America/New_York"


def daily() -> pd.Series:
    parts = [pd.read_csv(p) for p in sorted(D.glob("tsa_daily_*.csv"))]
    t = pd.concat(parts)
    t["date"] = pd.to_datetime(t.date, format="%m/%d/%Y")
    s = t.drop_duplicates("date").set_index("date").n.astype(float)
    cur, extra = None, {}
    while True:
        p = {"series_ticker": "KXTSAD", "status": "settled", "limit": 1000}
        if cur:
            p["cursor"] = cur
        r = get("/markets", p)
        for m in r.get("markets", []):
            v = str(m.get("expiration_value") or "").replace(",", "")
            if v.replace(".", "").isdigit():
                extra[pd.to_datetime(m["event_ticker"].split("-")[1], format="%y%b%d")] = float(v)
        cur = r.get("cursor")
        if not cur or not r.get("markets"):
            break
    for d, v in extra.items():
        if d not in s.index:
            s.loc[d] = v
    return s.sort_index()


def weekly(t: pd.Series, wend: pd.Timestamp) -> float:
    days = pd.date_range(wend - pd.Timedelta(days=6), wend)
    return float(t.loc[days].mean()) if days.isin(t.index).all() else np.nan


def forecast(t: pd.Series, wend: pd.Timestamp, T: pd.Timestamp) -> float:
    sat = T.tz_convert(ET).normalize().tz_localize(None)
    while sat.dayofweek != 5:
        sat -= pd.Timedelta(days=1)
    s = pd.date_range(sat - pd.Timedelta(days=13), sat)
    ly = s - pd.Timedelta(days=364)
    wk = pd.date_range(wend - pd.Timedelta(days=6), wend) - pd.Timedelta(days=364)
    if not (s.isin(t.index).all() and ly.isin(t.index).all() and wk.isin(t.index).all()):
        return np.nan
    return float(t.loc[s].sum() / t.loc[ly].sum() * t.loc[wk].mean())


def resid(t: pd.Series, T: pd.Timestamp) -> pd.Series:
    lim = T.tz_convert(None).normalize() - pd.Timedelta(days=8)
    out = {}
    for w in pd.date_range("2024-01-21", lim, freq="W-SUN"):
        Tw = pd.Timestamp(w - pd.Timedelta(days=6), tz=ET) + pd.Timedelta(hours=12)
        f, a = forecast(t, w, Tw), weekly(t, w)
        if not (np.isnan(f) or np.isnan(a)):
            out[w] = np.log(a / f)
    return pd.Series(out)


def rows() -> list[dict]:
    return [json.loads(x) for x in LOG.read_text().splitlines()] if LOG.exists() else []


def append(r: dict) -> None:
    with LOG.open("a") as f:
        f.write(json.dumps(r, default=str) + "\n")


def cmd_tick() -> int:
    now = pd.Timestamp.now(tz="UTC")
    ms = get("/markets", {"series_ticker": "KXTSAW", "status": "open", "limit": 200}).get("markets", [])
    if not ms:
        print("no open KXTSAW markets")
        return 0
    df = pd.DataFrame(ms)
    df["open"] = pd.to_datetime(df.open_time, format="ISO8601")
    ev = df.sort_values("open").event_ticker.iloc[-1]
    e = df[df.event_ticker == ev]
    age_h = (now - e.open.min()).total_seconds() / 3600
    if age_h > 12:
        print(f"{ev} listed {age_h:.1f} h ago (> 12 h): decision window missed, not logging")
        return 0
    if any(r.get("event") == ev and r.get("kind") == "post" for r in rows()):
        print(f"{ev} already logged")
        return 0
    t = daily()
    wend = pd.to_datetime(ev.split("-")[1], format="%y%b%d")
    F = forecast(t, wend, now)
    r = resid(t, now)
    if np.isnan(F) or len(r) < 20:
        print(f"{ev}: no forecast (F={F}, n_resid={len(r)}); last data day {t.index.max().date()}")
        return 1
    hi, lo = F * np.exp(r.max() + M), F * np.exp(r.min() - M)
    n = 0
    for _, m in e.iterrows():
        K = float(m.floor_strike)
        yb, ya = float(m.yes_bid_dollars), float(m.yes_ask_dollars)
        side = "no" if K >= hi else ("yes" if K <= lo else "")
        if not side or ya - yb < 0.02 - 1e-9:
            continue
        if side == "no" and ya > 0.99 + 1e-9:
            continue
        if side == "yes" and yb < 0.01 - 1e-9:
            continue
        q = round(1 - ya + 0.01, 2) if side == "no" else round(yb + 0.01, 2)
        append(dict(kind="post", ts=now.isoformat(), event=ev, ticker=m.ticker, K=K, side=side, q=q, yes_bid=yb,
                    yes_ask=ya, F=round(F), hi=round(hi), lo=round(lo), n_resid=len(r),
                    last_data=str(t.index.max().date())))
        n += 1
    print(f"{ev}: F={F:,.0f} band [{lo:,.0f}, {hi:,.0f}] posted {n} paper orders")
    return 0


def _yp(k: dict) -> float:
    v = float(k.get("yes_price_dollars", k.get("yes_price")))
    return v / 100 if v > 1.5 else v


def cmd_fills() -> int:
    rs = rows()
    posts = {r["ticker"]: r for r in rs if r["kind"] == "post"}
    done_fill = {r["ticker"] for r in rs if r["kind"] == "fill"}
    done_set = {r["ticker"] for r in rs if r["kind"] == "settle"}
    for tk, p in posts.items():
        if tk not in done_fill:
            t0 = pd.Timestamp(p["ts"])
            tr = get("/markets/trades", {"ticker": tk, "min_ts": int(t0.timestamp()), "limit": 1000}).get("trades", [])
            for k in sorted(tr, key=lambda k: k["created_time"]):
                if pd.Timestamp(k["created_time"]) <= t0:
                    continue
                if (p["side"] == "no" and k["taker_side"] == "yes" and _yp(k) >= 1 - p["q"] - 1e-9) or \
                   (p["side"] == "yes" and k["taker_side"] == "no" and _yp(k) <= p["q"] + 1e-9):
                    append(dict(kind="fill", ticker=tk, event=p["event"], fill_ts=k["created_time"], q=p["q"]))
                    done_fill.add(tk)
                    break
        if tk not in done_set:
            m = get(f"/markets/{tk}").get("market", {})
            if m.get("result") in ("yes", "no"):
                append(dict(kind="settle", ticker=tk, event=p["event"], result=m["result"]))
    return cmd_report()


def cmd_report() -> int:
    rs = rows()
    posts = {r["ticker"]: r for r in rs if r["kind"] == "post"}
    fills = {r["ticker"]: r for r in rs if r["kind"] == "fill"}
    res = {r["ticker"]: r["result"] for r in rs if r["kind"] == "settle"}
    st = [(tk, posts[tk]) for tk in sorted(fills, key=lambda k: fills[k]["fill_ts"]) if tk in res]
    if not st:
        print(f"posted {len(posts)}, filled {len(fills)}, settled fills 0")
        return 0
    pnl = np.array([(1.0 if res[tk] == p["side"] else 0.0) - p["q"] for tk, p in st])
    cost = np.array([p["q"] for _, p in st])
    losses = [res[tk] != p["side"] for tk, p in st]
    print(f"posted {len(posts)}, filled {len(fills)} ({len(fills) / max(len(posts), 1):.1%}), settled fills {len(st)}, "
          f"losses {sum(losses)}, P&L {pnl.sum():+.2f} units on {cost.sum():.2f} staked, ROI {pnl.sum() / cost.sum():+.2%}")
    if sum(losses[:30]) >= 3:
        print("KILL: >= 3 losses in the first 30 settled fills")
    elif len(st) >= 60:
        print("JUDGE: >= 60 settled fills ->", "KILL (ROI <= 0)" if pnl.sum() <= 0 else "ROI > 0: PM decision")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    sys.exit(dict(tick=cmd_tick, fills=cmd_fills, report=cmd_report)[cmd]())
