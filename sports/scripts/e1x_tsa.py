"""E1X-T: KXTSAW weekly TSA throughput consensus-reject (taker T1, maker T2). Prereg: notes/e1x_tsa.md.
Stages: audit -> build -> candles -> trades -> score --split dev|sealed
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_family_scan import get  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "e1x"
LEDGER = ROOT / "lab" / "ledger.jsonl"
SERIES = "KXTSAW"
SEED = 20261103
NBOOT = 10_000
M = 0.01
TAKER = 0.07


def tsa() -> pd.Series:
    parts = [pd.read_csv(D / f"tsa_daily_{y}.csv") for y in (2023, 2024, 2025, 2026)]
    t = pd.concat(parts)
    t["date"] = pd.to_datetime(t.date, format="%m/%d/%Y")
    t = t.drop_duplicates("date").set_index("date").n.sort_index().astype(float)
    full = pd.date_range(t.index.min(), t.index.max())
    miss = full.difference(t.index)
    assert len(miss) == 0, f"missing days {list(miss)[:10]}"
    return t


def markets() -> pd.DataFrame:
    df = pd.read_parquet(D / f"{SERIES}_markets.parquet")
    # KXTSAW-* and the legacy TSAW-* ladders (same contract, renamed late 2024)
    df = df[df.event_ticker.str.match(r"^(KX)?TSAW-\d\d[A-Z]{3}\d\d$")].copy()
    df["wend"] = pd.to_datetime(df.event_ticker.str.split("-").str[1], format="%y%b%d")
    kt = df.ticker.str.extract(r"-A(\d+\.\d+)$")[0].astype(float) * 1e6
    df["K"] = df.floor_strike.astype(float).fillna(kt)
    df = df[df.K.notna()]
    n = df.groupby("event_ticker").ticker.transform("size")
    return df[n >= 10]


def weekly(t: pd.Series, wend: pd.Timestamp, sunsat: bool = False) -> float:
    end = wend - pd.Timedelta(days=1) if sunsat else wend
    days = pd.date_range(end - pd.Timedelta(days=6), end)
    if days.min() < t.index.min() or days.max() > t.index.max():
        return np.nan
    return float(t.loc[days].mean())


def cmd_audit() -> int:
    t = tsa()
    df = markets()
    df = df[df.result.isin(["yes", "no"])]
    for sunsat in (False, True):
        ok, n, bad = 0, 0, []
        for ev, g in df.groupby("event_ticker"):
            a = weekly(t, g.wend.iloc[0], sunsat)
            if np.isnan(a):
                continue
            n += 1
            good = all((a > k) == (r == "yes") for k, r in zip(g.K, g.result))
            ok += good
            if not good:
                bad.append((ev, round(a), sorted(g[g.result == "yes"].K)[-1:] , sorted(g[g.result == "no"].K)[:1]))
        print("Sun..Sat" if sunsat else "Mon..Sun", f"consistent {ok}/{n} = {ok / max(n, 1):.3f}")
        for b in bad[:12]:
            print("   ", b)
    return 0


def forecast(t: pd.Series, wend: pd.Timestamp, T: pd.Timestamp) -> float:
    sat = (T.tz_convert("America/New_York").normalize().tz_localize(None))
    while sat.dayofweek != 5:
        sat -= pd.Timedelta(days=1)
    if sat > t.index.max():
        return np.nan
    s = pd.date_range(sat - pd.Timedelta(days=13), sat)
    ly = s - pd.Timedelta(days=364)
    if ly.min() < t.index.min():
        return np.nan
    R = t.loc[s].sum() / t.loc[ly].sum()
    wk = pd.date_range(wend - pd.Timedelta(days=6), wend) - pd.Timedelta(days=364)
    if wk.min() < t.index.min():
        return np.nan
    return float(R * t.loc[wk].mean())


def cmd_build() -> int:
    t = tsa()
    df = markets()
    ev = df.groupby("event_ticker").agg(wend=("wend", "first"), open=("open_time", "min")).reset_index()
    ev["T"] = pd.to_datetime(ev.open, format="ISO8601") + pd.Timedelta(hours=2)
    # residual history for every Sunday week end from 2024-01-21 (forecast built as if decided that Monday +2h)
    sundays = pd.date_range("2024-01-21", t.index.max(), freq="W-SUN")
    hist = []
    for w in sundays:
        Tw = pd.Timestamp(w - pd.Timedelta(days=6), tz="America/New_York") + pd.Timedelta(hours=12)
        f = forecast(t, w, Tw)
        a = weekly(t, w)
        if not (np.isnan(f) or np.isnan(a)):
            hist.append((w, np.log(a / f)))
    h = pd.DataFrame(hist, columns=["wend", "r"])
    h.to_parquet(D / "tsa_resid.parquet")
    rows = []
    for _, e in ev.iterrows():
        f = forecast(t, e.wend, e["T"])
        past = h[h.wend <= (e["T"].tz_convert(None).normalize() - pd.Timedelta(days=8))]
        rows.append(dict(event=e.event_ticker, wend=e.wend, T=e["T"], F=f, n_resid=len(past),
                         rmin=past.r.min() if len(past) else np.nan, rmax=past.r.max() if len(past) else np.nan))
    b = pd.DataFrame(rows)
    b["hi"] = b.F * np.exp(b.rmax + M)
    b["lo"] = b.F * np.exp(b.rmin - M)
    b.to_parquet(D / "tsa_build.parquet")
    print("residual sd", round(h.r.std(), 4), "min/max", round(h.r.min(), 4), round(h.r.max(), 4), "n", len(h))
    print(b.tail(8).to_string())
    return 0


def rungs(split: str | None = None) -> pd.DataFrame:
    b = pd.read_parquet(D / "tsa_build.parquet")
    df = markets()
    x = df.merge(b, left_on="event_ticker", right_on="event")
    x = x[(x.wend >= "2024-07-07") & (x.wend <= "2026-10-25") & (x.n_resid >= 20) & x.F.notna()]
    x["side"] = np.where(x.K >= x.hi, "no", np.where(x.K <= x.lo, "yes", ""))
    x = x[x.side != ""]
    if split:
        evs = np.sort(x.wend.unique())
        cut = evs[int(round(0.7 * len(evs))) - 1]
        x = x[x.wend <= cut] if split == "dev" else x[x.wend > cut]
    return x


def cmd_candles() -> int:
    x = rungs()
    cache = D / "tsa_candles.json"
    out = json.loads(cache.read_text()) if cache.exists() else {}
    for _, m in x.iterrows():
        if m.ticker in out:
            continue
        ts = int(pd.Timestamp(m["T"]).timestamp())
        p = {"start_ts": ts - 6 * 3600, "end_ts": ts, "period_interval": 60}
        if m["_src"] == "/historical/markets":
            r = get(f"/historical/markets/{m.ticker}/candlesticks", p)
        else:
            r = get(f"/series/{SERIES}/markets/{m.ticker}/candlesticks", p)
        out[m.ticker] = r.get("candlesticks", [])
    cache.write_text(json.dumps(out))
    print("rejected rungs", len(x), "events", x.event_ticker.nunique(), "with candles", sum(bool(v) for v in out.values()))
    return 0


def cmd_trades() -> int:
    x = rungs()
    cache = D / "tsa_trades.json"
    out = json.loads(cache.read_text()) if cache.exists() else {}
    for _, m in x.iterrows():
        if m.ticker in out:
            continue
        ep = "/historical/trades" if m["_src"] == "/historical/markets" else "/markets/trades"
        tr, cur = [], None
        while True:
            p = {"ticker": m.ticker, "min_ts": int(pd.Timestamp(m["T"]).timestamp()), "limit": 1000}
            if cur:
                p["cursor"] = cur
            r = get(ep, p)
            tr += r.get("trades", [])
            cur = r.get("cursor")
            if not cur or not r.get("trades"):
                break
        out[m.ticker] = tr
    cache.write_text(json.dumps(out))
    print("tickers", len(out), "trades", sum(len(v) for v in out.values()))
    return 0


def _px(c, side):
    v = c.get(side, {}).get("close_dollars")
    return np.nan if v is None else float(v)


def _yp(t):
    return float(t.get("yes_price_dollars"))


def boot(f: pd.DataFrame, rng):
    g = f.groupby("event_ticker").agg(pnl=("pnl", "sum"), cost=("cost", "sum"))
    roi = g.pnl.sum() / g.cost.sum()
    bs = np.empty(NBOOT)
    pn, co = g.pnl.to_numpy(), g.cost.to_numpy()
    for i in range(NBOOT):
        s = rng.integers(0, len(g), len(g))
        bs[i] = pn[s].sum() / co[s].sum()
    return roi, np.percentile(bs, 2.5), np.percentile(bs, 97.5), float((bs <= 0).mean())


def cmd_score(split: str) -> int:
    x = rungs(split)
    x = x[x.result.isin(["yes", "no"])].copy()
    cs = json.loads((D / "tsa_candles.json").read_text())
    tr = json.loads((D / "tsa_trades.json").read_text())
    yb, ya = [], []
    for tk, T in zip(x.ticker, x["T"]):
        ts = int(pd.Timestamp(T).timestamp())
        c = [k for k in cs.get(tk, []) if k["end_period_ts"] <= ts]
        yb.append(_px(c[-1], "yes_bid") if c else np.nan)
        ya.append(_px(c[-1], "yes_ask") if c else np.nan)
    x["yb"], x["ya"] = yb, ya
    x["win"] = x.result == x.side
    print(f"{split}: rejected rungs {len(x)} in {x.event_ticker.nunique()} events; quote present {x.yb.notna().mean():.3f}; "
          f"rejected-side wrong (outcome) rate {1 - x.win.mean():.4f}")
    rng = np.random.default_rng(SEED)
    rows = []
    # T1 taker
    t1 = x[x.yb.notna()].copy()
    t1["ask"] = np.where(t1.side == "no", 1 - t1.yb, t1.ya)
    t1 = t1[t1.ask <= 0.99 + 1e-9]
    t1["cost"] = t1.ask + TAKER * t1.ask * (1 - t1.ask)
    # T2 maker
    t2 = x[x.yb.notna()].copy()
    t2 = t2[(t2.ya - t2.yb) >= 0.02 - 1e-9]
    t2 = t2[np.where(t2.side == "no", t2.ya <= 0.99 + 1e-9, t2.yb >= 0.01 - 1e-9)]
    t2["q"] = np.where(t2.side == "no", 1 - t2.ya + 0.01, t2.yb + 0.01).round(2)
    fill = []
    for tk, side, q, T in zip(t2.ticker, t2.side, t2.q, t2["T"]):
        t0 = pd.Timestamp(T)
        ok = False
        for k in tr.get(tk, []):
            if pd.Timestamp(k["created_time"]) <= t0:
                continue
            if side == "no" and k["taker_side"] == "yes" and _yp(k) >= 1 - q - 1e-9:
                ok = True
            if side == "yes" and k["taker_side"] == "no" and _yp(k) <= q + 1e-9:
                ok = True
            if ok:
                break
        fill.append(ok)
    t2["filled"] = fill
    posted2 = t2.copy()
    t2 = t2[t2.filled].copy()
    t2["cost"] = t2.q
    for name, f in (("T1", t1), ("T2", t2)):
        extra = {}
        if name == "T2":
            extra = dict(posted=len(posted2), fill_rate=round(len(t2) / max(len(posted2), 1), 3),
                         loss_rate_filled=round(1 - t2.win.mean(), 4) if len(t2) else None,
                         loss_rate_unfilled=round(1 - posted2[~posted2.filled].win.mean(), 4)
                         if (~posted2.filled).any() else None)
        if f.empty:
            r = dict(trial=name, bets=0, **extra)
            print(json.dumps(r))
            rows.append(r)
            continue
        f = f.copy()
        f["pnl"] = f.win.astype(float) - f.cost
        roi, lo, hi, p = boot(f, rng)
        dd = np.sort(f.wend.unique())
        h1, h2 = f[f.wend.isin(dd[: len(dd) // 2])], f[f.wend.isin(dd[len(dd) // 2:])]
        r1 = h1.pnl.sum() / h1.cost.sum() if len(h1) else np.nan
        r2 = h2.pnl.sum() / h2.cost.sum() if len(h2) else np.nan
        r = dict(trial=name, bets=len(f), events=int(f.event_ticker.nunique()), losses=int((~f.win).sum()),
                 roi=round(roi, 4), ci=[round(lo, 4), round(hi, 4)], p=round(p, 4), half1=round(r1, 4),
                 half2=round(r2, 4), mean_cost=round(f.cost.mean(), 3), pnl_units=round(f.pnl.sum(), 3),
                 sides=f.side.value_counts().to_dict(), passed=bool(p < 0.025 and r1 > 0 and r2 > 0), **extra)
        print(json.dumps(r))
        if roi > 0.10 and len(f) > 200:
            print("TOO-GOOD GUARD")
        rows.append(r)
        f.to_csv(D / f"tsa_bets_{name}_{split}.csv", index=False)
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    with LEDGER.open("a") as fh:
        for i, r in enumerate(rows, 1):
            fh.write(json.dumps(dict(
                ts=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), strategy=f"e1x_tsa_{r['trial']}",
                sha=sha, dataset="e1x_tsa",
                hypothesis="KXTSAW consensus-reject: far side of rungs outside F*exp(r_min/max -+ .01) at open+2h, "
                           + ("taker at ask+fee" if r["trial"] == "T1" else "maker 1c inside, tape fills, no cancel"),
                source="agent E1-pattern search 2026-11-01 (notes/e1x_tsa.md)", status="ok",
                split=f"{split.upper()} (70/30 by week end, 2024-07-07..2026-10-25)",
                trial_number=i, study_trials=2, **r)) + "\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    fn = dict(audit=cmd_audit, build=cmd_build, candles=cmd_candles, trades=cmd_trades)
    sys.exit(cmd_score(a.split) if a.stage == "score" else fn[a.stage]())
