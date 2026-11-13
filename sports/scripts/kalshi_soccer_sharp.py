"""KS12: S12 (soft book vs de-vigged Pinnacle, edge >= 3%) with Kalshi soccer as the soft book. DEV / paper only.

    python scripts/kalshi_soccer_sharp.py census     # flagged-history counts by threshold (no outcomes)
    python scripts/kalshi_soccer_sharp.py build      # fetch Kalshi results + pre-kickoff candles for KS12-P bets
    python scripts/kalshi_soccer_sharp.py score      # pre-registered KS12-P (notes/kalshi_soccer_sharp.md)
    python scripts/kalshi_soccer_sharp.py shin       # live snapshot: power vs Shin fair, flags under each

Pre-registration: notes/kalshi_soccer_sharp.md (written before any outcome was fetched).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

QK = Path.home() / ".claude" / "quant-kalshi" / "data"
OUT = ROOT / "results" / "kalshi_soccer_sharp"
SOC = ["epl", "laliga", "seriea", "bund", "ligue1", "efl", "mls", "ucl", "ered"]
THRESH = 0.03
TS0 = pd.Timestamp("1970-01-01", tz="UTC")


def flagged() -> pd.DataFrame:
    g = pd.read_json(QK / "gap_log.jsonl", lines=True)
    s = g[g.league.isin(SOC)].copy()
    s["ts"] = pd.to_datetime(s.ts, utc=True, format="ISO8601")
    s["start"] = pd.to_datetime(s.start, utc=True, format="ISO8601")
    s["event"] = s.ticker.str.rsplit("-", n=1).str[0]
    s["out"] = s.ticker.str.rsplit("-", n=1).str[1]
    return s.sort_values("ts")


def bets(th: float = THRESH) -> pd.DataFrame:
    s = flagged()
    return s[s.ev >= th].drop_duplicates(["ticker", "side"], keep="first").reset_index(drop=True)


def cmd_census() -> int:
    s = flagged()
    print(f"soccer flagged rows {len(s)}  {s.ts.min()} .. {s.ts.max()}")
    for th in (0.01, 0.02, 0.03, 0.05):
        x = s[s.ev >= th].drop_duplicates(["ticker", "side"])
        print(f"EV>={th:.0%}: bets {len(x)} matches {x.event.nunique()} tie-share {(x.out == 'TIE').mean():.2f} "
              f"sides {x.side.value_counts().to_dict()} leagues {x.league.value_counts().to_dict()}")
    t = pd.read_json(QK / "gap_ticks.jsonl", lines=True)
    t["ts"] = pd.to_datetime(t.ts, utc=True, format="ISO8601")
    t["soc"] = t.seen.apply(lambda d: sum(v for k, v in d.items() if k in SOC and isinstance(v, int)))
    print(f"ticks {len(t)} {t.ts.min()} .. {t.ts.max()}; soccer quotes/tick median {t.soc.median():.0f}")
    print(t.groupby(t.ts.dt.date).agg(n=("soc", "size"), soc=("soc", "median")).to_string())
    return 0


def cmd_build() -> int:
    from sports.http import fetch
    from kalshi_longshot import API
    b = bets(0.01)  # build for all >=1% so descriptive thresholds share one fetch
    rows = []
    for ev, gg in b.groupby("event"):
        m = fetch(f"{API}/markets", {"event_ticker": ev, "limit": 100}, cache=False).get("markets", [])
        if not m:
            m = fetch(f"{API}/historical/markets", {"event_ticker": ev, "limit": 100}, cache=True).get("markets", [])
        start = gg.start.iloc[0]
        for mk in m:
            if mk["ticker"] not in set(gg.ticker):
                continue
            series = ev.split("-")[0]
            st = int((start - pd.Timedelta(hours=3) - TS0) // pd.Timedelta(seconds=1))
            en = int((start - TS0) // pd.Timedelta(seconds=1))
            c = fetch(f"{API}/series/{series}/markets/{mk['ticker']}/candlesticks",
                      {"start_ts": st, "end_ts": en, "period_interval": 1}, cache=True).get("candlesticks", [])
            close_bid = close_ask = None
            for x in c:  # keep the last candle with a two-sided quote
                yb, ya = x.get("yes_bid", {}), x.get("yes_ask", {})
                cb = yb.get("close_dollars", yb.get("close"))
                ca = ya.get("close_dollars", ya.get("close"))
                if cb is not None and ca is not None:
                    cb, ca = float(cb), float(ca)
                    if ca > 1.5:  # cents
                        cb, ca = cb / 100, ca / 100
                    if 0 < cb < ca < 1:
                        close_bid, close_ask, close_t = cb, ca, x["end_period_ts"]
            rows.append({"ticker": mk["ticker"], "event": ev, "status": mk.get("status"), "result": mk.get("result"),
                         "close_bid": close_bid, "close_ask": close_ask,
                         "close_min_before_ko": None if close_bid is None else (en - close_t) / 60})
        print(ev, len(m), flush=True)
    r = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_parquet(OUT / "outcomes.parquet", index=False)
    print(f"-> {OUT / 'outcomes.parquet'} {len(r)} rows; results {r.result.value_counts().to_dict()}")
    return 0


def boot(df: pd.DataFrame, rng, n=10000):
    e = df.groupby("event").agg(p=("pnl", "sum"), c=("cost", "sum"))
    P, C = e.p.to_numpy(), e.c.to_numpy()
    idx = rng.integers(0, len(e), (n, len(e)))
    roi = P[idx].sum(1) / C[idx].sum(1)
    return np.percentile(roi, [2.5, 97.5]), float((roi <= 0).mean())


def frame(th: float) -> pd.DataFrame:
    from sports.live import kalshi_fee
    b = bets(th)
    o = pd.read_parquet(OUT / "outcomes.parquet")
    d = b.merge(o, on=["ticker", "event"], how="left")
    d["fee"] = kalshi_fee(d.ask)
    d["cost"] = d.ask + d.fee
    d["won"] = np.where(d.side == "yes", d.result == "yes", d.result == "no")
    d["pnl"] = d.won.astype(float) - d.cost
    mid = (d.close_bid + d.close_ask) / 2
    d["close_val"] = np.where(d.side == "yes", mid, 1 - mid)
    d["clv"] = d.close_val - d.ask
    d["fair"] = np.where(d.side == "yes", d.fair_lo, 1 - d.fair_hi)
    return d


def cmd_score() -> int:
    rng = np.random.default_rng(0)
    for th in (0.01, 0.02, 0.03, 0.05):
        d = frame(th)
        done = d[d.result.isin(["yes", "no"])]
        tag = "KS12-P (PRE-REGISTERED)" if th == THRESH else "descriptive"
        if done.empty:
            print(th, "no settled bets"); continue
        ci, p = boot(done, rng)
        roi = done.pnl.sum() / done.cost.sum()
        cl = done.clv.dropna()
        print(f"EV>={th:.0%} [{tag}] bets {len(done)}/{len(d)} settled, matches {done.event.nunique()}  ROI {roi:+.2%} "
              f"95%CI [{ci[0]:+.2%},{ci[1]:+.2%}] p {p:.3f}  units {done.pnl.sum():+.2f}  "
              f"Kalshi-close CLV mean {cl.mean():+.4f} (n {len(cl)}, >0 share {(cl > 0).mean():.2f})  "
              f"exp EV/ct {(done.fair - done.cost).mean():+.4f} hit {done.won.mean():.3f} vs fair {done.fair.mean():.3f}")
        if th == THRESH:
            done.to_csv(OUT / "ks12p_bets.csv", index=False)
            print(done.groupby(["out_kind", "side"] if "out_kind" in done else ["side"]).agg(
                n=("pnl", "size"), pnl=("pnl", "sum"), clv=("clv", "mean")).round(3).to_string())
            k = done.assign(o=np.where(done.out == "TIE", "D", "team"))
            print(k.groupby(["o", "side"]).agg(n=("pnl", "size"), roi=("pnl", "sum"), cost=("cost", "sum"),
                                               clv=("clv", "mean"), hit=("won", "mean"), fair=("fair", "mean")).round(3).to_string())
            print(k.groupby("league").agg(n=("pnl", "size"), pnl=("pnl", "sum"), clv=("clv", "mean")).round(3).to_string())
    return 0


# ---------------- Shin vs power on a live snapshot ----------------
def shin(r: np.ndarray) -> np.ndarray:
    """Shin (1993) via the Jullien-Salanie / Strumpf iteration: p_i = (sqrt(z^2 + 4(1-z) r_i^2/S) - z) / (2(1-z))."""
    S = r.sum()
    lo, hi = 0.0, 0.5  # sum(p) falls as z rises; bisect for sum(p) = 1
    for _ in range(100):
        z = (lo + hi) / 2
        p = (np.sqrt(z * z + 4 * (1 - z) * r * r / S) - z) / (2 * (1 - z))
        if p.sum() > 1:
            lo = z
        else:
            hi = z
    return p / p.sum()


def cmd_shin() -> int:
    import gap_logger as gl
    from sports import live
    from sports.live import PIN, PIN_HEADERS, _get, am_to_dec, kalshi_fee
    rows = []
    ts = live.now()
    for lg in SOC:
        lid, ser = gl.SOCCER[lg]
        try:
            mu = _get(f"{PIN}/leagues/{lid}/matchups", headers=PIN_HEADERS)
            mk = _get(f"{PIN}/leagues/{lid}/markets/straight", headers=PIN_HEADERS)
            live.LEAGUES.setdefault(lg, (lid, ser, None))
            k = live.kalshi(lg)
        except Exception as e:
            print(lg, "fail", e); continue
        games = {}
        for m in mu:
            if m.get("type") != "matchup" or m.get("parentId") or m.get("isLive"):
                continue
            side = {p["alignment"]: p["name"] for p in m.get("participants", [])}
            if "home" in side and "away" in side:
                games[m["id"]] = (pd.Timestamp(m["startTime"]), side["home"], side["away"])
        for x in mk:
            g = games.get(x["matchupId"])
            if g is None or x.get("type") != "moneyline" or x.get("period") != 0 or x.get("isAlternate") or x.get("status") != "open":
                continue
            pr = {p["designation"]: am_to_dec(p["price"]) for p in x["prices"]}
            if set(pr) != {"home", "draw", "away"} or k.empty:
                continue
            r = np.array([1 / pr["home"], 1 / pr["draw"], 1 / pr["away"]])
            fp, fs = gl._power(r), shin(r)
            hn, an = gl._first(g[1]), gl._first(g[2])
            cand = k[k.event.str.contains(g[0].strftime("%y%b%d").upper(), regex=False)]
            evs = [e for e, gg in cand.groupby("event")
                   if any(hn in gl._first(t) for t in gg.team) and any(an in gl._first(t) for t in gg.team)]
            ok = len(evs) == 1
            for i, s in enumerate("HDA"):
                row = {"lg": lg, "start": g[0], "home": g[1], "away": g[2], "out": s, "dec": [pr["home"], pr["draw"], pr["away"]][i],
                       "overround": r.sum() - 1, "power": fp[i], "shin": fs[i], "prop": r[i] / r.sum(), "matched": ok}
                if ok:
                    q = cand[cand.event == evs[0]]
                    qq = q[q.ticker.str.endswith("-TIE")] if s == "D" else q[q.team.apply(lambda t: (hn if s == "H" else an) in gl._first(t))]
                    if len(qq) == 1:
                        qq = qq.iloc[0]
                        row.update(ticker=qq.ticker, yes_bid=qq.yes_bid, yes_ask=qq.yes_ask, no_ask=qq.no_ask)
                rows.append(row)
    d = pd.DataFrame(rows)
    d = d[(d.start - ts) > pd.Timedelta(minutes=10)]
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / f"shin_snapshot_{ts:%Y%m%dT%H%M}.parquet", index=False)
    print(f"{ts:%F %T} Pinnacle 1X2 rows {len(d)} ({d.drop_duplicates(['home','away']).shape[0]} matches), "
          f"matched to Kalshi {d.ticker.notna().sum() if 'ticker' in d else 0}")
    print("mean (shin - power) by outcome:", d.assign(x=d.shin - d.power).groupby("out").x.agg(["mean", "min", "max"]).round(4).to_string())
    if "ticker" not in d:
        return 0
    q = d.dropna(subset=["ticker"])
    q = q[q.yes_ask.between(0.02, 0.98) & q.no_ask.between(0.02, 0.98)]
    for name in ("power", "shin"):
        fy = (q[name] - q.yes_ask - kalshi_fee(q.yes_ask)) / (q.yes_ask + kalshi_fee(q.yes_ask))
        fn = ((1 - q[name]) - q.no_ask - kalshi_fee(q.no_ask)) / (q.no_ask + kalshi_fee(q.no_ask))
        best = np.maximum(fy, fn)
        print(f"{name:5s}: sides with EV>=1% {(fy >= .01).sum() + (fn >= .01).sum()}, >=3% {(fy >= .03).sum() + (fn >= .03).sum()} "
              f"of {2 * len(q)}; median best-EV {best.median():+.4f}")
        q = q.assign(**{f"ev_yes_{name}": fy, f"ev_no_{name}": fn})
    q["mid_minus_power"] = (q.yes_bid + q.yes_ask) / 2 - q.power
    q["mid_minus_shin"] = (q.yes_bid + q.yes_ask) / 2 - q.shin
    print(q.groupby("out")[["mid_minus_power", "mid_minus_shin"]].mean().round(4).to_string())
    print(q.sort_values("ev_yes_power", ascending=False).head(8)[["lg", "home", "away", "out", "power", "shin", "yes_bid", "yes_ask", "ev_yes_power", "ev_yes_shin"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"census": cmd_census, "build": cmd_build, "score": cmd_score, "shin": cmd_shin}.get(
        cmd, lambda: print(__doc__) or 1)())
