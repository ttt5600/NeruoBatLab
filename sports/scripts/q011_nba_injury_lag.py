"""Q011 (NBA): Kalshi reaction lag to official NBA injury-report status changes. Pre-reg: notes/q011_nba_injury_lag.md

    ../.venv_quant/bin/python scripts/q011_nba_injury_lag.py events   # news side only (no prices) -> data/q011/events.parquet, split
    ../.venv_quant/bin/python scripts/q011_nba_injury_lag.py prices   # Kalshi 1-min candles for DEV games only -> data/q011/candles_dev.parquet
    ../.venv_quant/bin/python scripts/q011_nba_injury_lag.py dev [--ledger]

Needs scripts/q011_reports.py fetch + table first.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports.http import fetch
from q011_reports import norm

API = "https://api.elections.kalshi.com/trade-api/v2"
OUTD = ROOT / "data" / "q011"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "q011_nba_injury_lag"
ALIAS = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
LEVEL = {"Out": 2, "Doubtful": 2, "Questionable": 1, "Probable": 0, "Available": 0}
FEE = 0.07
TRIALS = ["T1_game_all", "T2_game_stars30", "T3_spread_mid50", "T4_total_mid50"]
BAR = 0.05 / 4
SERIES = {"T1_game_all": "KXNBAGAME", "T2_game_stars30": "KXNBAGAME", "T3_spread_mid50": "KXNBASPREAD", "T4_total_mid50": "KXNBATOTAL"}


def fee(p):
    return FEE * p * (1 - p)


def ts(x) -> int:
    return int((pd.Timestamp(x) - EPOCH) // pd.Timedelta(seconds=1))


def games() -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(ROOT / "data" / "espn" / "nba" / f"games_{y}.parquet") for y in (2026, 2027)])
    g = g[g.season_type != "preseason"].copy()
    for c in ("home", "away"):
        g[c] = g[c].map(lambda x: ALIAS.get(x, x))
    g["et_date"] = g.start_utc.dt.tz_convert("America/New_York").dt.strftime("%m/%d/%Y")
    g["matchup"] = g.away + "@" + g.home
    return g


def minutes_table(g_all: pd.DataFrame) -> pd.DataFrame:
    b = pd.concat([pd.read_parquet(ROOT / "data" / "espn" / "nba" / f"box_{y}.parquet") for y in (2025, 2026, 2027)])
    b = b[b.stat == "minutes"].copy()
    b["min"] = pd.to_numeric(b.value, errors="coerce")
    gg = pd.concat([pd.read_parquet(ROOT / "data" / "espn" / "nba" / f"games_{y}.parquet") for y in (2025, 2026, 2027)])
    tid = pd.concat([gg[["event_id", "home_team_id", "home"]].set_axis(["event_id", "team_id", "abbr"], axis=1),
                     gg[["event_id", "away_team_id", "away"]].set_axis(["event_id", "team_id", "abbr"], axis=1)])
    b = b.merge(gg[["event_id", "start_utc", "season_type"]], on="event_id").merge(tid, on=["event_id", "team_id"], how="left")
    b = b[(b.season_type != "preseason") & (b["min"] > 0)]
    b["abbr"] = b.abbr.map(lambda x: ALIAS.get(x, x))
    b["pkey"] = b.name.map(norm)
    return b[["athlete_id", "pkey", "abbr", "start_utc", "min"]].sort_values("start_utc")


def cmd_events() -> int:
    idx = pd.read_parquet(OUTD / "report_index.parquet")
    rows = pd.read_parquet(OUTD / "report_rows.parquet")
    idx = idx[idx.http == 200].copy()
    idx["pub"] = idx[["lm", "hdr"]].max(axis=1)
    disagree = ((idx.lm - idx.hdr).abs() > pd.Timedelta(minutes=10)).sum()
    idx["era"] = np.where(idx.label.str.count("_") == 1, "hourly", "q15")
    print(f"reports 200: {len(idx)} (hourly {int((idx.era == 'hourly').sum())}, 15-min {int((idx.era == 'q15').sum())}); "
          f"|LastModified - header| > 10 min: {disagree}; header missing {idx.hdr.isna().sum()}, LM missing {idx.lm.isna().sum()}")
    lmh = (idx.lm - idx.hdr).dt.total_seconds()
    print(f"LastModified - header (s): median {lmh.median():.0f}, p1 {lmh.quantile(.01):.0f}, p99 {lmh.quantile(.99):.0f}, max {lmh.max():.0f}")
    rows = rows.merge(idx[["label", "pub", "era"]], on="label")
    g = games()
    gk = g.set_index(["et_date", "matchup"])
    rows = rows[rows.gdate.notna() & rows.matchup.notna()]
    rows = rows.join(gk[["event_id", "start_utc", "home", "away"]], on=["gdate", "matchup"], how="inner")
    print(f"rows joined to ESPN games: {len(rows)}; games {rows.event_id.nunique()}")
    mins = minutes_table(g)
    ev = []
    for eid, d in rows.groupby("event_id"):
        tip = d.start_utc.iloc[0]
        reps = d[["label", "pub", "era"]].drop_duplicates("label").sort_values("pub")
        by = {lab: x for lab, x in d.groupby("label")}
        labs = reps.label.tolist()
        for a, bl in zip(labs[:-1], labs[1:]):
            ra, rb = reps.set_index("label").loc[a], reps.set_index("label").loc[bl]
            gap = (rb.pub - ra.pub).total_seconds() / 60
            if gap > (65 if rb.era == "hourly" else 20):
                continue
            if not (tip - pd.Timedelta(hours=10) <= rb.pub <= tip - pd.Timedelta(minutes=10)):
                continue
            A, B = by[a], by[bl]
            for team in (d.home.iloc[0], d.away.iloc[0]):
                ta, tb = A[A.team == team], B[B.team == team]
                if (ta.status == "NOT_SUBMITTED").any() or (tb.status == "NOT_SUBMITTED").any() or ta.empty or tb.empty:
                    continue
                la = {k: LEVEL[s] for k, s in zip(ta.pkey, ta.status) if s in LEVEL}
                lb = {k: LEVEL[s] for k, s in zip(tb.pkey, tb.status) if s in LEVEL}
                for k in set(la) | set(lb):
                    x, y = la.get(k, 0), lb.get(k, 0)
                    if x != y:
                        ev.append({"event_id": eid, "tip": tip, "home": d.home.iloc[0], "away": d.away.iloc[0], "team": team,
                                   "pkey": k, "lv_prev": x, "lv_cur": y, "t0": rb.pub, "t_prev": ra.pub, "label": bl, "era": rb.era})
    ev = pd.DataFrame(ev)
    print(f"raw status changes in window: {len(ev)}")
    # impact join (strictly before the game date)
    out = []
    unmatched = ambiguous = 0
    for r in ev.itertuples():
        m = mins[(mins.pkey == r.pkey) & (mins.start_utc < r.tip - pd.Timedelta(hours=6))]
        if m.empty:
            unmatched += 1
            continue
        if m.athlete_id.nunique() > 1:
            m2 = m[m.abbr == r.team]
            if m2.athlete_id.nunique() != 1:
                ambiguous += 1
                continue
            m = m[m.athlete_id == m2.athlete_id.iloc[0]]
        last = m.tail(10)
        out.append({**r._asdict(), "n_games10": len(last), "min10": last["min"].mean()})
    ev = pd.DataFrame(out).drop(columns="Index")
    print(f"name join: unmatched {unmatched} (rookies/two-ways/no box), ambiguous {ambiguous}")
    ev["impact"] = (ev.n_games10 >= 5) & (ev.min10 >= 25.0)
    ev["dir"] = np.sign(ev.lv_cur - ev.lv_prev).astype(int)
    ev["benefit"] = np.where(ev.dir > 0, np.where(ev.team == ev.home, ev.away, ev.home), ev.team)
    ev["et_date"] = ev.tip.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    q = ev[ev.impact].copy()
    # flip-back check (data quality): change reverted in the very next report
    print(f"impact events: {len(q)}; by transition:\n{q.groupby(['lv_prev', 'lv_cur']).size().to_string()}")
    dates = sorted(q.et_date.unique())
    cut = dates[int(np.floor(0.7 * len(dates))) - 1]
    q["split"] = np.where(q.et_date <= cut, "dev", "sealed")
    print(f"qualifying dates {len(dates)} ({dates[0]} .. {dates[-1]}); DEV <= {cut}")
    print(q.groupby("split").agg(events=("pkey", "size"), games=("event_id", "nunique"), stars=("min10", lambda s: (s >= 30).sum())).to_string())
    print("events by era:", q.groupby(["split", "era"]).size().to_dict())
    print("t0 minutes before tip (dev):", ((q[q.split == 'dev'].tip - q[q.split == 'dev'].t0).dt.total_seconds() / 60).describe().round(0).to_dict())
    ev.to_parquet(OUTD / "events_all.parquet", index=False)
    q.to_parquet(OUTD / "events.parquet", index=False)
    (OUTD / "split.json").write_text(json.dumps({"dev_last_date": cut, "n_dates": len(dates), "frozen_utc": datetime.now(timezone.utc).isoformat()}))
    return 0


def markets(series: str) -> pd.DataFrame:
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
    m = pd.DataFrame(rows).drop_duplicates("ticker")
    m["kdate"] = m.event_ticker.str.split("-").str[1].str[:7]
    m["teams"] = m.event_ticker.str.split("-").str[1].str[7:]
    m["code"] = m.ticker.str.split("-").str[-1]
    m["open_ts"] = pd.to_datetime(m.open_time, utc=True, format="ISO8601")
    if "floor_strike" not in m:
        m["floor_strike"] = np.nan
    return m[["ticker", "event_ticker", "kdate", "teams", "code", "open_ts", "result", "floor_strike"]]


def candles(ticker: str, series: str, s0: int, e0: int) -> pd.DataFrame:
    c = None
    for path in (f"{API}/historical/markets/{ticker}/candlesticks", f"{API}/series/{series}/markets/{ticker}/candlesticks"):
        try:
            j = fetch(path, {"start_ts": s0, "end_ts": e0, "period_interval": 1}, cache=True)
        except Exception:
            continue
        if j.get("candlesticks"):
            c = j["candlesticks"]
            break
    if not c:
        return pd.DataFrame()
    def f(x, kk):  # historical: 'close' = dollars string; live: 'close_dollars' (cents int in 'close' -> /100)
        d = x.get(kk) or {}
        v = pd.to_numeric(d.get("close_dollars", d.get("close")), errors="coerce")
        return v / 100 if v == v and v > 1.5 else v
    return pd.DataFrame([{"end": x["end_period_ts"], "bid": f(x, "yes_bid"), "ask": f(x, "yes_ask"),
                          "vol": pd.to_numeric(x.get("volume_fp", x.get("volume")), errors="coerce")} for x in c]).assign(ticker=ticker)


def ladder_axis(m: pd.DataFrame, s: str, home: str) -> pd.DataFrame:
    """map ladder markets to one axis x with p(x) = P(home margin > x) (spread) or P(total > x), decreasing in x."""
    m = m.copy()
    k = pd.to_numeric(m.floor_strike, errors="coerce")
    k = k.where(k.notna(), pd.to_numeric(m.code.str.extract(r"(\d+)$")[0], errors="coerce"))
    if s == "KXNBASPREAD":
        m["team"] = m.code.str.rstrip("0123456789")
        m["x"] = np.where(m.team == home, k, -k)
        m["flip"] = m.team != home
    else:
        m["x"], m["flip"] = k, False
    return m.dropna(subset=["x"]).sort_values("x").reset_index(drop=True)


def cmd_prices() -> int:
    """KXNBAGAME: both markets. Ladders: expand outward from the middle strike until, at every decision time
    (t0 + 2/3/5 min), the fetched range brackets p = 0.5 (or hits the ladder end). With a monotone ladder the
    market closest to 0.50 (the pre-registered choice) is then always inside the fetched range; this only saves
    requests (Kalshi was returning 429s, shared with another session's pull)."""
    q = pd.read_parquet(OUTD / "events.parquet")
    q = q[q.split == "dev"]
    gm = q.groupby("event_id").agg(tip=("tip", "first"), home=("home", "first"), away=("away", "first"), t0min=("t0", "min"),
                                    t0s=("t0", lambda x: sorted(set(x)))).reset_index()
    print(f"dev games to price: {len(gm)}", flush=True)
    frames, nmk, nreq = [], {}, {}
    import os
    for s in os.environ.get("Q011_SERIES", "KXNBAGAME,KXNBASPREAD,KXNBATOTAL").split(","):
        mk = markets(s)
        print(f"{s}: {len(mk)} settled markets", flush=True)
        for i, r in enumerate(gm.itertuples()):
            kd = r.tip.tz_convert("America/New_York").strftime("%y%b%d").upper()
            m = mk[(mk.kdate == kd) & (mk.teams == r.away + r.home) & (mk.open_ts < r.tip)]
            if m.empty:
                nmk[s] = nmk.get(s, 0) + 1
                continue
            s0, e0 = ts(r.t0min) - 75 * 60, ts(r.tip) + 120
            tag = lambda cd, k: cd.assign(event_id=r.event_id, series=s, code=k.code, result=k.result, strike=k.floor_strike)
            if s == "KXNBAGAME":
                for k in m.itertuples():
                    cd = candles(k.ticker, s, s0, e0)
                    nreq[s] = nreq.get(s, 0) + 1
                    if not cd.empty:
                        frames.append(tag(cd, k))
                continue
            L = ladder_axis(m, s, r.home)
            got: dict[int, pd.DataFrame] = {}
            def load(j):
                if j not in got:
                    k = L.iloc[j]
                    got[j] = candles(k.ticker, s, s0, e0)
                    nreq[s] = nreq.get(s, 0) + 1
            def pval(j, t):
                c = got[j]
                if c.empty:
                    return None
                qa = quote_at(c, t)
                if not qa or qa[1] - qa[0] > 0.10:
                    return None
                p = (qa[0] + qa[1]) / 2
                return 1 - p if L.flip.iloc[j] else p
            lo = hi = len(L) // 2
            load(lo)
            tds = [ts(t) + 60 * d for t in r.t0s for d in (2, 3, 5)]
            for _ in range(len(L)):
                need_lo = need_hi = False
                for t in tds:
                    plo, phi = pval(lo, t), pval(hi, t)
                    if lo > 0 and (plo is None or plo < 0.5):
                        need_lo = True
                    if hi < len(L) - 1 and (phi is None or phi > 0.5):
                        need_hi = True
                if not (need_lo or need_hi):
                    break
                if need_lo:
                    lo -= 1
                    load(lo)
                if need_hi:
                    hi += 1
                    load(hi)
            for j, cd in got.items():
                if not cd.empty:
                    frames.append(tag(cd, L.iloc[j]))
            if i % 50 == 0:
                print(f"  {s} {i}/{len(gm)} requests so far {nreq}", flush=True)
    cd = pd.concat(frames, ignore_index=True)
    cd.to_parquet(OUTD / os.environ.get("Q011_CANDLES", "candles_dev.parquet"), index=False)
    print(f"-> candles_dev.parquet {len(cd)} rows, {cd.ticker.nunique()} markets; games without markets by series {nmk}; requests {nreq}")
    return 0


def quote_at(c: pd.DataFrame, t: int):
    """last two-sided candle with end <= t -> (bid, ask) or None"""
    x = c[(c.end <= t)]
    x = x[(x.bid > 0) & (x.ask < 1) & (x.bid <= x.ask)]
    if x.empty:
        return None
    r = x.iloc[-1]
    return r.bid, r.ask


def first_entry(c: pd.DataFrame, td: int, tip: int):
    x = c[(c.end >= td) & (c.end < tip) & (c.end <= td + 600)]
    x = x[(x.bid > 0) & (x.ask < 1) & (x.bid <= x.ask) & (x.ask - x.bid <= 0.10)]
    return None if x.empty else x.iloc[0]


def side_px(bid, ask, yes: bool):
    """(entry ask, mid) for the side bought"""
    return (ask, (bid + ask) / 2) if yes else (1 - bid, 1 - (bid + ask) / 2)


def build_bets(q: pd.DataFrame, cd: pd.DataFrame, trial: str, L: int = 3) -> pd.DataFrame:
    s = SERIES[trial]
    qq = q[q.min10 >= 30] if trial == "T2_game_stars30" else q
    cs = {t: x.sort_values("end") for t, x in cd[cd.series == s].groupby("ticker")}
    meta = cd[cd.series == s].drop_duplicates("ticker").set_index("ticker")
    out, miss = [], {"no_market": 0, "no_entry": 0, "no_tip": 0}
    for r in qq.itertuples():
        t0, tip = ts(r.t0), ts(r.tip)
        td = t0 + L * 60
        mk = meta[meta.event_id == r.event_id]
        if mk.empty:
            miss["no_market"] += 1
            continue
        if s == "KXNBAGAME":
            tk = mk[mk.code == r.benefit]
            if tk.empty:
                miss["no_market"] += 1
                continue
            ticker, yes = tk.index[0], True
        else:  # ladder: market with mid closest to .5 at td
            best = None
            for t in mk.index:
                qa = quote_at(cs[t], td)
                if qa and qa[1] - qa[0] <= 0.10:
                    dist = abs((qa[0] + qa[1]) / 2 - 0.5)
                    if best is None or dist < best[0]:
                        best = (dist, t)
            if best is None:
                miss["no_entry"] += 1
                continue
            ticker = best[1]
            if s == "KXNBASPREAD":
                yes = meta.loc[ticker].code.rstrip("0123456789") == r.benefit
            else:
                yes = r.dir < 0  # player back -> over
        c = cs[ticker]
        e = first_entry(c, td, tip)
        if e is None:
            miss["no_entry"] += 1
            continue
        qt = quote_at(c, tip)
        if qt is None:
            miss["no_tip"] += 1
            continue
        px, mid_e = side_px(e.bid, e.ask, yes)
        _, mid_tip = side_px(qt[0], qt[1], yes)
        pre = {}
        for lab, dt in (("m60", -3600), ("m15", -900), ("p0", 0), ("p5", 300), ("p15", 900), ("p60", 3600)):
            qa = quote_at(c, t0 + dt) if t0 + dt < tip else qt
            pre[lab] = side_px(qa[0], qa[1], yes)[1] if qa else np.nan
        won = float((meta.loc[ticker].result == "yes") == yes)
        out.append({"event_id": r.event_id, "t0": r.t0, "et_date": r.et_date, "ticker": ticker, "yes": yes, "pkey": r.pkey,
                    "dir": r.dir, "lv_prev": r.lv_prev, "lv_cur": r.lv_cur, "min10": r.min10, "entry_end": e.end, "lag_s": e.end - t0,
                    "px": px, "mid_entry": mid_e, "mid_tip": mid_tip, "won": won, **pre,
                    "clv": mid_tip - px - fee(px), "clv_ceil": mid_tip - px - np.ceil(100 * fee(px)) / 100,
                    "roi": (won - px - fee(px)) / px, "drift": mid_tip - mid_e})
    b = pd.DataFrame(out)
    if b.empty:
        return b, miss
    # dedup: same game, same t0, same market side -> one bet; opposite sides at same t0 in one game -> drop both
    key = b.groupby(["event_id", "t0"]).apply(lambda x: (x.ticker + x.yes.astype(str)).nunique(), include_groups=False)
    conflict = key[key > 1].index
    n_conf = b.set_index(["event_id", "t0"]).index.isin(conflict).sum()
    b = b[~b.set_index(["event_id", "t0"]).index.isin(conflict)]
    b = b.drop_duplicates(["event_id", "t0", "ticker", "yes"]).reset_index(drop=True)
    miss["conflict_dropped"] = int(n_conf)
    return b, miss


def boot(v: np.ndarray, g: np.ndarray, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in ug}
    sums = np.array([v[idx[k]].sum() for k in ug])
    cnt = np.array([len(idx[k]) for k in ug])
    pick = rng.integers(0, len(ug), size=(n, len(ug)))
    means = sums[pick].sum(1) / cnt[pick].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((means <= 0).mean()), float(means.mean())


def cmd_dev(ledger: bool) -> int:
    q = pd.read_parquet(OUTD / "events.parquet")
    q = q[q.split == "dev"]
    import os
    cd = pd.read_parquet(OUTD / os.environ.get("Q011_CANDLES", "candles_dev.parquet"))
    sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    res = {}
    for i, t in enumerate(TRIALS, 1):
        if SERIES[t] not in set(cd.series):
            print(f"{t}: no {SERIES[t]} candles in this file; skipped")
            continue
        b, miss = build_bets(q, cd, t)
        b.to_csv(OUTD / f"bets_{t}.csv", index=False)
        v, g = b.clv.values, b.event_id.values
        lo, hi, p, bm = boot(v, g)
        med = np.sort(b.et_date.unique())[len(b.et_date.unique()) // 2]
        h1, h2 = b[b.et_date < med], b[b.et_date >= med]
        rl, rh, rp, _ = boot(b.roi.values, g)
        dl, dh, dp, _ = boot(b.drift.values, g)
        sens = {}
        for L in (2, 5):
            bl, _ = build_bets(q, cd, t, L)
            sl = boot(bl.clv.values, bl.event_id.values)
            sens[f"L{L}"] = {"n": len(bl), "clv": float(bl.clv.mean()), "ci": [sl[0], sl[1]], "p": sl[2]}
        bdir = b.assign(pre60=b.p0 - b.m60, pre15=b.p0 - b.m15, tot=b.mid_tip - b.m60)
        passed = bool(v.mean() > 0 and p < BAR and h1.clv.mean() > 0 and h2.clv.mean() > 0)
        r = {"trial": t, "bets": len(b), "games": int(b.event_id.nunique()), "miss": miss,
             "clv": float(v.mean()), "ci": [lo, hi], "p": p, "boot_mean": bm,
             "half1": float(h1.clv.mean()), "half2": float(h2.clv.mean()), "half1_n": len(h1), "half2_n": len(h2),
             "clv_ceil_fee": float(b.clv_ceil.mean()), "drift": float(b.drift.mean()), "drift_ci": [dl, dh],
             "roi": float(b.roi.mean()), "roi_ci": [rl, rh], "roi_p": rp, "win_rate": float(b.won.mean()), "mean_px": float(b.px.mean()),
             "mean_spread_paid": float((b.px - b.mid_entry).mean()), "median_lag_s": float(b.lag_s.median()),
             "path": {k: float((b[k] - b.p0).mean()) for k in ("m60", "m15", "p5", "p15", "p60", "mid_tip")},
             "pre60_mean": float(bdir.pre60.mean()), "share_pre60_ge2c": float((bdir.pre60 >= 0.02).mean()),
             "share_pre60_le_m2c": float((bdir.pre60 <= -0.02).mean()),
             "share_of_move_before_t0": float(bdir.pre60.sum() / bdir.tot.sum()) if bdir.tot.sum() != 0 else None,
             "by_dir": {int(k): {"n": len(x), "clv": float(x.clv.mean())} for k, x in b.groupby("dir")},
             "sens_latency": sens, "passed": passed}
        res[t] = r
        print(json.dumps(r, indent=1, default=float))
        if ledger:
            rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"q011_{t}", "sha": sha,
                   "dataset": DATASET, "hypothesis": "pre-registered in notes/q011_nba_injury_lag.md; buy benefiting side at ask "
                   "first quote >= report pub + 3 min; metric net CLV to tip mid", "source": "QUESTIONS Q011 (NBA)", "status": "ok",
                   "split": "DEV (first 70% of event dates)", "trial_number": i, "study_trials": 4, "bonferroni_bar": BAR,
                   "bets": len(b), "units": float(b.clv.sum()), "roi": float(b.roi.mean()), "ci_lo": lo, "ci_hi": hi, "p": p,
                   "metric": "net_clv_per_contract", "clv": float(v.mean()), "roi_ci": [rl, rh],
                   "half1": r["half1"], "half2": r["half2"], "half1_n": len(h1), "half2_n": len(h2), "boot_mean": bm,
                   "game_clustered": True, "passed": passed, "verdict": "PASS (dev)" if passed else "FAIL"}
            with open(LEDGER, "a") as f:
                f.write(json.dumps(rec) + "\n")
    (OUTD / "dev_results.json").write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    a = sys.argv[1]
    sys.exit(cmd_events() if a == "events" else cmd_prices() if a == "prices" else cmd_dev("--ledger" in sys.argv))
