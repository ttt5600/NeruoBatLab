"""nfl_wx_updates: does the Kalshi KXNFLTOTAL ladder lag as-issued weather-forecast updates? Prereg: notes/nfl_wx_updates.md.

    python scripts/nfl_wx_updates.py mos        # IEM NBS + GFS MOS per station x season (cached)
    python scripts/nfl_wx_updates.py events     # forecast vintages -> window values -> events (no prices)
    python scripts/nfl_wx_updates.py kalshi     # KXNFLTOTAL markets + 1-min candles for event/placebo games
    python scripts/nfl_wx_updates.py score      # T1 + descriptive reaction table
Dev = 2025 + 2026 KO < 2026-11-01 (ET). Holdout (KO >= 2026-11-01) is never fetched.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))
from kx_throttle import kget  # noqa: E402
from nfl_kicker_wx import STATIONS, kickoff_utc  # noqa: E402
from kalshi_early_nfl_totals import implied  # noqa: E402

OUT = ROOT / "results" / "nfl_wx_updates"
MOSD = OUT / "mos"
CUT = pd.Timestamp("2026-11-01 04:00")  # 2026-11-01 00:00 ET in UTC (EDT)
LAT = {"NBS": pd.Timedelta(hours=1.5), "GFS": pd.Timedelta(hours=4)}
KT = 1.15078
MAXSPR = 0.10
ALIAS = {"LA": ["LAR", "LA"], "JAX": ["JAC", "JAX"], "WAS": ["WSH", "WAS"], "LV": ["LVR", "LV"]}
SEASON_RANGE = {2025: ("2025-08-25", "2026-02-16"), 2026: ("2026-08-25", "2026-11-02")}


def secs(ts: pd.Series | pd.Timestamp):
    """naive-UTC Timestamp(s) -> epoch seconds (// Timedelta(seconds=1), never // 1e9)."""
    return (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)


def games() -> pd.DataFrame:
    s = pd.read_csv(ROOT / "data" / "nfl_schedule.csv")
    s = s[(s.season.isin([2025, 2026])) & (s.roof == "outdoors") & s.stadium_id.isin(STATIONS)].copy()
    s["ko"] = kickoff_utc(s)
    s = s[s.ko.notna() & (s.ko < CUT)]
    return s.reset_index(drop=True)


def _get(url):
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                t = r.read().decode()
            if "slow down" in t[:200].lower():
                raise RuntimeError("slow down")
            return t
        except Exception as e:  # noqa: BLE001
            time.sleep(15 * (i + 1))
            err = e
    raise err


def cmd_mos():
    MOSD.mkdir(parents=True, exist_ok=True)
    g = games()
    for sid in sorted(g.stadium_id.unique()):
        for y in sorted(g[g.stadium_id == sid].season.unique()):
            for model in ["NBS", "GFS"]:
                for st in STATIONS[sid]:
                    f = MOSD / f"{st}_{model}_{y}.csv"
                    if not f.exists():
                        a, b = SEASON_RANGE[y]
                        url = (f"https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?station={st}&model={model}"
                               f"&sts={a}T00:00Z&ets={b}T00:00Z&format=csv")
                        txt = _get(url)
                        f.write_text(txt)
                        print(sid, st, model, y, txt.count("\n"), flush=True)
                        time.sleep(4)
                    if f.read_text().count("\n") > 2000:
                        break


def load_mos(sid: str, y: int, model: str) -> pd.DataFrame:
    for st in STATIONS[sid]:
        f = MOSD / f"{st}_{model}_{y}.csv"
        if f.exists() and f.read_text().count("\n") > 2000:
            m = pd.read_csv(f, usecols=["runtime", "ftime", "wsp", "p06"])
            m["runtime"] = pd.to_datetime(m.runtime)
            m["ftime"] = pd.to_datetime(m.ftime)
            m["station"] = st
            return m
    return pd.DataFrame()


def window_vals(m: pd.DataFrame, ko: pd.Timestamp) -> pd.DataFrame:
    """Per runtime: wind (mph) over ftime in [KO-1.5h, KO+3h], max p06 over 6h periods overlapping [KO, KO+3h]."""
    rows = []
    for rt, r in m.groupby("runtime"):
        if r.ftime.max() < ko + pd.Timedelta(hours=1):  # KO must be inside the horizon
            continue
        w = r[(r.ftime >= ko - pd.Timedelta(hours=1.5)) & (r.ftime <= ko + pd.Timedelta(hours=3))].wsp.dropna()
        p = r[r.p06.notna() & (r.ftime > ko) & (r.ftime - pd.Timedelta(hours=6) < ko + pd.Timedelta(hours=3))].p06
        rows.append(dict(runtime=rt, wind=w.mean() * KT if len(w) else np.nan, pop=p.max() if len(p) else np.nan))
    return pd.DataFrame(rows)


def cmd_events():
    g = games()
    ev, allupd = [], []
    for _, x in g.iterrows():
        for model in ["NBS", "GFS"]:
            m = load_mos(x.stadium_id, x.season, model)
            if m.empty:
                continue
            m = m[(m.runtime <= x.ko) & (m.runtime >= x.ko - pd.Timedelta(hours=84))]
            v = window_vals(m, x.ko).sort_values("runtime")
            v = v[v.wind.notna()]
            for (_, a), (_, b) in zip(v.iloc[:-1].iterrows(), v.iloc[1:].iterrows()):
                if (b.runtime - a.runtime) != pd.Timedelta(hours=6):
                    continue
                pub = b.runtime + LAT[model]
                # traps: the update must be issued (runtime) and public before the decision; KO after public time
                if pub > x.ko - pd.Timedelta(minutes=20):   # not public in time to trade before KO
                    continue
                assert b.runtime < pub
                dW = b.wind - a.wind
                dP = (b["pop"] - a["pop"]) if pd.notna(a["pop"]) and pd.notna(b["pop"]) else np.nan
                wev = abs(dW) >= 5 and max(a.wind, b.wind) >= 12
                pev = pd.notna(dP) and abs(dP) >= 30 and max(a["pop"], b["pop"]) >= 40
                dirs = set()
                if wev:
                    dirs.add(np.sign(dW))
                if pev:
                    dirs.add(np.sign(dP))
                row = dict(game_id=x.game_id, season=x.season, week=x.week, home=x.home_team, away=x.away_team,
                           gameday=x.gameday, ko=x.ko, total_line=x.total_line, model=model, runtime=b.runtime,
                           pub=pub, h_before=(x.ko - pub) / pd.Timedelta(hours=1), w_old=a.wind, w_new=b.wind,
                           p_old=a["pop"], p_new=b["pop"], dW=dW, dP=dP, wind_ev=wev, pop_ev=pev,
                           kind="event" if (dirs and len(dirs) == 1) else
                           ("conflict" if dirs else ("placebo" if abs(dW) < 2 and (pd.isna(dP) or abs(dP) < 10) else "other")),
                           direction=("UNDER" if dirs == {1.0} else "OVER" if dirs == {-1.0} else ""))
                allupd.append(row)
    u = pd.DataFrame(allupd)
    u = u[(u.h_before <= 72) & (u.h_before >= 20 / 60)]
    OUT.mkdir(parents=True, exist_ok=True)
    u.to_parquet(OUT / "updates.parquet")
    e = u[u.kind == "event"]
    print("games", len(g), "games w/ updates", u.game_id.nunique(), "updates", len(u))
    print(u.kind.value_counts().to_dict())
    print("event games", e.game_id.nunique(), "by model", e.groupby("model").size().to_dict(),
          "dir", e.drop_duplicates("game_id").direction.value_counts().to_dict())


# ---------------------------------------------------------------------------------------------- kalshi
def kmarkets() -> pd.DataFrame:
    f = OUT / "kmarkets.parquet"
    if f.exists():
        return pd.read_parquet(f)
    rows = []
    for path, extra in [("/historical/markets", {}), ("/markets", {"status": "settled"})]:
        cursor = None
        while True:
            p = {"series_ticker": "KXNFLTOTAL", "limit": 1000} | extra | ({"cursor": cursor} if cursor else {})
            d = kget(path, p) or {}
            rows += d.get("markets", [])
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                break
    m = pd.DataFrame(rows).drop_duplicates("ticker")
    m = m[["ticker", "event_ticker", "floor_strike", "result", "open_time", "close_time"]].copy()
    m["strike"] = pd.to_numeric(m.floor_strike, errors="coerce")
    m["date"] = m.event_ticker.str.split("-").str[1].str[:7]
    m["teams"] = m.event_ticker.str.split("-").str[1].str[7:]
    m.to_parquet(f)
    return m


def match(u: pd.DataFrame, m: pd.DataFrame) -> dict:
    ev = m.groupby(["date", "teams"]).event_ticker.first()
    out = {}
    for gid, x in u.drop_duplicates("game_id").set_index("game_id").iterrows():
        d = pd.Timestamp(x.gameday).strftime("%y%b%d").upper()
        hits = {ev[(d, a + h)] for a in ALIAS.get(x.away, [x.away]) for h in ALIAS.get(x.home, [x.home])
                if (d, a + h) in ev.index}
        if len(hits) == 1:
            out[gid] = hits.pop()
    return out


def candles(t: str, s: int, e: int, live: bool) -> list:
    path = (f"/series/KXNFLTOTAL/markets/{t}/candlesticks" if live else f"/historical/markets/{t}/candlesticks")
    j = kget(path, {"start_ts": s, "end_ts": e, "period_interval": 1}) or {}
    out = []
    for c in j.get("candlesticks", []):
        # historical endpoint: "close"/"volume"; live endpoint: "close_dollars"/"volume_fp"
        yb, ya = c.get("yes_bid", {}), c.get("yes_ask", {})
        b = yb.get("close", yb.get("close_dollars"))
        a = ya.get("close", ya.get("close_dollars"))
        out.append((int(c["end_period_ts"]), float(b) if b is not None else np.nan,
                    float(a) if a is not None else np.nan, float(c.get("volume", c.get("volume_fp")) or 0)))
    return out


def cmd_kalshi():
    u = pd.read_parquet(OUT / "updates.parquet")
    use = u[u.kind.isin(["event", "placebo"])]
    m = kmarkets()
    mp = match(use, m)
    print("games", use.game_id.nunique(), "matched", len(mp), flush=True)
    cdir = OUT / "candles"
    cdir.mkdir(exist_ok=True)
    for gid, evt in mp.items():
        f = cdir / f"{gid}.parquet"
        if f.exists():
            continue
        x = use[use.game_id == gid].iloc[0]
        ko = int(secs(x.ko))
        mm = m[m.event_ticker == evt]
        line = x.total_line if pd.notna(x.total_line) else mm.strike.median()
        mm = mm[(mm.strike - line).abs() <= 7.5]
        rows = []
        for _, r in mm.iterrows():
            s = ko - 76 * 3600
            cs = candles(r.ticker, s, ko, live=False)
            if not cs:
                cs = candles(r.ticker, s, ko, live=True)
            rows += [(r.ticker, r.strike, r.result, *c) for c in cs]
        pd.DataFrame(rows, columns=["ticker", "strike", "result", "end", "bid", "ask", "vol"]).to_parquet(f)
        print(gid, evt, len(mm), len(rows), flush=True)


# ---------------------------------------------------------------------------------------------- score
def state_at(c: pd.DataFrame, ts: int):
    """Forward-filled quote per strike at epoch ts (last candle with end <= ts)."""
    c = c[c.end <= ts]
    if c.empty:
        return pd.DataFrame()
    last = c.sort_values("end").groupby("ticker").tail(1).copy()
    # Kalshi omits 1-min candles with no quote change and no trade, so the last candle's quote stands until replaced
    ok = (last.bid > 0) & (last.ask < 1) & ((last.ask - last.bid) <= MAXSPR + 1e-9)
    return last[ok]


def imp_at(c, ts):
    q = state_at(c, ts)
    if len(q) < 2:
        return np.nan
    return implied(q.strike.to_numpy(float), ((q.bid + q.ask) / 2).to_numpy(float))


def boot(v: np.ndarray, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    bs = rng.choice(v, (n, len(v))).mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cmd_score():
    u = pd.read_parquet(OUT / "updates.parquet")
    cdir = OUT / "candles"
    # ---- T1
    ev = u[u.kind == "event"].sort_values("pub").drop_duplicates("game_id")
    bets = []
    for _, x in ev.iterrows():
        f = cdir / f"{x.game_id}.parquet"
        if not f.exists():
            continue
        c = pd.read_parquet(f)
        ko = int(secs(x.ko))
        t = int(secs(x.pub + pd.Timedelta(minutes=5)))
        q = state_at(c, t)
        if q.empty:
            bets.append(dict(game_id=x.game_id, skip="no valid quote")); continue
        q = q.copy()
        under = x.direction == "UNDER"
        q["sask"] = (1 - q.bid) if under else q.ask
        r = q.iloc[(q.sask - 0.5).abs().argsort()].iloc[0]
        # prereg: the strike's last VALID quote before KO (the KO-minute candle is often already pulled wide)
        cc = c[(c.ticker == r.ticker) & (c.end < ko) & (c.bid > 0) & (c.ask < 1) & ((c.ask - c.bid) <= MAXSPR + 1e-9)]
        cl = cc.sort_values("end").tail(1)
        if cl.empty:
            bets.append(dict(game_id=x.game_id, skip="no close quote")); continue
        cm = float((cl.bid.iloc[0] + cl.ask.iloc[0]) / 2)
        close_side = (1 - cm) if under else cm
        ask = float(r.sask)
        fee = 0.07 * ask * (1 - ask)
        win = (r.result == "no") if under else (r.result == "yes")
        bets.append(dict(game_id=x.game_id, season=x.season, week=x.week, model=x.model, direction=x.direction,
                         wind_ev=x.wind_ev, pop_ev=x.pop_ev, dW=x.dW, dP=x.dP, h_before=x.h_before,
                         ticker=r.ticker, strike=r.strike, ask=ask, fee=fee, close_side=close_side,
                         clv=close_side - ask - fee, win=int(win), roi=(int(win) - ask - fee) / (ask + fee),
                         quote_age_min=(t - r.end) / 60, skip=""))
    b = pd.DataFrame(bets)
    b.to_csv(OUT / "t1_bets.csv", index=False)
    ok = b[b.skip == ""]
    res = dict(n=len(ok), skipped=b[b.skip != ""].skip.value_counts().to_dict())
    if len(ok):
        res["clv"] = boot(ok.clv.to_numpy())
        res["roi"] = boot(ok.roi.to_numpy(), seed=1)
        res["win_rate"] = float(ok.win.mean())
        res["mean_ask"] = float(ok.ask.mean())
        res["by_dir"] = {d: dict(n=len(s), clv=float(s.clv.mean()), roi=float(s.roi.mean()))
                         for d, s in ok.groupby("direction")}
        res["by_season"] = {int(d): dict(n=len(s), clv=float(s.clv.mean()), roi=float(s.roi.mean()))
                            for d, s in ok.groupby("season")}
        res["pass"] = bool(len(ok) >= 20 and res["clv"][1] > 0 and res["roi"][1] > 0)
    # ---- descriptive reaction: signed implied move from runtime / public time to +15/60/180 min, and to KO
    rows = []
    for _, x in u[u.kind.isin(["event", "placebo"])].iterrows():
        f = cdir / f"{x.game_id}.parquet"
        if not f.exists():
            continue
        c = pd.read_parquet(f)
        ko = int(secs(x.ko))
        sgn = -1.0 if x.direction == "UNDER" else 1.0 if x.direction == "OVER" else np.nan
        rt, pub = int(secs(x.runtime)), int(secs(x.pub))
        i_rt, i_pub = imp_at(c, rt), imp_at(c, pub)
        d = dict(game_id=x.game_id, kind=x.kind, model=x.model, direction=x.direction, dW=x.dW, dP=x.dP,
                 h_before=x.h_before, i_rt=i_rt, i_pub=i_pub, i_ko=imp_at(c, ko - 60))
        for k in [15, 60, 180]:
            d[f"i_{k}"] = imp_at(c, min(pub + 60 * k, ko - 60))
        d["sgn"] = sgn
        rows.append(d)
    r = pd.DataFrame(rows)
    r.to_parquet(OUT / "reaction.parquet")
    e = r[r.kind == "event"].copy()
    desc = {}
    for lab, a, bcol in [("rt_to_pub", "i_rt", "i_pub"), ("pub_to_15", "i_pub", "i_15"), ("pub_to_60", "i_pub", "i_60"),
                         ("pub_to_180", "i_pub", "i_180"), ("pub_to_ko", "i_pub", "i_ko"), ("rt_to_ko", "i_rt", "i_ko")]:
        mv = ((e[bcol] - e[a]) * e.sgn).dropna()
        pl = (r[r.kind == "placebo"][bcol] - r[r.kind == "placebo"][a]).dropna()
        desc[lab] = dict(event_signed=boot(mv.to_numpy()) if len(mv) > 2 else None, n_event=len(mv),
                         placebo_abs_mean=float(pl.abs().mean()) if len(pl) else None, n_placebo=len(pl))
    res["reaction_pts_toward_event_side"] = desc
    # pre-runtime drift: did the ladder move toward the event side in the 6 h BEFORE the runtime (other models/sources)?
    pre = []
    for _, x in e.iterrows():
        c = pd.read_parquet(cdir / f"{x.game_id}.parquet")
        rt = int(secs(u[(u.game_id == x.game_id) & (u.kind == "event")].runtime.iloc[0]))
        pre.append((x.i_rt - imp_at(c, rt - 6 * 3600)) * x.sgn)
    pre = pd.Series(pre).dropna()
    res["pre_runtime_6h_signed"] = boot(pre.to_numpy()) if len(pre) > 2 else None
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"mos": cmd_mos, "events": cmd_events, "kalshi": cmd_kalshi, "score": cmd_score}[sys.argv[1]]()
