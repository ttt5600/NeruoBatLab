"""L38 IN-GAME LONGSHOT: do Kalshi in-game prices overprice TRAILING longshots enough to sell them after fees? DEV ONLY.

    python scripts/ingame_longshot.py build-nhl     # ESPN NHL 2025 Apr-Jun games + KXNHLGAME 1-min candles (DEV only)
    python scripts/ingame_longshot.py snap          # snapshots per sport/bin (NO outcomes touched) + trade tapes
    python scripts/ingame_longshot.py dev [--ledger]

Source of the idea: L37 sealed run (results/nba_l37_sealed.json, descriptive) -- Kalshi NBA trailer mids ran ~1.5c above
realised in 0-15c bins. Those 2026-01-15+ games are SEALED/used and are NEVER read here. L28 (sealed fail): cheap
leaders were adversely selected -> this test buys the leader only at whatever ask exists, no "cheap leader" filter.

PRE-REGISTERED 2026-10-27, written before any outcome was joined to a snapshot in this study
----------------------------------------------------------------------------------------------
DATA / DEV WINDOWS (sealed data is filtered out at load, before anything else is computed):
  NBA : data/nba_ingame.parquet rows with date 2025-10-21 .. 2026-01-14 (ET) only. In-game span = first..last ESPN play
        wallclock (data/nba_ingame_plays.parquet). Tickers re-derived from settled KXNBAGAME exactly as nba_ingame.build.
  NFL : data/nfl_ingame.parquet games with season 2025, REG, week <= 12 only (nfl_ingame_games.parquet start/end).
  NHL : NHL 2025-26 is SEALED by the frozen MANIFEST (sealed nhl: 2026; notes/nhl_simple_model.md), so the brief's
        "first 40% of 2025-26" is NOT used. NHL DEV = KXNHLGAME games dated 2025-04-01 .. 2025-06-30 (end of the 2024-25
        regular season + 2025 playoffs), ESPN completed games; in-game span = first..last ESPN play wallclock.
        Only market LISTINGS (counts by month) were looked at before this was written; no NHL price was pulled.
SNAPSHOT: 1-min candles (close yes_bid / yes_ask) for both team markets of a game, joined on candle end. A candle is
  in-game iff first play wallclock < end <= last play wallclock. Trailer = the side whose own-market mid < 0.5.
  Eligible: trailer market two-sided (0 < bid < ask < 1) with spread <= 3c; leader market ask in (0, 1).
  BINS on trailer mid m: B1 [0.01, 0.05), B2 [0.05, 0.10), B3 [0.10, 0.15), B4 [0.15, 0.25).
  For each game and each bin: the FIRST eligible in-game candle whose trailer mid is in that bin (one trade per game
  per bin; a game can contribute to several bins).
TAKER (primary): buy 1 leader YES at the leader market's close ask a in that same candle; cost a + 0.07*a*(1-a)
  (unrounded = large-order fee; the 1-contract ceil-to-cent fee is a sensitivity, not scored). Payoff 1 iff the
  leader market settles yes. Sensitivities (not trials): (i) buy trailer-NO at 1 - trailer bid instead; (ii) 1-min
  latency = leader ask of the NEXT candle (skip if missing); (iii) rounded fee.
MAKER: at the snapshot candle end T rest a sell of trailer YES at p = ceil-to-cent(m) (== NO-on-trailer bid at
  1 - p), good for 10 minutes. FILL iff the trailer market's public tape (GET /historical/trades, fallback
  /markets/trades) shows a trade in (T, T+600] with yes_price > p (strict print-through; '>= p' reported as
  sensitivity). Payoff = p - 1{trailer wins} - maker fee 0.0175*p*(1-p) (repo convention, conservative). Unfilled =
  no bet. If tapes cannot be fetched for >= 80% of the snapshots in a bin, that maker trial is 'not scored'.
  Adverse-selection check: trailer win rate conditional on fill vs all snapshots in the bin.
METRICS: per sport x bin and pooled: n games, mean trailer mid, realised trailer win rate, (realised - mid) with
  game-bootstrap 95% CI; taker P&L per contract (cents), ROI = P&L / cost; both chronological halves (split at the
  median game date within each sport). BREAK-EVEN printed explicitly: required leader win rate = mean(a + fee), i.e.
  the trailer must win less than (1 - a - fee), so the needed overpricing vs mid = spread/2-ish + fee.
TRIALS (dataset ingame_longshot, new): pooled over sports, per bin x execution: taker_B1..B4, maker_B1..B4 -> 8 trials,
  bar p < 0.05/8 = 0.00625. Inference: one-sided game bootstrap (10,000 draws, seed 0) of the mean P&L per trade;
  p = share of draws with mean <= 0. PASS = mean > 0, p < 0.00625, and same sign in both halves. Per-sport rows are
  descriptive only. A pass here is a DEV pass -> forward/sealed test only after the user's yes.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports.http import fetch
from kalshi_longshot import API, settled

DATA = ROOT / "data"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "ingame_longshot"
NHL_C = DATA / "ingame_longshot_nhl_candles.parquet"
NHL_G = DATA / "ingame_longshot_nhl_games.parquet"
SNAP = DATA / "ingame_longshot_snaps.parquet"
TAPE = DATA / "ingame_longshot_tapes.parquet"
OUTJ = ROOT / "results" / "ingame_longshot_dev.json"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
BINS = [("B1", 0.01, 0.05), ("B2", 0.05, 0.10), ("B3", 0.10, 0.15), ("B4", 0.15, 0.25)]
TAKER, MAKER = 0.07, 0.0175
NBA_LO, NBA_HI = pd.Timestamp("2025-10-21"), pd.Timestamp("2026-01-14")
NHL_LO, NHL_HI = "20250401", "20250630"
NBA_ALIAS = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
NHL_ESPN = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"
NBA_ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
TRIALS = [f"{x}_{b}" for x in ("taker", "maker") for b, _, _ in BINS]


def esec(t: pd.Series) -> np.ndarray:
    return ((t - EPOCH) // pd.Timedelta(seconds=1)).to_numpy()


def kcandles(series: str, tk: str, s: int, e: int) -> pd.DataFrame:
    rows = []
    for path in (f"{API}/historical/markets/{tk}/candlesticks", f"{API}/series/{series}/markets/{tk}/candlesticks"):
        try:
            j = fetch(path, {"start_ts": s, "end_ts": e, "period_interval": 1}, cache=True)
        except Exception:
            continue
        c = j.get("candlesticks", [])
        if c:
            for x in c:
                g = lambda k: pd.to_numeric((x.get(k) or {}).get("close_dollars", (x.get(k) or {}).get("close")), errors="coerce")
                rows.append({"end": x["end_period_ts"], "bid": g("yes_bid"), "ask": g("yes_ask")})
            break
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ NHL build (dev only)
def cmd_build_nhl() -> int:
    games = []
    for d in pd.date_range(NHL_LO, NHL_HI):
        sb = fetch(f"{NHL_ESPN}/scoreboard", {"dates": d.strftime("%Y%m%d")}, cache=True)
        for e in sb.get("events", []):
            c = e["competitions"][0]
            if c.get("status", {}).get("type", {}).get("completed") is not True:
                continue
            tm = {x["homeAway"]: x["team"]["abbreviation"] for x in c["competitors"]}
            games.append({"eid": e["id"], "home": tm["home"], "away": tm["away"], "stype": e.get("season", {}).get("type")})
    g = pd.DataFrame(games).drop_duplicates("eid")
    span = []
    for eid in g.eid:
        s = fetch(f"{NHL_ESPN}/summary", {"event": eid}, cache=True)
        w = [p["wallclock"] for p in s.get("plays", []) if p.get("wallclock")]
        span.append((pd.to_datetime(min(w), utc=True) if w else pd.NaT, pd.to_datetime(max(w), utc=True) if w else pd.NaT))
    g["start"], g["end"] = [a for a, _ in span], [b for _, b in span]
    g = g.dropna(subset=["start"])
    mk = settled("KXNHLGAME")
    mk["code"] = mk.ticker.str.split("-").str[-1]
    mk["kdate"] = mk.event_ticker.str.split("-").str[1].str[:7]
    alias = {"TB": "TBL", "NJ": "NJD", "LA": "LAK", "SJ": "SJS", "UTAH": "UTA", "WSH": "WSH"}
    frames, unmatched, rows = [], [], []
    for _, r in g.iterrows():
        kd = r.start.tz_convert("America/New_York").strftime("%y%b%d").upper()
        codes = {}
        for side, ab in (("home", r.home), ("away", r.away)):
            cand = {ab, alias.get(ab, ab)} | {k for k, v in alias.items() if v == ab}
            m = mk[(mk.kdate == kd) & mk.code.isin(cand)]
            if len(m) == 1:
                codes[side] = m.iloc[0]
        if len(codes) != 2 or codes["home"].event_ticker != codes["away"].event_ticker:
            unmatched.append(f"{kd} {r.away}@{r.home}")
            continue
        s0, e0 = int(r.start.timestamp()) - 900, int(r.end.timestamp()) + 900
        for side, k in codes.items():
            c = kcandles("KXNHLGAME", k.ticker, s0, e0)
            if c.empty:
                continue
            frames.append(c.assign(gid=r.eid, side=side, ticker=k.ticker, result=k.result))
        rows.append(r)
    pd.concat(frames, ignore_index=True).to_parquet(NHL_C, index=False)
    pd.DataFrame(rows).to_parquet(NHL_G, index=False)
    print(f"NHL dev: {len(g)} ESPN games, {len(rows)} matched, unmatched {len(unmatched)}: {unmatched[:20]}")
    return 0


# ------------------------------------------------------------------------------------------------ loaders (dev filter first)
def load_nba():
    cd = pd.read_parquet(DATA / "nba_ingame.parquet")
    cd = cd[(cd.date >= NBA_LO) & (cd.date <= NBA_HI)].copy()          # sealed rows dropped before anything else
    plays = pd.read_parquet(DATA / "nba_ingame_plays.parquet", columns=["eid", "t"])
    plays = plays[plays.eid.isin(cd.eid.unique())]
    span = plays.groupby("eid").t.agg(["min", "max"])
    # tickers (for tapes): same matching as nba_ingame.build
    mk = settled("KXNBAGAME")
    mk["code"] = mk.ticker.str.split("-").str[-1]
    mk["kdate"] = mk.event_ticker.str.split("-").str[1].str[:7]
    gm = cd.drop_duplicates("eid")[["eid", "date"]]
    home_away = cd.drop_duplicates(["eid", "side"])
    tick = {}
    for eid in gm.eid:
        if eid not in span.index:
            continue
        kd = span.loc[eid, "min"].tz_convert("America/New_York").strftime("%y%b%d").upper()
        summ = fetch(f"{NBA_ESPN}/summary", {"event": eid}, cache=True)
        comp = summ["header"]["competitions"][0]["competitors"]
        tm = {x["homeAway"]: NBA_ALIAS.get(x["team"]["abbreviation"], x["team"]["abbreviation"]) for x in comp}
        m = mk[(mk.kdate == kd) & mk.code.isin([tm["home"], tm["away"]])]
        if m.event_ticker.nunique() == 1 and len(m) == 2:
            for _, k in m.iterrows():
                tick[(eid, "home" if k.code == tm["home"] else "away")] = k.ticker
    cd["ticker"] = [tick.get((e, s)) for e, s in zip(cd.eid, cd.side)]
    games = pd.DataFrame({"gid": span.index, "start": esec(span["min"]), "stop": esec(span["max"])})
    games = games.merge(gm.rename(columns={"eid": "gid"}), on="gid")
    return cd.rename(columns={"eid": "gid"}), games


def load_nfl():
    gg = pd.read_parquet(DATA / "nfl_ingame_games.parquet")
    gg = gg[(gg.season == 2025) & (gg.stype == "REG") & (gg.week <= 12)]   # sealed weeks dropped first
    cd = pd.read_parquet(DATA / "nfl_ingame.parquet")
    cd = cd[cd.game_id.isin(gg.index)].rename(columns={"game_id": "gid"})
    games = pd.DataFrame({"gid": gg.index, "start": esec(gg.start), "stop": esec(gg.end),
                          "date": gg.start.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()})
    return cd, games


def load_nhl():
    g = pd.read_parquet(NHL_G)
    cd = pd.read_parquet(NHL_C)
    games = pd.DataFrame({"gid": g.eid, "start": esec(g.start), "stop": esec(g.end),
                          "date": g.start.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()})
    return cd, games


# ------------------------------------------------------------------------------------------------ snapshots
def snapshots(cd: pd.DataFrame, games: pd.DataFrame, sport: str) -> pd.DataFrame:
    """No outcomes used: the result column is carried along but never read here."""
    out = []
    gi = games.set_index("gid")
    for gid, c in cd.groupby("gid"):
        if gid not in gi.index or c.side.nunique() != 2:
            continue
        st, sp = gi.loc[gid, "start"], gi.loc[gid, "stop"]
        h = c[c.side == "home"].set_index("end")
        a = c[c.side == "away"].set_index("end")
        j = h.join(a, lsuffix="_h", rsuffix="_a", how="inner").sort_index()
        j = j[(j.index > st) & (j.index <= sp)]
        if j.empty:
            continue
        ends = j.index.to_numpy()
        for tr, ld in (("h", "a"), ("a", "h")):
            tb, ta = j[f"bid_{tr}"].to_numpy(), j[f"ask_{tr}"].to_numpy()
            la = j[f"ask_{ld}"].to_numpy()
            mid = (tb + ta) / 2
            ok = (tb > 0) & (ta < 1) & (tb < ta) & (ta - tb <= 0.03 + 1e-9) & (la > 0) & (la < 1) & (mid < 0.5)
            for b, lo, hi in BINS:
                idx = np.flatnonzero(ok & (mid >= lo - 1e-9) & (mid < hi - 1e-9))
                if not len(idx):
                    continue
                i = idx[0]
                nxt = la[i + 1] if i + 1 < len(la) and 0 < la[i + 1] < 1 else np.nan
                out.append({"sport": sport, "gid": gid, "bin": b, "T": int(ends[i]), "trailer": "home" if tr == "h" else "away",
                            "t_bid": tb[i], "t_ask": ta[i], "mid": mid[i], "l_ask": la[i], "l_ask_next": nxt,
                            "l_bid": j[f"bid_{ld}"].to_numpy()[i],
                            "t_ticker": j[f"ticker_{tr}"].iloc[i] if f"ticker_{tr}" in j else None,
                            "date": gi.loc[gid, "date"]})
    # one per game per bin: the earliest candle across both possible trailers
    s = pd.DataFrame(out).sort_values(["sport", "gid", "bin", "T"]).drop_duplicates(["sport", "gid", "bin"])
    return s


def tape(tk: str, t0: int, t1: int) -> list[dict] | None:
    out = []
    try:
        for path in ("historical/trades", "markets/trades"):
            cursor = None
            while True:
                p = {"ticker": tk, "min_ts": t0, "max_ts": t1, "limit": 1000}
                if cursor:
                    p["cursor"] = cursor
                d = fetch(f"{API}/{path}", p, cache=True)
                out += d.get("trades", [])
                cursor = d.get("cursor")
                if not cursor or not d.get("trades"):
                    break
            if out:
                break
    except Exception:
        return None
    rows = []
    for r in out:
        y = r.get("yes_price_dollars")
        y = float(y) if y is not None else float(r.get("yes_price")) / 100
        ts = pd.Timestamp(r["created_time"]).timestamp()
        if t0 < ts <= t1:
            rows.append({"ticker": tk, "ts": int(ts), "yes_price": y})
    return rows


def cmd_snap() -> int:
    parts = []
    for sport, fn in (("NBA", load_nba), ("NFL", load_nfl), ("NHL", load_nhl)):
        cd, games = fn()
        s = snapshots(cd, games, sport)
        parts.append(s)
        print(f"{sport}: {len(games)} dev games, {s.gid.nunique()} with a snapshot; per bin {s.bin.value_counts().sort_index().to_dict()}; "
              f"tickers {s.t_ticker.notna().mean():.1%}", flush=True)
    s = pd.concat(parts, ignore_index=True)
    s["gid"] = s.gid.astype(str)
    s.to_parquet(SNAP, index=False)
    rows, ok = [], []
    for _, r in s.iterrows():
        if not r.t_ticker:
            ok.append(False)
            continue
        t = tape(r.t_ticker, int(r["T"]), int(r["T"]) + 600)
        ok.append(t is not None)
        rows += [dict(x, T=int(r["T"])) for x in (t or [])]
    s["tape_ok"] = ok
    s.to_parquet(SNAP, index=False)
    pd.DataFrame(rows, columns=["ticker", "ts", "yes_price", "T"]).to_parquet(TAPE, index=False)
    print(f"tapes: fetched for {np.mean(ok):.1%} of {len(s)} snapshots, {len(rows)} trades")
    return 0


# ------------------------------------------------------------------------------------------------ scoring
def results_map() -> dict:
    """(sport, gid, side) -> trailer/leader result, from the same dev-filtered loaders."""
    m = {}
    for sport, fn in (("NBA", load_nba), ("NFL", load_nfl), ("NHL", load_nhl)):
        cd, _ = fn()
        for (g, side), r in cd.groupby(["gid", "side"]).result.first().items():
            m[(sport, str(g), side)] = r
    return m


def boot(x: np.ndarray, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(b, .025)), float(np.quantile(b, .975)), float(np.mean(b <= 0))


def cmd_dev() -> int:
    s = pd.read_parquet(SNAP)
    tp = pd.read_parquet(TAPE)
    res = results_map()
    other = {"home": "away", "away": "home"}
    s["t_res"] = [res.get((sp, g, t)) for sp, g, t in zip(s.sport, s.gid, s.trailer)]
    s["l_res"] = [res.get((sp, g, other[t])) for sp, g, t in zip(s.sport, s.gid, s.trailer)]
    bad = s[~s.t_res.isin(["yes", "no"]) | ~s.l_res.isin(["yes", "no"])]
    print(f"snapshots {len(s)}, missing/odd results {len(bad)}; both-yes {((s.t_res=='yes')&(s.l_res=='yes')).sum()}, "
          f"both-no (ties?) {((s.t_res=='no')&(s.l_res=='no')).sum()}")
    s = s.drop(bad.index)
    s["tw"] = (s.t_res == "yes").astype(float)
    s["lw"] = (s.l_res == "yes").astype(float)
    fee = lambda p: TAKER * p * (1 - p)
    s["cost"] = s.l_ask + fee(s.l_ask)
    s["pnl"] = s.lw - s.cost
    s["pnl_no"] = (1 - s.tw) - ((1 - s.t_bid) + fee(1 - s.t_bid))
    s["pnl_lat"] = s.lw - (s.l_ask_next + fee(s.l_ask_next))
    s["pnl_round"] = s.lw - (s.l_ask + np.ceil(np.round(fee(s.l_ask) * 100, 6)) / 100)
    # maker
    s["p_sell"] = np.ceil(np.round(s.mid * 100, 6)) / 100
    tp = tp.merge(s[["t_ticker", "T", "p_sell"]].rename(columns={"t_ticker": "ticker"}), on=["ticker", "T"])
    thr = tp[tp.yes_price > tp.p_sell + 1e-9].groupby(["ticker", "T"]).size()
    at = tp[tp.yes_price >= tp.p_sell - 1e-9].groupby(["ticker", "T"]).size()
    key = list(zip(s.t_ticker, s["T"]))
    s["fill"] = [k in thr.index for k in key]
    s["fill_at"] = [k in at.index for k in key]
    s["mk_pnl"] = s.p_sell - s.tw - MAKER * s.p_sell * (1 - s.p_sell)
    # halves within sport
    med = s.drop_duplicates(["sport", "gid"]).groupby("sport").date.median()
    s["half"] = np.where(s.date <= s.sport.map(med), "H1", "H2")

    def row(x: pd.DataFrame) -> dict:
        gap = (x.tw - x.mid).to_numpy()
        lo, hi, _ = boot(gap, 4000)
        plo, phi, p = boot(x.pnl.to_numpy())
        return {"n": int(len(x)), "mid": float(x.mid.mean()), "spread": float((x.t_ask - x.t_bid).mean()),
                "realised": float(x.tw.mean()), "real_minus_mid": float(gap.mean()), "rmm_ci": [lo, hi],
                "leader_ask": float(x.l_ask.mean()), "breakeven_leader_win": float(x.cost.mean()),
                "leader_win": float(x.lw.mean()), "pnl_c": float(100 * x.pnl.mean()), "pnl_ci_c": [100 * plo, 100 * phi],
                "roi": float(x.pnl.sum() / x.cost.sum()), "p": p,
                "pnl_c_H1": float(100 * x[x.half == "H1"].pnl.mean()), "pnl_c_H2": float(100 * x[x.half == "H2"].pnl.mean()),
                "pnl_c_trailerNO": float(100 * x.pnl_no.mean()), "pnl_c_latency": float(100 * x.pnl_lat.dropna().mean()),
                "pnl_c_roundfee": float(100 * x.pnl_round.mean())}

    out = {"by_sport_bin": {}, "pooled": {}, "maker": {}}
    print("\nCALIBRATION + TAKER (buy leader at ask), cents per contract")
    hdr = f"{'cell':12s} {'n':>5s} {'mid':>6s} {'real':>6s} {'r-mid':>6s} {'95%CI':>15s} {'BE':>6s} {'Lwin':>6s} {'pnl':>6s} {'95%CI':>15s} {'p':>6s} {'H1':>6s} {'H2':>6s}"
    print(hdr)
    for sp in ("NBA", "NFL", "NHL", "ALL"):
        for b, _, _ in BINS + [("ALL", 0, 0)]:
            x = s if sp == "ALL" else s[s.sport == sp]
            x = x if b == "ALL" else x[x.bin == b]
            if not len(x):
                continue
            r = row(x)
            (out["pooled"] if sp == "ALL" else out["by_sport_bin"])[f"{sp}_{b}"] = r
            print(f"{sp+'_'+b:12s} {r['n']:5d} {100*r['mid']:6.2f} {100*r['realised']:6.2f} {100*r['real_minus_mid']:+6.2f} "
                  f"[{100*r['rmm_ci'][0]:+5.1f},{100*r['rmm_ci'][1]:+5.1f}] {100*r['breakeven_leader_win']:6.2f} {100*r['leader_win']:6.2f} "
                  f"{r['pnl_c']:+6.2f} [{r['pnl_ci_c'][0]:+5.2f},{r['pnl_ci_c'][1]:+5.2f}] {r['p']:6.3f} {r['pnl_c_H1']:+6.2f} {r['pnl_c_H2']:+6.2f}")
    print("\nTAKER sensitivities (pooled, c/contract): trailer-NO / +1 min latency / rounded 1-lot fee")
    for b, _, _ in BINS:
        r = out["pooled"][f"ALL_{b}"]
        print(f"  {b}: {r['pnl_c_trailerNO']:+.2f} / {r['pnl_c_latency']:+.2f} / {r['pnl_c_roundfee']:+.2f}")
    print("\nMAKER (sell trailer YES at ceil(mid), 10 min, strict print-through)")
    for b, _, _ in BINS:
        x = s[s.bin == b]
        cov = float(x.tape_ok.mean())
        f = x[x.fill & x.tape_ok]
        lo, hi, p = boot(f.mk_pnl.to_numpy())
        fa = x[x.fill_at & x.tape_ok]
        r = {"snapshots": int(len(x)), "tape_coverage": cov, "fills": int(len(f)), "fill_rate": float(len(f) / max(1, x.tape_ok.sum())),
             "trailer_win_all": float(x.tw.mean()), "trailer_win_if_filled": float(f.tw.mean()) if len(f) else None,
             "mid_if_filled": float(f.mid.mean()) if len(f) else None, "p_sell_if_filled": float(f.p_sell.mean()) if len(f) else None,
             "pnl_c": float(100 * f.mk_pnl.mean()) if len(f) else None, "pnl_ci_c": [100 * lo, 100 * hi], "p": p,
             "pnl_c_H1": float(100 * f[f.half == "H1"].mk_pnl.mean()) if len(f) else None,
             "pnl_c_H2": float(100 * f[f.half == "H2"].mk_pnl.mean()) if len(f) else None,
             "at_fills": int(len(fa)), "at_pnl_c": float(100 * fa.mk_pnl.mean()) if len(fa) else None,
             "scored": cov >= 0.8}
        out["maker"][b] = r
        print(f"  {b}: snaps {r['snapshots']} tape {cov:.0%} fills {r['fills']} ({r['fill_rate']:.0%}); trailer win all {r['trailer_win_all']:.3f} "
              f"| if filled {r['trailer_win_if_filled']} at p_sell {r['p_sell_if_filled']}; pnl {r['pnl_c']} c [{100*lo:+.2f},{100*hi:+.2f}] p {p:.3f} "
              f"H1 {r['pnl_c_H1']} H2 {r['pnl_c_H2']}; '>=' fills {r['at_fills']} pnl {r['at_pnl_c']}")
    bar = 0.05 / len(TRIALS)
    recs = []
    for t in TRIALS:
        kind, b = t.split("_")
        if kind == "taker":
            r = out["pooled"][f"ALL_{b}"]
            m, p, h1, h2 = r["pnl_c"], r["p"], r["pnl_c_H1"], r["pnl_c_H2"]
            status = "ok"
            n = r["n"]
        else:
            r = out["maker"][b]
            m, p, h1, h2 = r["pnl_c"], r["p"], r["pnl_c_H1"], r["pnl_c_H2"]
            status = "ok" if r["scored"] and r["fills"] else "not_scored_no_tape"
            n = r["fills"]
        verdict = "pass" if status == "ok" and m is not None and m > 0 and p < bar and (h1 or 0) > 0 and (h2 or 0) > 0 else "fail"
        if status != "ok":
            verdict = "not_scored"
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"ingame_longshot_{t}",
                     "sha": hashlib.sha256(("ingame_longshot:" + t).encode()).hexdigest()[:12], "dataset": DATASET, "trial": t,
                     "hypothesis": f"pre-registered in scripts/ingame_longshot.py ({t})",
                     "source": "L37 sealed descriptive: Kalshi trailer mids ~1.5c above realised (forward idea L38)",
                     "status": status, "study_trials": len(TRIALS), "bonferroni_bar": bar, "bets": n,
                     "pnl_c_per_contract": m, "p": p, "pnl_c_H1": h1, "pnl_c_H2": h2, "verdict": verdict})
        print(f"TRIAL {t:9s} n {n:4d} pnl {m if m is None else round(m, 3)} c p {p if p is None else round(p, 4)} -> {verdict}")
    out["trials"] = recs
    out["bar"] = bar
    OUTJ.write_text(json.dumps(out, indent=1, default=str))
    print(f"-> {OUTJ}; bar p < {bar:.5f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        print("appended to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"build-nhl": cmd_build_nhl, "snap": cmd_snap, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
