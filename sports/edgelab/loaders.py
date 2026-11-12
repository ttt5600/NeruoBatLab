"""Dataset loaders. They WRAP the existing study code / cached files (no re-implementation of any study's pipeline);
each returns a frame with a `date` column (the vault's split key, tz-naive calendar date) plus raw columns for rules.
Only edgelab.vault calls these; rules never load data themselves.

Every loader returns (frame, sources) where sources = list of file paths the rows came from (hashed into evidence).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .paths import DATA, QS_DATA, SPORTS

SCRIPTS = SPORTS / "scripts"


def _imp_scripts():
    for p in (str(SPORTS), str(SCRIPTS)):
        if p not in sys.path:
            sys.path.insert(0, p)


# ------------------------------------------------------------------ Kalshi daily-high temperature (E1 / E2 / E4)
def _wx_picks(cohort: str):
    """Frozen-rule picks (kalshi_reject_safe_check for E1, wx_expand.picks for X1 = E2), built by
    scripts/kalshi_temp_maker.py build / build-sealed. Dev and sealed files are concatenated; the vault splits by date."""
    files = [DATA / "kalshi_temp_maker_picks.parquet", DATA / "kalshi_temp_maker_picks_sealed.parquet"]
    tapes = [DATA / "kalshi_temp_maker_trades.parquet", DATA / "kalshi_temp_maker_trades_sealed.parquet"]
    p = pd.concat([pd.read_parquet(f) for f in files if f.exists()], ignore_index=True)
    p = p[p.cohort == cohort].copy()
    p["date"] = pd.to_datetime(p["date"])
    tr = pd.concat([pd.read_parquet(f) for f in tapes if f.exists()], ignore_index=True)
    tr = tr[tr.ticker.isin(p.ticker)]
    p.attrs["tape"] = tr
    return p, [str(f) for f in files + tapes if f.exists()]


def kalshi_temp_e1():
    return _wx_picks("E1")


def kalshi_temp_x1():
    return _wx_picks("X1")


def kalshi_temp_all():
    """lab/datasets.py kalshi_temp table (every tradeable bracket, both sides, 22:00 local quote) -- placebo base."""
    f = DATA / "lab" / "kalshi_temp.parquet"
    d = pd.read_parquet(f)
    d["date"] = pd.to_datetime(d["date"])
    return d, [str(f)]


# ------------------------------------------------------------------ KXTSAW weekly TSA (E1X-T)
def kalshi_tsaw():
    """scripts/e1x_tsa.py rungs(): every consensus-rejected rung with its T (decision) quote inputs; plus the hourly
    candles and the public trade tape for maker fills / closes."""
    _imp_scripts()
    import e1x_tsa as T
    x = T.rungs(None)
    x = x[x.result.isin(["yes", "no"])].copy()
    x["date"] = pd.to_datetime(x["wend"])
    D = DATA / "e1x"
    x.attrs["candles"] = json.loads((D / "tsa_candles.json").read_text())
    x.attrs["trades"] = json.loads((D / "tsa_trades.json").read_text())
    return x, [str(D / "tsa_build.parquet"), str(D / "KXTSAW_markets.parquet"), str(D / "tsa_candles.json"),
               str(D / "tsa_trades.json")]


# ------------------------------------------------------------------ NBA totals vs opener (L32b)
def nba_totals_open():
    """L32b dev bet list (results/l32b_debiased_totals/dev_bets.csv, written by scripts/l32b_debiased_totals.py) joined
    to nba_simple_model.markets() for the QUOTED open odds and start times. Dev seasons only exist in this file."""
    _imp_scripts()
    import nba_simple_model as M
    f = SPORTS / "results" / "l32b_debiased_totals" / "dev_bets.csv"
    b = pd.read_csv(f)
    wf = pd.read_pickle(M.OUTD / "wf.pkl")
    m = M.markets(wf)
    keep = [c for c in ["event_id", "start_utc", "over_odds_open", "under_odds_open", "pf_over_close", "tot_close_same",
                        "total_close", "oo_c", "uo_c"] if c in m.columns]
    m = m[keep].drop_duplicates("event_id")
    m["event_id"] = m["event_id"].astype("int64")
    b = b.merge(m.drop(columns=[c for c in ("tot_close_same",) if c in m.columns]), on="event_id", how="left")
    b["date"] = pd.to_datetime(b["date"])
    return b, [str(f), str(M.OUTD / "wf.pkl")]


def nba_totals_open_forward():
    """L32b forward paper log (scripts/l32b_logger.py). The launchd copy under ~/.claude/quant-sports is preferred
    (read-only); falls back to the repo copy."""
    f = QS_DATA / "l32b_log.jsonl"
    if not f.exists():
        f = DATA / "l32b_log.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
    d = pd.DataFrame(rows)
    if d.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")}), [str(f)]
    d["date"] = pd.to_datetime(d["key"].str[:10], errors="coerce")  # ET game date (decision and settle rows alike)
    return d, [str(f)]


# ------------------------------------------------------------------ NBA sides, team rating (L2)
def nba_sides_l2():
    """results/l2_team_rating_sides/bets_*.csv (scripts/l2_team_rating_sides.py score; 2025-26 held out, ESPN close)."""
    D = SPORTS / "results" / "l2_team_rating_sides"
    fs = sorted(D.glob("bets_*.csv"))
    b = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    b["date"] = pd.to_datetime(b["date"])
    return b, [str(f) for f in fs]


# ------------------------------------------------------------------ NFL totals (placebo base, lab/datasets.py)
def nfl_total_lab():
    f = DATA / "lab" / "nfl_total.parquet"
    d = pd.read_parquet(f)
    d["date"] = pd.to_datetime(d["date"])
    return d, [str(f)]


# ------------------------------------------------------------------ forward-only logs (no settled-bet adapter yet)
def _jsonl(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    return rows


def nba_early_form():
    f = QS_DATA / "nba_early_form_log.jsonl"
    if not f.exists():
        f = DATA / "nba_early_form_log.jsonl"
    d = pd.DataFrame(_jsonl(f))
    d["date"] = pd.to_datetime(d["date"]) if "date" in d else pd.Series(dtype="datetime64[ns]")
    return d, [str(f)]


def kalshi_prop_ladders():
    f = DATA / "kalshi_prop_grades.jsonl"
    d = pd.DataFrame(_jsonl(f))
    d["date"] = pd.to_datetime(d["date"]) if "date" in d else pd.Series(dtype="datetime64[ns]")
    return d, [str(f)]


def q017_ercot():
    f = QS_DATA / "q017" / "fwd" / "picks.jsonl"
    d = pd.DataFrame(_jsonl(f))
    d["date"] = pd.to_datetime(d["date"]) if "date" in d else pd.Series(dtype="datetime64[ns]")
    return d, [str(f)]


def kalshi_tsaw_forward():
    f = QS_DATA / "e1x" / "tsa_paper.jsonl"
    if not f.exists():
        f = DATA / "e1x" / "tsa_paper.jsonl"
    d = pd.DataFrame(_jsonl(f))
    if d.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")}), [str(f)]
    d["date"] = pd.to_datetime(d["event"].str.split("-").str[1], format="%y%b%d", errors="coerce")
    return d, [str(f)]


# ------------------------------------------------------------------ CBB in-game vs Kalshi (scripts/cbb_ingame.py)
def cbb_ingame():
    """data/cbb_ingame/snapshots.parquet (one row per game x 60-s decision time; built by scripts/cbb_ingame.py build)
    + the 1-min candles per game in attrs (maker fills from trade low/high, CLV mark)."""
    D = DATA / "cbb_ingame"
    f, fc = D / "snapshots.parquet", D / "candles.parquet"
    x = pd.read_parquet(f)
    x["date"] = pd.to_datetime(x["date"])
    cd = pd.read_parquet(fc)
    cd = cd[cd["eid"].isin(set(x["eid"]))]
    x.attrs["candles"] = {e: c.sort_values("end").reset_index(drop=True) for e, c in cd.groupby("eid")}
    return x, [str(f), str(fc)]


LOADERS = {
    "kalshi_temp_e1": kalshi_temp_e1, "kalshi_temp_x1": kalshi_temp_x1, "kalshi_temp_all": kalshi_temp_all,
    "kalshi_tsaw": kalshi_tsaw, "nba_totals_open": nba_totals_open, "nba_totals_open_forward": nba_totals_open_forward,
    "nba_sides_l2": nba_sides_l2, "nfl_total_lab": nfl_total_lab, "nba_early_form": nba_early_form,
    "kalshi_prop_ladders": kalshi_prop_ladders, "q017_ercot": q017_ercot, "kalshi_tsaw_forward": kalshi_tsaw_forward,
    "cbb_ingame": cbb_ingame,
}


# ------------------------------------------------------------------ soccer 1X2 in-game vs Kalshi (scripts/kalshi_ingame_lowscore.py)
def kalshi_soccer_ingame():
    """data/ingame_lowscore_soccer_frame.parquet: one row per game x market x 1-min candle in play (model p + quotes)."""
    f = DATA / "ingame_lowscore_soccer_frame.parquet"
    x = pd.read_parquet(f)
    x["date"] = pd.to_datetime(x["date"])
    return x, [str(f)]


LOADERS["kalshi_soccer_ingame"] = kalshi_soccer_ingame


# ------------------------------------------------------------------ Polymarket maker NO (scripts/pm_maker_no.py)
def pm_maker_no():
    """data/pm_maker_no/frame.parquet (one row per mention/culture binary market: anchor S, t = S - 24 h, display
    YES p_t, closed_time, p_close, won = NO resolved) + attrs['trades'] = {ticker: prints after t} (trades.parquet)."""
    D = DATA / "pm_maker_no"
    f, ft = D / "frame.parquet", D / "trades.parquet"
    x = pd.read_parquet(f)
    x["date"] = pd.to_datetime(x["date"])
    tr = pd.read_parquet(ft)
    tr = tr[tr["ticker"].isin(set(x["ticker"]))]
    x.attrs["trades"] = {k: g[["timestamp", "price", "yes"]].reset_index(drop=True) for k, g in tr.groupby("ticker")}
    return x, [str(f), str(ft)]


LOADERS["pm_maker_no"] = pm_maker_no


# ------------------------------------------------------------------ PortWatch weekly chokepoints far-bracket (scripts/dl_portwatch.py)
def kalshi_portwatch():
    """scripts/dl_portwatch.py rungs(): every far-rejected rung (K outside last published week x residual range) with
    T; plus hourly candles and the public trade tape (maker fills strictly through q). date = week end (Sunday)."""
    _imp_scripts()
    import dl_portwatch as P
    x = P.rungs()
    x = x[x.result.isin(["yes", "no"])].copy()
    x["date"] = pd.to_datetime(x["wend"])
    D = DATA / "dl"
    x.attrs["candles"] = json.loads((D / "pw_candles.json").read_text())
    x.attrs["trades"] = json.loads((D / "pw_trades.json").read_text())
    return x, [str(D / "pw_build.parquet"), str(D / "portwatch_daily.parquet"), str(D / "pw_candles.json"),
               str(D / "pw_trades.json")]


LOADERS["kalshi_portwatch"] = kalshi_portwatch


# ------------------------------------------------------------------ KXYTVIEWSW weekly YouTube max daily views (E1X-M)
def kalshi_ytviews():
    """scripts/e1x_media.py yt_rungs(): consensus-rejected rungs at Sunday 16:00Z + hourly candles and trade tape."""
    _imp_scripts()
    import e1x_media as Y
    x = Y.yt_rungs()
    x = x[x.result.isin(["yes", "no"])].copy()
    x["date"] = pd.to_datetime(x["wend"])
    x = x.drop(columns=["open", "close"])
    D = DATA / "e1x_media"
    x.attrs["candles"] = json.loads((D / "ytv_candles.json").read_text())
    x.attrs["trades"] = json.loads((D / "ytv_trades.json").read_text())
    return x, [str(D / "ytv_build.parquet"), str(D / "KXYTVIEWSW_markets.parquet"), str(D / "ytv_candles.json"),
               str(D / "ytv_trades.json")]


LOADERS["kalshi_ytviews"] = kalshi_ytviews


# ------------------------------------------------------------------ Kalshi HOURLY temperature (scripts/kalshi_hourly_temp.py)
def kalshi_temp_hourly():
    """data/kalshi_hourly_temp/frame.parquet: one row per settled hourly-temperature market (inputs at T = open + 15 min,
    1-min quotes for input-selected markets) + attrs['tape'] = public trades (T, H] on the rule-selected tickers."""
    D = DATA / "kalshi_hourly_temp"
    f, ft = D / "frame.parquet", D / "tape.parquet"
    x = pd.read_parquet(f)
    x["date"] = pd.to_datetime(x["date"])
    x.attrs["tape"] = pd.read_parquet(ft)
    return x, [str(f), str(ft)]


LOADERS["kalshi_temp_hourly"] = kalshi_temp_hourly


# ------------------------------------------------------------------ Polymarket time-to-resolution favourites (scripts/pm_timing.py)
def pm_timing():
    """data/pm_timing/frame.parquet: one row per (binary market, horizon 7d|30d) with p0 at t = S - h, closed_time,
    p_close, result (hidden: winning token) + attrs['trades'] = {ticker: prints after t, tok0 flag} (trades.parquet)."""
    D = DATA / "pm_timing"
    f, ft = D / "frame.parquet", D / "trades.parquet"
    x = pd.read_parquet(f)
    x["date"] = pd.to_datetime(x["date"])
    tr = pd.read_parquet(ft)
    tr = tr[tr["ticker"].isin(set(x["ticker"]))]
    x.attrs["trades"] = {k: g[["timestamp", "price", "tok0"]].reset_index(drop=True) for k, g in tr.groupby("ticker")}
    return x, [str(f), str(ft)]


LOADERS["pm_timing"] = pm_timing


# ------------------------------------------------------------------ Polymarket jump / overreaction fade (scripts/pm_jumps.py)
def pm_jumps():
    """data/pm_jumps/frame_{dev,sealed}.parquet: one row per detected 5c/15-min jump (pre-decision columns only,
    decided_at = jump print + 60 s, result hidden) + attrs['trades'] = {ticker: DataFrame[ts, p]} = YES-equivalent
    public taker tape after the first decision (trades_{split}.parquet). Sealed frame exists only after its pull."""
    D = DATA / "pm_jumps"
    xs, trs, srcs = [], [], []
    for s in ("dev", "sealed"):
        f, ft = D / f"frame_{s}.parquet", D / f"trades_{s}.parquet"
        if f.exists():
            xs.append(pd.read_parquet(f)); trs.append(pd.read_parquet(ft)); srcs += [str(f), str(ft)]
    x = pd.concat(xs, ignore_index=True)
    x["date"] = pd.to_datetime(x["date"])
    tr = pd.concat(trs, ignore_index=True)
    x.attrs["trades"] = {k: g[["ts", "p"]].reset_index(drop=True) for k, g in tr.groupby("ticker")}
    return x, srcs


LOADERS["pm_jumps"] = pm_jumps


# ------------------------------------------------------------------ Kalshi -> Polymarket US in-game lag (scripts/pmus_lag.py)
def pmus_ingame_lag():
    """data/pmus_lag/frame_{dev,sealed}.parquet: one row per Kalshi in-game jump (pre-decision columns, result hidden;
    `ticker` = PM US slug so result_equals_side keys on it, Kalshi ticker in `kticker`) + attrs['pm'] = {slug:
    DataFrame[ts, long, short]} PM US minute observations (pm_{split}.parquet). Sealed exists only after its pull."""
    D = DATA / "pmus_lag"
    xs, hs, srcs = [], [], []
    for s in ("dev", "sealed"):
        f, fh = D / f"frame_{s}.parquet", D / f"pm_{s}.parquet"
        if f.exists():
            xs.append(pd.read_parquet(f)); hs.append(pd.read_parquet(fh)); srcs += [str(f), str(fh)]
    x = pd.concat(xs, ignore_index=True).rename(columns={"ticker": "kticker"})
    x["ticker"] = x["slug"]
    x["date"] = pd.to_datetime(x["date"])
    x["k_close_ts"] = (pd.to_datetime(x["k_close"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    h = pd.concat(hs, ignore_index=True)
    x.attrs["pm"] = {k: g[["ts", "long", "short"]].sort_values("ts").reset_index(drop=True) for k, g in h.groupby("slug")}
    return x, srcs


LOADERS["pmus_ingame_lag"] = pmus_ingame_lag


# ------------------------------------------------------------------ NHL in-game forward log (scripts/nhl_ingame_logger.py)
NHL_FRAME_COLS = ["date", "gid", "ticker", "team", "t", "inputs_as_of", "price_time", "event_start", "period",
                  "period_type", "intermission", "minutes_left", "margin", "p_team", "bid", "ask", "recent_goal",
                  "jump", "mark", "mark_src", "result"]


def _nhl_epoch(s) -> np.ndarray:
    return ((pd.to_datetime(s, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy()


def nhl_frame_from_log(rows: list[dict], results: list[dict]) -> pd.DataFrame:
    """Logger rows -> one row per tick x team market. Only games whose BOTH tickers have a Kalshi result are kept
    (completeness filter on settlement, not on the outcome's value). Decision columns use ticks with decided_at <= t
    only; `mark` (CLV) looks forward and is never a decision input."""
    ticks = [r for r in rows if r.get("kind") == "tick"]
    empty = pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "date" else object) for c in NHL_FRAME_COLS})
    if not ticks:
        return empty
    d = pd.DataFrame(ticks)
    d["t"] = _nhl_epoch(d["decided_at"])
    d = d.sort_values(["gid", "t"]).drop_duplicates(["gid", "t"]).reset_index(drop=True)
    asof = pd.json_normalize(d["as_of"].tolist())
    d["inputs_as_of"] = pd.concat([pd.to_datetime(asof[c], utc=True) for c in ("nhl", "kalshi", "anchor")],
                                  axis=1).max(axis=1)
    d["price_time"] = pd.to_datetime(asof["kalshi"], utc=True)
    # 3-min goal cooldown: a score change first observed at tick t blocks [t, t+180)
    tot = d["home_score"].fillna(0).astype(int) + d["away_score"].fillna(0).astype(int)
    prev = tot.groupby(d["gid"]).shift(1)
    chg = prev.notna() & tot.ne(prev)
    last = d["t"].where(chg).groupby(d["gid"]).ffill()
    d["recent_goal"] = (d["t"] - last) < 180
    # price jump: home mid vs each of the previous 3 ticks (within 300 s)
    d["hmid"] = (d["home_bid"] + d["home_ask"]) / 2
    jump = pd.Series(False, index=d.index)
    for k in (1, 2, 3):
        pm_, pt_ = d.groupby("gid")["hmid"].shift(k), d.groupby("gid")["t"].shift(k)
        jump |= ((d["hmid"] - pm_).abs() >= 0.08) & ((d["t"] - pt_) <= 300)
    d["jump"] = jump
    fin = {r["gid"]: r["decided_at"] for r in rows if r.get("kind") == "final"}
    res = {r["ticker"]: r for r in results}
    out = []
    for team in ("home", "away"):
        p = pd.to_numeric(d["p_model"], errors="coerce")
        m = d.assign(team=team, ticker=d[f"{team}_ticker"], bid=d[f"{team}_bid"], ask=d[f"{team}_ask"],
                     p_team=p if team == "home" else 1 - p)
        tt_all, mid_all = m["t"].to_numpy(), ((m["bid"] + m["ask"]) / 2).to_numpy()
        ok_all = ((m["bid"] > 0) & (m["ask"] < 1)).to_numpy()
        mark = np.full(len(m), np.nan)
        src = np.array([None] * len(m), dtype=object)
        for idx in m.groupby("gid").indices.values():
            tt, mm, ok = tt_all[idx], mid_all[idx], ok_all[idx]
            for j, i in enumerate(idx):
                for lo, hi, lab in ((300, 420, "+300s"), (60, 180, "+60s")):
                    w = np.where(ok & (tt >= tt[j] + lo) & (tt <= tt[j] + hi))[0]
                    if len(w):
                        mark[i], src[i] = mm[w[0]], lab
                        break
        m["mark"], m["mark_src"] = mark, src
        out.append(m)
    x = pd.concat(out, ignore_index=True)
    last_t = x.groupby("gid")["t"].transform("max")
    x["event_start"] = pd.to_datetime([fin[g] if g in fin else pd.Timestamp(int(lt) + 60, unit="s", tz="UTC")
                                       for g, lt in zip(x["gid"], last_t)], utc=True)
    x["result"] = [res[tk]["result"] if tk in res else None for tk in x["ticker"]]
    settled = x["result"].notna().groupby(x["gid"]).transform("all")
    x = x[settled.astype(bool)].copy()
    if x.empty:
        return empty
    x["date"] = pd.to_datetime(x["game_date"])
    x["intermission"] = x["intermission"].fillna(False).astype(bool)
    return x[NHL_FRAME_COLS].reset_index(drop=True)


def kalshi_nhl_ingame():
    """Forward-only: the launchd logger's data/live/nhl_ingame_paper.jsonl + nhl_ingame_results.jsonl (the deployed
    copy ~/.claude/quant-sports is the writer; the worktree copy after `qs_sync.sh pull` is the fallback)."""
    base = QS_DATA / "live" if (QS_DATA / "live" / "nhl_ingame_paper.jsonl").exists() else DATA / "live"
    f, fr = base / "nhl_ingame_paper.jsonl", base / "nhl_ingame_results.jsonl"
    return nhl_frame_from_log(_jsonl(f), _jsonl(fr)), [str(f), str(fr)]


LOADERS["kalshi_nhl_ingame"] = kalshi_nhl_ingame


# ------------------------------------------------------------------ physical ladders (scripts/phys_ladders.py)
def _phys(name):
    D = DATA / "phys_ladders"
    f, ft = D / f"{name}_frame.parquet", D / f"{name}_tape.parquet"
    x = pd.read_parquet(f)
    x["date"] = pd.to_datetime(x["date"])
    x.attrs["tape"] = pd.read_parquet(ft)
    return x, [str(f), str(ft)]


def kalshi_spacex_count():
    """data/phys_ladders/spacex_frame.parquet: one row per (monthly KXSPACEXCOUNT ticker, daily 16:00Z decision)."""
    return _phys("spacex")


def kalshi_spr_level():
    """data/phys_ladders/spr_frame.parquet: one row per settled KXSPRLVL market (decision = open + 1 h)."""
    return _phys("spr")


LOADERS["kalshi_spacex_count"] = kalshi_spacex_count
LOADERS["kalshi_spr_level"] = kalshi_spr_level


# ------------------------------------------------------------------ ForecastEx monthly climate (scripts/fx_climate.py)
def fx_climate():
    """data/fx_climate/frame.parquet: one row per settled GTM / MACD contract (nowcast inputs + hidden result) +
    attrs['days'] = per-CT-trade-day high/end/qty of each instrument that traded (days.parquet)."""
    D = DATA / "fx_climate"
    f, fd = D / "frame.parquet", D / "days.parquet"
    x = pd.read_parquet(f)
    x["date"] = pd.to_datetime(x["date"])
    dd = pd.read_parquet(fd)
    x.attrs["days"] = dd[dd["ticker"].isin(set(x["ticker"]))].reset_index(drop=True)
    return x, [str(f), str(fd)]


LOADERS["fx_climate"] = fx_climate


# ------------------------------------------------------------------ KXTORNADO monthly count (scripts/tornado_ladder.py)
def kalshi_tornado_count():
    """data/phys_ladders/tornado_frame.parquet: one row per (monthly KXTORNADO ticker, daily 16:00Z decision)."""
    return _phys("tornado")


LOADERS["kalshi_tornado_count"] = kalshi_tornado_count


# ------------------------------------------------------------------ daily rain NYC/SEA (scripts/raind_daily_rain.py)
def kalshi_rain_daily_nyc():
    """data/raind/rain_frame.parquet: one row per settled daily-rain ticker (KXRAINNYC/KXRAINSEA T0, KXRAIN-*-NYC)."""
    D = DATA / "raind"
    f, ft = D / "rain_frame.parquet", D / "rain_tape.parquet"
    x = pd.read_parquet(f)
    x["date"] = pd.to_datetime(x["date"])
    x.attrs["tape"] = pd.read_parquet(ft)
    return x, [str(f), str(ft)]


LOADERS["kalshi_rain_daily_nyc"] = kalshi_rain_daily_nyc


# ------------------------------------------------------------------ weekly ISO power, forward paper (scripts/power_week.py)
def kalshi_power_week():
    """Forward-only: data/live/power_week_ticks.jsonl + _results.jsonl + _tape.jsonl (deployed ~/.claude/quant-sports
    copy is the writer; worktree copy after `qs_sync.sh pull` is the fallback). Rows without a settled result drop."""
    base = QS_DATA / "live" if (QS_DATA / "live" / "power_week_ticks.jsonl").exists() else DATA / "live"
    f, fr, ft = base / "power_week_ticks.jsonl", base / "power_week_results.jsonl", base / "power_week_tape.jsonl"
    ticks, res, tape = pd.DataFrame(_jsonl(f)), pd.DataFrame(_jsonl(fr)), pd.DataFrame(_jsonl(ft))
    if ticks.empty or res.empty:
        x = pd.DataFrame(columns=list(dict.fromkeys(list(ticks.columns) + ["date", "result", "expiration_value", "mark_yes_bid", "mark_yes_ask"])))
    else:
        x = ticks.merge(res, on="ticker", how="inner", validate="many_to_one")
    x["date"] = pd.to_datetime(x["date"])
    x.attrs["tape"] = tape
    return x, [str(f), str(fr), str(ft)]


LOADERS["kalshi_power_week"] = kalshi_power_week


# ------------------------------------------------------------------ lock-in (scripts/lock_in.py)
def lock_in_nba_total():
    """data/lock_in/frame_{dev,sealed}.parquet: one row per NBA ticker whose YES is fixed by the public score (rules/lock_in.py)."""
    D = DATA / "lock_in"
    fs = [p for p in (D / "frame_dev.parquet", D / "frame_sealed.parquet") if p.exists()]
    ts = [p for p in (D / "tape_dev.parquet", D / "tape_sealed.parquet") if p.exists()]
    x = pd.concat([pd.read_parquet(p) for p in fs], ignore_index=True)
    x["date"] = pd.to_datetime(x["date"])
    x.attrs["tape"] = pd.concat([pd.read_parquet(p) for p in ts], ignore_index=True)
    return x, [str(p) for p in fs + ts]


LOADERS["lock_in_nba_total"] = lock_in_nba_total


# ------------------------------------------------------------------ game-release slump (scripts/game_release_slump.py)
def game_release_totals():
    """results/game_release_slump/edgelab_frame.parquet: one row per in-season game (NFL/NBA/NHL/CFB, dev seasons) in
    days 0..13 after a mega video-game launch (notes/game_release_slump.md), UNDER side at the closing total."""
    f = SPORTS / "results" / "game_release_slump" / "edgelab_frame.parquet"
    x = pd.read_parquet(f)
    x["date"] = pd.to_datetime(x["date"])
    return x, [str(f)]


LOADERS["game_release_totals"] = game_release_totals


def game_release_hype():
    """results/game_release_dose/edgelab_frame.parquet (scripts/game_release_dose.py sealed): one row per in-season game
    on days 0..6 after a launch cluster with H >= $600M; dev rows <= 2022-12-31 (SBR/lab), sealed rows 2023-01-01..
    2026-10-31 (ESPN lines). notes/game_release_dose.md."""
    f = SPORTS / "results" / "game_release_dose" / "edgelab_frame.parquet"
    x = pd.read_parquet(f)
    x["date"] = pd.to_datetime(x["date"])
    return x, [str(f)]


LOADERS["game_release_hype"] = game_release_hype


def nba_paint_sides():
    """results/nba_l30_paint/edgelab_frame.parquet (scripts/nba_l30_paint.py export): one row per (trial, game) the
    P-PAINT fade selects, dev 2016-17..2024-25 (notes/nba_l30_paint.md)."""
    f = SPORTS / "results" / "nba_l30_paint" / "edgelab_frame.parquet"
    x = pd.read_parquet(f)
    x["date"] = pd.to_datetime(x["date"])
    return x, [str(f)]


LOADERS["nba_paint_sides"] = nba_paint_sides
