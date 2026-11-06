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
