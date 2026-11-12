"""L30 P-PAINT: reproduce + audit the 'opp paint share' ATS effect, then the pre-registered walk-forward bet trials.

notes/nba_l30_paint.md (pre-registration). Dev only: 2015-16..2024-25 regular season. 2025-26 is NOT read.
Usage (from sports/): ../.venv_quant/bin/python scripts/nba_l30_paint.py {repro|bets|trend}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))
import traps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "nba_l30_paint"
OUT.mkdir(parents=True, exist_ok=True)
ESPN2NBA = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


def team_games() -> pd.DataFrame:
    t = pd.read_parquet(ROOT / "data" / "nba_comebacks" / "teamlogs.parquet")
    t = t[t.stype == "Regular Season"].copy()
    assert t.season.max() <= 2024, "2025-26 must not be read"
    t["date"] = pd.to_datetime(t.GAME_DATE.str[:10])
    o = t[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", "PTS_PAINT"]].rename(
        columns={"TEAM_ID": "o_id", "TEAM_ABBREVIATION": "opp", "PTS": "o_pts", "PTS_PAINT": "o_paint"})
    d = t.merge(o, on="GAME_ID")
    d = d[d.TEAM_ID != d.o_id].copy()
    assert d.GAME_ID.value_counts().eq(2).all()
    # audit (a): OPP_PTS_PAINT must equal the opponent's own PTS_PAINT
    d["paint_mismatch"] = d.OPP_PTS_PAINT != d.o_paint
    d["team"] = d.TEAM_ABBREVIATION
    d["is_home"] = d.MATCHUP.str.contains("vs.")
    d["ops"] = d.OPP_PTS_PAINT / d.o_pts
    d = d.sort_values(["team", "date"]).reset_index(drop=True)
    # audit (b): one game per team per date
    assert not d.duplicated(["team", "date"]).any()
    g = d.groupby(["team", "season"], sort=False)
    d["n_prior"] = g.cumcount()
    # strictly earlier games: cumulative sum of previous rows / count (dates strictly increase within team-season)
    d["ops_pit"] = (g.ops.cumsum() - d.ops) / d.n_prior.replace(0, np.nan)
    d["ops_leak"] = g.ops.cumsum() / (d.n_prior + 1)  # deliberate leak: includes the current game
    return d


def game_table() -> pd.DataFrame:
    d = team_games()
    one_home = d.groupby("GAME_ID").is_home.transform("sum") == 1  # 5 neutral-site games (both '@') dropped
    h = d[d.is_home & one_home].set_index("GAME_ID")
    a = d[~d.is_home & one_home].set_index("GAME_ID")
    G = pd.DataFrame({"date": h.date, "season": h.season, "h_team": h.team, "a_team": a.team,
                      "margin": h.PLUS_MINUS, "h_n": h.n_prior, "a_n": a.n_prior,
                      "x": h.ops_pit - a.ops_pit, "x_leak": h.ops_leak - a.ops_leak}).reset_index()
    traps.check_join_aligned(G.assign(a_date=a.date.reindex(G.GAME_ID).values), "date", "a_date")
    gm = pd.read_parquet(ROOT / "data" / "nba_comebacks" / "games.parquet")[["GAME_ID", "spread", "line_src", "h_s_opp_paint_share", "a_s_opp_paint_share"]]
    G = traps.checked_merge(G, gm, on="GAME_ID", validate="one_to_one")
    return G


def zs(x, s):
    return (x - x.groupby(s).transform("mean")) / x.groupby(s).transform("std")


def ols(y, x):
    m = sm.OLS(np.asarray(y, float), sm.add_constant(np.asarray(x, float))).fit(cov_type="HC1")
    return float(m.params[1]), float(m.bse[1]), float(m.pvalues[1]), int(m.nobs)


def espn_close() -> pd.DataFrame:
    """ESPN core odds, one row per event: DraftKings plain (closing) spread + odds, else ESPN BET. seasons 2022..2025
    (= 2021-22..2024-25). Home spread in ESPN sign (negative = home favoured)."""
    fr = []
    for y in (2022, 2023, 2024, 2025):
        l = pd.read_parquet(ROOT / "data" / "espn" / "nba" / f"lines_{y}.parquet")
        g = pd.read_parquet(ROOT / "data" / "espn" / "nba" / f"games_{y}.parquet")
        g = g[g.season_type.astype(str).str.contains("regular", case=False)]
        g = g[g.status == "STATUS_FINAL"]; traps.check_final_status(g, "status")
        for prov in ("DraftKings", "ESPN BET"):
            x = l[(l.provider == prov) & l.spread.notna() & l.home_spread_odds.notna() & l.away_spread_odds.notna()]
            x = x.drop_duplicates("event_id").assign(prov=prov)
            fr.append(x[["event_id", "spread", "home_spread_odds", "away_spread_odds", "prov"]].merge(
                g[["event_id", "start_utc", "home", "away"]], on="event_id"))
    e = pd.concat(fr).drop_duplicates("event_id", keep="first")  # DK first
    e["date"] = pd.to_datetime(e.start_utc, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    e["h_team"] = e.home.replace(ESPN2NBA)
    e["a_team"] = e.away.replace(ESPN2NBA)
    return e


def cmd_repro() -> dict:
    G = game_table()
    res = {"paint_mismatch_rows": int(team_games().paint_mismatch.sum())}
    q = G[G.spread.notna() & (G.h_n >= 10) & (G.a_n >= 10)].copy()
    q["ats"] = q.margin - q.spread
    # audit (c): my PIT feature vs the original's s_opp_paint_share
    res["max_abs_diff_vs_original_feature"] = float(np.nanmax(np.abs(q.x - (q.h_s_opp_paint_share - q.a_s_opp_paint_share))))
    res["n"] = len(q)
    res["repro_within_season_z"] = ols(q.ats, zs(q.x, q.season))
    res["leak_including_current_game"] = ols(q.ats, zs(q.x_leak, q.season))
    # (e) walk-forward scale: SD of x from all earlier seasons (needs season >= 2016)
    sd = {s: q[q.season < s].x.std() for s in q.season.unique() if s > q.season.min()}
    w = q[q.season > q.season.min()].copy()
    w["zwf"] = w.x / w.season.map(sd)
    res["walkforward_scale_2016_24"] = ols(w.ats, w.zwf)
    res["same_rows_within_z"] = ols(w.ats, zs(w.x, w.season))
    for src, x in q.groupby("line_src"):
        x = x[x.groupby("season").x.transform("size") > 50]
        res[f"by_line_src_{src}"] = ols(x.ats, zs(x.x, x.season))
    # (f) vs ESPN core odds DK / ESPN BET close
    e = espn_close()
    m = q.merge(e[["date", "h_team", "a_team", "spread", "prov"]].rename(columns={"spread": "esp_close"}),
                on=["date", "h_team", "a_team"], how="inner")
    res["espn_close_join_n"] = len(m)
    res["espn_close_sign_check_corr(-esp_close, main spread)"] = float(np.corrcoef(-m.esp_close, m.spread)[0, 1])
    m["ats_c"] = m.margin + m.esp_close
    res["vs_espn_close_2021_24"] = ols(m.ats_c, zs(m.x, m.season))
    res["main_line_same_rows_2021_24"] = ols(m.ats, zs(m.x, m.season))
    res["vs_espn_close_DK_only"] = ols(m[m.prov == "DraftKings"].ats_c, zs(m[m.prov == "DraftKings"].x, m[m.prov == "DraftKings"].season))
    # raw margin (is the trait even predictive of results?) and spread (does the market already move on it?)
    res["x_vs_raw_margin"] = ols(q.margin, zs(q.x, q.season))
    res["x_vs_spread"] = ols(q.spread, zs(q.x, q.season))
    print(json.dumps(res, indent=1))
    (OUT / "repro.json").write_text(json.dumps(res, indent=1))
    return res


def cmd_trend() -> dict:
    G = game_table()
    q = G[G.spread.notna() & (G.h_n >= 10) & (G.a_n >= 10)].copy()
    q["ats"] = q.margin - q.spread
    q["z"] = zs(q.x, q.season)
    rows = []
    for s, x in q.groupby("season"):
        b, se, p, n = ols(x.ats, x.z)
        rows.append({"season": f"{s}-{str(s + 1)[2:]}", "s": s, "b": b, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "p": p, "n": n})
    t = pd.DataFrame(rows)
    wls = sm.WLS(t.b, sm.add_constant(t.s - 2015), weights=1 / ((t.hi - t.b) / 1.96) ** 2).fit()
    out = {"per_season": t.round(4).to_dict("records"),
           "trend_pts_per_season": [float(wls.params.iloc[1]), float(wls.conf_int().iloc[1, 0]), float(wls.conf_int().iloc[1, 1]), float(wls.pvalues.iloc[1])],
           "early_2015_19": ols(q[q.season <= 2019].ats, q[q.season <= 2019].z),
           "late_2020_24": ols(q[q.season >= 2020].ats, q[q.season >= 2020].z),
           "last2_2023_24": ols(q[q.season >= 2023].ats, q[q.season >= 2023].z)}
    print(t.round(3).to_string()); print({k: v for k, v in out.items() if k != "per_season"})
    (OUT / "trend.json").write_text(json.dumps(out, indent=1))
    return out


def boot_date(pnl: pd.Series, dates: pd.Series, B=10000, seed=2026):
    df = pd.DataFrame({"p": pnl.values, "d": dates.values})
    agg = df.groupby("d").p.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(agg), (B, len(agg)))
    s, c = agg["sum"].values[idx].sum(1), agg["count"].values[idx].sum(1)
    r = s / c
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), float((r <= 0).mean())


def build_bets() -> pd.DataFrame:
    """Selections for PP1/PP2 (outcome columns kept for my own scoring; edgelab hides them)."""
    G = game_table()
    q = G[(G.h_n >= 10) & (G.a_n >= 10) & G.x.notna()].copy()
    sd = {s: q[q.season < s].x.std() for s in range(2016, 2025)}
    q = q[q.season >= 2016].copy()
    q["z"] = q.x / q.season.map(sd)
    q["side"] = np.where(q.z >= 1.0, "away", np.where(q.z <= -1.0, "home", None))
    return q


def score(b: pd.DataFrame, line_col: str, dec_h: np.ndarray, dec_a: np.ndarray, label: str) -> dict:
    """line_col: home spread in 'expected home margin' sign (home covers if margin - line > 0)."""
    res = b.margin - b[line_col]
    cov = np.where(b.side == "home", np.sign(res), -np.sign(res))
    dec = np.where(b.side == "home", dec_h, dec_a)
    pnl = np.where(cov > 0, dec - 1, np.where(cov < 0, -1.0, 0.0))
    b = b.assign(cv=cov, pnl=pnl, dec=dec)
    lo, hi, p = boot_date(b.pnl, b.date)
    half = b.sort_values("date")
    h1, h2 = half.iloc[: len(half) // 2], half.iloc[len(half) // 2:]
    nz = b[b.cv != 0]
    out = {"trial": label, "bets": len(b), "win": float((nz.cv > 0).mean()), "roi": float(b.pnl.mean()),
           "ci": [lo, hi], "p_one_sided": p, "half1": float(h1.pnl.mean()), "half2": float(h2.pnl.mean()),
           "mean_abs_z": float(b.z.abs().mean()),
           "per_season": b.groupby("season").agg(n=("pnl", "size"), roi=("pnl", "mean")).round(4).reset_index().to_dict("records")}
    for t in (1.5, 2.0):
        x = b[b.z.abs() >= t]
        out[f"desc_absz_ge_{t}"] = {"bets": len(x), "roi": float(x.pnl.mean()), "win": float((x[x.cv != 0].cv > 0).mean())}
    b.to_csv(OUT / f"bets_{label}.csv", index=False)
    return out


def cmd_bets() -> dict:
    q = build_bets()
    sel = q[q.side.notna()]
    out = {}
    b1 = sel[sel.spread.notna()].copy()
    out["PP1"] = score(b1, "spread", np.full(len(b1), 1 + 100 / 110), np.full(len(b1), 1 + 100 / 110), "PP1")
    e = espn_close()
    b2 = sel.merge(e[["date", "h_team", "a_team", "spread", "home_spread_odds", "away_spread_odds", "prov"]].rename(
        columns={"spread": "esp_close"}), on=["date", "h_team", "a_team"], how="inner")
    b2["line2"] = -b2.esp_close
    am = lambda a: np.where(a > 0, 1 + a / 100, 1 + 100 / -a)
    out["PP2"] = score(b2, "line2", am(b2.home_spread_odds.astype(float)), am(b2.away_spread_odds.astype(float)), "PP2")
    out["PP2"]["provider_counts"] = b2.prov.value_counts().to_dict()
    out["PP2"]["eligible_games_2021_24_with_signal"] = int((sel.season >= 2021).sum())
    print(json.dumps(out, indent=1, default=float))
    (OUT / "bets.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def cmd_export() -> None:
    """edgelab frame (loader nba_paint_sides): one row per (trial, selected game); dev rows only (2016-17..2024-25)."""
    fr = []
    for t in ("PP1", "PP2"):
        b = pd.read_csv(OUT / f"bets_{t}.csv", parse_dates=["date"], dtype={"GAME_ID": str})
        fr.append(pd.DataFrame({"date": b.date, "game_id": b.GAME_ID.str.zfill(10), "trial": t, "sel": b.side,
                                "z": b.z, "dec": b.dec, "won": (b.cv > 0).astype(int), "push": (b.cv == 0).astype(int)}))
    f = pd.concat(fr, ignore_index=True)
    assert f.date.max() < pd.Timestamp("2025-07-01")
    f.to_parquet(OUT / "edgelab_frame.parquet", index=False)
    print(f.groupby("trial").size())


if __name__ == "__main__":
    globals()["cmd_" + sys.argv[1]]()
