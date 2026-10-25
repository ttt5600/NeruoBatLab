"""NBA comebacks, matchups, coaching, refs: what predicts winning BEYOND the market?  (2026-10-25)

User ask (after DEN came back from 23 down vs NOP): "team passing rates, offense, defense, all the metrics that
correlate to which one wins, what types of comebacks are possible, what schemes play well against what defenses --
coaching, reffing, all factors."

    python scripts/nba_comebacks.py pull        # stats.nba.com team/player game logs + coaches -> data/nba_comebacks/
    python scripts/nba_comebacks.py timeouts    # ESPN cached summaries 2021-22..2024-25 -> timeouts / runs table
    python scripts/nba_comebacks.py build       # games + closing lines + season-to-date traits (no lookahead)
    python scripts/nba_comebacks.py q1          # comeback heterogeneity (+ game-day lookup table)
    python scripts/nba_comebacks.py q2          # four-factor / style matchups vs closing spread + total
    python scripts/nba_comebacks.py q3          # coaching: coach ATS, late-game, timeouts, rest
    python scripts/nba_comebacks.py q4          # refs: crew x team-style interactions vs ATS
    python scripts/nba_comebacks.py q5          # other: shot luck, altitude, travel, ATS streaks
    python scripts/nba_comebacks.py all         # q1..q5 -> results/nba_comebacks.json

SCOPE / SEALED. Seasons 2015-16 .. 2024-25 only (ESPN/meta season label 2015..2024 = start year). 2025-26 is the
sealed NBA season in lab/MANIFEST and data/nba_ingame.parquet is reserved for lead L28: neither is read. ESPN box
files box_2026/box_2027 are not read.

TESTING RULES (written before any result was looked at).
  * Target is always vs the market: ATS residual = home margin - closing home spread (OddsData `Spread` = expected
    home margin); total residual = points - closing OU; in-game: outcome - ESPN WP at the state, and outcome -
    a pregame-adjusted empirical baseline (probit on deficit/sqrt(tau) and closing-spread*sqrt(tau), fit
    leave-one-season-out).
  * Traits are season-to-date means of games BEFORE the game (shift 1), only after >= 10 prior games.
  * Every cell is a test. BH-FDR at q = 0.10 within each question AND Bonferroni over all tests reported; a
    survivor also needs the same sign in 2015-16..2019-20 and 2020-21..2024-25.
  * Effects are reported per 1 SD of the trait (z-scored within season).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import numpy as np
import pandas as pd

OUT = ROOT / "data" / "nba_comebacks"
RES = ROOT / "results" / "nba_comebacks.json"
SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2025)]  # 2015-16 .. 2024-25 ; 2025-26 SEALED
SEASON_TYPES = ["Regular Season", "Playoffs"]


def _retry(fn, tries=5):
    for i in range(tries):
        try:
            return fn()
        except Exception as ex:  # stats.nba.com throttles
            print("  retry", i, str(ex)[:80], flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError("stats.nba.com failed")


def cmd_pull() -> int:
    from nba_api.stats.endpoints import commonteamroster, playergamelogs, teamgamelogs
    from nba_api.stats.static import teams as st
    OUT.mkdir(parents=True, exist_ok=True)
    tl, pl = [], []
    for s in SEASONS:
        for t in SEASON_TYPES:
            frames = []
            for mt in ["Base", "Advanced", "Misc", "Scoring"]:
                d = _retry(lambda: teamgamelogs.TeamGameLogs(season_nullable=s, season_type_nullable=t,
                                                             measure_type_player_game_logs_nullable=mt,
                                                             timeout=60).get_data_frames()[0])
                d = d[[c for c in d.columns if not c.endswith("_RANK")]]
                frames.append(d.set_index(["TEAM_ID", "GAME_ID"]))
                time.sleep(0.7)
            m = frames[0]
            for f in frames[1:]:
                m = m.join(f[[c for c in f.columns if c not in m.columns]])
            m = m.reset_index()
            m["season"] = int(s[:4])
            m["stype"] = t
            tl.append(m)
            p = _retry(lambda: playergamelogs.PlayerGameLogs(season_nullable=s, season_type_nullable=t,
                                                             timeout=60).get_data_frames()[0])
            p = p[["SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE", "MIN", "PTS",
                   "FGA", "FTA", "TOV", "FG3A", "AST", "PLUS_MINUS"]].copy()
            p["season"] = int(s[:4])
            p["stype"] = t
            pl.append(p)
            print(s, t, len(m), len(p), flush=True)
            time.sleep(0.7)
    pd.concat(tl, ignore_index=True).to_parquet(OUT / "teamlogs.parquet", index=False)
    pd.concat(pl, ignore_index=True).to_parquet(OUT / "playerlogs.parquet", index=False)
    co = []
    for s in SEASONS:
        for tm in st.get_teams():
            r = _retry(lambda: commonteamroster.CommonTeamRoster(team_id=tm["id"], season=s, timeout=30).get_data_frames()[1])
            r["season"] = int(s[:4])
            co.append(r)
            time.sleep(0.6)
        print("coaches", s, flush=True)
    pd.concat(co, ignore_index=True).to_parquet(OUT / "coaches.parquet", index=False)
    return 0


ESPN2NBA = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
NAME2NBA = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Bobcats": "CHA",
            "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL",
            "Denver Nuggets": "DEN", "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU",
            "Indiana Pacers": "IND", "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
            "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
            "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
            "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
            "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"}
# arena coordinates (lat, lon, elevation ft) for travel / altitude; approximate city centres are enough here
ARENA = {"ATL": (33.76, -84.40, 1050), "BOS": (42.37, -71.06, 20), "BKN": (40.68, -73.98, 30), "CHA": (35.23, -80.84, 750),
         "CHI": (41.88, -87.67, 600), "CLE": (41.50, -81.69, 650), "DAL": (32.79, -96.81, 430), "DEN": (39.75, -105.01, 5280),
         "DET": (42.34, -83.06, 600), "GSW": (37.77, -122.39, 10), "HOU": (29.75, -95.36, 50), "IND": (39.76, -86.16, 715),
         "LAC": (34.04, -118.27, 300), "LAL": (34.04, -118.27, 300), "MEM": (35.14, -90.05, 260), "MIA": (25.78, -80.19, 10),
         "MIL": (43.04, -87.92, 620), "MIN": (44.98, -93.28, 830), "NOP": (29.95, -90.08, 5), "NYK": (40.75, -73.99, 30),
         "OKC": (35.46, -97.52, 1200), "ORL": (28.54, -81.38, 100), "PHI": (39.90, -75.17, 40), "PHX": (33.45, -112.07, 1090),
         "POR": (45.53, -122.67, 50), "SAC": (38.58, -121.50, 30), "SAS": (29.43, -98.44, 650), "TOR": (43.64, -79.38, 250),
         "UTA": (40.77, -111.90, 4230), "WAS": (38.90, -77.02, 50)}


def _secs(c):
    try:
        m, s = c.split(":") if ":" in c else ("0", c)
        return int(m) * 60 + float(s)
    except Exception:
        return np.nan


def cmd_timeouts() -> int:
    """Timeouts from ESPN summaries already in the disk cache (2021-22 .. 2024-25). Never fetches."""
    from concurrent.futures import ThreadPoolExecutor
    from sports.http import CACHE_DIR, _key
    m = pd.read_parquet(ROOT / "data" / "nba_big_lead_meta.parquet")
    m = m[m.cached & m.season.between(2021, 2024)]
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={}"

    def one(eid):
        f = CACHE_DIR / (_key(url.format(eid), None) + ".json")
        if not f.exists():
            return []
        j = json.loads(f.read_text())
        comp = j["header"]["competitions"][0]
        side = {c["id"]: c["homeAway"] for c in comp["competitors"]}
        out = []
        for p in j.get("plays") or []:
            tt = p.get("type", {}).get("text", "")
            if "imeout" not in tt:
                continue
            out.append((eid, int(p.get("sequenceNumber", 0)), p.get("period", {}).get("number", 0),
                        _secs(p.get("clock", {}).get("displayValue", "")), p.get("homeScore"), p.get("awayScore"),
                        tt, side.get((p.get("team") or {}).get("id"), "none")))
        return out

    rows = []
    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(one, m.eid)):
            rows += r
            if i % 1000 == 0:
                print(i, len(rows), flush=True)
    d = pd.DataFrame(rows, columns=["eid", "seq", "period", "clock_s", "hs", "as", "type", "side"])
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "timeouts.parquet", index=False)
    print(d.type.value_counts().head(10), d.side.value_counts())
    return 0


# ----------------------------------------------------------------------------------------------- build
TRAITS = ["pace", "ortg", "drtg", "net", "fg3a_rate", "ft_rate", "tov_pct", "orb_pct", "drb_pct", "ast_ratio",
          "pct_ast_fgm", "paint_share", "fb_share", "mr_share", "fg3_pct", "ft_pct", "opp_fg3a_rate", "opp_ft_rate",
          "opp_tov_pct", "opp_paint_share", "opp_fg3_pct", "opp_fb_share", "bench_min_share", "starter_net48",
          "bench_net48", "dropoff48", "star_share", "blowout_core_share", "margin", "ats_cov"]


def _odds() -> pd.DataFrame:
    import sqlite3
    c = sqlite3.connect(ROOT / "data" / "nba_odds" / "OddsData.sqlite")
    fr = []
    for y in range(2015, 2025):
        t = f"odds_{y}-{str(y + 1)[2:]}_new" if y < 2023 else f"{y}-{str(y + 1)[2:]}"
        d = pd.read_sql(f'select Date, Home, Away, OU, Spread, ML_Home, ML_Away from "{t}"', c)
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d["date"] = pd.to_datetime(d.Date)
    d["home"] = d.Home.map(NAME2NBA)
    d["away"] = d.Away.map(NAME2NBA)
    assert d.home.notna().all() and d.away.notna().all()
    return d.rename(columns={"OU": "total_line", "Spread": "spread", "ML_Home": "ml_h", "ML_Away": "ml_a"})[
        ["date", "home", "away", "total_line", "spread", "ml_h", "ml_a"]].drop_duplicates(["date", "home"])


def _player_agg() -> pd.DataFrame:
    p = pd.read_parquet(OUT / "playerlogs.parquet")
    p = p[p.MIN > 0].copy()
    p["use"] = p.FGA + 0.44 * p.FTA + p.TOV
    p = p.sort_values(["GAME_ID", "TEAM_ID", "MIN"], ascending=[True, True, False])
    p["rk"] = p.groupby(["GAME_ID", "TEAM_ID"]).cumcount()
    p["top5"] = p.rk < 5
    g = p.groupby(["GAME_ID", "TEAM_ID"])
    tot = g.agg(min_all=("MIN", "sum"), use_all=("use", "sum"), use_max=("use", "max"))
    s = p[p.top5].groupby(["GAME_ID", "TEAM_ID"]).agg(min5=("MIN", "sum"), pm5=("PLUS_MINUS", "sum"))
    b = p[~p.top5].groupby(["GAME_ID", "TEAM_ID"]).agg(minb=("MIN", "sum"), pmb=("PLUS_MINUS", "sum"))
    a = tot.join(s).join(b).fillna({"minb": 0, "pmb": 0})
    a["bench_min_share"] = a.minb / a.min_all
    a["starter_net48"] = a.pm5 / a.min5 * 48
    a["bench_net48"] = np.where(a.minb > 20, a.pmb / a.minb.clip(lower=1) * 48, np.nan)
    a["dropoff48"] = a.starter_net48 - a.bench_net48
    a["star_share"] = a.use_max / a.use_all
    a["core_share"] = a.min5 / a.min_all
    return a.reset_index()[["GAME_ID", "TEAM_ID", "bench_min_share", "starter_net48", "bench_net48", "dropoff48",
                            "star_share", "core_share"]]


def cmd_build() -> int:
    t = pd.read_parquet(OUT / "teamlogs.parquet")
    t["date"] = pd.to_datetime(t.GAME_DATE.str[:10])
    t["is_home"] = t.MATCHUP.str.contains("vs.")
    t["team"] = t.TEAM_ABBREVIATION
    d = pd.DataFrame({"GAME_ID": t.GAME_ID, "TEAM_ID": t.TEAM_ID, "team": t.team, "date": t.date, "season": t.season,
                      "stype": t.stype, "is_home": t.is_home, "pts": t.PTS, "margin": t.PLUS_MINUS,
                      "pace": t.PACE, "ortg": t.OFF_RATING, "drtg": t.DEF_RATING, "net": t.NET_RATING,
                      "fga": t.FGA, "fg3a": t.FG3A, "fg3m": t.FG3M, "fta": t.FTA, "ftm": t.FTM,
                      "tov_pct": t.TM_TOV_PCT, "orb_pct": t.OREB_PCT, "drb_pct": t.DREB_PCT, "ast_ratio": t.AST_RATIO,
                      "pct_ast_fgm": t.PCT_AST_FGM, "paint_share": t.PTS_PAINT / t.PTS, "fb_share": t.PTS_FB / t.PTS,
                      "mr_share": t.PCT_PTS_2PT_MR, "opp_paint": t.OPP_PTS_PAINT, "opp_fb": t.OPP_PTS_FB})
    o = d[["GAME_ID", "TEAM_ID", "team", "pts", "fga", "fg3a", "fg3m", "fta", "tov_pct"]].rename(
        columns=lambda c: c if c == "GAME_ID" else "o_" + c)
    d = d.merge(o, on="GAME_ID")
    d = d[d.TEAM_ID != d.o_TEAM_ID].copy()
    d["opp"] = d.o_team
    d["fg3a_rate"] = d.fg3a / d.fga
    d["ft_rate"] = d.fta / d.fga
    d["opp_fg3a_rate"] = d.o_fg3a / d.o_fga
    d["opp_ft_rate"] = d.o_fta / d.o_fga
    d["opp_tov_pct"] = d.o_tov_pct
    d["opp_paint_share"] = d.opp_paint / d.o_pts
    d["opp_fb_share"] = d.opp_fb / d.o_pts
    d = d.merge(_player_agg(), on=["GAME_ID", "TEAM_ID"], how="left")
    # closing lines (home perspective) -> ATS cover for each team-game (used as a trait: ATS form)
    od = _odds()
    hm = d[d.is_home][["GAME_ID", "date", "team", "opp"]].rename(columns={"team": "home", "opp": "away"})
    hm = hm.merge(od, on=["date", "home", "away"], how="left")
    miss = hm.spread.isna()
    if miss.any():  # date off by one in a few rows
        alt = od.assign(date=od.date - pd.Timedelta(days=1))
        fill = hm[miss].drop(columns=["total_line", "spread", "ml_h", "ml_a"]).merge(alt, on=["date", "home", "away"], how="left")
        hm.loc[miss, ["total_line", "spread", "ml_h", "ml_a"]] = fill[["total_line", "spread", "ml_h", "ml_a"]].values
    for c in ["total_line", "spread", "ml_h", "ml_a"]:
        hm[c] = pd.to_numeric(hm[c], errors="coerce")
    # OddsData `Spread` is UNSIGNED in older seasons (ATL +9.5 as a 300 dog): sign it by the moneyline favourite.
    hm["spread"] = hm.spread.abs() * np.sign(hm.ml_a - hm.ml_h)
    # ESPN pickcenter spread (meta) is preferred where it exists (2015-16..2022-23): OddsData disagrees badly in
    # 2019-20 (corr with ESPN 0.58, corr with margin 0.21 vs ESPN 0.43). OddsData only for 2023-24, 2024-25.
    mt = pd.read_parquet(ROOT / "data" / "nba_big_lead_meta.parquet")
    mt = mt[mt.season <= 2024].assign(home=lambda x: x.home.replace(ESPN2NBA))[["date", "home", "spread"]].rename(columns={"spread": "esp"})
    hm = hm.merge(mt.drop_duplicates(["date", "home"]), on=["date", "home"], how="left")
    hm["line_src"] = np.where(hm.esp.notna(), "espn_pickcenter", np.where(hm.spread.notna(), "oddsdata", "none"))
    hm["spread"] = np.where(hm.esp.notna(), -hm.esp, hm.spread)
    # totals: OddsData has junk rows (e.g. 1955.5); keep 170..270 only
    hm.loc[~hm.total_line.between(170, 270), "total_line"] = np.nan
    d = d.merge(hm[["GAME_ID", "spread", "total_line", "ml_h", "ml_a", "line_src"]], on="GAME_ID", how="left")
    d["team_spread"] = np.where(d.is_home, d.spread, -d.spread)  # expected margin for this team
    d["ats_res"] = d.margin - d.team_spread
    d["ats_cov"] = np.sign(d.ats_res)
    d = d.sort_values(["team", "date"]).reset_index(drop=True)
    gb = d.groupby(["team", "season"], sort=False)
    # season-to-date (strictly before this game)
    d["n_prior"] = gb.cumcount()
    for c in ["pace", "ortg", "drtg", "net", "tov_pct", "orb_pct", "drb_pct", "ast_ratio", "pct_ast_fgm", "paint_share",
              "fb_share", "mr_share", "opp_tov_pct", "opp_paint_share", "opp_fb_share", "bench_min_share", "starter_net48",
              "bench_net48", "dropoff48", "star_share", "margin", "ats_cov"]:
        d["s_" + c] = gb[c].transform(lambda x: x.shift(1).expanding().mean())
    for num, den, name in [("fg3a", "fga", "fg3a_rate"), ("fta", "fga", "ft_rate"), ("fg3m", "fg3a", "fg3_pct"),
                           ("ftm", "fta", "ft_pct"), ("o_fg3a", "o_fga", "opp_fg3a_rate"), ("o_fta", "o_fga", "opp_ft_rate"),
                           ("o_fg3m", "o_fg3a", "opp_fg3_pct")]:
        d["s_" + name] = gb[num].transform(lambda x: x.shift(1).cumsum()) / gb[den].transform(lambda x: x.shift(1).cumsum())
    # starter-pull habit: core (top-5 minutes) share in prior wins by 15+ (lower = pulls starters earlier)
    d["_bw"] = (d.margin >= 15).astype(float)
    d["_bwv"] = d.core_share.fillna(0) * d._bw
    nb = gb["_bw"].transform(lambda x: x.shift(1).cumsum())
    d["s_blowout_core_share"] = gb["_bwv"].transform(lambda x: x.shift(1).cumsum()) / nb.where(nb >= 3)
    d["s_last5_ats"] = gb["ats_cov"].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    # schedule
    d["prev_date"] = gb.date.shift(1)
    d["rest"] = (d.date - d.prev_date).dt.days.clip(upper=5)
    d["d2"] = gb.date.shift(2)
    d["three_in_four"] = ((d.date - gb.date.shift(2)).dt.days <= 3).astype(float)
    loc = np.where(d.is_home, d.team, d.opp)
    d["loc"] = loc
    d["prev_loc"] = gb["loc"].shift(1)
    lat = d["loc"].map(lambda k: ARENA[k][0]); lon = d["loc"].map(lambda k: ARENA[k][1])
    plat = d.prev_loc.map(lambda k: ARENA[k][0] if isinstance(k, str) else np.nan)
    plon = d.prev_loc.map(lambda k: ARENA[k][1] if isinstance(k, str) else np.nan)
    r = np.pi / 180
    d["travel_km"] = 6371 * np.arccos(np.clip(np.sin(lat * r) * np.sin(plat * r) + np.cos(lat * r) * np.cos(plat * r) * np.cos((lon - plon) * r), -1, 1))
    d["road_streak"] = gb["is_home"].transform(lambda x: (~x).groupby(x.cumsum()).cumsum())
    d["prev_margin"] = gb["margin"].shift(1)
    d["prev_ats_res"] = gb["ats_res"].shift(1)
    d = d.drop(columns=["_bw", "_bwv", "d2"])
    # bubble / neutral: 2019-20 restart (from 2020-07-30) is neutral-site
    d["bubble"] = (d.season == 2019) & (d.date >= "2020-07-30")
    d.to_parquet(OUT / "teamgames.parquet", index=False)
    # one row per game, home perspective
    S = [c for c in d.columns if c.startswith("s_")] + ["n_prior", "rest", "three_in_four", "travel_km", "road_streak",
                                                         "prev_margin", "prev_ats_res", "team", "TEAM_ID"]
    h = d[d.is_home].set_index("GAME_ID")
    a = d[~d.is_home].set_index("GAME_ID")
    g = h[["date", "season", "stype", "bubble", "line_src", "spread", "total_line", "ml_h", "ml_a", "pts", "o_pts", "margin"]].rename(
        columns={"pts": "hpts", "o_pts": "apts"})
    g = g.join(h[S].add_prefix("h_")).join(a[S].add_prefix("a_"))
    g["ats_res"] = g.margin - g.spread
    g["tot_res"] = g.hpts + g.apts - g.total_line
    # ESPN event id (for pbp / refs)
    m = pd.read_parquet(ROOT / "data" / "nba_big_lead_meta.parquet")
    m = m[m.season <= 2024]
    m["home_n"] = m.home.replace(ESPN2NBA)
    g = g.reset_index().merge(m[["eid", "date", "home_n", "spread", "p0_espn"]].rename(
        columns={"home_n": "h_team", "spread": "espn_spread"}), on=["date", "h_team"], how="left")
    g.to_parquet(OUT / "games.parquet", index=False)
    print(g.groupby(["season", "stype"]).agg(n=("GAME_ID", "size"), line=("spread", lambda x: x.notna().mean()),
                                             eid=("eid", lambda x: x.notna().mean())))
    ok = g.dropna(subset=["spread", "espn_spread"])
    print("OddsData spread vs ESPN pickcenter: corr", np.corrcoef(ok.spread, -ok.espn_spread)[0, 1].round(4),
          "mean |diff|", (ok.spread + ok.espn_spread).abs().mean().round(3), "n", len(ok))
    print("ATS residual mean/sd", g.ats_res.mean().round(3), g.ats_res.std().round(2), " total res", g.tot_res.mean().round(2), g.tot_res.std().round(2))
    return 0


# ----------------------------------------------------------------------------------------------- stats helpers
def zs(x: pd.Series, season: pd.Series) -> pd.Series:
    g = x.groupby(season)
    return (x - g.transform("mean")) / g.transform("std")


def ols1(y, x, cluster=None):
    """y ~ a + b x ; returns b, se (HC1 or cluster-robust), p, n."""
    import statsmodels.api as sm
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = np.asarray(y)[ok], np.asarray(x)[ok]
    X = sm.add_constant(x)
    if cluster is not None:
        r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(cluster)[ok]})
    else:
        r = sm.OLS(y, X).fit(cov_type="HC1")
    return float(r.params[1]), float(r.bse[1]), float(r.pvalues[1]), int(ok.sum())


def bh(p, q=0.10):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    passed = p[o] <= thr
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(n, bool)
    out[o[:k]] = True
    return out


def test_table(rows, q=0.10):
    t = pd.DataFrame(rows)
    t["bh"] = bh(t.p.values, q)
    t["halves_same_sign"] = np.sign(t.b_early) == np.sign(t.b_late)
    t["survivor"] = t.bh & t.halves_same_sign
    return t


def run_cell(df, ycol, xcol, name, family, scale=1.0, cluster=None):
    y, x = df[ycol].values, df[xcol].values
    b, se, p, n = ols1(y, x, None if cluster is None else df[cluster].values)
    e = df.season <= 2019
    be = ols1(y[e.values], x[e.values])[0] if e.sum() > 50 else np.nan
    bl = ols1(y[~e.values], x[~e.values])[0] if (~e).sum() > 50 else np.nan
    per = []
    for s, dd in df.groupby("season"):
        if dd[xcol].notna().sum() > 50:
            per.append(np.sign(ols1(dd[ycol].values, dd[xcol].values)[0]))
    return {"family": family, "test": name, "b": b * scale, "lo": (b - 1.96 * se) * scale, "hi": (b + 1.96 * se) * scale,
            "p": p, "n": n, "b_early": be * scale, "b_late": bl * scale,
            "seasons_same_sign": f"{int(np.sum(np.array(per) == np.sign(b)))}/{len(per)}"}


# ----------------------------------------------------------------------------------------------- Q1 comebacks
GRID = np.arange(60, 2880, 60)  # elapsed regulation seconds at which states are sampled


def states() -> pd.DataFrame:
    """One row per game per elapsed minute: home margin, ESPN WP, outcome. Seasons 2015-16..2024-25 only."""
    f = OUT / "states.parquet"
    if f.exists():
        return pd.read_parquet(f)
    g = pd.read_parquet(OUT / "games.parquet").dropna(subset=["eid"])
    p = pd.read_parquet(ROOT / "data" / "nba_big_lead_plays.parquet", columns=["eid", "seq", "period", "clock_s", "hs", "as", "espn_wp"])
    p = p[p.eid.isin(set(g.eid)) & (p.period <= 4)].dropna(subset=["clock_s"])
    p["el"] = (p.period - 1) * 720 + (720 - p.clock_s)
    p = p.sort_values(["eid", "el", "seq"])
    grid = pd.DataFrame({"eid": np.repeat(g.eid.values, len(GRID)), "el": np.tile(GRID, len(g))}).astype({"el": float})
    grid = grid.sort_values(["el"])
    p = p.sort_values(["el"])
    s = pd.merge_asof(grid, p[["eid", "el", "hs", "as", "espn_wp"]], on="el", by="eid", direction="backward")
    s["m"] = s.hs - s["as"]
    s = s.merge(g[["eid", "GAME_ID", "season", "stype", "spread", "margin", "date", "h_team", "a_team"]], on="eid")
    s["home_won"] = (s.margin > 0).astype(int)
    s = s.dropna(subset=["m"])
    s.to_parquet(f, index=False)
    return s


def baseline_features(d):
    tau = (2880 - d.el) / 2880
    st = np.sqrt(tau)
    z = -d.deficit / st
    # quadratic terms added after a first look at calibration: the linear version ran 1-5 pts too HIGH for
    # 20+ deficits (log loss .45139 -> .45124 LOSO). Choice made on calibration only, before any trait test was rerun.
    return np.column_stack([z, d.s_tr * st, d.deficit, d.s_tr, tau, d.deficit * tau, z ** 2, z * d.s_tr * st, (d.s_tr * st) ** 2])


def fit_baseline(s: pd.DataFrame) -> pd.DataFrame:
    """Trailer-perspective states (deficit >= 1) + leave-one-season-out logistic baseline p_base."""
    from sklearn.linear_model import LogisticRegression
    d = s[(s.m != 0)].copy()
    d["home_trails"] = d.m < 0
    d["deficit"] = d.m.abs()
    d["s_tr"] = np.where(d.home_trails, d.spread, -d.spread)  # trailer's pregame expected margin
    d["y"] = np.where(d.home_trails, d.home_won, 1 - d.home_won)
    d["wp_espn"] = np.where(d.home_trails, d.espn_wp, 1 - d.espn_wp)
    d["tr_team"] = np.where(d.home_trails, d.h_team, d.a_team)
    d["ld_team"] = np.where(d.home_trails, d.a_team, d.h_team)
    d = d.dropna(subset=["s_tr"])
    d["p_base"] = np.nan
    for se in sorted(d.season.unique()):
        tr = d.season != se
        lr = LogisticRegression(C=1e4, max_iter=2000).fit(baseline_features(d[tr]), d.y[tr])
        d.loc[~tr, "p_base"] = lr.predict_proba(baseline_features(d[~tr]))[:, 1]
    return d


def lookup_model(d):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1e4, max_iter=2000).fit(baseline_features(d), d.y)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def trait_frame(stype="Regular Season"):
    tg = pd.read_parquet(OUT / "teamgames.parquet")
    tg = tg[tg.stype == stype].copy()
    S = [c for c in tg.columns if c.startswith("s_")]
    ok = tg.n_prior >= 10
    for c in S:
        tg.loc[~ok, c] = np.nan
        tg["z" + c[1:]] = zs(tg[c], tg.season)
    return tg


Q1_TRAILER = ["pace", "ortg", "net", "fg3a_rate", "fg3_pct", "ft_rate", "orb_pct", "tov_pct", "ast_ratio", "bench_net48",
              "bench_min_share", "star_share"]
Q1_LEADER = ["drtg", "net", "pace", "opp_fg3a_rate", "tov_pct", "ft_pct", "ft_rate", "dropoff48", "bench_net48",
             "blowout_core_share"]
WINDOWS = {"halftime": 1440, "end_Q3": 2160, "Q4_6min": 2520}


def cmd_q1() -> dict:
    s = states()
    d = fit_baseline(s)
    res = {}
    # calibration of the baseline (LOSO) by deficit x time, deficit >= 10
    cal = d[d.deficit >= 10].copy()
    cal["db"] = pd.cut(cal.deficit, [9, 14, 19, 24, 60], labels=["10-14", "15-19", "20-24", "25+"])
    cal["qb"] = pd.cut(cal.el, [0, 720, 1440, 2160, 2880], labels=["Q1", "Q2", "Q3", "Q4"])
    one = cal.sort_values("el").groupby(["eid", "db", "qb"], observed=True).head(1)  # first state per game per cell
    c = one.groupby(["qb", "db"], observed=True).agg(n=("y", "size"), won=("y", "mean"), p_base=("p_base", "mean"),
                                                     espn=("wp_espn", "mean"))
    print("\nBaseline calibration (first state per game per cell, LOSO):\n", c.round(4))
    res["calibration"] = c.reset_index().astype({"qb": str, "db": str}).to_dict("records")

    # ---- comeback types: first time a team trails by >= D, by quarter of that moment and by pregame role
    rows = []
    for D in (15, 20, 25):
        f = d[d.deficit >= D].sort_values("el").groupby("eid").head(1).copy()
        f["q"] = (f.el // 720 + 1).clip(upper=4).astype(int)
        f["role"] = pd.cut(f.s_tr, [-99, -3.5, 3.5, 99], labels=["trailer dog >3.5", "pick +/-3.5", "trailer fav >3.5"])  # s_tr < 0 = trailer was the underdog
        for (q, role), x in f.groupby(["q", "role"], observed=True):
            lo, hi = wilson(x.y.sum(), len(x))
            rows.append({"D": D, "quarter": int(q), "role": str(role), "n": len(x), "won": int(x.y.sum()),
                         "rate": x.y.mean(), "lo": lo, "hi": hi, "p_base": x.p_base.mean(), "espn": x.wp_espn.mean()})
        for q, x in f.groupby("q"):
            lo, hi = wilson(x.y.sum(), len(x))
            rows.append({"D": D, "quarter": int(q), "role": "all", "n": len(x), "won": int(x.y.sum()), "rate": x.y.mean(),
                         "lo": lo, "hi": hi, "p_base": x.p_base.mean(), "espn": x.wp_espn.mean()})
    ct = pd.DataFrame(rows)
    print("\nComeback types (first time trailing by >= D):\n", ct[ct.role == "all"].round(3).to_string())
    print(ct[ct.role != "all"].round(3).to_string())
    res["comeback_types"] = ct.to_dict("records")

    # ---- game-day lookup from the full-sample model
    lm = lookup_model(d)
    times = {"Q2 9:00": 720 + 180, "Q2 6:00": 720 + 360, "Q2 3:00": 720 + 540, "halftime": 1440, "Q3 6:00": 1440 + 360,
             "end Q3": 2160, "Q4 9:00": 2160 + 180, "Q4 6:00": 2160 + 360, "Q4 3:00": 2160 + 540}
    look = []
    for tl, el in times.items():
        for D in (10, 15, 20, 25, 30):
            for sp in (-8, -4, 0, 4, 8):
                x = pd.DataFrame({"deficit": [D], "s_tr": [sp], "el": [el]})
                pr = lm.predict_proba(baseline_features(x))[0, 1]
                emp = d[(d.deficit.between(D - 2, D + 2)) & (d.el.between(el - 120, el + 120)) & (d.s_tr.between(sp - 3, sp + 3))]
                emp = emp.groupby("eid").head(1)
                look.append({"time": tl, "deficit": D, "trailer_spread": sp, "model": pr, "emp_n": len(emp),
                             "emp_rate": emp.y.mean() if len(emp) else np.nan})
    lk = pd.DataFrame(look)
    print("\nLookup (model; trailer_spread = trailer's pregame expected margin, + = trailer favoured):")
    print(lk.pivot_table(index=["time", "deficit"], columns="trailer_spread", values="model", sort=False).round(3).to_string())
    res["lookup"] = lk.to_dict("records")

    # ---- trait heterogeneity (residual vs LOSO baseline), regular season, deficit >= 10, one state per game per window
    tf = trait_frame()
    key = tf.set_index(["GAME_ID", "team"])
    zc = [c for c in tf.columns if c.startswith("z_")]
    rows = []
    wins = {}
    for wn, el in WINDOWS.items():
        w = d[(d.el == el) & (d.deficit >= 10) & (d.stype == "Regular Season")].copy()
        T = key.loc[list(zip(w.GAME_ID, w.tr_team)), zc].reset_index(drop=True).add_prefix("T")
        L = key.loc[list(zip(w.GAME_ID, w.ld_team)), zc].reset_index(drop=True).add_prefix("L")
        w = pd.concat([w.reset_index(drop=True), T, L], axis=1)
        w["res"] = w.y - w.p_base
        w["res_espn"] = w.y - w.wp_espn
        w["Tx3_vs_L3allowed"] = w["Tz_fg3a_rate"] * w["Lz_opp_fg3a_rate"]
        wins[wn] = w
        we = w[w.wp_espn.notna()]
        print(f"\nwindow {wn}: n {len(w)}, comeback rate {w.y.mean():.4f}, p_base {w.p_base.mean():.4f}; ESPN subset n {len(we)}: "
              f"rate {we.y.mean():.4f} p_base {we.p_base.mean():.4f} ESPN {we.wp_espn.mean():.4f}")
        res[f"window_{wn}"] = {"n": len(w), "rate": w.y.mean(), "p_base": w.p_base.mean(), "n_espn": len(we),
                               "rate_espn_subset": we.y.mean(), "espn": float(we.wp_espn.mean())}
        cells = [("T" + "z_" + c, f"trailer {c}") for c in Q1_TRAILER] + [("L" + "z_" + c, f"leader {c}") for c in Q1_LEADER] \
            + [("Tx3_vs_L3allowed", "trailer 3PA rate x leader 3PA allowed")]
        for col, nm in cells:
            r = run_cell(w, "res", col, f"{wn}: {nm}", "Q1", scale=100)
            ok = w.res_espn.notna()
            r["b_espn"] = ols1(w.res_espn[ok].values, w[col][ok].values)[0] * 100
            rows.append(r)
    t = test_table(rows)
    print(f"\nQ1 trait tests: {len(t)} cells, BH survivors {int(t.bh.sum())}, +halves {int(t.survivor.sum())}")
    print(t.sort_values("p").head(15).round(4).to_string())
    res["tests"] = t.to_dict("records")
    # descriptive: comeback rate by trailer 3PA-rate tercile at halftime / end Q3 (deficit >= 15)
    desc = []
    for wn in ("halftime", "end_Q3"):
        w = wins[wn][wins[wn].deficit >= 15].dropna(subset=["Tz_fg3a_rate"])
        w["terc"] = pd.qcut(w.Tz_fg3a_rate, 3, labels=["low 3PA", "mid", "high 3PA"])
        for k, x in w.groupby("terc", observed=True):
            desc.append({"window": wn, "trailer_3pa": str(k), "n": len(x), "rate": x.y.mean(), "p_base": x.p_base.mean()})
    print(pd.DataFrame(desc).round(4))
    res["desc_3pa"] = desc
    return res


# ----------------------------------------------------------------------------------------------- Q2 matchups
def game_frame(stype="Regular Season"):
    g = pd.read_parquet(OUT / "games.parquet")
    g = g[(g.stype == stype) & g.spread.notna() & (g.h_n_prior >= 10) & (g.a_n_prior >= 10)].copy()
    return g


ATS_DIFFS = ["net", "ortg", "drtg", "pace", "fg3a_rate", "ft_rate", "tov_pct", "orb_pct", "drb_pct", "ast_ratio",
             "pct_ast_fgm", "paint_share", "fb_share", "mr_share", "fg3_pct", "ft_pct", "opp_fg3a_rate", "opp_ft_rate",
             "opp_tov_pct", "opp_paint_share", "opp_fg3_pct", "opp_fb_share", "bench_min_share", "bench_net48",
             "dropoff48", "star_share", "blowout_core_share", "ats_cov", "last5_ats"]
# (offense trait, defence-allowed trait, sign): matchup = H_off * A_def - A_off * H_def ; for ORB use -DRB as "allowed"
MATCHUPS = [("fg3a_rate", "opp_fg3a_rate", 1), ("paint_share", "opp_paint_share", 1), ("orb_pct", "drb_pct", -1),
            ("tov_pct", "opp_tov_pct", 1), ("ft_rate", "opp_ft_rate", 1), ("fb_share", "opp_fb_share", 1),
            ("fg3_pct", "opp_fg3_pct", 1)]
TOT_SUMS = ["pace", "ortg", "drtg", "fg3a_rate", "ft_rate", "orb_pct", "tov_pct", "paint_share", "fb_share", "fg3_pct",
            "opp_fg3_pct", "opp_ft_rate", "opp_fg3a_rate", "bench_min_share"]


def q2_features(g):
    F = {}
    for c in ATS_DIFFS:
        F[f"ATS diff {c}"] = ("ats_res", g["h_s_" + c] - g["a_s_" + c])
    for o, dfn, sg in MATCHUPS:
        zh_o, za_o = zs(g["h_s_" + o], g.season), zs(g["a_s_" + o], g.season)
        zh_d, za_d = sg * zs(g["h_s_" + dfn], g.season), sg * zs(g["a_s_" + dfn], g.season)
        F[f"ATS matchup {o} x {dfn}"] = ("ats_res", zh_o * za_d - za_o * zh_d)
        F[f"TOT matchup {o} x {dfn}"] = ("tot_res", zh_o * za_d + za_o * zh_d)
    zp = zs(g.h_s_pace + g.a_s_pace, g.season)
    F["ATS pace-sum x favourite (fav covers more in fast games?)"] = ("ats_res", zp * np.sign(g.spread))
    F["ATS pace mismatch x home faster"] = ("ats_res", g.h_s_pace - g.a_s_pace)
    for c in TOT_SUMS:
        F[f"TOT sum {c}"] = ("tot_res", g["h_s_" + c] + g["a_s_" + c])
    F["TOT pace mismatch |diff|"] = ("tot_res", (g.h_s_pace - g.a_s_pace).abs())
    F["TOT |spread| (blowout risk)"] = ("tot_res", g.spread.abs())
    return F


def cmd_q2() -> dict:
    g = game_frame()
    print(f"Q2 games {len(g)}  seasons {g.season.min()}-{g.season.max()}  ATS res mean {g.ats_res.mean():.2f} sd {g.ats_res.std():.2f}"
          f"  TOT res mean {g.tot_res.mean():.2f} sd {g.tot_res.std():.2f}")
    rows = []
    for name, (y, x) in q2_features(g).items():
        g["_x"] = zs(x, g.season)
        rows.append(run_cell(g, y, "_x", name, "Q2"))
    t = test_table(rows)
    print(f"Q2 cells {len(t)}; BH survivors {int(t.bh.sum())}; +halves {int(t.survivor.sum())}; min p {t.p.min():.4g}")
    print(t.sort_values("p").head(15).round(4).to_string())
    res = {"n_games": len(g), "tests": t.to_dict("records")}
    # priced-ness sanity: raw outcome vs net-rating diff (good teams win) and vs ATS residual
    x = zs(g.h_s_net - g.a_s_net, g.season)
    res["net_vs_margin"] = ols1(g.margin.values, x.values)
    res["net_vs_ats"] = ols1(g.ats_res.values, x.values)
    print("net-rating diff per SD -> raw margin", np.round(res["net_vs_margin"][:3], 4), " -> ATS residual", np.round(res["net_vs_ats"][:3], 4))
    # walk-forward check for every BH survivor: fit slope on prior seasons, bet sign when |pred| >= 1 pt, -110
    res["walk_forward"] = walk_forward(g, q2_features(g), t[t.survivor].test.tolist())
    # robustness 1: survivors re-scored vs the multi-book median line in refs_dev (2021-22..2024-25)
    rf = pd.read_parquet(ROOT / "data" / "espn" / "nba" / "refs_dev.parquet")
    rf = rf[rf.season <= 2025]
    rf["date"] = pd.to_datetime(rf.start_utc).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    rf["h_team"] = rf.home.replace(ESPN2NBA)
    gg = g.merge(rf[["date", "h_team", "mkt_total", "home_spread"]], on=["date", "h_team"], how="inner")
    gg["ats_res"] = gg.margin + gg.home_spread  # ESPN sign: negative = home favoured
    gg["tot_res"] = gg.hpts + gg.apts - gg.mkt_total
    F2 = q2_features(gg)
    rob = {}
    for nm in t[t.survivor].test:
        y, x = F2[nm]
        gg["_x"] = zs(x, gg.season)
        b, se, p, n = ols1(gg[y].values, gg["_x"].values)
        b0 = ols1(g[g.season >= 2021][y].values, zs(q2_features(g)[nm][1], g.season)[g.season >= 2021].values)[0]
        rob[nm] = {"b_multibook_2021_24": b, "se": se, "p": p, "n": n, "b_main_line_same_seasons": b0}
        print("robust (multi-book line 2021-24)", nm, {k: round(v, 4) for k, v in rob[nm].items()})
    res["robust_multibook"] = rob
    # robustness 2: is opp_paint_share just 3P defence luck? joint model with opp_fg3_pct and opp_fg3a_rate diffs
    import statsmodels.api as sm
    X = pd.DataFrame({c: zs(g["h_s_" + c] - g["a_s_" + c], g.season) for c in ["opp_paint_share", "opp_fg3_pct", "opp_fg3a_rate", "paint_share", "net"]})
    r = sm.OLS(g.ats_res.values, sm.add_constant(X.values), missing="drop").fit(cov_type="HC1")
    res["joint_paint"] = dict(zip(["const"] + list(X.columns), zip(r.params.round(4), r.pvalues.round(5))))
    print("joint ATS model (pts per SD, p):", res["joint_paint"])
    print("corr opp_paint_share vs opp_fg3_pct diffs", round(X.corr().loc["opp_paint_share", "opp_fg3_pct"], 3),
          " vs opp_fg3a_rate", round(X.corr().loc["opp_paint_share", "opp_fg3a_rate"], 3))
    return res


def walk_forward(g, F, names, thr=1.0):
    out = {}
    for nm in names:
        y, x = F[nm]
        gg = g.assign(_x=zs(x, g.season), _y=g[y])
        bets = []
        for se in sorted(gg.season.unique())[3:]:
            tr, te = gg[gg.season < se].dropna(subset=["_x", "_y"]), gg[gg.season == se].dropna(subset=["_x", "_y"])
            b = np.polyfit(tr._x, tr._y, 1)[0]
            pred = b * te._x
            sel = te[pred.abs() >= thr]
            res = np.sign(pred[pred.abs() >= thr]) * sel._y
            bets += [(se, r) for r in res if r != 0]
        bb = pd.DataFrame(bets, columns=["season", "r"])
        if len(bb):
            w = (bb.r > 0).mean()
            h = 1.96 * np.sqrt(w * (1 - w) / len(bb))
            out[nm] = {"bets": len(bb), "win": w, "win_lo": w - h, "win_hi": w + h, "roi_-110": w * 1.9091 - 1,
                       "by_season": bb.groupby("season").r.apply(lambda r: round((r > 0).mean(), 3)).to_dict()}
            print("walk-forward", nm, out[nm])
    return out


# ----------------------------------------------------------------------------------------------- Q3 coaching / rest
def coach_map():
    c = pd.read_parquet(OUT / "coaches.parquet")
    # head coach as listed by stats.nba.com for that team-season (end-of-season roster: mid-season firings mislabel some games)
    hc = c[c.COACH_TYPE.str.strip().eq("Head Coach")]
    hc = hc.sort_values("SORT_SEQUENCE").groupby(["TEAM_ID", "season"]).head(1)
    return hc.set_index(["TEAM_ID", "season"]).COACH_NAME


def cmd_q3() -> dict:
    res = {}
    rows = []
    g = game_frame()
    cm = coach_map()
    g["h_coach"] = [cm.get((t, s)) for t, s in zip(g.h_TEAM_ID, g.season)]
    g["a_coach"] = [cm.get((t, s)) for t, s in zip(g.a_TEAM_ID, g.season)]
    print("coach coverage", g.h_coach.notna().mean().round(3))
    # team-perspective rows
    tp = pd.concat([pd.DataFrame({"season": g.season, "coach": g.h_coach, "team": g.h_team, "r": g.ats_res, "GAME_ID": g.GAME_ID}),
                    pd.DataFrame({"season": g.season, "coach": g.a_coach, "team": g.a_team, "r": -g.ats_res, "GAME_ID": g.GAME_ID})])
    # C1 coach ATS persistence: coach's ATS mean over all PRIOR seasons (>= 82 games) -> this game's ATS residual
    cs = tp.groupby(["coach", "season"]).r.agg(["sum", "count"]).reset_index().sort_values(["coach", "season"])
    cs["prior_mean"] = cs.groupby("coach")["sum"].transform(lambda x: x.shift(1).cumsum()) / cs.groupby("coach")["count"].transform(lambda x: x.shift(1).cumsum())
    cs["prior_n"] = cs.groupby("coach")["count"].transform(lambda x: x.shift(1).cumsum())
    cs.loc[cs.prior_n < 82, "prior_mean"] = np.nan
    tp = tp.merge(cs[["coach", "season", "prior_mean"]], on=["coach", "season"], how="left")
    hh = tp.iloc[:len(g)].reset_index(drop=True); aa = tp.iloc[len(g):].reset_index(drop=True)
    g = g.reset_index(drop=True)
    g["coach_prior_diff"] = hh.prior_mean.values - aa.prior_mean.values
    g["coach_prior_diff"] = zs(g.coach_prior_diff, g.season)
    rows.append(run_cell(g, "ats_res", "coach_prior_diff", "C1 coach prior-seasons ATS mean (H-A) -> ATS", "Q3"))
    # coach-season ATS split-half reliability (descriptive)
    tp["half"] = tp.groupby(["coach", "season"]).cumcount() % 2
    sh = tp.groupby(["coach", "season", "half"]).r.mean().unstack()
    res["coach_season_splithalf_r"] = float(sh.corr().iloc[0, 1])
    # C2 late-game: residual in close Q4 states vs baseline, coach prior-seasons mean -> current
    s = states()
    d = fit_baseline(s)
    late = d[(d.el == 2580) & (d.deficit <= 5) & (d.stype == "Regular Season")].copy()  # Q4 5:00, within 5
    late["tr_id"] = late.tr_team
    tg = pd.read_parquet(OUT / "teamgames.parquet")[["GAME_ID", "team", "TEAM_ID"]]
    late = late.merge(tg.rename(columns={"team": "tr_team", "TEAM_ID": "tr_tid"}), on=["GAME_ID", "tr_team"])
    late = late.merge(tg.rename(columns={"team": "ld_team", "TEAM_ID": "ld_tid"}), on=["GAME_ID", "ld_team"])
    late["tr_coach"] = [cm.get((t, s_)) for t, s_ in zip(late.tr_tid, late.season)]
    late["ld_coach"] = [cm.get((t, s_)) for t, s_ in zip(late.ld_tid, late.season)]
    late["res"] = late.y - late.p_base
    lp = pd.concat([pd.DataFrame({"coach": late.tr_coach, "season": late.season, "r": late.res}),
                    pd.DataFrame({"coach": late.ld_coach, "season": late.season, "r": -late.res})])
    ls = lp.groupby(["coach", "season"]).r.agg(["sum", "count"]).reset_index().sort_values(["coach", "season"])
    ls["pm"] = ls.groupby("coach")["sum"].transform(lambda x: x.shift(1).cumsum()) / ls.groupby("coach")["count"].transform(lambda x: x.shift(1).cumsum())
    ls["pn"] = ls.groupby("coach")["count"].transform(lambda x: x.shift(1).cumsum())
    ls.loc[ls.pn < 40, "pm"] = np.nan
    k = ls.set_index(["coach", "season"]).pm
    late["coach_late_diff"] = [k.get((a, s_), np.nan) - k.get((b, s_), np.nan) for a, b, s_ in zip(late.tr_coach, late.ld_coach, late.season)]
    late["coach_late_diff"] = zs(late.coach_late_diff, late.season)
    r = run_cell(late, "res", "coach_late_diff", "C2 coach prior close-game residual (trailer-leader) -> close-game outcome", "Q3", scale=100)
    rows.append(r)
    res["late_n"] = len(late)
    # rest / schedule vs ATS
    g["rest_diff"] = g.h_rest - g.a_rest
    g["h_b2b"] = (g.h_rest == 1).astype(float)
    g["a_b2b"] = (g.a_rest == 1).astype(float)
    g["t34_diff"] = g.h_three_in_four - g.a_three_in_four
    g["travel_diff"] = np.where(g.bubble, np.nan, g.h_travel_km - g.a_travel_km)
    g["road_streak_away"] = g.a_road_streak
    for c in ("rest_diff", "travel_diff", "road_streak_away"):
        g[c] = zs(g[c], g.season)  # per SD
    for col, nm in [("rest_diff", "R1 rest days diff (H-A, capped 5)"), ("h_b2b", "R2 home on back-to-back"),
                    ("a_b2b", "R3 away on back-to-back"), ("t34_diff", "R4 3-in-4 nights diff"),
                    ("travel_diff", "R5 travel km since last game diff"), ("road_streak_away", "R6 away team road-trip game #")]:
        rows.append(run_cell(g, "ats_res", col, nm, "Q3"))
    # timeouts (2021-22 .. 2024-25): after an opponent 8-0 run, timeout vs not -> net points next 4 game-minutes
    res["timeouts"] = timeout_analysis(rows, g, cm)
    t = test_table(rows)
    print(f"Q3 cells {len(t)}; BH survivors {int(t.bh.sum())}; +halves {int(t.survivor.sum())}")
    print(t.round(4).to_string())
    res["tests"] = t.to_dict("records")
    return res


def timeout_analysis(rows, g, cm):
    to = pd.read_parquet(OUT / "timeouts.parquet")
    to = to[to.type == "Full Timeout"]
    p = pd.read_parquet(ROOT / "data" / "nba_big_lead_plays.parquet", columns=["eid", "seq", "period", "clock_s", "hs", "as"])
    p = p[p.eid.isin(set(to.eid)) & (p.period <= 4)].dropna(subset=["clock_s"])
    p["el"] = (p.period - 1) * 720 + (720 - p.clock_s)
    p = p.sort_values(["eid", "el", "seq"])
    to["el"] = (to.period - 1) * 720 + (720 - to.clock_s)
    tod = {e: x for e, x in to[to.period <= 4].groupby("eid")}
    ev = []
    for eid, x in p.groupby("eid", sort=False):
        hs, as_, el = x.hs.values, x["as"].values, x.el.values
        dh, da = np.diff(hs, prepend=0), np.diff(as_, prepend=0)
        run_side, run = None, 0
        T = tod.get(eid)
        for i in range(len(x)):
            for side, dv, ov in (("home", dh[i], da[i]), ("away", da[i], dh[i])):
                if dv > 0 and ov == 0:
                    if run_side == side:
                        run += dv
                    else:
                        run_side, run = side, dv
                    if run >= 8 and run - dv < 8 and el[i] < 2700:  # opponent of `side` just went down an 8-0 run
                        victim = "away" if side == "home" else "home"
                        # did the victim call a timeout before its next score (or within 60 s)?
                        # decision window W = 30 game-seconds after the run reaches 8. Group A: victim calls a
                        # timeout inside W. Group B: no victim timeout inside W AND the score did not change inside W
                        # (so "no timeout" is not just "answered immediately"). Outcome: victim's net points from
                        # el+W to el+W+240. Observational; descriptive only (not a market test).
                        W = 30
                        k0 = np.searchsorted(el, el[i] + W, side="right") - 1
                        called = T is not None and ((T.side == victim) & (T.el >= el[i]) & (T.el <= el[i] + W)).any()
                        quiet = (hs[k0] == hs[i]) and (as_[k0] == as_[i])
                        if not (called or quiet):
                            continue
                        j = np.searchsorted(el, el[i] + W + 240, side="right") - 1
                        vm0 = (hs[k0] - as_[k0]) * (1 if victim == "home" else -1)
                        vm1 = (hs[j] - as_[j]) * (1 if victim == "home" else -1)
                        ev.append((eid, victim, el[i], vm0, called, vm1 - vm0))
    e = pd.DataFrame(ev, columns=["eid", "victim", "el", "margin0", "called", "net_next4"])
    e["q"] = (e.el // 720 + 1).astype(int)
    e["mb"] = pd.cut(e.margin0, [-99, -10, -3, 3, 10, 99])
    print(f"\nTimeouts: {len(e)} opponent 8-0 runs in {e.eid.nunique()} games; victim called TO {e.called.mean():.3f}")
    print(e.groupby("called").net_next4.agg(["mean", "count"]).round(3))
    # stratified difference (quarter x margin bucket), game-cluster SE via OLS with fixed effects
    import statsmodels.formula.api as smf
    e["c"] = e.called.astype(float)
    m = smf.ols("net_next4 ~ c + C(q) + C(mb)", data=e).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(e.eid)[0]})
    out = {"n_runs": len(e), "to_rate": e.called.mean(), "effect_pts": float(m.params["c"]), "se": float(m.bse["c"]), "p": float(m.pvalues["c"])}
    print("TO effect on next-4-min net points (adj quarter + margin)", out)
    # coach timeout propensity -> ATS (walk-forward: prior-season propensity of the same coach)
    gm = pd.read_parquet(OUT / "games.parquet")[["eid", "season", "h_TEAM_ID", "a_TEAM_ID"]].dropna(subset=["eid"])
    e = e.merge(gm, on="eid")
    e["tid"] = np.where(e.victim == "home", e.h_TEAM_ID, e.a_TEAM_ID)
    e["coach"] = [cm.get((t, s)) for t, s in zip(e.tid, e.season)]
    cp = e.groupby(["coach", "season"]).called.agg(["mean", "count"])
    out["coach_propensity_spread"] = {"sd_between_coach_seasons": float(cp["mean"].std()), "n_coach_seasons": len(cp)}
    cpk = cp["mean"].reset_index()
    cpk["prev"] = cpk.sort_values("season").groupby("coach")["mean"].shift(1)
    out["propensity_year_to_year_r"] = float(cpk[["mean", "prev"]].dropna().corr().iloc[0, 1])
    print("coach timeout propensity:", out["coach_propensity_spread"], "year-to-year r", round(out["propensity_year_to_year_r"], 3))
    pk = cpk.set_index(["coach", "season"]).prev
    gg = g[g.season >= 2022].copy()
    gg["to_prop_diff"] = [pk.get((cm.get((h, s)), s), np.nan) - pk.get((cm.get((a, s)), s), np.nan)
                          for h, a, s in zip(gg.h_TEAM_ID, gg.a_TEAM_ID, gg.season)]
    gg["to_prop_diff"] = zs(gg.to_prop_diff, gg.season)
    r = run_cell(gg, "ats_res", "to_prop_diff", "T1 coach prior-season timeout-after-run propensity (H-A) -> ATS", "Q3")
    r["b_early"], r["b_late"] = (r["b"], r["b"])  # only 2022-24: halves not available -> sign check vs itself (flag)
    rows.append(r)
    out["prop_test_note"] = "T1 uses 2022-23..2024-25 only (prior season needed); halves check not possible"
    return out


# ----------------------------------------------------------------------------------------------- Q4 refs (ATS only)
def cmd_q4() -> dict:
    rf = pd.read_parquet(ROOT / "data" / "espn" / "nba" / "refs_dev.parquet")
    rf = rf[rf.season <= 2025]  # ESPN label 2025 = 2024-25 ; 2026 never loaded
    rf["date"] = pd.to_datetime(rf.start_utc).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    rf["home_n"] = rf.home.replace(ESPN2NBA)
    g = pd.read_parquet(OUT / "games.parquet")
    g = g[g.spread.notna()]
    g = g.merge(rf[["event_id", "home_n", "date", "ref1", "ref2", "ref3", "home_fta", "away_fta", "home_pf", "away_pf"]].rename(
        columns={"home_n": "h_team"}), on=["date", "h_team"], how="inner")
    g = g.sort_values("date").reset_index(drop=True)
    print("Q4 games with refs", len(g), g.season.value_counts().sort_index().to_dict())
    g["fta_tot"] = g.home_fta + g.away_fta
    g["fta_dev"] = g.fta_tot - g.groupby("season").fta_tot.transform("mean")
    g["home_fta_edge"] = (g.home_fta - g.away_fta)
    # walk-forward ref ratings: mean over the ref's PRIOR games (>= 30) of fta_dev, home ATS residual, home FTA edge
    long = pd.concat([g[["date", "GAME_ID", r, "fta_dev", "ats_res", "home_fta_edge"]].rename(columns={r: "ref"}) for r in ("ref1", "ref2", "ref3")])
    long = long.dropna(subset=["ref"]).sort_values("date")
    gr = long.groupby("ref")
    for c in ("fta_dev", "ats_res", "home_fta_edge"):
        cum = gr[c].transform(lambda x: x.shift(1).cumsum())
        n = gr.cumcount()
        long["r_" + c] = (cum / n).where(n >= 30)
    crew = long.groupby("GAME_ID")[["r_fta_dev", "r_ats_res", "r_home_fta_edge"]].mean()
    g = g.merge(crew, left_on="GAME_ID", right_index=True, how="left")
    g = g[g.stype == "Regular Season"]
    ok = (g.h_n_prior >= 10) & (g.a_n_prior >= 10)
    g = g[ok].copy()
    zc = zs(g.r_fta_dev, g.season)
    g["r_ats_res"] = zs(g.r_ats_res, g.season)
    g["r_home_fta_edge"] = zs(g.r_home_fta_edge, g.season)
    g["X_fta_x_ftr"] = zc * zs(g.h_s_ft_rate - g.a_s_ft_rate, g.season)
    g["X_fta_x_foulers"] = zc * zs(g.h_s_opp_ft_rate - g.a_s_opp_ft_rate, g.season)
    g["X_fta_x_star"] = zc * zs(g.h_s_star_share - g.a_s_star_share, g.season)
    g["X_fta_x_fav"] = zc * np.sign(g.spread)
    rows = []
    for col, nm in [("r_ats_res", "F1 crew prior home-ATS rating -> ATS"),
                    ("r_home_fta_edge", "F2 crew prior home FTA edge -> ATS"),
                    ("X_fta_x_ftr", "F3 whistle-heavy crew x FT-rate drawers (H-A) -> ATS"),
                    ("X_fta_x_foulers", "F4 whistle-heavy crew x foul-prone defence (H-A) -> ATS"),
                    ("X_fta_x_star", "F5 whistle-heavy crew x star-usage (H-A) -> ATS"),
                    ("X_fta_x_fav", "F6 whistle-heavy crew x favourite -> ATS")]:
        r = run_cell(g, "ats_res", col, nm, "Q4")
        # halves inside 2021-22..2024-25: first two vs last two seasons
        e = g.season <= 2022
        r["b_early"] = ols1(g.ats_res[e].values, g[col][e].values)[0]
        r["b_late"] = ols1(g.ats_res[~e].values, g[col][~e].values)[0]
        rows.append(r)
    t = test_table(rows)
    print(f"Q4 cells {len(t)}; BH survivors {int(t.bh.sum())}; +halves {int(t.survivor.sum())}")
    print(t.round(4).to_string())
    return {"n_games": len(g), "tests": t.to_dict("records")}


# ----------------------------------------------------------------------------------------------- Q5 other
def cmd_q5() -> dict:
    g = game_frame()
    g["alt_home"] = np.where(g.bubble, np.nan, g.h_team.isin(["DEN", "UTA"]).astype(float))
    g["prev_margin_diff"] = g.h_prev_margin - g.a_prev_margin
    g["prev_ats_diff"] = g.h_prev_ats_res - g.a_prev_ats_res
    g["h_bounce"] = (g.h_prev_margin <= -20).astype(float) - (g.a_prev_margin <= -20).astype(float)
    g["prev_margin_diff"] = zs(g.prev_margin_diff, g.season)
    g["prev_ats_diff"] = zs(g.prev_ats_diff, g.season)
    g["late_season"] = np.where(g.date.dt.month.isin([3, 4]), np.sign(g.spread), 0.0)
    g["home"] = 1.0
    rows = []
    for col, nm in [("alt_home", "O1 altitude home (DEN/UTA)"), ("prev_margin_diff", "O2 previous-game margin diff (H-A)"),
                    ("prev_ats_diff", "O3 previous-game ATS residual diff (overreaction)"),
                    ("h_bounce", "O4 bounce-back after 20+ loss (H-A)"),
                    ("late_season", "O6 favourite in March/April")]:
        rows.append(run_cell(g, "ats_res", col, nm, "Q5"))
    # O7 home ATS overall (intercept): is the home side mispriced on average?
    hm = g.ats_res.mean(); hs = g.ats_res.std() / np.sqrt(len(g))
    rows.append({"family": "Q5", "test": "O7 home ATS mean residual", "b": hm, "lo": hm - 1.96 * hs, "hi": hm + 1.96 * hs,
                 "p": float(2 * (1 - __import__("scipy").stats.norm.cdf(abs(hm / hs)))), "n": len(g),
                 "b_early": g[g.season <= 2019].ats_res.mean(), "b_late": g[g.season > 2019].ats_res.mean(),
                 "seasons_same_sign": f"{int((g.groupby('season').ats_res.mean() * np.sign(hm) > 0).sum())}/{g.season.nunique()}"})
    # O8 playoffs: team that lost the previous game of the series ("zig-zag") vs ATS
    po = pd.read_parquet(OUT / "games.parquet")
    po = po[(po.stype == "Playoffs") & po.spread.notna()].sort_values("date").copy()
    po["pair"] = [tuple(sorted(x)) for x in zip(po.h_team, po.a_team)]
    po["prev_home_margin"] = np.nan
    last = {}
    for i, r in po.iterrows():
        k = (r.season, r.pair)
        if k in last:
            pw, pm = last[k]
            po.at[i, "prev_home_margin"] = pm if pw == r.h_team else -pm  # current home team's margin last game
        last[k] = (r.h_team, r.margin)
    po["zig"] = -np.sign(po.prev_home_margin)  # +1 if current home team lost last game
    rows.append(run_cell(po.dropna(subset=["zig"]), "ats_res", "zig", "O8 playoff zig-zag (home lost last series game)", "Q5"))
    t = test_table(rows)
    print(f"Q5 cells {len(t)}; BH survivors {int(t.bh.sum())}; +halves {int(t.survivor.sum())}")
    print(t.round(4).to_string())
    return {"tests": t.to_dict("records")}


# ----------------------------------------------------------------------------------------------- frozen candidate rules
RULE_FEATURES = {
    "ATS_PAINT": ("ats_res", ["ATS diff opp_paint_share"]),
    "TOT_3PT": ("tot_res", ["TOT sum fg3_pct", "TOT sum opp_fg3_pct", "TOT |spread| (blowout risk)"]),
}


def cmd_rules() -> dict:
    """Freeze coefficients on ALL dev seasons (2015-16..2024-25) for the forward-paper rules, and show the
    walk-forward (fit on seasons < s, score s) record of the same combined model. POST-HOC: the combination
    was chosen after the Q2 survivors were seen; it is a candidate for forward paper, not a pass."""
    import statsmodels.api as sm
    g = game_frame()
    F = q2_features(g)
    out = {}
    for rule, (y, feats) in RULE_FEATURES.items():
        X = pd.DataFrame({f: zs(F[f][1], g.season) for f in feats})
        ok = X.notna().all(axis=1) & g[y].notna()
        full = sm.OLS(g[y][ok].values, sm.add_constant(X[ok].values)).fit()
        # z-scoring uses season mean/sd; forward use needs the dev-pooled raw mean/sd of each feature
        raw = pd.DataFrame({f: F[f][1] for f in feats})
        out[rule] = {"target": y, "coef_per_sd": dict(zip(["const"] + feats, np.round(full.params, 4))),
                     "raw_mean": raw[ok].mean().round(5).to_dict(), "raw_sd": raw[ok].std().round(5).to_dict()}
        for thr in (1.0, 1.5, 2.0):
            rec = []
            for se in sorted(g.season.unique())[3:]:
                tr, te = ok & (g.season < se), ok & (g.season == se)
                m = sm.OLS(g[y][tr].values, sm.add_constant(X[tr].values)).fit()
                pred = m.predict(sm.add_constant(X[te].values, has_constant="add")) - m.params[0]
                r = np.sign(pred) * g[y][te].values
                sel = (np.abs(pred) >= thr) & (r != 0)
                rec += [(se, v) for v in r[sel]]
            bb = pd.DataFrame(rec, columns=["season", "r"])
            w = (bb.r > 0).mean() if len(bb) else np.nan
            h = 1.96 * np.sqrt(w * (1 - w) / max(len(bb), 1))
            out[rule][f"walk_forward_thr{thr}"] = {"bets": len(bb), "win": w, "win_lo": w - h, "win_hi": w + h,
                                                   "roi_-110": w * 1.9091 - 1,
                                                   "early_2018_20": (bb[bb.season <= 2020].r > 0).mean(),
                                                   "late_2021_24": (bb[bb.season > 2020].r > 0).mean()}
        print(rule, json.dumps(out[rule], indent=1, default=float))
    return out


def cmd_all() -> int:
    out = {}
    for q in ("q1", "q2", "q3", "q4", "q5", "rules"):
        print(f"\n================ {q}", flush=True)
        out[q] = globals()["cmd_" + q]()
    allt = pd.concat([pd.DataFrame(out[q]["tests"]) for q in out if "tests" in out[q]], ignore_index=True)
    allt["bonferroni_all"] = allt.p < 0.05 / len(allt)
    allt["bh_all"] = bh(allt.p.values, 0.10)
    print(f"\nALL: {len(allt)} tests; Bonferroni (p < {0.05 / len(allt):.2e}) {int(allt.bonferroni_all.sum())}; "
          f"BH-all {int(allt.bh_all.sum())}; BH-all + halves {int((allt.bh_all & allt.halves_same_sign).sum())}")
    print(allt[allt.bh_all].round(4).to_string())
    out["all_tests"] = allt.to_dict("records")
    RES.parent.mkdir(exist_ok=True)
    RES.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    r = globals()["cmd_" + cmd]()
    sys.exit(r if isinstance(r, int) else 0)
