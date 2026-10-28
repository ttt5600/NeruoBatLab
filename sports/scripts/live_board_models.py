"""In-game win-probability models for the unified live board (scripts/live_board.py). DEV DATA ONLY.

    python scripts/live_board_models.py build-nhl   # fetch NHL api-web pbp for dev games -> data/live_board/nhl_dev_states.parquet
    python scripts/live_board_models.py fit-nhl     # -> results/nhl_ingame_fit.json
    python scripts/live_board_models.py fit-nfl     # -> results/nfl_ingame_fit.json

NBA: reuses the frozen L37 probit (results/nba_midgame_fit.json, scripts/nba_midgame_scan.py) -- not refit here.

NFL (nflfastR vegas_wp-style logistic, possession-team perspective). DEV = 2006-2019 only (NFL lockbox 2020-2025 per
  scripts/nfl_comebacks.py + data/frozen MANIFEST; the 2024 season suggested for calibration is IN the lockbox, so it is
  not read). Fit seasons 2006-2016, calibration/holdout 2017-2019, then the frozen coefficients are refit on 2006-2019.
  Features (posteam view, e = elapsed share = 1 - game_seconds_remaining/3600, as nflfastR):
    score_differential, score_differential / exp(-4 e) ["diff_time_ratio"], spread_time = posteam_spread * exp(-4 e),
    posteam_spread, yardline_100, down 1-4 dummies (1 = base), log(ydstogo), half_seconds_remaining/1800, e,
    posteam / defteam timeouts, receive_2h_ko (posteam gets the 2nd-half kickoff, 1st half only), home (posteam is home).
  posteam_spread = posteam expected margin from nflverse spread_line (spread_line = home expected margin).
  Rows: regular + post, qtr <= 4, down not null (live plays), result != 0 (ties dropped). Outcome: posteam won.

NHL (logistic, home perspective). DEV seasons by SBR/NHL-API season label (2017 = 2016-17, ..., 2022 = 2021-22). NHL
  sealed = 2026 (2025-26) per MANIFEST; never read. States: NHL api-web /gamecenter/{id}/play-by-play, one state every
  30 s of regulation (0..3600 elapsed): goal diff, man advantage from situationCode (home skaters - away skaters),
  empty nets. Pregame = de-vigged SBR closing moneyline (data/nhl_luck/games.pkl). Fit 2017-2019, calibrate 2022.
  Features: logit(p_pre), gd_eff, gd_eff/sqrt(tau), logit(p_pre)*tau, tau, home / away empty-net indicators;
  gd_eff = gd + 0.10 * man advantage (fixed from the dev next-2-minute goal rate).
  tau = regulation time left / 3600. OT/SO games count by the final winner.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
NFL_FIT = RES / "nfl_ingame_fit.json"
NHL_FIT = RES / "nhl_ingame_fit.json"
NHL_STATES = DATA / "live_board" / "nhl_dev_states.parquet"
NFL_STATES = DATA / "live_board" / "nfl_dev_states.parquet"
NHL_HIST = DATA / "live_board" / "nhl_dev_hist.parquet"  # states + pregame prob + outcome (shipped to the deploy)
NHL_SEASONS_FIT, NHL_SEASON_CAL = [2017, 2018, 2019], 2022
NFL_FIT_SEASONS, NFL_CAL_SEASONS = list(range(2006, 2017)), [2017, 2018, 2019]
NFL_LOCKBOX = set(range(2020, 2027))
NHL_SEALED = {2026}

NFL_FEATS = ["score_differential", "diff_time_ratio", "spread_time", "posteam_spread", "yardline_100", "down2",
             "down3", "down4", "log_ydstogo", "half_frac", "elapsed", "pos_to", "def_to", "receive_2h_ko", "home",
             "sd_rt", "lead1_rt", "trail1_rt", "tie_rt", "trail1_yl_rt", "tie_yl_rt", "to_diff_rt"]
NHL_FEATS = ["lp", "gd", "gd_rt", "lp_tau", "tau", "en_home", "en_away"]  # gd includes PP_GOALS * pp
PP_GOALS = 0.10


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(float(c - h), 4), round(float(c + h), 4))


def calib_table(p, y, gid):
    x = pd.DataFrame({"p": p, "y": y, "gid": gid})
    x["bin"] = pd.cut(x.p, [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1], include_lowest=True)
    out = []
    for b, g in x.groupby("bin", observed=True):
        g1 = g.drop_duplicates("gid")  # one state per game per bin for the CI
        out.append({"bin": str(b), "states": len(g), "games": len(g1), "model": round(g.p.mean(), 4),
                    "realised": round(g.y.mean(), 4), "ci95_1pg": wilson(int(g1.y.sum()), len(g1))})
    return out


def scores(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return {"n": int(len(y)), "brier": round(float(np.mean((p - y) ** 2)), 5),
            "logloss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5),
            "mean_model": round(float(np.mean(p)), 4), "mean_realised": round(float(np.mean(y)), 4)}


def fit_logit(X, y):
    import statsmodels.api as sm
    return sm.Logit(np.asarray(y, dtype=float), sm.add_constant(np.asarray(X, dtype=float), has_constant="add")).fit(
        disp=0, maxiter=200)


# ============================================================================================ NFL
def nfl_feats(df: pd.DataFrame) -> pd.DataFrame:
    """df needs: score_differential, game_seconds_remaining, half_seconds_remaining, posteam_spread, yardline_100,
    down, ydstogo, pos_to, def_to, receive_2h_ko, home, qtr (all posteam perspective)."""
    e = 1 - df.game_seconds_remaining.astype(float) / 3600
    x = pd.DataFrame(index=df.index)
    x["score_differential"] = df.score_differential.astype(float)
    x["diff_time_ratio"] = x.score_differential / np.exp(-4 * e)
    x["spread_time"] = df.posteam_spread.astype(float) * np.exp(-4 * e)
    x["posteam_spread"] = df.posteam_spread.astype(float)
    x["yardline_100"] = df.yardline_100.astype(float)
    d = df.down.astype(float)
    for k in (2, 3, 4):
        x[f"down{k}"] = (d == k).astype(float)
    x["log_ydstogo"] = np.log(np.clip(df.ydstogo.astype(float), 1, 99))
    x["half_frac"] = df.half_seconds_remaining.astype(float) / 1800
    x["elapsed"] = e
    x["pos_to"] = df.pos_to.astype(float)
    x["def_to"] = df.def_to.astype(float)
    x["receive_2h_ko"] = np.where(df.qtr.astype(float) <= 2, df.receive_2h_ko.astype(float), 0.0)
    x["home"] = df.home.astype(float)
    # late-game terms (v2, added after v1 gave P = .18 to 'down 4, 0:30 left, opponent ball'): scale by 1/sqrt(time left)
    rt = 1 / np.sqrt(np.clip(df.game_seconds_remaining.astype(float), 5, 3600) / 3600)
    sd = x.score_differential
    x["sd_rt"] = np.clip(sd, -28, 28) * rt
    x["lead1_rt"] = ((sd > 0) & (sd <= 8)).astype(float) * rt
    x["trail1_rt"] = ((sd < 0) & (sd >= -8)).astype(float) * rt
    x["tie_rt"] = (sd == 0).astype(float) * rt
    x["trail1_yl_rt"] = x.trail1_rt * x.yardline_100 / 100
    x["tie_yl_rt"] = x.tie_rt * x.yardline_100 / 100
    x["to_diff_rt"] = (x.pos_to - x.def_to) * rt / 3
    return x[NFL_FEATS]


def nfl_predict_pos(fitd: dict, df: pd.DataFrame) -> np.ndarray:
    b = np.asarray(fitd["coef"])
    z = b[0] + nfl_feats(df).values @ b[1:]
    a, c = fitd.get("platt", [1.0, 0.0])
    return sigmoid(a * z + c)


def nfl_load(seasons):
    cols = ["game_id", "season", "home_team", "away_team", "posteam", "qtr", "down", "ydstogo", "yardline_100",
            "game_seconds_remaining", "half_seconds_remaining", "score_differential", "posteam_timeouts_remaining",
            "defteam_timeouts_remaining", "spread_line", "result", "vegas_wp", "play_type"]
    out = []
    for s in seasons:
        assert s not in NFL_LOCKBOX
        p = pd.read_parquet(DATA / "pbp" / f"play_by_play_{s}.parquet", columns=cols)
        # first kickoff of the game: its posteam is the receiver of the opening kick -> the OTHER team gets the 2nd half
        k = p[(p.play_type == "kickoff") & (p.qtr == 1) & p.posteam.notna()].drop_duplicates("game_id")
        p = p.merge(k[["game_id", "posteam"]].rename(columns={"posteam": "recv1"}), on="game_id", how="left")
        out.append(p)
    p = pd.concat(out, ignore_index=True)
    p = p[p.down.notna() & p.posteam.notna() & (p.qtr <= 4) & p.spread_line.notna() & p.result.notna()
          & (p.result != 0) & p.score_differential.notna() & p.yardline_100.notna()
          & p.game_seconds_remaining.notna() & p.half_seconds_remaining.notna()].copy()
    p["home"] = (p.posteam == p.home_team).astype(int)
    p["posteam_spread"] = np.where(p.home == 1, p.spread_line, -p.spread_line)
    p["pos_to"] = p.posteam_timeouts_remaining.fillna(3)
    p["def_to"] = p.defteam_timeouts_remaining.fillna(3)
    p["receive_2h_ko"] = (p.recv1.notna() & (p.recv1 != p.posteam)).astype(int)
    p["y"] = np.where(p.home == 1, p.result > 0, p.result < 0).astype(int)
    return p


def fit_nfl() -> int:
    tr, te = nfl_load(NFL_FIT_SEASONS), nfl_load(NFL_CAL_SEASONS)
    # Platt slope/intercept: base fit 2006-2013 -> linear predictor on 2014-2016 -> logistic on it (v1/v2 were
    # S-shaped under-confident on the holdout). Then base refit on 2006-2016 + those Platt params = FROZEN model,
    # scored once on 2017-2019 (never used for fitting). No refit on 2017-2019.
    a_tr, a_pl = tr[tr.season <= 2013], tr[tr.season >= 2014]
    mb = fit_logit(nfl_feats(a_tr), a_tr.y.values)
    zpl = np.column_stack([np.ones(len(a_pl)), nfl_feats(a_pl).values]) @ np.asarray(mb.params)
    pl = fit_logit(zpl.reshape(-1, 1), a_pl.y.values)
    platt = [float(pl.params[1]), float(pl.params[0])]
    m1 = fit_logit(nfl_feats(tr), tr.y.values)
    m = m1
    fitd_ = {"coef": [float(x) for x in m1.params], "platt": platt}
    p_raw = np.asarray(m1.predict(np.column_stack([np.ones(len(te)), nfl_feats(te).values])))
    pt = nfl_predict_pos(fitd_, te)
    vw = te.vegas_wp.astype(float).values
    ok = ~np.isnan(vw)
    home_p = np.where(te.home == 1, pt, 1 - pt)
    home_y = np.where(te.home == 1, te.y, 1 - te.y)
    allp = pd.concat([tr, te], ignore_index=True)
    by_q = []
    for q in (1, 2, 3, 4):
        mk = te.qtr.values == q
        by_q.append({"qtr": q, **scores(pt[mk], te.y.values[mk]),
                     "vegas_wp_brier": scores(vw[mk & ok], te.y.values[mk & ok])["brier"]})
    out = {"model": "NFL in-game WP (nflfastR vegas_wp-style logistic, posteam perspective)",
           "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "data": "nflverse pbp 2006-2019 (DEV); lockbox 2020-2025 never read", "features": NFL_FEATS,
           "coef": [float(x) for x in m.params], "platt": platt,
           "platt_note": "P = sigmoid(a * (coef . [1, x]) + c), [a, c] fit on 2014-16 from a 2006-13 base fit",
           "n_plays_fit": int(len(tr)), "n_games_fit": int(tr.game_id.nunique()),
           "calibration_holdout": {"fit": "2006-2016 (+ Platt 2014-16)", "test": "2017-2019 (out of sample, scored once)",
                                   "ours": scores(pt, te.y.values), "ours_without_platt": scores(p_raw, te.y.values),
                                   "nflfastR_vegas_wp_same_plays": scores(vw[ok], te.y.values[ok]),
                                   "ours_same_plays": scores(pt[ok], te.y.values[ok]),
                                   "by_quarter": by_q,
                                   "table_home_view": calib_table(home_p, home_y, te.game_id.values)}}
    # pregame: P(home win) = Phi(home expected margin / sigma), for the Kalshi-pregame-mid -> spread fallback
    import statsmodels.api as sm
    g1 = allp.drop_duplicates("game_id")
    hm = g1.spread_line.astype(float).values
    yh = (g1.result > 0).astype(int).values
    out["pregame_sigma"] = float(1 / sm.Probit(yh, hm.reshape(-1, 1)).fit(disp=0).params[0])
    NFL_FIT.parent.mkdir(exist_ok=True)
    NFL_FIT.write_text(json.dumps(out, indent=1))
    # compact dev states for the live board's "history n" neighbour count (home view)
    hv = pd.DataFrame({"game_id": allp.game_id.values, "qtr": allp.qtr.astype("int8").values,
                       "gsr": allp.game_seconds_remaining.astype("int16").values,
                       "sd_home": np.where(allp.home == 1, allp.score_differential, -allp.score_differential).astype("int16"),
                       "hm": allp.spread_line.astype("float32").values,
                       "y_home": np.where(allp.home == 1, allp.y, 1 - allp.y).astype("int8")})
    NFL_STATES.parent.mkdir(parents=True, exist_ok=True)
    hv.to_parquet(NFL_STATES, index=False)
    print(f"pregame sigma {out['pregame_sigma']:.2f}; dev states -> {NFL_STATES}")
    c = out["calibration_holdout"]
    print(f"NFL fit plays {out['n_plays_fit']} games {out['n_games_fit']}; platt {platt}")
    print("holdout 2017-19 ours", c["ours"], "\n  vegas_wp", c["nflfastR_vegas_wp_same_plays"], "\n  ours same", c["ours_same_plays"])
    print(pd.DataFrame(by_q).to_string(index=False))
    print(pd.DataFrame(c["table_home_view"]).to_string(index=False))
    return 0


# ============================================================================================ NHL
def nhl_feats(df: pd.DataFrame) -> pd.DataFrame:
    """df needs p_pre (home pregame win prob), gd (home - away), el (elapsed regulation s, 0..3600), pp (home skaters
    - away skaters), en_home / en_away (goalie pulled)."""
    tau = np.clip(1 - df.el.astype(float) / 3600, 1 / 120, 1)
    x = pd.DataFrame(index=df.index)
    x["lp"] = logit(df.p_pre)
    # v2: a man advantage enters as an effective goal-difference shift (dev: next-2-min goal diff +0.105 per man on
    # the PP, -0.09 short-handed); v1's free pp / pp/sqrt(tau) terms fit ~0 effect (+0.7 pts at a tied mid-game vs an
    # empirical ~+4 pts), so they were replaced. PP_GOALS is fixed, not fitted.
    x["gd"] = df.gd.astype(float) + PP_GOALS * df.pp.astype(float).clip(-2, 2)
    x["gd_rt"] = x.gd / np.sqrt(tau)
    x["lp_tau"] = x.lp * tau
    x["tau"] = tau
    x["en_home"] = df.en_home.astype(float)
    x["en_away"] = df.en_away.astype(float)
    return x[NHL_FEATS]


def nhl_predict_home(fitd: dict, df: pd.DataFrame) -> np.ndarray:
    b = np.asarray(fitd["coef"])
    return sigmoid(b[0] + nhl_feats(df).values @ b[1:])


def _mmss(s):
    m, x = s.split(":")
    return int(m) * 60 + int(x)


def nhl_states_from_pbp(d: dict, step: int = 30) -> list[dict]:
    """Regulation states every `step` s from an api-web play-by-play json. situationCode digits =
    [away goalie, away skaters, home skaters, home goalie]."""
    ev = []
    for p in d.get("plays", []):
        per = p["periodDescriptor"]["number"]
        if per > 3:
            break
        ev.append(((per - 1) * 1200 + _mmss(p["timeInPeriod"]), p.get("situationCode"), p["typeDescKey"],
                   p.get("details") or {}))
    hid = d["homeTeam"]["id"]
    rows, hg, ag, sit, i = [], 0, 0, "1551", 0
    for t in range(0, 3600, step):
        while i < len(ev) and ev[i][0] <= t:
            _, sc, typ, det = ev[i]
            if sc and len(sc) == 4 and sc.isdigit():
                sit = sc
            if typ == "goal":
                if det.get("homeScore") is not None:
                    hg, ag = det["homeScore"], det["awayScore"]
                elif det.get("eventOwnerTeamId") == hid:
                    hg += 1
                else:
                    ag += 1
            i += 1
        ag_, ask, hsk, hg_ = (int(c) for c in sit)
        rows.append({"el": t, "gd": hg - ag, "pp": hsk - ask, "en_home": int(hg_ == 0), "en_away": int(ag_ == 0)})
    return rows


def build_nhl() -> int:
    import requests
    g = pd.read_pickle(DATA / "nhl_luck" / "games.pkl")
    seasons = NHL_SEASONS_FIT + [NHL_SEASON_CAL]
    assert not set(seasons) & NHL_SEALED
    g = g[g.season.isin(seasons) & g.ml_close_h.notna() & g.ml_close_a.notna() & g.gameId.notna()]
    g = g[g.gameId.astype("int64").astype(str).str[4:6] == "02"]  # regular season
    have = pd.read_parquet(NHL_STATES) if NHL_STATES.exists() else pd.DataFrame(columns=["gameId"])
    todo = sorted(set(g.gameId.astype("int64")) - set(have.gameId.astype("int64")))
    print(f"{len(g)} dev games, {len(todo)} to fetch", flush=True)
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"  # the default python-requests UA gets HTTP 429 from api-web

    def one(gid):
        for k in range(5):
            try:
                r = s.get(f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play", timeout=20)
                r.raise_for_status()
                return [{"gameId": gid, **x} for x in nhl_states_from_pbp(r.json())]
            except Exception:  # noqa: BLE001
                time.sleep(5 * (k + 1))
        print("FAILED", gid, flush=True)
        return []

    parts = [have] if len(have) else []
    with ThreadPoolExecutor(4) as ex:
        for j, rows in enumerate(ex.map(one, todo)):
            if rows:
                parts.append(pd.DataFrame(rows))
            if j % 500 == 0:
                print(j, flush=True)
    st = pd.concat(parts, ignore_index=True)
    st = st.astype({"gameId": "int64", "el": "int16", "gd": "int8", "pp": "int8", "en_home": "int8", "en_away": "int8"})
    NHL_STATES.parent.mkdir(parents=True, exist_ok=True)
    st.to_parquet(NHL_STATES, index=False)
    print(f"states {len(st)} games {st.gameId.nunique()}")
    return 0


def am_prob(a):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 100 / (a + 100), -a / (-a + 100))


def fit_nhl() -> int:
    g = pd.read_pickle(DATA / "nhl_luck" / "games.pkl")
    ph, pa = am_prob(g.ml_close_h), am_prob(g.ml_close_a)
    g = g.assign(p_pre=ph / (ph + pa))[["gameId", "season", "p_pre", "home_win"]].dropna()
    g["gameId"] = g.gameId.astype("int64")
    st = pd.read_parquet(NHL_STATES).merge(g, on="gameId")
    assert not set(st.season.unique()) & NHL_SEALED
    st["y"] = st.home_win.astype(int)
    tr, te = st[st.season.isin(NHL_SEASONS_FIT)], st[st.season == NHL_SEASON_CAL]
    m1 = fit_logit(nhl_feats(tr), tr.y.values)
    pt = np.asarray(m1.predict(np.column_stack([np.ones(len(te)), nhl_feats(te).values])))
    m = fit_logit(nhl_feats(st), st.y.values)
    by_t = []
    for lo, hi in [(0, 1200), (1200, 2400), (2400, 3300), (3300, 3600)]:
        mk = (te.el.values >= lo) & (te.el.values < hi)
        by_t.append({"window_el": f"{lo}-{hi}", **scores(pt[mk], te.y.values[mk]),
                     "pregame_only_brier": scores(te.p_pre.values[mk], te.y.values[mk])["brier"]})
    pp_mk = te.pp.values != 0
    out = {"model": "NHL in-game WP (logistic, home perspective)",
           "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "data": "NHL api-web pbp, regular season 2016-17..2018-19 + 2021-22 (DEV); pregame = de-vigged SBR close ML; "
                   "2025-26 sealed never read", "features": NHL_FEATS,
           "coef": [float(x) for x in m.params], "coef_fit_2017_2019": [float(x) for x in m1.params],
           "n_states": int(len(st)), "n_games": int(st.gameId.nunique()),
           "calibration_holdout": {"fit": "2016-17..2018-19", "test": "2021-22 (out of sample)",
                                   "ours": scores(pt, te.y.values), "pregame_only": scores(te.p_pre.values, te.y.values),
                                   "power_play_states": scores(pt[pp_mk], te.y.values[pp_mk]),
                                   "by_time": by_t,
                                   "table": calib_table(pt, te.y.values, te.gameId.values)}}
    NHL_FIT.parent.mkdir(exist_ok=True)
    NHL_FIT.write_text(json.dumps(out, indent=1))
    st[["gameId", "el", "gd", "p_pre", "y"]].astype({"p_pre": "float32", "y": "int8"}).to_parquet(NHL_HIST, index=False)
    c = out["calibration_holdout"]
    print(f"NHL fit states {out['n_states']} games {out['n_games']}; coef "
          f"{dict(zip(['const'] + NHL_FEATS, np.round(np.asarray(m.params), 3)))}")
    print("holdout 2021-22 ours", c["ours"], "\n  pregame only", c["pregame_only"], "\n  PP states", c["power_play_states"])
    print(pd.DataFrame(by_t).to_string(index=False))
    print(pd.DataFrame(c["table"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build-nhl": build_nhl, "fit-nhl": fit_nhl, "fit-nfl": fit_nfl}[cmd]())
