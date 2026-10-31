"""NFL kicker x forecast wind, run/pass matchups, player x stadium. Pre-registration: notes/nfl_kicker_wx_matchups.md.

    python scripts/nfl_kicker_wx.py mos        # fetch as-issued GFS MOS (IEM archive) per station-season, dev 2007-2019
    python scripts/nfl_kicker_wx.py gates      # B0 + C gates (pbp 2006-2019 only)
    python scripts/nfl_kicker_wx.py features   # A forecast-wind table + B matchup terms (no lines, no outcomes)
    python scripts/nfl_kicker_wx.py trials [--ledger]   # join SBR lines, score A1 (+B1/B2 if B0 passed)
    python scripts/nfl_kicker_wx.py kicker     # fit kicker make model (dev) -> json coefficients
    python scripts/nfl_kicker_wx.py today      # 2026-11-01 descriptive table

DEV only: seasons <= 2019. Lockbox 2020-2025 is never read except kick attempts for the live kicker-skill table.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MOS_DIR = DATA / "nfl_mos"
OUT = DATA / "lab" / "nfl_kicker_wx"
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "scripts"))

DEV = list(range(2006, 2020))
LINE_SEASONS = list(range(2007, 2020))
STATIONS = {  # stadium_id -> ASOS stations with GFS MOS, nearest first
    "NYC00": ["KTEB", "KEWR"], "NYC01": ["KTEB", "KEWR"], "GNB00": ["KGRB"], "BAL00": ["KBWI"],
    "KAN00": ["KMCI"], "CAR00": ["KCLT"], "NAS00": ["KBNA"], "TAM00": ["KTPA"], "WAS00": ["KDCA", "KIAD"],
    "MIA00": ["KOPF", "KMIA"], "OAK00": ["KOAK"], "CHI98": ["KMDW", "KORD"], "CLE00": ["KBKL", "KCLE"],
    "BOS00": ["KOWD", "KBOS"], "JAX00": ["KJAX"], "PIT00": ["KAGC", "KPIT"], "CIN00": ["KLUK", "KCVG"],
    "BUF00": ["KBUF"], "DEN00": ["KBKF", "KDEN"], "SEA00": ["KBFI", "KSEA"], "SDG00": ["KSAN"],
    "PHI00": ["KPHL"], "SFO00": ["KSFO"], "DAL99": ["KDAL", "KDFW"], "SFO01": ["KSJC"],
    "LAX99": ["KLAX"], "LAX97": ["KTOA", "KLAX"], "MIN98": ["KMSP"], "PHO99": ["KPHX"],
}
ALT_TEAM_STADIUM = {"DEN00"}


def _get(url, tries=4):
    import urllib.request
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read().decode()
        except Exception as e:  # noqa: BLE001
            if getattr(e, "code", None) == 404:
                return '{"data": []}'
            time.sleep(10 * (i + 1))
            err = e
    raise err


def cmd_mos():
    MOS_DIR.mkdir(parents=True, exist_ok=True)
    for sid, sts in STATIONS.items():
        for y in LINE_SEASONS:
            for st in sts:
                f = MOS_DIR / f"{st}_{y}.csv"
                if not f.exists():
                    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?station=" + st +
                           f"&model=GFS&sts={y}-09-01T00:00Z&ets={y + 1}-02-20T00:00Z&format=csv")
                    txt = _get(url)
                    f.write_text(txt)
                    print(sid, st, y, txt.count("\n"), flush=True)
                    time.sleep(3)
                if f.read_text().count("\n") > 5000:   # a full season; skip fallbacks
                    break


def kickoff_utc(s: pd.DataFrame) -> pd.Series:
    loc = pd.to_datetime(s["gameday"].astype(str) + " " + s["gametime"].astype(str), errors="coerce")
    return loc.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC").dt.tz_localize(None)


def load_mos(st: str, y: int) -> pd.DataFrame:
    f = MOS_DIR / f"{st}_{y}.csv"
    if not f.exists() or f.stat().st_size < 200:
        return pd.DataFrame()
    m = pd.read_csv(f, usecols=["runtime", "ftime", "tmp", "wsp", "p06"])
    m["runtime"] = pd.to_datetime(m["runtime"])
    m["ftime"] = pd.to_datetime(m["ftime"])
    return m.dropna(subset=["wsp"])


def load_mos_day(st: str, ko: pd.Timestamp) -> pd.DataFrame:
    """Runtimes from kickoff-54h to kickoff-6h for one game (small IEM request, cached per station x date)."""
    if pd.isna(ko):
        return pd.DataFrame()
    d = MOS_DIR / "days"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{st}_{ko:%Y%m%d%H%M}.json"
    if not f.exists():
        # latest 6-hourly cycle with runtime <= kickoff - 6 h; step back one cycle at a time if that run is missing
        rt = (ko - pd.Timedelta(hours=6)).floor("6h")
        recs = []
        for _ in range(4):
            url = ("https://mesonet.agron.iastate.edu/api/1/mos.json?station=" + st +
                   f"&model=GFS&runtime={rt:%Y-%m-%dT%H:%M}Z")
            recs = json.loads(_get(url)).get("data", [])
            if recs:
                break
            rt -= pd.Timedelta(hours=6)
        f.write_text(json.dumps(recs))
    recs = json.loads(f.read_text())
    if not recs:
        return pd.DataFrame()
    m = pd.DataFrame(recs)[["runtime", "ftime", "tmp", "wsp", "p06"]]
    m["runtime"] = pd.to_datetime(m["runtime"]).dt.tz_localize(None) if pd.to_datetime(m["runtime"]).dt.tz is not None \
        else pd.to_datetime(m["runtime"])
    m["ftime"] = pd.to_datetime(m["ftime"]).dt.tz_localize(None) if pd.to_datetime(m["ftime"]).dt.tz is not None \
        else pd.to_datetime(m["ftime"])
    return m.dropna(subset=["wsp"])
    m = pd.read_csv(f, usecols=["runtime", "ftime", "tmp", "wsp", "p06"])
    m["runtime"] = pd.to_datetime(m["runtime"])
    m["ftime"] = pd.to_datetime(m["ftime"])
    return m.dropna(subset=["wsp"])


def forecast_for(m: pd.DataFrame, ko: pd.Timestamp):
    if m.empty or pd.isna(ko):
        return None
    ok = m[m["runtime"] <= ko - pd.Timedelta(hours=6)]
    if ok.empty:
        return None
    rt = ok["runtime"].max()
    if ko - rt > pd.Timedelta(hours=48):
        return None
    r = ok[ok["runtime"] == rt]
    w = r[(r["ftime"] >= ko) & (r["ftime"] <= ko + pd.Timedelta(hours=3))]
    if w.empty:
        mid = ko + pd.Timedelta(minutes=90)
        d = (r["ftime"] - mid).abs()
        w = r[d == d.min()]
        if d.min() > pd.Timedelta(hours=3):
            return None
    return dict(fc_runtime=rt, fc_lead_h=(ko - rt) // pd.Timedelta(seconds=1) / 3600,
                fc_wind_mph=float(w["wsp"].mean() * 1.15078), fc_temp_f=float(w["tmp"].mean()))


def schedule() -> pd.DataFrame:
    s = pd.read_csv(DATA / "nfl_schedule.csv")
    s = s[s["season"].isin(DEV)].copy()   # DEV ONLY
    return s


def cmd_features_a():
    s = schedule()
    s = s[s["season"].isin(LINE_SEASONS) & (s["roof"] == "outdoors")].copy()
    s["kickoff_utc"] = kickoff_utc(s)
    rows, cache = [], {}
    for _, g in s.iterrows():
        rec = dict(game_id=g["game_id"], season=g["season"], week=g["week"], stadium_id=g["stadium_id"],
                   kickoff_utc=g["kickoff_utc"], obs_wind=g["wind"], obs_temp=g["temp"])
        for st in STATIONS.get(g["stadium_id"], []):
            k = (st, g["season"])
            if k not in cache:
                cache[k] = load_mos(*k)
            mm = cache[k] if not cache[k].empty else load_mos_day(st, g["kickoff_utc"])
            fc = forecast_for(mm, g["kickoff_utc"])
            if fc:
                rec.update(fc, station=st)
                break
        rows.append(rec)
    a = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    a.to_parquet(OUT / "a_forecast.parquet")
    has = a["fc_wind_mph"].notna()
    print(f"outdoor games {len(a)}, with forecast {has.sum()}, missing by stadium:",
          a[~has]["stadium_id"].value_counts().head(8).to_dict())
    print("lead hours:", a["fc_lead_h"].describe().round(1).to_dict())
    print("forecast wind >= 15 mph:", int((a["fc_wind_mph"] >= 15).sum()))
    c = a[has & a["obs_wind"].notna()]
    print(f"corr(forecast, schedule-observed wind) = {np.corrcoef(c.fc_wind_mph, c.obs_wind)[0, 1]:.3f} (n {len(c)})")


TEAM_MAP = {"SD": "LAC", "OAK": "LV", "STL": "LA", "LAR": "LA"}


def boot_r(x, y, n=4000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    xs, ys = x[idx], y[idx]
    xs = xs - xs.mean(1, keepdims=True)
    ys = ys - ys.mean(1, keepdims=True)
    rs = (xs * ys).sum(1) / np.sqrt((xs ** 2).sum(1) * (ys ** 2).sum(1))
    return float(np.corrcoef(x, y)[0, 1]), float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def load_pbp_dev(cols):
    out = []
    for y in DEV:   # DEV ONLY
        out.append(pd.read_parquet(DATA / "pbp" / f"play_by_play_{y}.parquet", columns=cols))
    p = pd.concat(out, ignore_index=True)
    for c in ("posteam", "defteam", "home_team", "away_team"):
        if c in p:
            p[c] = p[c].replace(TEAM_MAP)
    return p


def cmd_gates():
    res = {}
    # ---- B0: defence rush-vs-pass split persistence, weeks 1-8 vs 9+
    p = load_pbp_dev(["season", "week", "season_type", "posteam", "defteam", "rush", "pass", "epa", "qb_kneel", "qb_spike"])
    p = p[(p["season_type"] == "REG") & p["epa"].notna() & p["defteam"].notna()
          & (p["qb_kneel"] != 1) & (p["qb_spike"] != 1) & ((p["rush"] == 1) | (p["pass"] == 1))]
    p["half"] = np.where(p["week"] <= 8, 1, 2)
    p["kind"] = np.where(p["pass"] == 1, "pass", "rush")
    lg = p.groupby(["season", "half", "kind"])["epa"].mean().rename("lg")
    d = p.groupby(["season", "defteam", "half", "kind"])["epa"].agg(["mean", "size"]).join(lg, on=["season", "half", "kind"])
    d["dev"] = d["mean"] - d["lg"]
    w = d["dev"].unstack(["half", "kind"])
    S1 = w[(1, "rush")] - w[(1, "pass")]
    S2 = w[(2, "rush")] - w[(2, "pass")]
    ok = S1.notna() & S2.notna()
    rB = boot_r(S1[ok], S2[ok])
    rr = boot_r(w[(1, "rush")][ok], w[(2, "rush")][ok])
    rp = boot_r(w[(1, "pass")][ok], w[(2, "pass")][ok])
    o = p.groupby(["season", "posteam", "half"])["rush"].mean().unstack("half")
    ro = boot_r(o[1], o[2])
    res["B0"] = dict(n_team_seasons=int(ok.sum()), r_split=rB, r_def_rush=rr, r_def_pass=rp, r_off_rush_rate=ro,
                     sd_split_h1=float(S1[ok].std()), passed=bool(rB[0] >= 0.15 and rB[1] > 0))
    print(f"B0 defence split (rushEPA - passEPA allowed), H1 vs H2, n={ok.sum()}: r={rB[0]:.3f} [{rB[1]:.3f},{rB[2]:.3f}]"
          f"  | rush-D r={rr[0]:.3f} [{rr[1]:.3f},{rr[2]:.3f}] pass-D r={rp[0]:.3f} [{rp[1]:.3f},{rp[2]:.3f}]"
          f"  | offence rush rate r={ro[0]:.3f}  -> {'PASS' if res['B0']['passed'] else 'DEAD'}")
    # ---- C: player x stadium yards residual
    q = load_pbp_dev(["game_id", "season", "season_type", "posteam", "home_team", "stadium_id",
                      "rusher_player_id", "rushing_yards", "receiver_player_id", "receiving_yards",
                      "passer_player_id", "passing_yards"])
    q = q[q["season_type"] == "REG"]
    stad = q.groupby("game_id")[["stadium_id", "home_team"]].first()
    out = {}
    for kind, pid, yd, cnt_min in (("rush", "rusher_player_id", "rushing_yards", 5),
                                   ("rec", "receiver_player_id", "receiving_yards", 3),
                                   ("pass", "passer_player_id", "passing_yards", 15)):
        x = q[q[pid].notna()].copy()
        x[yd] = x[yd].fillna(0)
        g = x.groupby([pid, "game_id", "season", "posteam"])[yd].agg(["sum", "size"]).reset_index()
        g = g.rename(columns={"sum": "y", "size": "n"})
        g = g[g["n"] >= cnt_min].join(stad, on="game_id")
        tot = g.groupby([pid, "season"])["y"].agg(["sum", "size"]).rename(columns={"sum": "S", "size": "N"})
        g = g.join(tot, on=[pid, "season"])
        g = g[g["N"] >= 4]
        g["res"] = g["y"] - (g["S"] - g["y"]) / (g["N"] - 1)
        g["home"] = g["posteam"] == g["home_team"]
        g = g.sort_values("game_id")
        for tag, sub in (("all", g), ("away_only", g[~g["home"]])):
            sub = sub[sub.groupby([pid, "stadium_id"])["y"].transform("size") >= 4].copy()
            sub["k"] = sub.groupby([pid, "stadium_id"]).cumcount()
            h = sub.groupby([pid, "stadium_id", sub["k"] % 2])["res"].mean().unstack().dropna()
            r = boot_r(h[0], h[1]) if len(h) > 10 else (np.nan, np.nan, np.nan)
            out[f"{kind}_{tag}"] = dict(cells=int(len(h)), r=r)
            print(f"C {kind:4s} {tag:9s} cells={len(h):5d} r={r[0]:+.3f} [{r[1]:+.3f},{r[2]:+.3f}]")
    res["C"] = out
    res["C_passed"] = bool(any(v["r"][0] >= 0.15 for v in out.values()))
    print("C ->", "PASS" if res["C_passed"] else "DEAD")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gates.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ kicker make model
KSK, KDECAY = 40.0, 0.8
ALT_STADIUMS = {"DEN00", "MEX00"}
FG_COLS = ["game_id", "season", "week", "play_id", "kicker_player_id", "kicker_player_name", "field_goal_result",
           "kick_distance", "roof", "wind", "temp", "stadium_id", "play_type"]


def fg_attempts(seasons):
    out = []
    for y in seasons:
        p = pd.read_parquet(DATA / "pbp" / f"play_by_play_{y}.parquet", columns=FG_COLS)
        out.append(p[(p["play_type"] == "field_goal") & p["kick_distance"].notna() & p["kicker_player_id"].notna()])
    a = pd.concat(out, ignore_index=True)
    a["make"] = (a["field_goal_result"] == "made").astype(int)
    a["out"] = a["roof"].isin(["outdoors", "open"]).astype(int)
    a["wind"] = np.where(a["out"] == 1, a["wind"].fillna(a["wind"].median()), 0.0)
    a["temp"] = np.where(a["out"] == 1, a["temp"].fillna(60.0), 70.0)
    a["alt"] = a["stadium_id"].isin(ALT_STADIUMS).astype(int)
    return a.sort_values(["season", "week", "game_id", "play_id"]).reset_index(drop=True)


def base_X(a):
    d = a["kick_distance"].astype(float)
    return np.column_stack([np.ones(len(a)), (d - 40) / 10, ((d - 40) / 10) ** 2, a["out"] * a["wind"] / 10,
                            a["out"] * a["wind"] / 10 * (d - 40) / 10, a["out"] * (a["temp"] - 60) / 10, a["alt"],
                            a["alt"] * (d - 40) / 10])


BASE_NAMES = ["const", "dist10", "dist10_sq", "wind10_out", "wind10_x_dist10", "temp10_out", "alt", "alt_x_dist10"]


def kicker_skill(a, p_base):
    """skill before each attempt = decayed sum of (make - p_base) / (decayed n + KSK), prior attempts only."""
    res = a["make"].to_numpy() - p_base
    sk = np.zeros(len(a))
    st = {}
    for i, (kid, s) in enumerate(zip(a["kicker_player_id"].to_numpy(), a["season"].to_numpy())):
        S, N, ls = st.get(kid, (0.0, 0.0, s))
        if s != ls:
            S, N = S * KDECAY ** (s - ls), N * KDECAY ** (s - ls)
        sk[i] = S / (N + KSK)
        st[kid] = (S + res[i], N + 1, s)
    return sk, st


def fit_kicker():
    import statsmodels.api as sm
    a = fg_attempts(DEV)   # DEV ONLY for the coefficients
    Xb = base_X(a)
    mb = sm.Logit(a["make"].to_numpy(), Xb).fit(disp=0)
    pb = mb.predict(Xb)
    sk, _ = kicker_skill(a, pb)
    X2 = np.column_stack([Xb, sk * 10])
    m2 = sm.Logit(a["make"].to_numpy(), X2).fit(disp=0)
    return a, mb, m2


def cmd_kicker():
    a, mb, m2 = fit_kicker()
    names = BASE_NAMES + ["skill_x10"]
    tab = {n: (float(c), float(se)) for n, c, se in zip(names, m2.params, m2.bse)}
    print(f"kicker model, dev FG attempts n={len(a)} (2006-2019); coef (se):")
    for n, (c, se) in tab.items():
        print(f"  {n:16s} {c:+.3f} ({se:.3f})")
    # what 20 mph does at 50 yds outdoors, 50F, average kicker
    for w in (0, 10, 20):
        row = pd.DataFrame(dict(kick_distance=[50], out=[1], wind=[w], temp=[50.0], alt=[0]))
        x = np.column_stack([base_X(row), [0.0]])
        print(f"  50 yd, 50F, wind {w:2d} mph -> P(make) {m2.predict(x)[0]:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kicker_model.json").write_text(json.dumps(dict(n=len(a), coef=tab, base_coef=list(map(float, mb.params))), indent=1))


# ------------------------------------------------------------------ A1 trial
def wk_boot(v, clusters, n=4000, seed=0):
    df = pd.DataFrame({"v": v, "c": clusters})
    g = df.groupby("c")["v"].agg(["sum", "size"])
    s, k = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n, len(s)))
    means = s[idx].sum(1) / k[idx].sum(1)
    return float(np.mean(means <= 0)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cmd_trials():
    from nfl_simple_model import load_sbr, load_schedule
    ledger = "--ledger" in sys.argv
    a = pd.read_parquet(OUT / "a_forecast.parquet")
    sched = load_schedule()
    sched = sched[sched["season"].isin(LINE_SEASONS)]   # DEV ONLY
    sbr, _ = load_sbr(sched)
    m = a.merge(sbr, on="game_id", how="inner", validate="1:1").merge(
        sched[["game_id", "home_score", "away_score"]], on="game_id", how="left", validate="1:1")
    m["pts"] = m["home_score"] + m["away_score"]
    print(f"A universe: {len(a)} outdoor games, {a.fc_wind_mph.notna().sum()} with forecast, {len(m)} joined to SBR, "
          f"{m.total_close.notna().sum()} with close total")
    b = m[(m["fc_wind_mph"] >= 15) & m["total_close"].notna()].copy()
    r = np.sign(b["total_close"] - b["pts"])
    b["u"] = np.where(r > 0, 100 / 110, np.where(r < 0, -1.0, 0.0))
    b["wk"] = b["season"].astype(str) + "_" + b["week"].astype(str)
    p, lo, hi = wk_boot(b["u"].to_numpy(), b["wk"].to_numpy())
    h1, h2 = b[b.season <= 2013]["u"].mean(), b[b.season >= 2014]["u"].mean()
    roi = b["u"].mean()
    bar = 0.05 / 3
    passed = bool(p < bar and h1 > 0 and h2 > 0)
    # descriptive: is the close already moved? (no bet logic)
    lin = m[m["total_close"].notna() & m["fc_wind_mph"].notna()]
    import statsmodels.api as sm
    X = sm.add_constant(lin[["fc_wind_mph"]].astype(float))
    fit_close = sm.OLS((lin["pts"] - lin["total_close"]).astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(lin["season"].astype(str) + lin["week"].astype(str))[0]})
    fit_move = sm.OLS((lin["total_close"] - lin["total_open"]).astype(float), X, missing="drop").fit()
    res = dict(trial="A1", bets=int(len(b)), wins=int((b.u > 0).sum()), pushes=int((b.u == 0).sum()),
               units=float(b.u.sum()), roi=float(roi), ci=[lo, hi], p=p, halves=[float(h1), float(h2)],
               n_h=[int((b.season <= 2013).sum()), int((b.season >= 2014).sum())], bar=bar, passed=passed,
               mean_fc_wind=float(b.fc_wind_mph.mean()), mean_obs_wind=float(b.obs_wind.mean()),
               lead_h_median=float(b.fc_lead_h.median()),
               resid_vs_close_per_mph=[float(fit_close.params["fc_wind_mph"]), float(fit_close.bse["fc_wind_mph"])],
               open_to_close_move_per_mph=[float(fit_move.params["fc_wind_mph"]), float(fit_move.bse["fc_wind_mph"])],
               by_season={int(k): [int(v.size), float(v.mean())] for k, v in b.groupby("season")["u"]})
    print(json.dumps({k: v for k, v in res.items() if k != "by_season"}, default=float))
    print("by season (n, roi):", {k: (v[0], round(v[1], 3)) for k, v in res["by_season"].items()})
    se = b["u"].std() / np.sqrt(len(b))
    print(f"power: per-bet SD {b.u.std():.3f}, SE {se:.3f}; ROI needed for p<{bar:.4f} ~ {2.13 * se:+.3f}")
    (OUT / "trials.json").write_text(json.dumps(res, indent=1, default=float))
    b.to_csv(OUT / "bets_A1.csv", index=False)
    if ledger:
        import subprocess
        sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT).stdout.strip()
        rows = [dict(ts=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), strategy="kwm_A1_fc_wind15_unders_close",
                     sha=sha, dataset="nfl_kicker_wx_matchups",
                     hypothesis="pre-registered in notes/nfl_kicker_wx_matchups.md; total UNDER at SBR close when as-issued "
                                "GFS MOS wind (runtime <= kickoff-6h) >= 15 mph, outdoor games",
                     source="user ask 10-31", status="ok", split="DEV 2007-2019", trial_number=1, study_trials=3,
                     bonferroni_bar=bar, price="SBR close total, -110 assumed", bets=res["bets"], dates=int(b.wk.nunique()),
                     units=res["units"], roi=res["roi"], ci=res["ci"], p=res["p"], halves=res["halves"], passed=passed,
                     verdict="PASS" if passed else "FAIL")]
        g = json.loads((OUT / "gates.json").read_text())
        rows.append(dict(ts=rows[0]["ts"], strategy="kwm_B0_def_split_gate", sha=sha, dataset="nfl_kicker_wx_matchups",
                         hypothesis="gate: defence rushEPA-passEPA split persists wk1-8 -> wk9+ (r >= .15)",
                         source="user ask 10-31", status="ok", split="DEV 2006-2019", trial_number=None, study_trials=3,
                         n=g["B0"]["n_team_seasons"], r=g["B0"]["r_split"][0], ci=g["B0"]["r_split"][1:],
                         passed=g["B0"]["passed"], verdict="DEAD (gate) -> B1/B2 not run"))
        rows.append(dict(ts=rows[0]["ts"], strategy="kwm_C_player_stadium_gate", sha=sha, dataset="nfl_kicker_wx_matchups",
                         hypothesis="gate: player x stadium yards residual split-half r >= .15",
                         source="user ask 10-31", status="ok", split="DEV 2006-2019", trial_number=None, study_trials=3,
                         cells={k: v["cells"] for k, v in g["C"].items()}, r={k: v["r"] for k, v in g["C"].items()},
                         passed=g["C_passed"], verdict="DEAD (gate)"))
        with open(ROOT / "lab" / "ledger.jsonl", "a") as f:
            for r_ in rows:
                f.write(json.dumps(r_, default=float) + "\n")
        print("ledger: wrote", len(rows), "rows")


# ------------------------------------------------------------------ today (descriptive only)
TODAY_COORDS = {"Highmark Stadium": (42.774, -78.787), "Paycor Stadium": (39.095, -84.516),
                "MetLife Stadium": (40.814, -74.074), "Acrisure Stadium": (40.447, -80.016),
                "Raymond James Stadium": (27.976, -82.503), "EverBank Stadium": (30.324, -81.637),
                "Empower Field at Mile High": (39.744, -105.020), "Hard Rock Stadium": (25.958, -80.239),
                "Northwest Stadium": (38.908, -76.865), "SoFi Stadium": (33.953, -118.339)}
ESPN_ABBR = {"WSH": "WAS", "LAR": "LA"}


def cmd_today():
    date = sys.argv[2] if len(sys.argv) > 2 else "20261101"
    sb = json.loads(_get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={date}"))
    a, mb, m2 = fit_kicker()                     # coefficients: DEV fit only
    allk = fg_attempts(range(2006, 2027))        # skill state: all attempts through 2026 (descriptive, no lines)
    _, st = kicker_skill(allk, mb.predict(base_X(allk)))
    last = allk[allk["season"] == 2026].copy()
    k26 = pd.read_parquet(DATA / "pbp" / "play_by_play_2026.parquet",
                          columns=["posteam", "kicker_player_id", "kicker_player_name", "week", "play_type"])
    k26 = k26[k26["play_type"].isin(["field_goal", "extra_point"]) & k26["kicker_player_id"].notna()]
    k26 = k26.sort_values("week").groupby("posteam").tail(1).set_index("posteam")
    rows = []
    for e in sb["events"]:
        c = e["competitions"][0]
        v = c.get("venue", {})
        ko = pd.Timestamp(e["date"]).tz_convert("UTC").tz_localize(None)
        o = (c.get("odds") or [{}])[0]
        teams = {x["homeAway"]: x["team"]["abbreviation"] for x in c["competitors"]}
        outdoor = (not v.get("indoor", True)) and v.get("fullName") != "SoFi Stadium"
        rec = dict(game=e["shortName"], ko_utc=ko, venue=v.get("fullName"), outdoor=outdoor,
                   spread=o.get("details"), total=o.get("overUnder"), book=o.get("provider", {}).get("name"))
        if v.get("fullName") in TODAY_COORDS:
            lat, lon = TODAY_COORDS[v["fullName"]]
            w = json.loads(_get(
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,"
                "wind_speed_10m,wind_gusts_10m,precipitation,precipitation_probability&wind_speed_unit=mph&"
                f"temperature_unit=fahrenheit&precipitation_unit=inch&timezone=GMT&start_date={ko:%Y-%m-%d}&"
                f"end_date={ko + pd.Timedelta(days=1):%Y-%m-%d}"))
            h = pd.DataFrame(w["hourly"])
            h["time"] = pd.to_datetime(h["time"])
            h = h[(h["time"] >= ko.floor("h")) & (h["time"] <= ko + pd.Timedelta(hours=3))]
            rec.update(wind=h["wind_speed_10m"].mean(), gust=h["wind_gusts_10m"].max(), temp=h["temperature_2m"].mean(),
                       precip_in=h["precipitation"].sum(), pop=h["precipitation_probability"].max())
        wind = rec.get("wind", 0.0) if outdoor else 0.0
        temp = rec.get("temp", 70.0) if outdoor else 70.0
        alt = int(v.get("fullName") == "Empower Field at Mile High")
        for side in ("away", "home"):
            t = ESPN_ABBR.get(teams[side], teams[side])
            if t not in k26.index:
                continue
            kid, kn = k26.loc[t, "kicker_player_id"], k26.loc[t, "kicker_player_name"]
            S, N, _ = st.get(kid, (0.0, 0.0, 2026))
            sk = S / (N + KSK)
            for dist in (40, 50):
                row = pd.DataFrame(dict(kick_distance=[dist], out=[int(outdoor)], wind=[wind], temp=[temp], alt=[alt]))
                calm = row.assign(wind=0.0, temp=70.0 if not outdoor else 60.0)
                rec[f"{side}_k"] = f"{t} {kn}"
                rec[f"{side}_p{dist}"] = float(m2.predict(np.column_stack([base_X(row), [sk * 10]]))[0])
                rec[f"{side}_p{dist}_calm"] = float(m2.predict(np.column_stack([base_X(calm), [sk * 10]]))[0])
            rec[f"{side}_skill"] = sk
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"today_{date}.csv", index=False)
    pd.set_option("display.width", 250)
    show = df[["game", "ko_utc", "outdoor", "spread", "total", "wind", "gust", "temp", "pop", "away_k", "away_p40", "away_p50",
               "home_k", "home_p40", "home_p50"]]
    print(show.round(2).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"mos": cmd_mos, "features_a": cmd_features_a, "gates": cmd_gates, "kicker": cmd_kicker,
     "trials": cmd_trials, "today": cmd_today}[cmd]()
