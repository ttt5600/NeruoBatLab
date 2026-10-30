"""NBA style matchups: shot-zone profile x opponent zone defense (+ passing), point-in-time. notes/nba_style_matchups.md

    ../.venv_quant/bin/python scripts/nba_style_matchups.py parse        # ESPN cached summaries -> data/nba_pbp/
    ../.venv_quant/bin/python scripts/nba_style_matchups.py calib        # coverage + coordinate/3PT-line calibration
    ../.venv_quant/bin/python scripts/nba_style_matchups.py reliability  # split-half (odd/even games) per metric
    ../.venv_quant/bin/python scripts/nba_style_matchups.py build        # PIT profiles + matchup terms (no outcomes)
    ../.venv_quant/bin/python scripts/nba_style_matchups.py dist         # signal distributions only (no outcomes)
    ../.venv_quant/bin/python scripts/nba_style_matchups.py score [--ledger] [--sealed]   # the one scoring run

Data: ESPN site summary `plays` (cached under data/cache by sports.http; every 2020-21..2025-26 regular-season final
is already cached, so `parse` makes no network calls). ESPN season label = year the season ends.
DEV = 2022..2025 (2021-22..2024-25); history-only = 2021 (2020-21, for walk-forward thresholds and EB k);
SEALED = 2026 (2025-26), scored only with --sealed and only for trials that passed DEV.
No orders; reads no keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

ESPN_D = ROOT / "data" / "espn" / "nba"
PBP = ROOT / "data" / "nba_pbp"
OUT = ROOT / "results" / "nba_style_matchups"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nba_style_matchups"
API = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
HIST, DEV, SEALED = [2021], [2022, 2023, 2024, 2025], 2026
H1, H2 = [2022, 2023], [2024, 2025]
PROP_DEV = [2023, 2024, 2025]
ZONES = ["rim", "short", "mid", "long2", "c3", "ab3"]
ZVAL = {"rim": 2, "short": 2, "mid": 2, "long2": 2, "c3": 3, "ab3": 3}
Y0 = 1.0          # basket y in ESPN coordinate feet (calibrated in `calib`: x centre 25, rim at y ~ 1)
CORNER_Y = 9      # a three with coordinate y <= 9 is a corner three (straight part of the line, ~14 ft from baseline)
K_TRIALS = 4
BAR = 0.05 / K_TRIALS
NB = 10000
SEED = 2026


# ============================================================================================== parse
def _zone(three: bool, dist: float, y: float) -> str:
    if three:
        return "c3" if (y is not None and y <= CORNER_Y) else "ab3"
    if dist <= 3:
        return "rim"
    if dist <= 9:
        return "short"
    if dist <= 15:
        return "mid"
    return "long2"


def _parse_one(args):
    eid, season = args
    from sports.http import CACHE_DIR, _key
    f = CACHE_DIR / (_key(f"{API}/summary?event={eid}", None) + ".json")
    if not f.exists():
        return eid, None, None, "no_cache"
    j = json.loads(f.read_text())
    plays = j.get("plays") or []
    if not plays:
        return eid, None, None, "no_plays"
    shots, ev = [], []
    seen = set()
    for p in plays:
        pid = p.get("id")
        if pid in seen:
            continue
        seen.add(pid)
        typ = (p.get("type") or {}).get("text", "") or ""
        txt = p.get("text", "") or ""
        team = (p.get("team") or {}).get("id")
        parts = [((x.get("athlete") or {}).get("id")) for x in (p.get("participants") or [])]
        per = (p.get("period") or {}).get("number", 0)
        if p.get("shootingPlay") and "Free Throw" not in typ:
            c = p.get("coordinate") or {}
            x, y = c.get("x"), c.get("y")
            if x is None or y is None or x < -1000 or y < -1000:
                x = y = np.nan
            m = re.search(r"(\d+)-foot", txt)
            tl = txt.lower()
            three = "three point" in tl or (p.get("scoringPlay") and p.get("scoreValue") == 3)
            dft = float(m.group(1)) if m else np.nan
            dco = float(np.hypot(x - 25, y - Y0)) if not np.isnan(x) else np.nan
            dist = dft if m else dco
            made = bool(p.get("scoringPlay"))
            blocked = " blocks " in tl
            # shooter: for "X blocks Y's shot" ESPN lists the shooter first? verified in `calib` via team membership
            shooter = parts[0] if parts else None
            ast = parts[1] if (made and "assists)" in tl and len(parts) > 1) else None
            shots.append((eid, season, per, team, shooter, ast, made, bool(three), dft, dco, x, y, blocked, typ, txt[:120]))
        elif "Turnover" in typ or "turnover" in txt.lower():
            stl = None
            if "steals)" in txt.lower() and len(parts) > 1:
                stl = parts[1]
            ev.append((eid, season, per, team, parts[0] if parts else None, "tov", stl is not None, typ))
    return eid, shots, ev, "ok"


def cmd_parse(a) -> int:
    PBP.mkdir(parents=True, exist_ok=True)
    cov = {}
    for s in (a.seasons or SEASONS):
        g = pd.read_parquet(ESPN_D / f"games_{s}.parquet")
        g = g[(g.season_type == "regular-season") & (g.status == "STATUS_FINAL")]
        todo = [(str(e), s) for e in g.event_id]
        S, E, st = [], [], {}
        with ProcessPoolExecutor(max_workers=6) as ex:
            for eid, sh, ev, status in ex.map(_parse_one, todo, chunksize=8):
                st[status] = st.get(status, 0) + 1
                if sh:
                    S += sh
                if ev:
                    E += ev
        sh = pd.DataFrame(S, columns=["event_id", "season", "period", "team_id", "shooter", "assister", "made", "three",
                                      "dist_text", "dist_coord", "x", "y", "blocked", "type", "text"])
        sh["dist"] = sh.dist_text.fillna(sh.dist_coord)
        # ESPN text often omits "three point" on MISSED threes ("misses 25-foot step back jumpshot"): a miss is a three
        # if it is in the corner box or >= 24 ft (text or coordinates). Checked against box 3PA in `calib`.
        corner = (sh.y <= CORNER_Y) & ((sh.x - 25).abs() >= 22)
        sh["three_text"] = sh.three
        sh["three"] = sh.three | (~sh.made & (corner | (sh.dist_coord >= 24) | (sh.dist_text >= 24)))
        sh["zone"] = [_zone(t, d if not np.isnan(d) else 0.0, y if not np.isnan(y) else None)
                      for t, d, y in zip(sh.three, sh.dist, sh.y)]
        ev = pd.DataFrame(E, columns=["event_id", "season", "period", "team_id", "player", "kind", "steal", "type"])
        sh.to_parquet(PBP / f"shots_{s}.parquet", index=False)
        ev.to_parquet(PBP / f"tov_{s}.parquet", index=False)
        cov[s] = {"games": len(g), **st, "shots": len(sh), "tov_events": len(ev)}
        print(s, cov[s], flush=True)
    (PBP / "parse_coverage.json").write_text(json.dumps(cov, indent=1))
    return 0


# ============================================================================================== shared loaders
def games(seasons) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(ESPN_D / f"games_{s}.parquet") for s in seasons], ignore_index=True)
    g = g[(g.season_type == "regular-season") & (g.status == "STATUS_FINAL")].copy()
    nba = g.home_team_id.value_counts()
    nba = set(nba[nba >= 20].index)
    g = g[g.home_team_id.isin(nba) & g.away_team_id.isin(nba)]
    g["event_id"] = g.event_id.astype(str)
    g["home_team_id"] = g.home_team_id.astype(str)
    g["away_team_id"] = g.away_team_id.astype(str)
    g["et_date"] = g.start_utc.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    g["margin"] = g.home_score - g.away_score
    g["pts"] = g.home_score + g.away_score
    return g.sort_values("start_utc").reset_index(drop=True)


def box_team(seasons) -> pd.DataFrame:
    out = []
    for s in seasons:
        b = pd.read_parquet(ESPN_D / f"box_{s}.parquet")
        out.append(b)
    b = pd.concat(out, ignore_index=True)
    return b


def team_box_totals(seasons, g) -> pd.DataFrame:
    """Per (event, team) box totals from player rows: FGA, FGM, 3PA, 3PM, FTA, OREB, TOV, AST, STL, BLK, PTS."""
    b = box_team(seasons)
    b["event_id"] = b.event_id.astype(str)
    b["team_id"] = b.team_id.astype(str)
    b = b[b.event_id.isin(set(g.event_id))]
    want = {"fieldGoalsAttempted": "fga", "fieldGoalsMade": "fgm", "threePointFieldGoalsAttempted": "fg3a",
            "threePointFieldGoalsMade": "fg3m", "freeThrowsAttempted": "fta", "offensiveRebounds": "orb",
            "turnovers": "tov", "assists": "ast", "steals": "stl", "blocks": "blk", "points": "pts"}
    b = b[b.stat.isin(want)]
    t = b.pivot_table(index=["event_id", "team_id"], columns="stat", values="value", aggfunc="sum").reset_index()
    t = t.rename(columns=want)
    t["poss"] = t.fga - t.orb + t.tov + 0.44 * t.fta
    return t


def load_shots(seasons) -> pd.DataFrame:
    sh = pd.concat([pd.read_parquet(PBP / f"shots_{s}.parquet") for s in seasons], ignore_index=True)
    sh["event_id"] = sh.event_id.astype(str)
    sh["team_id"] = sh.team_id.astype(str)
    return sh


def load_tov(seasons) -> pd.DataFrame:
    e = pd.concat([pd.read_parquet(PBP / f"tov_{s}.parquet") for s in seasons], ignore_index=True)
    e["event_id"] = e.event_id.astype(str)
    e["team_id"] = e.team_id.astype(str)
    return e


# ============================================================================================== calib
def cmd_calib(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for s in SEASONS:
        g = games([s])
        sh = load_shots([s])
        tv = load_tov([s])
        bt = team_box_totals([s], g)
        b = box_team([s])
        b["event_id"], b["team_id"], b["athlete_id"] = b.event_id.astype(str), b.team_id.astype(str), b.athlete_id.astype(str)
        mem = b.drop_duplicates(["event_id", "athlete_id"]).set_index(["event_id", "athlete_id"]).team_id
        sh["shooter"] = sh.shooter.astype(str)
        k = pd.MultiIndex.from_arrays([sh.event_id, sh.shooter])
        st = mem.reindex(k).to_numpy()
        r = {"games": int(g.event_id.nunique()), "games_with_shots": int(sh.event_id.nunique()),
             "fga": int(len(sh)), "shooter_on_play_team": float(np.mean(st == sh.team_id.to_numpy())),
             "shooter_in_box": float(np.mean(pd.notna(st))),
             "coord_missing": float(sh.x.isna().mean()), "text_dist_share": float(sh.dist_text.notna().mean())}
        v = sh.dropna(subset=["dist_text", "x"])
        r["coord_vs_text_mae_ft"] = float((v.dist_coord - v.dist_text).abs().mean())
        r["coord_vs_text_within1"] = float(((v.dist_coord - v.dist_text).abs() <= 1).mean())
        t3 = sh[sh.three & sh.dist_text.notna()]
        r["three_text_dist_q"] = {z: [float(x) for x in t3[t3.zone == z].dist_text.quantile([.01, .05, .5]).values]
                                  for z in ("c3", "ab3")}
        r["three_under_22ft_share"] = float((t3.dist_text < 22).mean())
        t2 = sh[~sh.three & sh.dist_text.notna()]
        r["two_over_24ft_share"] = float((t2.dist_text > 24).mean())
        r["zone_share"] = {z: float((sh.zone == z).mean()) for z in ZONES}
        r["zone_fg"] = {z: float(sh[sh.zone == z].made.mean()) for z in ZONES}
        # pbp vs box team totals
        agg = sh.groupby(["event_id", "team_id"]).agg(fga=("made", "size"), fgm=("made", "sum"),
                                                     fg3a=("three", "sum"),
                                                     fg3m=("three", lambda x: int((x & sh.loc[x.index, "made"]).sum())),
                                                     ast=("assister", lambda x: int(x.notna().sum())))
        tt = tv.groupby(["event_id", "team_id"]).agg(tov=("kind", "size"))
        stl = tv[tv.steal].groupby(["event_id", "team_id"]).size().rename("stl_against")
        j = bt.set_index(["event_id", "team_id"]).join(agg, rsuffix="_p").join(tt, rsuffix="_p").join(stl)
        for c in ("fga", "fgm", "fg3a", "fg3m", "ast", "tov"):
            d = (j[f"{c}_p"] - j[c]).dropna()
            r[f"{c}_pbp_minus_box"] = {"mean": float(d.mean()), "exact": float((d == 0).mean()), "abs_le1": float((d.abs() <= 1).mean())}
        res[s] = r
        print(s, json.dumps({k: r[k] for k in r if k not in ("zone_fg",)}), flush=True)
    (OUT / "calib.json").write_text(json.dumps(res, indent=1))
    return 0


# ============================================================================================== team-game table
def zone_counts(sh: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    c = sh.groupby(by + ["zone"]).made.agg(["size", "sum"]).unstack("zone", fill_value=0)
    c.columns = [f"{'fga' if a == 'size' else 'fgm'}_{z}" for a, z in c.columns]
    for z in ZONES:
        for p in ("fga", "fgm"):
            if f"{p}_{z}" not in c:
                c[f"{p}_{z}"] = 0
    c["fga"] = c[[f"fga_{z}" for z in ZONES]].sum(1)
    c["fgm"] = c[[f"fgm_{z}" for z in ZONES]].sum(1)
    return c


def team_games(seasons) -> pd.DataFrame:
    """One row per (event, team): offense zone counts, defense (opponent) zone counts, box passing counts."""
    g = games(seasons)
    sh = load_shots(seasons)
    sh = sh[sh.event_id.isin(set(g.event_id))]
    oc = zone_counts(sh, ["event_id", "team_id"])
    rb = sh[(sh.zone == "rim") & sh.blocked].groupby(["event_id", "team_id"]).size().rename("rimblk_against")
    oc = oc.join(rb).fillna({"rimblk_against": 0})
    bt = team_box_totals(seasons, g).set_index(["event_id", "team_id"])[["ast", "tov", "stl", "poss", "fgm"]]
    bt = bt.rename(columns={"fgm": "fgm_box"})
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({"event_id": g.event_id, "season": g.season, "et_date": g.et_date,
                                  "start_utc": g.start_utc, "team_id": g[f"{side}_team_id"],
                                  "opp_id": g[f"{opp}_team_id"], "team": g[side], "opp": g[opp],
                                  "is_home": side == "home"}))
    t = pd.concat(rows, ignore_index=True)
    o = oc.join(bt, how="left")
    o.columns = [f"o_{c}" for c in o.columns]
    d = o.copy()
    d.columns = [f"d_{c[2:]}" for c in d.columns]
    t = t.join(o, on=["event_id", "team_id"]).join(d, on=["event_id", "opp_id"])
    t = t.sort_values(["team_id", "start_utc"]).reset_index(drop=True)
    t["game_no"] = t.groupby(["team_id", "season"]).cumcount() + 1
    return t


def metric_defs() -> dict:
    """name -> (numerator col, denominator col, kind). kind: 'mix'/'fg' per zone, else passing."""
    M = {}
    for z in ZONES:
        M[f"om_{z}"] = (f"o_fga_{z}", "o_fga", "mix")
        M[f"op_{z}"] = (f"o_fgm_{z}", f"o_fga_{z}", "fg")
        M[f"dm_{z}"] = (f"d_fga_{z}", "d_fga", "mix")
        M[f"dp_{z}"] = (f"d_fgm_{z}", f"d_fga_{z}", "fg")
    M["o_ast_sh"] = ("o_ast", "o_fgm_box", "pass")
    M["d_ast_sh"] = ("d_ast", "d_fgm_box", "pass")
    M["o_tov_r"] = ("o_tov", "o_poss", "pass")
    M["d_tov_r"] = ("d_tov", "d_poss", "pass")
    M["d_stl_r"] = ("o_stl", "d_poss", "pass")       # this team's steals per opponent possession
    M["d_rimblk"] = ("d_rimblk_against", "d_fga_rim", "pass")  # opponents' rim FGA blocked (rim protection)
    return M


# ============================================================================================== reliability
def cmd_reliability(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t = team_games(HIST + DEV)
    M = metric_defs()
    t["half"] = t.game_no % 2
    res = {}
    for nm, (num, den, kind) in M.items():
        h = t.groupby(["season", "team_id", "half"])[[num, den]].sum()
        r = (h[num] / h[den]).unstack("half")
        r = r[r.index.get_level_values("season").isin(DEV)].dropna()
        r = r - r.groupby(level="season").transform("mean")
        x, y = r[0].to_numpy(), r[1].to_numpy()
        rr = float(np.corrcoef(x, y)[0, 1])
        rng = np.random.default_rng(1)
        bs = []
        for _ in range(2000):
            k = rng.integers(0, len(x), len(x))
            bs.append(np.corrcoef(x[k], y[k])[0, 1])
        lo, hi = np.quantile(bs, [.025, .975])
        # season 2020-21 reported separately (history)
        res[nm] = {"r_split_half": rr, "ci": [float(lo), float(hi)], "n_team_seasons": int(len(x)),
                   "spearman_brown_full": float(2 * rr / (1 + rr)) if rr > -1 else None,
                   "stable": bool(rr >= 0.30 and lo > 0), "kind": kind}
    (OUT / "reliability.json").write_text(json.dumps(res, indent=1))
    print(f"{'metric':<12}{'r(odd,even)':>12}{'95% CI':>18}{'SB full':>9}  stable")
    for nm, v in res.items():
        print(f"{nm:<12}{v['r_split_half']:>12.3f}   [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]{v['spearman_brown_full']:>9.3f}  "
              f"{'YES' if v['stable'] else 'no'}")
    print(f"n team-seasons = {next(iter(res.values()))['n_team_seasons']} (DEV {DEV})")
    return 0


def gate() -> dict:
    return {k: v["stable"] for k, v in json.loads((OUT / "reliability.json").read_text()).items()}


# ============================================================================================== PIT build
def eb_k(num: np.ndarray, den: np.ndarray) -> float:
    """Method-of-moments EB prior strength for a binomial-type rate across units (inf if no true variance)."""
    ok = den > 0
    num, den = num[ok], den[ok]
    L = num.sum() / den.sum()
    r = num / den
    w = den / den.sum()
    var_obs = float(np.sum(w * (r - L) ** 2))
    samp = float(np.sum(w * L * (1 - L) / den))
    tau2 = var_obs - samp
    return float(L * (1 - L) / tau2) if tau2 > 0 else float("inf")


def league_rates(t: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per (season, et_date): league sums over team-games strictly before that date in the season, or season s-1
    full totals when fewer than 300 team-games are available. Returns sums (not rates) + team-game count."""
    out = []
    full = t.groupby("season")[cols].sum()
    full["n_tg"] = t.groupby("season").size()
    for s, x in t.groupby("season"):
        dd = x.groupby("et_date")[cols].sum().sort_index()
        dd["n_tg"] = x.groupby("et_date").size()
        cum = dd.cumsum().shift(1).fillna(0)
        if s - 1 in full.index:
            prev = full.loc[s - 1]
            use_prev = cum.n_tg < 300
            cum.loc[use_prev, :] = prev.values
        else:
            cum.loc[cum.n_tg < 300, :] = full.loc[s].values  # first history season only (2020-21; not scored)
        cum["season"] = s
        out.append(cum.reset_index())
    return pd.concat(out, ignore_index=True)


def build_team_pit(t: pd.DataFrame, G: dict) -> pd.DataFrame:
    """Adds PIT shrunk deviations (gated) for offense and defense to each team-game row."""
    M = metric_defs()
    cols = sorted({c for v in M.values() for c in v[:2]})
    t = t.copy()
    grp = t.groupby(["team_id", "season"], sort=False)
    cum = grp[cols].cumsum() - t[cols]                     # strictly earlier games of this team-season
    LG = league_rates(t, cols).set_index(["season", "et_date"])
    Lk = LG.reindex(pd.MultiIndex.from_arrays([t.season, t.et_date]))
    seasons_tot = t.groupby(["season", "team_id"])[cols].sum()
    kk = {}
    for s in sorted(t.season.unique()):
        src = s - 1 if s - 1 in seasons_tot.index.get_level_values(0) else s
        st = seasons_tot.loc[src]
        kk[s] = {nm: eb_k(st[num].to_numpy(float), st[den].to_numpy(float)) for nm, (num, den, _) in M.items()}
    t["n_prior"] = grp.cumcount()
    for nm, (num, den, kind) in M.items():
        L = (Lk[num] / Lk[den]).to_numpy()
        k = t.season.map({s: kk[s][nm] for s in kk}).to_numpy(float)
        cn, cd = cum[num].to_numpy(float), cum[den].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            est = np.where(np.isinf(k), L, (cn + np.where(np.isinf(k), 0, k) * L) / (cd + np.where(np.isinf(k), 1, k)))
        dev = est - L
        if not G.get(nm, False):
            dev = np.zeros_like(dev)
        t[f"L_{nm}"] = L
        t[f"dev_{nm}"] = dev
    t["L_fga_pg"] = (Lk["o_fga"] / Lk["n_tg"]).to_numpy()
    t.attrs["k"] = kk
    return t


def interaction(off: pd.DataFrame, de: pd.DataFrame, fga) -> np.ndarray:
    """I = FGA x sum_z [a_z e_z + b_z c_z]; off supplies a (dev_om_z) and c (value x dev_op_z), de supplies
    b (dev_dm_z) and e (value x dev_dp_z). Arrays aligned row-by-row."""
    tot = 0.0
    for z in ZONES:
        a_ = off[f"dev_om_{z}"].to_numpy(float)
        c_ = ZVAL[z] * off[f"dev_op_{z}"].to_numpy(float)
        b_ = de[f"dev_dm_{z}"].to_numpy(float)
        e_ = ZVAL[z] * de[f"dev_dp_{z}"].to_numpy(float)
        tot = tot + a_ * e_ + b_ * c_
    return np.asarray(fga, float) * tot


def game_terms(tp: pd.DataFrame) -> pd.DataFrame:
    h = tp[tp.is_home].set_index("event_id")
    a = tp[~tp.is_home].set_index("event_id")
    a = a.reindex(h.index)
    I_h = interaction(h, a, h.L_fga_pg)       # home offense vs away defense
    I_a = interaction(a, h, h.L_fga_pg)
    out = pd.DataFrame({"event_id": h.index, "season": h.season.values, "et_date": h.et_date.values,
                        "home": h.team.values, "away": h.opp.values, "home_team_id": h.team_id.values,
                        "away_team_id": h.opp_id.values, "n_prior_h": h.n_prior.values, "n_prior_a": a.n_prior.values,
                        "I_h": I_h, "I_a": I_a})
    out["I_total"] = out.I_h + out.I_a
    out["I_margin"] = out.I_h - out.I_a
    # descriptive extras: plain (marginal) expected points per shot deviation, and the passing interaction
    for side, X, Y in (("h", h, a), ("a", a, h)):
        pl = np.zeros(len(X))
        for z in ZONES:
            Lm, Lp = X[f"L_om_{z}"].to_numpy(), ZVAL[z] * X[f"L_op_{z}"].to_numpy()
            a_, c_ = X[f"raw_om_{z}"].to_numpy() - Lm, ZVAL[z] * X[f"raw_op_{z}"].to_numpy() - Lp
            b_, e_ = Y[f"raw_dm_{z}"].to_numpy() - Lm, ZVAL[z] * Y[f"raw_dp_{z}"].to_numpy() - Lp
            pl += a_ * Lp + Lm * c_ + b_ * Lp + Lm * e_
        out[f"plain_{side}"] = pl * h.L_fga_pg.values
        out[f"pass_{side}"] = X["raw_o_ast_sh"].to_numpy() - X["L_o_ast_sh"].to_numpy()
        out[f"stl_{side}"] = Y["raw_d_stl_r"].to_numpy() - Y["L_d_stl_r"].to_numpy()
    return out


def add_raw(t: pd.DataFrame) -> pd.DataFrame:
    """Ungated shrunk estimates (raw_*) for descriptive terms (plain marginal rating, passing)."""
    G_all = {k: True for k in metric_defs()}
    r = build_team_pit(t, G_all)
    for nm in metric_defs():
        t[f"raw_{nm}"] = r[f"L_{nm}"] + r[f"dev_{nm}"]
    return t


# ---------------------------------------------------------------------------------------------- players
def player_pit(seasons, tp: pd.DataFrame, G: dict) -> pd.DataFrame:
    """Per player-game (minutes > 0): PIT player zone deviations (last 82 games played, shrunk), FGA_p (last 20),
    opponent's PIT defense deviations from tp, and adj_p."""
    from nba_player_opp_venue import load_box
    b = load_box(sorted(set(seasons) | {min(seasons) - 1}) if min(seasons) - 1 >= 2019 else seasons)
    b["event_id"], b["athlete_id"], b["team_id"] = b.event_id.astype(str), b.athlete_id.astype(str), b.team_id.astype(str)
    b["opp_id"] = b.opp_id.astype(str)
    b = b[b.season_type == "regular-season"] if "season_type" in b else b
    sh = load_shots(sorted({s for s in SEASONS if s >= min(b.season)}))
    sh["shooter"] = sh.shooter.astype(str)
    pc = zone_counts(sh, ["event_id", "shooter"])
    pc.index = pc.index.set_names(["event_id", "athlete_id"])
    b = b.join(pc, on=["event_id", "athlete_id"])
    cnt = [f"fga_{z}" for z in ZONES] + [f"fgm_{z}" for z in ZONES] + ["fga"]
    b[cnt] = b[cnt].fillna(0)
    b = b.sort_values(["athlete_id", "start_utc"]).reset_index(drop=True)
    gp = b.groupby("athlete_id", sort=False)
    roll = gp[cnt].transform(lambda x: x.shift(1).rolling(82, min_periods=1).sum())
    b["fga_p"] = gp.fga.transform(lambda x: x.shift(1).rolling(20, min_periods=10).mean())
    b["n_prior_p"] = gp.cumcount()
    # player EB k per season from player-seasons of s-1 with >= 100 FGA
    ps = b.groupby(["season", "athlete_id"])[cnt].sum()
    kk = {}
    for s in sorted(b.season.unique()):
        if s - 1 not in ps.index.get_level_values(0):
            continue
        x = ps.loc[s - 1]
        x = x[x.fga >= 100]
        kk[s] = {}
        for z in ZONES:
            kk[s][f"om_{z}"] = eb_k(x[f"fga_{z}"].to_numpy(float), x.fga.to_numpy(float))
            kk[s][f"op_{z}"] = eb_k(x[f"fgm_{z}"].to_numpy(float), x[f"fga_{z}"].to_numpy(float))
    b = b[b.season.isin(seasons) & b.season.isin(list(kk))].copy()
    roll = roll.loc[b.index]
    # league rates and opponent defense from the team PIT table (same event)
    td = tp.set_index(["event_id", "team_id"])
    opp = td.reindex(pd.MultiIndex.from_arrays([b.event_id, b.opp_id]))
    adj = np.zeros(len(b))
    for z in ZONES:
        for nm, num, den in ((f"om_{z}", f"fga_{z}", "fga"), (f"op_{z}", f"fgm_{z}", f"fga_{z}")):
            L = opp[f"L_{nm}"].to_numpy(float)
            k = b.season.map({s: kk[s][nm] for s in kk}).to_numpy(float)
            kf = np.where(np.isinf(k), 0, k)
            with np.errstate(invalid="ignore", divide="ignore"):
                est = np.where(np.isinf(k), L, (roll[num].to_numpy(float) + kf * L) / (roll[den].to_numpy(float) + kf))
            dev = est - L
            if not G.get(nm, False):
                dev = np.zeros_like(dev)
            b[f"pdev_{nm}"] = dev
        a_, c_ = b[f"pdev_om_{z}"].to_numpy(), ZVAL[z] * b[f"pdev_op_{z}"].to_numpy()
        bb, e_ = opp[f"dev_dm_{z}"].to_numpy(float), ZVAL[z] * opp[f"dev_dp_{z}"].to_numpy(float)
        adj += a_ * e_ + bb * c_
    b["adj_p"] = b.fga_p.to_numpy() * adj
    b["eligible"] = b.fga_p.notna() & (b.n_prior_p >= 10) & np.isfinite(b.adj_p)
    b.attrs["k"] = kk
    return b


def cmd_build(a) -> int:
    G = gate()
    t = team_games(SEASONS)
    t = add_raw(t)
    tp = build_team_pit(t, G)
    gt = game_terms(tp)
    PBP.mkdir(parents=True, exist_ok=True)
    tk = tp.attrs.pop("k")
    tp.to_parquet(PBP / "team_pit.parquet", index=False)
    gt.attrs = {}
    gt.to_parquet(PBP / "game_terms.parquet", index=False)
    pp = player_pit(SEASONS[1:], tp, G)
    keep = ["event_id", "athlete_id", "name", "team_id", "opp_id", "season", "date", "start_utc", "minutes", "points",
            "fga", "fga_p", "n_prior_p", "adj_p", "eligible"]
    pk = pp.attrs.pop("k")
    pp[keep].to_parquet(PBP / "player_terms.parquet", index=False)
    kinfo = {"team_k": {str(s): v for s, v in tk.items()}, "player_k": {str(s): v for s, v in pk.items()},
             "gate": G}
    (OUT / "build_info.json").write_text(json.dumps(kinfo, indent=1, default=float))
    print(f"team-games {len(tp)}, games {len(gt)}, player-games {len(pp)} (eligible {int(pp.eligible.sum())})")
    print("stable (gated in):", [k for k, v in G.items() if v])
    return 0


def thresholds() -> dict:
    gt = pd.read_parquet(PBP / "game_terms.parquet")
    pt = pd.read_parquet(PBP / "player_terms.parquet")
    X = {}
    for s in DEV + [SEALED]:
        p = gt[gt.season == s - 1]
        q = pt[(pt.season == s - 1) & pt.eligible]
        X[s] = {"X1": float(np.quantile(p.I_total.abs(), .8)), "X2": float(np.quantile(p.I_margin.abs(), .8)),
                "Y": float(np.quantile(q.adj_p.abs(), .9)) if len(q) else float("nan")}
    return X


def cmd_dist(a) -> int:
    """Signal distributions only (no outcomes, no lines)."""
    gt = pd.read_parquet(PBP / "game_terms.parquet")
    pt = pd.read_parquet(PBP / "player_terms.parquet")
    X = thresholds()
    print("walk-forward thresholds:", json.dumps(X, indent=0))
    for s, x in gt.groupby("season"):
        print(s, "I_total sd %.3f q90|.| %.3f  I_margin sd %.3f  share|I_tot|>=X1 %s" % (
            x.I_total.std(), x.I_total.abs().quantile(.9), x.I_margin.std(),
            f"{(x.I_total.abs() >= X[s]['X1']).mean():.3f}" if s in X else "-"))
        print("   corr(I_h, plain_h) %.3f   sd plain_h %.3f   sd I_h %.3f" % (
            np.corrcoef(x.I_h, x.plain_h)[0, 1], x.plain_h.std(), x.I_h.std()))
    for s, x in pt[pt.eligible].groupby("season"):
        print(s, "adj_p sd %.3f q90|.| %.3f n %d" % (x.adj_p.std(), x.adj_p.abs().quantile(.9), len(x)))
    return 0


# ============================================================================================== live (nba_sim --matchup)
def ensure_season_shots(S: int, before) -> int:
    """Fetch (<= 2 req/s, cached forever) and parse ESPN summaries for season S finals dated before `before`."""
    import sports.http as http
    g = pd.read_parquet(ESPN_D / f"games_{S}.parquet")
    g = g[(g.season_type == "regular-season") & (g.status == "STATUS_FINAL")]
    g = g[g.start_utc.dt.tz_convert("America/New_York").dt.date < before]
    old = http.MIN_INTERVAL
    http.MIN_INTERVAL = max(old, 0.5)
    try:
        for e in g.event_id.astype(str):
            f = http.CACHE_DIR / (http._key(f"{API}/summary?event={e}", None) + ".json")
            if not f.exists():
                http.fetch(f"{API}/summary?event={e}", cache=True)
    finally:
        http.MIN_INTERVAL = old
    S_rows = []
    for eid, sh, ev, status in map(_parse_one, [(str(e), S) for e in g.event_id]):
        if sh:
            S_rows += sh
    cols = ["event_id", "season", "period", "team_id", "shooter", "assister", "made", "three", "dist_text", "dist_coord",
            "x", "y", "blocked", "type", "text"]
    sh = pd.DataFrame(S_rows, columns=cols)
    sh["dist"] = sh.dist_text.fillna(sh.dist_coord)
    corner = (sh.y <= CORNER_Y) & ((sh.x - 25).abs() >= 22)
    sh["three_text"] = sh.three
    sh["three"] = sh.three.astype(bool) | (~sh.made.astype(bool) & (corner | (sh.dist_coord >= 24) | (sh.dist_text >= 24)))
    sh["zone"] = [_zone(t, d if not np.isnan(d) else 0.0, y if not np.isnan(y) else None)
                  for t, d, y in zip(sh.three, sh.dist, sh.y)]
    PBP.mkdir(parents=True, exist_ok=True)
    sh.to_parquet(PBP / f"shots_{S}.parquet", index=False)
    return int(g.event_id.nunique())


def live_terms(away: str, home: str, d) -> dict:
    """Style interaction for one upcoming game (abbreviations as in ESPN games files), state strictly before date d."""
    d = pd.Timestamp(d).date()
    S = d.year + 1 if d.month >= 8 else d.year
    ensure_season_shots(S, d)
    t = team_games([S - 1, S])
    gS = pd.read_parquet(ESPN_D / f"games_{S}.parquet")
    ids = {**dict(zip(gS.home, gS.home_team_id.astype(str))), **dict(zip(gS.away, gS.away_team_id.astype(str)))}
    if home not in ids or away not in ids:
        raise SystemExit(f"--matchup: team abbreviation not found in games_{S}: {home} / {away}")
    ts = pd.Timestamp(d).tz_localize("America/New_York").tz_convert("UTC") + pd.Timedelta(hours=23)
    fake = []
    for tm, op, ih in ((home, away, True), (away, home, False)):
        fake.append({"event_id": "LIVE", "season": S, "et_date": pd.Timestamp(d), "start_utc": ts, "team_id": ids[tm],
                     "opp_id": ids[op], "team": tm, "opp": op, "is_home": ih})
    f = pd.DataFrame(fake)
    for c in t.columns:
        if c not in f:
            f[c] = 0
    t = pd.concat([t, f[t.columns]], ignore_index=True).sort_values(["team_id", "start_utc"]).reset_index(drop=True)
    t = add_raw(t)
    tp = build_team_pit(t, gate())
    gt = game_terms(tp)
    r = gt[gt.event_id == "LIVE"].iloc[0]
    return {"season": S, "I_h": float(r.I_h), "I_a": float(r.I_a), "I_margin": float(r.I_margin),
            "I_total": float(r.I_total), "n_prior_h": int(r.n_prior_h), "n_prior_a": int(r.n_prior_a)}


# ============================================================================================== scoring
def ml_dec(o):
    o = np.asarray(o, float)
    return np.where(o > 0, 1 + o / 100, 1 + 100 / np.abs(o))


def cboot(v, clus, seed=SEED, nb=NB):
    c = pd.Series(np.asarray(v, float)).groupby(pd.factorize(pd.Series(clus))[0])
    s, k = c.sum().to_numpy(), c.size().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), (nb, len(s)))
    return s[idx].sum(1) / k[idx].sum(1)


def game_frame(seasons) -> pd.DataFrame:
    from nba_clock_calendar_arena import lines
    g = games(seasons)
    gt = pd.read_parquet(PBP / "game_terms.parquet")
    x = g.merge(gt[["event_id", "I_total", "I_margin", "I_h", "I_a", "plain_h", "plain_a", "pass_h", "pass_a",
                    "stl_h", "stl_a", "n_prior_h", "n_prior_a"]], on="event_id", how="inner")
    L = pd.concat([lines(s) for s in seasons], ignore_index=True)
    x = x.merge(L, on="event_id", how="left")
    x["date"] = x.et_date
    return x


def bets_m1(x: pd.DataFrame, X: dict) -> pd.DataFrame:
    thr = x.season.map({s: X[s]["X1"] for s in X})
    side = np.where(x.I_total >= thr, "over", np.where(x.I_total <= -thr, "under", ""))
    y = x[side != ""].copy()
    y["side"] = side[side != ""]
    y = y.dropna(subset=["total"])
    s = np.where(y.side == "over", y.pts - y.total, y.total - y.pts)
    odds = np.where(y.side == "over", y.over_odds, y.under_odds)
    y["odds"] = odds
    y = y[pd.notna(y.odds)]
    s = np.where(y.side == "over", y.pts - y.total, y.total - y.pts)
    d = ml_dec(y.odds)
    y["win"] = (s > 0).astype(int)
    y["push"] = (s == 0).astype(int)
    y["profit"] = np.where(s > 0, d - 1, np.where(s < 0, -1.0, 0.0))
    y["signal"] = y.I_total
    return y


def bets_m2(x: pd.DataFrame, X: dict) -> pd.DataFrame:
    thr = x.season.map({s: X[s]["X2"] for s in X})
    side = np.where(x.I_margin >= thr, "home_ats", np.where(x.I_margin <= -thr, "away_ats", ""))
    y = x[side != ""].copy()
    y["side"] = side[side != ""]
    y = y.dropna(subset=["spread"])
    y["odds"] = np.where(y.side == "home_ats", y.home_spread_odds, y.away_spread_odds)
    y = y[pd.notna(y.odds)]
    cov = y.margin + y.spread                         # home cover margin (spread = home line)
    s = np.where(y.side == "home_ats", cov, -cov)
    d = ml_dec(y.odds)
    y["win"] = (s > 0).astype(int)
    y["push"] = (s == 0).astype(int)
    y["profit"] = np.where(s > 0, d - 1, np.where(s < 0, -1.0, 0.0))
    y["signal"] = y.I_margin
    return y


def summarize(b: pd.DataFrame, clus: list[str], halves: str, bar: float) -> dict:
    if len(b) == 0:
        return {"bets": 0, "passed": False}
    v = b.profit.to_numpy()
    out = {"bets": int(len(b)), "units": float(v.sum()), "roi": float(v.mean()), "win_rate": float(b.win.mean()),
           "push_rate": float(b.push.mean()) if "push" in b else 0.0}
    ps = []
    for c in clus:
        bs = cboot(v, b[c].astype(str).to_numpy())
        out[f"ci_{c}"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        out[f"p_{c}"] = float(np.mean(bs <= 0))
        ps.append(out[f"p_{c}"])
    out["p"] = max(ps)
    out["ci_lo"], out["ci_hi"] = out[f"ci_{clus[0]}"]
    if halves == "season":
        hs = [b[b.season.isin(H1)], b[b.season.isin(H2)]]
    else:
        ds = np.sort(pd.to_datetime(b.date).unique())
        med = ds[len(ds) // 2]
        hs = [b[pd.to_datetime(b.date) < med], b[pd.to_datetime(b.date) >= med]]
    out["half1"], out["half2"] = [float(h.profit.mean()) if len(h) else None for h in hs]
    out["half_n"] = [int(len(h)) for h in hs]
    out["by_season"] = {int(s): {"n": int(len(y)), "roi": float(y.profit.mean())} for s, y in b.groupby("season")}
    out["by_side"] = {str(s): {"n": int(len(y)), "roi": float(y.profit.mean())} for s, y in b.groupby("side")}
    out["passed"] = bool(out["p"] < bar and (out["half1"] or -1) > 0 and (out["half2"] or -1) > 0)
    return out


def ols_boot(y, X, nb=NB, seed=SEED):
    n = len(y)
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b0
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var(ddof=X.shape[1]))
    rng = np.random.default_rng(seed)
    bs = np.empty((nb, X.shape[1]))
    for i in range(nb):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        Xw = X * w[:, None]
        bs[i] = np.linalg.solve(X.T @ Xw, Xw.T @ y)
    return b0, se, bs


def m4(x: pd.DataFrame, bar: float) -> dict:
    out = {}
    for nm, ycol, ccol, icol in (("margin", "margin", "close_margin", "I_margin"), ("total", "pts", "total", "I_total")):
        y = x.dropna(subset=[ccol, icol])
        X = np.c_[np.ones(len(y)), y[ccol].to_numpy(float), y[icol].to_numpy(float)]
        b, se, bs = ols_boot(y[ycol].to_numpy(float), X)
        g = bs[:, 2]
        halves = []
        for H in (H1, H2):
            yy = y[y.season.isin(H)]
            XX = np.c_[np.ones(len(yy)), yy[ccol].to_numpy(float), yy[icol].to_numpy(float)]
            halves.append(float(np.linalg.lstsq(XX, yy[ycol].to_numpy(float), rcond=None)[0][2]))
        # residual-on-signal view: (actual - close) per 1 pt of I
        r = {"n": int(len(y)), "const": float(b[0]), "coef_close": float(b[1]), "gamma": float(b[2]),
             "gamma_se_ols": float(se[2]), "gamma_ci_boot": [float(np.quantile(g, .025)), float(np.quantile(g, .975))],
             "p_one_sided": float(np.mean(g <= 0)), "halves": halves, "sd_I": float(y[icol].std())}
        r["passed"] = bool(r["p_one_sided"] < bar / 2 and all(h > 0 for h in halves))
        out[nm] = r
    out["passed"] = bool(out["margin"]["passed"] or out["total"]["passed"])
    return out


def props_frame(seasons) -> pd.DataFrame:
    from nba_player_opp_venue import quotes
    q = quotes(seasons, at="open")
    q = q[q.stat == "points"].copy()
    q["event_id"], q["athlete_id"] = q.event_id.astype(str), q.athlete_id.astype(str)
    pt = pd.read_parquet(PBP / "player_terms.parquet")
    m = q.merge(pt, on=["event_id", "athlete_id"], how="left", indicator=True)
    m.attrs["audit"] = {"quotes": int(len(q)), "matched": int((m._merge == "both").sum()),
                        "actual_eq_box_points": float((m.loc[m._merge == "both", "actual"]
                                                       == m.loc[m._merge == "both", "points"]).mean())}
    m = m[m._merge == "both"].drop(columns="_merge")
    return m


def bets_m3(m: pd.DataFrame, X: dict) -> pd.DataFrame:
    m = m[m.eligible.astype(bool)].copy()
    thr = m.season.map({s: X[s]["Y"] for s in X})
    side = np.where(m.adj_p >= thr, "over", np.where(m.adj_p <= -thr, "under", ""))
    y = m[side != ""].copy()
    y["side"] = side[side != ""]
    ov = y.side == "over"
    y["win"] = np.where(ov, y.over, 1 - y.over)
    y["dec"] = np.where(ov, y.d_o, y.d_u)
    y["p_side"] = np.where(ov, y.p_over, 1 - y.p_over)
    y["profit"] = np.where(y.win == 1, y.dec - 1, -1.0)
    y["push"] = 0
    y["signal"] = y.adj_p
    return y


def ledger_row(trial: str, n: int, r: dict, split: str) -> dict:
    sha = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"style_{trial}_{split.lower()}",
           "sha": sha, "dataset": DATASET, "hypothesis": "pre-registered in notes/nba_style_matchups.md",
           "source": "user hypothesis 10-30 (style matchups)", "status": "ok", "split": split, "trial_number": n,
           "study_trials": K_TRIALS, "bonferroni_bar": BAR, "trial": trial}
    row.update({k: v for k, v in r.items() if k not in ("by_season", "by_side")})
    return row


def cmd_score(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    X = thresholds()
    res = {"thresholds": {str(k): v for k, v in X.items()}}
    x = game_frame(DEV)
    x["close_margin"] = -x.spread
    res["games_dev"] = {"games": int(len(x)), "with_total": int(x.total.notna().sum()), "with_spread": int(x.spread.notna().sum()),
                        "provider": x.provider.value_counts().to_dict()}
    b1, b2 = bets_m1(x, X), bets_m2(x, X)
    res["M1"] = summarize(b1, ["event_id"], "season", BAR)
    res["M2"] = summarize(b2, ["event_id"], "season", BAR)
    m = props_frame(PROP_DEV)
    res["props_audit"] = m.attrs["audit"]
    b3 = bets_m3(m, X)
    res["M3"] = summarize(b3, ["event_id", "date"], "date", BAR)
    res["M3"]["hit_minus_novig"] = float(b3.win.mean() - b3.p_side.mean()) if len(b3) else None
    res["M4"] = m4(x, BAR)
    # descriptive (not trials)
    d = {}
    x["res_total"], x["res_margin"] = x.pts - x.total, x.margin - x.close_margin
    for col, sig in (("res_total", "I_total"), ("res_margin", "I_margin")):
        y = x.dropna(subset=[col])
        q = pd.qcut(y[sig], 5, labels=False)
        d[f"{col}_by_{sig}_quintile"] = {int(k): {"n": int(len(v)), "mean_sig": float(v[sig].mean()),
                                                   "mean_res": float(v[col].mean())} for k, v in y.groupby(q)}
    y = x.dropna(subset=["res_margin", "res_total"])
    pass_m = (y.pass_h * y.stl_a) - (y.pass_a * y.stl_h)
    pass_t = (y.pass_h * y.stl_a) + (y.pass_a * y.stl_h)
    for nm, col, sig in (("pass_margin", "res_margin", pass_m), ("pass_total", "res_total", pass_t)):
        z = (sig - sig.mean()) / sig.std()
        Xd = np.c_[np.ones(len(y)), z.to_numpy()]
        b, se, bs = ols_boot(y[col].to_numpy(float), Xd, nb=2000)
        d[nm] = {"pts_per_sd": float(b[1]), "ci": [float(np.quantile(bs[:, 1], .025)), float(np.quantile(bs[:, 1], .975))]}
    # plain (marginal) zone rating vs close residual -- is the marginal priced? (descriptive)
    pm = y.plain_h - y.plain_a
    Xd = np.c_[np.ones(len(y)), ((pm - pm.mean()) / pm.std()).to_numpy()]
    b, se, bs = ols_boot(y.res_margin.to_numpy(float), Xd, nb=2000)
    d["plain_zone_margin_vs_close_res"] = {"pts_per_sd": float(b[1]), "ci": [float(np.quantile(bs[:, 1], .025)),
                                                                            float(np.quantile(bs[:, 1], .975))]}
    res["descriptive"] = d
    trials = {"M1": res["M1"], "M2": res["M2"], "M3": res["M3"],
              "M4": {"passed": res["M4"]["passed"], "p": min(res["M4"]["margin"]["p_one_sided"], res["M4"]["total"]["p_one_sided"]),
                     "gamma_margin": res["M4"]["margin"]["gamma"], "gamma_margin_ci": res["M4"]["margin"]["gamma_ci_boot"],
                     "gamma_total": res["M4"]["total"]["gamma"], "gamma_total_ci": res["M4"]["total"]["gamma_ci_boot"],
                     "halves_margin": res["M4"]["margin"]["halves"], "halves_total": res["M4"]["total"]["halves"]}}
    for nm, b in (("M1", b1), ("M2", b2), ("M3", b3)):
        b.to_csv(OUT / f"bets_{nm}_dev.csv", index=False)
    # sealed only with --sealed and only for passed trials
    passed = [k for k, v in trials.items() if v["passed"]]
    res["dev_passed"] = passed
    if a.sealed and passed:
        bar_s = 0.05 / len(passed)
        xs = game_frame([SEALED])
        xs["close_margin"] = -xs.spread
        sres = {}
        for k in passed:
            if k == "M1":
                sres[k] = summarize(bets_m1(xs, X), ["event_id"], "date", bar_s)
            elif k == "M2":
                sres[k] = summarize(bets_m2(xs, X), ["event_id"], "date", bar_s)
            elif k == "M3":
                sres[k] = summarize(bets_m3(props_frame([SEALED]), X), ["event_id", "date"], "date", bar_s)
            elif k == "M4":
                sres[k] = m4(xs.assign(season=np.where(xs.et_date < xs.et_date.median(), H1[0], H2[0])), 0.05 / len(passed) * 2)
        res["sealed"] = sres
    (OUT / "score.json").write_text(json.dumps(res, indent=1, default=str))
    if a.ledger:
        with open(LEDGER, "a") as f:
            for i, (k, v) in enumerate(trials.items(), 1):
                f.write(json.dumps(ledger_row(k, i, v, "DEV"), default=str) + "\n")
    print(json.dumps({k: res[k] for k in res if k not in ("descriptive",)}, indent=1, default=str)[:6000])
    print(json.dumps(res["descriptive"], indent=1)[:3000])
    return 0


# ============================================================================================== CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse")
    p.add_argument("--seasons", type=int, nargs="*")
    for c in ("calib", "reliability", "build", "dist"):
        sub.add_parser(c)
    sc = sub.add_parser("score")
    sc.add_argument("--ledger", action="store_true")
    sc.add_argument("--sealed", action="store_true")
    a = ap.parse_args(argv)
    cmds = {"parse": cmd_parse, "calib": cmd_calib}
    for nm in ("reliability", "build", "dist", "score"):
        if f"cmd_{nm}" in globals():
            cmds[nm] = globals()[f"cmd_{nm}"]
    return cmds[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
