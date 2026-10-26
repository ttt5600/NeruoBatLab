"""NHL simple model vs the OPENING line (DEV ONLY). Pre-registered 2026-10-25 before any model-vs-market join.

    ../.venv_quant/bin/python scripts/nhl_simple_model.py build              # features + walk-forward preds, no market
    ../.venv_quant/bin/python scripts/nhl_simple_model.py backtest [--ledger] # pre-registered trials vs ESPN BET opener

Template: notes/nba_simple_model.md (simple linear model led the NBA opener, not the close).

DATA (dev only; NHL 2026 = 2025-26 is SEALED per data/frozen/2026-10-14/MANIFEST.json and is never read):
  - ESPN games/box/lines seasons 2022..2025 (= 2021-22 .. 2024-25), regular season, STATUS_FINAL, not neutral.
  - data/nhl_games.parquet, read with a pyarrow filter date < 2025-08-01 (so 2025-26 rows are never loaded):
    starting goalie names and the overtime/shootout flag.
  - Step-0 data check (before this docstring): ESPN BET has open+close ML/total WITH PRICES for ~95% of 2023-24
    and ~100% of 2024-25 games. 2021-22 and 2022-23 have NO openers anywhere in our data. Seen before writing this:
    line coverage per book, the distribution of total_open (5.5/6.5) and its odds. No outcome or model was joined.

MODEL (all features from games strictly before the game date; walk-forward):
  Unit = team-game (two rows per game). Target = REGULATION goals (final score minus the OT/SO winner's goal when
  nhl_games.overtime is true).
  r_gf   ridge: goals = league + att[team] - def[opp] + home, exp decay half-life 90 d, lambda 10, refit each date.
  g_opp  opposing starting goalie quality in goals: rating x 30 shots. rating = shrunk GSAA per shot =
         sum_w(saves - shots_against x league_sv) / (sum_w(shots_against) + 1500), decay half-life 365 d;
         league_sv = decayed league save% over the same window. All goalie appearances count (incl. relief).
  b2b_t, b2b_o  team / opponent played the previous calendar day.
  S1_shots  z(team decayed shots-for per game) + z(opp decayed shots-against per game)      (decay 60 d)
  S2_pen    z(opp decayed PIM taken per game) + z(team decayed PIM drawn per game)           (PP/PK proxy, 60 d)
  S3_sh     z(team decayed shooting % (goals/shots for), 60 d)                              (PDO-luck regression)
  Stack: OLS  goals ~ r_gf + g_opp + b2b_t + b2b_o + S1 + S2 + S3. Walk-forward by season: for scored season s,
  fit on seasons 2023..s-1 (2021-22 = warm-up only, ratings cold). Scored seasons 2024 (23-24) and 2025 (24-25).
  lambda = max(pred, 0.5). Home/away goals independent Poisson.
  AMENDMENT A1 (2026-10-25, after the accuracy-only `build`, BEFORE any market join): the first build under-predicted
  the 2024-25 total by ~0.4 goals because league shots/game fell (31.3 -> 28.4) and the S1/S2/S3 raw features were
  z-scored against fixed training-season means. Fix: each raw style feature is first expressed RELATIVE to the
  league's decayed (60 d) mean over games before the date, then z-scored with training SDs. No other change.
  P(home win) = P(H > A) + 0.5 * P(H = A).
  Total T = H + A + 1{H = A} (OT/SO adds exactly one goal to the final). P(over L) from that distribution; for an
  integer L, P(over | no push).
  Goalie variants: ACT = the actual starter (from nhl_games; known ~pre-game but AFTER the open => lookahead vs the
  open). STRICT = projected starter = most starts in the team's last 10 games (tie -> most recent). Stack is fit on
  ACT features (the goalie effect); STRICT preds plug the projected starter in. goalie_change game = actual starter
  != projected starter for either team.

MARKET: ESPN BET only (the one book with openers in both dev seasons). Open = *_open, close = same book *_close.
  Fair probs = proportional de-vig (sports.market.devig_two_way). Offered price = the open American odds.
  Totals CLV in GOALS: implied market mean mu solved so that P(over L | mu split evenly, model's Poisson+OT rule)
  equals the de-vigged over prob; CLV = side x (mu_close - mu_open). This handles 5.5 <-> 6.5 moves and juice moves.

TRIALS (8; bar p < 0.05/8 = 0.00625, one-sided game bootstrap 4,000; pass also needs effect > 0; report per season,
fav/dog, CIs):
  T1 ML STRICT vs open: bet side where |p_model - p_open_fair| >= 0.03; CLV in prob points (p_close_fair - p_open_fair)
  T2 same bets, ROI at the no-vig opening price
  T3 same bets, ROI at the offered opening price
  T4 totals STRICT vs open: bet over/under where |P_over_model - p_over_open_fair| >= 0.04; CLV in goals (above)
  T5 same bets, ROI at no-vig opening price (push = 0)
  T6 same bets, ROI at offered opening price
  T7 ML ACT, goalie_change games only, gap >= 0.03: CLV prob points (lookahead upper bound: openers vs late starters)
  T8 same bets, ROI at no-vig opening price
DESCRIPTIVE (not trials): log loss model/open/close; logistic stack (logit open + logit model) vs open and vs close;
  fav/dog splits; gap dose; goalie-change vs not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from scipy.stats import poisson

from sports.market import american_to_prob, devig_two_way

ESPN = ROOT / "data" / "espn" / "nhl"
LEDGER = ROOT / "lab" / "ledger.jsonl"
OUTD = ROOT / "data" / "nhl_simple_model"
DATASET = "nhl_simple_model"
DEV = [2022, 2023, 2024, 2025]
SCORED = [2024, 2025]
assert 2026 not in DEV
MP = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL", "UTAH": "UTA"}
H_TEAM, LAM, H_G, K_G, H_S, SHOTS = 90.0, 10.0, 365.0, 1500.0, 60.0, 30.0
ML_X, TOT_X, N_TRIALS = 0.03, 0.04, 8
FEAT = ["r_gf", "g_opp", "b2b_t", "b2b_o", "S1_shots", "S2_pen", "S3_sh"]
G = np.arange(0, 16)


def lastkey(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower().replace(".", " ")
    return s.split()[-1] if s.split() else ""


# ------------------------------------------------------------------------------------------------------------- data
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    g = pd.concat([pd.read_parquet(ESPN / f"games_{y}.parquet").assign(season=y) for y in DEV], ignore_index=True)
    g = g[(g.season_type == "regular-season") & (g.status == "STATUS_FINAL")].copy()
    g["home"], g["away"] = g.home.replace(MP), g.away.replace(MP)
    g["date"] = g.start_utc.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    n = pq.read_table(ROOT / "data" / "nhl_games.parquet",
                      columns=["date", "home", "away", "home_starter", "away_starter", "overtime", "home_score", "away_score"],
                      filters=[("date", "<", pd.Timestamp("2025-08-01"))]).to_pandas()
    assert n.date.max() < pd.Timestamp("2025-08-01")
    g = g.merge(n.rename(columns={"home_score": "hs_n", "away_score": "as_n"}), on=["date", "home", "away"], how="inner")
    assert (g.hs_n == g.home_score).all() and (g.as_n == g.away_score).all()
    g["hr"] = g.home_score - (g.overtime & (g.home_score > g.away_score)).astype(int)
    g["ar"] = g.away_score - (g.overtime & (g.away_score > g.home_score)).astype(int)
    assert (~g.overtime | (g.hr == g.ar)).all()
    g = g.sort_values(["date", "event_id"]).reset_index(drop=True)

    box = pd.concat([pd.read_parquet(ESPN / f"box_{y}.parquet") for y in DEV], ignore_index=True)
    box = box[box.event_id.isin(g.event_id)]
    gl = box[box.group == "goalies"].pivot_table(index=["event_id", "team_id", "athlete_id", "name"], columns="stat",
                                                 values="value", aggfunc="first").reset_index()
    sk = box[box.group != "goalies"].pivot_table(index=["event_id", "team_id"], columns="stat", values="value",
                                                 aggfunc="sum")[["shotsTotal", "penaltyMinutes", "goals"]].reset_index()
    return g, gl, sk


def team_rows(g, gl, sk) -> pd.DataFrame:
    """Two rows per game: team t vs opp o, with box aggregates and starters (ESPN athlete ids)."""
    tid = pd.concat([g[["event_id", "home", "home_team_id"]].rename(columns={"home": "team", "home_team_id": "team_id"}),
                     g[["event_id", "away", "away_team_id"]].rename(columns={"away": "team", "away_team_id": "team_id"})])
    gl = gl.merge(tid, on=["event_id", "team_id"])
    gl["shots_ag"] = gl.shotsAgainst.fillna(0)
    gl["sv"] = gl.saves.fillna(0)
    gl["key"] = gl.name.map(lastkey)
    rows = []
    for side, o in (("home", "away"), ("away", "home")):
        r = g[["event_id", "season", "date", side, o, f"{side}_starter", f"{o}_starter"]].copy()
        r.columns = ["event_id", "season", "date", "team", "opp", "starter_name", "opp_starter_name"]
        r["home_ind"] = int(side == "home")
        r["goals"] = g["hr" if side == "home" else "ar"].to_numpy()
        r["goals_final"] = g[f"{side}_score"].to_numpy()
        rows.append(r)
    t = pd.concat(rows, ignore_index=True)
    # starter id: match nhl_games starter last name to ESPN goalie on that team; else most shots faced
    gl_s = gl.sort_values("shots_ag", ascending=False)
    first = gl_s.drop_duplicates(["event_id", "team"]).set_index(["event_id", "team"]).athlete_id
    bykey = gl.drop_duplicates(["event_id", "team", "key"]).set_index(["event_id", "team", "key"]).athlete_id
    k = list(zip(t.event_id, t.team, t.starter_name.map(lastkey)))
    t["starter"] = [bykey.get(x, first.get(x[:2])) for x in k]
    t["starter_matched"] = [x in bykey.index for x in k]
    # shots for = opp goalies' shots against; PIM; shots against
    sa = gl.groupby(["event_id", "team"]).agg(shots_ag=("shots_ag", "sum"), ga=("goalsAgainst", "sum")).reset_index()
    t = t.merge(sa, on=["event_id", "team"], how="left")
    t = t.merge(sa.rename(columns={"team": "opp", "shots_ag": "shots_for", "ga": "ga_opp"}), on=["event_id", "opp"], how="left")
    sk = sk.merge(tid, on=["event_id", "team_id"])
    t = t.merge(sk[["event_id", "team", "penaltyMinutes"]].rename(columns={"penaltyMinutes": "pim"}), on=["event_id", "team"], how="left")
    t = t.merge(sk[["event_id", "team", "penaltyMinutes"]].rename(columns={"team": "opp", "penaltyMinutes": "pim_drawn"}),
                on=["event_id", "opp"], how="left")
    return t.sort_values(["date", "event_id", "home_ind"]).reset_index(drop=True), gl


# --------------------------------------------------------------------------------------------------------- features
def ridge_gf(t: pd.DataFrame) -> np.ndarray:
    teams = sorted(set(t.team) | set(t.opp))
    ti = {k: i for i, k in enumerate(teams)}
    nT = len(teams)
    X = np.zeros((len(t), 2 * nT + 2))
    X[np.arange(len(t)), t.team.map(ti)] = 1
    X[np.arange(len(t)), nT + t.opp.map(ti).to_numpy()] = 1
    X[:, -2] = t.home_ind
    X[:, -1] = 1
    y = t.goals.to_numpy(float)
    dn = t.date.map(pd.Timestamp.toordinal).to_numpy()
    pen = np.r_[np.full(2 * nT, LAM), 1e-6, 1e-6]
    out = np.full(len(t), np.nan)
    for d in np.unique(dn):
        te, tr = dn == d, dn < d
        if tr.sum() < 200:
            continue
        w = 0.5 ** ((d - dn[tr]) / H_TEAM)
        Xt = X[tr]
        beta = np.linalg.solve((Xt * w[:, None]).T @ Xt + np.diag(pen), (Xt * w[:, None]).T @ y[tr])
        out[te] = X[te] @ beta
    return out


def goalie_ratings(gl: pd.DataFrame, dates: np.ndarray) -> dict:
    """{date_ordinal: {athlete_id: rating}} using goalie games strictly before each date."""
    gl = gl.copy()
    dn = gl.date.map(pd.Timestamp.toordinal).to_numpy()
    out = {}
    for d in dates:
        m = dn < d
        if m.sum() < 100:
            out[d] = {}
            continue
        w = 0.5 ** ((d - dn[m]) / H_G)
        sv, sh = gl.sv.to_numpy()[m], gl.shots_ag.to_numpy()[m]
        lg = (w * sv).sum() / (w * sh).sum()
        df = pd.DataFrame({"a": gl.athlete_id.to_numpy()[m], "g": w * (sv - sh * lg), "s": w * sh})
        s = df.groupby("a").sum()
        out[d] = (s.g / (s.s + K_G)).to_dict()
    return out


def decayed_team(t: pd.DataFrame, col: str, by: str, H: float, denom: str | None = None) -> np.ndarray:
    """decayed mean per game (or ratio sum col / sum denom) of `col` for entity `by`, games strictly before date."""
    out = np.full(len(t), np.nan)
    dn = t.date.map(pd.Timestamp.toordinal).to_numpy()
    for k, idx in t.groupby(by).indices.items():
        idx = np.sort(idx)
        d = dn[idx]
        v = t[col].to_numpy(float)[idx]
        den = t[denom].to_numpy(float)[idx] if denom else np.ones(len(idx))
        for j, i in enumerate(idx):
            m = d < d[j]
            if m.sum() < 3:
                continue
            w = 0.5 ** ((d[j] - d[m]) / H)
            out[i] = (w * v[m]).sum() / (w * den[m]).sum()
    return out


def league_decayed(t: pd.DataFrame, num: str, den: str, H: float) -> np.ndarray:
    dn = t.date.map(pd.Timestamp.toordinal).to_numpy()
    daily = pd.DataFrame({"d": dn, "n": t[num].to_numpy(float), "q": t[den].to_numpy(float)}).groupby("d").sum()
    dd, nn, qq = daily.index.to_numpy(), daily.n.to_numpy(), daily.q.to_numpy()
    val = {}
    for d in dd:
        m = dd < d
        w = 0.5 ** ((d - dd[m]) / H)
        val[d] = (w * nn[m]).sum() / (w * qq[m]).sum() if m.any() else np.nan
    return np.array([val[d] for d in dn])


def build_frame() -> pd.DataFrame:
    g, gl, sk = load()
    t, gl = team_rows(g, gl, sk)
    gl = gl.merge(g[["event_id", "date"]], on="event_id")
    t["r_gf"] = ridge_gf(t)
    dates = np.unique(t.date.map(pd.Timestamp.toordinal))
    R = goalie_ratings(gl, dates)
    # projected starter: most starts in the team's last 10 games (tie -> most recent)
    proj = []
    for team, idx in t.groupby("team").indices.items():
        idx = np.sort(idx)
        st = t.starter.to_numpy()[idx]
        for j, i in enumerate(idx):
            last = list(st[max(0, j - 10):j])
            if not last:
                proj.append((i, None))
                continue
            cnt = {a: last.count(a) for a in last}
            best = max(cnt.values())
            cand = [a for a in reversed(last) if cnt[a] == best][0]
            proj.append((i, cand))
    pj = dict(proj)
    t["proj_starter"] = [pj[i] for i in range(len(t))]
    # descriptive only (added after build 1, before market join): share of the team's last 10 starts by tonight's starter
    sh = {}
    for team, idx in t.groupby("team").indices.items():
        idx = np.sort(idx)
        st = t.starter.to_numpy()[idx]
        for j, i in enumerate(idx):
            last = list(st[max(0, j - 10):j])
            sh[i] = last.count(st[j]) / len(last) if last else np.nan
    t["starter_share10"] = [sh[i] for i in range(len(t))]
    dn = t.date.map(pd.Timestamp.toordinal).to_numpy()
    t["g_rating_act"] = [R[d].get(a, 0.0) for d, a in zip(dn, t.starter)]
    t["g_rating_strict"] = [R[d].get(a, 0.0) if a is not None else 0.0 for d, a in zip(dn, t.proj_starter)]
    # style
    t["sf_pg"] = decayed_team(t, "shots_for", "team", H_S)
    t["sa_pg"] = decayed_team(t, "shots_ag", "team", H_S)
    t["pim_pg"] = decayed_team(t, "pim", "team", H_S)
    t["pimd_pg"] = decayed_team(t, "pim_drawn", "team", H_S)
    t["gf_tot"] = t.goals_final
    t["shpct"] = decayed_team(t, "gf_tot", "team", H_S, denom="shots_for")
    # A1: express style raw features relative to the league's decayed mean before the date
    t["one"] = 1.0
    for c, num, den in (("sf_pg", "shots_for", "one"), ("sa_pg", "shots_ag", "one"), ("pim_pg", "pim", "one"),
                        ("pimd_pg", "pim_drawn", "one"), ("shpct", "gf_tot", "shots_for")):
        t[c] = t[c] - league_decayed(t, num, den, H_S)
    # rest
    prev = t.groupby("team").date.shift(1)
    t["b2b"] = ((t.date - prev).dt.days == 1).astype(int)
    # attach opponent's columns
    oc = t[["event_id", "team", "g_rating_act", "g_rating_strict", "sa_pg", "pim_pg", "b2b", "starter", "proj_starter",
            "starter_share10"]]
    oc = oc.rename(columns={c: "o_" + c for c in oc.columns if c not in ("event_id",)}).rename(columns={"o_team": "opp"})
    t = t.merge(oc, on=["event_id", "opp"], how="left")
    t["b2b_t"], t["b2b_o"] = t.b2b, t.o_b2b
    t["g_opp_act"] = t.o_g_rating_act * SHOTS
    t["g_opp_strict"] = t.o_g_rating_strict * SHOTS
    t["gchange"] = (t.starter != t.proj_starter) | (t.o_starter != t.o_proj_starter)
    t["rare"] = (t.starter_share10 <= 0.2) | (t.o_starter_share10 <= 0.2)
    return t, g


def zfit(tr: pd.DataFrame) -> dict:
    cols = ["sf_pg", "sa_pg", "pim_pg", "pimd_pg", "shpct"]
    return {c: (float(tr[c].mean()), float(tr[c].std())) for c in cols}


def style(t: pd.DataFrame, zc: dict) -> pd.DataFrame:
    z = lambda s, c: (s - zc[c][0]) / zc[c][1]
    t = t.copy()
    t["S1_shots"] = z(t.sf_pg, "sf_pg") + z(t.o_sa_pg, "sa_pg")
    t["S2_pen"] = z(t.o_pim_pg, "pim_pg") + z(t.pimd_pg, "pimd_pg")
    t["S3_sh"] = z(t.shpct, "shpct")
    return t


def walk_forward(t: pd.DataFrame) -> pd.DataFrame:
    t = t.dropna(subset=["r_gf", "sf_pg", "o_sa_pg", "pim_pg", "o_pim_pg", "pimd_pg", "shpct"]).copy()
    preds, fits = [], {}
    for s in SCORED:
        tr = t[(t.season >= 2023) & (t.season < s)]
        zc = zfit(tr)
        trs = style(tr, zc).assign(g_opp=lambda d: d.g_opp_act)
        X = sm.add_constant(trs[FEAT].to_numpy(float))
        f = sm.OLS(trs.goals.to_numpy(float), X).fit()
        fits[s] = f
        te = style(t[t.season == s], zc)
        for v in ("act", "strict"):
            Xe = sm.add_constant(te.assign(g_opp=te[f"g_opp_{v}"])[FEAT].to_numpy(float), has_constant="add")
            te[f"lam_{v}"] = np.maximum(Xe @ f.params, 0.5)
        preds.append(te)
    return pd.concat(preds), fits


# ------------------------------------------------------------------------------------------------------- probability
def probs(lh, la, line=None):
    ph, pa = poisson.pmf(G[None, :], np.asarray(lh)[:, None]), poisson.pmf(G[None, :], np.asarray(la)[:, None])
    J = ph[:, :, None] * pa[:, None, :]                       # [n, h, a]
    H, A = np.meshgrid(G, G, indexing="ij")
    p_home = (J * (H > A)).sum((1, 2)) + 0.5 * (J * (H == A)).sum((1, 2))
    if line is None:
        return p_home, None
    T = H + A + (H == A)
    L = np.asarray(line)[:, None, None]
    po, pu = (J * (T > L)).sum((1, 2)), (J * (T < L)).sum((1, 2))
    return p_home, po / (po + pu)


def implied_mu(line, p_over):
    """mean total mu with lh = la = mu/2 such that model P(over|no push) = p_over (bisection, vectorised)."""
    lo, hi = np.full(len(line), 2.0), np.full(len(line), 11.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        _, po = probs(mid / 2, mid / 2, line)
        lo, hi = np.where(po < p_over, mid, lo), np.where(po < p_over, hi, mid)
    return (lo + hi) / 2


def game_frame(t: pd.DataFrame) -> pd.DataFrame:
    h = t[t.home_ind == 1].set_index("event_id")
    a = t[t.home_ind == 0].set_index("event_id")
    d = pd.DataFrame({"season": h.season, "date": h.date, "home": h.team, "away": h.opp,
                      "lh_act": h.lam_act, "la_act": a.lam_act, "lh_strict": h.lam_strict, "la_strict": a.lam_strict,
                      "hs": h.goals_final, "as_": a.goals_final, "gchange": h.gchange, "rare": h.rare}).dropna(subset=["la_act"])
    d["home_win"] = (d.hs > d.as_).astype(int)
    d["total"] = d.hs + d.as_
    for v in ("act", "strict"):
        d[f"ph_{v}"], _ = probs(d[f"lh_{v}"], d[f"la_{v}"])
    return d.reset_index()


# -------------------------------------------------------------------------------------------------------- commands
def cmd_build():
    t, g = build_frame()
    print(f"team-games {len(t)}; starter name matched {t.starter_matched.mean():.3f}; goalie-change games "
          f"{t[t.home_ind == 1].gchange.mean():.3f}")
    p, fits = walk_forward(t)
    for s, f in fits.items():
        print(f"\nstack for scored season {s} (n {int(f.nobs)}, resid SD {np.sqrt(f.scale):.3f})")
        for n_, b, se in zip(["const"] + FEAT, f.params, f.bse):
            print(f"  {n_:10s} {b:+.4f} ({se:.4f})")
    d = game_frame(p)
    for v in ("act", "strict"):
        pr = np.clip(d[f"ph_{v}"], 1e-6, 1 - 1e-6)
        ll = -np.mean(d.home_win * np.log(pr) + (1 - d.home_win) * np.log(1 - pr))
        print(f"{v}: games {len(d)} log loss {ll:.4f}  mean model total {(d[f'lh_{v}'] + d[f'la_{v}']).mean():.3f} "
              f"vs actual {d.total.mean():.3f}; home win {d.home_win.mean():.3f} vs model {d[f'ph_{v}'].mean():.3f}")
    OUTD.mkdir(exist_ok=True, parents=True)
    d.to_pickle(OUTD / "wf_games.pkl")
    json.dump({str(s): {"names": ["const"] + FEAT, "params": list(map(float, f.params)), "bse": list(map(float, f.bse)),
                        "n": int(f.nobs), "sigma": float(np.sqrt(f.scale))} for s, f in fits.items()},
              open(OUTD / "stack_fits.json", "w"), indent=1)
    return 0


def markets() -> pd.DataFrame:
    rows = []
    for y in SCORED:
        ln = pd.read_parquet(ESPN / f"lines_{y}.parquet")
        o = ln[(ln.provider == "ESPN BET")].dropna(subset=["home_ml_open"]).drop_duplicates("event_id")
        rows.append(o)
    m = pd.concat(rows)
    m["p_open"] = devig_two_way(m.home_ml_open, m.away_ml_open)[0]
    m["p_close"] = devig_two_way(m.home_ml_close, m.away_ml_close)[0]
    m["po_open"] = devig_two_way(m.over_odds_open, m.under_odds_open)[0]
    m["po_close"] = devig_two_way(m.over_odds_close, m.under_odds_close)[0]
    return m[["event_id", "p_open", "p_close", "po_open", "po_close", "home_ml_open", "away_ml_open", "total_open",
              "total_close", "over_odds_open", "under_odds_open"]]


def dec(am):
    am = np.asarray(am, float)
    return np.where(am > 0, 1 + am / 100, 1 + 100 / np.abs(am))


def boot(v, season, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0:
        return {"bets": 0, "roi": float("nan"), "p": 1.0}
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (4000, len(v)))].mean(1)
    per = pd.Series(v, index=season.index).groupby(season).agg(["mean", "size"])
    return {"bets": int(len(v)), "roi": float(v.mean()), "ci_lo": float(np.quantile(bs, .025)),
            "ci_hi": float(np.quantile(bs, .975)), "p": float(np.mean(bs <= 0)),
            "folds_positive": int((per["mean"] > 0).sum()), "folds": int(len(per)),
            "per_fold": {str(k): [round(float(a), 4), int(b)] for k, (a, b) in per.iterrows()}}


def ml_bets(d, v, thr):
    x = d.dropna(subset=["p_open", "p_close"]).copy()
    gap = x[f"ph_{v}"] - x.p_open
    x = x[gap.abs() >= thr].copy()
    side = np.sign(gap[x.index])
    x["side"] = side
    pf = np.where(side > 0, x.p_open, 1 - x.p_open)
    win = np.where(side > 0, x.home_win == 1, x.home_win == 0)
    odds = np.where(side > 0, dec(x.home_ml_open), dec(x.away_ml_open))
    x["clv"] = side * (x.p_close - x.p_open)
    x["pnl_novig"] = np.where(win, 1 / pf - 1, -1.0)
    x["pnl_off"] = np.where(win, odds - 1, -1.0)
    x["fav"] = pf > 0.5
    x["win"] = win.astype(float)
    return x


def tot_bets(d, v, thr):
    x = d.dropna(subset=["po_open", "po_close", "total_open", "total_close"]).copy()
    _, pm = probs(x[f"lh_{v}"], x[f"la_{v}"], x.total_open.to_numpy())
    x["pm_over"] = pm
    gap = pm - x.po_open
    x = x[np.abs(gap) >= thr].copy()
    side = np.sign(gap[x.index])
    x["side"] = side
    mu_o = implied_mu(x.total_open.to_numpy(), x.po_open.to_numpy())
    mu_c = implied_mu(x.total_close.to_numpy(), x.po_close.to_numpy())
    x["clv"] = side * (mu_c - mu_o)
    res = side * (x.total - x.total_open)
    pf = np.where(side > 0, x.po_open, 1 - x.po_open)
    odds = np.where(side > 0, dec(x.over_odds_open), dec(x.under_odds_open))
    x["pnl_novig"] = np.where(res > 0, 1 / pf - 1, np.where(res == 0, 0.0, -1.0))
    x["pnl_off"] = np.where(res > 0, odds - 1, np.where(res == 0, 0.0, -1.0))
    x["win"] = (res > 0).astype(float)
    x["push"] = res == 0
    x["fav"] = side > 0                       # 'fav' = over for totals
    return x


def cmd_backtest(ledger=False):
    d = pd.read_pickle(OUTD / "wf_games.pkl")
    d = d.merge(markets(), on="event_id", how="left")
    d["gchange"] = d.gchange.fillna(False).astype(bool)
    d["rare"] = d.rare.fillna(False).astype(bool)
    print(f"scored games {len(d)}; with ESPN BET ML open {d.p_open.notna().sum()}; total open {d.po_open.notna().sum()}")
    bar = 0.05 / N_TRIALS
    res = {"trials": {}, "descriptive": {}}
    mlS, totS = ml_bets(d, "strict", ML_X), tot_bets(d, "strict", TOT_X)
    gc = ml_bets(d[d.gchange], "act", ML_X)
    spec = [("T1_ml_strict_clv", mlS, "clv"), ("T2_ml_strict_novig", mlS, "pnl_novig"), ("T3_ml_strict_offered", mlS, "pnl_off"),
            ("T4_tot_strict_clv_goals", totS, "clv"), ("T5_tot_strict_novig", totS, "pnl_novig"),
            ("T6_tot_strict_offered", totS, "pnl_off"), ("T7_ml_act_gchange_clv", gc, "clv"),
            ("T8_ml_act_gchange_novig", gc, "pnl_novig")]
    recs = []
    for name, x, metric in spec:
        r = boot(x[metric], x.season)
        r["win_rate"] = float(x.win.mean()) if len(x) else float("nan")
        r["fav_dog"] = {"fav_or_over": [float(x[x.fav][metric].mean()), int(x.fav.sum())],
                        "dog_or_under": [float(x[~x.fav][metric].mean()), int((~x.fav).sum())]}
        r["passed"] = bool(r["p"] < bar and r["roi"] > 0)
        res["trials"][name] = r
        print(f"  {name:26s} n {r['bets']:5d} mean {r['roi']:+.4f} CI [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p {r['p']:.4f} "
              f"win {r['win_rate']:.3f} seasons {r['per_fold']} fav/dog {r['fav_dog']} {'PASS' if r['passed'] else ''}")
        sha = hashlib.sha256((DATASET + ":" + name).encode()).hexdigest()[:12]
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"nhl_simple_{name}",
                     "sha": sha, "dataset": DATASET, "metric": metric, "trial": name,
                     "hypothesis": f"pre-registered in scripts/nhl_simple_model.py ({name})",
                     "source": "simple NHL team+goalie+style model vs ESPN BET opener (2026-10-25)", "status": "ok",
                     "trial_number": len(recs) + 1, "study_trials": N_TRIALS, "bonferroni_bar": bar, **r})

    print("\nDESCRIPTIVE (not trials)")
    x = d.dropna(subset=["p_open", "p_close", "ph_strict", "ph_act"])
    lg =lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    ll = lambda p: float(-np.mean(x.home_win * np.log(np.clip(p, 1e-6, 1)) + (1 - x.home_win) * np.log(np.clip(1 - p, 1e-6, 1))))
    desc = {"logloss": {k: ll(x[c]) for k, c in (("strict", "ph_strict"), ("act", "ph_act"), ("open", "p_open"), ("close", "p_close"))}}
    print("  log loss", {k: round(v, 4) for k, v in desc["logloss"].items()}, "n", len(x))
    for ref in ("p_open", "p_close"):
        for v in ("strict", "act"):
            f = sm.Logit(x.home_win.to_numpy(float), sm.add_constant(np.column_stack([lg(x[ref]), lg(x[f"ph_{v}"])]))).fit(disp=0)
            ci = f.conf_int()[2]
            desc[f"stack_{ref}_{v}"] = [float(f.params[2]), float(ci[0]), float(ci[1])]
            print(f"  logit stack given {ref:7s} model {v:6s}: b_model {f.params[2]:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  b_mkt {f.params[1]:+.3f}")
    y = d.dropna(subset=["po_open", "total_open", "lh_strict", "la_strict", "lh_act", "la_act"])
    for v in ("strict", "act"):
        _, pm = probs(y[f"lh_{v}"], y[f"la_{v}"], y.total_open.to_numpy())
        nz = y.total != y.total_open
        ov = (y.total > y.total_open)[nz].astype(float)
        f = sm.Logit(ov.to_numpy(), sm.add_constant(np.column_stack([lg(y.po_open[nz]), lg(pm[nz.to_numpy()])]))).fit(disp=0)
        ci = f.conf_int()[2]
        desc[f"tot_stack_open_{v}"] = [float(f.params[2]), float(ci[0]), float(ci[1])]
        print(f"  totals logit stack given open, model {v}: b_model {f.params[2]:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]; "
              f"mean model total {(y[f'lh_{v}'] + y[f'la_{v}']).mean():.3f} actual {y.total.mean():.3f}")
    for thr in (0.02, 0.03, 0.05, 0.07):
        m1 = ml_bets(d, "strict", thr)
        print(f"  dose ML strict gap>={thr:.2f}: n {len(m1):4d} CLV {m1.clv.mean():+.4f} novig {m1.pnl_novig.mean():+.4f} "
              f"offered {m1.pnl_off.mean():+.4f} dog share {(~m1.fav).mean():.2f}")
    for thr in (0.02, 0.04, 0.06, 0.08):
        t1 = tot_bets(d, "strict", thr)
        print(f"  dose TOT strict gap>={thr:.2f}: n {len(t1):4d} CLV {t1.clv.mean():+.4f} novig {t1.pnl_novig.mean():+.4f} "
              f"offered {t1.pnl_off.mean():+.4f} over share {t1.fav.mean():.2f}")
    for lab, dd in (("gchange", d[d.gchange]), ("no-change", d[~d.gchange]), ("rare<=2/10", d[d.rare]),
                    ("not rare", d[~d.rare])):
        for v in ("act", "strict"):
            m1 = ml_bets(dd, v, ML_X)
            print(f"  ML {v:6s} {lab:9s}: n {len(m1):4d} CLV {m1.clv.mean():+.4f} novig {m1.pnl_novig.mean():+.4f}")
    a = ml_bets(d, "act", ML_X)
    ta = tot_bets(d, "act", TOT_X)
    print(f"  ML act all: n {len(a)} CLV {a.clv.mean():+.4f} novig {a.pnl_novig.mean():+.4f}; "
          f"TOT act all: n {len(ta)} CLV {ta.clv.mean():+.4f} novig {ta.pnl_novig.mean():+.4f}")
    print(f"  goalie-change share of scored games {d.gchange.mean():.3f}")
    res["descriptive"] = desc
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "nhl_simple_model.json").write_text(json.dumps(res, indent=1, default=float))
    if ledger:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=float) + "\n")
        print(f"appended {len(recs)} trials to {LEDGER}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "backtest"])
    ap.add_argument("--ledger", action="store_true")
    a = ap.parse_args()
    return cmd_build() if a.cmd == "build" else cmd_backtest(a.ledger)


if __name__ == "__main__":
    sys.exit(main())
