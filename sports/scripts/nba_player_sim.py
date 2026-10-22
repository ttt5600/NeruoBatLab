"""NBA game models: SIMPLE ridge team rating vs PLAYER SIMULATION ("the RotoBot way"), scored vs OPEN and CLOSE.

    ../.venv_quant/bin/python scripts/nba_player_sim.py [--ledger]

User 2026-10-22: "Simulating player scores might be a good way to predict games (the RotoBot way). Otherwise plain
analytics could work well too; simple models often beat convoluted ones."

PRE-REGISTERED 2026-10-22, written BEFORE either model's probability was joined to any market price.

DATA (dev only): data/espn/nba/{games,box,lines}_{2022..2025}.parquet = seasons 2021-22..2024-25.
NBA 2026 (= 2025-26) is SEALED and is never read by this script (ESPN_DEV below; asserted).
Games: STATUS_FINAL, not preseason, 30 real teams (all-star rows dropped). Ratings learn from regular season +
play-in + playoffs; SCORED games = regular season, not neutral site, both teams >= 5 earlier games that season.
All state is updated in ET-date batches: every game on date D is predicted from games dated < D only.

MODEL A  SIMPLE (ridge team rating):
  margin_home = h*home + r_home - r_away + b1*b2b_home + b2*b2b_away + b3*restdiff(cap 3) + b4*travel_home_1000km
                + b5*travel_away_1000km + e. Weighted ridge, weights 0.5^(age_days/H), penalty lam on the 30 team
  ratings only. Refit every date on all earlier games. P(home) = Phi(mu / sigma).
  Tuned on season 2022 games dated >= 2021-12-01 ONLY: H in {30, 60, 120, 240} days, lam in {1, 5, 20} (min MAE
  of margin), then sigma by min log loss. Frozen for 2023-2025.

MODEL B  PLAYER SIM:
  Candidate roster for team T = players whose most recent appearance (any team) was for T and who appeared in at
  least one of T's last 10 games (pregame-known). Availability P(play | k) where k = consecutive T-games missed
  (buckets 0,1,2,3-5,6-9), table estimated on season 2022 only. Expected minutes when playing = EWMA(0.3) of the
  player's past minutes. Points per minute = decayed (half-life Hp days) and shrunk to the league mean with k_min
  prior minutes. Defense adjustment per team = decayed mean of (points allowed - opponent's pregame expected
  points), shrunk with 5 pseudo-games. Home = constant fitted on 2022.
  Each of N = 2000 sims: Bernoulli availability per player -> minutes rescaled to 240 (individual cap 44) ->
  player points = 2*Poisson(rate*min/2) -> team pts = sum + defense adj + home/2 (+-) + team shock N(0, s_team).
  P(home) = share of sims with margin > 0 (ties 1/2). Tuned on 2022 (>= 2021-12-01): Hp in {60, 180},
  k_min in {100, 300} by margin MAE; then s_team in {6, 8, 10, 12} by log loss. Frozen.
  LIMITATION (stated up front): no historical injury feed. ESPN's summary `injuries` block for past games returns
  the CURRENT injury list (checked 2026-10-22 on event 401656363: entries dated 2026-10-18), so it cannot be used.
  The sim knows only who played recently; a player who sat last game and returns, or a late scratch, is missed.
  Using who ACTUALLY played in the scored game would be a leak and is not done.

MARKETS (proportional de-vig, overround in [0, 0.10]):
  CLOSE: real books (drop accuscore/teamrankings/betegy/consensus) median de-vig of home_ml/away_ml (= close,
         see notes/nba_public_models.md); bet price = median decimal odds over books. Seasons 2023-2025.
  OPEN : ESPN BET home_ml_open/away_ml_open (the only book with openers in both 2024 and 2025); bet at that opening
         price; CLV reference = ESPN BET home_ml_close/away_ml_close de-vig. Seasons 2024-2025.
         The time ESPN records as "open" is not documented (UNVERIFIED that it is the first posted line).

TRIALS (dataset nba_player_sim; 6 trials -> bar one-sided p < 0.05/6 = 0.0083; flat 1u, one bet per game, side =
whichever team the model rates >= x above the market, x = 0.04; one-sided bootstrap over games, 4000 resamples):
  S1 simple_open_roi : model A vs open, ROI at opening price          pass = p < bar and ROI > 0
  S2 sim_open_roi    : model B vs open, ROI at opening price
  S3 simple_open_clv : same bets as S1; metric = de-vig close prob of bet side - de-vig open prob of bet side
  S4 sim_open_clv    : same for S2                                     pass = p(mean CLV <= 0) < bar
  S5 simple_close_roi: model A vs median close, ROI at median closing price
  S6 sim_close_roi   : model B vs close
DESCRIPTIVE (not trials): log loss / Brier of A, B, open, close on the same games with paired bootstrap; margin MAE
vs opening/closing spread; stacking coefficient y ~ logit(mkt) + logit(model); corr(model - open, close - open).
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from sports.market import american_to_decimal, brier, devig_two_way, log_loss, paired_logloss_bootstrap

ESPN = ROOT / "data" / "espn" / "nba"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nba_player_sim"
ESPN_DEV = [2022, 2023, 2024, 2025]
assert 2026 not in ESPN_DEV
NOT_BOOKS = {"consensus", "accuscore", "teamrankings", "betegy"}
N_TRIALS, X_THR, N_SIM = 6, 0.04, 2000
TUNE_FROM = pd.Timestamp("2021-12-01").date()

# approximate arena-city coordinates (lat, lon)
LOC = {"ATL": (33.76, -84.40), "BOS": (42.37, -71.06), "BKN": (40.68, -73.98), "CHA": (35.23, -80.84),
       "CHI": (41.88, -87.67), "CLE": (41.50, -81.69), "DAL": (32.79, -96.81), "DEN": (39.75, -105.01),
       "DET": (42.34, -83.06), "GS": (37.77, -122.39), "HOU": (29.75, -95.36), "IND": (39.76, -86.16),
       "LAC": (34.04, -118.27), "LAL": (34.04, -118.27), "MEM": (35.14, -90.05), "MIA": (25.78, -80.19),
       "MIL": (43.04, -87.92), "MIN": (44.98, -93.28), "NO": (29.95, -90.08), "NY": (40.75, -73.99),
       "OKC": (35.46, -97.52), "ORL": (28.54, -81.38), "PHI": (39.90, -75.17), "PHX": (33.45, -112.07),
       "POR": (45.53, -122.67), "SAC": (38.58, -121.50), "SA": (29.43, -98.44), "TOR": (43.64, -79.38),
       "UTAH": (40.77, -111.90), "WSH": (38.90, -77.02)}
CITY = {"Paris": (48.86, 2.35), "Mexico City": (19.43, -99.13), "Las Vegas": (36.17, -115.14)}
TEAMS = sorted(LOC)
TI = {t: i for i, t in enumerate(TEAMS)}


def km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


# ----------------------------------------------------------------------------------------------------------- data
def load_games() -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(ESPN / f"games_{y}.parquet").assign(season=y) for y in ESPN_DEV], ignore_index=True)
    g = g[(g.status == "STATUS_FINAL") & (g.season_type != "preseason") & g.home.isin(LOC) & g.away.isin(LOC)].copy()
    g["date"] = g.start_utc.dt.tz_convert("US/Eastern").dt.date
    g = g.sort_values("start_utc").reset_index(drop=True)
    g["margin"] = g.home_score - g.away_score
    g["loc"] = [CITY.get(c, LOC[h]) if n else LOC[h] for c, h, n in zip(g.city, g.home, g.neutral.astype(bool))]
    # rest / b2b / travel per team, from the team's previous non-preseason game (pregame-known schedule)
    last = {}
    for side in ("home", "away"):
        g[f"rest_{side}"], g[f"trav_{side}"] = np.nan, np.nan
    for i, r in g.iterrows():
        for side in ("home", "away"):
            t = r[side]
            if t in last and last[t][2] == r.season:
                d0, l0, _ = last[t]
                g.at[i, f"rest_{side}"] = min((r.date - d0).days, 4)
                g.at[i, f"trav_{side}"] = km(l0, r["loc"]) / 1000
            else:
                g.at[i, f"rest_{side}"], g.at[i, f"trav_{side}"] = 4, 0.0
            last[t] = (r.date, r["loc"], r.season)
    g["b2b_home"], g["b2b_away"] = (g.rest_home == 1).astype(float), (g.rest_away == 1).astype(float)
    g["restdiff"] = (g.rest_home.clip(upper=3) - g.rest_away.clip(upper=3))
    g["home_ind"] = (~g.neutral.astype(bool)).astype(float)
    # games played earlier this season by each team
    for side in ("home", "away"):
        g[f"n_{side}"] = 0
    cnt = {}
    for i, r in g.iterrows():
        for side in ("home", "away"):
            k = (r[side], r.season)
            g.at[i, f"n_{side}"] = cnt.get(k, 0)
            cnt[k] = cnt.get(k, 0) + 1
    g["scored"] = (g.season_type == "regular-season") & (g.home_ind == 1) & (g.n_home >= 5) & (g.n_away >= 5)
    return g


def load_box(g: pd.DataFrame) -> pd.DataFrame:
    b = pd.concat([pd.read_parquet(ESPN / f"box_{y}.parquet", columns=["event_id", "team_id", "athlete_id", "stat",
                                                                       "value"]) for y in ESPN_DEV])
    b = b[b.stat.isin(["minutes", "points"])]
    b = b.pivot_table(index=["event_id", "team_id", "athlete_id"], columns="stat", values="value",
                      aggfunc="first").reset_index()
    b = b.dropna(subset=["minutes", "points"])
    b = b[b.minutes > 0]
    ids = pd.concat([g[["event_id", "home_team_id", "home"]].rename(columns={"home_team_id": "team_id", "home": "team"}),
                     g[["event_id", "away_team_id", "away"]].rename(columns={"away_team_id": "team_id", "away": "team"})])
    return b.merge(ids, on=["event_id", "team_id"], how="inner")


# ----------------------------------------------------------------------------------------------------- model A
def ridge_preds(g: pd.DataFrame, H: float, lam: float, only_from=None) -> np.ndarray:
    hi, ai = g.home.map(TI).to_numpy(), g.away.map(TI).to_numpy()
    extra = g[["home_ind", "b2b_home", "b2b_away", "restdiff", "trav_home", "trav_away"]].to_numpy(float)
    X = np.zeros((len(g), 30 + extra.shape[1]))
    X[np.arange(len(g)), hi] += 1
    X[np.arange(len(g)), ai] -= 1
    X[:, 30:] = extra
    y = g.margin.to_numpy(float)
    dn = np.array([d.toordinal() for d in g.date])
    pen = np.r_[np.full(30, lam), np.full(extra.shape[1], 1e-6)]
    out = np.full(len(g), np.nan)
    for d in np.unique(dn):
        if only_from is not None and d < only_from:
            continue
        te = dn == d
        tr = dn < d
        if tr.sum() < 50:
            continue
        w = 0.5 ** ((d - dn[tr]) / H)
        Xt = X[tr]
        A = (Xt * w[:, None]).T @ Xt + np.diag(pen)
        beta = np.linalg.solve(A, (Xt * w[:, None]).T @ y[tr])
        out[te] = X[te] @ beta
    return out


# ----------------------------------------------------------------------------------------------------- model B
BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 9)]


def bucket(k):
    for j, (a, b) in enumerate(BUCKETS):
        if a <= k <= b:
            return j
    return None


class Sim:
    def __init__(self, Hp, kmin, s_team, home, pplay, n=N_SIM, seed=0):
        self.Hp, self.kmin, self.s_team, self.home, self.pplay, self.n = Hp, kmin, s_team, home, pplay, n
        self.rng = np.random.default_rng(seed)
        self.P = {}          # athlete -> dict(smin, spts, t, em, team, last_tidx)
        self.T = {}          # team -> games played count (index)
        self.D = {}          # team -> (sres, sw, t)
        self.lg = [0.0, 0.0]  # league decayed pts, min

    def _decay(self, t0, t):
        return 0.5 ** ((t - t0) / self.Hp)

    def roster(self, team, t):
        n = self.T.get(team, 0)
        lg_rate = self.lg[0] / self.lg[1] if self.lg[1] > 0 else 0.46
        out = []
        for a, s in self.P.items():
            if s["team"] != team:
                continue
            k = n - s["last_tidx"] - 1
            j = bucket(k)
            if j is None:
                continue
            dec = self._decay(s["t"], t)
            smin, spts = s["smin"] * dec, s["spts"] * dec
            rate = (spts + self.kmin * lg_rate) / (smin + self.kmin)
            out.append((self.pplay[j], s["em"], rate))
        return np.array(out) if out else np.zeros((0, 3))

    def dadj(self, team, t):
        if team not in self.D:
            return 0.0
        sres, sw, t0 = self.D[team]
        dec = self._decay(t0, t)
        return sres * dec / (sw * dec + 5)

    def team_pts(self, R, n):
        """R rows (p, em, rate) -> (expected pts deterministic, simulated pts array n)."""
        if len(R) == 0:
            return 100.0, np.full(n, 100.0)
        p, em, rate = R[:, 0], R[:, 1], R[:, 2]
        avail = self.rng.random((n, len(R))) < p
        m = avail * em
        tot = m.sum(1, keepdims=True)
        m = np.minimum(m * 240 / np.maximum(tot, 1), 44)
        mu = m * rate
        pts = 2 * self.rng.poisson(mu / 2)
        # deterministic expectation (availability at its mean)
        me = p * em
        me = np.minimum(me * 240 / max(me.sum(), 1), 44)
        return float((me * rate).sum()), pts.sum(1)

    def predict(self, r, t):
        Rh, Ra = self.roster(r.home, t), self.roster(r.away, t)
        eh, ph = self.team_pts(Rh, self.n)
        ea, pa = self.team_pts(Ra, self.n)
        dh, da = self.dadj(r.away, t), self.dadj(r.home, t)   # away D adjusts home scoring, etc.
        hm = self.home * r.home_ind
        sh = ph + dh + hm / 2 + self.rng.normal(0, self.s_team, self.n)
        sa = pa + da - hm / 2 + self.rng.normal(0, self.s_team, self.n)
        mar = sh - sa
        p = float(np.mean(mar > 0) + 0.5 * np.mean(mar == 0))
        return p, float(np.mean(mar)), (eh, ea)

    def update(self, r, t, exp_pts, box_g):
        # defense residuals: points the defense allowed minus opponent's pregame expected offense
        for team, allowed, exp in ((r.away, r.home_score, exp_pts[0]), (r.home, r.away_score, exp_pts[1])):
            sres, sw, t0 = self.D.get(team, (0.0, 0.0, t))
            dec = self._decay(t0, t)
            self.D[team] = (sres * dec + (allowed - exp), sw * dec + 1, t)
        for team in (r.home, r.away):
            idx = self.T.get(team, 0)
            for a, mins, pts in box_g.get(team, []):
                s = self.P.get(a)
                if s is None:
                    s = {"smin": 0.0, "spts": 0.0, "t": t, "em": mins}
                dec = self._decay(s["t"], t)
                s["smin"], s["spts"], s["t"] = s["smin"] * dec + mins, s["spts"] * dec + pts, t
                s["em"] = 0.7 * s["em"] + 0.3 * mins
                s["team"], s["last_tidx"] = team, idx
                self.P[a] = s
                self.lg[0] += pts
                self.lg[1] += mins
            self.T[team] = idx + 1
        self.lg = [self.lg[0] * 0.999, self.lg[1] * 0.999]


def run_sim(g, box, Hp, kmin, s_team, home, pplay, n=N_SIM, stop=None):
    sim = Sim(Hp, kmin, s_team, home, pplay, n)
    bx = {eid: {} for eid in g.event_id}
    for row in box.itertuples():
        bx[row.event_id].setdefault(row.team, []).append((row.athlete_id, row.minutes, row.points))
    P, M = np.full(len(g), np.nan), np.full(len(g), np.nan)
    dn = np.array([d.toordinal() for d in g.date])
    for d in np.unique(dn):
        if stop is not None and d > stop:
            break
        idx = np.where(dn == d)[0]
        exps = {}
        for i in idx:
            r = g.iloc[i]
            P[i], M[i], exps[i] = sim.predict(r, d)
        for i in idx:
            sim.update(g.iloc[i], d, exps[i], bx[g.event_id.iat[i]])
    return P, M


def estimate_pplay(g, box):
    """P(play | k consecutive team games missed), season 2022 only."""
    g22 = g[g.season == 2022]
    played = box[box.event_id.isin(g22.event_id)].groupby(["event_id", "team"]).athlete_id.apply(set).to_dict()
    last_team, last_idx, tcount = {}, {}, {}
    hit, tot = np.zeros(len(BUCKETS)), np.zeros(len(BUCKETS))
    for r in g22.itertuples():
        for team in (r.home, r.away):
            n = tcount.get(team, 0)
            pl = played.get((r.event_id, team), set())
            for a, tm in last_team.items():
                if tm != team:
                    continue
                j = bucket(n - last_idx[a] - 1)
                if j is not None:
                    tot[j] += 1
                    hit[j] += a in pl
            for a in pl:
                last_team[a], last_idx[a] = team, n
            tcount[team] = n + 1
    return hit / tot, tot


# ------------------------------------------------------------------------------------------------------- markets
def markets(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in ESPN_DEV:
        if y < 2023:
            continue
        ln = pd.read_parquet(ESPN / f"lines_{y}.parquet")
        bk = ln[~ln.provider.isin(NOT_BOOKS)].dropna(subset=["home_ml", "away_ml"]).copy()
        ph, _, ov = devig_two_way(bk.home_ml, bk.away_ml)
        bk["p"], bk["over"] = ph, ov
        bk = bk[(bk.over >= 0) & (bk.over <= 0.10)]
        bk["dh"], bk["da"] = american_to_decimal(bk.home_ml), american_to_decimal(bk.away_ml)
        c = bk.groupby("event_id").agg(p_close=("p", "median"), dec_h=("dh", "median"), dec_a=("da", "median"),
                                       spread_close=("spread", "median")).reset_index()
        eb = ln[(ln.provider == "ESPN BET")].dropna(subset=["home_ml_open", "away_ml_open", "home_ml_close",
                                                           "away_ml_close"]).copy()
        po, _, ovo = devig_two_way(eb.home_ml_open, eb.away_ml_open)
        pc, _, ovc = devig_two_way(eb.home_ml_close, eb.away_ml_close)
        eb = eb.assign(p_open=po, p_eb_close=pc, ovo=ovo, ovc=ovc,
                       dec_oh=american_to_decimal(eb.home_ml_open), dec_oa=american_to_decimal(eb.away_ml_open))
        eb = eb[(eb.ovo >= 0) & (eb.ovo <= .10) & (eb.ovc >= 0) & (eb.ovc <= .10)]
        eb = eb.drop_duplicates("event_id")[["event_id", "p_open", "p_eb_close", "dec_oh", "dec_oa",
                                            "home_spread_open", "home_spread_close"]]
        rows.append(c.merge(eb, on="event_id", how="left"))
    return g.merge(pd.concat(rows), on="event_id", how="left")


# ------------------------------------------------------------------------------------------------------- scoring
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def pick(d, pm, pmkt):
    diff = d[pm] - d[pmkt]
    home, away = diff >= X_THR, diff <= -X_THR
    m = (home | away) & d[pm].notna() & d[pmkt].notna()
    return d[m], home[m].to_numpy()


def boot(v, season, rng):
    v = np.asarray(v, float)
    if len(v) == 0:
        return {"bets": 0}
    bs = np.array([rng.choice(v, len(v)).sum() for _ in range(4000)])
    per = pd.Series(v, index=season.index).groupby(season).sum().round(3)
    return {"bets": int(len(v)), "staked": float(len(v)), "units": float(v.sum()), "roi": float(v.mean()),
            "ci_lo": float(np.quantile(bs, .025)), "ci_hi": float(np.quantile(bs, .975)), "p": float(np.mean(bs <= 0)),
            "folds_positive": int((per > 0).sum()), "folds": int(len(per)),
            "per_fold": {str(k): float(v) for k, v in per.items()}}


def roi_trial(d, pm, pmkt, dh, da):
    x, hs = pick(d, pm, pmkt)
    won = np.where(hs, x.y == 1, x.y == 0)
    dec = np.where(hs, x[dh], x[da])
    return boot(np.where(won, dec - 1, -1.0), x.season, np.random.default_rng(0)), x, hs


def clv_trial(d, pm):
    x, hs = pick(d, pm, "p_open")
    clv = np.where(hs, x.p_eb_close - x.p_open, x.p_open - x.p_eb_close)
    return boot(clv, x.season, np.random.default_rng(0))


def describe(tag, d, pm, pmkt):
    x = d[d[pm].notna() & d[pmkt].notna()]
    b = paired_logloss_bootstrap(x[pm], x[pmkt], x.y, n_boot=2000)
    X = sm.add_constant(np.column_stack([logit(x[pmkt]), logit(x[pm])]))
    f = sm.Logit(x.y.to_numpy(), X).fit(disp=0)
    ci = f.conf_int()
    print(f"  {tag:30s} n {len(x):5d} model LL {log_loss(x[pm], x.y):.4f} Brier {brier(x[pm], x.y):.4f} | mkt LL "
          f"{log_loss(x[pmkt], x.y):.4f} Brier {brier(x[pmkt], x.y):.4f} | dLL {b['observed_diff']:+.4f} "
          f"[{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] | stack b_model {f.params[2]:+.2f} [{ci[2][0]:+.2f},{ci[2][1]:+.2f}]")


def main() -> int:
    g = load_games()
    box = load_box(g)
    chk = box.groupby(["event_id", "team"]).points.sum()
    ok = np.mean([abs(chk.get((e, h), -1) - s) < 0.5 for e, h, s in zip(g.event_id, g.home, g.home_score)])
    print(f"games {len(g)} (scored {g.scored.sum()}); box players {len(box)}; box pts == home score in {ok:.1%}")
    t0 = TUNE_FROM.toordinal()
    tune = (g.season == 2022) & (pd.Series([d.toordinal() for d in g.date]) >= t0) & g.scored

    # ---- tune A on 2022
    best = None
    for H in (30, 60, 120, 240):
        for lam in (1, 5, 20):
            mu = ridge_preds(g[g.season == 2022].reset_index(drop=True), H, lam)
            mae = np.nanmean(np.abs(mu[tune[g.season == 2022].to_numpy()] - g.margin[tune].to_numpy()))
            print(f"  A tune H {H:3d} lam {lam:2d}  MAE {mae:.3f}")
            if best is None or mae < best[0]:
                best = (mae, H, lam)
    _, H, lam = best
    mu_all = ridge_preds(g, H, lam)
    sig = min(np.arange(9, 16.01, 0.25), key=lambda s: log_loss(norm.cdf(mu_all[tune] / s), (g.margin[tune] > 0)))
    print(f"A frozen: H {H}, lam {lam}, sigma {sig}")
    g["mu_A"], g["p_A"] = mu_all, norm.cdf(mu_all / sig)

    # ---- tune B on 2022
    pplay, n_obs = estimate_pplay(g, box)
    print("B P(play | k missed) 2022:", {f"{a}-{b}": f"{p:.3f} (n {int(n)})" for (a, b), p, n in zip(BUCKETS, pplay, n_obs)})
    g22 = g[g.season == 2022].reset_index(drop=True)
    b22 = box[box.event_id.isin(g22.event_id)]
    tune22 = tune[g.season == 2022].to_numpy()
    # home constant: mean residual of sim (no home) on 2022 tuning games, default Hp/kmin
    _, M0 = run_sim(g22, b22, 180, 300, 0.0, 0.0, pplay, n=300)
    home = float(np.nanmean(g22.margin.to_numpy()[tune22] - M0[tune22]))
    print(f"B home constant (2022) {home:.2f}")
    best = None
    for Hp in (60, 180):
        for kmin in (100, 300):
            _, M = run_sim(g22, b22, Hp, kmin, 0.0, home, pplay, n=300)
            mae = np.nanmean(np.abs(M[tune22] - g22.margin.to_numpy()[tune22]))
            print(f"  B tune Hp {Hp:3d} kmin {kmin:3d}  MAE {mae:.3f}")
            if best is None or mae < best[0]:
                best = (mae, Hp, kmin)
    _, Hp, kmin = best
    best = None
    for s in (6, 8, 10, 12):
        P, _ = run_sim(g22, b22, Hp, kmin, s, home, pplay, n=1000)
        ll = log_loss(np.clip(P[tune22], .001, .999), g22.margin.to_numpy()[tune22] > 0)
        print(f"  B tune s_team {s:2d}  LL {ll:.4f}")
        if best is None or ll < best[0]:
            best = (ll, s)
    s_team = best[1]
    print(f"B frozen: Hp {Hp}, kmin {kmin}, s_team {s_team}, home {home:.2f}")
    P, M = run_sim(g, box, Hp, kmin, s_team, home, pplay, n=N_SIM)
    g["p_B"], g["mu_B"] = np.clip(P, 1 / (2 * N_SIM), 1 - 1 / (2 * N_SIM)), M

    # ---- score
    d = markets(g[g.scored & (g.season >= 2023)].copy())
    d["y"] = (d.margin > 0).astype(int)
    d = d.dropna(subset=["p_A", "p_B"])
    print(f"\nscored dev games 2023-2025: {len(d)}; with close {d.p_close.notna().sum()}; with ESPN BET open "
          f"{d.p_open.notna().sum()} (seasons {sorted(d[d.p_open.notna()].season.unique())})")
    print("ACCURACY (model margin MAE):")
    for s in (2023, 2024, 2025):
        x = d[d.season == s]
        print(f"  {s}: A {np.mean(np.abs(x.mu_A - x.margin)):.2f}  B {np.mean(np.abs(x.mu_B - x.margin)):.2f}  "
              f"close spread {np.nanmean(np.abs(-x.spread_close - x.margin)):.2f}  "
              f"open spread {np.nanmean(np.abs(-x.home_spread_open - x.margin)):.2f}  "
              f"acc A {np.mean((x.p_A > .5) == x.y):.3f} B {np.mean((x.p_B > .5) == x.y):.3f} "
              f"close {np.mean((x.p_close > .5) == x.y):.3f}")
    print("DESCRIPTIVE log loss vs market on the same games (dLL > 0 = model worse):")
    for pm in ("p_A", "p_B"):
        describe(f"{pm} vs close 2023-25", d, pm, "p_close")
        describe(f"{pm} vs ESPN BET open 2024-25", d, pm, "p_open")
    describe("ESPN BET open vs ESPN BET close", d[d.p_open.notna()], "p_open", "p_eb_close")
    d["p_AB"] = (d.p_A + d.p_B) / 2
    describe("mean(A,B) vs open", d, "p_AB", "p_open")
    o = d[d.p_open.notna()]
    for pm in ("p_A", "p_B"):
        c = np.corrcoef(o[pm] - o.p_open, o.p_eb_close - o.p_open)[0, 1]
        print(f"  corr({pm} - open, close - open) = {c:+.3f} (n {len(o)}); mean |close-open| {np.mean(np.abs(o.p_eb_close - o.p_open)):.4f}")

    plan = [("simple_open_roi", "roi", "p_A", "p_open", "dec_oh", "dec_oa"),
            ("sim_open_roi", "roi", "p_B", "p_open", "dec_oh", "dec_oa"),
            ("simple_open_clv", "clv", "p_A", None, None, None),
            ("sim_open_clv", "clv", "p_B", None, None, None),
            ("simple_close_roi", "roi", "p_A", "p_close", "dec_h", "dec_a"),
            ("sim_close_roi", "roi", "p_B", "p_close", "dec_h", "dec_a")]
    assert len(plan) == N_TRIALS
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r.get("dataset") == DATASET}
    recs = []
    print(f"\nTRIALS (x = {X_THR}; bar p < {0.05 / N_TRIALS:.4f})")
    for name, kind, pm, pmkt, dh, da in plan:
        if kind == "roi":
            r, _, _ = roi_trial(d, pm, pmkt, dh, da)
        else:
            r = clv_trial(d, pm)
        sha = hashlib.sha256((DATASET + ":" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        lab = "ROI" if kind == "roi" else "mean CLV (prob pts)"
        print(f"  {name:17s} bets {r['bets']:5d} sum {r['units']:+8.2f} {lab} {r['roi']:+.4f} "
              f"CI(sum) [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] p {r['p']:.4f} seasons+ {r['folds_positive']}/{r['folds']} "
              f"{r['per_fold']}")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"nba_sim_{name}",
                     "sha": sha, "dataset": DATASET, "metric": kind,
                     "hypothesis": f"pre-registered in scripts/nba_player_sim.py ({name})",
                     "source": "player sim vs simple ridge, vs open + close (user 2026-10-22)", "status": "ok",
                     "trial_number": len(shas),
                     "passed": bool(r["p"] < 0.05 / N_TRIALS and r["roi"] > 0), **r})
    d.to_parquet(ROOT / "data" / "lab" / "nba_player_sim_scored.parquet")   # for post-hoc diagnostics
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"appended {len(recs)} trials to {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
