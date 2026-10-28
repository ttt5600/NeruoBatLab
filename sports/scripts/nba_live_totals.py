"""NBA LIVE TOTALS: a remaining-points model vs Kalshi's in-game KXNBATOTAL ladders. DEV ONLY, no orders.

Motivation (2026-10-27, user, live POR@LAL tied 121-121 Q4 3:49, total 242, pregame 233.5): unconditional history
P(remaining >= 24/26/28) ~ .35-.39/.26-.30/.18-.22 vs Kalshi over-265.5/267.5/269.5 mids .345/.265/.25 -- roughly
fair, but history ignored the game's pace. Question: does a model of remaining points that uses pregame total, own
pace, time left and margin (fouling / OT) beat Kalshi's live total ladder?

    python scripts/nba_live_totals.py build      # checkpoint states 2015-16 .. 2025-26(dev only) + pregame totals
    python scripts/nba_live_totals.py fit        # fit 2015-16..2023-24, calibrate on 2024-25 (PIT, reliability)
    python scripts/nba_live_totals.py kalshi     # pull KXNBATOTAL 1-min candles for DEV games (2025-10-21..2026-01-14)
    python scripts/nba_live_totals.py dev [--ledger]

PRE-REGISTRATION (written 2026-10-27 BEFORE any KXNBATOTAL price was pulled or joined; only the dev-window market
LIST -- tickers/strikes, 602 events x 11 strikes -- was looked at to learn the ladder format):

DATA
  pbp      : data/nba_big_lead_plays.parquet (ESPN; hs/as running score, period, clock_s, wallclock) + _meta.
             Seasons 2015-16 .. 2024-25 for fitting/calibration. 2025-26 rows are loaded ONLY for games dated
             2025-10-21 .. 2026-01-14 (the Kalshi DEV window); everything dated >= 2026-01-15 is dropped at load.
             Score at a point = max running hs / as over plays at or before it (scores are monotone).
             Final total = max hs + max as over all plays (includes OT).
  pregame  : total T0. Fit/calibration: data/nba_odds/OddsData.sqlite "OU" (joined by date + home team).
             DEV games: ESPN summary pickcenter[0].overUnder (cached summaries), fallback OddsData "OU".
             Amendment 1 (2026-10-27, after `build`, before any Kalshi price): drop games with T0 outside
             [170, 280] -- OddsData has ~12 typo lines (1.5, 5.0, 24.0, 1105, 1955.5); no dev game is affected.
  SEALED   : Kalshi NBA candles dated >= 2026-01-15 and data/nba_ingame.parquet rows in that range: never read.

CHECKPOINTS (c, regulation seconds left s_c): Q1end 2160, HALF 1440, Q3end 720, Q4_6 360, Q4_3 180.
  Historical state at c = scores of all plays with period < p, or period == p and clock >= c (Q4); for period ends,
  all plays with period <= p.

MODEL (per checkpoint c; fit by OLS on games 2015-16 .. 2023-24 with a pregame total; all game types)
  r0 = T0 / 48 (pregame points per minute); P = points so far; e = elapsed regulation minutes; s = s/60 minutes left;
  pace = P / e; w = e / (e + K) with K = 24 minutes FIXED A PRIORI (half a game of evidence = the prior's weight);
  remaining R = b1_c * s*r0 + b2_c * s*w*(pace - r0) + eps          (no intercept; b1, b2 fitted per c)
  eps distribution = EMPIRICAL residuals of the training games at checkpoint c within the |margin| bucket
  {0-3, 4-6, 7-10, 11-15, 16+}. This carries late fouling (close games score more late), OT (a ~+20 pt tail in
  tied/close Q4 states) and garbage time (blowouts) non-parametrically.
  P(final > L) = mean_j 1[ P + mu + eps_j > L ].
  Off-checkpoint state (Kalshi snapshot state is up to ~1 min of wall time later): mu uses the state's actual s,
  e, P; residuals are rescaled eps' = (s/s_c)*mean_b + sqrt(s/s_c)*(eps - mean_b). s from the state's own play.
  CALIBRATION (out of sample, 2024-25 season games): PIT histogram (mid-PIT for integer totals) per checkpoint;
  reliability of P(final > L) for L = round(P + mu) + {-12.5, -7.5, -2.5, +2.5, +7.5, +12.5}; log loss of the same
  vs a NO-PACE baseline (b2 = 0) and vs the unconditional "history" model (mu = mean R at c, residuals pooled).
  FROZEN DEV MODEL = same spec refit on 2015-16 .. 2024-25.

KALSHI SNAPSHOTS (DEV = games dated 2025-10-21 .. 2026-01-14, ET)
  event time w_c = wallclock of the last play at the checkpoint (period end: last play of the period; Q4_6/Q4_3:
  last Q4 play with clock >= c). Candle = the first 1-min candle (per strike) with end in [w_c + 60, w_c + 360] s.
  State = plays with wallclock <= candle end - 60 s (60 s data latency, as in nba_late_wp.py).
  Strike qualifies when 0 < yes_bid, yes_ask < 1 and yes_ask - yes_bid <= 0.04 (two-sided, spread <= 4c).
  YES = "Over <floor_strike> points scored"; settlement = Kalshi market result. Fee = 0.07*p*(1-p) per contract.
  T1 calib    : over all qualifying (game, checkpoint, strike) rows, D = Brier(Kalshi mid) - Brier(model);
                success if D > 0 (model better). Also reported: Kalshi mid reliability bins vs realised.
  T2 rule     : per game-checkpoint, candidate bets YES at ask (edge q - ask - fee(ask)) and NO at 1 - bid
                (edge (1-q) - (1-bid) - fee(1-bid)) over qualifying strikes; take the single max-edge candidate if
                edge >= 0.05; 1 contract, hold to settlement. Statistic = summed P&L; ROI = P&L / cost.
  T3 close Q4 : T2 restricted to Q4_6 and Q4_3 snapshots with |margin| <= 6 at the state.
  TEST        : game-clustered bootstrap (5000 resamples, seed 0), one-sided p = P(stat* <= 0).
                3 trials -> bar p < 0.05/3 = 0.0167 AND the point estimate > 0 in both halves (DEV games split at
                the median game date). Passing nothing => Kalshi live totals are efficient vs this model.
  If a trial passes: propose it to `lb` as decision support with the frozen rule (no orders), sealed untouched.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports.http import CACHE_DIR, _key, fetch, fetch_many

PLAYS = ROOT / "data" / "nba_big_lead_plays.parquet"
META = ROOT / "data" / "nba_big_lead_meta.parquet"
ODDS = ROOT / "data" / "nba_odds" / "OddsData.sqlite"
STATES = ROOT / "data" / "nba_live_totals_states.parquet"
DEVPLAYS = ROOT / "data" / "nba_live_totals_devplays.parquet"
CANDLES = ROOT / "data" / "nba_live_totals_candles.parquet"
MODEL = ROOT / "data" / "nba_live_totals_model.json"
RESULTS = ROOT / "results" / "nba_live_totals_dev.json"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nba_live_totals"
API = "https://api.elections.kalshi.com/trade-api/v2"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
DEV_START, SEALED = pd.Timestamp("2025-10-21"), pd.Timestamp("2026-01-15")
CKPTS = {"Q1end": (1, 0, 2160), "HALF": (2, 0, 1440), "Q3end": (3, 0, 720), "Q4_6": (4, 360, 360), "Q4_3": (4, 180, 180)}
K = 24.0
BUCKETS = [0, 4, 7, 11, 16, 999]
TRIALS = ["T1_calib_brier", "T2_rule", "T3_close_q4"]
EDGE = 0.05
FEE = 0.07
KX = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
NAMES = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Bobcats": "CHA",
         "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL",
         "Denver Nuggets": "DEN", "Detroit Pistons": "DET", "Golden State Warriors": "GS", "Houston Rockets": "HOU",
         "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
         "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
         "New Orleans Pelicans": "NO", "New York Knicks": "NY", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
         "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
         "San Antonio Spurs": "SA", "Toronto Raptors": "TOR", "Utah Jazz": "UTAH", "Washington Wizards": "WSH"}


def fee(p):
    return FEE * p * (1 - p)


def bucket(m):
    return np.searchsorted(BUCKETS, np.asarray(m), side="right") - 1


# ----------------------------------------------------------------------------------------------------------- build
def odds_totals() -> pd.DataFrame:
    c = sqlite3.connect(ODDS)
    tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
    use = [t for t in tabs if t.endswith("_new")] + ["2023-24", "2024-25", "odds_2025-26"]
    d = pd.concat([pd.read_sql(f'select Date, Home, Away, OU from "{t}"', c) for t in use if t in tabs])
    d["date"] = pd.to_datetime(d.Date, errors="coerce")
    d["home"] = d.Home.map(NAMES)
    return d.dropna(subset=["date", "home", "OU"]).drop_duplicates(["date", "home"])[["date", "home", "OU"]]


def allowed_meta() -> pd.DataFrame:
    m = pd.read_parquet(META)
    m = m[(m.season <= 2024) | ((m.season == 2025) & (m.date >= DEV_START) & (m.date < SEALED))]
    return m


def cmd_build() -> int:
    m = allowed_meta()
    ok = set(m.eid)
    p = pd.read_parquet(PLAYS)
    p = p[p.eid.isin(ok)]  # sealed rows dropped before anything else
    p = p.sort_values(["eid", "period", "clock_s", "seq"], ascending=[True, True, False, True])
    fin = p.groupby("eid").agg(fh=("hs", "max"), fa=("as", "max"), maxper=("period", "max"))
    rows = []
    for name, (per, clk, sc) in CKPTS.items():
        sel = (p.period < per) | ((p.period == per) & (p.clock_s >= clk)) if clk else (p.period <= per)
        g = p[sel].groupby("eid").agg(hs=("hs", "max"), as_=("as", "max"), lastper=("period", "max"))
        g = g[g.lastper == per]  # game actually reached this checkpoint in the pbp
        g["ckpt"], g["s_c"] = name, sc
        rows.append(g.reset_index())
    st = pd.concat(rows, ignore_index=True)
    st = st.merge(fin.reset_index(), on="eid")
    st["P"] = st.hs + st.as_
    st["margin"] = (st.hs - st.as_).abs()
    st["final"] = st.fh + st.fa
    st["R"] = st.final - st.P
    od = odds_totals()
    mm = m[["eid", "date", "home", "away", "season", "game_type"]].merge(od, on=["date", "home"], how="left")
    # ESPN pickcenter overUnder for DEV games (cached summaries only)
    dev = mm[mm.date >= DEV_START]
    ou = {}
    for e in dev.eid:
        url = f"{ESPN}/summary?event={e}"
        if (CACHE_DIR / (_key(url, None) + ".json")).exists():
            pc = (fetch(url).get("pickcenter") or [{}])[0]
            ou[e] = pc.get("overUnder")
    mm["T0_espn"] = mm.eid.map(ou)
    mm["T0"] = np.where(mm.date >= DEV_START, mm.T0_espn.fillna(mm.OU), mm.OU)
    st = st.merge(mm, on="eid")
    st = st.dropna(subset=["T0"])
    st = st[(st.T0 >= 170) & (st.T0 <= 280)]  # amendment 1: OddsData has typo lines (1.5, 24, 1955.5)
    st.to_parquet(STATES, index=False)
    p[p.eid.isin(dev.eid)].to_parquet(DEVPLAYS, index=False)
    chk = mm[mm.date >= DEV_START].dropna(subset=["T0_espn", "OU"])
    print(f"states {len(st)} rows, {st.eid.nunique()} games; by season:\n{st.groupby('season').eid.nunique().to_string()}")
    print(f"DEV games {dev.eid.nunique()}, ESPN overUnder {mm.T0_espn.notna().sum()}, ESPN vs OddsData |diff| median "
          f"{(chk.T0_espn - chk.OU).abs().median():.2f} (n {len(chk)})")
    f = st[st.ckpt == "HALF"]
    print("final - T0 by season: mean / sd")
    print(f.assign(d=f.final - f.T0).groupby("season").d.agg(["mean", "std", "size"]).round(2).to_string())
    return 0


# ------------------------------------------------------------------------------------------------------------- model
def feats(d: pd.DataFrame, s_sec=None):
    s = (d.s_c if s_sec is None else s_sec) / 60.0
    e = 48.0 - s
    r0 = d.T0 / 48.0
    pace = d.P / e
    w = e / (e + K)
    return np.c_[s * r0, s * w * (pace - r0)]


def fit(train: pd.DataFrame) -> dict:
    out = {}
    for c in CKPTS:
        d = train[train.ckpt == c]
        X = feats(d)
        b, *_ = np.linalg.lstsq(X, d.R.to_numpy(float), rcond=None)
        res = d.R.to_numpy(float) - X @ b
        bk = bucket(d.margin)
        out[c] = {"b": b.tolist(), "res": {int(k): np.sort(res[bk == k]).tolist() for k in np.unique(bk)},
                  "s_c": float(d.s_c.iloc[0]), "n": int(len(d)), "meanR": float(d.R.mean()), "allres": None}
    return out


def mu_res(mdl, c, d: pd.DataFrame, s_sec):
    """Return per-row mean mu and a list of residual arrays (rescaled to the state's own s)."""
    mc = mdl[c]
    X = feats(d, s_sec)
    mu = X @ np.array(mc["b"])
    ratio = np.asarray(s_sec, float) / mc["s_c"]
    bk = bucket(d.margin)
    res = []
    for k, r in zip(bk, ratio):
        e = np.asarray(mc["res"][k] if k in mc["res"] else mc["res"][str(k)], float)
        mb = e.mean()
        res.append(r * mb + np.sqrt(r) * (e - mb))
    return mu, res


def p_over(P, mu, res, L):
    return float(np.mean(P + mu + res > L))


def cmd_fit() -> int:
    st = pd.read_parquet(STATES)
    tr, ca = st[st.season <= 2023], st[st.season == 2024]
    mdl = fit(tr)
    rep = {}
    rng = np.random.default_rng(0)
    for c in CKPTS:
        d = ca[ca.ckpt == c]
        mu, res = mu_res(mdl, c, d, d.s_c.to_numpy(float))
        # no-pace baseline and unconditional history
        b1 = np.linalg.lstsq(feats(tr[tr.ckpt == c])[:, :1], tr[tr.ckpt == c].R.to_numpy(float), rcond=None)[0]
        mu_np = feats(d)[:, :1] @ b1
        hist_res = (tr[tr.ckpt == c].R - tr[tr.ckpt == c].R.mean()).to_numpy(float)
        pit, rel, ll = [], [], {"model": [], "nopace": [], "history": []}
        for i, (P, y) in enumerate(zip(d.P.to_numpy(), d.final.to_numpy())):
            tot = P + mu[i] + res[i]
            pit.append(np.mean(tot < y - 0.5) + 0.5 * np.mean(np.abs(tot - y) < 0.5))
            base = np.round(P + mu[i])
            tot_np = P + mu_np[i] + res[i]  # same residuals, no-pace mean
            tot_h = P + mdl[c]["meanR"] + hist_res
            for off in (-12.5, -7.5, -2.5, 2.5, 7.5, 12.5):
                L = base + off
                o = float(y > L)
                q = np.clip([np.mean(tot > L), np.mean(tot_np > L), np.mean(tot_h > L)], 1e-3, 1 - 1e-3)
                rel.append((q[0], o))
                for k, qq in zip(ll, q):
                    ll[k].append(-(o * np.log(qq) + (1 - o) * np.log(1 - qq)))
        pit = np.array(pit)
        rel = np.array(rel)
        bins = np.linspace(0, 1, 11)
        rb = pd.DataFrame(rel, columns=["q", "o"]).assign(b=lambda x: pd.cut(x.q, bins, include_lowest=True))
        rep[c] = {"n_cal_games": int(len(d)), "b": mdl[c]["b"], "n_train": mdl[c]["n"],
                  "pit_deciles": np.histogram(pit, bins)[0].tolist(),
                  "logloss": {k: float(np.mean(v)) for k, v in ll.items()},
                  "reliability": rb.groupby("b", observed=True).agg(q=("q", "mean"), o=("o", "mean"), n=("o", "size")).round(3).reset_index(drop=True).to_dict("records"),
                  "resid_sd_by_bucket": {int(k): float(np.std(v)) for k, v in mdl[c]["res"].items()},
                  "resid_mean_by_bucket": {int(k): float(np.mean(v)) for k, v in mdl[c]["res"].items()}}
        print(f"{c:6s} b={np.round(mdl[c]['b'], 3)} n_train {mdl[c]['n']} cal {len(d)} | logloss model "
              f"{rep[c]['logloss']['model']:.4f} nopace {rep[c]['logloss']['nopace']:.4f} history {rep[c]['logloss']['history']:.4f}")
        print("   PIT deciles", rep[c]["pit_deciles"])
        print("   resid mean/sd by |margin| bucket", {k: (round(rep[c]['resid_mean_by_bucket'][k], 1), round(v, 1)) for k, v in rep[c]["resid_sd_by_bucket"].items()})
        print("   reliability", [(r["q"], r["o"], r["n"]) for r in rep[c]["reliability"]])
    final = fit(st[st.season <= 2024])
    MODEL.write_text(json.dumps({"spec": "see docstring", "K": K, "fit_seasons": "2015-2024", "model": final,
                                 "calibration_2024_25": rep}))
    print(f"-> frozen DEV model (2015-16..2024-25) {MODEL}")
    return 0


# ------------------------------------------------------------------------------------------------------------ kalshi
def dev_games() -> pd.DataFrame:
    st = pd.read_parquet(STATES)
    g = st[(st.date >= DEV_START) & (st.date < SEALED)].drop_duplicates("eid")[["eid", "date", "home", "away", "T0", "final"]]
    return g


def cmd_kalshi() -> int:
    from kalshi_longshot import settled
    mk = settled("KXNBATOTAL")
    mk = mk[(mk.event_date >= DEV_START) & (mk.event_date < SEALED)].copy()  # sealed dropped before any candle call
    mk["code"] = mk.event_ticker.str.split("-").str[1].str[7:]
    g = dev_games()
    dp = pd.read_parquet(DEVPLAYS)
    span = dp.groupby("eid").wall.agg(["min", "max"])
    ev = {}
    for _, r in g.iterrows():
        if r.eid not in span.index or pd.isna(span.loc[r.eid, "min"]):
            continue
        kd = span.loc[r.eid, "min"].tz_convert("America/New_York").strftime("%y%b%d").upper()
        a, h = KX.get(r.away, r.away), KX.get(r.home, r.home)
        hit = mk[(mk.event_ticker.str.split("-").str[1].str[:7] == kd) & (mk.code.isin([a + h, h + a]))]
        if hit.event_ticker.nunique() == 1:
            ev[r.eid] = hit
    print(f"DEV games {len(g)}, matched to a KXNBATOTAL event {len(ev)}", flush=True)
    urls, keys = [], []
    for eid, hit in ev.items():
        s0 = int((span.loc[eid, "min"] - EPOCH) // pd.Timedelta(seconds=1)) - 600
        e0 = int((span.loc[eid, "max"] - EPOCH) // pd.Timedelta(seconds=1)) + 600
        for _, k in hit.iterrows():
            urls.append(f"{API}/historical/markets/{k.ticker}/candlesticks?start_ts={s0}&end_ts={e0}&period_interval=1")
            keys.append((eid, k.ticker, k.floor_strike, k.result))
    print(f"fetching {len(urls)} market candle series", flush=True)
    rows = []
    chunk = 400
    for i in range(0, len(urls), chunk):
        res = fetch_many(urls[i:i + chunk], cache=False)
        for (eid, tk, strike, result), j in zip(keys[i:i + chunk], res):
            for x in j.get("candlesticks", []):
                f = lambda kk: pd.to_numeric((x.get(kk) or {}).get("close_dollars", (x.get(kk) or {}).get("close")), errors="coerce")
                rows.append((eid, tk, strike, result, x["end_period_ts"], f("yes_bid"), f("yes_ask")))
        print(f"  {min(i + chunk, len(urls))}/{len(urls)}", flush=True)
    cd = pd.DataFrame(rows, columns=["eid", "ticker", "strike", "result", "end", "bid", "ask"])
    if cd.bid.max() > 1.5:  # cents -> dollars
        cd[["bid", "ask"]] /= 100
    cd.to_parquet(CANDLES, index=False)
    print(f"-> {CANDLES}: {len(cd)} candles, {cd.eid.nunique()} games, {cd.ticker.nunique()} markets")
    return 0


# --------------------------------------------------------------------------------------------------------------- dev
def snapshots() -> pd.DataFrame:
    mdl = json.loads(MODEL.read_text())["model"]
    g = dev_games().set_index("eid")
    dp = pd.read_parquet(DEVPLAYS)
    cd = pd.read_parquet(CANDLES)
    dp = dp.dropna(subset=["wall"])
    dp["t"] = (dp.wall - EPOCH) // pd.Timedelta(seconds=1)
    out = []
    for eid, c in cd.groupby("eid"):
        p = dp[dp.eid == eid].sort_values(["period", "clock_s", "seq"], ascending=[True, False, True])
        if p.empty or eid not in g.index:
            continue
        gi = g.loc[eid]
        for name, (per, clk, sc) in CKPTS.items():
            sel = (p.period == per) & (p.clock_s >= clk) if clk else (p.period == per)
            if not sel.any():
                continue
            w = int(p[sel].t.max())
            cc = c[(c.end >= w + 60) & (c.end <= w + 360)]
            if cc.empty:
                continue
            for tk, ct in cc.sort_values("end").groupby("ticker"):
                x = ct.iloc[0]
                known = p[p.t <= x.end - 60]
                if known.empty:
                    continue
                last = known.loc[known.t.idxmax()]
                hs, as_ = known.hs.max(), known["as"].max()
                per_s, clk_s = int(last.period), float(last.clock_s)
                s_left = max(0, 4 - per_s) * 720 + clk_s if per_s <= 4 else np.nan
                out.append({"eid": eid, "date": gi.date, "ckpt": name, "ticker": tk, "strike": x.strike, "result": x.result,
                            "end": x.end, "bid": x.bid, "ask": x.ask, "P": hs + as_, "margin": abs(hs - as_),
                            "s_left": s_left, "period": per_s, "T0": gi.T0, "final": gi.final})
    s = pd.DataFrame(out)
    s = s[s.s_left.notna() & (s.s_left > 0)]
    q = np.empty(len(s))
    for c in CKPTS:
        idx = np.where(s.ckpt == c)[0]
        if not len(idx):
            continue
        d = s.iloc[idx]
        mu, res = mu_res(mdl, c, d.assign(s_c=mdl[c]["s_c"]), d.s_left.to_numpy(float))
        q[idx] = [p_over(P, m, r, L) for P, m, r, L in zip(d.P, mu, res, d.strike)]
    s["q"] = q
    s["won_yes"] = s.result == "yes"
    return s


def boot(per_game: pd.Series, n=5000):
    v = per_game.to_numpy(float)
    rng = np.random.default_rng(0)
    b = np.array([v[rng.integers(0, len(v), len(v))].sum() for _ in range(n)])
    return float(np.mean(b <= 0)), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def rule(s: pd.DataFrame) -> pd.DataFrame:
    yes = s.assign(side="yes", px=s.ask, pq=s.q, win=s.won_yes)
    no = s.assign(side="no", px=1 - s.bid, pq=1 - s.q, win=~s.won_yes)
    c = pd.concat([yes, no])
    c["edge"] = c.pq - c.px - fee(c.px)
    c = c[c.edge >= EDGE].sort_values("edge", ascending=False).drop_duplicates(["eid", "ckpt"])
    c["cost"] = c.px + fee(c.px)
    c["pnl"] = c.win.astype(float) - c.cost
    return c


def cmd_dev() -> int:
    s = snapshots()
    q = s[(s.bid > 0) & (s.ask < 1) & (s.ask - s.bid <= 0.04)].copy()
    q["mid"] = (q.bid + q.ask) / 2
    q["o"] = q.won_yes.astype(float)
    # sanity: Kalshi YES == final > strike
    agree = ((s.final > s.strike) == s.won_yes).mean()
    med_date = q.drop_duplicates("eid").date.median()
    q["half"] = np.where(q.date <= med_date, "H1", "H2")
    print(f"DEV snapshots {len(s)} rows, qualifying (spread<=4c) {len(q)} rows, {q.eid.nunique()} games, "
          f"{q.groupby(['eid', 'ckpt']).ngroups} game-checkpoints; settle agreement final>strike vs result {agree:.4f}")
    print(f"median |mid - model| {np.median(np.abs(q.mid - q.q)):.3f}; by checkpoint:\n"
          + q.groupby("ckpt").apply(lambda x: pd.Series({"rows": len(x), "games": x.eid.nunique(), "med|mid-q|": np.median(np.abs(x.mid - x.q)),
                                                           "brier_k": np.mean((x.mid - x.o) ** 2), "brier_m": np.mean((x.q - x.o) ** 2)})).round(4).to_string())
    rec, res = [], {}
    # T1
    q["d"] = (q.mid - q.o) ** 2 - (q.q - q.o) ** 2
    pg = q.groupby("eid").d.sum()
    p, lo, hi = boot(pg)
    halves = q.groupby("half").d.mean().to_dict()
    res["T1_calib_brier"] = {"rows": int(len(q)), "games": int(q.eid.nunique()), "brier_kalshi": float(np.mean((q.mid - q.o) ** 2)),
                             "brier_model": float(np.mean((q.q - q.o) ** 2)), "D_mean": float(q.d.mean()), "p": p,
                             "ci_sum_lo": lo, "ci_sum_hi": hi, "halves": halves}
    # outcome ~ logit(mid) + logit(model) (descriptive)
    import statsmodels.api as sm
    lg = lambda x: np.log(np.clip(x, .01, .99) / (1 - np.clip(x, .01, .99)))
    X = sm.add_constant(np.c_[lg(q.mid), lg(q.q)])
    try:
        fr = sm.GLM(q.o, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(q.eid)[0]})
        pa, se = np.asarray(fr.params), np.asarray(fr.bse)
        res["info_test"] = {"coef_kalshi": float(pa[1]), "se_kalshi": float(se[1]), "coef_model": float(pa[2]), "se_model": float(se[2])}
    except Exception as ex:
        res["info_test"] = {"err": str(ex)}
    rb = q.assign(b=pd.cut(q.mid, [0, .05, .15, .3, .45, .55, .7, .85, .95, 1], include_lowest=True)).groupby("b", observed=True).agg(
        n=("o", "size"), games=("eid", "nunique"), mid=("mid", "mean"), realised=("o", "mean"), model=("q", "mean"))
    res["kalshi_reliability"] = rb.round(4).reset_index().astype({"b": str}).to_dict("records")
    for name, sub in (("T2_rule", q), ("T3_close_q4", q[q.ckpt.isin(["Q4_6", "Q4_3"]) & (q.margin <= 6)])):
        b = rule(sub)
        if len(b):
            p, lo, hi = boot(b.groupby("eid").pnl.sum())
            hv = b.groupby("half").apply(lambda x: x.pnl.sum() / x.cost.sum()).to_dict()
        else:
            p, lo, hi, hv = 1.0, 0.0, 0.0, {}
        res[name] = {"bets": int(len(b)), "games": int(b.eid.nunique()), "cost": float(b.cost.sum()), "units": float(b.pnl.sum()),
                     "roi": float(b.pnl.sum() / b.cost.sum()) if len(b) else 0.0, "p": p, "ci_lo": lo, "ci_hi": hi, "halves_roi": hv,
                     "win_rate": float(b.win.mean()) if len(b) else None, "mean_model_p": float(b.pq.mean()) if len(b) else None,
                     "mean_price": float(b.px.mean()) if len(b) else None,
                     "by_ckpt": b.groupby("ckpt").agg(n=("pnl", "size"), units=("pnl", "sum"), cost=("cost", "sum")).round(3).to_dict("index"),
                     "by_side": b.groupby("side").agg(n=("pnl", "size"), units=("pnl", "sum"), cost=("cost", "sum")).round(3).to_dict("index")}
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r.get("dataset") == DATASET}
    for name in TRIALS:
        shas.add(hashlib.sha256(("nba_live_totals:" + name).encode()).hexdigest()[:12])
    bar = 0.05 / len(shas)
    for name in TRIALS:
        r = res[name]
        hv = r.get("halves") or r.get("halves_roi") or {}
        r["pass"] = bool(r["p"] < bar and len(hv) == 2 and all(v > 0 for v in hv.values()))
        print(f"{name:15s} " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("by_ckpt", "by_side")}, default=str))
    print("info test", res["info_test"])
    print("Kalshi reliability:\n" + rb.round(3).to_string())
    for name in ("T2_rule", "T3_close_q4"):
        print(name, "by ckpt", res[name]["by_ckpt"], "by side", res[name]["by_side"])
    print(f"trials {len(shas)} -> bar p < {bar:.4f}")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"bar": bar, **res}, indent=1, default=str))
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for i, name in enumerate(TRIALS):
                r = res[name]
                f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"nba_live_totals_{name}",
                                    "sha": hashlib.sha256(("nba_live_totals:" + name).encode()).hexdigest()[:12], "dataset": DATASET,
                                    "hypothesis": f"pre-registered in scripts/nba_live_totals.py ({name})",
                                    "source": "user idea 2026-10-27: live NBA totals vs Kalshi (POR@LAL)", "status": "ok",
                                    "trial_number": i + 1, "bar": bar,
                                    **{k: v for k, v in r.items() if k not in ("by_ckpt", "by_side")}}, default=str) + "\n")
        print("appended to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "fit": cmd_fit, "kalshi": cmd_kalshi, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
