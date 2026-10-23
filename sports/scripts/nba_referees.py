"""NBA referees: bias, crew effects and match-fixing *screens* against the closing market.

    python scripts/nba_referees.py build          # crews + team fouls/FTA from cached ESPN summaries -> parquet
    python scripts/nba_referees.py run [--ledger] # all pre-registered tests below

WHAT THIS CAN AND CANNOT SHOW (read before the numbers). Every test here asks whether officiating
leaves a footprint *the closing market did not price*. A pass means "a pattern exists that random
chance and the market do not explain". It can never by itself mean "fixing": biases (home crowd,
star treatment, compensation) produce the same footprints with no intent, and a competent fixer
on a few games is statistically invisible in thousands. Real fixing cases (Donaghy 2007, Porter
2024, the 2025 EDNY indictments) were caught by wiretaps and sportsbook betting-pattern alerts,
not by box-score statistics. No result here is evidence against any named individual; R5 output
is anonymised on purpose.

PRE-REGISTERED 2026-10-22, written before any result was looked at.
Seasons use ESPN labels (year the season ENDS). DEV = 2022..2025 (2021-22 .. 2024-25).
SEALED = 2026 (2025-26): never loaded here (MANIFEST nba sealed = 2026). 2027 = current season, unused.
Games: regular season + play-in + playoffs, finished. Crew tests need >= 3 officials in gameInfo.officials;
playoff/play-in games list 4 (the alternate is not distinguishable), all listed officials enter the crew.

Market. Per game, the median over books (excluding the 'accuscore' model) of the close total where
the book reports one, else its last ('current') total -- ESPN's current is approximately closing.
Home win prob = median over books of the proportionally de-vigged moneyline. Betting prices come
from one real book per season (ESPN BET where present, else DraftKings) at its own total and odds.

Trials (Bonferroni bar = 0.05 / 7 = 0.00714 over R1, R2, R3, R4a, R4b, R4c, B1):
  R1 crew -> totals. y = (home+away points) - market total, demeaned within season. Statistic = R^2
     of OLS of y on referee indicators (each game has three 1s). Null: shuffle whole crews across
     games within season (keeps each referee's games per season), 2000 permutations; p = share of
     null R^2 >= observed. Report excess R^2 and implied SD of the crew-sum effect in points.
  R2 crew -> home result. y = home_win - de-vigged home prob, non-neutral games; same statistic/null.
  R3 series extension. Playoff games numbered within each series; games 4, 5, 6 where the series is
     not tied before tip-off. y = trailing team win - its de-vigged prob. Null: wins ~ Bernoulli(p)
     (market calibrated), 200k simulations, two-sided. Secondary (not a trial): ATS vs median spread.
  R4 Last Two Minute reports (public compilation github.com/atlhawksfanatic/L2M, file
     1-tidy/L2M/L2M_stats_nba.csv, 2015-03 .. 2025-05, all <= DEV;
     official.nba.com robots.txt disallows the ?gameId pages, so no fresh scrape). Errors = IC
     (incorrect call: benefits the disadvantaged side) and INC (incorrect non-call: benefits the
     committing side); rows without a home/away side dropped. Per game, net = #benefit A - #benefit B.
     Null: per-game sign flip (errors symmetric), 20k flips, two-sided.
     R4a A = home. R4b A = pre-game moneyline favourite (SBR closing odds 2015-16..2023-24; pick'em
     dropped). R4c A = team trailing at the moment of the error (score from ESPN play-by-play, last
     play strictly before the error clock; tied dropped; L2M seasons 2022-2025, matched to the ESPN cache).
  B1 bet by crew. Walk-forward: at the start of each calendar month of seasons 2023-2025, ridge
     (lambda = 100) of R1's y (not demeaned) on referee indicators using ALL dev games before that
     month; crew prediction = sum of three referee coefficients (unseen referee = 0). Bet over if
     prediction >= +1.0 pt, under if <= -1.0, at the season's book total and price; pushes refund.
     p = share of 4000 date-cluster bootstrap resamples with units <= 0 (one-sided).
Descriptive only (no trial, no p in the bar):
  C0 positive control: the R1 statistic on total free-throw attempts and total fouls. Price & Wolfers
     find crews move fouls; if this test cannot see FTA effects it has no power on totals either.
  R5 Donaghy-style screen: seasons with open+close totals (2024, 2025). Flag the 5% of games with the
     largest |close - open| total move. Per referee (>= 40 games), flagged count vs expectation and the
     rate the result went WITH the move; compare the max over referees to the same max under crew
     shuffling (1000 perms). Referees are reported as anonymous ranks only.
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

from sports.espn import summary
from sports.market import american_to_decimal, devig_two_way

DATA = ROOT / "data" / "espn" / "nba"
OUT = DATA / "refs_dev.parquet"
L2M = ROOT / "data" / "l2m" / "L2M_stats_nba.csv"
SBR = ROOT / "data" / "nba_odds" / "OddsData.sqlite"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nba_referees"
DEV = [2022, 2023, 2024, 2025]
N_TRIALS = 7
BAR = 0.05 / N_TRIALS
ESPN2NBA = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


# ------------------------------------------------------------------ build

def _team_stats(s: dict) -> dict:
    out = {}
    for t in s.get("boxscore", {}).get("teams", []) or []:
        side = t.get("homeAway")
        st = {x.get("name"): x.get("displayValue") for x in t.get("statistics", [])}
        fta = st.get("freeThrowsMade-freeThrowsAttempted", "")
        out[f"{side}_fta"] = float(fta.split("-")[1]) if "-" in fta else np.nan
        out[f"{side}_pf"] = float(st["fouls"]) if st.get("fouls") not in (None, "") else np.nan
    return out


def _market(season: int) -> pd.DataFrame:
    l = pd.read_parquet(DATA / f"lines_{season}.parquet")
    l = l[l.provider != "accuscore"].copy()
    l["tot"] = l.total_close.fillna(l.total)
    l["hml"] = l.home_ml_close.fillna(l.home_ml)
    l["aml"] = l.away_ml_close.fillna(l.away_ml)
    l["hsp"] = l.home_spread_close.fillna(l.spread)
    ok = l.hml.notna() & l.aml.notna()
    l.loc[ok, "p_home"] = devig_two_way(l.loc[ok, "hml"].values, l.loc[ok, "aml"].values)[0]
    m = l.groupby("event_id").agg(mkt_total=("tot", "median"), p_home=("p_home", "median"),
                                  home_spread=("hsp", "median"), n_books=("provider", "nunique"))
    book = "ESPN BET" if (l.provider == "ESPN BET").sum() > 0.8 * m.shape[0] else "DraftKings"
    b = l[l.provider == book].drop_duplicates("event_id").set_index("event_id")
    m["bk_total"] = b.tot
    m["bk_over"] = b.over_odds_close.fillna(b.over_odds)
    m["bk_under"] = b.under_odds_close.fillna(b.under_odds)
    m["bk_total_open"] = b.total_open
    m["book"] = book
    return m


def build() -> int:
    rows = []
    for season in DEV:
        g = pd.read_parquet(DATA / f"games_{season}.parquet")
        g = g[(g.season_type != "preseason") & (g.status == "STATUS_FINAL")]
        mk = _market(season)
        for r in g.itertuples():
            try:
                s = summary("nba", r.event_id)
            except Exception:
                continue
            offs = (s.get("gameInfo") or {}).get("officials") or []
            refs = sorted(o.get("fullName") or o.get("displayName") for o in offs)
            rec = {"event_id": r.event_id, "season": season, "season_type": r.season_type,
                   "start_utc": r.start_utc, "neutral": r.neutral, "home": r.home, "away": r.away,
                   "home_score": r.home_score, "away_score": r.away_score, "n_refs": len(refs),
                   "ref1": refs[0] if len(refs) > 0 else None, "ref2": refs[1] if len(refs) > 1 else None,
                   "ref3": refs[2] if len(refs) > 2 else None, "ref4": refs[3] if len(refs) > 3 else None,
                   **_team_stats(s)}
            rows.append(rec)
        print(season, len(g), "games")
    d = pd.DataFrame(rows)
    allm = pd.concat([_market(s).assign(season=s) for s in DEV]).drop(columns="season")
    d = d.merge(allm, left_on="event_id", right_index=True, how="left")
    d.to_parquet(OUT, index=False)
    print(f"wrote {OUT} {d.shape}; refs per game {d.n_refs.value_counts().to_dict()}")
    return 0


# ------------------------------------------------------------------ helpers

def load() -> pd.DataFrame:
    d = pd.read_parquet(OUT)
    d = d[d.season.isin(DEV)].copy()
    assert not d.season.isin([2026, 2027]).any()
    d["total"] = d.home_score + d.away_score
    d["home_win"] = (d.home_score > d.away_score).astype(float)
    return d.sort_values("start_utc").reset_index(drop=True)


def crews(d: pd.DataFrame) -> pd.DataFrame:
    return d[d.n_refs >= 3].reset_index(drop=True)


REFCOLS = ("ref1", "ref2", "ref3", "ref4")


def ref_matrix(d: pd.DataFrame, refs: list[str] | None = None):
    if refs is None:
        refs = sorted(set(pd.concat([d[c] for c in REFCOLS]).dropna()))
    idx = {r: i for i, r in enumerate(refs)}
    X = np.zeros((len(d), len(refs)))
    for c in REFCOLS:
        j = d[c].map(idx)
        ok = j.notna().values
        X[np.where(ok)[0], j[ok].astype(int).values] = 1.0
    return X, refs


def r2(X: np.ndarray, y: np.ndarray) -> float:
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return 1 - (e @ e) / (y @ y)  # y is demeaned


def perm_r2(d: pd.DataFrame, y: np.ndarray, n_perm: int, seed: int = 0) -> dict:
    """R^2 of y on referee indicators vs whole-crew shuffles within season."""
    y = np.asarray(y, float)
    X, refs = ref_matrix(d)
    keep = X.sum(0) >= 1
    X = X[:, keep]
    obs = r2(X, y)
    rng = np.random.default_rng(seed)
    groups = [np.where(d.season.values == s)[0] for s in sorted(d.season.unique())]
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = np.arange(len(d))
        for gi in groups:
            perm[gi] = rng.permutation(gi)
        null[k] = r2(X[perm], y)
    excess = obs - null.mean()
    return {"n": int(len(y)), "n_refs": int(keep.sum()), "r2": float(obs), "r2_null_mean": float(null.mean()),
            "r2_null_p95": float(np.quantile(null, .95)), "excess_r2": float(excess),
            "implied_crew_sd": float(np.sqrt(max(excess, 0) * y.var())), "resid_sd": float(y.std()),
            "p": float((1 + (null >= obs).sum()) / (1 + n_perm))}


def demean(d, col):
    return (d[col] - d.groupby("season")[col].transform("mean")).values


def signflip(net: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    net = net[net != 0]
    rng = np.random.default_rng(seed)
    obs = net.sum()
    sims = (net[None, :] * rng.choice([-1, 1], size=(n, len(net)))).sum(1)
    return float((1 + (np.abs(sims) >= abs(obs)).sum()) / (1 + n))


# ------------------------------------------------------------------ tests

def r1_r2_c0(d: pd.DataFrame) -> dict:
    out = {}
    t = d.dropna(subset=["mkt_total"]).reset_index(drop=True)
    t["y"] = t.total - t.mkt_total
    out["R1"] = perm_r2(t, demean(t, "y"), 2000, seed=1)
    out["R1"]["mean_total_resid"] = float(t.y.mean())
    h = d[~d.neutral].dropna(subset=["p_home"]).reset_index(drop=True)
    yh = (h.home_win - h.p_home).values
    out["R2"] = perm_r2(h, yh - yh.mean(), 2000, seed=2)
    out["R2"]["home_win_minus_market"] = float(yh.mean())
    for col, name in (("fta", "C0_fta"), ("pf", "C0_fouls")):
        c = d.dropna(subset=[f"home_{col}", f"away_{col}"]).reset_index(drop=True)
        c["v"] = c[f"home_{col}"] + c[f"away_{col}"]
        out[name] = perm_r2(c, demean(c, "v"), 500, seed=3)
        out[name]["mean"] = float(c.v.mean())
    # split-half persistence (descriptive): referee mean residual, seasons 2022+2024 vs 2023+2025
    long = pd.concat([t[["season", c, "y"]].rename(columns={c: "ref"}) for c in REFCOLS]).dropna(subset=["ref"])
    long["half"] = long.season.isin([2022, 2024])
    g = long.groupby(["ref", "half"]).y.agg(["mean", "size"]).unstack()
    g = g[(g["size"] >= 60).all(axis=1)]
    out["R1"]["split_half_corr"] = float(np.corrcoef(g["mean"][True], g["mean"][False])[0, 1])
    out["R1"]["split_half_refs"] = int(len(g))
    return out


def series_games(d: pd.DataFrame) -> pd.DataFrame:
    p = d[d.season_type == "post-season"].sort_values("start_utc").copy()
    p["pair"] = [tuple(sorted(x)) for x in zip(p.home, p.away)]
    rows = []
    for (s, pair), g in p.groupby(["season", "pair"], sort=False):
        wins = {pair[0]: 0, pair[1]: 0}
        for i, r in enumerate(g.itertuples(), 1):
            hw, aw = wins[r.home], wins[r.away]
            if i in (4, 5, 6) and hw != aw:
                trail_home = hw < aw
                pt = r.p_home if trail_home else 1 - r.p_home
                won = r.home_score > r.away_score if trail_home else r.away_score > r.home_score
                margin = (r.home_score - r.away_score) * (1 if trail_home else -1)
                spread = r.home_spread if trail_home else -r.home_spread
                rows.append({"season": s, "game_no": i, "trail_home": trail_home, "elim": max(hw, aw) == 3,
                             "p_trail": pt, "trail_win": float(won), "trail_cover": margin + spread,
                             "event_id": r.event_id})
            wins[r.home if r.home_score > r.away_score else r.away] += 1
    return pd.DataFrame(rows)


def r3(d: pd.DataFrame) -> dict:
    s = series_games(d).dropna(subset=["p_trail"])
    y = s.trail_win - s.p_trail
    rng = np.random.default_rng(4)
    sims = (rng.random((200000, len(s))) < s.p_trail.values).mean(1) - s.p_trail.mean()
    obs = y.mean()
    cov = s.trail_cover.dropna()
    cov = cov[cov != 0]
    return {"n": int(len(s)), "trail_win_rate": float(s.trail_win.mean()), "trail_market_prob": float(s.p_trail.mean()),
            "excess": float(obs), "p": float((1 + (np.abs(sims) >= abs(obs)).sum()) / 200001),
            "elim_games": int(s.elim.sum()), "elim_excess": float((y[s.elim]).mean()),
            "ats_n": int(len(cov)), "ats_cover_rate": float((cov > 0).mean()),
            "by_game": {int(k): f"{v.trail_win.mean() - v.p_trail.mean():+.3f}/{len(v)}" for k, v in s.groupby("game_no")}}


def l2m_errors() -> pd.DataFrame:
    l = pd.read_csv(L2M, low_memory=False)
    l = l[l.decision.isin(["IC", "INC"])].copy()
    l["benefit"] = np.where(l.decision == "INC", l.committing_side, l.disadvantaged_side)
    l = l[l.benefit.isin(["home", "away"])]
    l["gid"] = l.gid.astype(str)
    return l


def sbr_favs() -> pd.DataFrame:
    import sqlite3
    c = sqlite3.connect(SBR)
    tabs = [f"odds_{y}-{str(y + 1)[2:]}_new" for y in range(2014, 2024)]
    names = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
    fs = [pd.read_sql(f'select Date, Home, Away, ML_Home, ML_Away from "{t}"', c) for t in tabs if t in names]
    f = pd.concat(fs)
    for k in ("ML_Home", "ML_Away"):
        f[k] = pd.to_numeric(f[k], errors="coerce")
    return f.dropna()


NAME2TRI = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Bobcats": "CHA",
            "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL",
            "Denver Nuggets": "DEN", "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU",
            "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
            "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
            "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
            "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
            "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"}


def _clock(s) -> float:
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + float(sec)
    except Exception:
        return np.nan


def _score_before(plays: list, period: int, t_err: float):
    hs = as_ = None
    for p in plays:
        per = (p.get("period") or {}).get("number")
        clk = _clock((p.get("clock") or {}).get("displayValue"))
        if per is None or np.isnan(clk):
            continue
        if per < period or (per == period and clk > t_err):
            hs, as_ = p.get("homeScore"), p.get("awayScore")
        else:
            break
    return hs, as_


def r4(n_flip: int = 20000) -> dict:
    l = l2m_errors()
    out = {"n_errors": int(len(l)), "n_games": int(l.gid.nunique()),
           "seasons": f"{int(l.season.min())}-{int(l.season.max())}"}
    # R4a home
    net = l.groupby("gid").benefit.apply(lambda b: (b == "home").sum() - (b == "away").sum()).values
    out["R4a"] = {"share_home": float((l.benefit == "home").mean()), "n": int(len(l)), "p": signflip(net, n_flip, 5),
                  "by_type": {k: f"{(v.benefit == 'home').mean():.3f}/{len(v)}" for k, v in l.groupby("decision")}}
    # R4b favourite
    f = sbr_favs()
    f["home"] = f.Home.map(NAME2TRI)
    f["date"] = pd.to_datetime(f.Date).dt.strftime("%Y-%m-%d")
    f = f[f.ML_Home != f.ML_Away].drop_duplicates(["date", "home"])
    f["fav_side"] = np.where(f.ML_Home < f.ML_Away, "home", "away")
    l["home3"] = l.home.replace({"BRK": "BKN", "PHO": "PHX", "CHO": "CHA", "NOH": "NOP"})
    lb = l.merge(f[["date", "home", "fav_side"]], left_on=["date", "home3"], right_on=["date", "home"], how="inner")
    lb["fav_benefit"] = lb.benefit == lb.fav_side
    netb = lb.groupby("gid").fav_benefit.apply(lambda b: b.sum() - (~b).sum()).values
    out["R4b"] = {"share_fav": float(lb.fav_benefit.mean()), "n": int(len(lb)), "games": int(lb.gid.nunique()),
                  "matched_share": float(lb.gid.nunique() / l.gid.nunique()), "p": signflip(netb, n_flip, 6)}
    # R4c trailing at time of error (ESPN play-by-play)
    eg = pd.concat([pd.read_parquet(DATA / f"games_{s}.parquet") for s in (2022, 2023, 2024, 2025)])
    eg = eg[eg.season_type != "preseason"].copy()
    eg["date"] = eg.start_utc.dt.tz_convert("US/Eastern").dt.strftime("%Y-%m-%d")
    eg["home3"] = eg.home.replace(ESPN2NBA)
    eg = eg.drop_duplicates(["date", "home3"]).set_index(["date", "home3"]).event_id
    lc = l[l.season.isin([2022, 2023, 2024, 2025])].copy()
    lc["period_n"] = lc.period.astype(str).str.extract(r"Q(\d)")[0].astype(float)
    lc["t_err"] = lc.time.map(_clock)
    res, cache = [], {}
    for r in lc.itertuples():
        eid = eg.get((r.date, r.home3))
        if eid is None or np.isnan(r.period_n) or np.isnan(r.t_err):
            res.append(np.nan)
            continue
        if eid not in cache:
            try:
                cache[eid] = summary("nba", eid).get("plays") or []
            except Exception:
                cache[eid] = []
        hs, as_ = _score_before(cache[eid], int(r.period_n), r.t_err)
        if hs is None or int(hs) == int(as_):
            res.append(np.nan)
            continue
        res.append("home" if int(hs) < int(as_) else "away")
    lc["trail_side"] = res
    lt = lc.dropna(subset=["trail_side"])
    lt = lt.assign(tb=lt.benefit == lt.trail_side)
    nett = lt.groupby("gid").tb.apply(lambda b: b.sum() - (~b).sum()).values
    out["R4c"] = {"share_trailing": float(lt.tb.mean()), "n": int(len(lt)), "games": int(lt.gid.nunique()),
                  "of_errors_in_window": int(len(lc)), "p": signflip(nett, n_flip, 7),
                  "by_type": {k: f"{v.tb.mean():.3f}/{len(v)}" for k, v in lt.groupby("decision")}}
    return out


def ridge(X, y, lam):
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


def b1(d: pd.DataFrame, lam: float = 100.0, thr: float = 1.0) -> dict:
    t = d.dropna(subset=["mkt_total"]).reset_index(drop=True)
    t["y"] = t.total - t.mkt_total
    t["month"] = t.start_utc.dt.tz_convert("US/Eastern").dt.to_period("M")
    bets = []
    for m in sorted(t[t.season >= 2023].month.unique()):
        train = t[t.month < m]
        test = t[(t.month == m) & t.bk_total.notna() & t.bk_over.notna() & t.bk_under.notna()]
        if test.empty:
            continue
        X, refs = ref_matrix(train)
        b = ridge(X, train.y.values - train.y.mean(), lam)
        Xt, _ = ref_matrix(test, refs)
        pred = Xt @ b
        for r, pr in zip(test.itertuples(), pred):
            if abs(pr) < thr:
                continue
            side = "over" if pr > 0 else "under"
            odds = r.bk_over if side == "over" else r.bk_under
            dec = float(american_to_decimal(np.array([odds]))[0])
            diff = r.total - r.bk_total
            if diff == 0:
                pnl = 0.0
            else:
                won = diff > 0 if side == "over" else diff < 0
                pnl = dec - 1 if won else -1.0
            bets.append({"date": r.start_utc.date(), "season": r.season, "side": side, "pred": pr, "pnl": pnl})
    bt = pd.DataFrame(bets)
    rng = np.random.default_rng(8)
    days = bt.groupby("date").pnl.sum().values
    boots = np.array([rng.choice(days, len(days)).sum() for _ in range(4000)])
    per = bt.groupby("season").pnl.agg(["sum", "size", "mean"]).round(3)
    return {"bets": int(len(bt)), "units": float(bt.pnl.sum()), "roi": float(bt.pnl.mean()),
            "win_rate": float((bt.pnl > 0).mean()), "overs": int((bt.side == "over").sum()),
            "ci_lo": float(np.quantile(boots, .025)), "ci_hi": float(np.quantile(boots, .975)),
            "p": float(np.mean(boots <= 0)), "folds_positive": int((per["sum"] > 0).sum()),
            "per_fold": {str(k): f"{v['sum']:+.1f}u/{int(v['size'])} ({v['mean']:+.2%})" for k, v in per.iterrows()}}


def r5(d: pd.DataFrame, n_perm: int = 1000) -> dict:
    t = d[d.season.isin([2024, 2025]) & d.bk_total_open.notna() & d.bk_total.notna()].reset_index(drop=True)
    t["move"] = t.bk_total - t.bk_total_open
    cut = t.move.abs().quantile(0.95)
    t["flag"] = t.move.abs() >= cut
    t["with_move"] = np.sign(t.total - t.bk_total) == np.sign(t.move)
    X, refs = ref_matrix(t)
    ok = X.sum(0) >= 40
    X = X[:, ok]
    n = X.sum(0)
    base = t.flag.mean()

    def stats(perm):
        f = t.flag.values[perm].astype(float)
        w = (t.flag.values & t.with_move.values)[perm].astype(float)
        cnt = X.T @ f
        z = (cnt - n * base) / np.sqrt(n * base * (1 - base))
        wr = np.where(cnt > 0, (X.T @ w) / np.maximum(cnt, 1), np.nan)
        return z, wr, cnt

    z, wr, cnt = stats(np.arange(len(t)))
    rng = np.random.default_rng(9)
    groups = [np.where(t.season.values == s)[0] for s in (2024, 2025)]
    zmax_null, wr_null = [], []
    for _ in range(n_perm):
        perm = np.arange(len(t))
        for gi in groups:
            perm[gi] = rng.permutation(gi)
        zn, wn, cn = stats(perm)
        zmax_null.append(zn.max())
        wr_null.append(np.nanmax(np.where(cn >= 5, wn, np.nan)))
    order = np.argsort(-z)[:5]
    return {"games": int(len(t)), "move_cut_pts": float(cut), "flagged": int(t.flag.sum()),
            "with_move_rate_flagged": float(t[t.flag].with_move.mean()), "refs_screened": int(ok.sum()),
            "max_z": float(z.max()), "max_z_null_p95": float(np.quantile(zmax_null, .95)),
            "max_z_perm_p": float(np.mean(np.array(zmax_null) >= z.max())),
            "max_with_rate_min5": float(np.nanmax(np.where(cnt >= 5, wr, np.nan))),
            "max_with_rate_null_p95": float(np.nanquantile(wr_null, .95)),
            "top5_anon": [f"rank{i + 1}: {int(cnt[j])} flagged of {int(n[j])} (exp {n[j] * base:.1f}), z {z[j]:+.2f}, "
                          f"with-move {wr[j]:.2f}" for i, j in enumerate(order)]}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        return build()
    d = load()
    print(f"DEV games: {len(d)} by season {d.season.value_counts().sort_index().to_dict()}; with crew {len(crews(d))}")
    t = d.dropna(subset=["mkt_total"])
    print(f"sanity: total - market mean {np.mean(t.total - t.mkt_total):+.2f}, sd {np.std(t.total - t.mkt_total):.2f}; "
          f"home win {d.home_win.mean():.3f} vs market {d.p_home.mean():.3f}")
    c = crews(d)
    res = r1_r2_c0(c)
    res["R3"] = r3(d)
    res["R4"] = r4()
    res["B1"] = b1(c)
    res["R5"] = r5(c)
    print(json.dumps(res, indent=1, default=str))
    trials = {"R1_crew_totals": res["R1"], "R2_crew_home": res["R2"], "R3_series_extension": res["R3"],
              "R4a_l2m_home": res["R4"]["R4a"], "R4b_l2m_favourite": res["R4"]["R4b"],
              "R4c_l2m_trailing": res["R4"]["R4c"], "B1_bet_by_crew": res["B1"]}
    print(f"\nBonferroni bar p < {BAR:.5f} over {N_TRIALS} trials")
    for k, v in trials.items():
        print(f"  {k:22s} p {v['p']:.4f}  {'PASS' if v['p'] < BAR else 'fail'}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "nba_referees.json").write_text(json.dumps(res, indent=1, default=str))
    if "--ledger" in sys.argv:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with LEDGER.open("a") as f:
            for i, (k, v) in enumerate(trials.items(), 1):
                sha = hashlib.sha256(("nba_referees:" + k).encode()).hexdigest()[:12]
                f.write(json.dumps({"ts": ts, "strategy": f"nba_ref_{k}", "sha": sha, "dataset": DATASET,
                                    "hypothesis": f"pre-registered in scripts/nba_referees.py ({k})",
                                    "source": "user ask 2026-10-22 (referee bias / fixing); lead L6",
                                    "status": "ok", "trial_number": i, "study_trials": N_TRIALS,
                                    "bonferroni_bar": BAR, "passed": bool(v["p"] < BAR), **v}, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
