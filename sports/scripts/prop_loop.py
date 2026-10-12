"""Search loop: which pre-game information beats NFL player-prop lines, in units?

    python scripts/prop_loop.py build        # props x player history x defense x game -> data/prop_frame.parquet
    python scripts/prop_loop.py corr         # what correlates with (actual - line)? per market, BH-corrected
    python scripts/prop_loop.py loop         # models x feature sets x markets, walk-forward units -> ledger
    python scripts/prop_loop.py ledger       # every trial so far + the bar it has to clear
    python scripts/prop_loop.py holdout NAME --confirm   # HUMAN ONLY: score one trial on the sealed season

Honesty rules (the eval is units, so these are what keep the units real):
  * Features use only games BEFORE the prop's game (player weekly stats shift(1);
    defense allowed/style shift(1)). The line and opening line are known when betting.
  * Walk-forward by season: test season Y uses a model fit on seasons < Y only.
  * SEALED season: HOLDOUT is never scored by `loop` or `corr`. One shot per trial,
    by a human, after the search is done.
  * Every (market, model, feature set) run is a TRIAL written to the ledger; the
    promotion bar is p < 0.05 / #trials. Picking the best of 200 loop results
    without that correction is how backtests lie.
  * Bet rule is fixed for every trial: per prop take the side with the higher
    model EV at the quoted price, bet 1u if EV >= 5%.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from sports.http import fetch

warnings.filterwarnings("ignore")
DATA = ROOT / "data"
FRAME = DATA / "prop_frame.parquet"
LEDGER = ROOT / "lab" / "reports" / "prop_loop_ledger.jsonl"
HOLDOUT = 2025
EV_MIN = 0.05
MIN_WEEKS = 6   # weeks of props before the first test block
BLOCK = 3       # weeks per walk-forward block
PLAYERS = "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet"
TEAM = {"WSH": "WAS", "LAR": "LA"}

# ESPN market -> nflverse weekly stat (sum of columns)
STAT = {
    "total passing yards": ["passing_yards"], "total pass completions": ["completions"],
    "total passing attempts": ["attempts"], "total passing touchdowns": ["passing_tds"],
    "total passing interceptions": ["passing_interceptions"],
    "total rushing yards": ["rushing_yards"], "total carries": ["carries"],
    "total receiving yards": ["receiving_yards"], "total receptions": ["receptions"],
    "total rushing plus receiving yards": ["rushing_yards", "receiving_yards"],
    "total passing plus rushing yards": ["passing_yards", "rushing_yards"],
    "total sacks": ["def_sacks"], "total field goals made": ["fg_made"],
    "total extra points made": ["pat_made"],
    "total tackles plus assists": ["def_tackles_solo", "def_tackle_assists"],
}
USAGE = ["targets", "carries", "attempts", "receptions", "target_share", "air_yards_share", "wopr",
         "passing_epa", "rushing_epa", "receiving_epa"]


# ------------------------------------------------------------------ build

def _props() -> pd.DataFrame:
    fs = sorted((DATA / "espn" / "nfl").glob("graded_*.parquet"))
    p = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    p = p[p["market_n"].isin(STAT) & p["side"].isin(["over", "under"]) & p["athlete_id"].notna()]
    key = ["event_id", "provider", "athlete_id", "market_n", "line"]
    o = p[p["side"] == "over"].drop_duplicates(key).set_index(key)
    u = p[p["side"] == "under"].drop_duplicates(key).set_index(key)
    j = o[["line_open", "dec", "dec_open", "actual"]].rename(columns={"dec": "dec_over", "dec_open": "dec_over_open"}) \
        .join(u[["dec", "dec_open"]].rename(columns={"dec": "dec_under", "dec_open": "dec_under_open"}), how="inner")
    return j.reset_index()


def _espn_games() -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(f) for f in sorted((DATA / "espn" / "nfl").glob("games_*.parquet"))])
    g["date"] = pd.to_datetime(g["start_utc"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    g["home_n"] = g["home"].replace(TEAM)
    nv = pd.read_parquet(DATA / "nfl_games.parquet")
    nv["date"] = pd.to_datetime(nv["start_utc"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    g = g.merge(nv[["date", "home", "game_id", "spread_line", "total_line", "roof", "wind_mph"]]
                .rename(columns={"home": "home_n"}), on=["date", "home_n"], how="left")
    return g[["event_id", "season", "date", "game_id", "spread_line", "total_line", "roof", "wind_mph"]]


def _player_hist(pw: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per player-week: priors of the summed stat + usage, from strictly earlier weeks."""
    pw = pw.sort_values(["player_id", "season", "week"]).copy()
    pw["y"] = pw[cols].sum(axis=1, min_count=1)
    g = pw.groupby("player_id")
    out = pw[["player_id", "game_id", "team", "opponent_team", "position", "season", "week", "y"]].copy()
    s = g["y"].shift(1)
    out["p_ewm"] = s.groupby(pw["player_id"]).transform(lambda x: x.ewm(span=8, min_periods=1).mean())
    out["p_last3"] = s.groupby(pw["player_id"]).transform(lambda x: x.rolling(3, min_periods=1).mean())
    out["p_last1"] = s
    out["p_std8"] = s.groupby(pw["player_id"]).transform(lambda x: x.rolling(8, min_periods=3).std())
    out["p_n"] = g.cumcount()
    out["p_season_mean"] = pw.groupby(["player_id", "season"])["y"].transform(lambda x: x.shift(1).expanding().mean())
    for u in USAGE:
        if u in pw:
            out[f"u_{u}"] = g[u].shift(1).groupby(pw["player_id"]).transform(lambda x: x.ewm(span=6, min_periods=1).mean())
    # opponent: what this defense allowed to this position, earlier weeks only
    dp = pw.groupby(["opponent_team", "position", "season", "week"])["y"].sum().reset_index() \
        .sort_values(["season", "week"])
    dp["d_allowed"] = dp.groupby(["opponent_team", "position"])["y"].transform(
        lambda x: x.shift(1).ewm(span=8, min_periods=2).mean())
    out = out.merge(dp[["opponent_team", "position", "season", "week", "d_allowed"]],
                    on=["opponent_team", "position", "season", "week"], how="left")
    pos_mean = dp.groupby(["position", "season"])["y"].transform("mean")
    dp["d_allowed_rel"] = dp["d_allowed"] / pos_mean.replace(0, np.nan)
    out = out.merge(dp[["opponent_team", "position", "season", "week", "d_allowed_rel"]],
                    on=["opponent_team", "position", "season", "week"], how="left")
    return out


def cmd_build() -> None:
    pr = _props()
    gm = _espn_games()
    pr = pr.merge(gm, on="event_id", how="left")
    pl = pd.read_parquet(io.BytesIO(fetch(PLAYERS, binary=True)))[["gsis_id", "espn_id"]].dropna()
    pl["espn_id"] = pl["espn_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    pr = pr.merge(pl.rename(columns={"espn_id": "athlete_id", "gsis_id": "player_id"}), on="athlete_id", how="left")
    pw = pd.read_parquet(DATA / "nfl_player_week.parquet")
    pw = pw[pw["season_type"] == "REG"] if "season_type" in pw else pw
    parts = []
    for m, cols in STAT.items():
        sub = pr[pr["market_n"] == m]
        if sub.empty:
            continue
        h = _player_hist(pw, cols)
        parts.append(sub.merge(h, on=["player_id", "game_id"], how="inner"))
    f = pd.concat(parts, ignore_index=True)
    # defense style of the opponent (from the play-by-play team-game table, earlier games)
    tg = DATA / "nfl_fg_teamgames.parquet"
    if tg.exists():
        t = pd.read_parquet(tg)[["game_id", "defteam", "x_def_opp_zone", "x_def_opp_blitz", "x_def_opp_box",
                                 "x_def_rz_td_rate", "x_def_range_rate"]]
        f = f.merge(t.rename(columns={"defteam": "opponent_team"}).drop_duplicates(["game_id", "opponent_team"]),
                    on=["game_id", "opponent_team"], how="left")
    # game script + market features
    home = f["game_id"].str.split("_").str[3] == f["team"]
    f["team_spread"] = np.where(home, f["spread_line"], -f["spread_line"])
    f["implied_pts"] = (f["total_line"] + f["team_spread"]) / 2
    f["outdoor_wind"] = np.where(f["roof"].isin(["outdoors", "open"]), f["wind_mph"].fillna(0), 0)
    io_, iu = 1 / f["dec_over"], 1 / f["dec_under"]
    f["mkt_p_over"] = io_ / (io_ + iu)
    f["line_move"] = f["line"] - f["line_open"]
    f["price_move"] = (1 / f["dec_over"]) - (1 / f["dec_over_open"])
    f["z_ewm"] = (f["p_ewm"] - f["line"]) / f["p_std8"].replace(0, np.nan)
    f["gap_ewm"] = (f["p_ewm"] - f["line"]) / f["line"].replace(0, np.nan)
    f["gap_last3"] = (f["p_last3"] - f["line"]) / f["line"].replace(0, np.nan)
    f["resid"] = f["actual"] - f["line"]
    f["over"] = (f["actual"] > f["line"]).astype(float).where(f["actual"] != f["line"])
    f.to_parquet(FRAME, index=False)
    print(f"prop frame: {len(f):,} props, seasons {sorted(f['season_x'].unique()) if 'season_x' in f else sorted(f['season'].unique())}")
    print(f.groupby("market_n").size().sort_values(ascending=False).to_string())


def load() -> pd.DataFrame:
    f = pd.read_parquet(FRAME)
    if "season_x" in f:
        f = f.rename(columns={"season_x": "season"}).drop(columns=["season_y"], errors="ignore")
    return f[f["over"].notna()]


FEATS = {
    "market": ["mkt_p_over", "line_move", "price_move", "line"],
    "player": ["gap_ewm", "gap_last3", "z_ewm", "p_n", "p_std8"],
    "usage": [c for c in (f"u_{u}" for u in USAGE)],
    "matchup": ["d_allowed_rel", "x_def_opp_zone", "x_def_opp_blitz", "x_def_opp_box",
                "x_def_rz_td_rate", "x_def_range_rate"],
    "script": ["implied_pts", "team_spread", "total_line", "outdoor_wind"],
}


# ------------------------------------------------------------------ corr

def cmd_corr() -> None:
    f = load()
    f = f[f["season"] != HOLDOUT]
    rows = []
    for m, g in f.groupby("market_n"):
        if len(g) < 300:
            continue
        for fam, cols in FEATS.items():
            for c in cols:
                if c not in g or c == "mkt_p_over":
                    continue
                ok = g[c].notna() & g["resid"].notna()
                if ok.sum() < 200 or g.loc[ok, c].std() == 0:
                    continue
                # target = surprise vs the market's own (de-vigged) probability, not raw "over hit":
                # a feature that only restates the price would otherwise look predictive
                r, p = stats.spearmanr(g.loc[ok, c], g.loc[ok, "over"] - g.loc[ok, "mkt_p_over"])
                rows.append({"market": m, "family": fam, "feature": c, "n": int(ok.sum()), "rho_vs_surprise": r, "p": p})
    r = pd.DataFrame(rows)
    o = np.argsort(r["p"].to_numpy())
    q = np.empty(len(r))
    q[o] = np.minimum.accumulate((r["p"].to_numpy()[o] * len(r) / np.arange(1, len(r) + 1))[::-1])[::-1]
    r["q_bh"] = np.minimum(q, 1)
    r = r.sort_values("p")
    r.to_csv(ROOT / "lab" / "reports" / "prop_corr.csv", index=False)
    print(f"{len(r)} feature x market tests (non-holdout seasons); BH q<0.10: {(r.q_bh < .1).sum()}")
    print("rho = Spearman with (over hit - market de-vigged P(over)): nonzero = the price is wrong in that direction")
    print(r.head(30).to_string(index=False, float_format=lambda v: f"{v:.3g}"))


# ------------------------------------------------------------------ loop

def _models():
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    base = lambda *m: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), *m)
    return {
        "logit_l2": lambda: base(LogisticRegression(C=0.1, max_iter=500)),
        "pca5_logit": lambda: base(PCA(n_components=5), LogisticRegression(C=0.1, max_iter=500)),
        "pca10_logit": lambda: base(PCA(n_components=10), LogisticRegression(C=0.1, max_iter=500)),
        "hgb": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                     HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                    max_iter=200, l2_regularization=1.0)),
    }


SETS = {
    "player": ["player"], "player+usage": ["player", "usage"], "matchup": ["matchup"],
    "script": ["script"], "market": ["market"], "all": ["market", "player", "usage", "matchup", "script"],
}


def _bets(test: pd.DataFrame, p_over: np.ndarray) -> pd.DataFrame:
    ev_o = p_over * test["dec_over"] - 1
    ev_u = (1 - p_over) * test["dec_under"] - 1
    over = ev_o >= ev_u
    ev = np.where(over, ev_o, ev_u)
    won = np.where(over, test["over"] == 1, test["over"] == 0)
    dec = np.where(over, test["dec_over"], test["dec_under"])
    b = test.assign(ev=ev, bet_over=over, profit=np.where(won, dec - 1, -1.0))
    return b[b["ev"] >= EV_MIN]


def _trial_count() -> int:
    return sum(1 for _ in LEDGER.open()) if LEDGER.exists() else 0


def cmd_loop() -> None:
    f = load()
    f = f[f["season"] != HOLDOUT]
    # Walk-forward by WEEK BLOCK: free yardage props start in 2024 and 2025 is sealed, so
    # season-level folds would leave nothing to test. Each 3-week block is predicted by a
    # model fit on every earlier week (all markets' rows dated before the block).
    f = f.assign(wk=f["season"] * 100 + f["week"])
    weeks = sorted(f["wk"].unique())
    blocks = [weeks[i:i + BLOCK] for i in range(MIN_WEEKS, len(weeks), BLOCK)]
    seasons = sorted(f["season"].unique())
    markets = [m for m, g in f.groupby("market_n") if len(g) >= 400] + ["ALL"]
    models = _models()
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    res = []
    for m in markets:
        g = f if m == "ALL" else f[f["market_n"] == m]
        for sname, fams in SETS.items():
            cols = [c for fam in fams for c in FEATS[fam] if c in g]
            # every set includes the market's own probability as the anchor to correct
            cols = list(dict.fromkeys(["mkt_p_over"] + cols))
            for mname, mk in models.items():
                if mname.startswith("pca") and len(cols) <= int(mname[3:].split("_")[0]):
                    continue
                bets = []
                for blk in blocks:
                    tr, te = g[g["wk"] < blk[0]], g[g["wk"].isin(blk)]
                    if len(tr) < 200 or te.empty:
                        continue
                    model = mk().fit(tr[cols], tr["over"])
                    bets.append(_bets(te, model.predict_proba(te[cols])[:, 1]))
                b = pd.concat(bets) if bets else pd.DataFrame()
                n = len(b)
                units = float(b["profit"].sum()) if n else 0.0
                p = float(np.mean(np.random.default_rng(0).choice(b["profit"].to_numpy(), (4000, n)).sum(1) <= 0)) if n else 1.0
                rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "market": m,
                       "features": sname, "model": mname, "seasons": [int(s) for s in seasons],
                       "test_from_week": int(blocks[0][0]) if blocks else None, "bets": n,
                       "units": units, "roi": units / n if n else 0.0, "p": p,
                       "name": f"{m}|{sname}|{mname}"}
                with LEDGER.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                res.append(rec)
    r = pd.DataFrame(res).sort_values("units", ascending=False)
    n = _trial_count()
    print(f"{len(r)} trials this run, {n} in ledger -> bar p < {0.05 / n:.5f}")
    pd.set_option("display.width", 200)
    print(r[["market", "features", "model", "bets", "units", "roi", "p"]].head(25).to_string(
        index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nworst 5 (the spread between best and worst is roughly what luck alone does):")
    print(r[["market", "features", "model", "bets", "units", "roi"]].tail(5).to_string(
        index=False, float_format=lambda v: f"{v:.3g}"))


def cmd_ledger() -> None:
    if not LEDGER.exists():
        print("ledger empty")
        return
    r = pd.DataFrame([json.loads(l) for l in LEDGER.open()])
    print(f"{len(r)} trials; bar p < {0.05 / len(r):.5f}; meeting bar: {(r.p < 0.05 / len(r)).sum()}")
    print(r.sort_values("p")[["name", "bets", "units", "roi", "p"]].head(20).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "corr", "loop", "ledger", "holdout"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    if a.cmd == "holdout":
        print("holdout scoring is a human decision and is not implemented until a trial is chosen")
        return 1
    {"build": cmd_build, "corr": cmd_corr, "loop": cmd_loop, "ledger": cmd_ledger}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
