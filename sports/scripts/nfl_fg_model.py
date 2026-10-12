"""Field goals made per team-game: how defense style + game script drive them, and whether
it beats the kicker "Total Field Goals Made" prop.

    python scripts/nfl_fg_model.py tendencies   # team defense styles + offense results vs man/zone
    python scripts/nfl_fg_model.py model        # Poisson model, out-of-sample 2022-2025
    python scripts/nfl_fg_model.py props        # EV vs ESPN kicker prop lines (needs collect.py nfl)

Mechanism being tested: a field goal is a drive that reaches scoring range and then
stalls. So FGs ~ (drives that reach range) x (1 - red-zone TD rate) x (coach kicks
rather than goes for it) x (kicker makes it). Each factor has a pre-game estimate:

  reach range   implied team total (market), offense trailing drives-to-range rate,
                opponent defense trailing range-allowed rate
  stall         offense red-zone TD rate, opponent red-zone TD rate allowed,
                opponent zone/blitz/box tendencies (bend-don't-break defenses)
  kick vs go    coach 4th-down go rate inside the 40 (trailing)
  make          kicker trailing FG%, wind, roof
  game script   |spread| (close games keep kicking late), total

Every trailing feature uses only EARLIER games (shift(1), exponentially weighted
across season boundaries, ~16-game memory).

PROTOCOL (fixed before any 2022+ result was looked at):
  fit  2016-2021      evaluate 2022-2025 once
  bet  one side per prop, only if model EV >= 5% at the quoted price, 1 unit flat

AMENDMENT (2026-10-12, after the `model` calibration table and BEFORE any prop
result was computed): the fixed 2016-21 fit under-predicts 2022-25 at every
quintile (league FG rate rose). Added walk-forward -- season Y predicted by a
fit on all seasons < Y -- as the primary rule; the fixed fit is still reported.

AMENDMENT 2 (2026-10-12, before any prop result): 2025 is the sealed season for
every props search (prop_loop.py), so `props` scores 2024 only (ESPN has no
kicker O/U lines for 2022-23). 2025 is scored once, later, by a human.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from sports import nflpbp

SEASONS = range(2016, 2026)
FIT, TEST = range(2016, 2022), range(2022, 2026)
OUT = ROOT / "lab" / "reports"
CACHE = ROOT / "data" / "nfl_fg_teamgames.parquet"
SPAN = 16
EV_MIN = 0.05
# ESPN -> nflverse team codes where they differ
ESPN2NFLV = {"WSH": "WAS", "LAR": "LA"}


def load_plays() -> pd.DataFrame:
    return pd.concat([nflpbp.plays(y) for y in SEASONS], ignore_index=True)


def team_games(p: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offense team) with outcomes and that game's tendencies."""
    p = p[p["season_type"] == "REG"].copy()
    p["fg_made"] = (p["field_goal_result"] == "made").astype(float)
    p["is_fga"] = (p["field_goal_attempt"] == 1).astype(float)
    p["zone"] = (p["defense_man_zone_type"] == "ZONE_COVERAGE").astype(float).where(
        p["defense_man_zone_type"].isin(["ZONE_COVERAGE", "MAN_COVERAGE"]))
    p["blitz"] = (p["n_blitzers"] > 0).astype(float).where(p["n_blitzers"].notna() & (p["pass"] == 1))
    p["box"] = p["defenders_in_box"].where(p["defenders_in_box"] > 0)
    p["go4_range"] = ((p["down"] == 4) & (p["yardline_100"] <= 40) & p["play_type"].isin(["pass", "run"])).astype(float)
    p["dec4_range"] = ((p["down"] == 4) & (p["yardline_100"] <= 40) &
                       p["play_type"].isin(["pass", "run", "field_goal"])).astype(float)
    off = p[p["posteam"].notna()]
    # drive level: did it reach the 30 (FG range) / the 20 (red zone); how did it end
    dr = off.groupby(["game_id", "posteam", "fixed_drive"]).agg(
        min_yl=("yardline_100", "min"), result=("fixed_drive_result", "last"))
    dr["range"] = dr["min_yl"] <= 30
    dr["rz"] = dr["min_yl"] <= 20
    dr["rz_td"] = dr["rz"] & (dr["result"] == "Touchdown")
    d = dr.groupby(["game_id", "posteam"]).agg(drives=("range", "size"), range_drives=("range", "sum"),
                                                rz_drives=("rz", "sum"), rz_tds=("rz_td", "sum"))
    g = off.groupby(["game_id", "posteam"]).agg(
        season=("season", "first"), week=("week", "first"), date=("game_date", "first"),
        defteam=("defteam", "first"), home_team=("home_team", "first"),
        spread_line=("spread_line", "first"), total_line=("total_line", "first"),
        roof=("roof", "first"), wind=("wind", "first"),
        fga=("is_fga", "sum"), fgm=("fg_made", "sum"),
        go4=("go4_range", "sum"), dec4=("dec4_range", "sum"),
        opp_zone=("zone", "mean"), opp_blitz=("blitz", "mean"), opp_box=("box", "mean"),
        epa_vs_zone=("epa", lambda s: s[off.loc[s.index, "zone"] == 1].mean()),
        epa_vs_man=("epa", lambda s: s[off.loc[s.index, "zone"] == 0].mean()))
    t = g.join(d).reset_index().rename(columns={"posteam": "team"})
    t["is_home"] = t["team"] == t["home_team"]
    # market: spread_line > 0 = home favoured; implied points for this team
    t["team_spread"] = np.where(t["is_home"], t["spread_line"], -t["spread_line"])
    t["implied_pts"] = (t["total_line"] + t["team_spread"]) / 2
    kick = off[off["is_fga"] == 1].groupby(["game_id", "posteam"])["kicker_player_name"].agg(
        lambda s: s.mode().iloc[0] if len(s) else None)
    t = t.merge(kick.rename("kicker").reset_index().rename(columns={"posteam": "team"}),
                on=["game_id", "team"], how="left")
    return t


def ewm_prior(t: pd.DataFrame, by: str, col: str) -> pd.Series:
    t = t.sort_values(["date", "game_id"])
    return t.groupby(by)[col].transform(lambda s: s.shift(1).ewm(span=SPAN, min_periods=4).mean())


def features(t: pd.DataFrame) -> pd.DataFrame:
    t = t.sort_values(["date", "game_id"]).reset_index(drop=True)
    t["range_rate"] = t["range_drives"] / t["drives"]
    t["rz_td_rate"] = t["rz_tds"] / t["rz_drives"].replace(0, np.nan)
    t["go_rate"] = t["go4"] / t["dec4"].replace(0, np.nan)
    t["fg_pct"] = t["fgm"] / t["fga"].replace(0, np.nan)
    f = pd.DataFrame(index=t.index)
    # offense (this team) priors
    for c in ("fgm", "range_rate", "rz_td_rate", "go_rate"):
        f[f"off_{c}"] = ewm_prior(t, "team", c)
    f["kick_pct"] = ewm_prior(t, "kicker", "fg_pct") if t["kicker"].notna().any() else np.nan
    # defense (opponent) priors: what it allowed / how it plays, keyed by defteam
    for c in ("fgm", "range_rate", "rz_td_rate", "opp_zone", "opp_blitz", "opp_box"):
        f[f"def_{c}"] = ewm_prior(t, "defteam", c)
    f["implied_pts"] = t["implied_pts"]
    f["abs_spread"] = t["team_spread"].abs()
    f["total_line"] = t["total_line"]
    f["outdoor_wind"] = np.where(t["roof"].isin(["outdoors", "open"]), t["wind"].fillna(0), 0)
    f["dome"] = t["roof"].isin(["dome", "closed"]).astype(float)
    return pd.concat([t, f.add_prefix("x_")], axis=1)


def fit_poisson(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    from scipy.optimize import minimize
    Xb = np.column_stack([np.ones(len(X)), X])

    def nll(b):
        eta = Xb @ b
        return np.sum(np.exp(eta) - y * eta) + l2 * np.sum(b[1:] ** 2)

    def grad(b):
        mu = np.exp(Xb @ b)
        g = Xb.T @ (mu - y)
        g[1:] += 2 * l2 * b[1:]
        return g
    return minimize(nll, np.zeros(Xb.shape[1]), jac=grad, method="L-BFGS-B").x


def design(df, cols, mu_sd=None):
    X = df[cols].astype(float)
    if mu_sd is None:
        mu_sd = (X.mean(), X.std().replace(0, 1))
    X = ((X - mu_sd[0]) / mu_sd[1]).fillna(0.0)  # missing prior -> league average
    return X.to_numpy(), mu_sd


def poisson_dev(y, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * np.sum(t - (y - mu))


MARKET_ONLY = ["x_implied_pts", "x_abs_spread", "x_total_line"]
FULL = MARKET_ONLY + ["x_off_fgm", "x_off_range_rate", "x_off_rz_td_rate", "x_off_go_rate", "x_kick_pct",
                      "x_def_fgm", "x_def_range_rate", "x_def_rz_td_rate", "x_def_opp_zone", "x_def_opp_blitz",
                      "x_def_opp_box", "x_outdoor_wind", "x_dome"]


def build() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    t = features(team_games(load_plays()))
    t.to_parquet(CACHE, index=False)
    return t


def fitted(t: pd.DataFrame, cols: list[str]):
    tr = t[t["season"].isin(FIT)]
    X, ms = design(tr, cols)
    b = fit_poisson(X, tr["fgm"].to_numpy())
    Xt, _ = design(t, cols, ms)
    return np.exp(np.column_stack([np.ones(len(Xt)), Xt]) @ b), b


def walk_forward(t: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """mu for each TEST season from a fit on every earlier season (NaN elsewhere)."""
    mu = np.full(len(t), np.nan)
    for y in TEST:
        tr = t[t["season"] < y]
        X, ms = design(tr, cols)
        b = fit_poisson(X, tr["fgm"].to_numpy())
        m = (t["season"] == y).to_numpy()
        Xt, _ = design(t[m], cols, ms)
        mu[m] = np.exp(np.column_stack([np.ones(len(Xt)), Xt]) @ b)
    return mu


def cmd_tendencies(t: pd.DataFrame) -> None:
    last = t["season"].max()
    s = t[t["season"] == last]
    d = s.groupby("defteam").agg(zone_rate=("opp_zone", "mean"), blitz_rate=("opp_blitz", "mean"),
                                 avg_box=("opp_box", "mean"), fg_allowed_pg=("fgm", "mean"),
                                 rz_td_allowed=("rz_tds", "sum"), rz_trips_allowed=("rz_drives", "sum"))
    d["rz_td_pct_allowed"] = d["rz_td_allowed"] / d["rz_trips_allowed"]
    o = s.groupby("team").agg(epa_vs_zone=("epa_vs_zone", "mean"), epa_vs_man=("epa_vs_man", "mean"))
    o["zone_minus_man"] = o["epa_vs_zone"] - o["epa_vs_man"]
    pd.set_option("display.width", 200)
    print(f"DEFENSE styles {last} (as defense):")
    print(d.drop(columns=["rz_td_allowed", "rz_trips_allowed"]).sort_values("zone_rate").round(3).to_string())
    print(f"\nOFFENSE EPA/play vs zone and vs man {last}:")
    print(o.sort_values("zone_minus_man").round(3).to_string())
    d.to_csv(OUT / f"nfl_def_tendencies_{last}.csv")
    o.to_csv(OUT / f"nfl_off_vs_coverage_{last}.csv")
    # does defensive style predict FGs allowed? (all seasons, team-season level)
    ts = t.groupby(["season", "defteam"]).agg(zone=("opp_zone", "mean"), blitz=("opp_blitz", "mean"),
                                              box=("opp_box", "mean"), fg=("fgm", "mean"),
                                              rz=("rz_tds", "sum"), rzd=("rz_drives", "sum")).dropna()
    ts["rz_td_pct"] = ts["rz"] / ts["rzd"]
    print("\nteam-season correlation of defensive style with FGs allowed / RZ TD% allowed "
          f"(n={len(ts)} team-seasons, 2016-{last}):")
    for c in ("zone", "blitz", "box"):
        r1 = stats.pearsonr(ts[c], ts["fg"])
        r2 = stats.pearsonr(ts[c], ts["rz_td_pct"])
        print(f"  {c:6s} vs FG/game r={r1[0]:+.2f} (p {r1[1]:.3f})   vs RZ TD% r={r2[0]:+.2f} (p {r2[1]:.3f})")


def cmd_model(t: pd.DataFrame) -> None:
    t = t[t["total_line"].notna()]
    te = t["season"].isin(TEST).to_numpy()
    y = t["fgm"].to_numpy()
    base = t.loc[t["season"].isin(FIT), "fgm"].mean()
    rows = []
    for name, cols in (("league mean", None), ("market only", MARKET_ONLY), ("full", FULL)):
        mu = np.full(len(t), base) if cols is None else fitted(t, cols)[0]
        rows.append({"model": name, "test_deviance": poisson_dev(y[te], mu[te]),
                     "test_mae": np.abs(y[te] - mu[te]).mean(),
                     "corr": np.corrcoef(y[te], mu[te])[0, 1] if cols else 0.0})
    r = pd.DataFrame(rows)
    r["dev_gain_vs_mean_%"] = 100 * (1 - r["test_deviance"] / r["test_deviance"].iloc[0])
    print(f"Out-of-sample 2022-2025 ({te.sum()} team-games), fit 2016-2021:")
    print(r.round(4).to_string(index=False))
    mu, b = fitted(t, FULL)
    print("\nstandardised coefficients (log-rate per 1 sd), full model:")
    for c, v in sorted(zip(FULL, b[1:]), key=lambda x: -abs(x[1])):
        print(f"  {c[2:]:<20} {v:+.3f}")
    # calibration of P(FGM >= 2) by decile
    t = t.assign(mu=mu)
    tt = t[te]
    tt = tt.assign(p2=1 - stats.poisson.cdf(1, tt["mu"]), hit=(tt["fgm"] >= 2).astype(float))
    tt["dec"] = pd.qcut(tt["p2"], 5, labels=False, duplicates="drop")
    print("\ncalibration P(>=2 FG made), test seasons, by model quintile:")
    print(tt.groupby("dec").agg(pred=("p2", "mean"), actual=("hit", "mean"), n=("hit", "size")).round(3).to_string())


def cmd_props(t: pd.DataFrame) -> None:
    espn = ROOT / "data" / "espn" / "nfl"
    gp = sorted(espn.glob("graded_*.parquet"))
    if not gp:
        print("no ESPN graded props yet: run collect.py all --league nfl first")
        return
    pr = pd.concat([pd.read_parquet(f) for f in gp], ignore_index=True)
    pr = pr[(pr["market_n"] == "total field goals made") & pr["profit"].notna() & (pr["side"] != "yes")]
    games = pd.concat([pd.read_parquet(f) for f in sorted(espn.glob("games_*.parquet"))])
    box = pd.concat([pd.read_parquet(f, columns=["event_id", "athlete_id", "team_id"]) for f in
                     sorted(espn.glob("box_*.parquet"))]).drop_duplicates(["event_id", "athlete_id"])
    pr = pr.merge(box, on=["event_id", "athlete_id"], how="left", suffixes=("", "_box"))
    g = games.set_index("event_id")
    pr["date"] = pd.to_datetime(pr["event_id"].map(g["start_utc"]), utc=True).dt.tz_convert(
        "America/New_York").dt.strftime("%Y-%m-%d")
    is_home = pr["team_id_box"].astype(str) == pr["event_id"].map(g["home_team_id"]).astype(str)
    pr["team"] = np.where(is_home, pr["event_id"].map(g["home"]), pr["event_id"].map(g["away"]))
    pr["team"] = pr["team"].replace(ESPN2NFLV)
    tl = t[t["total_line"].notna()].copy()
    tl["mu_fixed"] = fitted(tl, FULL)[0]
    tl["mu_wf"] = walk_forward(tl, FULL)
    tm = tl[["date", "team", "season", "mu_fixed", "mu_wf", "fgm"]].copy()
    tm["date"] = pd.to_datetime(tm["date"]).dt.strftime("%Y-%m-%d")
    j0 = pr.merge(tm, on=["date", "team"], how="inner")
    j0 = j0[j0["season"].isin(TEST) & (j0["season"] != 2025)]  # 2025 sealed
    for label, col in (("WALK-FORWARD (primary)", "mu_wf"), ("FIXED 2016-21 fit", "mu_fixed")):
        print(f"\n######## {label}")
        _prop_eval(j0.assign(mu=j0[col]))


def _prop_eval(j: pd.DataFrame) -> None:
    print(f"kicker FG props matched to model: {len(j)} sides "
          f"({j['event_id'].nunique()} games); actual vs pbp FGM agree: {(j['actual'] == j['fgm']).mean():.1%}")
    k = np.floor(j["line"])
    p_over = 1 - stats.poisson.cdf(k, j["mu"])
    j["p_model"] = np.where(j["side"] == "over", p_over, 1 - p_over)
    j["ev"] = j["p_model"] * j["dec"] - 1
    j["implied"] = 1 / j["dec"]
    best = j.sort_values("ev", ascending=False).drop_duplicates(["event_id", "athlete_id", "line"])
    bets = best[best["ev"] >= EV_MIN]
    print(f"\nall sides at quoted price: ROI {j['profit'].mean():+.2%} (n={len(j)})  -- the market's hold")
    print(f"model bets (EV >= {EV_MIN:.0%}): {len(bets)} bets, {bets['profit'].sum():+.1f}u, "
          f"ROI {bets['profit'].mean():+.2%}, win {bets['won'].mean():.1%}")
    if len(bets):
        print(bets.groupby(["season", "side"]).agg(n=("profit", "size"), units=("profit", "sum"),
                                                     roi=("profit", "mean")).round(3).to_string())
        boots = np.random.default_rng(0).choice(bets["profit"].to_numpy(), (5000, len(bets))).mean(1)
        print(f"  ROI 95% CI [{np.percentile(boots, 2.5):+.2%}, {np.percentile(boots, 97.5):+.2%}]")
    print("\ncalibration: model P vs implied P vs actual win rate, by side")
    print(j.groupby("side").agg(model=("p_model", "mean"), implied=("implied", "mean"),
                                actual=("won", "mean"), n=("won", "size")).round(3).to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["tendencies", "model", "props"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t = build()
    {"tendencies": cmd_tendencies, "model": cmd_model, "props": cmd_props}[a.cmd](t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
