"""First look: sanity checks, an Elo baseline, and the "is it priced in?" test.

    python scripts/baseline.py [--leagues NFL MLB NBA NHL]

Three questions, in the order that protects against fooling ourselves:

1. Do the data reproduce facts everyone already knows (home advantage, the
   empty-stadium 2020 season, Coors Field)? If not, the pipeline is wrong and
   nothing downstream is worth reading.
2. Does a simple a-priori Elo beat a coin flip out of sample? It should --
   good teams win. That is a positive control, not an edge.
3. NFL only, because nflverse ships closing lines: does a feature predict the
   outcome *net of the market*? A feature can predict results and still be
   worthless to a bettor, because the line already moved for it. Only the
   residual against the closing number is money.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sports import features, market

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(league: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"{league.lower()}_games.parquet")
    return df[df["home_score"].notna()].reset_index(drop=True)


def mean_ci(s: pd.Series) -> str:
    s = s.dropna()
    if len(s) < 2:
        return f"n={len(s)}"
    m, se = s.mean(), s.std(ddof=1) / np.sqrt(len(s))
    return f"{m:+.2f} [{m - 1.96 * se:+.2f}, {m + 1.96 * se:+.2f}]  n={len(s)}"


# ------------------------------------------------------------------ 1
def home_advantage(df: pd.DataFrame, league: str) -> pd.DataFrame:
    reg = df[(df["game_type"] == "regular") & ~df["neutral"].astype(bool)]
    y = market.home_result(reg)
    t = pd.DataFrame({"season": reg["season"], "home_win": y,
                      "margin": reg["home_score"] - reg["away_score"]})
    out = t.groupby("season").agg(games=("home_win", "size"),
                                  home_win_pct=("home_win", "mean"),
                                  home_margin=("margin", "mean")).round(3)
    print(f"\n[{league}] home advantage by season (regular season, non-neutral)")
    print(out.T.to_string())
    return out.assign(league=league)


# ------------------------------------------------------------------ 2
def elo_baseline(df: pd.DataFrame, league: str) -> dict:
    f = features.build(df)
    first = f["season"].min()
    test = f[(f["season"] > first) & (f["game_type"] == "regular")]  # first season = burn-in
    y = market.home_result(test)
    ok = y.notna()
    p, y = test.loc[ok, "p_home_elo"], y[ok]

    # Naive benchmark: the home-win rate of all *earlier* seasons (no peeking).
    hist = market.home_result(f[f["game_type"] == "regular"]).groupby(f["season"]).mean()
    prior = {s: hist[hist.index < s].mean() for s in test["season"].unique()}
    p_home_rate = test.loc[ok, "season"].map(prior)

    r = {
        "league": league, "n": int(ok.sum()),
        "seasons": f"{first + 1}-{f['season'].max()}",
        "logloss_coin": market.log_loss(np.full(len(y), 0.5), y),
        "logloss_home_rate": market.log_loss(p_home_rate, y),
        "logloss_elo": market.log_loss(p, y),
        "acc_elo": float(((p > 0.5) == (y == 1)).mean()),
    }
    print(f"\n[{league}] Elo (a-priori parameters), out of sample {r['seasons']}, n={r['n']}")
    print(f"  log loss: coin {r['logloss_coin']:.4f} | prior home rate "
          f"{r['logloss_home_rate']:.4f} | Elo {r['logloss_elo']:.4f}   accuracy {r['acc_elo']:.3f}")
    return r, f


# ------------------------------------------------------------------ 3 (NFL)
def nfl_market(f: pd.DataFrame) -> dict:
    g = f[(f["season"] > f["season"].min()) & f["home_ml"].notna() & f["away_ml"].notna()].copy()
    y = market.home_result(g)
    ok = y.notna()
    g, y = g[ok], y[ok]
    p_mkt, _, over = market.devig_two_way(g["home_ml"], g["away_ml"])
    bt = market.paired_logloss_bootstrap(g["p_home_elo"], p_mkt, y, n_boot=2000)
    print(f"\n[NFL] Elo vs the devigged closing moneyline, n={bt['n']}")
    print(f"  log loss: market {market.log_loss(p_mkt, y):.4f} | Elo {market.log_loss(g['p_home_elo'], y):.4f}")
    print(f"  Elo - market per game: {bt['observed_diff']:+.4f}  95% CI "
          f"[{bt['ci95'][0]:+.4f}, {bt['ci95'][1]:+.4f}]   (negative would mean Elo better)")
    print(f"  median moneyline overround (vig): {np.median(over):.2%}")
    return bt


def nfl_priced_in(f: pd.DataFrame) -> None:
    """Raw effect vs. effect net of the closing line, for three famous angles."""
    g = f[f["spread_line"].notna() & f["total_line"].notna()].copy()
    g["margin"] = g["home_score"] - g["away_score"]
    g["total"] = g["home_score"] + g["away_score"]
    g["ats_resid"] = g["margin"] - g["spread_line"]   # >0: home beat the spread
    g["tot_resid"] = g["total"] - g["total_line"]     # >0: went over

    print("\n[NFL] Is it priced in?  Each row: raw outcome, then outcome NET of the closing line.")
    print("      An angle is only bettable if the second column is reliably non-zero.")

    outdoor = g[g["roof"].isin(["outdoors", "open"]) & g["wind_mph"].notna()]
    print(f"\n  WIND (outdoor games with a reading, n={len(outdoor)})")
    print("    bin           total points                 total minus closing total")
    for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 99)]:
        b = outdoor[(outdoor["wind_mph"] >= lo) & (outdoor["wind_mph"] < hi)]
        print(f"    {lo:>2}-{hi:<3} mph  {mean_ci(b['total']):<30} {mean_ci(b['tot_resid'])}")

    print("\n  REST ADVANTAGE (home rest minus away rest, regular season)")
    print("    bin               home margin                  home margin minus spread")
    reg = g[g["game_type"] == "regular"]
    for name, m in [("away rested more (<= -4d)", reg["rest_diff"] <= -4),
                    ("even (-3..+3d)", reg["rest_diff"].between(-3, 3)),
                    ("home rested more (>= 4d)", reg["rest_diff"] >= 4)]:
        b = reg[m]
        print(f"    {name:<26} {mean_ci(b['margin']):<30} {mean_ci(b['ats_resid'])}")

    print("\n  HOME UNDERDOGS (spread_line < 0 means the home team is getting points)")
    for name, m in [("home favourite", g["spread_line"] > 0),
                    ("home underdog", g["spread_line"] < 0)]:
        b = g[m]
        won, lost = (b["ats_resid"] > 0).sum(), (b["ats_resid"] < 0).sum()
        print(f"    {name:<16} ATS residual {mean_ci(b['ats_resid']):<32} "
              f"home cover rate {won / (won + lost):.3f} ({won}-{lost}, pushes excluded)")
    print("    (break-even at -110 is a 52.4% cover rate)")


def mlb_conditions(f: pd.DataFrame) -> None:
    g = f[f["game_type"] == "regular"].copy()
    g["total"] = g["home_score"] + g["away_score"]
    sealed = g["condition"].isin(["Dome", "Roof Closed"])
    open_air = g[~sealed & g["temp_f"].notna()]
    print(f"\n[MLB] runs per game by first-pitch temperature (open-air games, n={len(open_air)})")
    for lo, hi in [(0, 55), (55, 65), (65, 75), (75, 85), (85, 120)]:
        b = open_air[(open_air["temp_f"] >= lo) & (open_air["temp_f"] < hi)]
        print(f"    {lo:>3}-{hi:<3}F  {mean_ci(b['total'])}")
    print(f"    roof sealed  {mean_ci(g.loc[sealed, 'total'])}")
    by_park = g.groupby("venue")["total"].agg(["mean", "size"])
    by_park = by_park[by_park["size"] >= 400].sort_values("mean")
    print("  highest- and lowest-scoring parks (>= 400 games):")
    for v, r in pd.concat([by_park.tail(3), by_park.head(3)]).iterrows():
        print(f"    {v:<28} {r['mean']:.2f} runs/game  n={int(r['size'])}")
    print("  (no MLB closing lines yet, so none of this says anything about value)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="+", default=["NFL", "MLB", "NBA", "NHL"])
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    ha, elos = [], []
    for lg in args.leagues:
        if not (DATA / f"{lg.lower()}_games.parquet").exists():
            print(f"\n[{lg}] not pulled yet; skipping")
            continue
        df = load(lg)
        ha.append(home_advantage(df, lg))
        r, f = elo_baseline(df, lg)
        elos.append(r)
        if lg == "NFL":
            r.update({f"vs_market_{k}": v for k, v in nfl_market(f).items()})
            nfl_priced_in(f)
        if lg == "MLB":
            mlb_conditions(f)

    if ha:
        pd.concat(ha).to_csv(RESULTS / "home_advantage.csv")
        pd.DataFrame(elos).to_csv(RESULTS / "elo_baseline.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
