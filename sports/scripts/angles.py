"""Test published NFL betting angles at the closing price, as money.

    python scripts/angles.py

Each angle is a rule that picks bets from pre-game information only. It is
scored by what it would actually have paid: flat 1-unit stakes at the closing
moneyline, or at -110 for spreads (nflverse's spread odds are not carried into
the common schema, so -110 is assumed and stated). ROI gets a bootstrap CI and
the set gets a Holm correction, because testing several angles and reporting
the best is the 22-config SMA sweep again.

The rules come from the literature, not from this data, so the period tested
here (2016 on) is new evidence for each of them. Published anomalies
typically shrink once published; that is the thing being measured.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sports import features, market

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
SPREAD_PAYOUT = 100 / 110  # profit per unit at -110


def roi_test(profit: pd.Series, n_boot: int = 5000, seed: int = 0) -> dict:
    p = profit.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(p, (n_boot, len(p))).mean(axis=1)
    return {"bets": len(p), "roi": p.mean(),
            "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
            "p_value": float(np.mean(boots <= 0))}  # one-sided: is ROI > 0?


def ml_profit(won: pd.Series, ml: pd.Series) -> pd.Series:
    dec = pd.Series(market.american_to_decimal(ml), index=ml.index)
    return np.where(won, dec - 1.0, -1.0)


def spread_profit(resid: pd.Series) -> pd.Series:
    """resid > 0 wins, < 0 loses, == 0 pushes (stake returned)."""
    return pd.Series(np.where(resid > 0, SPREAD_PAYOUT, np.where(resid < 0, -1.0, 0.0)),
                     index=resid.index)


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m, out, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        out[i] = running
    return out


def main() -> int:
    g = pd.read_parquet(DATA / "nfl_games.parquet")
    g = features.build(g[g["home_score"].notna()])
    g = g[(g["season"] >= 2016) & g["home_ml"].notna() & g["away_ml"].notna()].copy()
    g["margin"] = g["home_score"] - g["away_score"]
    g["p_home_mkt"], _, _ = market.devig_two_way(g["home_ml"], g["away_ml"])
    home_won, away_won = g["margin"] > 0, g["margin"] < 0
    ties = g["margin"] == 0

    angles: dict[str, pd.Series] = {}

    # 1. Away moneyline in competitive games (market p_home in [0.3, 0.7]).
    m = g["p_home_mkt"].between(0.3, 0.7) & ~ties
    angles["away ML, close games (p_home 0.3-0.7)"] = pd.Series(
        ml_profit(away_won[m], g.loc[m, "away_ml"]), index=g.index[m])

    # 2. Holdover bias: in week 1, bet against last season's playoff teams
    #    when they face a non-playoff team.
    allg = pd.read_parquet(DATA / "nfl_games.parquet")
    post = allg[allg["game_type"] == "post"]
    playoff = {(s + 1, t) for s, ts in post.groupby("season")
               for t in set(ts["home"]) | set(ts["away"])}
    reg = g[g["game_type"] == "regular"]
    wk1 = reg[reg["home_games_played"].eq(0) & reg["away_games_played"].eq(0)]
    hp = [(s, h) in playoff for s, h in zip(wk1["season"], wk1["home"])]
    ap = [(s, a) in playoff for s, a in zip(wk1["season"], wk1["away"])]
    hp, ap = np.array(hp), np.array(ap)
    resid = wk1["margin"] - wk1["spread_line"]
    fade = pd.concat([spread_profit(-resid[hp & ~ap]),   # bet the non-playoff away team
                      spread_profit(resid[ap & ~hp])])    # bet the non-playoff home team
    angles["week 1: fade last year's playoff teams (ATS)"] = fade

    # 3. Favourite-longshot: every favourite vs every underdog, moneyline.
    fav_home = g["p_home_mkt"] > 0.5
    nt = ~ties
    fav = pd.Series(np.where(fav_home, ml_profit(home_won, g["home_ml"]),
                             ml_profit(away_won, g["away_ml"])), index=g.index)[nt]
    dog = pd.Series(np.where(fav_home, ml_profit(away_won, g["away_ml"]),
                             ml_profit(home_won, g["home_ml"])), index=g.index)[nt]
    angles["all favourites, moneyline"] = fav
    angles["all underdogs, moneyline"] = dog
    big_dog = g["p_home_mkt"].lt(0.25) | g["p_home_mkt"].gt(0.75)
    angles["big underdogs only (p < 0.25), moneyline"] = dog[big_dog[nt]]

    rows = [{"angle": k, **roi_test(v)} for k, v in angles.items()]
    df = pd.DataFrame(rows)
    df["holm_p"] = holm(df["p_value"].tolist())

    print(f"NFL {g['season'].min()}-{g['season'].max()}, closing prices, flat 1-unit stakes\n")
    print(f"{'angle':<46}{'bets':>6}{'ROI':>9}{'95% CI':>20}{'p':>7}{'Holm':>7}")
    for r in df.itertuples():
        print(f"{r.angle:<46}{r.bets:>6}{r.roi:>+9.2%}   [{r.ci_lo:+.2%}, {r.ci_hi:+.2%}]"
              f"{r.p_value:>7.3f}{r.holm_p:>7.3f}")
    print("\nROI is per unit staked, after the vig. Zero is break-even; p tests ROI > 0.")

    # Calibration of the devigged close: is there a favourite-longshot bias at all?
    y = home_won[nt].astype(float)
    bins = pd.cut(g.loc[nt, "p_home_mkt"], [0, .2, .35, .5, .65, .8, 1])
    cal = pd.DataFrame({"p": g.loc[nt, "p_home_mkt"], "y": y, "bin": bins}) \
        .groupby("bin", observed=True).agg(n=("y", "size"), market=("p", "mean"), actual=("y", "mean"))
    cal["se"] = np.sqrt(cal["actual"] * (1 - cal["actual"]) / cal["n"])
    print("\nCalibration of the devigged closing moneyline (home win probability):")
    print(cal.round(3).to_string())

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "nfl_angles.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
