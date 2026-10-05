"""Does player/context data predict what the strikeout-prop MARKET misses?

    python scripts/props_residual.py --lines PATH/bt_ensemble_2025_edges.csv

Predicting strikeouts is the wrong target: the line already does that, better
than our model (props_backtest.py). The question with money in it is whether a
feature moves the outcome *after* conditioning on the market price. So:

    logit P(over) = logit(p_market) + b . features      (market is an offset)

fitted walk-forward by month through 2025 -- each month is predicted by a
model trained only on earlier 2025 months. If no feature carries information
the market lacks, the fitted b shrinks toward 0 and nothing beats the market.

Features (all knowable before first pitch, fixed before looking at results):
  disagree   logit(our model) - logit(market)
  ump_k      plate umpire's trailing K/PA vs league, shrunk (assignments are
             public the morning of the game)
  park_k     venue trailing K/PA vs league, shrunk
  temp, wind first-pitch observations (forecast proxy; see schema caveat)
  rest       days since the pitcher's previous start
  pitches3   mean pitch count over his previous 3 starts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from sports import market, props

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
FEATS = ["disagree", "ump_k", "park_k", "temp", "wind", "rest", "pitches3"]
EV_THRESHOLD = 0.05
L2 = 50.0  # ridge strength on standardised coefficients: shrink toward "market is right"


def trailing_rate(bat: pd.DataFrame, key: str, prior: float = 2000.0) -> pd.Series:
    """Per game: key's K/PA over prior 365 days (exclusive), shrunk, as a ratio to league."""
    g = bat.dropna(subset=[key]).groupby(["game_pk", key, "start_utc"])[["bat_k", "bat_pa"]].sum() \
        .reset_index().sort_values("start_utc")
    out = []
    for _, d in g.groupby(key, sort=False):
        d = d.set_index("start_utc")
        r = d[["bat_k", "bat_pa"]].rolling("365D", closed="left").sum()
        out.append(pd.DataFrame({"game_pk": d["game_pk"].to_numpy(),
                                 "k": r["bat_k"].fillna(0).to_numpy(),
                                 "pa": r["bat_pa"].fillna(0).to_numpy()}))
    t = pd.concat(out)
    lg = bat["bat_k"].sum() / bat["bat_pa"].sum()
    return t.set_index("game_pk").eval(f"(k + {prior} * {lg}) / (pa + {prior}) / {lg}").rename(key)


def fit_offset_logit(X, y, off, l2):
    def nll(b):
        z = off + X @ b
        return -(y * z - np.logaddexp(0, z)).sum() + l2 * (b @ b)
    def grad(b):
        p = expit(off + X @ b)
        return -X.T @ (y - p) + 2 * l2 * b
    return minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B").x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True, type=Path)
    args = ap.parse_args()

    pit = pd.read_parquet(DATA / "mlb_pitching.parquet")
    bat = pd.read_parquet(DATA / "mlb_team_batting.parquet")
    bat = bat[bat["game_type"] == "regular"].copy()
    bat["start_utc"] = pd.to_datetime(bat["start_utc"], utc=True)

    st = props.strikeout_features(pit, bat)
    fit = st[st["season"].between(2021, 2023)]
    league_bf = float(fit["bf"].mean())
    r = props.fit_dispersion(props.predict_mean(fit, league_bf).to_numpy(), fit["k"].to_numpy())

    st = st.sort_values("start_utc")
    gp = st.groupby("pitcher_id")
    st["rest"] = gp["start_utc"].diff().dt.total_seconds() / 86400
    st["pitches3"] = gp["pitches"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    ump = trailing_rate(bat, "officials")
    park = trailing_rate(bat, "venue")
    st = st.join(ump.groupby(level=0).first().rename("ump_k"), on="game_pk")
    st = st.join(park.groupby(level=0).first().rename("park_k"), on="game_pk")

    s25 = st[st["season"] == 2025].copy()
    s25["mu"] = props.predict_mean(s25, league_bf)
    lines = pd.read_csv(args.lines, low_memory=False,
                        usecols=["game_pk", "pitcher_id", "line", "over_odds", "under_odds", "fetched_at"])
    lines["fetched_at"] = pd.to_datetime(lines["fetched_at"], utc=True)
    L = lines.merge(s25, on=["game_pk", "pitcher_id"], how="inner")
    L = L[L["fetched_at"] < L["start_utc"]].drop_duplicates(["game_pk", "pitcher_id", "line"]).copy()

    L["p_mkt"], _, _ = market.devig_two_way(L["over_odds"], L["under_odds"])
    L["p_model"] = props.p_over(L["mu"], L["line"], r)
    eps = 1e-4
    L["disagree"] = logit(L["p_model"].clip(eps, 1 - eps)) - logit(L["p_mkt"].clip(eps, 1 - eps))
    L["temp"], L["wind"] = L["temp_f"], L["wind_mph"]
    L.loc[L["condition"].isin(["Dome", "Roof Closed"]), "wind"] = 0.0
    L["y"] = (L["k"] > L["line"]).astype(float)
    L["month"] = pd.to_datetime(L["date"]).dt.month
    L = L.dropna(subset=["p_mkt"]).copy()
    for c in FEATS:  # impute with the TRAINING-free median of all 2021-2024 starts where possible
        L[c] = L[c].fillna(st.loc[st["season"] < 2025, c].median() if c in st else L[c].median())

    # Walk-forward by month: train on earlier 2025 months only.
    preds = pd.Series(np.nan, index=L.index)
    coefs = []
    for m in sorted(L["month"].unique())[1:]:
        tr, te = L[L["month"] < m], L[L["month"] == m]
        mu_, sd_ = tr[FEATS].mean(), tr[FEATS].std().replace(0, 1)
        Xtr, Xte = ((tr[FEATS] - mu_) / sd_).to_numpy(), ((te[FEATS] - mu_) / sd_).to_numpy()
        off_tr = logit(tr["p_mkt"].clip(eps, 1 - eps)).to_numpy()
        off_te = logit(te["p_mkt"].clip(eps, 1 - eps)).to_numpy()
        b = fit_offset_logit(Xtr, tr["y"].to_numpy(), off_tr, L2)
        preds[te.index] = expit(off_te + Xte @ b)
        coefs.append(pd.Series(b, index=FEATS, name=m))

    T = L[preds.notna()].copy()
    T["p_blend"] = preds[T.index]
    bt = market.paired_logloss_bootstrap(T["p_blend"], T["p_mkt"], T["y"])
    print(f"walk-forward months {sorted(T['month'].unique())}, {len(T)} lines")
    print(f"log loss: market {market.log_loss(T['p_mkt'], T['y']):.4f} | market+features "
          f"{market.log_loss(T['p_blend'], T['y']):.4f} | diff {bt['observed_diff']:+.4f} "
          f"[{bt['ci95'][0]:+.4f}, {bt['ci95'][1]:+.4f}]  (negative = features add information)")
    print("\nstandardised coefficients by month (0 = market already has it):")
    print(pd.DataFrame(coefs).round(3).to_string())

    dec_o = market.american_to_decimal(T["over_odds"])
    dec_u = market.american_to_decimal(T["under_odds"])
    T["ev_o"], T["ev_u"] = T["p_blend"] * dec_o - 1, (1 - T["p_blend"]) * dec_u - 1
    T["side_over"] = T["ev_o"] >= T["ev_u"]
    T["ev"] = T[["ev_o", "ev_u"]].max(axis=1)
    T["won"] = np.where(T["side_over"], T["y"] == 1, T["y"] == 0)
    T["profit"] = np.where(T["won"], np.where(T["side_over"], dec_o, dec_u) - 1, -1.0)
    best = T.sort_values("ev", ascending=False).drop_duplicates(["game_pk", "pitcher_id"])
    bets = best[best["ev"] >= EV_THRESHOLD]
    by_day = bets.groupby("date")["profit"].sum().to_numpy()
    boots = np.random.default_rng(0).choice(by_day, (5000, len(by_day))).sum(axis=1)
    print(f"\nbetting market+features, EV >= {EV_THRESHOLD:.0%}: {len(bets)} bets, "
          f"{bets['profit'].sum():+.1f} u, ROI {bets['profit'].mean():+.2%}, "
          f"95% CI [{np.percentile(boots, 2.5):+.1f}, {np.percentile(boots, 97.5):+.1f}] u")
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(coefs).to_csv(RESULTS / "k_props_residual_coefs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
