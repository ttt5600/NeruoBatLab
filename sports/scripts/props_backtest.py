"""Backtest the strikeout-prop model against real 2025 lines, in units.

    python scripts/props_backtest.py --lines PATH/bt_ensemble_2025_edges.csv

Protocol, fixed BEFORE the 2025 results were looked at:

  fit     2021-2023  league starter BF and NegBin dispersion (2 numbers)
  check   2024       calibration only -- no lines needed, no betting
  bet     2025       real lines snapshotted ~4h before first pitch

  rule    per start, take the single side/line with the highest model EV at
          the quoted price; bet 1 unit if EV >= 5%. Flat stakes, no Kelly.

The threshold sweep printed afterwards is secondary: picking the best
threshold from it would be fitting the 2025 results, and the headline stays
the pre-registered 5%.

Line source: the public pitcherKModel repo (github.com/Msuresh32/pitcherKModel),
data/processed_ensemble_wf2025/bt_ensemble_2025_edges.csv. Only its line,
odds, book and timestamp columns are used -- none of its model outputs.
Prices are the BEST across ~9 US books, so results assume line shopping.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sports import market, props

DATA = Path(__file__).resolve().parents[1] / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
EV_THRESHOLD = 0.05


def load_lines(path: Path, starts: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False,
                      usecols=["game_pk", "pitcher_id", "pitcher_name", "strikeouts", "line",
                               "over_odds", "under_odds", "over_bookmaker", "under_bookmaker",
                               "fetched_at"])
    raw["fetched_at"] = pd.to_datetime(raw["fetched_at"], utc=True)
    df = raw.merge(starts, on=["game_pk", "pitcher_id"], how="inner", suffixes=("", "_ours"))
    n0 = len(df)
    df = df[df["fetched_at"] < df["start_utc"]]          # no price seen after first pitch
    df = df.drop_duplicates(["game_pk", "pitcher_id", "line"])
    mism = (df["strikeouts"] != df["k"]).sum()
    print(f"lines: {len(raw)} raw, {n0} matched to our starts, {len(df)} pre-game unique; "
          f"their K count disagrees with the boxscore on {mism} rows")
    return df


def block_bootstrap_units(profit: pd.Series, dates: pd.Series, n_boot=5000, seed=0):
    """Resample whole betting DAYS -- bets on one slate share weather, umps, news."""
    by_day = profit.groupby(dates).sum()
    v = by_day.to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, (n_boot, len(v))).sum(axis=1)
    return np.percentile(boots, [2.5, 97.5]), float(np.mean(boots <= 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True, type=Path)
    args = ap.parse_args()

    pit = pd.read_parquet(DATA / "mlb_pitching.parquet")
    bat = pd.read_parquet(DATA / "mlb_team_batting.parquet")
    st = props.strikeout_features(pit, bat)
    st = st[st["k"].notna() & st["bf"].notna()]

    # ---------------- fit: 2021-2023 ----------------
    fit = st[st["season"].between(2021, 2023)]
    league_bf = float(fit["bf"].mean())
    mu_fit = props.predict_mean(fit, league_bf)
    r = props.fit_dispersion(mu_fit.to_numpy(), fit["k"].to_numpy())
    print(f"fit 2021-2023: {len(fit)} starts, league starter BF {league_bf:.2f}, "
          f"NegBin r {r:.1f} (Poisson would be r=inf)")

    # ---------------- check: 2024 calibration ----------------
    chk = st[st["season"] == 2024].copy()
    chk["mu"] = props.predict_mean(chk, league_bf)
    print(f"\n2024 calibration, {len(chk)} starts: mean K actual {chk['k'].mean():.3f} "
          f"vs predicted {chk['mu'].mean():.3f}; corr(mu, K) {chk[['mu', 'k']].corr().iloc[0, 1]:.3f}")
    print("  line   P(over) predicted   actual        n")
    for line in (3.5, 4.5, 5.5, 6.5, 7.5):
        p = props.p_over(chk["mu"], np.full(len(chk), line), r)
        y = (chk["k"] > line).astype(float)
        for lo, hi in ((0, .35), (.35, .5), (.5, .65), (.65, 1)):
            m = (p >= lo) & (p < hi)
            if m.sum() >= 100:
                print(f"  {line:>4}   {p[m].mean():.3f} [{lo:.2f}-{hi:.2f})   {y[m].mean():.3f} "
                      f"± {1.96 * np.sqrt(y[m].mean() * (1 - y[m].mean()) / m.sum()):.3f}   {m.sum():>5}")

    # ---------------- bet: 2025 ----------------
    s25 = st[st["season"] == 2025].copy()
    s25["mu"] = props.predict_mean(s25, league_bf)
    L = load_lines(args.lines, s25[["game_pk", "pitcher_id", "start_utc", "date", "k", "mu",
                                    "pitcher", "team", "opponent"]])
    L["p_over_model"] = props.p_over(L["mu"], L["line"], r)
    L["p_over_mkt"], _, L["overround"] = market.devig_two_way(L["over_odds"], L["under_odds"])
    L["went_over"] = (L["k"] > L["line"]).astype(float)

    bt = market.paired_logloss_bootstrap(L["p_over_model"], L["p_over_mkt"], L["went_over"])
    print(f"\n2025 forecast quality on {bt['n']} lines (log loss, lower is better):")
    print(f"  market (devigged best prices) {market.log_loss(L['p_over_mkt'], L['went_over']):.4f}"
          f" | model {market.log_loss(L['p_over_model'], L['went_over']):.4f}"
          f" | model - market {bt['observed_diff']:+.4f} [{bt['ci95'][0]:+.4f}, {bt['ci95'][1]:+.4f}]")

    dec_o = market.american_to_decimal(L["over_odds"])
    dec_u = market.american_to_decimal(L["under_odds"])
    L["ev_over"] = L["p_over_model"] * dec_o - 1
    L["ev_under"] = (1 - L["p_over_model"]) * dec_u - 1
    L["side"] = np.where(L["ev_over"] >= L["ev_under"], "over", "under")
    L["ev"] = L[["ev_over", "ev_under"]].max(axis=1)
    L["won"] = np.where(L["side"] == "over", L["went_over"] == 1, L["went_over"] == 0)
    L["dec"] = np.where(L["side"] == "over", dec_o, dec_u)
    L["profit"] = np.where(L["won"], L["dec"] - 1, -1.0)
    best = L.sort_values("ev", ascending=False).drop_duplicates(["game_pk", "pitcher_id"])

    def report(name, bets):
        if bets.empty:
            print(f"  {name:<34} no bets")
            return {}
        units = bets["profit"].sum()
        ci, p = block_bootstrap_units(bets["profit"], bets["date"])
        cum = bets.sort_values("start_utc")["profit"].cumsum()
        dd = float((cum - cum.cummax()).min())
        print(f"  {name:<34} {len(bets):>5} bets  {bets['won'].mean():.3f} win  "
              f"{units:+8.1f} u  ROI {units / len(bets):+.2%}  95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}] u  "
              f"p(units<=0) {p:.3f}  max DD {dd:.1f} u")
        return {"rule": name, "bets": len(bets), "win_rate": bets["won"].mean(), "units": units,
                "roi": units / len(bets), "ci_lo": ci[0], "ci_hi": ci[1], "p": p, "max_dd": dd}

    rows = []
    print(f"\n2025 BETTING, flat 1 unit, best-of-books prices ~4h pre-game")
    print("PRE-REGISTERED:")
    rows.append(report(f"model EV >= {EV_THRESHOLD:.0%}", best[best["ev"] >= EV_THRESHOLD]))

    print("\nbaselines (no model) -- the main line per start = closest to a coin flip:")
    main = L.assign(gap=(L["p_over_mkt"] - 0.5).abs()).sort_values("gap") \
        .drop_duplicates(["game_pk", "pitcher_id"])
    for side in ("over", "under"):
        b = main.copy()
        b["won"] = (b["went_over"] == 1) if side == "over" else (b["went_over"] == 0)
        d = market.american_to_decimal(b[f"{side}_odds"])
        b["profit"] = np.where(b["won"], d - 1, -1.0)
        rows.append(report(f"always {side}, main line", b))

    print("\nsecondary -- threshold sweep (do NOT pick the best of these):")
    for t in (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
        rows.append(report(f"model EV >= {t:.0%}", best[best["ev"] >= t]))

    picked = best[best["ev"] >= EV_THRESHOLD].copy()
    picked["month"] = pd.to_datetime(picked["date"]).dt.to_period("M")
    print("\npre-registered rule by month (units):")
    print(picked.groupby("month")["profit"].agg(["size", "sum"]).round(1).T.to_string())
    print("\nby side:", picked.groupby("side")["profit"].agg(["size", "sum", "mean"]).round(3).to_dict("index"))

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame([x for x in rows if x]).to_csv(RESULTS / "k_props_2025.csv", index=False)
    picked.drop(columns=["month"]).to_csv(RESULTS / "k_props_2025_bets.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
