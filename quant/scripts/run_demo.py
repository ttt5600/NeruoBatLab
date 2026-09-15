"""The experiment: how much of a backtest survives an honest evaluation.

Three parts.

A. OVERFITTING EXHIBIT -- sweep SMA crossover parameters on the full sample,
   report the winner's Sharpe, then deflate it for the number of trials and
   re-run the same grid walk-forward. The gap between those numbers is the
   cost of choosing parameters on the data you score them on.

B. HONEST HORSE RACE -- strategies whose parameters come from published work
   rather than from this sample, versus buy-and-hold, with block-bootstrap
   p-values on the difference.

C. NULL CONTROL -- the same sweep on synthetic geometric Brownian motion,
   where no timing edge exists by construction. Whatever Sharpe it "finds"
   there is the amount of Sharpe this procedure manufactures from nothing.

Run:  ../.venv_quant/bin/python scripts/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import strategies as S
from quant.backtest import CostModel, run_backtest
from quant.data import load_prices, synthetic_prices
from quant.stats import (
    bootstrap_sharpe_difference,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe,
    stationary_bootstrap_pvalue,
    summarise,
)
from quant.walkforward import expand_grid, grid_search_full_sample, walk_forward

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
START, END = "2007-01-01", "2026-09-12"
COSTS = CostModel()
RESULTS = Path(__file__).resolve().parents[1] / "results"

SMA_GRID = {
    "fast": [5, 10, 20, 50, 100],
    "slow": [50, 100, 150, 200, 250],
}


def _valid_sma(combos):
    return [c for c in combos if c["fast"] < c["slow"]]


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def fmt(d, keys):
    return "  ".join(f"{k}={d[k]:.3f}" for k in keys if k in d and np.isfinite(d[k]))


# --------------------------------------------------------------------------
def part_a(prices):
    banner("A. OVERFITTING EXHIBIT -- the same grid, scored two ways")

    combos = _valid_sma(expand_grid(SMA_GRID))
    print(f"grid: {len(combos)} valid (fast<slow) SMA crossover configurations")

    bench = run_backtest(prices, S.buy_and_hold(prices), costs=COSTS).net_returns
    print(f"benchmark: equal-weight buy-and-hold, Sharpe = {sharpe(bench):.3f}")

    rows, series, excess = [], {}, {}
    for p in combos:
        w = S.sma_crossover(prices, **p)
        res = run_backtest(prices, w, costs=COSTS)
        tag = f"{p['fast']}/{p['slow']}"
        series[tag] = res.net_returns
        excess[tag] = (res.net_returns - bench).dropna()
        rows.append({**p, "tag": tag, "sharpe": sharpe(res.net_returns),
                     "excess_sharpe": sharpe(excess[tag])})
    tbl = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    best = tbl.iloc[0]
    print(f"\nin-sample winner: SMA {best.tag}   Sharpe = {best.sharpe:.3f}")
    print(f"worst in grid:    SMA {tbl.iloc[-1].tag}   Sharpe = {tbl.iloc[-1].sharpe:.3f}")
    print("\ntop 5 configurations by full-sample Sharpe:")
    print(tbl.head(5).to_string(index=False))

    dsr = deflated_sharpe_ratio(series[best.tag], tbl["sharpe"].values)
    print("\n-- deflating that winner for the number of trials, vs ZERO --")
    print(f"  observed Sharpe             {dsr['sr_observed']:.3f}")
    print(f"  trials run                  {dsr['n_trials']}")
    print(f"  Sharpe reachable by luck    {dsr['sr_expected_max']:.3f}")
    print(f"  naive P(Sharpe>0)           {dsr['psr_vs_zero']:.3f}")
    print(f"  DSR vs zero                 {dsr['dsr']:.3f}")

    # Testing a long-only equity strategy against zero is the wrong question.
    # It is long stocks most of the time, so it collects the equity risk premium
    # whether or not the timing rule does anything. The null that matters is
    # buy-and-hold.
    dsr_x = deflated_sharpe_ratio(excess[best.tag], tbl["excess_sharpe"].values)
    print("\n-- the null that actually matters: EXCESS over buy-and-hold --")
    print(f"  excess Sharpe of the winner {dsr_x['sr_observed']:.3f}")
    print(f"  excess reachable by luck    {dsr_x['sr_expected_max']:.3f}")
    print(f"  DSR on excess returns       {dsr_x['dsr']:.3f}   <- what it is actually worth")

    pbo = probability_of_backtest_overfitting(pd.DataFrame(series), n_blocks=14, max_splits=3000)
    print(f"\n  PBO = {pbo['pbo']:.3f} over {pbo['n_splits']} splits "
          f"(0.5 = picking the in-sample winner tells you nothing)")

    print("\n-- now choosing parameters walk-forward instead --")
    wf = walk_forward(prices, S.sma_crossover, SMA_GRID, train_days=1260,
                      test_days=252, embargo_days=252, costs=COSTS)
    wf_sr = sharpe(wf.oos_returns)
    print(f"  folds: {wf.n_folds}   out-of-sample days: {len(wf.oos_returns)}")
    print(f"  walk-forward OOS Sharpe     {wf_sr:.3f}")
    print(f"  full-sample winner Sharpe   {best.sharpe:.3f}")
    print(f"  INFLATION FROM SELECTION    {best.sharpe - wf_sr:+.3f} Sharpe")

    if not wf.folds.empty:
        chosen = wf.folds[[c for c in wf.folds.columns if c.startswith("p_")]]
        print(f"\n  parameters chosen per fold (instability is itself the finding):")
        print("   " + wf.folds.assign(
            picked=chosen.astype(str).agg("/".join, axis=1)
        )[["test_start", "picked", "is_sharpe", "oos_sharpe"]].to_string(index=False).replace("\n", "\n   "))

    tbl.to_csv(RESULTS / "a_sma_grid_full_sample.csv", index=False)
    return {"best_tag": best.tag, "is_sharpe": float(best.sharpe), "wf_sharpe": float(wf_sr),
            "dsr_vs_zero": dsr["dsr"], "dsr_excess": dsr_x["dsr"],
            "excess_sharpe": dsr_x["sr_observed"], "pbo": pbo["pbo"]}


# --------------------------------------------------------------------------
def part_b(prices):
    banner("B. HONEST HORSE RACE -- a priori parameters, net of costs")

    bh_w = S.buy_and_hold(prices)
    bench = run_backtest(prices, bh_w, costs=COSTS).net_returns

    specs = {
        "buy_and_hold_EW": bh_w,
        "sixty_forty": S.sixty_forty(prices),
        "inverse_vol": S.inverse_vol(prices, lookback=60),
        "tsmom_12m_long_only": S.time_series_momentum(prices, 252, long_only=True),
        "tsmom_12m_long_short": S.time_series_momentum(prices, 252, long_only=False),
        "xsmom_12_1_top2": S.cross_sectional_momentum(prices, 252, 21, n_long=2),
        "meanrev_5d": S.mean_reversion_zscore(prices, lookback=5),
    }
    specs["inverse_vol__voltgt10"] = S.volatility_target(
        specs["inverse_vol"], prices, target_vol=0.10)
    specs["tsmom_12m_LO__voltgt10"] = S.volatility_target(
        specs["tsmom_12m_long_only"], prices, target_vol=0.10)

    bench_sr = sharpe(bench)
    rows = []
    for name, w in specs.items():
        res = run_backtest(prices, w, costs=COSTS)
        s = summarise(res)
        diff = (res.net_returns - bench).dropna()
        s["name"] = name
        # Signed excess makes the p-value readable: a one-sided p near 1.0 means
        # the strategy LOST to the benchmark, which is easy to misread as "very
        # significant" if you only look at the p column.
        s["p_mean_excess"] = stationary_bootstrap_pvalue(diff, n_boot=3000)["p_value"]
        d = bootstrap_sharpe_difference(res.net_returns, bench, n_boot=3000)
        s["vs_bh_sharpe"] = d["delta_sharpe"]
        s["ci_low"] = d["ci_low"]
        s["ci_high"] = d["ci_high"]
        s["p_beats_bh"] = d["p_value"]
        rows.append(s)

    tbl = pd.DataFrame(rows)

    # Nine strategies were tested against the same benchmark on the same data.
    # Reporting the one that cleared p<0.05 without saying so would be the exact
    # selection effect this whole script is about, committed one level up.
    tested = tbl[tbl["name"] != "buy_and_hold_EW"].copy()
    m = len(tested)
    order = tested["p_beats_bh"].rank(method="first")
    tested["p_holm"] = (tested["p_beats_bh"] * (m - order + 1)).clip(upper=1.0)
    tested["p_bh_fdr"] = (tested["p_beats_bh"] * m / order).clip(upper=1.0)
    tbl = tbl.merge(tested[["name", "p_holm", "p_bh_fdr"]], on="name", how="left")

    cols = ["name", "ann_return", "ann_vol", "sharpe", "sharpe_gross", "max_drawdown",
            "ann_turnover", "ann_cost_drag", "vs_bh_sharpe", "ci_low", "ci_high",
            "p_beats_bh", "p_mean_excess", "p_holm", "p_bh_fdr"]
    tbl = tbl[cols].sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 220)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print("\n  vs_bh_sharpe   = Sharpe minus equal-weight buy-and-hold, with 95% bootstrap CI.")
    print("  p_beats_bh     = one-sided P(delta Sharpe <= 0), paired block bootstrap.")
    print("                   Near 1.0 means it LOST, not that it won decisively.")
    print("  p_mean_excess  = the same test on MEAN RETURN instead of Sharpe. Compare the")
    print("                   two columns for risk-based books: cutting volatility lifts")
    print("                   Sharpe while leaving mean excess return at ~0, so the mean")
    print("                   test reports ~0.5 and misses the effect entirely.")
    print("  sharpe_gross vs sharpe = what friction removed.")
    print(f"  p_holm / p_bh_fdr = corrected for the {m} strategies tested here. A raw p just")
    print("                   under 0.05 out of 9 tries is roughly one expected false")
    print("                   positive -- the correction is what decides if it survives.")

    raw_hits = tbl[tbl["p_beats_bh"] < 0.05]["name"].tolist()
    holm_hits = tbl[tbl["p_holm"] < 0.05]["name"].tolist()
    fdr_hits = tbl[tbl["p_bh_fdr"] < 0.05]["name"].tolist()
    print(f"\n  beats buy-and-hold, raw p<0.05:        {raw_hits or 'none'}")
    print(f"  survives Holm (family-wise):           {holm_hits or 'none'}")
    print(f"  survives Benjamini-Hochberg (FDR):     {fdr_hits or 'none'}")
    tbl.to_csv(RESULTS / "b_horse_race.csv", index=False)
    return tbl


# --------------------------------------------------------------------------
def part_c():
    banner("C. NULL CONTROL -- the same sweep where no edge can exist")

    syn = synthetic_prices(n_days=len(pd.bdate_range(START, END)), n_assets=10, seed=42)
    combos = _valid_sma(expand_grid(SMA_GRID))
    rows, series = [], {}
    for p in combos:
        res = run_backtest(syn, S.sma_crossover(syn, **p), costs=COSTS)
        tag = f"{p['fast']}/{p['slow']}"
        series[tag] = res.net_returns
        rows.append({"tag": tag, "sharpe": sharpe(res.net_returns)})
    tbl = pd.DataFrame(rows).sort_values("sharpe", ascending=False)

    bh = sharpe(run_backtest(syn, S.buy_and_hold(syn), costs=COSTS).net_returns)
    best = tbl.iloc[0]
    dsr = deflated_sharpe_ratio(series[best.tag], tbl["sharpe"].values)
    pbo = probability_of_backtest_overfitting(pd.DataFrame(series), n_blocks=14, max_splits=3000)

    print("GBM has a fixed drift and NO timing signal at all -- by construction,")
    print("no SMA rule can have skill here. Yet:")
    print(f"  buy-and-hold Sharpe on the synthetic panel   {bh:.3f}")
    print(f"  best SMA config found by the sweep           {best.sharpe:.3f}  (SMA {best.tag})")
    print(f"  naive P(Sharpe>0) for that winner            {dsr['psr_vs_zero']:.3f}")
    print(f"  DSR vs zero for that winner                  {dsr['dsr']:.3f}")
    print(f"  PBO                                          {pbo['pbo']:.3f}")
    print(f"\n  THE LESSON: a strategy with provably zero timing skill still posts")
    print(f"  Sharpe {best.sharpe:.2f} and a DSR-vs-zero of {dsr['dsr']:.3f}. Significance against")
    print("  zero is nearly meaningless for a long-biased rule -- it certifies the")
    print("  drift, not the rule. Note it still fails to beat buy-and-hold")
    print(f"  ({best.sharpe:.3f} vs {bh:.3f}), which is the comparison that had any content.")
    tbl.to_csv(RESULTS / "c_null_control.csv", index=False)
    return {"bh": bh, "best": float(best.sharpe), "dsr": dsr["dsr"], "pbo": pbo["pbo"]}


def main():
    RESULTS.mkdir(exist_ok=True)
    print(f"universe: {', '.join(UNIVERSE)}")
    print(f"window:   {START} .. {END}")
    raw = load_prices(UNIVERSE, START, END)
    prices = raw.dropna(how="any")
    print(f"loaded:   {prices.shape[0]} trading days x {prices.shape[1]} assets "
          f"({prices.index[0].date()} .. {prices.index[-1].date()})")

    # A truncated panel is the failure mode that does not announce itself: one
    # short symbol silently amputates the common window and every number below
    # quietly describes a different, shorter history. Fail loudly instead.
    lost = (raw.index[-1] - raw.index[0]).days - (prices.index[-1] - prices.index[0]).days
    if lost > 370:
        short = {c: str(raw[c].dropna().index[0].date()) for c in raw.columns
                 if raw[c].dropna().index[0] > prices.index[0] + pd.Timedelta(days=5)}
        raise SystemExit(
            f"panel truncated by ~{lost} days vs requested window. Late starters: "
            f"{short}. Re-fetch with refresh=True or drop those symbols."
        )
    print(f"costs:    {COSTS.per_trade_bps:.1f} bps/trade, "
          f"{COSTS.short_borrow_bps_annual:.0f} bps/yr borrow, "
          f"{COSTS.financing_bps_annual:.0f} bps/yr financing")

    a = part_a(prices)
    b = part_b(prices)
    c = part_c()

    banner("VERDICT")
    print(f"  SMA sweep, full-sample winner        Sharpe {a['is_sharpe']:.3f}")
    print(f"  same grid, chosen walk-forward       Sharpe {a['wf_sharpe']:.3f}")
    print(f"  that winner's EXCESS over buy-hold   Sharpe {a['excess_sharpe']:+.3f}")
    print(f"  DSR vs zero (the flattering null)           {a['dsr_vs_zero']:.3f}")
    print(f"  DSR on excess (the honest null)             {a['dsr_excess']:.3f}")
    print(f"  PBO                                        {a['pbo']:.3f}")
    print(f"  identical sweep on pure noise        Sharpe {c['best']:.3f} (DSR {c['dsr']:.3f})")

    beat = b[b["p_beats_bh"] < 0.05]
    holm = b[b["p_holm"] < 0.05]
    print(f"\n  beat equal-weight buy-and-hold, raw p<0.05:  {len(beat)} of {len(b) - 1}"
          + (f"  ({', '.join(beat['name'])})" if len(beat) else ""))
    for _, r in beat.iterrows():
        print(f"    {r['name']:22s} dSharpe {r['vs_bh_sharpe']:+.3f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  "
              f"raw p={r['p_beats_bh']:.3f}  Holm p={r['p_holm']:.3f}  FDR p={r['p_bh_fdr']:.3f}")
    print(f"  survives multiple-testing correction:        "
          f"{', '.join(holm['name']) if len(holm) else 'NONE'}")
    print(f"\n  results written to {RESULTS}")


if __name__ == "__main__":
    main()
