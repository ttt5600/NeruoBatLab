"""Confidence-based (fractional-Kelly) sizing for E1/E2 (Kalshi daily-high consensus-reject). DEV picks only.

    python scripts/kalshi_sizing.py build        # dev picks + decision-time features -> data/kalshi_sizing_dev.parquet
    python scripts/kalshi_sizing.py calib        # time-ordered out-of-fold calibration of P(NO wins), Wilson CIs
    python scripts/kalshi_sizing.py backtest     # walk-forward Kelly sizing vs flat; sensitivity grid; night bootstrap
    python scripts/kalshi_sizing.py fit          # freeze the PRIMARY calibrator on ALL dev -> data/kalshi_sizing_params.json
    python scripts/kalshi_sizing.py sealed-descriptive   # AFTER fit: frozen params on the already-scored sealed picks,
                                                         # descriptive only (no fitting, no selection, not a trial)

The rule is unchanged (picks = exactly data/kalshi_temp_maker_picks.parquet: E1 dev = kalshi_temp dev quarters through
lab/strategies/kalshi_reject_safe_check.py, X1 dev = wx_expand highs before 2026-03-01). Only the STAKE changes.

PRE-SPECIFIED 2026-10-27 (written after the descriptive loss-rate tables by gap/price/side on all dev picks were looked at
-- so the model FORMS were chosen with that look; the walk-forward is time-honest in its fits, not in its form choice):
  Calibrators of P(loss) (loss = YES wins), fit on picks with date < (decision month start - 1 day):
    ratio : P(loss) = theta * yes_bid  (yes_bid = 1 - NO ask = market-implied loss). One parameter; theta_hi = exact
            Poisson (Garwood) one-sided upper bound of losses / sum(yes_bid) at `level`.     <- PRIMARY
    logit : logistic(loss ~ logit(yes_bid) + gap), p_lo by the delta method on the linear predictor.
    bins  : NO-ask bins (<.90, .90-.95, .95-.97, .97-.98, .98-.99, >=.99), p_lo = Wilson lower bound of the bin win rate.
  Sizing f = k * (p_lo - q_eff) / (1 - q_eff), zero if p_lo <= q_eff (SKIP), capped at PICK_CAP of bankroll; a night's
  total is scaled down to NIGHT_CAP. q_eff = NO ask + 0.07*a*(1-a) (taker) or the maker price q (fee 0, M3, fill tape).
  PRIMARY: ratio, level 0.80 one-sided, k 0.25, PICK_CAP 0.05, NIGHT_CAP 0.10 (= executor NIGHT_FRAC). Everything else
  (k 0.10/0.50, level 0.50/0.90/0.95, logit, bins, skip-only, maker) is sensitivity, reported in full.
  Baseline FLAT = executor TIER_FRAC (orig7/E1 1%, x1/X1 0.5% of current bankroll), same night cap.
  Eval window 2025-01-01..2026-02-28 for every strategy (the first 10 months only train).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy import stats

DATA = ROOT / "data"
DEV = DATA / "kalshi_sizing_dev.parquet"
PARAMS = DATA / "kalshi_sizing_params.json"
FEE = 0.07
MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "ncep_nbm_conus"]
TIER_FRAC = {"E1": 0.01, "X1": 0.005, "orig7": 0.01, "x1": 0.005}
PRIMARY = {"model": "ratio", "level": 0.80, "k": 0.25, "pick_cap": 0.05, "night_cap": 0.10}
EVAL_START = pd.Timestamp("2025-01-01")
BINS = [0.0, 0.90, 0.95, 0.97, 0.98, 0.99, 1.0001]


# ------------------------------------------------------------------------------------------------ pure pieces
def pick_features(strike_type: str, floor: float, cap: float, forecasts) -> dict:
    """Decision-time features from the 5 forecasts. gap = distance (F) from the NEAREST model forecast to the bracket
    edge, in the rule's own convention (the rule fires iff gap > 3). side: bracket above or below the forecasts."""
    v = [float(x) for x in (forecasts.values() if isinstance(forecasts, dict) else forecasts)
         if x is not None and x == x]
    if not v:
        return {"gap": math.nan, "gap_mean": math.nan, "spread": math.nan, "side": None, "n_models": 0}
    hi, lo, mu = max(v), min(v), sum(v) / len(v)
    if strike_type == "greater":
        gap, gm, side = floor + 1 - hi, floor + 1 - mu, "above"
    elif strike_type == "less":
        gap, gm, side = lo - (cap - 1), mu - (cap - 1), "below"
    else:
        a, b = floor - hi, lo - cap
        gap, gm, side = max(a, b), max(floor - mu, mu - cap), ("above" if a > b else "below")
    return {"gap": gap, "gap_mean": gm, "spread": hi - lo, "side": side, "n_models": len(v)}


def taker_cost(no_ask):
    return no_ask + FEE * no_ask * (1 - no_ask)


def kelly_fraction(p_lo: float, q_eff: float, k: float = 0.25, cap: float = 0.05) -> float:
    """Fraction of bankroll to SPEND on a binary contract costing q_eff that pays 1 with probability >= p_lo."""
    if not (0 < q_eff < 1) or not (p_lo == p_lo) or p_lo <= q_eff:
        return 0.0
    return float(min(cap, k * (p_lo - q_eff) / (1 - q_eff)))


def wilson(succ, n, z):
    """Wilson score interval (lo, hi) for a proportion; z from the one-sided level."""
    if n == 0:
        return 0.0, 1.0
    ph = succ / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _logit(x):
    x = np.clip(np.asarray(x, float), 0.005, 0.995)
    return np.log(x / (1 - x))


# ------------------------------------------------------------------------------------------------ calibrators
def fit_cal(train: pd.DataFrame, model: str) -> dict:
    """Fit on rows with columns yes_bid, gap, loss. Returns a JSON-able parameter dict."""
    L, n = int(train.loss.sum()), len(train)
    if model == "ratio":
        return {"model": "ratio", "losses": L, "expected": float(train.yes_bid.sum()), "n": n}
    if model == "bins":
        b = pd.cut(1 - train.yes_bid, BINS, right=False)
        g = train.groupby(b, observed=False).loss.agg(["sum", "size"])
        return {"model": "bins", "edges": BINS, "losses": g["sum"].astype(int).tolist(), "n": g["size"].astype(int).tolist()}
    if model == "logit":
        import statsmodels.api as sm
        X = sm.add_constant(np.column_stack([_logit(train.yes_bid), np.minimum(train.gap.to_numpy(float), 10)]))
        r = sm.GLM(train.loss.to_numpy(float), X, family=sm.families.Binomial()).fit()
        return {"model": "logit", "beta": r.params.tolist(), "cov": r.cov_params().tolist(), "n": n, "losses": L}
    raise ValueError(model)


def predict(par: dict, yes_bid, gap, level: float) -> tuple[np.ndarray, np.ndarray]:
    """-> (p_hat, p_lo) of P(NO wins). p_lo is the one-sided `level` lower bound (level .5 -> ~point estimate)."""
    yb = np.asarray(yes_bid, float)
    z = stats.norm.ppf(level)
    if par["model"] == "ratio":
        L, E = par["losses"], max(par["expected"], 1e-9)
        th = L / E
        th_hi = (stats.chi2.ppf(level, 2 * (L + 1)) / 2) / E if level > 0.5 else th
        return 1 - np.minimum(th * yb, 1), 1 - np.minimum(th_hi * yb, 1)
    if par["model"] == "bins":
        idx = np.clip(np.searchsorted(par["edges"], 1 - yb, side="right") - 1, 0, len(par["n"]) - 1)
        ph, pl = np.empty(len(yb)), np.empty(len(yb))
        for i, j in enumerate(idx):
            n, L = par["n"][j], par["losses"][j]
            ph[i] = (n - L + 0.5) / (n + 1) if n else 0.0          # light smoothing; empty bin -> p 0 -> skip
            pl[i] = wilson(n - L, n, z)[0] if (n and level > 0.5) else ph[i]
        return ph, pl
    if par["model"] == "logit":
        b, C = np.array(par["beta"]), np.array(par["cov"])
        X = np.column_stack([np.ones(len(yb)), _logit(yb), np.minimum(np.asarray(gap, float), 10)])
        eta = X @ b
        se = np.sqrt(np.einsum("ij,jk,ik->i", X, C, X))
        lo_eta = eta + (z * se if level > 0.5 else 0)
        return 1 - 1 / (1 + np.exp(-eta)), 1 - 1 / (1 + np.exp(-lo_eta))
    raise ValueError(par["model"])


def size_pick(pick: dict, params: dict, bankroll: float, *, k: float = PRIMARY["k"], level: float = PRIMARY["level"],
              pick_cap: float = PRIMARY["pick_cap"], style: str = "taker", maker_q: float | None = None) -> dict:
    """PURE. One live pick (kalshi_paper.jsonl row: no_price, strike_type, floor, cap, forecasts) + frozen calibrator
    -> {p_hat, p_lo, q_eff, frac, usd, skip}. The executor still applies its own order/day/open/depth caps after this.
    For style='maker' pass the frozen `calibrator_maker` (fit on FILLED picks) and the resting price maker_q."""
    no_ask = float(pick["no_price"])
    ft = pick_features(pick.get("strike_type"), float(pick.get("floor") or math.nan), float(pick.get("cap") or math.nan),
                       pick.get("forecasts") or {})
    q_eff = float(maker_q) if style == "maker" and maker_q is not None else float(taker_cost(no_ask))
    ph, pl = predict(params, [1 - no_ask], [ft["gap"]], level)
    f = kelly_fraction(float(pl[0]), q_eff, k, pick_cap)
    return {"p_hat": round(float(ph[0]), 5), "p_lo": round(float(pl[0]), 5), "q_eff": round(q_eff, 5),
            "frac": round(f, 5), "usd": round(f * bankroll, 2), "skip": f <= 0, **{k_: ft[k_] for k_ in ("gap", "side")}}


# ------------------------------------------------------------------------------------------------ data
def cmd_build() -> int:
    import warnings
    warnings.filterwarnings("ignore")
    import importlib.util
    from lab import datasets as D
    import wx_expand as X
    spec = importlib.util.spec_from_file_location("rule", ROOT / "lab" / "strategies" / "kalshi_reject_safe_check.py")
    rule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rule)
    df = D.load("kalshi_temp", "dev")
    e = df[rule.bet(None, df.drop(columns=["won", "push", "actual"], errors="ignore")) > 0].rename(columns={"opp_id": "ticker"})
    e = e.assign(cohort="E1", won=e.won.astype(int))
    x = pd.read_parquet(X.OUT)
    x["date"] = pd.to_datetime(x["date"])
    s = X.picks(x[(x.date < X.SPLIT) & (x.kind == "high")])
    s = s.assign(cohort="X1", won=(s.result == "no").astype(int))
    rows = []
    for d in (e, s):
        for r in d.itertuples(index=False):
            fc = [getattr(r, "fcsafe_" + m) for m in MODELS]
            ft = pick_features(r.strike_type, r.floor_strike, r.cap_strike, fc)
            rows.append({"ticker": r.ticker, "date": pd.Timestamp(r.date), "cohort": r.cohort,
                         "series": r.ticker.split("-")[0], "strike_type": r.strike_type,
                         "yes_bid": float(r.yes_bid), "yes_ask": float(r.yes_ask), "won": int(r.won),
                         "volume_to_decision": float(r.volume_to_decision), **ft})
    out = pd.DataFrame(rows)
    out["loss"] = 1 - out.won
    out["no_ask"] = 1 - out.yes_bid
    ref = pd.read_parquet(DATA / "kalshi_temp_maker_picks.parquet")
    assert set(ref.ticker) == set(out.ticker) and len(ref) == len(out), "pick set differs from kalshi_temp_maker dev picks"
    out = out.merge(_maker_cols(ref), on="ticker", how="left")
    out.sort_values(["date", "ticker"]).to_parquet(DEV, index=False)
    print(f"{len(out)} dev picks ({out.loss.sum()} losses), {out.date.min().date()}..{out.date.max().date()} -> {DEV}")
    return 0


def _maker_cols(ref: pd.DataFrame) -> pd.DataFrame:
    """M3_N10 maker price and filled flag per pick (fill tape, kalshi_temp_maker.py conventions)."""
    import kalshi_temp_maker as KM
    tr = pd.read_parquet(KM.TRADES)
    q = KM.maker_price(ref, "3")
    f = KM.fill_flags(ref, tr, q, 10, ref.close_ts.to_numpy())
    return pd.DataFrame({"ticker": ref.ticker, "maker_q": q, "maker_filled": f})


def load_dev() -> pd.DataFrame:
    if not DEV.exists():
        cmd_build()
    return pd.read_parquet(DEV)


# ------------------------------------------------------------------------------------------------ walk-forward
def walk_forward(d: pd.DataFrame, model: str, level: float, style: str = "taker") -> pd.DataFrame:
    """Out-of-fold p_hat/p_lo: each calendar month is predicted from picks dated before (month start - 1 day).
    For maker, the calibrator is fit on FILLED picks only (the bet exists only if filled: P(win | filled))."""
    d = d.sort_values("date").reset_index(drop=True)
    fit_rows = d[d.maker_filled] if style == "maker" else d
    ph, pl = np.full(len(d), np.nan), np.full(len(d), np.nan)
    months = d.date.dt.to_period("M")
    for m in months.unique():
        cut = m.start_time - pd.Timedelta(days=1)
        tr = fit_rows[fit_rows.date < cut]
        idx = np.where(months == m)[0]
        if tr.loss.sum() < 3 or len(tr) < 50:
            continue
        par = fit_cal(tr, model)
        a, b = predict(par, d.yes_bid.to_numpy()[idx], d.gap.to_numpy()[idx], level)
        ph[idx], pl[idx] = a, b
    return d.assign(p_hat=ph, p_lo=pl)


def fractions(d: pd.DataFrame, how: str, k: float, pick_cap: float, style: str = "taker") -> np.ndarray:
    q = d.maker_q.to_numpy(float) if style == "maker" else taker_cost(d.no_ask.to_numpy(float))
    if how == "flat":
        return d.cohort.map(TIER_FRAC).to_numpy(float)
    if how == "skip_only":  # flat size, but skip when the calibrated lower bound is below the cost
        return np.where(d.p_lo.to_numpy() > q, d.cohort.map(TIER_FRAC).to_numpy(float), 0.0)
    return np.array([kelly_fraction(p, c, k, pick_cap) for p, c in zip(d.p_lo, q)])


def night_returns(d: pd.DataFrame, f: np.ndarray, night_cap: float, style: str = "taker") -> pd.DataFrame:
    """Per night: scale fractions to the night cap; return = sum f_i * pnl_i per $ of bankroll."""
    q = d.maker_q.to_numpy(float) if style == "maker" else taker_cost(d.no_ask.to_numpy(float))
    live = d.maker_filled.to_numpy(bool) if style == "maker" else np.ones(len(d), bool)
    f = np.where(live, f, 0.0)
    pnl = np.where(d.won.to_numpy() == 1, 1 / q - 1, -1.0)
    t = pd.DataFrame({"date": d.date.to_numpy(), "f": f, "fp": f * pnl})
    g = t.groupby("date").agg(f=("f", "sum"), fp=("fp", "sum"))
    scale = np.where(g.f > night_cap, night_cap / g.f.where(g.f > 0, 1), 1.0)
    return pd.DataFrame({"staked": g.f * scale, "ret": g.fp * scale}, index=g.index)


def path_stats(r: np.ndarray) -> dict:
    eq = np.cumprod(1 + r)
    dd = 1 - eq / np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return {"multiple": float(eq[-1]) if len(eq) else 1.0, "max_dd": float(dd.max()) if len(dd) else 0.0,
            "worst_night": float(r.min()) if len(r) else 0.0}


def evaluate(d, how, model, level, k, pick_cap, night_cap, style="taker", rng=None, nboot=2000) -> dict:
    w = walk_forward(d, model, level, style) if how != "flat" else d.sort_values("date").reset_index(drop=True)
    w = w[w.date >= EVAL_START].reset_index(drop=True)
    if how != "flat" and w.p_lo.isna().any():
        raise RuntimeError("eval rows without a fitted calibrator")
    f = fractions(w, how, k, pick_cap, style)
    live = w.maker_filled.to_numpy(bool) if style == "maker" else np.ones(len(w), bool)
    nr = night_returns(w, f, night_cap, style)
    r = nr.ret.to_numpy()
    st = path_stats(r)
    rng = rng or np.random.default_rng(0)
    bs = np.array([[v for v in path_stats(r[rng.integers(0, len(r), len(r))]).values()] for _ in range(nboot)])
    bet = (f > 0) & live
    q = w.maker_q.to_numpy(float) if style == "maker" else taker_cost(w.no_ask.to_numpy(float))
    pnl = np.where(w.won.to_numpy() == 1, 1 / q - 1, -1.0)
    return {"how": how, "model": model if how != "flat" else "-", "level": level, "k": k, "style": style,
            "picks": int(live.sum()), "bets": int(bet.sum()), "skipped": float(1 - bet.sum() / max(live.sum(), 1)),
            "losses_bet": int(((w.won == 0).to_numpy() & bet).sum()), "losses_skipped": int(((w.won == 0).to_numpy() & live & ~bet).sum()),
            "unit_roi_bet": float(pnl[bet].mean()) if bet.any() else 0.0,
            "unit_roi_skipped": float(pnl[live & ~bet].mean()) if (live & ~bet).any() else float("nan"),
            "mean_frac_bet": float(f[bet].mean()) if bet.any() else 0.0, "avg_night_staked": float(nr.staked.mean()),
            "log_growth": float(np.log1p(r).sum()), **st,
            "boot_med_multiple": float(np.median(bs[:, 0])), "boot_p05_multiple": float(np.quantile(bs[:, 0], .05)),
            "P_final_below_1": float(np.mean(bs[:, 0] < 1)), "P_dd_gt_20": float(np.mean(bs[:, 1] > .20)),
            "P_dd_gt_40": float(np.mean(bs[:, 1] > .40)), "nights": int(len(r))}


# ------------------------------------------------------------------------------------------------ reports
def calib_table(w: pd.DataFrame, col: str, edges, z: float) -> pd.DataFrame:
    w = w.dropna(subset=[col])
    b = pd.cut(w[col], edges, right=False)
    rows = []
    for iv, g in w.groupby(b, observed=True):
        L, n = int(g.loss.sum()), len(g)
        lo, hi = wilson(L, n, z)
        rows.append({"bin": str(iv), "n": n, "losses": L, "loss_per_1000": round(1000 * L / n, 1),
                     "wilson95_per_1000": f"[{1000 * lo:.1f}, {1000 * hi:.1f}]",
                     "pred_loss_per_1000": round(1000 * (1 - g[col]).mean(), 1),
                     "market_yes_bid_per_1000": round(1000 * g.yes_bid.mean(), 1),
                     "taker_breakeven_per_1000": round(1000 * (1 - taker_cost(g.no_ask)).mean(), 1)})
    return pd.DataFrame(rows)


def cmd_calib() -> int:
    d = load_dev()
    z95 = stats.norm.ppf(0.975)
    edges = [0, .90, .95, .97, .98, .99, .995, .999, 1.0001]
    print(f"DEV {len(d)} picks, {int(d.loss.sum())} losses, {d.date.min().date()}..{d.date.max().date()}")
    print("\n== raw: loss rate by NO ask (all dev, no model) ==")
    print(calib_table(d.assign(p_mkt=d.no_ask), "p_mkt", edges, z95).to_string(index=False))
    out = {}
    for model in ("ratio", "logit", "bins"):
        w = walk_forward(d, model, 0.5)
        w = w[w.date >= EVAL_START]
        ll = -np.mean(w.loss * np.log(1 - w.p_hat.clip(1e-6, 1 - 1e-6)) + (1 - w.loss) * np.log(w.p_hat.clip(1e-6, 1 - 1e-6)))
        llm = -np.mean(w.loss * np.log(w.yes_bid.clip(1e-6)) + (1 - w.loss) * np.log(1 - w.yes_bid.clip(0, 1 - 1e-6)))
        print(f"\n== {model}: OUT-OF-FOLD (fit on earlier months only), eval {EVAL_START.date()}+, n={len(w)}; "
              f"log-loss model {ll:.4f} vs market yes_bid {llm:.4f} ==")
        print(calib_table(w, "p_hat", edges, z95).to_string(index=False))
        out[model] = {"logloss": ll, "logloss_market": llm}
    w = walk_forward(d, "logit", 0.5)
    par = fit_cal(d, "logit")
    se = np.sqrt(np.diag(par["cov"]))
    print("\nlogit on ALL dev (descriptive): beta", np.round(par["beta"], 3), "se", np.round(se, 3),
          "(const, logit(yes_bid), gap)")
    pr = fit_cal(d, "ratio")
    print(f"ratio on ALL dev: losses {pr['losses']} / market-implied {pr['expected']:.1f} -> theta {pr['losses'] / pr['expected']:.3f}")
    for c in ("E1", "X1"):
        pc = fit_cal(d[d.cohort == c], "ratio")
        print(f"  {c}: losses {pc['losses']} / implied {pc['expected']:.1f} -> theta {pc['losses'] / pc['expected']:.3f}")
    for s_ in ("above", "below"):
        pc = fit_cal(d[d.side == s_], "ratio")
        print(f"  side {s_}: losses {pc['losses']} / implied {pc['expected']:.1f} -> theta {pc['losses'] / pc['expected']:.3f}")
    return 0


def sizing_report(d: pd.DataFrame | None = None) -> pd.DataFrame:
    d = load_dev() if d is None else d
    P = PRIMARY
    runs = [("flat", "-", P["level"], P["k"], "taker"),
            ("kelly", "ratio", P["level"], P["k"], "taker"),
            ("skip_only", "ratio", P["level"], P["k"], "taker")]
    runs += [("kelly", "ratio", P["level"], k, "taker") for k in (0.10, 0.50)]
    runs += [("kelly", "ratio", lv, P["k"], "taker") for lv in (0.50, 0.90, 0.95)]
    runs += [("kelly", m, P["level"], P["k"], "taker") for m in ("logit", "bins")]
    runs += [("skip_only", m, P["level"], P["k"], "taker") for m in ("logit", "bins")]
    runs += [("flat", "-", P["level"], P["k"], "maker"), ("kelly", "ratio", P["level"], P["k"], "maker"),
             ("skip_only", "ratio", P["level"], P["k"], "maker")]
    res = [evaluate(d, h, m, lv, k, P["pick_cap"], P["night_cap"], st, np.random.default_rng(0)) for h, m, lv, k, st in runs]
    return pd.DataFrame(res)


def cmd_backtest() -> int:
    d = load_dev()
    r = sizing_report(d)
    pd.set_option("display.width", 250)
    cols = ["how", "model", "level", "k", "style", "picks", "bets", "skipped", "losses_bet", "losses_skipped",
            "unit_roi_bet", "unit_roi_skipped", "mean_frac_bet", "avg_night_staked", "multiple", "log_growth", "max_dd",
            "worst_night", "boot_med_multiple", "boot_p05_multiple", "P_final_below_1", "P_dd_gt_20", "P_dd_gt_40"]
    print(r[cols].round(4).to_string(index=False))
    r.to_json(DATA / "kalshi_sizing_backtest.json", orient="records", indent=1)
    # where do the skips fall? (primary)
    w = walk_forward(d, "ratio", PRIMARY["level"])
    w = w[w.date >= EVAL_START].copy()
    w["f"] = fractions(w, "kelly", PRIMARY["k"], PRIMARY["pick_cap"])
    w["bin"] = pd.cut(w.no_ask, BINS, right=False)
    q = taker_cost(w.no_ask)
    w["pnl"] = np.where(w.won == 1, 1 / q - 1, -1.0)
    g = w.groupby("bin", observed=True).agg(n=("f", "size"), losses=("loss", "sum"), bet_share=("f", lambda x: (x > 0).mean()),
                                            mean_frac=("f", "mean"), max_frac=("f", "max"), unit_roi=("pnl", "mean"))
    print("\nPRIMARY by NO-ask bin (eval window):\n", g.round(4).to_string())
    return 0


def cmd_fit() -> int:
    d = load_dev()
    par = fit_cal(d, PRIMARY["model"])
    pm = fit_cal(d[d.maker_filled], PRIMARY["model"])  # maker: P(win | filled), adverse selection included
    out = {"frozen_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"), "fit_on": "all dev picks",
           "n": len(d), **PRIMARY, "calibrator": par, "calibrator_maker": pm}
    PARAMS.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    for na in (0.80, 0.90, 0.95, 0.97, 0.98, 0.99):
        r = size_pick({"no_price": na, "strike_type": "greater", "floor": 90, "cap": None,
                       "forecasts": {"a": 80, "b": 81, "c": 82}}, par, 400.0)
        print(f"NO ask {na:.2f}: p_hat {r['p_hat']:.4f} p_lo {r['p_lo']:.4f} cost {r['q_eff']:.4f} -> frac {r['frac']:.4f} "
              f"(${r['usd']:.2f} of $400){'  SKIP' if r['skip'] else ''}")
    return 0


def cmd_sealed_descriptive() -> int:
    """Frozen params on the sealed picks (already scored once for MAKER). Descriptive only."""
    if not PARAMS.exists():
        print("run fit first (params must be frozen before this is looked at)")
        return 1
    fz = json.loads(PARAMS.read_text())
    s = pd.read_parquet(DATA / "kalshi_temp_maker_picks_sealed.parquet").sort_values("date").reset_index(drop=True)
    s = s.assign(no_ask=1 - s.yes_bid, loss=1 - s.won, gap=np.nan, maker_q=np.nan, maker_filled=False)
    ph, pl = predict(fz["calibrator"], s.yes_bid, s.gap, fz["level"])
    s["p_hat"], s["p_lo"] = ph, pl
    print(f"SEALED (descriptive, frozen {fz['frozen_at']}): {len(s)} picks, {int(s.loss.sum())} losses, "
          f"{s.date.min().date()}..{s.date.max().date()}")
    print(calib_table(s, "p_hat", [0, .90, .95, .97, .98, .99, .995, 1.0001], stats.norm.ppf(.975)).to_string(index=False))
    th = fz["calibrator"]["losses"] / fz["calibrator"]["expected"]
    print(f"sealed losses / market-implied = {s.loss.sum()} / {s.yes_bid.sum():.1f} = {s.loss.sum() / s.yes_bid.sum():.3f} "
          f"(dev theta {th:.3f})")
    res = []
    for how in ("flat", "kelly", "skip_only"):
        f = fractions(s, how, fz["k"], fz["pick_cap"])
        r = night_returns(s, f, fz["night_cap"]).ret.to_numpy()
        q = taker_cost(s.no_ask.to_numpy())
        pnl = np.where(s.won == 1, 1 / q - 1, -1.0)
        bet = f > 0
        res.append({"how": how, "bets": int(bet.sum()), "skipped": float(1 - bet.mean()), "losses_bet": int((s.loss[bet]).sum()),
                    "unit_roi_bet": float(pnl[bet].mean()), "unit_roi_skipped": float(pnl[~bet].mean()) if (~bet).any() else math.nan,
                    "log_growth": float(np.log1p(r).sum()), **path_stats(r)})
    print(pd.DataFrame(res).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "calib"
    sys.exit({"build": cmd_build, "calib": cmd_calib, "backtest": cmd_backtest, "fit": cmd_fit,
              "sealed-descriptive": cmd_sealed_descriptive}[cmd]())
