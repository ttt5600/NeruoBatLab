"""Stock strategy backtests on the frozen daily snapshot -- same honesty rules as the sports lab.

    python scripts/stocks_bt.py dev        # score every registered trial on DEV years, append ledger

Rules:
  * reads data/stocks/frozen_*.parquet (checksum verified); DEV = 2006-2018 only.
    2019-2025 is SEALED: dev mode drops those rows at load time, before any signal is built.
  * no lookahead: a position is decided from data up to the close of day t and earns the
    close(t) -> close(t+1) return; COST_BPS is charged on every change in position.
  * the question is "does it beat just holding the same thing", so each strategy is regressed
    on its own buy-and-hold benchmark; the edge is the intercept (alpha), HAC t-stat, one-sided p.
  * every trial is appended to lab/stocks_ledger.jsonl; the bar is p < 0.05 / (trials so far).
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "lab" / "stocks_ledger.jsonl"
DEV = ("2006-01-01", "2018-12-31")
COST_BPS = 2.0
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
MEGA_TECH = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]  # META starts 2012; left out of the dev signal


def load(window):
    path = sorted((ROOT / "data" / "stocks").glob("frozen_*.parquet"))[-1]
    want = path.with_suffix(".sha256").read_text().strip()
    if hashlib.sha256(path.read_bytes()).hexdigest() != want:
        sys.exit(f"{path} does not match its checksum; refusing to score")
    df = pd.read_parquet(path)
    df = df[(df.date >= window[0]) & (df.date <= window[1])]
    px = {f: df.pivot(index="date", columns="ticker", values=f) for f in ["open", "high", "low", "close"]}
    return px, path.name


def run_positions(pos: pd.DataFrame, close: pd.DataFrame) -> pd.Series:
    """pos[t] = weight held from close t to close t+1 (rows sum <= 1). Returns daily portfolio return."""
    ret = close.pct_change()
    held = pos.shift(1)
    gross = (held * ret[pos.columns]).sum(axis=1, min_count=1)
    cost = pos.diff().abs().sum(axis=1).shift(1) * COST_BPS / 1e4
    return (gross - cost).dropna()


def score(name, params, strat: pd.Series, bench: pd.Series, exposure: float):
    d = pd.concat([strat.rename("s"), bench.rename("b")], axis=1).dropna()
    fit = sm.OLS(d.s, sm.add_constant(d.b)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    a, t = fit.params["const"], fit.tvalues["const"]

    def summ(r):
        eq = (1 + r).cumprod()
        yrs = len(r) / 252
        return {"cagr": eq.iloc[-1] ** (1 / yrs) - 1, "sharpe": r.mean() / r.std() * np.sqrt(252),
                "maxdd": (eq / eq.cummax() - 1).min()}

    s, b = summ(d.s), summ(d.b)
    return {"strategy": name, "params": params, "days": len(d), "exposure": round(exposure, 3),
            "alpha_ann": a * 252, "alpha_t": t, "p": float(stats.norm.sf(t)), "beta": fit.params["b"],
            "cagr": s["cagr"], "sharpe": s["sharpe"], "maxdd": s["maxdd"],
            "bh_cagr": b["cagr"], "bh_sharpe": b["sharpe"], "bh_maxdd": b["maxdd"]}


# ---------------------------------------------------------------- strategies
def dip_rip(px, asset, band, lo=0.5, n=20):
    """User's rule: trim to `lo` when price runs `band` above its n-day mean, back to 100% on a `band` dip."""
    c = px["close"][asset]
    ma = c.rolling(n).mean()
    w = pd.Series(np.nan, index=c.index)
    w[c > ma * (1 + band)] = lo
    w[c < ma * (1 - band)] = 1.0
    w = w.ffill().fillna(1.0)
    pos = w.to_frame(asset)
    return run_positions(pos, px["close"]), run_positions(pd.DataFrame(1.0, index=c.index, columns=[asset]), px["close"]), w.mean()


def tech_gate(px, signal):
    """Hold the 8 non-tech sectors tomorrow only if tech rose today (tech 'feeds' the rest)."""
    others = [s for s in SECTORS if s != "XLK"]
    c = px["close"]
    sig = c[signal].pct_change().mean(axis=1) if isinstance(signal, list) else c[signal].pct_change()
    on = (sig > 0).astype(float)
    pos = pd.DataFrame({s: on / len(others) for s in others})
    bh = pd.DataFrame(1.0 / len(others), index=c.index, columns=others)
    return run_positions(pos, c), run_positions(bh, c), on.mean()


def graph_leadlag(px, k=3, window=252, lam=1e-3):
    """Learned lead-lag graph: B[i,j] = how sector i's return today predicts sector j's tomorrow.
    Ridge-fit on the trailing `window` days, refit at each month end, applied out of sample next month.
    Hold the top-k sectors whose predicted return is > 0."""
    c = px["close"][SECTORS]
    r = c.pct_change().dropna()
    pos = pd.DataFrame(0.0, index=r.index, columns=SECTORS)
    month_ends = r.index[r.index.to_series().dt.month.diff().shift(-1).fillna(1) != 0]
    for i, m in enumerate(month_ends[:-1]):
        hist = r.loc[:m].iloc[-window - 1:]
        if len(hist) < window:
            continue
        X, Y = hist.values[:-1], hist.values[1:]
        B = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
        days = r.index[(r.index >= m) & (r.index < month_ends[i + 1])]
        for t in days:
            pred = r.loc[t].values @ B
            top = np.argsort(pred)[::-1][:k]
            top = [j for j in top if pred[j] > 0]
            if top:
                pos.loc[t, [SECTORS[j] for j in top]] = 1.0 / k
    pos = pos.loc[pos.index >= month_ends[0]]
    bh = pd.DataFrame(1.0 / len(SECTORS), index=pos.index, columns=SECTORS)
    return run_positions(pos, c), run_positions(bh, c), (pos.sum(axis=1)).mean()


def loss_cap(px, asset, cap):
    """Buy-and-hold with a hard daily stop: if today's low breaches prev_close*(1-cap), sell at the stop
    (or at the open if it gapped through), re-buy at the close. Fill at the stop exactly is OPTIMISTIC."""
    o, l, c = px["open"][asset], px["low"][asset], px["close"][asset]
    pc = c.shift(1)
    stop = pc * (1 - cap)
    hit = l <= stop
    exit_px = np.minimum(o, stop)
    r = c / pc - 1
    r[hit] = exit_px[hit] / pc[hit] - 1 - 2 * COST_BPS / 1e4  # sell + re-buy
    return r.dropna(), (c / pc - 1).dropna(), 1 - hit.mean()


def _rebalance(c: pd.DataFrame, score_fn, when: pd.Series, k: int, top: bool) -> pd.DataFrame:
    """Equal-weight the k best (top=True) or worst sectors by score_fn at each `when` close; hold until next."""
    sc = score_fn(c)
    pos = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for t in c.index[when.reindex(c.index, fill_value=False).to_numpy()]:
        s = sc.loc[t].dropna()
        if len(s) < k:
            continue
        pick = s.sort_values(ascending=not top).index[:k]
        pos.loc[t] = 0.0
        pos.loc[t, pick] = 1.0 / k
    return pos.ffill().fillna(0.0)


def sector_momentum(px, k=3):
    """Industry momentum (Moskowitz & Grinblatt 1999, J Finance): at each month end hold the k sectors with the
    best 12-month return skipping the latest month (12-1); benchmark = equal-weight all 9 sectors."""
    c = px["close"][SECTORS]
    idx = c.index.to_series()
    month_end = idx.dt.month != idx.shift(-1).dt.month
    pos = _rebalance(c, lambda x: x.shift(21) / x.shift(252) - 1, month_end, k, top=True)
    bh = pd.DataFrame(1.0 / len(SECTORS), index=c.index, columns=SECTORS)
    live = pos.sum(axis=1) > 0
    first = live.idxmax()
    return run_positions(pos.loc[first:], c), run_positions(bh.loc[first:], c), pos.loc[first:].sum(axis=1).mean()


def sector_reversal(px, k=3):
    """Short-term reversal (Lehmann 1990; Jegadeesh 1990 for stocks): each week's last trading day hold the k
    sectors with the WORST prior-week return; benchmark = equal-weight 9 sectors."""
    c = px["close"][SECTORS]
    idx = c.index.to_series()
    week_end = (idx.dt.isocalendar().week != idx.shift(-1).dt.isocalendar().week).fillna(True).astype(bool)
    pos = _rebalance(c, lambda x: x / x.shift(5) - 1, week_end, k, top=False)
    bh = pd.DataFrame(1.0 / len(SECTORS), index=c.index, columns=SECTORS)
    first = (pos.sum(axis=1) > 0).idxmax()
    return run_positions(pos.loc[first:], c), run_positions(bh.loc[first:], c), pos.loc[first:].sum(axis=1).mean()


def vol_target(px, asset, target=0.10, n=20):
    """Volatility-managed exposure (Moreira & Muir 2017, J Finance): weight = min(1, target / trailing n-day
    realised vol). Long-only, no leverage. Benchmark = buy-and-hold the same asset."""
    c = px["close"][asset]
    rv = c.pct_change().rolling(n).std() * np.sqrt(252)
    w = (target / rv).clip(upper=1.0).dropna()
    pos = w.to_frame(asset)
    bh = pd.DataFrame(1.0, index=w.index, columns=[asset])
    return run_positions(pos, px["close"]), run_positions(bh, px["close"]), w.mean()


TRIALS = (
    [("dip_rip", {"asset": a, "band": b}) for a in ["SPY", "QQQ"] for b in [0.03, 0.05, 0.10]]
    + [("tech_gate", {"signal": "XLK"}), ("tech_gate", {"signal": "MEGA_TECH"})]
    + [("graph_leadlag", {"k": 3})]
    + [("loss_cap", {"asset": a, "cap": 0.02}) for a in ["SPY", "QQQ"]]
    # 2026-10-17: published-literature sector/vol ideas (registered before scoring)
    + [("sector_momentum", {"k": 3}), ("sector_reversal", {"k": 3}), ("vol_target", {"asset": "SPY", "target": 0.10})]
)


def run_trial(px, name, p):
    if name == "dip_rip":
        return dip_rip(px, p["asset"], p["band"])
    if name == "tech_gate":
        return tech_gate(px, MEGA_TECH if p["signal"] == "MEGA_TECH" else p["signal"])
    if name == "graph_leadlag":
        return graph_leadlag(px, p["k"])
    if name == "loss_cap":
        return loss_cap(px, p["asset"], p["cap"])
    if name == "sector_momentum":
        return sector_momentum(px, p["k"])
    if name == "sector_reversal":
        return sector_reversal(px, p["k"])
    if name == "vol_target":
        return vol_target(px, p["asset"], p["target"])
    raise KeyError(name)


def cmd_dev():
    px, snap = load(DEV)
    done = {(r["strategy"], json.dumps(r["params"], sort_keys=True))
            for r in map(json.loads, LEDGER.read_text().splitlines())} if LEDGER.exists() else set()
    new = []
    for name, p in TRIALS:
        if (name, json.dumps(p, sort_keys=True)) in done:
            continue
        s, b, expo = run_trial(px, name, p)
        rec = score(name, p, s, b, expo)
        rec.update({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "window": "dev", "snapshot": snap})
        new.append(rec)
    with LEDGER.open("a") as f:
        for r in new:
            f.write(json.dumps(r, default=float) + "\n")
    allr = [json.loads(l) for l in LEDGER.read_text().splitlines()]
    bar = 0.05 / len(allr)
    t = pd.DataFrame(allr)
    t["params"] = t["params"].map(lambda d: ",".join(f"{k}={v}" for k, v in d.items()))
    t["passes"] = (t.p < bar) & (t.alpha_ann > 0)
    cols = ["strategy", "params", "exposure", "alpha_ann", "alpha_t", "p", "cagr", "bh_cagr", "sharpe", "bh_sharpe", "maxdd", "bh_maxdd", "passes"]
    pd.set_option("display.width", 250)
    print(f"DEV {DEV[0]}..{DEV[1]}, {len(allr)} trials in ledger, bar one-sided p < {bar:.4f}, cost {COST_BPS} bps/trade")
    print(t[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    {"dev": cmd_dev}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
