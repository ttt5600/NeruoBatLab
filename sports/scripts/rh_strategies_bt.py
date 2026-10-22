"""Robinhood strategy evidence: leverage, vol-targeting, option-overlay proxies. Generic (no account data).

Pre-registered 2026-10-22 before looking at results:
  DEV  = start..2018-12-31 (design), OOS = 2019-01-01..today (reported, not tuned on).
  Financing: margin rate = 13-week T-bill (^IRX) + SPREAD (default 2.0%/yr; broker rates are an assumption, check the app).
  A) Constant daily-rebalanced leverage L in {1,1.5,2,2.5,3,3.3} on SMH (semis proxy for an AI/semis book) and QQQ.
  B) Lever-once-and-hold (how a margin account actually behaves): every 21 trading days open at leverage L, hold 252 days,
     record whether equity/gross ever falls below maintenance m (0.30 = Reg-T floor; 0.50 = typical house req on
     volatile semis) -> P(margin call within a year).
  C) Kelly leverage on DEV = (mu - r_margin) / sigma^2 (arithmetic daily, annualised).
  D) Vol-target: w_t = min(CAP, TGT / sigma_21d(t-1)); TGT = the asset's DEV-period vol (so average exposure ~1), CAP in
     {1.0, 1.5}; 5 bps per unit turnover; w>1 financed at the margin rate. Compared with buy-and-hold.
  E) Option-overlay proxies (no free option history): CBOE ^PUT (cash-secured S&P put-write) vs SPY; QYLD vs QQQ and
     XYLD vs SPY (ATM covered-call ETFs, total return).
Usage: ../.venv_quant/bin/python scripts/rh_strategies_bt.py
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

SPREAD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
DEV_END, OOS_START = "2018-12-31", "2019-01-01"
TICK = ["SPY", "QQQ", "SMH", "^PUT", "^IRX", "QYLD", "XYLD"]

px = yf.download(TICK, start="1996-01-01", auto_adjust=True, progress=False)["Close"]
rf_d = (px["^IRX"].ffill() / 100 / 252).reindex(px.index).ffill()
mg_d = rf_d + SPREAD / 252
ret = px.drop(columns=["^IRX"]).pct_change()


def stats(r, rfd):
    r = r.dropna()
    if len(r) < 50:
        return dict(n=len(r))
    rfd = rfd.reindex(r.index)
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(252)
    sh = (r - rfd).mean() / r.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    return dict(cagr=round(cagr * 100, 1), vol=round(vol * 100, 1), sharpe=round(sh, 2), maxdd=round(dd * 100, 1),
                wiped=bool((eq <= 0.01).any()))


def split(r):
    return {"DEV": stats(r[:DEV_END], rf_d), "OOS": stats(r[OOS_START:], rf_d)}


print(f"margin spread over T-bill = {SPREAD:.1%}; latest T-bill {px['^IRX'].dropna().iloc[-1]:.2f}%\n")

print("A) constant daily-rebalanced leverage")
for a in ["SMH", "QQQ"]:
    r = ret[a].dropna()
    for L in [1, 1.5, 2, 2.5, 3, 3.3]:
        rl = (L * r - (L - 1) * mg_d.reindex(r.index)).clip(lower=-1)
        # once equity hits zero it stays there
        eq = (1 + rl).cumprod()
        print(f"  {a} L={L}: {split(rl)}")

print("\nB) lever once, hold 252d: P(equity/gross < m at any point)")
for a in ["SMH", "QQQ"]:
    p = px[a].dropna()
    mg = mg_d.reindex(p.index)
    for L in [1.5, 2, 2.5, 3, 3.3]:
        for per, lo, hi in [("DEV", p.index[0], pd.Timestamp(DEV_END)), ("OOS", pd.Timestamp(OOS_START), p.index[-1])]:
            idx = np.where((p.index >= lo) & (p.index <= hi))[0]
            starts = [i for i in idx[::21] if i + 252 < len(p)]
            hit30 = hit50 = 0
            r12 = []
            for i in starts:
                g = L * p.iloc[i:i + 253].values / p.iloc[i]
                debt = (L - 1) * np.cumprod(1 + mg.iloc[i:i + 253].values)
                ratio = (g - debt) / g
                hit30 += (ratio < 0.30).any()
                hit50 += (ratio < 0.50).any()
                r12.append(g[-1] - debt[-1])  # equity multiple after 1y (start equity = 1)
            r12 = np.array(r12)
            print(f"  {a} L={L} {per}: windows={len(starts)} P(call@30%)={hit30/len(starts):.0%} "
                  f"P(call@50%)={hit50/len(starts):.0%} 1y equity x: p5={np.percentile(r12,5):.2f} "
                  f"median={np.median(r12):.2f} p95={np.percentile(r12,95):.2f}")

print("\nC) Kelly leverage (DEV, and OOS for reference)")
for a in ["SMH", "QQQ", "SPY"]:
    for per, sl in [("DEV", slice(None, DEV_END)), ("OOS", slice(OOS_START, None))]:
        r = ret[a][sl].dropna()
        m = mg_d.reindex(r.index)
        k = (r - m).mean() / r.var()
        print(f"  {a} {per}: mu={r.mean()*252:.1%} vol={r.std()*np.sqrt(252):.1%} kelly={k:.2f}x half-kelly={k/2:.2f}x")

print("\nD) vol-target (TGT = DEV vol of the asset)")
for a in ["SMH", "QQQ", "SPY"]:
    r = ret[a].dropna()
    tgt = r[:DEV_END].std() * np.sqrt(252)
    sig = r.rolling(21).std().shift(1) * np.sqrt(252)
    for cap in [1.0, 1.5]:
        w = (tgt / sig).clip(upper=cap).fillna(0)
        cost = 0.0005 * w.diff().abs().fillna(0)
        rv = w * r - (w - 1).clip(lower=0) * mg_d.reindex(r.index) + (1 - w).clip(lower=0) * rf_d.reindex(r.index) - cost
        rv = rv[sig.notna()]
        print(f"  {a} tgt={tgt:.0%} cap={cap}: avg w DEV={w[:DEV_END][sig.notna()].mean():.2f} "
              f"OOS={w[OOS_START:].mean():.2f} {split(rv)}")
    print(f"  {a} buy&hold: {split(r[sig.notna()])}")

print("\nE) option-overlay proxies")
for ov, base in [("^PUT", "SPY"), ("QYLD", "QQQ"), ("XYLD", "SPY")]:
    j = ret[[ov, base]].dropna()
    print(f"  {ov} vs {base} ({j.index[0].date()}..): {ov} {split(j[ov])}")
    print(f"  {'':>5}    {base}: {split(j[base])}")
    print(f"  {'':>5}    full-period beta={np.cov(j[ov], j[base])[0,1]/j[base].var():.2f}")
