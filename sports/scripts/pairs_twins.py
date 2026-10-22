"""Twin-security pairs + GGR cointegration screen (user 2026-10-21: "maybe we can see if there are other twin companies").

    python scripts/pairs_twins.py data              # download/cache + coverage report, NO P&L computed
    python scripts/pairs_twins.py dev [--ledger]    # score every pre-registered trial on DEV only

PRE-REGISTERED 2026-10-21, written before any price for these tickers was downloaded or any P&L was computed.
Prior is weak: classic pairs (pairs_bt.py, KO/PEP + 6) made +1.89%/yr dev, ~0 after 2009 (Do & Faff 2010).

WINDOWS
  DEV    = scored days 2006-01-03 .. 2024-10-18 (prices from 2004-06 for warm-up/formation).
  SEALED = 2024-10-21 .. 2026-10-20 (most recent ~2 years). `dev` cuts every price row after DEV_END right after
           load, before any signal is built. Sealed is scored once, only on the user's explicit "yes".
  Note: stocks_bt.py keeps 2019-2025 sealed for ITS 14 ETF rules; these are different strategies/tickers, the lead asked
  for a ~2-year holdout here, so 2019-2024 is dev for this file (it is no longer a clean holdout for twin pairs).

UNIVERSES (fixed lists; a pair is scored only if both legs have >= 500 overlapping DEV days; dropped pairs are listed)
  U1 dual-class shares (same company, different vote/share class):
     GOOGL/GOOG BRK-B/BRK-A FOXA/FOX NWSA/NWS Z/ZG LBRDK/LBRDA LEN/LEN-B BF-B/BF-A HEI/HEI-A MOG-A/MOG-B UAA/UA
     LBTYK/LBTYA FWONK/FWONA BATRK/BATRA GEF/GEF-B CRD-A/CRD-B BIO/BIO-B MKC/MKC-V TAP/TAP-A AGM/AGM-A RUSHA/RUSHB
     GTN/GTN-A HVT/HVT-A CWEN/CWEN-A KELYA/KELYB BELFB/BELFA
  U2 ETF twins (same index or same physical metal):
     SPY/IVV SPY/VOO QQQ/QQQM GLD/IAU GLD/SGOL SLV/SIVR IWM/VTWO MDY/IJH
  U3 Canadian interlisted (US line vs TSX line converted at yfinance CAD=X; same trading hours; NOT tradable on a
     US-stocks-only app, included because the lead asked for dual-listed where free data allows):
     TD/TD.TO RY/RY.TO BNS/BNS.TO ENB/ENB.TO CNQ/CNQ.TO SU/SU.TO CNI/CNR.TO CP/CP.TO
     FX caveat: CAD=X daily close is not stamped at 16:00 ET -> timing noise that mean-reverts; the 1-day execution
     lag below removes most of the mechanical part, not necessarily all.
  DATA-CHECK AMENDMENTS (after `data`, before any P&L was computed, 2026-10-22 03:30 UTC):
     - CNI.TO was a ticker typo for CN Rail on TSX -> CNR.TO (same company).
     - LBRDK and CWEN-A return no data from yfinance -> LBRDK/LBRDA and CWEN/CWEN-A unscored (not trials).
     - yfinance back-fills GOOG before 2014-04-03 as an exact copy of GOOGL (log ratio constant to 1e-7) -> guard:
       no entry signal when the 60-day sd of the spread < 1 bp (no real second line exists; would only burn costs).
  Closed-end fund vs NAV: not attempted -- no free, reliable daily NAV history in yfinance (stated, not tested).
  U4 GGR screen: current S&P 500 constituents (Wikipedia, fetched at download time) grouped by GICS Sub-Industry.
     SURVIVORSHIP BIAS: only today's members, so every name survived to 2026 -> optimistic; delisted twins
     (DISCA/DISCK, VIAC/VIACA, CMCSA/CMCSK, LSXMA/LSXMK, RDS-A/RDS-B ...) are missing from U1 for the same reason.

TWIN RULE (U1-U3), per pair A/B, adjusted closes (yfinance auto_adjust)
  spread s_t = log A_t - log B_t  (hedge ratio fixed at 1: same underlying claim; no fitted beta)
  mu, sd     = mean, sd of s over the 60 trading days t-60..t-1 (strictly past)
  z_t        = (s_t - mu) / sd
  entry      : flat and |z_t| >= 2 -> long the cheap leg, short the rich leg, $1 per leg (fixed shares thereafter)
  exit       : |z_t| <= 0.5, or 20 trading days held (time stop), or |z_t| >= 5 (break stop)
  execution  : signal at close t, trade at close t+1 (1-day lag: kills bid-ask bounce / stale-print / async artefacts);
               an entry is skipped if either leg printed zero volume on t or t+1; an exit with a zero-volume leg
               waits for the next day.
  variants   : BASE as above;  GATED = BASE but enter only if |s_t - mu| >= 2 x round-trip cost of the pair
               (expected reversion must cover costs twice) -- the obvious fix if the edge is sub-cost.

GGR RULE (U4), Gatev-Goetzmann-Rouwenhorst (2006) with a cointegration filter
  every month: formation = previous 252 trading days; candidates = pairs within the same GICS Sub-Industry,
  both legs with no missing close in formation. Normalized price = adj close / formation-start close.
  rank by SSD of normalized prices; take the first 20 (lowest SSD) that also pass Engle-Granger on log prices
  (statsmodels coint, maxlag=1, p < 0.05). Trading = next 126 days: spread = nA - nB, sigma = formation sd;
  open when |spread| > 2 sigma, close when spread crosses zero or at period end. Same 1-day execution lag.
  Cohort return = sum of pair P&L / 20 (committed capital, $1/leg); portfolio = mean of the up-to-6 live cohorts.

COSTS (per leg per side) -- tiered by trailing 60-day median dollar volume (close x volume, shifted 1 day):
  >=$1B 0.5 bp | >=$100M 1.5 bp | >=$20M 4 bp | >=$5M 10 bp | >=$1M 25 bp | <$1M 60 bp   (half-spread + impact)
  borrow on the short leg: 0.30%/yr if $ADV >= $20M else 2.0%/yr (thin share classes can be hard to borrow).
  Reported: GROSS, NET (spread + borrow), NET-NO-BORROW (= the long-only "class swap" an existing holder can do:
  hold B, switch to A when A is cheap -> excess return equals the pair P&L with no short), and NET at 2x costs.

TRIALS (all appended to lab/ledger.jsonl, one line each; Bonferroni bar = 0.05 / trials in THIS study)
  BUG FIX (after the first dev run, 2026-10-22): the GGR leg selected zero pairs in every month because the cache index
  includes TSX/FX-only days (NaN for every US stock) -> U4 restricted to SPY trading days; rerun with `dev --only-ggr`.
  The first U4 ledger line (all zeros) is voided by an appended record; rules unchanged.
  universe level: U1_base U1_gated U2_base U2_gated U3_base U3_gated U4_ggr        (7)
  pair level    : every scored pair under BASE                                        (+ n pairs)
  metric        : daily NET P&L per $1/leg (equal-weight over live pairs); Newey-West(10) HAC t; one-sided p.
  Also: annual return, Sharpe, round trips/yr per pair, avg days held, % days in position, sub-periods
  2006-12 / 2013-18 / 2019-24 (description only, not trials).
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "stocks" / "twins_prices.parquet"
SPX_CACHE = ROOT / "data" / "stocks" / "twins_spx_members.csv"
LEDGER = ROOT / "lab" / "ledger.jsonl"
START, DEV_START, DEV_END, SEALED_END = "2004-06-01", "2006-01-03", "2024-10-18", "2026-10-20"

U1 = [("GOOGL", "GOOG"), ("BRK-B", "BRK-A"), ("FOXA", "FOX"), ("NWSA", "NWS"), ("Z", "ZG"), ("LBRDK", "LBRDA"),
      ("LEN", "LEN-B"), ("BF-B", "BF-A"), ("HEI", "HEI-A"), ("MOG-A", "MOG-B"), ("UAA", "UA"), ("LBTYK", "LBTYA"),
      ("FWONK", "FWONA"), ("BATRK", "BATRA"), ("GEF", "GEF-B"), ("CRD-A", "CRD-B"), ("BIO", "BIO-B"),
      ("MKC", "MKC-V"), ("TAP", "TAP-A"), ("AGM", "AGM-A"), ("RUSHA", "RUSHB"), ("GTN", "GTN-A"), ("HVT", "HVT-A"),
      ("CWEN", "CWEN-A"), ("KELYA", "KELYB"), ("BELFB", "BELFA")]
U2 = [("SPY", "IVV"), ("SPY", "VOO"), ("QQQ", "QQQM"), ("GLD", "IAU"), ("GLD", "SGOL"), ("SLV", "SIVR"),
      ("IWM", "VTWO"), ("MDY", "IJH")]
U3 = [("TD", "TD.TO"), ("RY", "RY.TO"), ("BNS", "BNS.TO"), ("ENB", "ENB.TO"), ("CNQ", "CNQ.TO"), ("SU", "SU.TO"),
      ("CNI", "CNR.TO"), ("CP", "CP.TO")]
UNIVERSES = {"U1": U1, "U2": U2, "U3": U3}
TIERS = [(1e9, 0.5), (1e8, 1.5), (2e7, 4.0), (5e6, 10.0), (1e6, 25.0), (0.0, 60.0)]


# ------------------------------------------------------------------------------------------------ data
def spx_members() -> pd.DataFrame:
    if SPX_CACHE.exists():
        return pd.read_csv(SPX_CACHE)
    import requests
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers={"User-Agent": "Mozilla/5.0 research"}, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    t = t.rename(columns={"Symbol": "ticker", "GICS Sub-Industry": "subind", "GICS Sector": "sector"})
    t["ticker"] = t.ticker.str.replace(".", "-", regex=False)
    t = t[["ticker", "sector", "subind"]]
    t.to_csv(SPX_CACHE, index=False)
    return t


def download() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import yfinance as yf
    tk = sorted({t for u in UNIVERSES.values() for p in u for t in p} | set(spx_members().ticker) | {"CAD=X"})
    frames = []
    for i in range(0, len(tk), 100):
        d = yf.download(tk[i:i + 100], start=START, end="2026-10-21", auto_adjust=True, progress=False,
                        group_by="column", threads=True)
        frames.append(d[["Close", "Volume"]])
    px = pd.concat(frames, axis=1)
    px.columns = [f"{f}|{t}" for f, t in px.columns]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    px.to_parquet(CACHE)
    return px


def load_dev():
    px = download()
    px = px[px.index <= DEV_END]                     # SEALED rows dropped before anything else is computed
    close = px[[c for c in px.columns if c.startswith("Close|")]]
    vol = px[[c for c in px.columns if c.startswith("Volume|")]]
    close.columns = [c.split("|", 1)[1] for c in close.columns]
    vol.columns = [c.split("|", 1)[1] for c in vol.columns]
    # TSX lines -> USD at CAD=X (USD/CAD: CAD per USD)
    fx = close["CAD=X"].ffill()
    for c in [c for c in close.columns if c.endswith(".TO")]:
        close[c] = close[c] / fx
    return close, vol


def half_spread_bps(close: pd.Series, vol: pd.Series) -> pd.Series:
    adv = (close * vol).rolling(60, min_periods=20).median().shift(1)
    out = pd.Series(60.0, index=close.index)
    for lo, bps in reversed(TIERS):
        out[adv >= lo] = bps
    return out, adv


# ------------------------------------------------------------------------------------------------ engine
def simulate(dates, la, lb, ra, rb, vola, volb, hsa, hsb, bra, brb, open_sig, exit_fn):
    """Generic fixed-share pair engine with 1-day execution lag.
    open_sig[i] in {-1,0,+1}: +1 = long A/short B. exit_fn(i, d, days) -> bool. Returns DataFrame of daily components."""
    n = len(dates)
    gross, cost, borrow, inpos, opened = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    pos, days, pend = 0, 0, None
    vA = vB = 0.0          # signed dollar value of each leg (+ long, - short)
    trips = 0
    for i in range(1, n):
        if pos != 0:
            pa, pb = vA * ra[i], vB * rb[i]
            gross[i] = pa + pb
            vA, vB = vA * (1 + ra[i]), vB * (1 + rb[i])
            short_val, short_br = (-vA, bra[i]) if vA < 0 else (-vB, brb[i])
            borrow[i] = short_val * short_br / 252
            inpos[i] = 1
        if pend is not None:                      # execute at close i
            if pend == "close":
                if vola[i] > 0 and volb[i] > 0:
                    cost[i] += abs(vA) * hsa[i] + abs(vB) * hsb[i]
                    pos, vA, vB, pend = 0, 0.0, 0.0, None
            else:
                if vola[i] > 0 and volb[i] > 0:
                    pos, days, trips = pend, 0, trips + 1
                    vA, vB = float(pos), float(-pos)
                    cost[i] += hsa[i] + hsb[i]
                    opened[i] = 1
                pend = None
        if pend is None:                          # decide at close i
            if pos == 0:
                d = open_sig[i]
                if d != 0 and vola[i] > 0 and volb[i] > 0:
                    pend = int(d)
            else:
                days += 1
                if exit_fn(i, pos, days):
                    pend = "close"
    df = pd.DataFrame({"gross": gross, "cost": cost, "borrow": borrow, "inpos": inpos, "opened": opened}, index=dates)
    df.attrs["trips"] = trips
    return df


def twin_pair(close, vol, a, b, gated=False):
    df = pd.DataFrame({"a": close[a], "b": close[b], "va": vol[a], "vb": vol[b]}).dropna()
    df = df[(df.a > 0) & (df.b > 0)]
    if len(df[df.index >= DEV_START]) < 500:
        return None
    hsa, adva = half_spread_bps(df.a, df.va)
    hsb, advb = half_spread_bps(df.b, df.vb)
    s = np.log(df.a) - np.log(df.b)
    mu = s.rolling(60).mean().shift(1)
    sd = s.rolling(60).std().shift(1)
    z = ((s - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    dev = (s - mu).abs().fillna(0).to_numpy()
    rt_cost = (2 * (hsa + hsb) / 1e4).to_numpy()
    sig = np.where(np.abs(z) >= 2, -np.sign(z), 0)        # z>0: A rich -> short A (d=-1)
    sig = np.where(sd.fillna(0).to_numpy() >= 1e-4, sig, 0)  # data guard: identical back-filled series
    if gated:
        sig = np.where(dev >= 2 * rt_cost, sig, 0)
    bra = np.where(adva >= 2e7, 0.003, 0.02)
    brb = np.where(advb >= 2e7, 0.003, 0.02)
    out = simulate(df.index, None, None, df.a.pct_change().fillna(0).to_numpy(), df.b.pct_change().fillna(0).to_numpy(),
                   df.va.to_numpy(), df.vb.to_numpy(), (hsa / 1e4).to_numpy(), (hsb / 1e4).to_numpy(), bra, brb,
                   sig, lambda i, d, days: abs(z[i]) <= 0.5 or days >= 20 or abs(z[i]) >= 5)
    out = out[out.index >= DEV_START]
    return out


# ------------------------------------------------------------------------------------------------ GGR
def ggr(close, vol, members):
    from statsmodels.tsa.stattools import coint
    us_days = close["SPY"].notna()                  # cache index also holds TSX/FX-only days -> US sessions only
    close, vol = close[us_days], vol[us_days]
    tick = [t for t in members.ticker if t in close.columns]
    C = close[tick]
    V = vol[tick]
    R = C.pct_change()
    HS, BR = {}, {}
    for t in tick:
        h, adv = half_spread_bps(C[t], V[t])
        HS[t] = h / 1e4
        BR[t] = pd.Series(np.where(adv >= 2e7, 0.003, 0.02), index=C.index)
    HS, BR = pd.DataFrame(HS), pd.DataFrame(BR)
    groups = members[members.ticker.isin(tick)].groupby("subind").ticker.apply(list)
    idx = C.index
    month_starts = [i for i in range(1, len(idx)) if idx[i].month != idx[i - 1].month and i >= 252]
    cohorts = []
    npicked = []
    for m in month_starts:
        if idx[m] < pd.Timestamp(DEV_START) - pd.Timedelta(days=200):
            continue
        f0, f1 = m - 252, m                       # formation rows f0..f1-1
        t1 = min(m + 126, len(idx))
        if idx[m] > pd.Timestamp(DEV_END):
            break
        F = C.iloc[f0:f1]
        cand = []
        for g in groups:
            ok = [t for t in g if F[t].notna().all()]   # formation data only (no look-ahead)
            if len(ok) < 2:
                continue
            N = (F[ok] / F[ok].iloc[0]).to_numpy()
            for i in range(len(ok)):
                for j in range(i + 1, len(ok)):
                    cand.append((float(((N[:, i] - N[:, j]) ** 2).sum()), ok[i], ok[j]))
        cand.sort()
        picked = []
        for ssd, a, b in cand:
            p = coint(np.log(F[a].to_numpy()), np.log(F[b].to_numpy()), maxlag=1, autolag=None)[1]
            if p < 0.05:
                picked.append((a, b))
            if len(picked) == 20:
                break
        npicked.append(len(picked))
        rows = idx[m - 1:t1]                      # include last formation day as the first decision day
        coh = pd.DataFrame(0.0, index=idx[m:t1], columns=["gross", "cost", "borrow", "inpos", "opened"])
        trips = 0
        for a, b in picked:
            na = C[a].iloc[m - 1:t1] / F[a].iloc[0]
            nb = C[b].iloc[m - 1:t1] / F[b].iloc[0]
            sig_sd = float((F[a] / F[a].iloc[0] - F[b] / F[b].iloc[0]).std())
            spr = (na - nb).to_numpy()
            n = len(rows)
            sig = np.where(np.abs(spr) > 2 * sig_sd, -np.sign(spr), 0)
            sig[-1] = 0
            last = n - 1

            def ex(i, d, days, spr=spr, last=last):
                # d=+1 means long A: entered when spr < 0; close when spr crosses back to >= 0 (and vice versa);
                # period end: decide at last-1 so the close executes on the final trading day
                return (d > 0 and spr[i] >= 0) or (d < 0 and spr[i] <= 0) or i >= last - 1
            sig[last - 1:] = 0
            out = simulate(rows, None, None, R[a].iloc[m - 1:t1].fillna(0).to_numpy(), R[b].iloc[m - 1:t1].fillna(0).to_numpy(),
                           V[a].iloc[m - 1:t1].fillna(0).to_numpy(), V[b].iloc[m - 1:t1].fillna(0).to_numpy(),
                           HS[a].iloc[m - 1:t1].to_numpy(), HS[b].iloc[m - 1:t1].to_numpy(),
                           BR[a].iloc[m - 1:t1].to_numpy(), BR[b].iloc[m - 1:t1].to_numpy(), sig, ex)
            coh = coh.add(out.iloc[1:], fill_value=0)
            trips += out.opened.sum()
        cohorts.append((coh / 20.0, trips, len(picked)))
    return cohorts, npicked


# ------------------------------------------------------------------------------------------------ stats
def stats(x: pd.Series) -> dict:
    x = x.dropna()
    v = x.to_numpy()
    n = len(v)
    if n < 50 or v.std() == 0:
        return {"ann": float(v.mean() * 252) if n else 0.0, "sharpe": 0.0, "t": 0.0, "p": 1.0, "days": n}
    d = v - v.mean()
    s = np.dot(d, d) / n + 2 * sum((1 - k / 11) * np.dot(d[k:], d[:-k]) / n for k in range(1, 11))
    t = v.mean() / np.sqrt(s / n)
    return {"ann": float(v.mean() * 252), "sharpe": float(v.mean() / v.std(ddof=1) * np.sqrt(252)), "t": float(t),
            "p": float(norm.sf(t)), "days": n}


def legs(df):
    return {"gross": df.gross, "net": df.gross - df.cost - df.borrow, "net_noborrow": df.gross - df.cost,
            "net_2x": df.gross - 2 * df.cost - 2 * df.borrow}


def subperiods(x):
    return {k: round(float(x[(x.index >= a) & (x.index <= b)].mean() * 252), 4)
            for k, a, b in [("2006-12", "2006-01-01", "2012-12-31"), ("2013-18", "2013-01-01", "2018-12-31"),
                            ("2019-24", "2019-01-01", "2024-12-31")]}


# ------------------------------------------------------------------------------------------------ commands
def cmd_data() -> int:
    close, vol = load_dev()
    print(f"cached {CACHE.name}: {close.shape[1]} tickers, rows {close.index.min().date()}..{close.index.max().date()} (dev cut)")
    for u, pairs in UNIVERSES.items():
        for a, b in pairs:
            if a not in close or b not in close:
                print(f"  {u} {a}/{b}: MISSING ticker"); continue
            ok = close[[a, b]].dropna()
            ok = ok[ok.index >= DEV_START]
            zv = int(((vol[a].reindex(ok.index) == 0) | (vol[b].reindex(ok.index) == 0)).sum())
            print(f"  {u} {a}/{b}: {len(ok)} dev days from {ok.index.min().date() if len(ok) else '-'}, zero-vol days {zv}")
    m = spx_members()
    have = [t for t in m.ticker if t in close and close[t].notna().sum() > 0]
    print(f"S&P 500 members: {len(m)}, with any price {len(have)}, sub-industries {m.subind.nunique()}")
    return 0


def cmd_dev() -> int:
    close, vol = load_dev()
    results, port_series = [], {}
    for u, pairs in ({} if "--only-ggr" in sys.argv else UNIVERSES).items():
        for gated in (False, True):
            per, meta = {}, {}
            for a, b in pairs:
                if a not in close or b not in close:
                    continue
                out = twin_pair(close, vol, a, b, gated)
                if out is None:
                    continue
                per[f"{a}/{b}"] = out
                yrs = len(out) / 252
                meta[f"{a}/{b}"] = {"trips_yr": out.opened.sum() / yrs if yrs else 0,
                                    "in_pos": float(out.inpos.mean()), "days": len(out)}
            tag = f"{u}_{'gated' if gated else 'base'}"
            comp = {k: pd.DataFrame({p: legs(df)[k] for p, df in per.items()}).mean(axis=1) for k in
                    ["gross", "net", "net_noborrow", "net_2x"]}
            st = {k: stats(v) for k, v in comp.items()}
            tr = float(np.mean([m["trips_yr"] for m in meta.values()])) if meta else 0.0
            ip = float(np.mean([m["in_pos"] for m in meta.values()])) if meta else 0.0
            cost_ann = float(pd.DataFrame({p: df.cost for p, df in per.items()}).mean(axis=1).mean() * 252)
            results.append({"trial": tag, "level": "universe", "n_pairs": len(per), **{f"{k}_{m}": st[k][m] for k in st for m in ("ann", "sharpe", "p")},
                            "t": st["net"]["t"], "p": st["net"]["p"], "trips_yr_per_pair": tr, "in_pos": ip,
                            "cost_ann": cost_ann, "sub": subperiods(comp["net"]), "sub_gross": subperiods(comp["gross"])})
            port_series[tag] = comp["net"]
            if not gated:
                for p, df in per.items():
                    lg = legs(df)
                    sn, sg = stats(lg["net"]), stats(lg["gross"])
                    m = meta[p]
                    held = df.inpos.sum() / max(df.opened.sum(), 1)
                    results.append({"trial": f"{u}_pair_{p}", "level": "pair", "n_pairs": 1, "gross_ann": sg["ann"],
                                    "gross_sharpe": sg["sharpe"], "gross_p": sg["p"], "net_ann": sn["ann"],
                                    "net_sharpe": sn["sharpe"], "net_p": sn["p"],
                                    "net_noborrow_ann": stats(lg["net_noborrow"])["ann"], "net_2x_ann": stats(lg["net_2x"])["ann"],
                                    "t": sn["t"], "p": sn["p"], "trips_yr_per_pair": m["trips_yr"], "in_pos": m["in_pos"],
                                    "avg_hold_days": float(held), "cost_ann": float(df.cost.mean() * 252),
                                    "first": str(df.index.min().date()), "sub": subperiods(lg["net"])})
    # GGR screen
    cohorts, npicked = ggr(close, vol, spx_members())
    acc = {k: [] for k in ["gross", "net", "net_noborrow", "net_2x"]}
    for coh, _, _ in cohorts:
        lg = legs(coh)
        for k in acc:
            acc[k].append(lg[k])
    comp = {k: pd.concat(v, axis=1).mean(axis=1).sort_index() for k, v in acc.items()}
    comp = {k: v[(v.index >= DEV_START) & (v.index <= DEV_END)] for k, v in comp.items()}
    st = {k: stats(v) for k, v in comp.items()}
    trips = np.mean([c[1] / max(c[2], 1) for c in cohorts]) * 2      # per pair per 126d -> per year
    inpos = float(np.mean([c[0].inpos.mean() * 20 / max(c[2], 1) for c in cohorts]))
    results.append({"trial": "U4_ggr", "level": "universe", "n_pairs": int(np.mean(npicked)),
                    **{f"{k}_{m}": st[k][m] for k in st for m in ("ann", "sharpe", "p")},
                    "t": st["net"]["t"], "p": st["net"]["p"], "trips_yr_per_pair": float(trips), "in_pos": inpos,
                    "cost_ann": float(pd.concat([c[0].cost for c in cohorts], axis=1).mean(axis=1).mean() * 252),
                    "sub": subperiods(comp["net"]), "sub_gross": subperiods(comp["gross"]),
                    "cohorts": len(cohorts), "months_lt20": int(sum(1 for k in npicked if k < 20))})

    n = 47 if "--only-ggr" in sys.argv else len(results)   # study size fixed by the full dev run
    bar = 0.05 / n
    print(f"\n{n} trials in this study -> Bonferroni bar p < {bar:.5f}   (DEV {DEV_START}..{DEV_END}; per $1/leg)\n")
    print(f"{'trial':34s} {'gross%':>7s} {'net%':>7s} {'noBorr%':>8s} {'2xcost%':>8s} {'netSR':>6s} {'p':>7s} {'trips/yr':>8s} {'inpos':>6s} {'cost%':>6s}")
    for r in results:
        print(f"{r['trial']:34s} {r['gross_ann']*100:+7.2f} {r['net_ann']*100:+7.2f} {r['net_noborrow_ann']*100:+8.2f} "
              f"{r['net_2x_ann']*100:+8.2f} {r['net_sharpe']:+6.2f} {r['p']:7.4f} {r['trips_yr_per_pair']:8.1f} "
              f"{r['in_pos']:6.2f} {r['cost_ann']*100:6.2f}{'  PASS' if r['p'] < bar else ''}")
    print("\nsub-periods (net ann, universe level):")
    for r in results:
        if r["level"] == "universe":
            print(f"  {r['trial']:12s} net {r['sub']}  gross {r['sub_gross']}")
    print("\npair first dev date / avg hold days:")
    for r in results:
        if r["level"] == "pair":
            print(f"  {r['trial']:34s} {r['first']}  hold {r['avg_hold_days']:.1f}d  net sub {r['sub']}")
    print(f"\nGGR: {results[-1]['cohorts']} cohorts, months with <20 cointegrated pairs: {results[-1]['months_lt20']}")
    if "--ledger" in sys.argv:
        prior = [l for l in LEDGER.read_text().splitlines() if l.strip()] if LEDGER.exists() else []
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with LEDGER.open("a") as f:
            for k, r in enumerate(results):
                f.write(json.dumps({"ts": ts, "strategy": f"twins_{r['trial']}",
                                    "sha": hashlib.sha256(f"pairs_twins:{r['trial']}".encode()).hexdigest()[:12],
                                    "dataset": "stocks_twins_yf", "hypothesis": "pre-registered in scripts/pairs_twins.py",
                                    "source": "pairs_twins", "status": "ok", "trial_number": len(prior) + k + 1,
                                    "study_trials": n, "bonferroni_bar": bar, **r}) + "\n")
        print(f"appended {n} trials to {LEDGER}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"data": cmd_data, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
