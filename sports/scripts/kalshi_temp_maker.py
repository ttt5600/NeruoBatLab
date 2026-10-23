"""E1/E2 (temperature consensus-reject) executed as a MAKER instead of a taker. EXECUTION test, DEV picks only. Lead L24.

    python scripts/kalshi_temp_maker.py build            # dev pick lists + public trade tapes + T+10h quotes (cached)
    python scripts/kalshi_temp_maker.py quotes           # (re)fetch only the T+10h quotes (after a 429)
    python scripts/kalshi_temp_maker.py dev [--ledger]   # score the variants below; --ledger appends lab/ledger.jsonl

The RULE is unchanged (lab/strategies/kalshi_reject_safe_check.py for E1 = orig 7 cities, wx_expand.picks for X1 = 17
more highs). Only HOW the NO is bought changes. The sealed periods (E1 2025Q3+, X1 2026-03-01+) are NOT loaded here;
replaying the chosen execution on them is a separate one-shot that needs the user's explicit yes.

PRE-REGISTERED 2026-10-22, written before any trade tape was pulled or any maker P&L computed
------------------------------------------------------------------------------------------------
Picks : E1 dev = kalshi_temp DEV quarters (2021Q3..2025Q2; the 5-model rule only fires from 2024Q1) via the lab loader;
        X1 dev = wx_expand rows with kind == high and date < 2026-03-01, wx_expand.picks(). One pick = one bracket.
T     : the decision candle end (price_ts, ~22:00 station-local the day before), the same instant the taker baseline uses.
Quotes at T: yes_bid b, yes_ask a. NO ask = 1 - b (what the taker pays), NO best bid = 1 - a.
Baseline TAKER: buy NO at 1 - b, fee 0.07*p*(1-p) unrounded (direct-member $0.0001 precision, docs fee_rounding).
MAKER price q (post_only NO bid placed at T):
    M1 = NO ask - 1c ; M2 = NO ask - 2c ; M3 = NO mid (1 - (a+b)/2) floored to the cent, capped at NO ask - 1c.
FILL (public tape GET /historical/trades, fallback /markets/trades): consider trades with created_time in (T, close].
    A trade "at or below our bid" = no_price <= q (either taker side; while our bid rests at q, any print at <= q
    implies the book reached us). Queue caveat: filled iff the cumulative count of such trades >= N.
    N = 1 (optimistic: we are first in queue) and N = 10 (size-through: at least 10 contracts traded at/below q).
    Maker fee 0 (series fee_type 'quadratic' = General Trading Fees Table, no maker component -- API docs + live
    /series check 10-22; historical maker-fee schedule for weather before 2025 not verified -> sensitivity row with
    0.0175*q*(1-q), not a trial).
Unfilled -> no bet (variants M*_N*).
FALLBACK variants F*: rest at q only during (T, T + 10 h] (= ~08:00 local on the event day), fill rule N = 10;
    if unfilled, take at the T+10h NO ask (1 - yes_bid of the last hourly candle ending <= T+10h) with taker fee,
    subject to the executor's own guards: ask <= pick NO price + 1c and >= pick NO price - 3c, spread <= 10c and
    yes_bid > 0; otherwise no bet.
Trials (9, pooled E1+X1 picks): M1_N1 M1_N10 M2_N1 M2_N10 M3_N1 M3_N10 F1_N10 F2_N10 F3_N10. Bar p < 0.05/9.
PRIMARY statistic per variant: delta units = sum over ALL picks of (maker pnl - taker pnl), unfilled maker pnl = 0,
    each bet staked 1 unit of cost (pnl = 1/cost - 1 if NO wins else -1). One-sided cluster bootstrap over calendar
    dates (all cities on a date resample together), 4000 draws, p = P(delta <= 0). PASS = p < bar and delta > 0.
Reported (not trials): maker ROI per $ incl. fees (+ its own bootstrap p), fill rate, adverse selection = loss rate
    of filled picks vs all picks vs unfilled picks, units/month, per-cohort split, maker-fee sensitivity.
Dataset tag kalshi_temp_maker.

SEALED REPLAY (user "yes MAKER" 2026-10-23), written BEFORE any sealed tape was pulled or any sealed P&L computed:
    python scripts/kalshi_temp_maker.py build-sealed      # sealed picks + tapes only (no P&L)
    python scripts/kalshi_temp_maker.py sealed --confirm  # ONE shot -> lab/lockbox.jsonl
  Picks: E1 = kalshi_temp LOCKBOX quarters (2025Q3..) through the same frozen rule; X1 = wx_expand high rows with
  date >= 2026-03-01 through wx_expand.picks(). Same taker baseline, same fill rule, same bootstrap (4000, dates).
  PRIMARY (single promotion, bar p < 0.05): M3_N10 = post-only NO bid at the mid (floored, <= ask - 1c), filled iff
  >= 10 contracts print at <= q in (T, close]; delta units vs taker over ALL sealed picks > 0 and one-sided p < 0.05.
  Reported, not promotions: M2_N10, M3_N1, M2_N1, maker ROI, fill rate, adverse selection, by cohort, maker-fee row.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports import wx
from sports.http import fetch, fetch_many
import wx_expand as X

DATA = ROOT / "data"
PICKS = DATA / "kalshi_temp_maker_picks.parquet"
TRADES = DATA / "kalshi_temp_maker_trades.parquet"
QUOTES = DATA / "kalshi_temp_maker_quotes_t10.parquet"
PICKS_S = DATA / "kalshi_temp_maker_picks_sealed.parquet"
TRADES_S = DATA / "kalshi_temp_maker_trades_sealed.parquet"
LOCKBOX = ROOT / "lab" / "lockbox.jsonl"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "kalshi_temp_maker"
TAKER_FEE, MAKER_FEE_SENS = 0.07, 0.0175
K_HOURS = 10
TRIALS = ["M1_N1", "M1_N10", "M2_N1", "M2_N10", "M3_N1", "M3_N10", "F1_N10", "F2_N10", "F3_N10"]


# ------------------------------------------------------------------------------------------------ picks
def e1_dev_picks(split: str = "dev") -> pd.DataFrame:
    from lab import datasets as D
    spec = importlib.util.spec_from_file_location("rule", ROOT / "lab" / "strategies" / "kalshi_reject_safe_check.py")
    rule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rule)
    df = D.load("kalshi_temp", split)
    t = df[rule.bet(None, df.drop(columns=["won", "push", "actual"], errors="ignore")) > 0]
    raw = pd.concat([pd.read_parquet(f, columns=["ticker", "price_ts", "close_time"])
                     for f in sorted(DATA.glob("wx_*.parquet")) if f.stem.replace("wx_", "") in
                     {s.replace("KXHIGH", "") for s in wx.CITIES}]).drop_duplicates("ticker")
    t = t.rename(columns={"opp_id": "ticker"}).merge(raw, on="ticker", how="left")
    return pd.DataFrame({"cohort": "E1", "ticker": t.ticker, "event_ticker": t.event_ticker,
                         "series": t.ticker.str.split("-").str[0], "date": pd.to_datetime(t.date),
                         "yes_bid": t.yes_bid, "yes_ask": t.yes_ask, "won": t.won.astype(int),
                         "T": t.price_ts.astype("int64"), "close_time": t.close_time})


def x1_dev_picks(sealed: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(X.OUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df[((df.date >= X.SPLIT) if sealed else (df.date < X.SPLIT)) & (df.kind == "high")]  # sealed only via build-sealed
    s = X.picks(df)
    return pd.DataFrame({"cohort": "X1", "ticker": s.ticker, "event_ticker": s.event_ticker, "series": s.series,
                         "date": s.date, "yes_bid": s.yes_bid, "yes_ask": s.yes_ask,
                         "won": (s.result == "no").astype(int), "T": s.price_ts.astype("int64"),
                         "close_time": s.close_time})


# ------------------------------------------------------------------------------------------------ tapes
def _tape(tk: str, t0: int, t1: int) -> list[dict]:
    out = []
    for path in ("historical/trades", "markets/trades"):
        cursor = None
        while True:
            p = {"ticker": tk, "min_ts": t0, "max_ts": t1, "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            d = fetch(f"{wx.API}/{path}", p, cache=True)
            out += d.get("trades", [])
            cursor = d.get("cursor")
            if not cursor or not d.get("trades"):
                break
        if out:
            break
    rows = []
    for r in out:
        npx = r.get("no_price_dollars")
        npx = float(npx) if npx is not None else float(r.get("no_price")) / 100
        cnt = r.get("count_fp")
        cnt = float(cnt) if cnt is not None else float(r.get("count", 0))
        rows.append({"ticker": tk, "ts": int(pd.Timestamp(r["created_time"]).timestamp()), "no_price": npx,
                     "count": cnt, "taker_side": r.get("taker_side")})
    return rows


def _patient(urls: list[str]) -> list:
    """fetch_many, but a persistent 429 waits a minute and retries (cache makes re-runs cheap)."""
    import time
    def one(u):
        for i in range(20):
            try:
                return fetch(u)
            except Exception as ex:  # noqa: BLE001 -- 429 after fetch's own backoff
                if "429" not in str(ex):
                    raise
                time.sleep(60)
        raise RuntimeError(f"429 x20: {u}")
    with ThreadPoolExecutor(max_workers=1) as ex:
        return list(ex.map(one, urls))


def _quotes_at(picks: pd.DataFrame) -> pd.DataFrame:
    """yes bid/ask of the last hourly candle ending <= T + K_HOURS, per pick."""
    out = []
    ev = picks.groupby(["series", "event_ticker"])["T"].min().reset_index()
    urls = [f"{wx.API}/series/{s}/events/{e}/candlesticks?start_ts={t}&end_ts={t + K_HOURS * 3600}&period_interval=60"
            for s, e, t in ev.itertuples(index=False)]
    want = set(picks.ticker)
    missing = []
    for (s, e, t), d in zip(ev.itertuples(index=False), _patient(urls)):
        tks = d.get("market_tickers", [])
        if not tks:
            missing.append(e)
        for tk, cs in zip(tks, d.get("market_candlesticks", [])):
            if tk in want:
                r = wx._row(tk, e, cs, t + K_HOURS * 3600)
                if r:
                    out.append(r)
    leg = picks[picks.event_ticker.isin(missing)]
    urls = [f"{wx.API}/historical/markets/{tk}/candlesticks?start_ts={t}&end_ts={t + K_HOURS * 3600}&period_interval=60"
            for tk, t in zip(leg.ticker, leg["T"])]
    for (tk, e, t), d in zip(zip(leg.ticker, leg.event_ticker, leg["T"]), _patient(urls)):
        r = wx._row(tk, e, d.get("candlesticks", []), t + K_HOURS * 3600)
        if r:
            out.append(r)
    q = pd.DataFrame(out).rename(columns={"yes_bid": "yes_bid_k", "yes_ask": "yes_ask_k", "price_ts": "price_ts_k"})
    return q[["ticker", "price_ts_k", "yes_bid_k", "yes_ask_k"]] if len(q) else q


def cmd_build(sealed: bool = False) -> int:
    p = pd.concat([e1_dev_picks("lockbox" if sealed else "dev"), x1_dev_picks(sealed)], ignore_index=True)
    p["close_ts"] = pd.to_datetime(p.close_time, utc=True).map(lambda x: int(x.timestamp()))
    assert p["T"].notna().all() and (p.close_ts > p["T"]).all()
    p.to_parquet(PICKS_S if sealed else PICKS, index=False)
    print(f"picks: {p.cohort.value_counts().to_dict()}  dates {p.date.min().date()}..{p.date.max().date()}", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, r in enumerate(ex.map(lambda a: _tape(*a), zip(p.ticker, p["T"], p.close_ts))):
            rows += r
            if i % 250 == 0:
                print(f"  tapes {i}/{len(p)}", flush=True)
    tr = pd.DataFrame(rows, columns=["ticker", "ts", "no_price", "count", "taker_side"])
    tr.to_parquet(TRADES_S if sealed else TRADES, index=False)
    print(f"trades: {len(tr)} rows on {tr.ticker.nunique()} of {len(p)} pick tickers", flush=True)
    return 0 if sealed else cmd_quotes()


def cmd_quotes() -> int:
    """T+10h quotes for the fallback variants (separate step: resumable from the HTTP cache after a 429)."""
    p = pd.read_parquet(PICKS)
    q = _quotes_at(p)
    q.to_parquet(QUOTES, index=False)
    print(f"T+{K_HOURS}h quotes: {len(q)} of {len(p)}")
    return 0


# ------------------------------------------------------------------------------------------------ scoring
def _cent_floor(x):
    return np.floor(np.round(x * 100, 6)) / 100


def maker_price(p: pd.DataFrame, which: str) -> np.ndarray:
    ask = 1 - p.yes_bid.to_numpy(float)
    if which == "1":
        q = ask - 0.01
    elif which == "2":
        q = ask - 0.02
    else:
        q = np.minimum(_cent_floor(1 - (p.yes_bid.to_numpy(float) + p.yes_ask.to_numpy(float)) / 2), ask - 0.01)
    return np.round(q, 4)


def pnl(won: np.ndarray, cost: np.ndarray) -> np.ndarray:
    return np.where(won == 1, 1 / cost - 1, -1.0)


def taker(p: pd.DataFrame) -> np.ndarray:
    a = 1 - p.yes_bid.to_numpy(float)
    return pnl(p.won.to_numpy(), a + TAKER_FEE * a * (1 - a))


def fill_flags(p: pd.DataFrame, tr: pd.DataFrame, q: np.ndarray, n_min: int, t_end: np.ndarray) -> np.ndarray:
    """True iff >= n_min contracts printed at no_price <= q in (T, t_end]."""
    g = {k: v for k, v in tr.groupby("ticker")}
    out = np.zeros(len(p), bool)
    for i, (tk, t0, t1) in enumerate(zip(p.ticker, p["T"], t_end)):
        d = g.get(tk)
        if d is None:
            continue
        m = (d.ts.to_numpy() > t0) & (d.ts.to_numpy() <= t1) & (d.no_price.to_numpy() <= q[i] + 1e-9)
        out[i] = d["count"].to_numpy()[m].sum() >= n_min
    return out


def variant(p, tr, qk, name, maker_fee=0.0):
    """-> per-pick maker pnl (0 if no bet), bet flag, filled-as-maker flag."""
    which, n = name[1], int(name.split("_N")[1])
    q = maker_price(p, which)
    won = p.won.to_numpy()
    mk_pnl = pnl(won, q + maker_fee * q * (1 - q))
    if name[0] == "M":
        f = fill_flags(p, tr, q, n, p.close_ts.to_numpy())
        return np.where(f, mk_pnl, 0.0), f, f
    f = fill_flags(p, tr, q, n, p["T"].to_numpy() + K_HOURS * 3600)
    k = p[["ticker"]].merge(qk, on="ticker", how="left")
    b, a = k.yes_bid_k.to_numpy(float), k.yes_ask_k.to_numpy(float)
    ask_k = 1 - b
    pick = 1 - p.yes_bid.to_numpy(float)
    ok = (b > 0) & (a < 1) & ((a - b) <= 0.10) & (ask_k <= pick + 0.01 + 1e-9) & (ask_k >= pick - 0.03 - 1e-9)
    ok = np.nan_to_num(ok, nan=0).astype(bool) & ~f
    tk_pnl = pnl(won, ask_k + TAKER_FEE * ask_k * (1 - ask_k))
    return np.where(f, mk_pnl, np.where(ok, tk_pnl, 0.0)), f | ok, f


def boot(dates: np.ndarray, x: np.ndarray, rng, n=4000, denom=None):
    """Cluster bootstrap over dates of sum(x) (or sum(x)/sum(denom))."""
    u, inv = np.unique(dates, return_inverse=True)
    sx = np.bincount(inv, weights=x, minlength=len(u))
    sd = np.bincount(inv, weights=denom, minlength=len(u)) if denom is not None else None
    idx = rng.integers(0, len(u), size=(n, len(u)))
    if sd is None:
        return sx[idx].sum(1)
    d = sd[idx].sum(1)
    return np.where(d > 0, sx[idx].sum(1) / np.maximum(d, 1e-12), 0.0)


def summarize(p, mk, bet, filled, base, rng):
    dates = p.date.dt.strftime("%Y-%m-%d").to_numpy()
    won = p.won.to_numpy()
    months = max(1.0, (p.date.max() - p.date.min()).days / 30.4)
    delta = mk - base
    bd = boot(dates, delta, rng)
    br = boot(dates, mk, rng, denom=bet.astype(float))
    return {"picks": int(len(p)), "bets": int(bet.sum()), "maker_fills": int(filled.sum()),
            "fill_rate": float(filled.mean()), "bet_rate": float(bet.mean()),
            "units": float(mk.sum()), "taker_units": float(base.sum()), "delta_units": float(delta.sum()),
            "delta_ci_lo": float(np.quantile(bd, .025)), "delta_ci_hi": float(np.quantile(bd, .975)),
            "p": float(np.mean(bd <= 0)),
            "roi": float(mk[bet].mean()) if bet.any() else 0.0, "taker_roi": float(base.mean()),
            "roi_ci_lo": float(np.quantile(br, .025)), "roi_ci_hi": float(np.quantile(br, .975)),
            "roi_p": float(np.mean(br <= 0)),
            "loss_rate_all": float(1 - won.mean()),
            "loss_rate_filled": float(1 - won[filled].mean()) if filled.any() else float("nan"),
            "loss_rate_unfilled": float(1 - won[~filled].mean()) if (~filled).any() else float("nan"),
            "units_per_month": float(mk.sum() / months), "taker_units_per_month": float(base.sum() / months),
            "bets_per_month": float(bet.sum() / months)}


def cmd_dev() -> int:
    p = pd.read_parquet(PICKS)
    p["date"] = pd.to_datetime(p["date"])
    assert ((p.cohort == "E1") & (p.date >= "2025-07-01")).sum() == 0 and ((p.cohort == "X1") & (p.date >= X.SPLIT)).sum() == 0
    tr = pd.read_parquet(TRADES)
    qk = pd.read_parquet(QUOTES)
    base = taker(p)
    print(f"DEV picks {len(p)} ({p.cohort.value_counts().to_dict()}), {p.date.min().date()}..{p.date.max().date()}; "
          f"tapes on {tr.ticker.nunique()} tickers ({p.ticker.isin(tr.ticker).mean():.1%}), {len(tr)} trades; "
          f"T+{K_HOURS}h quotes on {p.ticker.isin(qk.ticker).mean():.1%}")
    print(f"TAKER baseline: units {base.sum():+.2f}  ROI {base.mean():+.2%}  loss rate {1 - p.won.mean():.2%}  "
          f"avg NO ask {(1 - p.yes_bid).mean():.3f}  spread>1c {((p.yes_ask - p.yes_bid) > 0.011).mean():.1%}")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    new = {hashlib.sha256(("kalshi_temp_maker:" + n).encode()).hexdigest()[:12] for n in TRIALS}
    bar = 0.05 / len(shas | new)
    recs = []
    hdr = f"{'variant':8s} {'fill':>6s} {'bets':>5s} {'ROI':>7s} {'units':>7s} {'d_units':>8s} {'95% CI':>15s} {'p':>7s}  loss%: all/filled/unfilled  u/mo"
    print(hdr)
    for name in TRIALS:
        mk, bet, filled = variant(p, tr, qk, name)
        r = summarize(p, mk, bet, filled, base, np.random.default_rng(0))
        r["passed"] = bool(r["p"] < bar and r["delta_units"] > 0)
        coh = {}
        for c in ("E1", "X1"):
            m = (p.cohort == c).to_numpy()
            s = summarize(p[m], mk[m], bet[m], filled[m], base[m], np.random.default_rng(1))
            coh[c] = {k: round(s[k], 4) for k in ("picks", "bets", "fill_rate", "roi", "taker_roi", "delta_units", "p",
                                                  "loss_rate_filled", "loss_rate_all", "units_per_month")}
        mf, mb, _ = variant(p, tr, qk, name, maker_fee=MAKER_FEE_SENS)
        r["delta_units_if_maker_fee"] = float((mf - base).sum())
        print(f"{name:8s} {r['fill_rate']:6.1%} {r['bets']:5d} {r['roi']:+7.2%} {r['units']:+7.2f} {r['delta_units']:+8.2f} "
              f"[{r['delta_ci_lo']:+6.1f},{r['delta_ci_hi']:+6.1f}] {r['p']:7.4f}  {r['loss_rate_all']:.2%}/"
              f"{r['loss_rate_filled']:.2%}/{r['loss_rate_unfilled']:.2%}  {r['units_per_month']:+.2f} "
              f"(taker {r['taker_units_per_month']:+.2f})  {'PASS' if r['passed'] else ''}")
        print(f"          E1 {coh['E1']}\n          X1 {coh['X1']}\n          with maker fee 0.0175: d_units "
              f"{r['delta_units_if_maker_fee']:+.2f}")
        sha = hashlib.sha256(("kalshi_temp_maker:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"kalshi_temp_maker_{name}",
                     "sha": sha, "dataset": DATASET, "hypothesis": f"pre-registered in scripts/kalshi_temp_maker.py ({name}); "
                     "E1+X1 dev picks executed as maker vs taker; primary = delta units, cluster bootstrap by date",
                     "source": "lead L24 (social_edges #1)", "status": "ok", "trial_number": len(shas),
                     "study_trials": len(TRIALS), "bonferroni_bar": bar, "trial": name, **r, "by_cohort": coh})
    print(f"{DATASET}: {len(shas)} trials -> bar p < {bar:.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        print("appended to ledger")
    return 0


def cmd_sealed() -> int:
    if "--confirm" not in sys.argv:
        print("sealed needs --confirm AND the user's explicit yes")
        return 1
    if any(json.loads(l).get("strategy") == "kalshi_temp_maker_M3_N10" for l in LOCKBOX.read_text().splitlines() if l.strip()):
        print("MAKER already scored on the lockbox -- never re-score")
        return 1
    p = pd.read_parquet(PICKS_S)
    p["date"] = pd.to_datetime(p["date"])
    assert ((p.cohort == "E1") & (p.date < "2025-07-01")).sum() == 0 and ((p.cohort == "X1") & (p.date < X.SPLIT)).sum() == 0
    tr = pd.read_parquet(TRADES_S)
    base = taker(p)
    sha = hashlib.sha256(PICKS_S.read_bytes() + TRADES_S.read_bytes()).hexdigest()[:12]
    print(f"SEALED picks {len(p)} ({p.cohort.value_counts().to_dict()}), {p.date.min().date()}..{p.date.max().date()}; "
          f"tapes on {p.ticker.isin(tr.ticker).mean():.1%} of tickers; data sha {sha}")
    print(f"TAKER baseline: units {base.sum():+.2f} ROI {base.mean():+.2%} loss rate {1 - p.won.mean():.2%}")
    out = {}
    for name in ("M3_N10", "M2_N10", "M3_N1", "M2_N1"):
        mk, bet, filled = variant(p, tr, None, name)
        r = summarize(p, mk, bet, filled, base, np.random.default_rng(0))
        coh = {}
        for c in ("E1", "X1"):
            m = (p.cohort == c).to_numpy()
            if m.any():
                cr = summarize(p[m], mk[m], bet[m], filled[m], base[m], np.random.default_rng(1))
                coh[c] = {k: round(cr[k], 4) for k in ("picks", "fill_rate", "roi", "taker_roi", "delta_units", "p")}
        mf, _, _ = variant(p, tr, None, name, maker_fee=MAKER_FEE_SENS)
        r["delta_units_if_maker_fee"] = float((mf - base).sum())
        r["by_cohort"] = coh
        out[name] = r
        tag = ("PASS" if r["delta_units"] > 0 and r["p"] < 0.05 else "FAIL") if name == "M3_N10" else "(reported)"
        print(f"{name:7s} fill {r['fill_rate']:.1%} bets {r['bets']} ROI {r['roi']:+.2%} (taker {r['taker_roi']:+.2%}) "
              f"d_units {r['delta_units']:+.2f} [{r['delta_ci_lo']:+.1f},{r['delta_ci_hi']:+.1f}] p {r['p']:.4f} "
              f"loss all/filled {r['loss_rate_all']:.2%}/{r['loss_rate_filled']:.2%} "
              f"maker-fee d {r['delta_units_if_maker_fee']:+.2f} {tag}")
        print(f"        {coh}")
    prim = out["M3_N10"]
    verdict = "PASS" if prim["delta_units"] > 0 and prim["p"] < 0.05 else "FAIL"
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": "kalshi_temp_maker_M3_N10",
           "sha": sha, "dataset": DATASET, "promotion_number": 1, **prim,
           "secondary": {k: {kk: v[kk] for kk in ("delta_units", "p", "fill_rate", "roi")} for k, v in out.items()
                         if k != "M3_N10"}, "verdict": verdict}
    with LOCKBOX.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print("lockbox: appended", verdict)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "quotes": cmd_quotes, "dev": cmd_dev, "build-sealed": lambda: cmd_build(True),
                      "sealed": cmd_sealed}.get(cmd, lambda: print(__doc__) or 1)())
