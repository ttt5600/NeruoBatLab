"""Q003: resting MAKER NO bids in thin retail Kalshi categories (Mentions). DEV replay on the public trade tape.

    ../.venv_quant/bin/python scripts/q003_retail_maker.py census        # outcome-free: split date, sample, counts
    ../.venv_quant/bin/python scripts/q003_retail_maker.py tapes         # pull trade tapes for the DEV sample (cached)
    ../.venv_quant/bin/python scripts/q003_retail_maker.py dev [--ledger]

PRE-REGISTERED 2026-10-28, written before any trade tape was pulled and before any outcome was joined to a quote
---------------------------------------------------------------------------------------------------------------
Question  : taker NO on overpriced mention YES died sealed (M3, notes/kalshi_mentions.md); maker favourite bids died
            to adverse selection in daily non-sports markets (K-FAV, notes/kalshi_fav_maker.md). Does a RESTING NO bid
            in Mentions (thin, retail-heavy, YES overpriced 4-10pp in every 5-80c bucket) earn money net of
            adverse selection? Literature: makers earn ~1.9c/contract in YES-overbought single-name markets.
Category  : Kalshi category Mentions, the eligible binary word-mention markets of scripts/kalshi_mentions.py
            (65,574 markets / 3,806 events; all groups: earnings, fed, trump, sports announcers, other = TV/podcast/
            culture single-name). Entertainment/culture series outside Mentions are NOT in this test: there are no
            decision-time quotes for them at scale (the 10-27 smallmarket scan has ~280 rows) -> out of budget.
Quotes    : already frozen, outcome-free: data/kalshi_mentions_prices.parquet + _sealed.parquet (yes_bid/yes_ask of
            the last hourly candle ending <= D). D24 = anchor - 24h, D6 = anchor - 6h; anchor = min(first close in
            event, end of busiest volume hour) -- the M3 timing, unchanged. Same filters as M3: drop an event at a
            decision time if >= 50% of its quoted markets sit at an extreme; bid > 0, ask < 1, spread <= 10c.
Band      : YES mid in [0.05, 0.35] at the decision time (the overbought-YES longshot band).
Split     : by event anchor date over all eligible in-band market-decisions (dev + sealed files pooled):
            DEV = anchor earlier than the 70th percentile of anchor; Q003-SEALED = the rest. Sealed outcomes are never
            loaded by this script (its rows are dropped before any `result` column is touched). CAVEAT stated up
            front: the M3 sealed scoring (taker NO <= 15c, anchor >= 2025-10-01) already read outcomes in part of
            this DEV window, for a different execution; Q003 DEV is therefore not virgin data for "mention NO".
Sample    : credit budget -> at most 1,500 DEV tickers, uniform random (seed 3003) among DEV tickers in-band at D24
            or D6. Drawn before any tape is pulled; outcome-free.
Quote rule: post-only NO bid at T = D, size S = 10 contracts. NO ask = 1 - yes_bid, NO best bid = 1 - yes_ask.
            MID : q = NO mid (1 - (b+a)/2) floored to the cent, capped at NO ask - 1c.
            IN1 : q = NO best bid + 1c, capped at NO ask - 1c (joins the best bid when the spread is 1c).
Cancel    : the order is cancelled at anchor - 2h (we do not rest through the live event, where a said word
            picks the bid off at certainty). Window = (T, min(anchor - 2h, market close)].
Fill model: public tape (GET /historical/trades, fallback /markets/trades), one print = (no_price, count).
            Back-of-queue conservative:
              q > NO best bid at T (we are alone at a new level) -> filled iff contracts printed at no_price <= q
                in the window >= S;
              q == NO best bid (we join an existing queue of unknown size) -> filled iff contracts printed STRICTLY
                below q >= S (trade-through: the whole queue at q must have been consumed).
            Unfilled -> no position. Filled -> hold to settlement.
Fees      : maker fee per series. All 464 Mentions series are fee_type 'quadratic' (no maker component) -> 0.
            Sensitivity (not a trial): 0.0175*q*(1-q) per contract.
P&L       : per contract: (1 - q - fee) if result == no else -(q + fee); stake = q + fee. ROI = sum pnl / sum stake.
Trials (4): MID_d24, MID_d6, IN1_d24, IN1_d6. Dataset q003_retail_maker. Bar p < 0.05/4 = 0.0125.
Statistic : ROI over filled contracts; one-sided event-clustered bootstrap (5,000 draws over events), p = P(ROI <= 0).
PASS      : ROI > 0 AND p < 0.0125 AND both chronological DEV halves (by anchor, split at the DEV median date)
            have ROI > 0. Kill: filled loss rate above the break-even loss rate (mean q) -> adverse selection wins.
Reported, not trials: fill rate; P&L per filled contract (cents); adverse-selection decomposition = YES rate on
            filled vs unfilled vs all quoted, and "no-selection" P&L if every quoted market had filled at q minus
            the actual per-fill P&L; taker-NO baseline on the same quoted markets (1 - yes_bid + taker fee);
            sensitivities: S = 1 touch fill, rest-to-close window (T, close], maker fee 0.0175; by group; by half.
fund_stats verdict on every trial after logging.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports.http import fetch, fetch_many  # noqa: F401

API = "https://api.elections.kalshi.com/trade-api/v2"
DATA = ROOT / "data"
PX_DEV, PX_SEAL = DATA / "kalshi_mentions_prices.parquet", DATA / "kalshi_mentions_prices_sealed.parquet"
SAMPLE = DATA / "q003_sample.parquet"
TAPES = DATA / "q003_tapes.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "q003_retail_maker"
TRIALS = ["MID_d24", "MID_d6", "IN1_d24", "IN1_d6"]
BAR = 0.05 / len(TRIALS)
S = 10
SEED = 3003
N_SAMPLE = 1500
BAND = (0.05, 0.35)
TAKER_FEE, MAKER_FEE_SENS = 0.07, 0.0175


def quotes(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """M3's quote filters at one decision time (outcome-free)."""
    d = df.rename(columns={f"bid_{tag}": "bid", f"ask_{tag}": "ask"})
    d = d[d.bid.notna() & d.ask.notna()].copy()
    ext = (d.ask <= 0.03) | (d.bid >= 0.97)
    over = ext.groupby(d.event_ticker).transform("mean") >= 0.5
    d = d[~over]
    d = d[(d.bid > 0) & (d.ask < 1) & (d.ask - d.bid <= 0.10)].copy()
    d["mid"] = (d.bid + d.ask) / 2
    d = d[(d.mid >= BAND[0] - 1e-9) & (d.mid <= BAND[1] + 1e-9)].copy()
    d["tag"] = tag
    d["T"] = (d.anchor_ts - {"d24": 24, "d6": 6}[tag] * 3600).astype(int)
    return d


def load_inband(drop_result: bool = True) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(PX_DEV), pd.read_parquet(PX_SEAL)], ignore_index=True)
    if drop_result:
        df = df.drop(columns=["result"])
    return pd.concat([quotes(df, "d24"), quotes(df, "d6")], ignore_index=True)


def split_ts(q: pd.DataFrame) -> float:
    return float(np.quantile(q.anchor_ts.astype(float), 0.70))


def cmd_census() -> int:
    q = load_inband(drop_result=True)
    cut = split_ts(q)
    dev = q[q.anchor_ts < cut]
    print(f"in-band market-decisions: {len(q)} ({q.ticker.nunique()} tickers, {q.event_ticker.nunique()} events)")
    print(f"split anchor = {pd.Timestamp(cut, unit='s', tz='UTC')}; DEV {len(dev)} decisions / "
          f"{dev.ticker.nunique()} tickers / {dev.event_ticker.nunique()} events; by tag {dev.tag.value_counts().to_dict()}")
    tk = np.sort(dev.ticker.unique())
    rng = np.random.default_rng(SEED)
    pick = rng.choice(tk, size=min(N_SAMPLE, len(tk)), replace=False)
    s = dev[dev.ticker.isin(pick)]
    print(f"sample: {len(pick)} tickers, {len(s)} decisions, {s.event_ticker.nunique()} events, "
          f"anchor {pd.Timestamp(s.anchor_ts.min(), unit='s').date()}..{pd.Timestamp(s.anchor_ts.max(), unit='s').date()}")
    s.to_parquet(SAMPLE, index=False)
    print(f"-> {SAMPLE}  split_ts={cut}")
    return 0


class Uncached(Exception):
    pass


def _fetch(url, params):
    """fetch; with env Q003_CACHE_ONLY=1 a page not already on disk raises Uncached (budget cap, no new calls)."""
    import os
    from sports import http as H
    if os.environ.get("Q003_CACHE_ONLY") and not (H.CACHE_DIR / (H._key(url, params) + ".json")).exists():
        raise Uncached(url)
    return fetch(url, params, cache=True)


def _tape(tk: str, t0: int, t1: int) -> list[dict] | None:
    out = []
    for path in ("historical/trades", "markets/trades"):
        cursor = None
        while True:
            p = {"ticker": tk, "min_ts": t0, "max_ts": t1, "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            try:
                d = _fetch(f"{API}/{path}", p)
            except Uncached:
                if path == "historical/trades" and cursor is None:
                    d = {}  # first historical page never cached = it 404ed (errors are not cached) -> try live
                else:
                    return None
            except Exception as ex:  # noqa: BLE001 -- 404 on one endpoint -> try the other
                d = {}
                if "404" not in str(ex) and "400" not in str(ex):
                    print(f"  {tk} {path}: {str(ex)[:80]}", flush=True)
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


def cmd_tapes() -> int:
    import time
    from concurrent.futures import ThreadPoolExecutor
    s = pd.read_parquet(SAMPLE)
    s["close_ts"] = ((pd.to_datetime(s.close_time, utc=True, format="ISO8601") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype(int)
    w = s.groupby("ticker").agg(t0=("T", "min"), t1=("close_ts", "max")).reset_index()
    t_start, rows, done, done_tk, skipped = time.time(), [], 0, [], []
    deadline = t_start + 45 * 60
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = []
        for r in w.itertuples():
            futs.append((r.ticker, ex.submit(_tape, r.ticker, int(r.t0), int(r.t1))))
        for tk, f in futs:
            if time.time() > deadline:
                print("45-min budget hit; stopping", flush=True)
                for _, g in futs:
                    g.cancel()
                break
            x = f.result()
            if x is None:
                skipped.append(tk)
                continue
            rows += x
            done_tk.append(tk)
            done += 1
            if done % 100 == 0:
                print(f"{done}/{len(w)} tickers, {len(rows)} trades, {time.time() - t_start:.0f}s", flush=True)
    tr = pd.DataFrame(rows, columns=["ticker", "ts", "no_price", "count", "taker_side"])
    tr.to_parquet(TAPES, index=False)
    pd.DataFrame({"ticker": done_tk}).to_parquet(DATA / "q003_tapes_done.parquet", index=False)
    print(f"skipped (not downloaded before the budget cap): {len(skipped)} tickers")
    print(f"-> {TAPES}: {len(tr)} trades on {tr.ticker.nunique()} of {done} pulled tickers")
    return 0


# ------------------------------------------------------------------------------------------------ scoring
def _cent_floor(x):
    return np.floor(np.round(x * 100, 6)) / 100


def maker_q(d: pd.DataFrame, rule: str) -> np.ndarray:
    no_ask, no_bid = 1 - d.bid.to_numpy(float), 1 - d.ask.to_numpy(float)
    if rule == "MID":
        q = _cent_floor(1 - (d.bid.to_numpy(float) + d.ask.to_numpy(float)) / 2)
    else:
        q = no_bid + 0.01
    return np.round(np.minimum(q, no_ask - 0.01), 4)


def fills(d: pd.DataFrame, tr: dict, q: np.ndarray, t_end: np.ndarray, s_min: float) -> np.ndarray:
    no_bid = np.round(1 - d.ask.to_numpy(float), 4)
    out = np.zeros(len(d), bool)
    for i, (tk, t0) in enumerate(zip(d.ticker, d["T"])):
        x = tr.get(tk)
        if x is None:
            continue
        ts, px, c = x
        m = (ts > t0) & (ts <= t_end[i])
        if q[i] > no_bid[i] + 1e-9:
            m &= px <= q[i] + 1e-9
        else:
            m &= px < q[i] - 1e-9
        out[i] = c[m].sum() >= s_min
    return out


def boot_roi(ev: np.ndarray, pnl: np.ndarray, stake: np.ndarray, rng, n=5000):
    u, inv = np.unique(ev, return_inverse=True)
    sp = np.bincount(inv, weights=pnl, minlength=len(u))
    ss = np.bincount(inv, weights=stake, minlength=len(u))
    idx = rng.integers(0, len(u), size=(n, len(u)))
    den = ss[idx].sum(1)
    return np.where(den > 0, sp[idx].sum(1) / np.maximum(den, 1e-12), 0.0)


def score(d, filled, q, fee_c, rng, boot=True):
    yes = (d.result == "yes").to_numpy()
    stake = q + fee_c
    pnl = np.where(yes, -stake, 1 - stake)
    f = filled
    r = {"quoted": int(len(d)), "fills": int(f.sum()), "fill_rate": float(f.mean()) if len(d) else 0.0,
         "events_filled": int(d.event_ticker[f].nunique())}
    if f.sum() == 0:
        return r | {"roi": 0.0, "p": 1.0}
    roi = float(pnl[f].sum() / stake[f].sum())
    r |= {"roi": roi, "units": float(pnl[f].sum()), "pnl_per_fill_c": float(100 * pnl[f].mean()),
          "mean_q_filled": float(q[f].mean()),
          "yes_rate_filled": float(yes[f].mean()), "yes_rate_unfilled": float(yes[~f].mean()) if (~f).any() else None,
          "yes_rate_all": float(yes.mean()), "breakeven_yes_rate": float(1 - stake[f].mean()),
          "pnl_per_contract_if_all_filled_c": float(100 * pnl.mean()),
          "roi_if_all_filled": float(pnl.sum() / stake.sum())}
    r["adverse_selection_c"] = r["pnl_per_contract_if_all_filled_c"] - r["pnl_per_fill_c"]
    if boot:
        b = boot_roi(d.event_ticker.to_numpy()[f], pnl[f], stake[f], rng)
        r |= {"roi_ci_lo": float(np.quantile(b, .025)), "roi_ci_hi": float(np.quantile(b, .975)),
              "p": float(np.mean(b <= 0))}
    return r


def cmd_dev(ledger: bool) -> int:
    rng = np.random.default_rng(SEED)
    cut = split_ts(load_inband(drop_result=True))
    s = pd.read_parquet(SAMPLE)
    assert (s.anchor_ts < cut).all(), "sample contains sealed rows"
    done = set(pd.read_parquet(DATA / "q003_tapes_done.parquet").ticker)
    s = s[s.ticker.isin(done)].copy()
    # join outcomes for DEV tickers only
    res = pd.concat([pd.read_parquet(PX_DEV, columns=["ticker", "result", "anchor_ts"]),
                     pd.read_parquet(PX_SEAL, columns=["ticker", "result", "anchor_ts"])])
    res = res[res.anchor_ts < cut].drop_duplicates("ticker")[["ticker", "result"]]
    s = s.merge(res, on="ticker", how="left")
    s = s[s.result.isin(["yes", "no"])].copy()
    s["close_ts"] = ((pd.to_datetime(s.close_time, utc=True, format="ISO8601") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype(int)
    trd = pd.read_parquet(TAPES)
    tr = {k: (g.ts.to_numpy(), g.no_price.to_numpy(), g["count"].to_numpy()) for k, g in trd.groupby("ticker")}
    s["date"] = pd.to_datetime(s.anchor_ts, unit="s")
    mid_date = s.anchor_ts.median()
    print(f"DEV sample with tapes+outcome: {len(s)} decisions, {s.ticker.nunique()} tickers, {s.event_ticker.nunique()} events;"
          f" tickers with >=1 trade in tape {s.ticker.isin(tr).mean():.1%}")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    n_prior = sum(1 for r in prior if r.get("dataset") == DATASET and r.get("status") == "ok")
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    out = []
    for k, name in enumerate(TRIALS):
        rule, tag = name.split("_")
        d = s[s.tag == tag].reset_index(drop=True)
        q = maker_q(d, rule)
        t_cancel = np.minimum(d.anchor_ts.to_numpy() - 2 * 3600, d.close_ts.to_numpy())
        f = fills(d, tr, q, t_cancel, S)
        r = score(d, f, q, 0.0, rng)
        h1 = d.anchor_ts.to_numpy() < mid_date
        r["half1"] = score(d[h1], f[h1], q[h1], 0.0, rng, boot=False).get("roi", 0.0)
        r["half2"] = score(d[~h1].reset_index(drop=True), f[~h1], q[~h1], 0.0, rng, boot=False).get("roi", 0.0)
        r["half1_fills"], r["half2_fills"] = int(f[h1].sum()), int(f[~h1].sum())
        # taker baseline on the same quoted markets
        a = 1 - d.bid.to_numpy(float)
        tk = score(d, np.ones(len(d), bool), a, TAKER_FEE * a * (1 - a), rng, boot=False)
        r["taker_roi_same_markets"] = tk["roi"]
        # sensitivities (not trials)
        sens = {}
        sens["S1_touch"] = score(d, fills(d, tr, q, t_cancel, 1), q, 0.0, rng, boot=False)
        f_close = fills(d, tr, q, d.close_ts.to_numpy(), S)
        sens["rest_to_close"] = score(d, f_close, q, 0.0, rng, boot=False)
        sens["maker_fee_0175"] = score(d, f, q, MAKER_FEE_SENS * q * (1 - q), rng, boot=False)
        r["sens"] = {k2: {x: v[x] for x in ("fills", "fill_rate", "roi", "yes_rate_filled") if x in v}
                     for k2, v in sens.items()}
        r["by_group"] = {}
        for g in sorted(d.group.unique()):
            m = (d.group == g).to_numpy()
            x = score(d[m].reset_index(drop=True), f[m], q[m], 0.0, rng, boot=False)
            r["by_group"][g] = {k2: x.get(k2) for k2 in ("fills", "roi", "yes_rate_filled", "yes_rate_unfilled")}
        r["passed"] = bool(r["roi"] > 0 and r["p"] < BAR and r["half1"] > 0 and r["half2"] > 0)
        out.append((name, r))
        print(f"\n== {name}: quoted {r['quoted']} fills {r['fills']} ({r['fill_rate']:.1%}) ROI {r['roi']:+.2%} "
              f"[{r.get('roi_ci_lo', 0):+.2%},{r.get('roi_ci_hi', 0):+.2%}] p {r['p']:.4f} halves {r['half1']:+.2%}/{r['half2']:+.2%}"
              f" pass={r['passed']}")
        print(f"   pnl/fill {r.get('pnl_per_fill_c', 0):+.2f}c  YES rate filled {r.get('yes_rate_filled')} unfilled "
              f"{r.get('yes_rate_unfilled')} all {r.get('yes_rate_all')} breakeven {r.get('breakeven_yes_rate')}")
        print(f"   no-selection pnl/contract {r.get('pnl_per_contract_if_all_filled_c', 0):+.2f}c -> adverse selection "
              f"{r.get('adverse_selection_c', 0):.2f}c; taker same mkts {r['taker_roi_same_markets']:+.2%}")
        print(f"   sens {r['sens']}")
        print(f"   by group {r['by_group']}")
        if ledger:
            row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"q003_{name}",
                   "sha": sha, "dataset": DATASET,
                   "hypothesis": f"pre-registered in scripts/q003_retail_maker.py ({name}); resting NO bid in Kalshi "
                                 "Mentions, YES mid 5-35c, back-of-queue tape fill, cancel anchor-2h, hold to settle",
                   "source": "fund Q003", "status": "ok", "trial_number": n_prior + k + 1,
                   "study_trials": len(TRIALS), "bonferroni_bar": BAR, "trial": name, "split_ts": cut,
                   "sample_tickers": int(s.ticker.nunique())} | r
            with LEDGER.open("a") as fh:
                fh.write(json.dumps(row, default=float) + "\n")
    if ledger:
        print(f"\nappended {len(out)} rows to {LEDGER}")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else "census"
    sys.exit({"census": cmd_census, "tapes": cmd_tapes}.get(cmd, lambda: cmd_dev("--ledger" in a))())
