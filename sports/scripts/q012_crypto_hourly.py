"""Q012: Kalshi hourly crypto above/below (KXBTCD, KXETHD) vs a spot + realised-vol fair value.

Pre-registration: notes/q012_crypto_hourly.md (written before any quote was joined to any outcome).

Stages (run in order):
  draw    -- seeded, outcome-free draw of DEV slots (series x ET date x ET hour)
  spot    -- Coinbase public 1-min candles BTC-USD / ETH-USD -> data/q012/spot_{BTC,ETH}.parquet
  markets -- strike lists for drawn events; outcomes split off into data/q012/dev_results.parquet (score reads it)
  build   -- 1-min Kalshi candles for spot-selected strikes, quotes at T-30/15/10, model probabilities
             -> data/q012/dev_decisions.parquet (NO outcomes)
  counts  -- per-trial bet/event counts from the decision file (extension rule), no outcomes
  score   -- the one scoring run: trials, calibration, ledger rows
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "q012"
D.mkdir(parents=True, exist_ok=True)
LEDGER = ROOT / "lab" / "ledger.jsonl"
API = "https://api.elections.kalshi.com/trade-api/v2"
CB = "https://api.exchange.coinbase.com/products/{}/candles"
SERIES = {"KXBTCD": "BTC", "KXETHD": "ETH"}
FRAME_START, FRAME_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-10-27")
SPLIT = pd.Timestamp("2026-04-11")  # DEV = ET date < SPLIT
ET = "America/New_York"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
MS = (30, 15, 10)
ZSEL = 3.5
TAKER = 0.07
SEED = 12012
N_DRAW = 1600
MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# ---------------------------------------------------------------- polite HTTP (<= ~4.5 req/s per host)
_last: dict[str, float] = {}
_lock = threading.Lock()
_local = threading.local()


def _throttle(host: str, gap: float) -> None:
    with _lock:
        now = time.monotonic()
        slot = max(now, _last.get(host, 0.0) + gap)
        _last[host] = slot
    if slot > now:
        time.sleep(slot - now)


def get(url: str, params: dict | None = None, gap: float = 0.22, retries: int = 16):
    host = url.split("/")[2]
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    for a in range(retries):
        _throttle(host, gap)
        try:
            r = _local.s.get(url, params=params, timeout=60, headers={"User-Agent": "research backtest"})
        except requests.RequestException:
            time.sleep(min(2 ** a, 30))
            continue
        if r.status_code == 404:
            return {}
        if r.status_code in (429, 500, 502, 503, 504):
            print(f"  http {r.status_code} attempt {a} {url.split('/v2')[-1][:60]}", flush=True)
            time.sleep(min(2 ** (a + 1), 60))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"failed {url} {params}")


# ---------------------------------------------------------------- draw
def slot_ticker(series: str, date: pd.Timestamp, hour: int) -> str:
    return f"{series}-{date:%y}{MON[date.month - 1]}{date:%d}{hour:02d}"


def dev_slots() -> list[str]:
    days = pd.date_range(FRAME_START, SPLIT - pd.Timedelta(days=1), freq="D")
    return [slot_ticker(s, d, h) for s in SERIES for d in days for h in range(24)]


def cmd_draw(which: int = 1) -> int:
    slots = dev_slots()
    if which == 1:
        pick = np.random.default_rng(SEED).choice(len(slots), N_DRAW, replace=False)
    else:
        first = set(json.loads((D / "draw1.json").read_text()))
        rest = [s for s in slots if s not in first]
        pick = np.random.default_rng(SEED + 1).choice(len(rest), N_DRAW, replace=False)
        slots = rest
    out = sorted(slots[i] for i in pick)
    (D / f"draw{which}.json").write_text(json.dumps(out))
    print(f"DEV slots {len(dev_slots())}; drew {len(out)} -> draw{which}.json")
    return 0


def drawn() -> list[str]:
    out = []
    for w in (1, 2):
        p = D / f"draw{w}.json"
        if p.exists():
            out += json.loads(p.read_text())
    return out


# ---------------------------------------------------------------- spot
def cmd_spot() -> int:
    start = pd.Timestamp("2024-10-25", tz="UTC")
    end = pd.Timestamp(SPLIT, tz="UTC") + pd.Timedelta(days=1)
    for prod, a in (("BTC-USD", "BTC"), ("ETH-USD", "ETH")):
        path = D / f"spot_{a}.parquet"
        if path.exists():
            print(a, "exists")
            continue
        wins = []
        t = start
        while t < end:
            wins.append(t)
            t += pd.Timedelta(minutes=300)

        def one(t0):
            p = {"granularity": 60, "start": t0.isoformat(), "end": (t0 + pd.Timedelta(minutes=299)).isoformat()}
            return get(CB.format(prod), p, gap=0.2) or []

        with ThreadPoolExecutor(4) as ex:
            rows = [r for chunk in ex.map(one, wins) for r in chunk]
        df = pd.DataFrame(rows, columns=["start", "low", "high", "open", "close", "volume"])
        df = df.drop_duplicates("start").sort_values("start").reset_index(drop=True)
        df.to_parquet(path)
        print(a, len(df), "candles", df.start.min(), df.start.max())
    return 0


def load_spot(a: str) -> pd.DataFrame:
    """Per-minute frame indexed by CLOSE time c (= candle start + 60 s), unix seconds. Forward-filled."""
    df = pd.read_parquet(D / f"spot_{a}.parquet")
    c = df.start.to_numpy(np.int64) + 60
    full = np.arange(c.min(), c.max() + 60, 60, dtype=np.int64)
    s = pd.Series(df.close.to_numpy(float), index=c).reindex(full)
    have = s.notna().to_numpy()
    s = s.ffill()
    r = np.log(s).diff()
    r2 = r ** 2
    sig2 = 0.5 * r2.rolling(60, min_periods=50).mean() + 0.5 * r2.rolling(1440, min_periods=1200).mean()
    # last REAL candle close time at or before c (staleness guard: spot must be a real candle within 5 min)
    last_real = pd.Series(np.where(have, full, np.nan), index=full).ffill().to_numpy()
    return pd.DataFrame({"S": s.to_numpy(), "sig2": sig2.to_numpy(), "last_real": last_real}, index=full)


def spot_at(sp: pd.DataFrame, t: int):
    """S_t, sigma2_t using only candles that CLOSED at or before t."""
    c = (t // 60) * 60
    if c not in sp.index:
        return None
    row = sp.loc[c]
    if not np.isfinite(row.S) or not np.isfinite(row.sig2) or row.sig2 <= 0:
        return None
    if not np.isfinite(row.last_real) or c - row.last_real > 240:
        return None
    assert c <= t
    return float(row.S), float(row.sig2), int(c)


_cdf_cache: dict = {}


def emp_cdf(sp: pd.DataFrame, a: str, m: int, day_utc: pd.Timestamp) -> np.ndarray:
    """Sorted standardised m-min returns, t' on 5-min grid, all t'+m <= 00:00 UTC of the decision day, 60 days back."""
    key = (a, m, day_utc)
    if key in _cdf_cache:
        return _cdf_cache[key]
    end = int(day_utc.timestamp())
    t0 = end - 60 * 86400
    grid = np.arange(t0 - t0 % 300, end - m * 60 + 1, 300, dtype=np.int64)
    grid = grid[np.isin(grid, sp.index) & np.isin(grid + m * 60, sp.index)]
    S0 = sp.S.reindex(grid).to_numpy()
    S1 = sp.S.reindex(grid + m * 60).to_numpy()
    sg = sp.sig2.reindex(grid).to_numpy()
    z = np.log(S1 / S0) / np.sqrt(sg * m)
    z = np.sort(z[np.isfinite(z)])
    _cdf_cache[key] = z
    return z


def p_yes_emp(z_sorted: np.ndarray, zk: float) -> float:
    n = len(z_sorted)
    above = n - np.searchsorted(z_sorted, zk, side="right")
    return (above + 0.5) / (n + 1)


# ---------------------------------------------------------------- markets
def cmd_markets() -> int:
    evs = drawn()
    mk_path, res_path = D / "dev_markets.parquet", D / "dev_results.parquet"
    done = set(pd.read_parquet(mk_path).event_ticker) if mk_path.exists() else set()
    miss_path = D / "missing_events.json"
    missing = set(json.loads(miss_path.read_text())) if miss_path.exists() else set()
    todo = [e for e in evs if e not in done and e not in missing]
    print(f"{len(evs)} drawn, {len(todo)} to fetch")

    def one(e):
        return e, get(f"{API}/historical/markets", {"event_ticker": e, "limit": 1000}).get("markets", [])

    for i in range(0, len(todo), 200):
        _markets_chunk(todo[i:i + 200], one, missing, mk_path, res_path)
        miss_path.write_text(json.dumps(sorted(missing)))
        print(f"  {min(i + 200, len(todo))}/{len(todo)}", flush=True)
    m = pd.read_parquet(mk_path)
    print(f"events with markets {m.event_ticker.nunique()}, markets {len(m)}, missing slots {len(missing)}")
    return 0


def _markets_chunk(todo, one, missing, mk_path, res_path):
    mrows, rrows = [], []
    with ThreadPoolExecutor(4) as ex:
        for e, ms in ex.map(one, todo):
            if not ms:
                missing.add(e)
                continue
            for x in ms:
                mrows.append({"event_ticker": e, "ticker": x["ticker"], "series": e.split("-")[0],
                              "strike": float(x["floor_strike"]) if x.get("floor_strike") is not None else np.nan,
                              "strike_type": x.get("strike_type"), "open_time": x.get("open_time"),
                              "close_time": x.get("close_time"), "status": x.get("status")})
                rrows.append({"ticker": x["ticker"], "result": x.get("result"),
                              "expiration_value": x.get("expiration_value")})
    if mrows:
        m = pd.DataFrame(mrows)
        r = pd.DataFrame(rrows)
        if mk_path.exists():
            m = pd.concat([pd.read_parquet(mk_path), m])
            r = pd.concat([pd.read_parquet(res_path), r])
        m.to_parquet(mk_path)
        r.to_parquet(res_path)  # outcomes: written here, read only by `score`


# ---------------------------------------------------------------- build
def _q(c, side):
    v = (c.get(side) or {}).get("close")
    return float(v) if v is not None else np.nan


def cmd_build() -> int:
    m = pd.read_parquet(D / "dev_markets.parquet")
    m = m[(m.strike_type == "greater") & m.strike.notna()].copy()
    m["close_ts"] = (pd.to_datetime(m.close_time, utc=True) - EPOCH) // pd.Timedelta(seconds=1)
    m["open_ts"] = (pd.to_datetime(m.open_time, utc=True) - EPOCH) // pd.Timedelta(seconds=1)
    m = m[m.close_ts % 3600 == 0]
    spots = {a: load_spot(a) for a in ("BTC", "ETH")}

    # strike selection on spot only
    sel = []
    for (ev, ser), g in m.groupby(["event_ticker", "series"]):
        a = SERIES[ser]
        H = int(g.close_ts.iloc[0])
        keep = np.zeros(len(g), bool)
        for mm in MS:
            s = spot_at(spots[a], H - mm * 60)
            if s is None:
                continue
            S, sig2, _ = s
            z = np.log(g.strike.to_numpy() / S) / np.sqrt(sig2 * (mm - 1 + 1 / 3))
            keep |= np.abs(z) <= ZSEL
        sel.append(g[keep])
    sel = pd.concat(sel)
    print(f"selected {len(sel)} markets of {len(m)} in {sel.event_ticker.nunique()} events")

    raw_path = D / "candles_raw.jsonl"
    have = set()
    if raw_path.exists():
        with raw_path.open() as f:
            for line in f:
                have.add(json.loads(line)["ticker"])
    todo = sel[~sel.ticker.isin(have)]
    print(f"candles to fetch: {len(todo)}")

    def one(row):
        st = max(int(row.open_ts), int(row.close_ts) - 4 * 3600)
        d = get(f"{API}/historical/markets/{row.ticker}/candlesticks",
                {"start_ts": st, "end_ts": int(row.close_ts), "period_interval": 1})
        cs = [[c["end_period_ts"], _q(c, "yes_bid"), _q(c, "yes_ask"), float(c.get("volume") or 0)]
              for c in d.get("candlesticks", [])]
        return {"ticker": row.ticker, "c": cs}

    with raw_path.open("a") as f, ThreadPoolExecutor(5) as ex:
        n = 0
        for out in ex.map(one, list(todo.itertuples())):
            f.write(json.dumps(out) + "\n")
            n += 1
            if n % 2000 == 0:
                f.flush()
                print(f"  {n}/{len(todo)}", flush=True)

    candles = {}
    with raw_path.open() as f:
        for line in f:
            o = json.loads(line)
            candles[o["ticker"]] = o["c"]

    rows = []
    for r in sel.itertuples():
        a = SERIES[r.series]
        H = int(r.close_ts)
        cs = candles.get(r.ticker, [])
        for mm in MS:
            t = H - mm * 60
            s = spot_at(spots[a], t)
            pre = [c for c in cs if c[0] <= t]
            if s is None or not pre:
                continue
            S, sig2, c_spot = s
            last = pre[-1]
            m_eff = mm - 1 + 1 / 3
            zk = np.log(r.strike / S) / np.sqrt(sig2 * m_eff)
            day = pd.Timestamp(t, unit="s", tz="UTC").normalize()
            zs = emp_cdf(spots[a], a, mm, day)
            from math import erf, sqrt
            rows.append({"event_ticker": r.event_ticker, "ticker": r.ticker, "series": r.series, "asset": a,
                         "close_ts": H, "m": mm, "t_dec": t, "spot_close_ts": c_spot, "quote_ts": int(last[0]),
                         "S": S, "sig_min": float(np.sqrt(sig2)), "strike": r.strike, "z": float(zk),
                         "p_emp": float(p_yes_emp(zs, zk)), "p_gauss": float(0.5 * (1 - erf(zk / sqrt(2)))),
                         "n_cdf": len(zs), "yes_bid": last[1], "yes_ask": last[2],
                         "vol_pre": float(sum(c[3] for c in pre))})
    dd = pd.DataFrame(rows)
    assert (dd.spot_close_ts <= dd.t_dec).all() and (dd.quote_ts <= dd.t_dec).all()
    dd.to_parquet(D / "dev_decisions.parquet")
    print(f"decisions {len(dd)} rows; by m: {dd.groupby('m').size().to_dict()}")
    return 0


# ---------------------------------------------------------------- trials
def fee(p):
    return TAKER * p * (1 - p)


def fee_rounded(p):
    return np.ceil(TAKER * p * (1 - p) * 100 - 1e-9) / 100


TRIALS = {"T1_FADE_T15": (15, "fade"), "T2_FADE_T30": (30, "fade"), "T3_FADE_T10": (10, "fade"),
          "T4_ANY_T15": (15, "any")}


def bets(dd: pd.DataFrame, trial: str, pcol: str = "p_emp", feefn=fee) -> pd.DataFrame:
    mm, kind = TRIALS[trial]
    d = dd[dd.m == mm].copy()
    yb, ya = d.yes_bid.to_numpy(), d.yes_ask.to_numpy()
    py = d[pcol].to_numpy()
    spread = ya - yb
    ask_yes, ask_no = ya, 1 - yb
    ok_yes = (ask_yes >= 0.01) & (ask_yes <= 0.99)
    ok_no = (ask_no >= 0.01) & (ask_no <= 0.99)
    e_yes = np.where(ok_yes, py - ask_yes - feefn(ask_yes), -9)
    e_no = np.where(ok_no, (1 - py) - ask_no - feefn(ask_no), -9)
    if kind == "fade":
        # F side = the model favourite when the other side is a longshot (model prob <= 0.10)
        f_is_no = py <= 0.10
        f_is_yes = py >= 0.90
        side = np.where(f_is_no, "no", np.where(f_is_yes, "yes", ""))
        edge = np.where(f_is_no, e_no, np.where(f_is_yes, e_yes, -9))
        take = (side != "") & (edge >= 0.03) & (spread <= 0.02)
    else:
        side = np.where(e_yes >= e_no, "yes", "no")
        edge = np.maximum(e_yes, e_no)
        take = (edge >= 0.05) & (spread <= 0.03)
    d["side"], d["edge"] = side, edge
    d = d[take].copy()
    d["ask"] = np.where(d.side == "yes", d.yes_ask, 1 - d.yes_bid)
    d["fee"] = feefn(d.ask.to_numpy())
    d["p_side"] = np.where(d.side == "yes", d[pcol], 1 - d[pcol])
    return d


def cmd_counts() -> int:
    dd = pd.read_parquet(D / "dev_decisions.parquet")
    print(f"decision rows {len(dd)}, events {dd.event_ticker.nunique()}")
    for t in TRIALS:
        b = bets(dd, t)
        print(f"{t}: bets {len(b)}, events {b.event_ticker.nunique()}, by side {b.side.value_counts().to_dict()}, "
              f"mean ask {b.ask.mean():.3f}, mean edge {b.edge.mean():.3f}")
    return 0


def boot(b: pd.DataFrame, B: int = 5000, seed: int = SEED):
    g = b.groupby("event_ticker").agg(pnl=("pnl", "sum"), cost=("cost", "sum"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    P, C = g.pnl.to_numpy()[idx].sum(1), g.cost.to_numpy()[idx].sum(1)
    roi = P / C
    return float(np.mean(P <= 0)), float(np.quantile(roi, 0.025)), float(np.quantile(roi, 0.975)), float(roi.mean())


def score_bets(b: pd.DataFrame) -> dict:
    b = b.copy()
    b["cost"] = b.ask + b.fee
    b["pnl"] = np.where(b.won, 1 - b.cost, -b.cost)
    if len(b) == 0:
        return {"bets": 0}
    p, lo, hi, bm = boot(b)
    ev = b.groupby("event_ticker").close_ts.first().sort_values()
    h1 = set(ev.index[: len(ev) // 2])
    b1, b2 = b[b.event_ticker.isin(h1)], b[~b.event_ticker.isin(h1)]
    roi = lambda x: float(x.pnl.sum() / x.cost.sum()) if len(x) else float("nan")
    return {"bets": int(len(b)), "events": int(b.event_ticker.nunique()), "units": float(b.pnl.sum()),
            "staked": float(b.cost.sum()), "roi": roi(b), "ci_lo": lo, "ci_hi": hi, "p": p, "boot_mean": bm,
            "half1": roi(b1), "half2": roi(b2), "half1_n": int(len(b1)), "half2_n": int(len(b2)),
            "win_rate": float(b.won.mean()), "mean_p_model": float(b.p_side.mean()), "mean_cost": float(b.cost.mean()),
            "by_asset": {a: {"n": int(len(x)), "roi": roi(x)} for a, x in b.groupby("asset")}}


def cmd_score() -> int:
    dd = pd.read_parquet(D / "dev_decisions.parquet")
    res = pd.read_parquet(D / "dev_results.parquet").drop_duplicates("ticker")
    dd = dd.merge(res, on="ticker", how="left", validate="many_to_one")
    assert dd.result.isin(["yes", "no"]).all(), dd.result.value_counts(dropna=False)
    dd["yes_won"] = dd.result == "yes"
    sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                         cwd=ROOT).stdout.strip()
    k = len(TRIALS)
    bar = 0.05 / k
    out = {}
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = []
    for i, t in enumerate(TRIALS, 1):
        b = bets(dd, t)
        b["won"] = np.where(b.side == "yes", b.yes_won, ~b.yes_won)
        r = score_bets(b)
        sens = {}
        for nm, kw in (("fee_1c_rounded", {"feefn": fee_rounded}), ("gauss_model", {"pcol": "p_gauss"})):
            bs = bets(dd, t, **kw)
            bs["won"] = np.where(bs.side == "yes", bs.yes_won, ~bs.yes_won)
            sr = score_bets(bs)
            sens[nm] = {x: sr.get(x) for x in ("bets", "roi", "ci_lo", "ci_hi", "p")}
        enough = r.get("bets", 0) >= 300 and r.get("events", 0) >= 100
        passed = bool(enough and r["roi"] > 0 and r["p"] < bar and r["half1"] > 0 and r["half2"] > 0)
        verdict = "PASS" if passed else ("UNDERPOWERED" if not enough else "FAIL")
        row = {"ts": ts, "strategy": f"q012_{t}", "sha": sha, "dataset": "q012_crypto_hourly",
               "hypothesis": "pre-registered in notes/q012_crypto_hourly.md; Kalshi KXBTCD/KXETHD ask vs Coinbase spot "
                             "+ realised-vol empirical-CDF fair value, taker at ask + 0.07p(1-p)",
               "source": "QUESTIONS Q012", "status": "ok", "split": "DEV ET date 2025-01-01..2026-04-10 (seeded draw)",
               "trial_number": i, "study_trials": k, "bonferroni_bar": bar, **r, "sens": sens,
               "passed": passed, "verdict": verdict}
        out[t] = row
        lines.append(json.dumps(row, default=float))
        b.to_parquet(D / f"bets_{t}.parquet")
    with LEDGER.open("a") as f:
        for ln in lines:
            f.write(ln + "\n")

    # descriptive calibration at T-15
    c = dd[(dd.m == 15) & (dd.yes_bid >= 0) & (dd.yes_ask <= 1) & (dd.yes_ask - dd.yes_bid <= 0.10)].copy()
    c = c[(c.yes_ask > c.yes_bid)]
    c["mid"] = (c.yes_bid + c.yes_ask) / 2
    edges = [0, .02, .05, .10, .20, .35, .50, .65, .80, .90, .95, .98, 1.0001]
    c["bucket"] = pd.cut(c.mid, edges, right=False)
    cal = c.groupby("bucket", observed=True).agg(n=("mid", "size"), events=("event_ticker", "nunique"),
                                                 mean_mid=("mid", "mean"), yes_rate=("yes_won", "mean"),
                                                 mean_p_emp=("p_emp", "mean"), mean_p_gauss=("p_gauss", "mean"))
    brier = {"mid": float(((c.mid - c.yes_won) ** 2).mean()), "p_emp": float(((c.p_emp - c.yes_won) ** 2).mean()),
             "p_gauss": float(((c.p_gauss - c.yes_won) ** 2).mean()), "n": int(len(c))}
    ev = res.copy()
    summary = {"trials": out, "calibration_T15": cal.reset_index().astype({"bucket": str}).to_dict("records"),
               "brier_T15": brier}
    (ROOT / "results" / "q012_crypto_hourly").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "q012_crypto_hourly" / "dev_score.json").write_text(json.dumps(summary, indent=1, default=float))
    for t, r in out.items():
        print(f"{t}: {r['verdict']} bets {r.get('bets')} ev {r.get('events')} roi {r.get('roi')} "
              f"[{r.get('ci_lo')}, {r.get('ci_hi')}] p {r.get('p')} halves {r.get('half1')}/{r.get('half2')}")
    print(cal.to_string())
    print(brier)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "draw":
        sys.exit(cmd_draw(int(sys.argv[2]) if len(sys.argv) > 2 else 1))
    sys.exit({"spot": cmd_spot, "markets": cmd_markets, "build": cmd_build, "counts": cmd_counts,
              "score": cmd_score}[cmd]())
