"""Q017: Kalshi ERCOT daily peak load (KXTXERCOTPEAKD) vs ERCOT's day-ahead load forecast (via EIA-930 DF).

Pre-registration: notes/q017_ercot_load.md (written before any Kalshi quote was joined to a forecast or an outcome).

Stages (run in order):
  markets -- all KXTXERCOTPEAKD markets; operating day parsed from rules; outcomes split into data/q017/results.parquet
  eia     -- EIA-930 ERCO hourly DF (day-ahead forecast) and D (actual demand), 2023-01-01 .. now (api_key=DEMO_KEY)
  audit   -- vintage check: which ERCOT NP3-560 7-day-forecast vintage does EIA DF match? (lookahead guard)
  candles -- hourly Kalshi candles for every settled market
  build   -- decision rows at 06:00 / 10:00 CT on D with model probabilities (NO outcomes)
  score   -- trials on DEV (or --split sealed), ledger rows
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "q017"
D.mkdir(parents=True, exist_ok=True)
LEDGER = ROOT / "lab" / "ledger.jsonl"
API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXTXERCOTPEAKD"
EIA = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
ERCOT_LIST = "https://www.ercot.com/misapp/GetReports.do?reportTypeId={}&reportTitle=x&showHTMLView=&mimicKey"
ERCOT_DL = "https://www.ercot.com/misdownload/servlets/mirDownload?mimic_duns=000000000&doclookupId={}"
CT = "America/Chicago"
TIMES = (6, 10)  # CT decision hours on operating day D
SEED = 17017
NBOOT = 10_000
TAKER = 0.07
FIRST_KALSHI_DAY = pd.Timestamp("2026-08-28")

_last: dict[str, float] = {}
_lock = threading.Lock()


def get(url: str, params: dict | None = None, gap: float = 0.25, retries: int = 14, raw: bool = False):
    host = url.split("/")[2]
    for a in range(retries):
        with _lock:
            now = time.monotonic()
            slot = max(now, _last.get(host, 0.0) + gap)
            _last[host] = slot
        if slot > now:
            time.sleep(slot - now)
        try:
            r = requests.get(url, params=params, timeout=60, headers={"User-Agent": "Mozilla/5.0 research backtest"})
        except requests.RequestException:
            time.sleep(min(2 ** a, 30))
            continue
        if r.status_code == 404:
            return {} if not raw else None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** (a + 1), 60))
            continue
        r.raise_for_status()
        return r.content if raw else r.json()
    raise RuntimeError(f"failed {url} {params}")


def fee(p):
    return TAKER * p * (1 - p)


# ---------------------------------------------------------------- markets
def opday_from_rules(text: str) -> pd.Timestamp:
    m = re.search(r"on ([A-Z][a-z]+ \d{1,2}, \d{4})", text)
    return pd.Timestamp(pd.to_datetime(m.group(1), format="%B %d, %Y"))


def cmd_markets() -> int:
    ms = []
    for ep in ("markets", "historical/markets"):
        cur = None
        while True:
            p = {"series_ticker": SERIES, "limit": 1000}
            if cur:
                p["cursor"] = cur
            r = get(f"{API}/{ep}", p)
            ms += [dict(m, _src=ep) for m in r.get("markets", [])]
            cur = r.get("cursor")
            if not cur:
                break
    df = pd.DataFrame(ms).drop_duplicates("ticker")
    df["opday"] = [opday_from_rules(t) for t in df.rules_primary]
    df["K"] = df.floor_strike.astype(float)
    keep = ["ticker", "event_ticker", "opday", "K", "strike_type", "open_time", "close_time", "status", "_src",
            "volume_fp"]
    df[keep].to_parquet(D / "markets.parquet")
    res = df[df.result.isin(["yes", "no"])][["ticker", "result", "expiration_value"]].copy()
    res["expiration_value"] = pd.to_numeric(res.expiration_value, errors="coerce")
    res.to_parquet(D / "results.parquet")
    ev = df.groupby("event_ticker").opday.nunique()
    assert (ev == 1).all(), "an event spans two op days"
    tick_day = pd.to_datetime(df.event_ticker.str.split("-").str[1], format="%y%b%d")
    print("ticker date - opday (days):", (tick_day - df.opday).dt.days.value_counts().to_dict())
    print("strike types:", df.strike_type.value_counts().to_dict(), "status:", df.status.value_counts().to_dict())
    print(f"markets {len(df)}, events {df.event_ticker.nunique()}, settled markets {len(res)}, "
          f"settled op days {df[df.ticker.isin(res.ticker)].opday.nunique()}")
    return 0


# ---------------------------------------------------------------- eia
def cmd_eia() -> int:
    for typ in ("DF", "D"):
        rows, off = [], 0
        while True:
            p = {"frequency": "hourly", "data[0]": "value", "facets[respondent][]": "ERCO", "facets[type][]": typ,
                 "start": "2023-01-01T00", "sort[0][column]": "period", "sort[0][direction]": "asc",
                 "offset": off, "length": 5000, "api_key": "DEMO_KEY"}
            d = get(EIA, p, gap=2.0)["response"]
            rows += d["data"]
            off += 5000
            if off >= int(d["total"]):
                break
        s = pd.DataFrame(rows)[["period", "value"]]
        s["value"] = pd.to_numeric(s.value, errors="coerce")
        s.to_parquet(D / f"eia_{typ}.parquet")
        print(typ, len(s), s.period.min(), s.period.max())
    return 0


def cmd_eia930() -> int:
    """Fallback for D: EIA-930 keyless six-month BALANCE files (DEMO_KEY allows 10 requests/day; it ran out after DF).
    Files: data/q017/eia930/EIA930_BALANCE_{2023..2026}_Jul_Dec.csv from eia.gov/electricity/gridmonitor/sixMonthFiles.
    Writes eia_D.parquet in the API's format (UTC hour-ending period) and cross-checks DF against the API pull."""
    parts = []
    for f in sorted((D / "eia930").glob("EIA930_BALANCE_*.csv")):
        x = pd.read_csv(f, usecols=["Balancing Authority", "UTC Time at End of Hour", "Demand Forecast (MW)",
                                    "Demand (MW)"], thousands=",", low_memory=False)
        x = x[x["Balancing Authority"] == "ERCO"]
        t = pd.to_datetime(x["UTC Time at End of Hour"], format="%m/%d/%Y %I:%M:%S %p")
        parts.append(pd.DataFrame({"period": t.dt.strftime("%Y-%m-%dT%H"),
                                   "D": pd.to_numeric(x["Demand (MW)"], errors="coerce"),
                                   "DF": pd.to_numeric(x["Demand Forecast (MW)"], errors="coerce")}))
    s = pd.concat(parts).drop_duplicates("period").sort_values("period")
    s[["period"]].assign(value=s.D).to_parquet(D / "eia_D.parquet")
    api = pd.read_parquet(D / "eia_DF.parquet").set_index("period").value
    j = s.set_index("period").DF.to_frame().join(api.rename("api"), how="inner").dropna()
    print(f"D rows {len(s)} {s.period.min()}..{s.period.max()}; DF file-vs-API overlap {len(j)}, "
          f"exact-equal share {(j.DF == j.api).mean():.4f}, max |diff| {(j.DF - j.api).abs().max()}")
    return 0


def eia_daily(typ: str) -> pd.DataFrame:
    """Daily max and hour count per CT operating day. EIA periods are UTC hour-ending."""
    s = pd.read_parquet(D / f"eia_{typ}.parquet").dropna()
    start = pd.to_datetime(s.period, format="%Y-%m-%dT%H").dt.tz_localize("UTC") - pd.Timedelta(hours=1)
    s["day"] = start.dt.tz_convert(CT).dt.tz_localize(None).dt.normalize()
    g = s.groupby("day").value.agg(["max", "count"])
    return g


# ---------------------------------------------------------------- audit (EIA DF vintage vs ERCOT MIS vintages)
def cmd_audit() -> int:
    html = get(ERCOT_LIST.format(12311), raw=True).decode()
    pairs = re.findall(r"<td class='labelOptional_ind'>([^<]+_csv\.zip)</td>.*?doclookupId=(\d+)", html, flags=re.S)
    rows = []

    def one(pair):
        name, did = pair
        parts = name.split(".")
        stamp = pd.Timestamp(parts[3] + parts[4][:6])  # CT local publication stamp
        z = zipfile.ZipFile(io.BytesIO(get(ERCOT_DL.format(did), raw=True, gap=0.3)))
        df = pd.read_csv(z.open(z.namelist()[0]))
        df["day"] = pd.to_datetime(df.DeliveryDate, format="%m/%d/%Y")
        return [{"vintage": stamp, "day": d, "peak": v} for d, v in df.groupby("day").SystemTotal.max().items()]

    with ThreadPoolExecutor(3) as ex:
        for out in ex.map(one, pairs):
            rows += out
    v = pd.DataFrame(rows)
    v.to_parquet(D / "ercot_vintages.parquet")
    df_ = eia_daily("DF")
    out = []
    for day, g in v.groupby("day"):
        if day not in df_.index or df_.loc[day, "count"] < 23:
            continue
        e = df_.loc[day, "max"]
        g = g.assign(diff=(g.peak - e).abs())
        best = g.loc[g["diff"].idxmin()]
        before = g[g.vintage < day]
        last_before = before.loc[before.vintage.idxmax()] if len(before) else None
        out.append({"day": day.date(), "eia_DF_peak": e, "best_vintage": best.vintage, "best_diff": best["diff"],
                    "best_before_D": bool(best.vintage < day), "n_vintages": len(g),
                    "last_before_D_vintage": None if last_before is None else last_before.vintage,
                    "last_before_pct": None if last_before is None else 100 * (last_before.peak - e) / e,
                    "first_on_D_pct": (100 * (g[g.vintage >= day].sort_values("vintage").peak.iloc[0] - e) / e)
                    if (g.vintage >= day).any() else None})
    a = pd.DataFrame(out)
    a.to_csv(D / "audit.csv", index=False)
    print(a.to_string())
    full = a[a.last_before_D_vintage.notna()]  # days with at least one vintage before D (else unevaluable)
    ok = bool(len(full) > 0 and (full.best_before_D.all() or (full.last_before_pct.abs() <= 1.0).all()))
    print("AUDIT", "PASS" if ok else "FAIL", f"({len(full)} audit days)")
    (D / "audit_verdict.json").write_text(json.dumps({"pass": ok, "days": len(full)}))
    return 0


# ---------------------------------------------------------------- candles
def cmd_candles() -> int:
    mk = pd.read_parquet(D / "markets.parquet")
    res = pd.read_parquet(D / "results.parquet")
    mk = mk[mk.ticker.isin(res.ticker)]
    path = D / "candles.jsonl"
    have = set()
    if path.exists():
        have = {json.loads(l)["ticker"] for l in path.open()}
    todo = mk[~mk.ticker.isin(have)].rename(columns={"_src": "src"})
    print("candles to fetch", len(todo))

    def one(row):
        st = int(pd.Timestamp(row.open_time).timestamp())
        en = int(pd.Timestamp(row.close_time).timestamp())
        base = f"{API}/historical/markets/{row.ticker}" if row.src == "historical/markets" else \
            f"{API}/series/{SERIES}/markets/{row.ticker}"
        d = get(f"{base}/candlesticks", {"start_ts": st - 3600, "end_ts": en, "period_interval": 60})
        cs = []
        for c in d.get("candlesticks", []):
            yb, ya = c.get("yes_bid", {}), c.get("yes_ask", {})
            def q(x):
                v = x.get("close_dollars", x.get("close"))
                if v is None:
                    return None
                v = float(v)
                return v / 100 if "close_dollars" not in x and v > 1 else v
            vol = c.get("volume_fp", c.get("volume"))
            cs.append([c["end_period_ts"], q(yb), q(ya), float(vol or 0)])
        return {"ticker": row.ticker, "c": cs}

    with path.open("a") as f, ThreadPoolExecutor(3) as ex:
        for i, out in enumerate(ex.map(one, list(todo.itertuples()))):
            f.write(json.dumps(out) + "\n")
            if i % 100 == 0:
                print(" ", i, flush=True)
    return 0


# ---------------------------------------------------------------- model
def error_sample() -> np.ndarray:
    df_, a_ = eia_daily("DF"), eia_daily("D")
    j = df_.join(a_, lsuffix="_f", rsuffix="_a", how="inner")
    j = j[(j.count_f >= 23) & (j.count_a >= 23) & (j.index < FIRST_KALSHI_DAY)]
    j = j[j.index.month.isin([8, 9, 10, 11]) & j.index.year.isin([2023, 2024, 2025])]
    return np.sort(np.log(j.max_a / j.max_f).to_numpy())


def p_yes(r: np.ndarray, F: float, K: float, c: float = 1.0) -> float:
    n = len(r)
    above = int((c * F * np.exp(r) > K).sum())
    return (above + 0.5) / (n + 1)


def cmd_build() -> int:
    if not json.loads((D / "audit_verdict.json").read_text())["pass"]:
        print("AUDIT FAILED -> no historical build (pre-registered)")
        return 1
    mk = pd.read_parquet(D / "markets.parquet")
    res = pd.read_parquet(D / "results.parquet")
    mk = mk[mk.ticker.isin(res.ticker)].copy()  # settled only (status is not an outcome)
    r = error_sample()
    print(f"error sample n={len(r)} mean {r.mean():+.4f} sd {r.std():.4f} "
          f"q05 {np.quantile(r, .05):+.4f} q95 {np.quantile(r, .95):+.4f}")
    np.save(D / "error_sample.npy", r)
    F = eia_daily("DF")
    cand = {}
    for l in (D / "candles.jsonl").open():
        o = json.loads(l)
        cand[o["ticker"]] = sorted(o["c"])
    rows = []
    for row in mk.itertuples():
        if row.opday not in F.index or F.loc[row.opday, "count"] < 23:
            continue
        Fd = float(F.loc[row.opday, "max"])
        op = int(pd.Timestamp(row.open_time).timestamp())
        for h in TIMES:
            T = int(pd.Timestamp(row.opday + pd.Timedelta(hours=h)).tz_localize(CT).tz_convert("UTC").timestamp())
            cs = [c for c in cand.get(row.ticker, []) if op <= c[0] <= T]
            yb = ya = None
            if cs:
                yb, ya = cs[-1][1], cs[-1][2]
            vol_before = sum(c[3] for c in cs)
            rows.append({"ticker": row.ticker, "opday": row.opday, "hour": h, "K": row.K, "F": Fd,
                         "yes_bid": yb, "yes_ask": ya, "vol_before": vol_before, "has_quote": bool(cs)})
    dd = pd.DataFrame(rows)
    dd.to_parquet(D / "decisions_raw.parquet")
    print("decision rows", len(dd), "with quote", int(dd.has_quote.sum()), "op days", dd.opday.nunique())
    return 0


def split_days(days) -> tuple[list, list]:
    days = sorted(set(days))
    k = int(np.floor(0.7 * len(days)))
    return days[:k], days[k:]


def decide(dd: pd.DataFrame, r: np.ndarray, c: float) -> pd.DataFrame:
    dd = dd.copy()
    dd["p"] = [p_yes(r, F, K, c) for F, K in zip(dd.F, dd.K)]
    ya = dd.yes_ask.astype(float)
    na = 1 - dd.yes_bid.astype(float)
    dd["ask_yes"] = ya.where((ya >= 0.01) & (ya <= 0.99))
    dd["ask_no"] = na.where((na >= 0.01) & (na <= 0.99))
    dd["edge_yes"] = dd.p - dd.ask_yes - fee(dd.ask_yes)
    dd["edge_no"] = (1 - dd.p) - dd.ask_no - fee(dd.ask_no)
    return dd


def trial_bets(dd: pd.DataFrame, trial: str) -> pd.DataFrame:
    h = 6 if trial.endswith("06") else 10
    x = dd[dd.hour == h].copy()
    out = []
    for row in x.itertuples():
        side = None
        if trial.startswith("A"):
            ey = row.edge_yes if np.isfinite(row.edge_yes) else -9
            en = row.edge_no if np.isfinite(row.edge_no) else -9
            if max(ey, en) >= 0.05:
                side = "yes" if ey >= en else "no"
        else:
            if row.p >= 0.97 and np.isfinite(row.ask_yes) and row.ask_yes <= 0.96:
                side = "yes"
            elif (1 - row.p) >= 0.97 and np.isfinite(row.ask_no) and row.ask_no <= 0.96:
                side = "no"
        if side:
            ask = row.ask_yes if side == "yes" else row.ask_no
            out.append({"ticker": row.ticker, "opday": row.opday, "K": row.K, "F": row.F, "p": row.p,
                        "side": side, "ask": ask, "fee": fee(ask)})
    return pd.DataFrame(out)


def boot(b: pd.DataFrame, rng) -> tuple[float, float, float]:
    g = b.groupby("opday").agg(pnl=("pnl", "sum"), cost=("cost", "sum"))
    P, C = g.pnl.to_numpy(), g.cost.to_numpy()
    idx = rng.integers(0, len(g), (NBOOT, len(g)))
    sp, sc = P[idx].sum(1), C[idx].sum(1)
    roi = sp / sc
    return float(np.quantile(roi, .025)), float(np.quantile(roi, .975)), float((sp <= 0).mean())


def cmd_score(split: str = "dev") -> int:
    dd = pd.read_parquet(D / "decisions_raw.parquet")
    dev, sealed = split_days(dd.opday)
    days = dev if split == "dev" else sealed
    res = pd.read_parquet(D / "results.parquet").set_index("ticker")
    mk = pd.read_parquet(D / "markets.parquet").set_index("ticker")
    r = np.load(D / "error_sample.npy")
    # settlement basis c from DEV days only (actual vs actual)
    A = eia_daily("D")
    ev = res.join(mk[["opday"]]).groupby("opday").expiration_value.first()
    ev_dev = ev[ev.index.isin(dev)]
    ratio = (ev_dev / A.loc[ev_dev.index, "max"]).dropna()
    c_med = float(ratio.median())
    c = c_med if abs(np.log(c_med)) > 0.01 else 1.0
    print(f"DEV days {len(dev)} ({dev[0].date()}..{dev[-1].date()}), SEALED {len(sealed)} "
          f"({sealed[0].date()}..{sealed[-1].date()}); scoring {split}")
    print(f"basis settlement/EIA-D peak on DEV: median {c_med:.4f} min {ratio.min():.4f} max {ratio.max():.4f} "
          f"-> c = {c}")
    x = decide(dd[dd.opday.isin(days)], r, c)
    x = x.join(res[["result"]], on="ticker")
    x["y"] = (x.result == "yes").astype(int)
    # descriptive: forecast error on these days, calibration vs mid
    Fd = x.groupby("opday").F.first()
    print(f"F_D vs settlement: MAPE {100 * (ev.loc[Fd.index] / Fd - 1).abs().mean():.2f}%, "
          f"mean ln(A/F) {np.log(ev.loc[Fd.index] / Fd).mean():+.4f}; pre-Kalshi sample n {len(r)} "
          f"mean {r.mean():+.4f} sd {r.std():.4f}")
    for h in TIMES:
        q = x[(x.hour == h) & x.ask_yes.notna() & x.yes_bid.notna() & ((x.yes_ask - x.yes_bid) <= 0.10)]
        mid = (q.yes_ask + q.yes_bid) / 2
        print(f"  {h:02d}:00 two-sided quotes (spread<=10c) n {len(q)} days {q.opday.nunique()}: "
              f"Brier model {((q.p - q.y) ** 2).mean():.4f} mid {((mid - q.y) ** 2).mean():.4f}; "
              f"quote coverage {x[x.hour == h].has_quote.mean():.2f}")
    rng = np.random.default_rng(SEED)
    sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                         cwd=ROOT).stdout.strip()
    trials = ["A06", "A10", "B06", "B10"]
    allbets = []
    for i, t in enumerate(trials, 1):
        b = trial_bets(x, t)
        rec = {"ts": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "strategy": f"q017_{t}",
               "sha": sha, "dataset": "q017_ercot_load", "hypothesis": "pre-registered in notes/q017_ercot_load.md",
               "source": "coordinator brief 10-29", "status": "ok",
               "split": f"{split.upper()} op days {days[0].date()}..{days[-1].date()}",
               "trial_number": i, "study_trials": 4, "bonferroni_bar": 0.05 / 4, "bets": int(len(b))}
        if len(b) == 0:
            print(f"{t}: 0 bets")
            rec.update(passed=False, note="0 bets")
            _ledger(rec)
            continue
        b = b.join(res[["result"]], on="ticker")
        b["win"] = (b.result == b.side).astype(int)
        b["cost"] = b.ask + b.fee
        b["pnl"] = b.win - b.cost
        b["pnl_r"] = b.win - b.ask - np.ceil(b.fee * 100 - 1e-9) / 100
        b["trial"] = t
        allbets.append(b)
        roi = b.pnl.sum() / b.cost.sum()
        roi_r = b.pnl_r.sum() / (b.ask + np.ceil(b.fee * 100 - 1e-9) / 100).sum()
        lo, hi, p = boot(b, rng)
        dsorted = sorted(b.opday.unique())
        half = sorted(days)[len(days) // 2]
        h1, h2 = b[b.opday < half], b[b.opday >= half]
        r1 = h1.pnl.sum() / h1.cost.sum() if len(h1) else np.nan
        r2 = h2.pnl.sum() / h2.cost.sum() if len(h2) else np.nan
        ndays = len(dsorted)
        powered = len(b) >= 30 and ndays >= 10
        bar = 0.05 / 4
        passed = bool(powered and roi > 0 and p < bar and r1 > 0 and r2 > 0)
        print(f"{t}: bets {len(b)} days {ndays} win {b.win.mean():.3f} avg ask {b.ask.mean():.3f} "
              f"units {b.pnl.sum():+.2f} ROI {100 * roi:+.2f}% [{100 * lo:+.2f}, {100 * hi:+.2f}] p {p:.4f} "
              f"halves {100 * r1:+.2f}%/{100 * r2:+.2f}% rounded-fee ROI {100 * roi_r:+.2f}% "
              f"yes/no {int((b.side == 'yes').sum())}/{int((b.side == 'no').sum())} "
              f"{'PASS' if passed else ('underpowered' if not powered else 'fail')}")
        rec.update(units=float(b.pnl.sum()), roi=float(roi), win_rate=float(b.win.mean()), ci_lo=lo, ci_hi=hi,
                   p=p, clusters=ndays, half1_roi=float(r1), half2_roi=float(r2), roi_fee_rounded=float(roi_r),
                   basis_c=c, passed=passed, note="" if powered else "underpowered (<30 bets or <10 days)")
        _ledger(rec)
    if allbets:
        pd.concat(allbets).to_csv(D / f"bets_{split}.csv", index=False)
    return 0


def _ledger(rec: dict) -> None:
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    fn = {"markets": cmd_markets, "eia": cmd_eia, "eia930": cmd_eia930, "audit": cmd_audit, "candles": cmd_candles, "build": cmd_build}
    if cmd == "score":
        split = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--split" else "dev"
        sys.exit(cmd_score(split))
    sys.exit(fn[cmd]())
