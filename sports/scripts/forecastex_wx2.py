"""ForecastEx daily weather BEYOND the E5 daily highs: does the consensus-reject rule work on other daily families?

    python scripts/forecastex_wx2.py build           # contracts + D-1 (or D-2) trade prices + outcomes + model forecasts
    python scripts/forecastex_wx2.py dev [--ledger]  # power, then T1/T2 per family on DEV rows only

WHY: E5 (scripts/forecastex_wx.py, UH* daily highs) passed sealed (PASS-UNDERPOWERED). ForecastEx favourites trade far
cheaper than Kalshi's (~0.88-0.90 vs ~0.97), which is the only reason to re-test families that were DEAD on Kalshi.

FAMILIES (from the /api/products catalog + the Q007 dev-window daily files, notes/q007_forecastex_breadth.md):
  F1 "UL"  US daily LOW temperature, 24 airports (UL<CODE>, same airports as UH; station K<CODE>). Kalshi analogue = X2
           lows, which FAILED dev (-6.4%) -> this is the same mechanism re-tested at ForecastEx prices.
  F2 "SH"  International daily HIGH, the 8 cities in the live catalog only (identified stations, hard-coded below):
           DDF Frankfurt EDDF, FPO Paris-Orly LFPO, JTT Tokyo-Haneda RJTT, SSS Singapore WSSS, SSY Sydney YSSY,
           YHC Vancouver Harbour CYHC, YUL Montreal CYUL, YYZ Toronto CYYZ. Strikes are CELSIUS (gate: median strike
           < 40). Delisted SH codes with unidentified stations (CAG, EMD, MAA, MDB, NOR) are excluded a priori.
  F3 "DH"  LEGACY US daily HIGH (DH<CODE>, Oct 2025 - Feb 2026, delisted). Same mechanism as E5 but a disjoint earlier
           period. Codes with an unambiguous K<CODE> airport only; NYC/TOR/YUL/YVR excluded a priori (station unknown).
  Not testable / excluded a priori: daily PRECIPITATION does not exist on ForecastEx (rain is monthly TR*, temperature
  monthly OA*, hourly HRU*); legacy DL* lows have ~1 trading day in the files (< 10 event days -> dropped, not a trial).
  No new US stations exist beyond the 24 (UH/UL share them); the only "beyond 24" stations are F2's.

CONTRACT DIRECTION: E5's UH wording is "EXCEED X" (YES iff value > X). For UL/SH/DH no filing text is on disk, so the
  direction is taken from the settlement ladder ALONE (no prices, no forecasts): if >= 95% of station-days have YES
  non-increasing in X the family is "exceed"; if >= 95% non-decreasing, "below" (YES iff value < X, rule mirrored);
  otherwise the family is aborted. Non-monotone station-days are dropped (counted).

RULE (E1/E5 frozen, nothing fitted): models {gfs_seamless, ecmwf_ifs025, icon_seamless, gem_seamless, ncep_nbm_conus}
  (NBM is CONUS-only, so F2 normally has 4), Open-Meteo previous-runs lead-1-day hourly 2 m temperature at the station,
  >= 3 models. Daily value per model:
    highs (F2, F3): max over local hours <= 18 (>= 15 values)  -- identical to E1/E5 (scripts/wx_expand.model_ext)
    lows  (F1):     min over local hours 0-23 (>= 20 values)   -- the contract is the calendar-day WU min; X2 used
                    hours <= 18, which misses late-evening minima. Chosen a priori, not tuned.
  hi/lo = max/min across models. For an "exceed X" contract (integer strike, 1-degree step):
    hi + M < X + 1  -> buy NO ;  lo - M > X -> buy YES.
  ("below X" contracts: YES iff value < X: lo - M > X - 1 -> buy NO ; hi + M < X -> buy YES.)
  Margin M = 3 F (F1, F3) and 3 F = 5/3 C for Celsius F2 (forecasts converted to C). No other margin is tried.

PRICING (daily files: OHLC/vwap of trades per Central-time trade date, no bid/ask, no timestamps):
  Price file: D-1 for stations at or west of US Eastern (all of F1, F3; F2 YUL/YYZ/YHC); D-2 for stations east of the
  Atlantic (F2 DDF/FPO/JTT/SSS/SSY), because the D-1 Central date ends 07:00-15:00 local on the EVENT day there
  (lookahead). Bought instrument only, and only if it traded on that file date (pair_quantity > 0).
    T1 (primary): last trade (end_price).   T2 (worst-price robustness): that day's high_price (worst any buyer paid).
  Fee $0.01/contract (ForecastEx per-side fee; IBKR commission 0). cost = price + 0.01, take only if 0 < price and
  cost < 1. 1 unit staked at cost; pnl = 1/cost - 1 if the bought side settles $1, else -1.
  Known bias, F1 East-coast stations: the D-1 Central date ends 01:00 ET on D, i.e. one hour of the event day (the
  midnight temperature bounds the day's min from above). Reported as an Eastern vs non-Eastern split (descriptive).

SETTLEMENT: YES settlement_price in the latest row dated >= the event date, only if exactly 0/1.

DATA/SPLIT: Q007 scratch pull (~/.claude/jobs/588c8b26/tmp/fx_days/*.parquet: files 2024-08-01..2026-08-14, contracts
  EXPIRING < 2026-08-15 only, rows that traded or are within 1 day of expiry). DEV ONLY: event dates < 2026-08-15; no
  sealed contract exists in the input. A later sealed test (event >= 2026-08-15) would need the user's yes.

TRIALS / BAR (pre-registered 2026-10-28, written before any price was joined to a forecast or outcome; dataset
  "forecastex_wx2"): F1_T1, F1_T2, F2_T1, F2_T2, F3_T1, F3_T2 = 6 trials -> bar p < 0.05/6 = 0.0083 (or 0.05 / #ok
  trials in this dataset in lab/ledger.jsonl if larger). Test: one-sided bootstrap over station-days, 5,000 resamples,
  seed 0, p = P(sum <= 0). Family verdict keys on T1:
    FAIL if no bets, p >= bar, units <= 0, or the loss check fails (Wilson 95% upper bound of the loss rate must be
    below the break-even loss rate 1 - mean cost);  T1 pass + T2 fail -> "-FRAGILE";
    PASS-UNDERPOWERED if T1 passes with < 500 bets (500-bet floor, as E5: thin, last-trade-priced, not resolvable);
    PASS only with >= 500 bets. No verdict authorises money; best case buys a sealed shot / forward paper.
  Reported per family: n, station-days, losses + Wilson CI vs break-even, units, ROI, bootstrap CI, halves by date,
  NO/YES side, by city, by month. MECHANISM note: F1 is Kalshi X2's mechanism (dead there); F3 is E5's own mechanism
  on an earlier period (a replication, not a new edge).
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

from sports import wx
from sports.http import fetch

SRC = Path.home() / ".claude/jobs/588c8b26/tmp/fx_days"
OUT = ROOT / "data" / "forecastex_wx2.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "forecastex_wx2"
SPLIT = pd.Timestamp("2026-08-15")
FEE = 0.01
TRIALS = {"T1": "end_price", "T2": "high_price"}
FAMS = ["UL", "SH", "DH"]
NAME = {"UL": "F1", "SH": "F2", "DH": "F3"}
MODELS = {"gfs_seamless": "2021-03-01", "ecmwf_ifs025": "2024-03-01", "icon_seamless": "2024-03-01",
          "gem_seamless": "2024-03-01", "ncep_nbm_conus": "2025-03-01"}
# F2 stations: code -> (lat, lon, tz, file lag days)
INTL = {"DDF": (50.0333, 8.5706, "Europe/Berlin", 2), "FPO": (48.7233, 2.3794, "Europe/Paris", 2),
        "JTT": (35.5523, 139.7798, "Asia/Tokyo", 2), "SSS": (1.3502, 103.9940, "Asia/Singapore", 2),
        "SSY": (-33.9461, 151.1772, "Australia/Sydney", 2), "YHC": (49.2875, -123.1167, "America/Vancouver", 1),
        "YUL": (45.4706, -73.7408, "America/Toronto", 1), "YYZ": (43.6772, -79.6306, "America/Toronto", 1)}
DH_EXCLUDE = {"NYC", "TOR", "YUL", "YVR"}


def load() -> pd.DataFrame:
    fr = []
    for f in sorted(SRC.glob("*.parquet")):
        t = pd.read_parquet(f)
        t = t[t.event_contract.astype(str).str.match(r"^(UL|SH|DH)[A-Z]{3}_")]
        if len(t):
            fr.append(t)
    raw = pd.concat(fr, ignore_index=True)
    p = raw.event_contract.str.extract(r"^(UL|SH|DH)([A-Z]{3})_(\d{6})_(-?\d+(?:\.\d+)?)$")
    raw["fam"], raw["code"] = p[0], p[1]
    raw["event_date"] = pd.to_datetime(p[2], format="%m%d%y", errors="coerce")
    raw["X"] = pd.to_numeric(p[3], errors="coerce")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw.fam.notna() & raw.event_date.notna() & (raw.event_date < SPLIT)]
    raw = raw[~((raw.fam == "SH") & ~raw.code.isin(list(INTL)))]
    raw = raw[~((raw.fam == "DH") & raw.code.isin(DH_EXCLUDE))]
    return raw


def model_daily(m, lat, lon, tz, start, end, low: bool, celsius: bool):
    frames, s = [], pd.Timestamp(start)
    while s <= pd.Timestamp(end):
        e = min(s + pd.Timedelta(days=364), pd.Timestamp(end))
        done = e < pd.Timestamp.today().normalize() - pd.Timedelta(days=3)
        d = fetch(wx.PREV_RUNS, {"latitude": lat, "longitude": lon, "models": m,
                                 "hourly": "temperature_2m_previous_day1",
                                 "temperature_unit": "celsius" if celsius else "fahrenheit",
                                 "timezone": tz, "start_date": s.date().isoformat(), "end_date": e.date().isoformat()}, cache=done)
        h = pd.DataFrame(d.get("hourly", {}))
        if not h.empty:
            t = pd.to_datetime(h["time"])
            h["day"] = t.dt.normalize()
            if low:
                g = h.groupby("day")["temperature_2m_previous_day1"]
                v = g.min().where(g.count() >= 20)
            else:
                g = h[t.dt.hour <= 18].groupby("day")["temperature_2m_previous_day1"]
                v = g.max().where(g.count() >= 15)
            frames.append(v.rename(f"fc_{m}"))
        s = e + pd.Timedelta(days=1)
    return pd.concat(frames) if frames else None


def cmd_build() -> int:
    raw = load()
    print("rows by family:", raw.groupby("fam").size().to_dict(), "event days:", raw.groupby("fam").event_date.nunique().to_dict())
    y = raw[(raw.subtype == "YES") & (raw.date >= raw.event_date)].sort_values("date")
    fin = y.groupby("event_contract").tail(1).set_index("event_contract")["settlement_price"]
    fin = fin.where(fin.isin([0.0, 1.0]))
    k = raw.drop_duplicates("event_contract").set_index("event_contract")[["fam", "code", "event_date", "X"]]
    k["lag"] = np.where((k.fam == "SH") & k.code.map(lambda c: INTL.get(c, (0, 0, 0, 1))[3] == 2), 2, 1)
    k["price_date"] = k.event_date - pd.to_timedelta(k.lag, unit="D")
    cols = ["high_price", "end_price", "vwap", "pair_quantity"]
    pr = raw.join(k[["price_date"]], on="event_contract")
    pr = pr[pr.date == pr.price_date]
    yes = pr[pr.subtype == "YES"].drop_duplicates("event_contract").set_index("event_contract")[cols].add_prefix("yes_")
    no = pr[pr.subtype == "NO"].drop_duplicates("event_contract").set_index("event_contract")[cols].add_prefix("no_")
    df = k.join(fin.rename("yes_won")).join(yes).join(no).reset_index()
    # direction from settlement ladder only
    def direction(g):
        g = g.dropna(subset=["yes_won"]).sort_values("X")
        d = np.diff(g.yes_won.to_numpy())
        return "dec" if (d <= 0).all() and (d < 0).any() else "inc" if (d >= 0).all() and (d > 0).any() else (
            "flat" if (d == 0).all() else "bad")
    dr = df.groupby(["fam", "code", "event_date"]).apply(direction).rename("ladder")
    df = df.join(dr, on=["fam", "code", "event_date"])
    sem = {}
    for fam, g in dr.groupby(level=0):
        v = g.value_counts()
        inf = v.get("dec", 0) + v.get("inc", 0)
        dec, inc = v.get("dec", 0) / max(inf, 1), v.get("inc", 0) / max(inf, 1)
        sem[fam] = "exceed" if dec >= .95 else "below" if inc >= .95 else "ABORT"
        print(f"{fam}: ladders {v.to_dict()} -> {sem[fam]}")
    df["sem"] = df.fam.map(sem)
    df["monotone"] = np.where(df["sem"] == "exceed", df.ladder.isin(["dec", "flat"]),
                              np.where(df["sem"] == "below", df.ladder.isin(["inc", "flat"]), False))
    sh = df[df.fam == "SH"].X
    print(f"SH median strike {sh.median()} (gate < 40 for Celsius)")
    out = []
    for (fam, code), g in df.groupby(["fam", "code"]):
        if fam == "SH":
            lat, lon, tz, _ = INTL[code]
        else:
            try:
                p = fetch(f"https://api.weather.gov/stations/K{code}")
                lon, lat = p["geometry"]["coordinates"][:2]
                tz = p["properties"]["timeZone"]
            except Exception as e:  # noqa: BLE001
                print(f"{fam}{code}: station lookup failed ({e}); skipped")
                continue
        g = g.copy()
        g["tz"] = tz
        for m, start in MODELS.items():
            try:
                f = model_daily(m, lat, lon, tz, max(start, str(g.event_date.min().date())), str(g.event_date.max().date()),
                                low=(fam == "UL"), celsius=(fam == "SH"))
            except Exception as e:  # noqa: BLE001
                print(f"  {fam}{code} {m}: {e}")
                f = None
            if f is not None:
                g = g.join(f, on="event_date")
        out.append(g)
        print(f"{fam}{code} {tz}: {len(g)} contracts, {g.event_date.nunique()} days", flush=True)
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(res)} rows)")
    return 0


def margin(fam):
    return 5 / 3 if fam == "SH" else 3.0


def candidates(df):
    cols = [c for c in df.columns if c.startswith("fc_")]
    n = df[cols].notna().sum(1)
    hi, lo = df[cols].max(1), df[cols].min(1)
    M = df.fam.map(margin)
    ex = df["sem"] == "exceed"
    side = np.where(ex, np.where(hi + M < df.X + 1, "NO", np.where(lo - M > df.X, "YES", "")),
                    np.where(lo - M > df.X - 1, "NO", np.where(hi + M < df.X, "YES", "")))
    d = df.assign(side=side)
    return d[(d.side != "") & (n >= 3) & d.monotone & d["sem"].isin(["exceed", "below"])]


def picks(df, field):
    d = candidates(df).copy()
    is_no = d.side == "NO"
    d["price"] = np.where(is_no, d[f"no_{field}"], d[f"yes_{field}"])
    d["qty"] = np.where(is_no, d.no_pair_quantity, d.yes_pair_quantity)
    d = d[(d.qty > 0) & (d.price > 0) & (d.price + FEE < 1.0) & d.yes_won.notna()].copy()
    d["cost"] = d.price + FEE
    d["won"] = np.where(d.side == "NO", d.yes_won == 0, d.yes_won == 1)
    d["pnl"] = np.where(d.won, 1 / d.cost - 1, -1.0)
    return d


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def score(t, rng):
    if t.empty:
        return {"bets": 0, "units": 0.0, "roi": 0.0, "p": 1.0, "losses": 0}
    t = t.assign(day=t.code + t.event_date.astype(str))
    s = t.groupby("day").pnl.sum().to_numpy()
    boots = np.array([rng.choice(s, len(s)).sum() for _ in range(5000)])
    mid = t.event_date.min() + (t.event_date.max() - t.event_date.min()) / 2
    h1, h2 = t[t.event_date <= mid], t[t.event_date > mid]
    lo, hi = wilson(int((~t.won).sum()), len(t))
    return {"bets": int(len(t)), "station_days": int(len(s)), "units": float(t.pnl.sum()), "roi": float(t.pnl.mean()),
            "ci_lo": float(np.quantile(boots, .025)), "ci_hi": float(np.quantile(boots, .975)), "p": float(np.mean(boots <= 0)),
            "losses": int((~t.won).sum()), "loss_rate": float((~t.won).mean()), "loss_ci": [float(lo), float(hi)],
            "breakeven_loss_rate": float(1 - t.cost.mean()), "mean_cost": float(t.cost.mean()),
            "units_H1": float(h1.pnl.sum()), "bets_H1": int(len(h1)), "units_H2": float(h2.pnl.sum()), "bets_H2": int(len(h2))}


def cmd_dev() -> int:
    df = pd.read_parquet(OUT)
    df["event_date"] = pd.to_datetime(df.event_date)
    df = df[df.event_date < SPLIT]
    for fam, g in df.groupby("fam"):
        print(f"{fam}: {g.code.nunique()} stations, {g.groupby('code').event_date.nunique().sum()} station-days, "
              f"{g.event_date.min().date()}..{g.event_date.max().date()}, sem {g["sem"].iloc[0]}, "
              f"non-monotone station-days dropped {g[~g.monotone].groupby(['code', 'event_date']).ngroups}")
    # power (prices only)
    c = candidates(df)
    for fam, g in c.groupby("fam"):
        is_no = g.side == "NO"
        pr = pd.Series(np.where(is_no, g.no_end_price, g.yes_end_price), index=g.index)
        q = pd.Series(np.where(is_no, g.no_pair_quantity, g.yes_pair_quantity), index=g.index)
        ok = (q > 0) & (pr > 0) & (pr + FEE < 1)
        print(f"  power {fam}: candidates {len(g)} (NO {int(is_no.sum())}), traded on price date {int((q > 0).sum())}, "
              f"takeable {int(ok.sum())}, mean cost {(pr[ok] + FEE).mean():.4f}, median qty {q[ok].median()}")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {x["sha"] for x in prior if x.get("dataset") == DATASET and x.get("status") == "ok"}
    names = [f"{NAME[f]}_{t}" for f in FAMS for t in TRIALS]
    for nm in names:
        shas.add(hashlib.sha256(f"forecastex_wx2:{nm}".encode()).hexdigest()[:12])
    bar = 0.05 / len(shas)
    recs, res = [], {}
    for fam in FAMS:
        for tr, field in TRIALS.items():
            nm = f"{NAME[fam]}_{tr}"
            t = picks(df[df.fam == fam], field)
            r = score(t, np.random.default_rng(0))
            loss_ok = r["bets"] > 0 and r["loss_ci"][1] < r["breakeven_loss_rate"]
            ok = r["bets"] > 0 and r["p"] < bar and r["units"] > 0 and loss_ok
            res[nm] = (t, r, ok)
            print(f"{nm} ({fam}): bets {r['bets']} sd {r.get('station_days', 0)} losses {r['losses']} "
                  f"loss {r.get('loss_rate', 0):.2%} CI {np.round(r.get('loss_ci', [0, 0]), 4).tolist()} vs b/e "
                  f"{r.get('breakeven_loss_rate', 0):.2%}; units {r['units']:+.2f} ROI {r['roi']:+.2%} CI "
                  f"[{r.get('ci_lo', 0):+.2f},{r.get('ci_hi', 0):+.2f}] p {r['p']:.4f} bar {bar:.4f} -> {'pass' if ok else 'fail'}")
            if len(t) and tr == "T1":
                print(f"   halves H1 {r['units_H1']:+.2f}/{r['bets_H1']}  H2 {r['units_H2']:+.2f}/{r['bets_H2']}")
                print("   side:", t.groupby("side").pnl.agg(["size", "sum"]).round(2).to_dict("index"))
                print("   city:", t.groupby("code").pnl.agg(["size", "sum"]).round(2).to_dict("index"))
                print("   month:", t.groupby(t.event_date.dt.strftime("%Y-%m")).pnl.agg(["size", "sum"]).round(2).to_dict("index"))
                print("   cost<0.80:", t[t.cost < .8].pnl.agg(["size", "sum"]).round(2).to_dict(),
                      " cost>=0.80:", t[t.cost >= .8].pnl.agg(["size", "sum"]).round(2).to_dict())
                if fam == "UL":
                    east = t.tz.isin(["America/New_York", "America/Detroit", "America/Indiana/Indianapolis"])
                    print("   Eastern:", t[east].pnl.agg(["size", "sum"]).round(2).to_dict(),
                          " non-Eastern:", t[~east].pnl.agg(["size", "sum"]).round(2).to_dict())
            sha = hashlib.sha256(f"forecastex_wx2:{nm}".encode()).hexdigest()[:12]
            recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"forecastex_wx2_{nm}",
                         "sha": sha, "dataset": DATASET, "trial": nm, "family": fam,
                         "hypothesis": "pre-registered in scripts/forecastex_wx2.py; E1/E5 consensus-reject on ForecastEx "
                         f"{fam} daily weather, D-1/D-2 daily-file trade price ({field}) + $0.01 fee",
                         "source": "E5 breadth (notes/q007_forecastex_breadth.md)", "status": "ok",
                         "study_trials": len(shas), "bonferroni_bar": bar, **r, "loss_check": bool(loss_ok),
                         "verdict": "pass" if ok else "fail", "underpowered": r["bets"] < 500})
    for fam in FAMS:
        _, r1, ok1 = res[f"{NAME[fam]}_T1"]
        _, _, ok2 = res[f"{NAME[fam]}_T2"]
        v = "FAIL" if not ok1 else ("PASS" if r1["bets"] >= 500 else "PASS-UNDERPOWERED") + ("" if ok2 else "-FRAGILE")
        print(f"VERDICT {NAME[fam]} {fam}: {v}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for x in recs:
                f.write(json.dumps(x, default=str) + "\n")
        print("appended to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
