"""ForecastEx (Interactive Brokers) daily-HIGH temperature contracts: does the E1/E2 consensus-reject rule work there?

    python scripts/forecastex_wx.py pull            # DEV-range public daily price files -> data/forecastex/ (weather rows only)
    python scripts/forecastex_wx.py build           # contracts + D-1 trade prices + final outcomes + 5-model forecasts
    python scripts/forecastex_wx.py dev [--ledger]  # power check, then T1/T2 on DEV rows; WU-vs-NWS disagreement; overlap with Kalshi
    (no `sealed` command exists yet: it is written only after the user's explicit yes, against this same docstring)

WHY: notes/venue_expansion.md section 3. ForecastEx lists daily highs at 24 US airports, 18 of them the same airports as
our Kalshi E1/E2 series, but settles on Weather Underground's "Daily Observations" table instead of the NWS CLI. A pass
here would be a second weather book with partly independent settlement. Nothing here places an order or needs a login.

CONTRACT (CFTC self-certification ptc08062614351, text read 2026-10-28 before any price was loaded):
  "Will the highest temperature in [region] EXCEED [##]" -> YES iff WU daily max > X (strict). Pays $1/$0. YES and NO are
  separate instruments (each trade is a YES/NO pair, so a NO trade price = 1 - the paired YES price). Last trade 23:59
  local on the contract date. Resolves on the final WU Daily Observations max once the next day's first observation posts.
  Ticker UH<CODE>_<MMDDYY>_<X>, e.g. UHLGA_101926_63. Earlier (Jan 2026) wording used WU's Summary "Actual" value; the
  Daily-Observations wording dates from the Aug 2026 filing (UNVERIFIED exact effective day) -> DEV is mostly the old
  wording, SEALED mostly the new one.

RULE (frozen from scripts/kalshi_paper.py / lab/strategies/kalshi_reject_safe_check.py; nothing fitted):
  models {gfs_seamless, ecmwf_ifs025, icon_seamless, gem_seamless, ncep_nbm_conus}, Open-Meteo previous-runs lead-1-day,
  local-day max over hours <= 18:00 (>= 15 hourly values; scripts/wx_expand.model_ext), at the WU station's coordinates
  (K<CODE> from api.weather.gov, e.g. KLGA for NYC, KBKF for Denver). >= 3 models available.
  An "exceed X" contract is the Kalshi "greater" bracket with floor X, and its complement "max <= X" is the Kalshi "less"
  bracket with cap X+1. So, identical to E1:
    hi + 3 < X + 1   (all models miss "> X")     -> buy NO  of contract X
    lo - 3 > X       (all models miss "<= X")    -> buy YES of contract X   (= E1's NO on a low-tail "less" bracket)
  Only daily HIGHS (Kalshi X2 lows failed dev). All 24 F-cities (UH*); Celsius/hourly products excluded.

PRICING (daily files have NO bid/ask and NO intraday time stamps -- the main limitation):
  Decision would be 22:00 station-local on D-1. The only D-1 prices are the file dated D-1: start/high/low/end/vwap of
  TRADES on that trade date (exchange date convention UNVERIFIED; expiration stamps are in -05/-06 Central offsets, so the
  date is most likely Central time). Bought instrument only, and only if it traded on D-1 (pair_quantity > 0), because a
  no-trade row's prices are marks of unknown construction.
    T1 (primary)   : price = bought instrument's D-1 end_price (last trade of D-1).
    T2 (robustness): price = bought instrument's D-1 high_price (the WORST price any buyer of that side paid on D-1).
  Fee $0.01/contract (IBKR commission 0 + ForecastEx exchange fee). cost = price + 0.01; take only if cost < 1.00
  (i.e. price <= 0.98; the analogue of E1's yes_ask > 0.01) and price > 0. Stake 1 unit of cost per bet;
  pnl = 1/cost - 1 if the bought side settles $1, else -1.
  Stated biases: (a) a last trade is not a quote: if it printed at the bid of our side, we could not have bought there,
  so T1 is optimistic by up to a half spread -- T2 bounds this from the other side; (b) the last trade may print after
  22:00 local (East: up to ~2 h of evening information) or many hours before (West: Central-day end = 22:00 PT); the
  forecasts are lead-1 runs issued before the decision either way; (c) requiring a D-1 trade drops untraded strikes.

SETTLEMENT: outcome = the YES instrument's settlement_price in the LATEST file row of that contract dated >= its
  expiration day, accepted only if exactly 0.00 or 1.00 (else the contract is dropped and counted). A station-day whose
  YES outcomes are not monotone in X is dropped (counted).

PRE-REGISTERED 2026-10-28, written before any ForecastEx price file was downloaded into this repo (the feasibility agent
had looked at 2026-02-01, 03-01 and 10-13..10-26 files for coverage/liquidity only -- 10-13+ is SEALED and is not used):
  dataset "forecastex_wx". Data run 2026-03-01 (first date with ~23 high cities) .. ~2026-10-26.
  SPLIT by EVENT date: DEV = 2026-03-01 .. 2026-08-14 (168 of ~240 days, 70%); SEALED = 2026-08-15 .. last settled.
  `pull` downloads files dated 2026-02-28 .. 2026-08-15 and keeps only rows of contracts with event date < 2026-08-15,
  so no sealed contract is ever written to disk by this script.
  Trials: T1, T2 (2 trials) -> bar p < 0.05/2 = 0.025 each (or 0.05/#ok trials in this dataset in lab/ledger.jsonl).
  Test: one-sided bootstrap over station-days (5,000 resamples of station-day pnl sums), p = P(sum <= 0).
  PASS(dev) for a trial = p < bar AND units > 0. Verdict for the venue keys on T1; T2 failing with T1 passing = "fragile".
  Also reported: n, loss count + Wilson 95% CI on the loss rate vs the break-even loss rate (1 - mean cost), ROI,
  first/second-half of DEV by date (sign check), by city, by direction (NO vs YES).
  Power, checked BEFORE outcomes are looked at: n picks and mean cost. E1 earns ~+1.5% at cost ~0.97 with per-bet sd
  ~0.18, so ~550 independent bets give p ~ .05 for a true +1.5%; with the $0.01 fee eating ~1%, a true edge equal to
  E1's gross would net ~+0.5% and need several thousand. If n < 500 the result is reported as "not resolvable".
  Independence (descriptive, not trials):
   W1 WU-vs-NWS: on station-days present on both venues, the WU max interval implied by ForecastEx outcomes
      (max > largest X with YES=1, max <= smallest X with YES=0) vs Kalshi's official value (NWS CLI / TWC);
      disagreement rate = Kalshi value outside the interval. Same-airport stations only.
   W2 overlap: station-days where both this rule (T1) and Kalshi E1/E2 picks exist (data/kalshi_temp_maker_picks*.parquet);
      share of ForecastEx losses that fall on a station-day where a Kalshi pick also lost.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import requests

from sports import wx
from sports.http import fetch

DATA = ROOT / "data" / "forecastex"
OUT = ROOT / "data" / "forecastex_wx.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "forecastex_wx"
FIRST_EVENT = date(2026, 3, 1)
SPLIT = date(2026, 8, 15)  # event dates >= SPLIT are SEALED
FEE = 0.01
MARGIN = 3.0
TRIALS = {"T1_end": "end_price", "T2_high": "high_price"}
URL = "https://forecastex.com/api/download"
H = {"User-Agent": "Mozilla/5.0"}
# ForecastEx code -> Kalshi series (same airport) for W1/W2
KALSHI = {"MDW": "KXHIGHCHI", "MIA": "KXHIGHMIA", "AUS": "KXHIGHAUS", "LAX": "KXHIGHLAX", "PHL": "KXHIGHPHIL",
          "ATL": "KXHIGHTATL", "BOS": "KXHIGHTBOS", "DFW": "KXHIGHTDAL", "DCA": "KXHIGHTDC", "HOU": "KXHIGHTHOU",
          "LAS": "KXHIGHTLV", "MSP": "KXHIGHTMIN", "MSY": "KXHIGHTNOLA", "OKC": "KXHIGHTOKC", "PHX": "KXHIGHTPHX",
          "SAT": "KXHIGHTSATX", "SEA": "KXHIGHTSEA", "SFO": "KXHIGHTSFO"}


def parse_ticker(s: pd.Series) -> pd.DataFrame:
    p = s.str.extract(r"^UH([A-Z]{3})_(\d{6})_(-?\d+(?:\.\d+)?)$")
    return pd.DataFrame({"code": p[0], "event_date": pd.to_datetime(p[1], format="%m%d%y", errors="coerce"),
                         "X": pd.to_numeric(p[2], errors="coerce")}, index=s.index)


def cmd_pull() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    d, last = FIRST_EVENT - timedelta(days=1), SPLIT
    n = 0
    while d <= last:
        f = DATA / f"uh_{d:%Y%m%d}.csv"
        if not f.exists():
            r = requests.get(URL, params={"type": "prices", "date": f"{d:%Y%m%d}"}, headers=H, timeout=120)
            r.raise_for_status()
            t = pd.read_csv(io.StringIO(r.text))
            t = t[t.event_contract.astype(str).str.startswith("UH")]
            meta = parse_ticker(t.event_contract.astype(str))
            t = t[meta.event_date.notna() & (meta.event_date < pd.Timestamp(SPLIT))]  # never write a sealed contract
            t.to_csv(f, index=False)
            n += 1
            print(f"{d}: {len(t)} UH rows (dev contracts)", flush=True)
        d += timedelta(days=1)
    print(f"pulled {n} new files into {DATA}")
    return 0


def station_meta(code: str):
    p = fetch(f"https://api.weather.gov/stations/K{code}")
    lon, lat = p["geometry"]["coordinates"][:2]
    return float(lat), float(lon), p["properties"]["timeZone"]


def cmd_build() -> int:
    from wx_expand import MODELS, model_ext
    fr = []
    for f in sorted(DATA.glob("uh_*.csv")):
        t = pd.read_csv(f)
        if len(t):
            fr.append(t)
    raw = pd.concat(fr, ignore_index=True)
    raw = raw.join(parse_ticker(raw.event_contract.astype(str)))
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw.code.notna() & (raw.event_date < pd.Timestamp(SPLIT)) & (raw.event_date >= pd.Timestamp(FIRST_EVENT))]
    # outcome: YES settlement in the latest row dated >= event date, only if exactly 0/1
    y = raw[(raw.subtype == "YES") & (raw.date >= raw.event_date)].sort_values("date")
    fin = y.groupby("event_contract").tail(1).set_index("event_contract")["settlement_price"]
    fin = fin.where(fin.isin([0.0, 1.0]))
    # D-1 trade prices of both instruments
    prev = raw[raw.date == raw.event_date - pd.Timedelta(days=1)]
    cols = ["start_price", "high_price", "low_price", "end_price", "vwap", "pair_quantity"]
    yes = prev[prev.subtype == "YES"].set_index("event_contract")[cols].add_prefix("yes_")
    no = prev[prev.subtype == "NO"].set_index("event_contract")[cols].add_prefix("no_")
    k = raw.drop_duplicates("event_contract").set_index("event_contract")[["code", "event_date", "X"]]
    df = k.join(fin.rename("yes_won")).join(yes).join(no).reset_index()
    print(f"contracts {len(df)}, final 0/1 outcome {df.yes_won.notna().sum()}, traded D-1 (YES or NO row qty>0) "
          f"{((df.yes_pair_quantity > 0) | (df.no_pair_quantity > 0)).sum()}")
    # monotonicity: YES must be 1 for every X below the max and 0 above
    def mono(g):
        g = g.dropna(subset=["yes_won"]).sort_values("X")
        return bool((np.diff(g.yes_won.to_numpy()) <= 0).all())
    ok = df.groupby(["code", "event_date"]).apply(mono).rename("monotone")
    df = df.join(ok, on=["code", "event_date"])
    print(f"station-days {len(ok)}, non-monotone {int((~ok).sum())}")
    out = []
    for code, g in df.groupby("code"):
        try:
            lat, lon, tz = station_meta(code)
        except requests.HTTPError as e:
            print(f"{code}: station lookup failed {e}; skipped")
            continue
        g = g.copy()
        g["tz"] = tz
        dd = g.set_index("event_date")
        for m, start in MODELS.items():
            fc = model_ext(m, lat, lon, tz, max(start, str(g.event_date.min().date())), str(g.event_date.max().date()), low=False)
            if fc is not None:
                g = g.join(fc, on="event_date")
        out.append(g)
        print(f"{code} {tz}: {len(g)} contracts, {g.event_date.nunique()} days", flush=True)
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(res)} rows)")
    return 0


def candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Rule side for each contract, no prices or outcomes used."""
    cols = [c for c in df.columns if c.startswith("fcsafe_")]
    n = df[cols].notna().sum(1)
    hi, lo = df[cols].max(1), df[cols].min(1)
    side = np.where(hi + MARGIN < df.X + 1, "NO", np.where(lo - MARGIN > df.X, "YES", ""))
    d = df.assign(side=side)
    return d[(d.side != "") & (n >= 3) & d.monotone]


def picks(df: pd.DataFrame, field: str) -> pd.DataFrame:
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
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def score(t: pd.DataFrame, rng) -> dict:
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
            "units_H1": float(h1.pnl.sum()), "roi_H1": float(h1.pnl.mean()) if len(h1) else 0.0, "bets_H1": int(len(h1)),
            "units_H2": float(h2.pnl.sum()), "roi_H2": float(h2.pnl.mean()) if len(h2) else 0.0, "bets_H2": int(len(h2))}


def kalshi_values(code: str) -> pd.Series:
    mk = wx.settled_markets(KALSHI[code])
    mk["date"] = mk.event_ticker.map(wx.event_date)
    return mk.groupby("date").expiration_value.max()


def cmd_dev() -> int:
    df = pd.read_parquet(OUT)
    df["event_date"] = pd.to_datetime(df.event_date)
    df = df[df.event_date < pd.Timestamp(SPLIT)]  # sealed never present, belt and braces
    print(f"DEV {df.code.nunique()} cities, {df.groupby('code').event_date.nunique().sum()} station-days, "
          f"{df.event_date.min().date()}..{df.event_date.max().date()}, {len(df)} contracts")
    # ---- power check (prices only, no outcomes)
    c = candidates(df)
    print(f"rule candidates (any price) {len(c)}: NO {int((c.side == 'NO').sum())}, YES {int((c.side == 'YES').sum())}")
    for name, field in TRIALS.items():
        is_no = c.side == "NO"
        pr = pd.Series(np.where(is_no, c[f"no_{field}"], c[f"yes_{field}"]), index=c.index)
        q = pd.Series(np.where(is_no, c.no_pair_quantity, c.yes_pair_quantity), index=c.index)
        ok = (q > 0) & (pr > 0) & (pr + FEE < 1)
        print(f"  power {name}: traded D-1 {int((q > 0).sum())}, takeable (price+fee<1) {int(ok.sum())}, "
              f"mean cost {(pr[ok] + FEE).mean():.4f}, station-days {c[ok].groupby(['code', 'event_date']).ngroups}")
    # ---- trials
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {x["sha"] for x in prior if x.get("dataset") == DATASET and x.get("status") == "ok"}
    recs, res = [], {}
    for name, field in TRIALS.items():
        shas.add(hashlib.sha256(f"forecastex_wx:{name}".encode()).hexdigest()[:12])
    bar = 0.05 / len(shas)
    for name, field in TRIALS.items():
        t = picks(df, field)
        r = score(t, np.random.default_rng(0))
        res[name] = (t, r)
        verdict = "pass" if (r["p"] < bar and r["units"] > 0) else "fail"
        print(f"{name}: bets {r['bets']} station-days {r.get('station_days', 0)} losses {r['losses']} "
              f"loss rate {r.get('loss_rate', 0):.3%} CI [{r.get('loss_ci', [0, 0])[0]:.3%}, {r.get('loss_ci', [0, 0])[1]:.3%}] "
              f"vs break-even {r.get('breakeven_loss_rate', 0):.3%}; units {r['units']:+.2f} ROI {r['roi']:+.2%} "
              f"95%CI [{r.get('ci_lo', 0):+.2f}, {r.get('ci_hi', 0):+.2f}] p {r['p']:.4f} (bar {bar:.4f}) -> {verdict.upper()}")
        if len(t):
            print(f"   halves: H1 {r['units_H1']:+.2f}/{r['bets_H1']} ({r['roi_H1']:+.2%})  H2 {r['units_H2']:+.2f}/{r['bets_H2']} ({r['roi_H2']:+.2%})")
            print("   by side:", {k: f"{v['sum']:+.2f}/{int(v['size'])}" for k, v in t.groupby('side').pnl.agg(['size', 'sum']).round(2).iterrows()})
            print("   by city:", {k: f"{v['sum']:+.2f}/{int(v['size'])}" for k, v in t.groupby('code').pnl.agg(['size', 'sum']).round(2).iterrows()})
            print("   by month:", t.groupby(t.event_date.dt.strftime('%Y-%m')).pnl.agg(['size', 'sum']).round(2).to_dict('index'))
        sha = hashlib.sha256(f"forecastex_wx:{name}".encode()).hexdigest()[:12]
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"forecastex_wx_{name}",
                     "sha": sha, "dataset": DATASET, "trial": name,
                     "hypothesis": "pre-registered in scripts/forecastex_wx.py; E1/E2 consensus-reject frozen, ported to "
                     "ForecastEx daily-high thresholds (WU settlement), D-1 daily-file trade price + $0.01 fee",
                     "source": "venue expansion (notes/venue_expansion.md sec 3)", "status": "ok",
                     "study_trials": len(shas), "bonferroni_bar": bar, **r, "verdict": verdict})
    # ---- W1 WU vs NWS
    rows = []
    for code in sorted(set(df.code) & set(KALSHI)):
        kv = kalshi_values(code)
        g = df[(df.code == code) & df.monotone & df.yes_won.notna()]
        for d, gg in g.groupby("event_date"):
            if d not in kv.index:
                continue
            lo = gg.loc[gg.yes_won == 1, "X"].max()  # max > lo
            hi = gg.loc[gg.yes_won == 0, "X"].min()  # max <= hi
            v = kv[d]
            out = (not np.isnan(lo) and v <= lo) or (not np.isnan(hi) and v > hi)
            rows.append({"code": code, "date": d, "k": v, "lo": lo, "hi": hi, "disagree": out,
                         "k_higher": (not np.isnan(hi) and v > hi), "k_lower": (not np.isnan(lo) and v <= lo),
                         "width": hi - lo})
    w = pd.DataFrame(rows)
    if len(w):
        print(f"W1 WU-vs-NWS: {len(w)} matched station-days ({w.code.nunique()} airports); Kalshi value outside the "
              f"ForecastEx-implied WU interval on {int(w.disagree.sum())} ({w.disagree.mean():.2%}): Kalshi higher "
              f"{int(w.k_higher.sum())}, lower {int(w.k_lower.sum())}; median interval width {w.width.median():.0f}F "
              f"(share with width 1 = exact: {(w.width == 1).mean():.1%})")
        ex = w[w.width == 1]
        if len(ex):
            print(f"   exact-identified days only: {len(ex)}, disagree {int(ex.disagree.sum())} ({ex.disagree.mean():.2%})")
        print("   by airport:", w.groupby("code").disagree.agg(["size", "sum"]).to_dict("index"))
    # ---- W2 overlap with Kalshi E1/E2 picks
    kp = pd.concat([pd.read_parquet(ROOT / "data" / "kalshi_temp_maker_picks.parquet"),
                    pd.read_parquet(ROOT / "data" / "kalshi_temp_maker_picks_sealed.parquet")]).drop_duplicates("ticker")
    inv = {v: k for k, v in KALSHI.items()}
    kp["code"] = kp.series.map(inv)
    kp = kp[kp.code.notna()]
    kp["date"] = pd.to_datetime(kp.date)
    kday = kp.groupby(["code", "date"]).agg(k_picks=("ticker", "size"), k_lost=("won", lambda s: int((~s.astype(bool)).sum())))
    t, r = res["T1_end"]
    if len(t):
        fday = t.groupby(["code", "event_date"]).agg(f_picks=("pnl", "size"), f_lost=("won", lambda s: int((~s).sum())), f_pnl=("pnl", "sum"))
        fday.index.names = ["code", "date"]
        same = fday[fday.index.get_level_values(0).isin(list(KALSHI))]
        j = same.join(kday, how="left")
        both = j.k_picks.notna()
        print(f"W2 overlap (T1, same airports, Kalshi picks 2024-03..2026-10): {len(j)} ForecastEx pick station-days, "
              f"{int(both.sum())} also have a Kalshi E1/E2 pick; ForecastEx losing days {int((j.f_lost > 0).sum())}, "
              f"of which Kalshi also lost that day {int(((j.f_lost > 0) & (j.k_lost > 0)).sum())}, Kalshi had a pick "
              f"{int(((j.f_lost > 0) & both).sum())}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for x in recs:
                f.write(json.dumps(x, default=str) + "\n")
        print("appended to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"pull": cmd_pull, "build": cmd_build, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
