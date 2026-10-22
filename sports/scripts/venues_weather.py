"""Port the sealed-passed Kalshi temperature consensus-reject rule to OTHER US venues + cross-venue arb check.

    python scripts/venues_weather.py build            # Polymarket US daily-high brackets + 22:00 quotes + 5-model fcsts + Kalshi twin quotes
    python scripts/venues_weather.py dev [--ledger]   # P1 on DEV rows only; cross-venue after-fee arb count (DEV rows)
    python scripts/venues_weather.py sealed --confirm P1   # ONE-TIME, only after the user's explicit yes
    python scripts/venues_weather.py paper tick       # forward paper logger for Polymarket US (no orders); see install note
    python scripts/venues_weather.py xlive            # live snapshot: same city/day/bracket on Kalshi vs Polymarket US books

VENUES WITH DAILY TEMPERATURE MARKETS (checked 2026-10-22, see notes/venues.md):
  * Polymarket US (CFTC DCM, own book): daily HIGH brackets for nyc/chicago/miami/LA/SF (since 2026-04-29) and philadelphia
    (since 2026-10-16); lows for some cities from 2026-10-16. Settles on the NWS CLI at KNYC, KMDW, KMIA, KLAX, KSFO, KPHL
    (docs.polymarket.us/faqs/weather-faqs.md + each market's text) = the SAME stations as Kalshi KXHIGHNY/CHI/MIA/LAX/PHIL and
    KXHIGHTSFO. Kalshi has settled via The Weather Company since 2026-08-14, which equals the NWS CLI value on 3,080/3,081 days
    (scripts/wx_source_check.py). Brackets: 2F inclusive bins + two tails, same encoding as Kalshi after mapping
    (polymarket_us.parse_bracket). Public price history: /v1/price-history custom range, ~1-minute book-derived quotes
    (longPrice = YES ask, shortPrice = 1 - YES bid), present back to 2026-05 (probed). -> BACKTESTABLE.
  * ForecastEx (IBKR; also reached via Robinhood): daily high/low "exceed X" thresholds, settles on Weather Underground
    Daily Observations at airport codes (e.g. LGA for NYC, not Central Park) -> different station AND source = a NEW trial,
    not a port. Only public data = daily OHLC/VWAP files (forecastex.com/api/download); no public quote at 22:00 and no
    public order book -> neither the exact backtest nor a key-free forward logger is possible. Not built.
  * No other US venue found with city temperature markets (Robinhood's are ForecastEx/Kalshi-routed; Coinbase/Webull route
    to Kalshi = same book).

RULE FROZEN (identical to lab/strategies/kalshi_reject_safe_check.py and scripts/kalshi_paper.py; nothing fitted here):
  models {gfs_seamless, ecmwf_ifs025, icon_seamless, gem_seamless, ncep_nbm_conus}, Open-Meteo previous-runs lead-1-day,
  local-day max over hours <= 18:00 (>= 15 hourly values); reject the bracket if even the most extreme available model
  (>= 3 available) misses it by > 3F (greater: hi+3 < floor+1; less: lo-3 > cap-1; between: hi+3 < floor or lo-3 > cap);
  buy NO at 1 - yes_bid if yes_ask > 0.01; tradeable quotes only (yes_bid > 0, yes_ask < 1, spread <= 10c).
  Decision = 22:00 station-local the day before; quote = last stored observation at or before 22:00 within the prior hour
  (Kalshi uses the hourly candle ending at or before 22:00 -- same instant).
  Fee: Polymarket US taker 0.0695 * p * (1-p) per contract (docs.polymarket.us/fees.md, effective 2026-10-19; applied to
  the whole history -- conservative vs any earlier lower schedule is UNVERIFIED either way).

PRE-REGISTERED 2026-10-22, written before any Polymarket temperature price or outcome was pulled in bulk (only the
handful of live quotes printed while probing the API's shape):
  dataset "pmus_temp"; cities = Polymarket US daily-HIGH series {nychigh, mdwhigh, miahigh, laxhigh, sfohigh, phlhigh};
  lows excluded (Kalshi X2 lows failed dev -6.4%).
  split : DEV = event dates < 2026-08-15 ; SEALED = 2026-08-15 .. last settled date (never loaded by `dev`; user's yes).
  P1    : the frozen rule on Polymarket US quotes with Polymarket fees. 1 unit of cost per bet.
          PASS(dev) = one-sided bootstrap p < 0.05 over station-days AND units > 0. (1 trial in this dataset.)
          Power warning, stated before scoring: the Kalshi edge is ~+1% ROI at NO ~0.96; a few hundred dev bets cannot
          reach p < 0.05 even if Polymarket has the same edge. A null here is "not resolvable", not "no edge".
  X1    : cross-venue (descriptive, not a hypothesis test, not ledgered): at the 22:00 decision, for every bracket listed
          identically on both venues, cost of YES on one + NO on the other incl. both fees; count cost < $1. Display
          quotes carry no size, so any hit is a candidate, not a fill. Settlement-source basis risk (TWC vs CLI) noted.
  SEALED P1: same rule/scorer on rows >= 2026-08-15; PASS iff p < 0.05 and units > 0. Run once, only on user's yes.

AMENDMENT 2026-10-22, after `build`, BEFORE any P1 P&L was computed (only quote-quality counts had been looked at):
  The price-history "display" quote is NOT always the book. Live check (`polymarket_us.py cmp`, 2026-10-22 ~08Z):
  with an EMPTY YES-bid side the display showed yes_bid 0.01, 0.02 (above the 0.01 ask -> crossed) and 0.09 (real book:
  no bid at all); with a 0.01 ask the display showed 0.06. In DEV rows 140/3,083 display quotes are locked/crossed
  (bid >= ask), impossible on a real book. Change: drop locked/crossed display quotes (yes_bid >= yes_ask) as artifacts.
  The undetectable case (empty bid shown as a plausible bid) REMAINS and biases the backtest UP (it buys NO against a
  bid that did not exist). So P1 on display quotes is indicative only; the clean test is `paper tick` on the live book.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import requests

import polymarket_us as pm
from sports import http, wx
from sports.http import fetch
from wx_expand import MODELS, model_ext  # same forecast builder as the X1 replication

OUT = ROOT / "data" / "venues_weather.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
LOCK = ROOT / "lab" / "lockbox.jsonl"
PAPER = ROOT / "data" / "venues_weather_paper.jsonl"
DATASET = "pmus_temp"
SPLIT = pd.Timestamp("2026-08-15")
FIRST = date(2026, 4, 25)  # probe start; first event found 2026-04-29
PM_FEE = 0.0695
MARGIN = 3.0
KALSHI = {"nychigh": "KXHIGHNY", "mdwhigh": "KXHIGHCHI", "miahigh": "KXHIGHMIA", "laxhigh": "KXHIGHLAX",
          "phlhigh": "KXHIGHPHIL", "sfohigh": "KXHIGHTSFO"}
H = {"User-Agent": "Mozilla/5.0"}


def pm_fee(p):
    return PM_FEE * p * (1 - p)


def decision_ts(day: date, tz: str) -> int:
    return int((datetime(day.year, day.month, day.day, 22, tzinfo=ZoneInfo(tz)) - timedelta(days=1)).timestamp())


def pm_quote(slug: str, ts: int, cache: bool) -> dict | None:
    d = fetch(f"{pm.GATEWAY}/price-history", {"symbol": slug, "timestamp.startTimestamp": ts - 3600,
                                             "timestamp.endTimestamp": ts, "fidelity": 1}, cache=cache)
    h = [x for x in d.get("history", []) if x["timestamp"] <= ts]
    if not h:
        return None
    x = h[-1]
    return {"pm_ts": x["timestamp"], "yes_ask": float(x["longPrice"]), "yes_bid": round(1 - float(x["shortPrice"]), 4)}


def kalshi_rows(series: str, tz: str, d0: date, d1: date) -> pd.DataFrame:
    mk = wx.settled_markets(series)
    if mk.empty:
        return mk
    mk["date"] = mk["event_ticker"].map(wx.event_date)
    mk = mk[(mk.date >= pd.Timestamp(d0)) & (mk.date <= pd.Timestamp(d1))]
    # Kalshi quotes at the same 22:00 decision: one event-candlestick call per day (cached once fetched). The candle
    # service answered 503 for every request for a stretch on 2026-10-22, so a failed day just gets NaN quotes
    # (the settlement value still comes from the markets endpoint); re-running `build` fills them from cache + retries.
    rows, fails = [], 0
    for ev in sorted(mk.event_ticker.unique()):
        t = wx.decision_ts(wx.event_date(ev), tz)
        p = {"start_ts": t - 3 * 3600, "end_ts": t, "period_interval": 60}
        url = f"{wx.API}/series/{series}/events/{ev}/candlesticks"
        if fails >= 5 and not (http.CACHE_DIR / (http._key(url, p) + ".json")).exists():
            continue  # service down: only use what is cached
        try:
            d = fetch(url, p, retries=2)
            fails = 0
        except requests.HTTPError:
            fails += 1
            continue
        for tk, candles in zip(d.get("market_tickers", []), d.get("market_candlesticks", [])):
            r = wx._row(tk, ev, candles, t)
            if r:
                rows.append(r)
    px = pd.DataFrame(rows, columns=["ticker", "event_ticker", "price_ts", "yes_ask", "yes_bid", "volume_to_decision"])
    print(f"  kalshi {series}: quotes for {px.event_ticker.nunique()}/{mk.event_ticker.nunique()} days", flush=True)
    df = mk.merge(px, on=["ticker", "event_ticker"], how="left")
    return df[["date", "strike_type", "floor_strike", "cap_strike", "yes_bid", "yes_ask", "result", "expiration_value"]]


def cmd_build() -> int:
    http.MIN_INTERVAL = 0.07  # gateway documents 20-25 req/s; Kalshi/Open-Meteo calls here are few and cached
    last = datetime.now(timezone.utc).date() - timedelta(days=2)  # settled by 8-11 am ET next day
    frames = []
    for city, (station, tz) in pm.TEMP_HIGH.items():
        rows, d = [], FIRST
        while d <= last:
            ev = fetch(f"{pm.GATEWAY}/events/slug/temp-{city}-{d.isoformat()}", cache=True) if _exists(city, d) else None
            for m in (ev or {}).get("event", ev or {}).get("markets", []) if ev else []:
                br = pm.parse_bracket(m["slug"])
                res = pm.winner(m)
                if br is None or res is None:
                    continue
                q = pm_quote(m["slug"], decision_ts(d, tz), cache=True)
                rows.append({"city": city, "station": station, "date": pd.Timestamp(d), "slug": m["slug"],
                             "strike_type": br[0], "floor_strike": br[1], "cap_strike": br[2], "result": res,
                             **(q or {"pm_ts": None, "yes_ask": np.nan, "yes_bid": np.nan})})
            d += timedelta(days=1)
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"{city}: no settled events")
            continue
        p = fetch(f"https://api.weather.gov/stations/{station}")
        lon, lat = p["geometry"]["coordinates"][:2]
        for mdl, start in MODELS.items():
            f = model_ext(mdl, lat, lon, tz, max(start, str(df.date.min().date())), str(df.date.max().date()), low=False)
            if f is not None:
                df = df.join(f, on="date")
        ks = kalshi_rows(KALSHI[city], tz, df.date.min().date(), df.date.max().date())
        ks = ks.rename(columns={"yes_bid": "k_yes_bid", "yes_ask": "k_yes_ask", "result": "k_result", "expiration_value": "k_value"})
        df = df.merge(ks, on=["date", "strike_type", "floor_strike", "cap_strike"], how="left")
        # day-level settlement cross-check: Kalshi official value vs Polymarket winning bracket
        kv = df.groupby("date")["k_value"].max()
        df["k_day_value"] = df["date"].map(kv)
        v = df["k_day_value"]
        implied = np.select([df.strike_type == "greater", df.strike_type == "less", df.strike_type == "between"],
                            [v > df.floor_strike, v < df.cap_strike, (v >= df.floor_strike) & (v <= df.cap_strike)], default=np.nan)
        implied = np.where(v.isna(), np.nan, implied.astype(float))
        df["settle_mismatch"] = (~np.isnan(implied)) & (implied != (df.result == "yes").astype(float))
        n_days = df.date.nunique()
        print(f"{city} {station}: {len(df)} brackets, {n_days} days {df.date.min().date()}..{df.date.max().date()}, "
              f"quoted {df.yes_ask.notna().sum()}, kalshi-matched {df.k_yes_ask.notna().sum()}, "
              f"settle mismatches vs Kalshi value {int(df.settle_mismatch.sum())} on "
              f"{df[df.settle_mismatch].date.nunique()} days, days with !=1 winner "
              f"{int((df.groupby('date').result.apply(lambda s: (s == 'yes').sum()) != 1).sum())}", flush=True)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out)} rows)")
    return 0


_EXISTS: dict = {}


def _exists(city: str, d: date) -> bool:
    """Missing events 404; fetch() would raise, so probe once (cached in-process)."""
    k = (city, d)
    if k not in _EXISTS:
        try:
            fetch(f"{pm.GATEWAY}/events/slug/temp-{city}-{d.isoformat()}", cache=True)
            _EXISTS[k] = True
        except requests.HTTPError:
            _EXISTS[k] = False
    return _EXISTS[k]


def rejected(st, fl, ca, hi, lo):
    return np.where(st == "greater", hi + MARGIN < fl + 1,
           np.where(st == "less", lo - MARGIN > ca - 1, (hi + MARGIN < fl) | (lo - MARGIN > ca)))


def picks(df: pd.DataFrame, bid="yes_bid", ask="yes_ask", fee=pm_fee) -> pd.DataFrame:
    d = df[(df[bid] > 0) & (df[ask] < 1) & ((df[ask] - df[bid]) <= 0.10) & (df[bid] < df[ask])].copy()  # amendment: no locked/crossed
    cols = [c for c in d.columns if c.startswith("fcsafe_")]
    n = d[cols].notna().sum(1)
    rej = rejected(d.strike_type.to_numpy(), d.floor_strike.to_numpy(float), d.cap_strike.to_numpy(float),
                   d[cols].max(1).to_numpy(float), d[cols].min(1).to_numpy(float))
    s = d[rej & (n >= 3) & (d[ask] > 0.01)].copy()
    no = 1 - s[bid]
    cost = no + fee(no)
    s["no_price"], s["stake"] = no, 1.0
    s["pnl"] = np.where(s.result == "no", 1 / cost - 1, -1.0)
    return s


def score(t: pd.DataFrame, rng) -> dict:
    if t.empty:
        return {"bets": 0, "staked": 0.0, "units": 0.0, "roi": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p": 1.0,
                "folds_positive": 0, "per_fold": {}, "losses": 0}
    t = t.assign(day=t.city + t.date.astype(str))
    pm_ = t.groupby("day")["pnl"].sum().to_numpy()
    boots = np.array([rng.choice(pm_, len(pm_)).sum() for _ in range(5000)])
    per = t.groupby(t.date.dt.strftime("%Y-%m"))["pnl"].sum().round(2)
    return {"bets": int(len(t)), "staked": float(len(t)), "units": float(t.pnl.sum()), "roi": float(t.pnl.mean()),
            "ci_lo": float(np.quantile(boots, .025)), "ci_hi": float(np.quantile(boots, .975)), "p": float(np.mean(boots <= 0)),
            "folds_positive": int((per > 0).sum()), "per_fold": per.to_dict(), "losses": int((t.pnl < 0).sum())}


def k_fee(p):
    return wx.FEE_RATE * p * (1 - p)


def arb_table(df: pd.DataFrame) -> pd.DataFrame:
    """Both directions of YES-on-one + NO-on-other at the 22:00 quotes, incl. both venues' taker fees."""
    d = df.dropna(subset=["yes_bid", "yes_ask", "k_yes_bid", "k_yes_ask"]).copy()
    d = d[(d.yes_bid > 0) & (d.k_yes_bid > 0) & (d.yes_ask < 1) & (d.k_yes_ask < 1)]
    pno, kno = 1 - d.yes_bid, 1 - d.k_yes_bid
    d["cost_kyes_pno"] = d.k_yes_ask + k_fee(d.k_yes_ask) + pno + pm_fee(pno)
    d["cost_pyes_kno"] = d.yes_ask + pm_fee(d.yes_ask) + kno + k_fee(kno)
    d["best_cost"] = d[["cost_kyes_pno", "cost_pyes_kno"]].min(1)
    d["mid_gap"] = ((d.yes_bid + d.yes_ask) - (d.k_yes_bid + d.k_yes_ask)) / 2
    return d


def cmd_dev() -> int:
    df = pd.read_parquet(OUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df.date < SPLIT]  # sealed rows never enter dev
    print(f"DEV {df.city.nunique()} cities, {df.groupby('city').date.nunique().sum()} station-days, "
          f"{df.date.min().date()}..{df.date.max().date()}; quoted brackets {df.yes_ask.notna().sum()}/{len(df)}; "
          f"settlement mismatches vs Kalshi {int(df.settle_mismatch.sum())}")
    t = picks(df[~df.settle_mismatch])
    r = score(t, np.random.default_rng(0))
    months = max(1, (t.date.max() - t.date.min()).days / 30.4) if len(t) else 1
    print(f"P1 polymarket consensus-reject: bets {r['bets']} (~{r['bets'] / months:.0f}/mo) losses {r['losses']} "
          f"units {r['units']:+.2f} ROI {r['roi']:+.2%} 95%CI [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] p {r['p']:.4f} "
          f"months+ {r['folds_positive']}/{len(r['per_fold'])} avg NO {t.no_price.mean() if len(t) else float('nan'):.3f}")
    print("   by city:", {k: f"{v['sum']:+.2f}/{int(v['size'])}" for k, v in t.groupby("city").pnl.agg(['size', 'sum']).round(2).iterrows()})
    print("   by month:", r["per_fold"])
    # Same days on Kalshi, same rule, Kalshi fee: does Polymarket add bets the Kalshi rule doesn't already take?
    k = df.dropna(subset=["k_yes_bid"]).rename(columns={"yes_bid": "p_yes_bid", "yes_ask": "p_yes_ask"})
    kt = picks(k, bid="k_yes_bid", ask="k_yes_ask", fee=k_fee)
    kt = kt.assign(result=kt.k_result)
    kt["pnl"] = np.where(kt.result == "no", 1 / (kt.no_price + k_fee(kt.no_price)) - 1, -1.0)
    both = t.merge(kt[["city", "date", "strike_type", "floor_strike", "cap_strike"]], how="inner",
                   on=["city", "date", "strike_type", "floor_strike", "cap_strike"])
    print(f"   [context, not a trial] Kalshi same city-days: {len(kt)} picks, ROI {kt.pnl.mean() if len(kt) else 0:+.2%}; "
          f"overlap with Polymarket picks {len(both)} -> Polymarket-only bets {len(t) - len(both)}")
    a = arb_table(df)
    hits = a[a.best_cost < 1]
    print(f"X1 cross-venue (DEV, descriptive): {len(a)} matched quoted brackets; median |mid gap| "
          f"{a.mid_gap.abs().median():.3f}; after-fee cost < $1 on {len(hits)} "
          f"({len(hits) / max(len(a), 1):.1%}); median best cost {a.best_cost.median():.3f}; "
          f"edge if hit: median {(1 - hits.best_cost).median() if len(hits) else 0:.3f}, sum {(1 - hits.best_cost).sum():.2f} $/contract")
    if len(hits):
        print(hits.sort_values("best_cost")[["city", "date", "slug", "yes_bid", "yes_ask", "k_yes_bid", "k_yes_ask",
                                             "best_cost"]].head(12).to_string(index=False))
    sha = hashlib.sha256(b"venues_weather:P1").hexdigest()[:12]
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {x["sha"] for x in prior if x.get("dataset") == DATASET and x.get("status") == "ok"} | {sha}
    bar = 0.05 / len(shas)
    print(f"{DATASET} trials {len(shas)} -> bar p < {bar:.4f}; P1 dev {'PASS' if (r['p'] < bar and r['units'] > 0) else 'FAIL'}")
    if "--ledger" in sys.argv:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": "venues_weather_P1_pmus", "sha": sha,
               "dataset": DATASET, "hypothesis": "pre-registered in scripts/venues_weather.py (P1); rule frozen from "
               "kalshi_reject_safe_check, ported to Polymarket US quotes + fees", "source": "venue diversification (user 2026-10-22)",
               "status": "ok", "trial_number": len(shas), "passed": bool(r["p"] < bar and r["units"] > 0), **r}
        with LEDGER.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print("appended to ledger")
    return 0


def cmd_sealed() -> int:
    """ONE-TIME. Requires the user's explicit yes. Same rule/scorer on rows >= SPLIT. Refuses a re-score."""
    if "--confirm" not in sys.argv or "P1" not in sys.argv:
        print("sealed scoring is the user's decision; pass --confirm P1 only after an explicit yes")
        return 1
    sha = hashlib.sha256(b"venues_weather:P1").hexdigest()[:12]
    prior = [json.loads(l) for l in LOCK.read_text().splitlines() if l.strip()] if LOCK.exists() else []
    if any(x["sha"] == sha for x in prior):
        print("P1 already scored on the lockbox; refusing")
        return 1
    df = pd.read_parquet(OUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df.date >= SPLIT]
    t = picks(df[~df.settle_mismatch])
    r = score(t, np.random.default_rng(0))
    verdict = "PASS" if (r["p"] < 0.05 and r["units"] > 0) else "FAIL"
    with LOCK.open("a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": "venues_weather_P1_pmus",
                            "sha": sha, "dataset": DATASET, **r, "verdict": verdict}, default=str) + "\n")
    print(f"SEALED P1 {df.date.min().date()}..{df.date.max().date()}: bets {r['bets']} units {r['units']:+.2f} "
          f"ROI {r['roi']:+.2%} CI [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] p {r['p']:.4f} -> {verdict}")
    return 0


# ---------------------------------------------------------------- forward paper (no orders)
def cmd_paper() -> int:
    """`paper tick`: hourly; for each Polymarket US high city in its 22:00-23:59 local window, apply the frozen rule to
    tomorrow's brackets at the live book (no orders), log picks + depth; then settle. `paper report` prints the tally.
    Install (user runs it; nothing installed by this script), e.g. a launchd StartCalendarInterval Minute=7 job running:
      cd <worktree>/sports && ../.venv_quant/bin/python scripts/venues_weather.py paper tick
    (launchd cannot read ~/Desktop -- copy the code outside Desktop as the kalshi-paper agent does; see
    scripts/install_kalshi_paper_agent.sh for the pattern)."""
    import kalshi_paper as kp
    sub = sys.argv[2] if len(sys.argv) > 2 else "report"
    rows = [json.loads(l) for l in PAPER.read_text().splitlines()] if PAPER.exists() else []
    if sub in ("tick", "validate"):  # validate = today's open markets, any hour, logs nothing
        now = datetime.now(timezone.utc)
        seen = {r["slug"] for r in rows}
        new = []
        for city, (station, tz) in pm.TEMP_HIGH.items():
            local = now.astimezone(ZoneInfo(tz))
            if sub == "tick" and local.hour not in (22, 23):
                continue
            tomorrow = (local + timedelta(days=1 if sub == "tick" else 0)).date()
            ev = pm.temp_event(city, tomorrow)
            if not ev:
                continue
            fc = kp.live_forecasts(*wx.station_coords(station), tz, tomorrow)
            vals = [v for v in fc.values() if v is not None]
            if len(vals) < 3:
                continue
            for m in ev.get("markets", []):
                br = pm.parse_bracket(m["slug"])
                if br is None or m["slug"] in seen:
                    continue
                b = pm.book(m["slug"])
                if not b or not b["bids"] or not b["offers"]:
                    continue
                bid, ask = b["bids"][0][0], b["offers"][0][0]
                if not (bid > 0 and ask < 1 and ask - bid <= 0.10 and ask > 0.01):
                    continue
                if not kp.rejected(br[0], br[1], br[2], vals):
                    continue
                d1 = sum(q for p, q in b["bids"] if p >= bid - 0.0101)
                new.append({"logged_at": now.isoformat(timespec="seconds"), "city": city, "slug": m["slug"],
                            "strike": br, "forecasts": fc, "yes_bid": bid, "yes_ask": ask, "no_price": round(1 - bid, 4),
                            "fee": round(pm_fee(1 - bid), 4), "depth_1c": d1, "result": None})
        if sub == "validate":
            for r in new:
                print("VALIDATE", r["slug"], "NO @", r["no_price"], "depth", r["depth_1c"], r["forecasts"])
            print(f"{len(new)} picks (validation only, not logged)")
            return 0
        with PAPER.open("a") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
        print(f"{len(new)} new paper picks")
        rows += new
        for r in rows:
            if r["result"] is None:
                mk = pm.market(r["slug"])
                r["result"] = pm.winner(mk) if mk else None
        PAPER.write_text("".join(json.dumps(r) + "\n" for r in rows))
    done = [r for r in rows if r["result"]]
    u = sum((1 / (r["no_price"] + r["fee"]) - 1) if r["result"] == "no" else -1.0 for r in done)
    print(f"polymarket paper: {len(rows)} picks, {len(done)} settled, {sum(r['result'] == 'no' for r in done)} wins, {u:+.2f} units")
    return 0


# ---------------------------------------------------------------- live cross-venue snapshot
def cmd_xlive() -> int:
    """Today's and tomorrow's brackets on both venues, best executable both-leg cost incl. fees and size at the top level."""
    today = datetime.now(timezone.utc).date()
    out = []
    for city, (station, tz) in pm.TEMP_HIGH.items():
        for d in (today, today + timedelta(days=1)):
            ev = pm.temp_event(city, d)
            if not ev:
                continue
            kev = f"{KALSHI[city]}-{d.strftime('%y%b%d').upper()}"
            try:
                kms = fetch(f"{wx.API}/markets", {"event_ticker": kev, "limit": 100}, cache=False, retries=4).get("markets", [])
            except requests.HTTPError as e:
                print(f"kalshi {kev}: {e} -- skipped")
                kms = []
            kmap = {}
            for km in kms:
                fl = float(km["floor_strike"]) if km.get("floor_strike") is not None else np.nan
                ca = float(km["cap_strike"]) if km.get("cap_strike") is not None else np.nan
                kmap[(km.get("strike_type"), None if np.isnan(fl) else fl, None if np.isnan(ca) else ca)] = km
            for m in ev.get("markets", []):
                st, fl, ca = pm.parse_bracket(m["slug"]) or (None, np.nan, np.nan)
                km = kmap.get((st, None if np.isnan(fl) else fl, None if np.isnan(ca) else ca))
                b = pm.book(m["slug"])
                if not km or not b or not b["bids"] or not b["offers"]:
                    out.append({"city": city, "date": d, "slug": m["slug"], "matched": km is not None})
                    continue
                kb, ka = float(km.get("yes_bid_dollars") or 0), float(km.get("yes_ask_dollars") or 0)
                pb, pa = b["bids"][0][0], b["offers"][0][0]
                c1 = ka + k_fee(ka) + (1 - pb) + pm_fee(1 - pb) if kb > 0 and ka < 1 else np.nan
                c2 = pa + pm_fee(pa) + (1 - kb) + k_fee(1 - kb) if kb > 0 and ka < 1 else np.nan
                out.append({"city": city, "date": d, "slug": m["slug"], "matched": True, "k_bid": kb, "k_ask": ka,
                            "p_bid": pb, "p_ask": pa, "p_bid_qty": b["bids"][0][1], "p_ask_qty": b["offers"][0][1],
                            "best_cost": np.nanmin([c1, c2])})
    t = pd.DataFrame(out)
    print(t.to_string(index=False, max_rows=200))
    m = t[t.matched & t.best_cost.notna()] if "best_cost" in t else t.iloc[:0]
    print(f"{len(t)} Polymarket brackets, {int(t.matched.sum())} matched to an identical Kalshi bracket, "
          f"after-fee cost < $1 on {int((m.best_cost < 1).sum())}; median best cost {m.best_cost.median():.3f}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "dev": cmd_dev, "sealed": cmd_sealed, "paper": cmd_paper,
                      "xlive": cmd_xlive}.get(cmd, lambda: print(__doc__) or 1)())
