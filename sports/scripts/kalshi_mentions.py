"""Kalshi MENTION markets: do buyers overpay for YES? (registry leads L10 / L23)

    python scripts/kalshi_mentions.py markets    # all settled markets of every 'Mentions' series -> data/kalshi_mentions_markets.parquet
    python scripts/kalshi_mentions.py describe   # counts / liquidity per month, NO outcomes printed
    python scripts/kalshi_mentions.py prices     # hourly candles -> leak-free decision quotes -> data/kalshi_mentions_prices.parquet
    python scripts/kalshi_mentions.py dev [--ledger]   # the pre-registered trials, DEV only
    python scripts/kalshi_mentions.py sealed --confirm M?  # ONE-TIME, user's yes required

Data (2026-10-22): the 464 series with category 'Mentions' in the full /series list (data/earnings/series.json,
2026-10-21); every settled market from /historical/markets + /markets?status=settled: 83,960 markets, 4,165 events.
All 464 series report fee_type 'quadratic', fee_multiplier 1 (series API) -> the general taker fee.

PRE-REGISTERED 2026-10-22, written after the outcome-free describe (counts, volume, rule text, timing fields) and
BEFORE any decision-time price or P&L was computed:

Universe (per market): binary; series not KXMVE* (multivariate combos); rules_primary is a word-mention contract
  (says/say/said/"is stated by"/mention) and NOT a window/duration contract (regex WINDOW below: "stated by ... before",
  "after ... and before", "late to", "speak(s) for", "minutes", template "||date"). Window markets are excluded because
  their decision time cannot be placed before the observation period.
Event timing (no scheduled start time exists in the API): A = min close_time over the event's markets (markets close
  at or after the end of the event; some close hours later after review). V = end of the hour with the most event-level
  traded volume in [A-96h, A] (in-event trading frenzy; outcome-free). anchor = min(A, V).
  D24 = anchor - 24h ; D6 = anchor - 6h. Quote = yes_bid / yes_ask close of the last hourly candle ending <= D.
  D6 is leak-free only if the busiest hour falls < ~5 h after the event starts (true for calls, pressers, rallies,
  games, shows). Guard (price-only, outcome-free): drop an EVENT at a decision time if >= 50% of its quoted markets
  already sit at an extreme (yes_ask <= 0.03 or yes_bid >= 0.97) -- that means the event is already over/underway.
Quote filter: yes_bid > 0, yes_ask < 1, yes_ask - yes_bid <= 0.10 (mention books are thinner than sports; set before
  looking at spreads). mid = (bid + ask) / 2.
Execution: taker, buy NO at the NO ask = 1 - yes_bid, 1 unit = 1 contract.
Fee: Kalshi general taker fee = ceil_to_cent(0.07 * C * P * (1 - P)) per order (P = price paid). Primary: orders of
  C = 10 contracts: fee per contract = ceil(100 * 0.07 * 10 * P * (1 - P)) / 100 / 10 dollars (see fee()). Robustness
  (not trials): C = 1 (min 1c per contract) and C = 100.
Trials (6 -> Bonferroni bar p < 0.05/6 = 0.0083, one-sided, bootstrap 3000x over EVENTS):
  M1_d24  buy NO on every eligible market at D24
  M2_d24  buy NO when YES mid in [0.15, 0.60] at D24
  M3_d24  buy NO when YES mid <= 0.15 at D24
  M1_d6, M2_d6, M3_d6  the same at D6
Split: event anchor (ET) < 2025-10-01 = DEV (~225 events); >= 2025-10-01 = SEALED (not priced, not loaded by `dev`).
Pass on dev = p < bar AND units > 0 AND positive in both halves of dev (by event date). A passing rule becomes a
  SEALED ASK; sealed is scored once, on the user's explicit yes.
Reported, not trials: ROI by group (earnings / fed / trump / sports / other), C=1 and C=100 fees, calibration table.

DEV RESULT 2026-10-22 (scored once, ledger): M3_d24 +4.32% / 293 bets / 79 events p<.0003 PASS; M3_d6 +2.99% / 394 /
106 p .0027 PASS; M1/M2 fail (M2_d6 +5.5% p .018 misses the bar). Sealed NOT scored -- see notes/kalshi_mentions.md.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sports.http import fetch, fetch_many

API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_JSON = ROOT / "data" / "earnings" / "series.json"   # full /series list pulled 2026-10-21
MK = ROOT / "data" / "kalshi_mentions_markets.parquet"
PX = ROOT / "data" / "kalshi_mentions_prices.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
LOCKBOX_LOG = ROOT / "lab" / "lockbox.jsonl"
DATASET = "kalshi_mentions"
SPLIT = pd.Timestamp("2025-10-01", tz="America/New_York")


def mention_series() -> list[str]:
    s = json.load(open(SERIES_JSON))["series"]
    return sorted({x["ticker"] for x in s if x.get("category") == "Mentions"})


def _pull(series: str) -> list[dict]:
    rows = []
    for path, extra, cache in ((f"{API}/historical/markets", {}, True), (f"{API}/markets", {"status": "settled"}, False)):
        cur = None
        while True:
            p = {"series_ticker": series, "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            try:
                d = fetch(path, p, cache=cache)
            except Exception as e:  # noqa: BLE001
                print("ERR", series, e, flush=True)
                break
            rows += d.get("markets", [])
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    for r in rows:
        r["series"] = series
    return rows


def cmd_markets() -> int:
    ser = mention_series()
    with ThreadPoolExecutor(4) as ex:
        rows = [r for rs in ex.map(_pull, ser) for r in rs]
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    keep = ["series", "ticker", "event_ticker", "title", "yes_sub_title", "result", "open_time", "close_time",
            "expected_expiration_time", "expiration_time", "occurrence_datetime", "settlement_ts", "volume_fp",
            "status", "market_type", "strike_type", "rules_primary", "can_close_early"]
    df = df[[c for c in keep if c in df]]
    df.to_parquet(MK, index=False)
    print(f"{len(ser)} series, {len(df)} markets, {df.event_ticker.nunique()} events -> {MK}")
    return 0


WORD = r"(?i)\bsays?\b|is stated by|mention|\bsaid\b"
WINDOW = r"(?i)is stated by .*\bbefore\b|\bafter\b.*\band before\b|late to|speaks? for|\bminutes\b|\|\|date"
TRIALS = ["M1_d24", "M2_d24", "M3_d24", "M1_d6", "M2_d6", "M3_d6"]
FEE = 0.07


def eligible(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rules_primary.fillna("")
    keep = (df.market_type == "binary") & ~df.series.str.startswith("KXMVE") & r.str.contains(WORD) & ~r.str.contains(WINDOW)
    keep &= ~df.series.str.contains("DURATION")
    return df[keep].copy()


def group(series: str, rule: str) -> str:
    r = rule.lower()
    if "EARN" in series or "earnings call" in r:
        return "earnings"
    if any(k in series for k in ("FED", "ECB", "POWELL", "WALLER", "BOWMAN", "MIRAN", "WARSH")) or "fomc" in r or "federal reserve" in r:
        return "fed"
    if "TRUMP" in series or "DJT" in series or "donald trump" in r:
        return "trump"
    if "commentator" in r or "announcer" in r:
        return "sports"
    return "other"


def load_markets() -> pd.DataFrame:
    df = eligible(pd.read_parquet(MK))
    for c in ("open_time", "close_time"):
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    df["vol"] = pd.to_numeric(df.volume_fp, errors="coerce")
    df["A"] = df.groupby("event_ticker").close_time.transform("min")
    df["group"] = [group(s, r or "") for s, r in zip(df.series, df.rules_primary)]
    return df


def cmd_describe() -> int:
    """Outcome-free: counts, volume, group mix per month (by A, ET)."""
    df = load_markets()
    df["mo"] = df.A.dt.tz_convert("America/New_York").dt.strftime("%Y-%m")
    t = df.groupby("mo").agg(events=("event_ticker", "nunique"), markets=("ticker", "size"),
                             vol_M=("vol", lambda x: x.sum() / 1e6), med_mkt_vol=("vol", "median"))
    print(t.round(2).to_string())
    print(df.groupby("group").agg(events=("event_ticker", "nunique"), markets=("ticker", "size"),
                                  vol_M=("vol", lambda x: x.sum() / 1e6)).round(1).to_string())
    dev = df.A < SPLIT
    print(f"eligible: {len(df)} markets / {df.event_ticker.nunique()} events; DEV(A<split) {dev.sum()} markets / "
          f"{df[dev].event_ticker.nunique()} events")
    return 0


def _q(c, side):
    v = c.get(side) or {}
    return pd.to_numeric(v.get("close_dollars", v.get("close")), errors="coerce")


def _vol(c):
    v = pd.to_numeric(c.get("volume_fp", c.get("volume")), errors="coerce")
    return 0.0 if pd.isna(v) else float(v)


def cmd_prices() -> int:
    """Hourly candles for DEV events only (sealed with --sealed; prices only, never scored without --confirm)."""
    df = load_markets()
    sealed = "--sealed" in sys.argv
    df = df[(df.A >= SPLIT) if sealed else (df.A < SPLIT)].copy()
    df["A_ts"] = (df.A - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    urls = [f"{API}/historical/markets/{tk}/candlesticks?start_ts={a - 96 * 3600}&end_ts={a}&period_interval=60"
            for tk, a in zip(df.ticker, df.A_ts)]
    print(f"fetching {len(urls)} market candle series", flush=True)
    res = fetch_many(urls)
    live = [i for i, d in enumerate(res) if not d.get("candlesticks")]
    if live:  # settled after the historical cutoff -> live endpoint
        lu = [f"{API}/series/{df.series.iloc[i]}/markets/{df.ticker.iloc[i]}/candlesticks?start_ts={df.A_ts.iloc[i] - 96 * 3600}"
              f"&end_ts={df.A_ts.iloc[i]}&period_interval=60" for i in live]
        def safe(u):
            try:
                return fetch(u, retries=3)
            except Exception:  # noqa: BLE001  (404 = no live candles either; market stays unpriced)
                return {}
        with ThreadPoolExecutor(4) as ex:
            for i, d in zip(live, ex.map(safe, lu)):
                res[i] = d
        print(f"{len(live)} markets had no historical candles; live endpoint filled "
              f"{sum(bool(res[i].get('candlesticks')) for i in live)}", flush=True)
    cand = {tk: d.get("candlesticks", []) for tk, d in zip(df.ticker, res)}
    # event busiest hour V (outcome-free)
    vols: dict[str, dict[int, float]] = {}
    for tk, ev in zip(df.ticker, df.event_ticker):
        h = vols.setdefault(ev, {})
        for c in cand[tk]:
            h[c["end_period_ts"]] = h.get(c["end_period_ts"], 0.0) + _vol(c)
    V = {ev: max(h, key=h.get) for ev, h in vols.items() if h and max(h.values()) > 0}
    df["V_ts"] = df.event_ticker.map(V)
    df["anchor_ts"] = np.fmin(df.A_ts.astype(float), df.V_ts.astype(float))
    for tag, hrs in (("d24", 24), ("d6", 6)):
        t = df.anchor_ts - hrs * 3600
        bid, ask = [], []
        for tk, tt in zip(df.ticker, t):
            cs = [c for c in cand[tk] if c["end_period_ts"] <= tt] if np.isfinite(tt) else []
            bid.append(_q(cs[-1], "yes_bid") if cs else np.nan)
            ask.append(_q(cs[-1], "yes_ask") if cs else np.nan)
        df[f"bid_{tag}"], df[f"ask_{tag}"] = bid, ask
    out = PX.with_name(PX.stem + "_sealed.parquet") if sealed else PX
    df.drop(columns=["rules_primary"]).to_parquet(out, index=False)
    for tag in ("d24", "d6"):
        print(f"{tag}: candle at decision for {df[f'bid_{tag}'].notna().sum()}/{len(df)} markets")
    print(f"-> {out}")
    return 0


def fee(p, c=10):
    return np.ceil(np.round(100 * FEE * c * p * (1 - p), 9)) / 100 / c


def quotes(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    d = df.rename(columns={f"bid_{tag}": "bid", f"ask_{tag}": "ask"})
    d = d[d.bid.notna() & d.ask.notna()].copy()
    ext = (d.ask <= 0.03) | (d.bid >= 0.97)
    over = ext.groupby(d.event_ticker).transform("mean") >= 0.5   # event already underway/over -> drop
    d = d[~over]
    d = d[(d.bid > 0) & (d.ask < 1) & (d.ask - d.bid <= 0.10)].copy()
    d["mid"] = (d.bid + d.ask) / 2
    return d


def trial_frame(df: pd.DataFrame, name: str, c=10) -> pd.DataFrame:
    rule, tag = name.split("_")
    d = quotes(df, tag)
    sel = {"M1": d.mid >= 0, "M2": (d.mid >= 0.15) & (d.mid <= 0.60), "M3": d.mid <= 0.15}[rule]
    d = d[sel]
    cost = (1 - d.bid).to_numpy()
    stake = cost + fee(cost, c)
    won = (d.result == "no").to_numpy()
    return pd.DataFrame({"event": d.event_ticker.to_numpy(), "group": d.group.to_numpy(), "series": d.series.to_numpy(),
                         "date": d.A.dt.tz_convert("America/New_York").dt.tz_localize(None).to_numpy(),
                         "pnl": np.where(won, 1 - stake, -stake), "stake": stake})


def score(t: pd.DataFrame, rng) -> dict:
    if t.empty:
        return {"bets": 0, "events": 0, "units": 0.0, "roi": 0.0, "p": 1.0}
    ev = t.groupby("event").agg(pnl=("pnl", "sum"), stake=("stake", "sum"), date=("date", "min"))
    idx = rng.integers(0, len(ev), size=(3000, len(ev)))
    boots = ev.pnl.to_numpy()[idx].sum(1)
    med = ev.date.median()
    h1, h2 = ev[ev.date < med].pnl.sum(), ev[ev.date >= med].pnl.sum()
    return {"bets": int(len(t)), "events": int(len(ev)), "staked": float(t.stake.sum()), "units": float(t.pnl.sum()),
            "roi": float(t.pnl.sum() / t.stake.sum()), "ci_lo": float(np.quantile(boots, 0.025)),
            "ci_hi": float(np.quantile(boots, 0.975)), "p": float(np.mean(boots <= 0)),
            "half1_units": float(h1), "half2_units": float(h2)}


def calibration(df: pd.DataFrame, tag: str):
    d = quotes(df, tag)
    d["b"] = pd.cut(d.mid, [0, .05, .10, .15, .25, .40, .60, .80, .95, 1.0])
    d["y"] = (d.result == "yes").astype(float)
    print(f"calibration {tag}: {'bucket':12s} {'n':>5s} {'ev':>4s} {'mid':>6s} {'hit':>6s} {'hit-mid':>8s}")
    for b, g in d.groupby("b", observed=True):
        print(f"                 {str(b):12s} {len(g):5d} {g.event_ticker.nunique():4d} {g.mid.mean():6.3f} {g.y.mean():6.3f} "
              f"{g.y.mean() - g.mid.mean():+8.3f}")


def cmd_dev() -> int:
    import hashlib
    from datetime import datetime, timezone
    df = pd.read_parquet(PX)
    df = df[df.A < SPLIT]  # sealed rows are never priced into PX; belt and braces
    print(f"DEV: {len(df)} markets, {df.event_ticker.nunique()} events, {df.A.min():%Y-%m-%d}..{df.A.max():%Y-%m-%d}")
    for tag in ("d24", "d6"):
        qq = quotes(df, tag)
        print(f"{tag}: candle {df[f'bid_{tag}'].notna().sum()}, after guard+spread filter {len(qq)} markets / "
              f"{qq.event_ticker.nunique()} events")
        calibration(df, tag)
    bar = 0.05 / len(TRIALS)
    recs = []
    for name in TRIALS:
        t = trial_frame(df, name)
        res = score(t, np.random.default_rng(0))
        passed = bool(res["p"] < bar and res["units"] > 0 and res.get("half1_units", 0) > 0 and res.get("half2_units", 0) > 0)
        rob = {}
        for c in (1, 100):
            x = trial_frame(df, name, c)
            rob[f"roi_c{c}"] = float(x.pnl.sum() / x.stake.sum()) if len(x) else 0.0
        by = {g: f"{x.pnl.sum() / x.stake.sum():+.1%}/{len(x)}/{x.event.nunique()}ev" for g, x in t.groupby("group")}
        print(f"{name:7s} bets {res['bets']:5d} ev {res['events']:4d} units {res['units']:+7.2f} ROI {res['roi']:+6.2%} "
              f"CI [{res.get('ci_lo', 0):+.1f},{res.get('ci_hi', 0):+.1f}] p {res['p']:.4f} halves "
              f"{res.get('half1_units', 0):+.1f}/{res.get('half2_units', 0):+.1f}  C1 {rob['roi_c1']:+.2%} "
              f"C100 {rob['roi_c100']:+.2%} {'PASS' if passed else 'fail'}")
        print(f"        by group: {by}")
        sha = hashlib.sha256(("kalshi_mentions:" + name).encode()).hexdigest()[:12]
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"kalshi_mentions_{name}",
                     "sha": sha, "dataset": DATASET, "hypothesis": f"pre-registered in scripts/kalshi_mentions.py ({name})",
                     "source": "L10/L23 mention markets buy NO (Deleep et al. SSRN 6322678 via Quantpedia)",
                     "status": "ok", "study_trials": len(TRIALS), "bonferroni_bar": bar, **res, **rob,
                     "by_group": by, "passed": passed})
    print(f"bar p < {bar:.4f}")
    if "--ledger" in sys.argv:
        n = sum(1 for l in LEDGER.read_text().splitlines() if l.strip())
        with LEDGER.open("a") as f:
            for i, r in enumerate(recs):
                f.write(json.dumps({**r, "trial_number": n + i + 1}, default=str) + "\n")
        print(f"appended {len(recs)} trials to {LEDGER}")
    return 0


def cmd_sealed() -> int:
    if "--confirm" not in sys.argv:
        print("sealed scoring is the user's decision; pass --confirm and the trial name(s)")
        return 1
    names = [a for a in sys.argv[2:] if a in TRIALS]
    if not names:
        print(f"name the promoted trial(s): {TRIALS}")
        return 1
    import hashlib
    from datetime import datetime, timezone
    prior = [json.loads(l) for l in LOCKBOX_LOG.read_text().splitlines() if l.strip()] if LOCKBOX_LOG.exists() else []
    df = pd.read_parquet(PX.with_name(PX.stem + "_sealed.parquet"))
    df = df[df.A >= SPLIT]
    n_prom = sum(r["dataset"] == DATASET for r in prior) + len(names)
    for name in names:
        sha = hashlib.sha256(("kalshi_mentions:" + name).encode()).hexdigest()[:12]
        if any(r["sha"] == sha for r in prior):
            print(f"{name} already scored on the lockbox; refusing")
            continue
        res = score(trial_frame(df, name), np.random.default_rng(0))
        bar = 0.05 / n_prom
        with LOCKBOX_LOG.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"kalshi_mentions_{name}",
                                "sha": sha, "dataset": DATASET, "promotion_number": n_prom, **res}, default=str) + "\n")
        print(f"SEALED {name}: {res} -> {'PASS' if res['p'] < bar and res['units'] > 0 else 'FAIL'} (bar {bar:.4f})")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"markets": cmd_markets, "describe": cmd_describe, "prices": cmd_prices, "dev": cmd_dev,
          "sealed": cmd_sealed}.get(cmd)
    raise SystemExit(fn() if fn else (print(__doc__) or 1))
