"""Consensus-reject on Kalshi ECONOMIC-DATA ladders (weather E1/E2/E5 mechanism transferred). DEV only, no orders.

    python scripts/econ_reject.py build      # markets + public anchors + DEV-event candles (sealed events never priced)
    python scripts/econ_reject.py dev [--ledger]

Why this might work / why it might not (read before trusting anything):
  Weather consensus-reject pays because temp ladders span the whole outcome range and the far tails stay quoted at
  2-5c YES after every forecast rules them out. Kalshi CPI (kalshi_cpi.py, N2) and daily gas (kalshi_gas.py, N6) died
  with ZERO bets: Kalshi econ ladders only list/quote rungs inside ~3 sd of the outcome; rungs outside the nowcast band
  had yes_bid 0. Diercks et al. (FEDS 2026-010): Kalshi beats Bloomberg consensus on the CPI centre -> we test TAILS.
  This script tests whether the same holds for claims / GDP / U3 / payrolls / core PCE, with k=2 added because k=3
  produced nothing on CPI.

PRE-REGISTERED 2026-10-28, written before any Kalshi price for these series was pulled (only market metadata --
tickers, strikes, close times, results -- was listed to size the universe and set the date split):

Series and anchors (all public, all available before the decision time unless flagged):
  KXJOBLESSCLAIMS (2025-06+, ladder, YES iff print >= strike) and KXJOBLESS (2021-07..2022-12, single "Above X" rung):
      anchor = FRED ICSA (SA) for the week ending S-7, S = target week-ending Saturday (new tickers carry the release
      Thursday R -> S = R-5; old tickers carry S). That week's value was released the Thursday before the decision.
      Caveat: FRED current vintage, so the anchor includes the small revision published WITH the target release.
      sd = std of weekly ICSA change, weeks 2015-01..2019-12 (pre-Kalshi, pre-COVID).
  KXU3 (U3-/KXU3-, monthly): anchor = FRED UNRATE for month M-1 (current vintage; only annual seasonal revisions).
      sd = std of monthly UNRATE change 2011-01..2019-12.
  KXPAYROLLS (PROLLS-/KXPAYROLLS-, strike = monthly NFP change): anchor = the previous month's Kalshi
      expiration_value (= that month's first print; leak-free). sd = std of month-to-month change of the PAYEMS
      monthly change, 2011-01..2019-12.
  KXGDP (advance real GDP q/q SAAR): anchor = Atlanta Fed GDPNow "Model Forecast Right Before BEA's Advance
      Estimate" (TrackRecord sheet), matched to the event by release date (+-3 days). sd = RMSE of GDPNow error for
      quarters 2011Q3..2019Q4. Caveat: the final GDPNow is sometimes published the release-eve; decision is close-24h.
  KXPCECORE (core PCE m/m): anchor = Cleveland Fed Core PCE nowcast (data/cleveland_nowcast.parquet, month kind),
      latest vintage dated <= decision date - 1 day. sd = std(first print - nowcast >= 3 days before release) on target
      months <= 2021-07 (as kalshi_cpi.band_sd).
  Excluded: KXPCEHEAD (1 event), KXCONTCLAIMS (9 events, median volume 25), consensus (Bloomberg) -- not public.

Rule (identical across series): band = [anchor - k*sd, anchor + k*sd].
  strike > anchor + k*sd  -> buy NO  at NO ask = 1 - yes_bid  (needs yes_bid >= 0.01)
  strike < anchor - k*sd  -> buy YES at yes_ask               (needs 0 < yes_ask <= 0.99)
  Decision = close_time - 24 h (the day before release); quote = last hourly candle ending at or before it.
  Fee (taker, exact): ceil(0.07 * C * p * (1-p) * 100) / 100 dollars for a C = 50 contract order, per contract.
  P&L per $1 staked: win -> 1/cost - 1, loss -> -1.
  MAKER variant (descriptive, NOT a trial): NO bought at the NO bid = 1 - yes_ask (needs yes_ask > yes_bid), fee
      ceil(0.0175*C*p*(1-p)*100)/100, ASSUMED FILLED -> an upper bound only.

Trials (2, pooled across all series): T1 k=3 (primary, same as CPI), T2 k=2. Bar p < 0.05/2 = 0.025, one-sided
  bootstrap over EVENTS (all rungs of one release = one draw). Report n, losses, ROI, 95% CI, both date halves of DEV,
  per-series breakdown.

DEV / SEALED split (per series, earliest ~70% of events by close date; SEALED never priced by this script):
  KXJOBLESS     close < 2022-07-06     KXJOBLESSCLAIMS close < 2026-06-25    KXU3      close < 2025-04-04
  KXPAYROLLS    close < 2025-10-03     KXGDP           close < 2025-04-30    KXPCECORE close < 2026-04-09
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports.http import fetch, fetch_many
from kalshi_longshot import API, _px

OUT = ROOT / "data" / "econ_reject_dev.parquet"
NOW = ROOT / "data" / "cleveland_nowcast.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "econ_reject"
DL = Path("/Users/jonathanwang/.claude/jobs/588c8b26/tmp/econ_dl")  # icsa.csv unrate.csv payems.csv gdpnow.xlsx
SERIES = ["KXJOBLESS", "KXJOBLESSCLAIMS", "KXU3", "KXPAYROLLS", "KXGDP", "KXPCECORE"]
CUT = {"KXJOBLESS": "2022-07-06", "KXJOBLESSCLAIMS": "2026-06-25", "KXU3": "2025-04-04",
       "KXPAYROLLS": "2025-10-03", "KXGDP": "2025-04-30", "KXPCECORE": "2026-04-09"}
TRIALS = {"T1_k3": 3.0, "T2_k2": 2.0}
C = 50
VERDICT = {  # written after the DEV run, see notes/econ_reject.md
    "T1_k3": "NOT AN EDGE: registered bootstrap p=0 is degenerate (0 losses); market-price null p .08; ROI driven by 6 "
             "mid-priced 2021-22 KXJOBLESS YES bets on a revised-vintage anchor; remaining 27 tail bets null p .48",
    "T2_k2": "FAIL: boot p .030 > bar .025; market-price null p .053; tail-only (cost>=.90) +0.9%/87",
}
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def fee(p, rate):
    return math.ceil(rate * C * p * (1 - p) * 100 - 1e-9) / 100 / C


def settled(series):
    rows = []
    for path, extra, cache in ((f"{API}/historical/markets", {}, True), (f"{API}/markets", {"status": "settled"}, False)):
        cur = None
        while True:
            p = {"series_ticker": series, "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            d = fetch(path, p, cache=cache)
            rows += d.get("markets", [])
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df = df[df.result.isin(["yes", "no"])].copy()
    df["series"] = series
    if "floor_strike" not in df:
        df["floor_strike"] = np.nan
    lab = df.yes_sub_title.fillna("").str.replace(",", "")
    num = pd.to_numeric(lab.str.extract(r"([-\d.]+)")[0], errors="coerce")
    df["strike"] = df.floor_strike.astype(float).where(df.floor_strike.notna(), num)
    df["close"] = pd.to_datetime(df.close_time, utc=True, format="ISO8601")
    df["decision_ts"] = ((df.close - pd.Timedelta(hours=24) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype(int)
    ev_close = df.groupby("event_ticker").close.transform("max")
    df["dev"] = ev_close < pd.Timestamp(CUT[series], tz="UTC")
    return df[["series", "ticker", "event_ticker", "strike", "result", "expiration_value", "close", "decision_ts", "dev"]]


def _tdate(ev):
    """date token of an event ticker -> (year, month, day|None)"""
    tok = ev.split("-")[1]
    m = re.match(r"(\d\d)([A-Z]{3})(\d\d)?", tok)
    if not m or m.group(2) not in MON:
        return None, None, None
    return 2000 + int(m.group(1)), MON[m.group(2)], (int(m.group(3)) if m.group(3) else None)


def fred(name):
    d = pd.read_csv(DL / f"{name.lower()}.csv")
    d.columns = ["date", "v"]
    d["date"] = pd.to_datetime(d.date)
    d["v"] = pd.to_numeric(d.v, errors="coerce")
    return d.dropna().set_index("date").v


def anchors_and_sd(mk):
    out, sd = {}, {}
    icsa, un, pay = fred("ICSA"), fred("UNRATE"), fred("PAYEMS")
    sd["claims"] = float(icsa["2015":"2019"].diff().dropna().std())
    sd["KXU3"] = float(un["2011":"2019"].diff().dropna().std())
    sd["KXPAYROLLS"] = float(pay.diff()["2011":"2019"].diff().dropna().std() * 1000)
    tr = pd.read_excel(DL / "gdpnow.xlsx", sheet_name="TrackRecord").iloc[:, [0, 1, 3, 5]]
    tr.columns = ["q", "fc", "rel", "err"]
    tr = tr.dropna(subset=["q"])
    tr["q"], tr["rel"] = pd.to_datetime(tr.q), pd.to_datetime(tr.rel)
    sd["KXGDP"] = float(np.sqrt((tr[(tr.q >= "2011-09-30") & (tr.q <= "2019-12-31")].err.astype(float) ** 2).mean()))
    nc = pd.read_parquet(NOW)
    nc = nc[nc.kind == "month"]
    errs = []
    for t, g in nc.groupby("target"):
        if pd.Period(t, "M") > pd.Period("2021-07", "M"):
            continue
        act = g[g.series == "Actual Core PCE Inflation"]
        fc = g[g.series == "Core PCE Inflation"].sort_values("date")
        if len(act) and len(fc):
            f = fc[fc.date <= act.date.min() - pd.Timedelta(days=3)]
            if len(f):
                errs.append(round(act.value.iloc[0], 1) - f.value.iloc[-1])
    sd["KXPCECORE"] = float(np.std(errs, ddof=1))
    pce = nc[nc.series == "Core PCE Inflation"]
    for ev, g in mk.groupby("event_ticker"):
        s = g.series.iloc[0]
        y, mo, dy = _tdate(ev)
        if y is None and s != "KXGDP":
            out[ev] = np.nan
            continue
        dec = pd.Timestamp(g.decision_ts.min(), unit="s")
        a = np.nan
        if s in ("KXJOBLESS", "KXJOBLESSCLAIMS"):
            S = pd.Timestamp(y, mo, dy) - (pd.Timedelta(days=5) if s == "KXJOBLESSCLAIMS" else pd.Timedelta(0))
            a = icsa.get(S - pd.Timedelta(days=7), np.nan)
        elif s == "KXU3":
            a = un.get(pd.Timestamp(y, mo, 1) - pd.DateOffset(months=1), np.nan)
        elif s == "KXPAYROLLS":
            prev = mk[(mk.series == s)].copy()
            prev["ym"] = prev.event_ticker.map(lambda e: _tdate(e)[:2])
            want = (y, mo - 1) if mo > 1 else (y - 1, 12)
            pv = prev[prev.ym == want].expiration_value
            a = pd.to_numeric(pv.astype(str).str.replace(",", "").str.extract(r"(-?[\d.]+)")[0], errors="coerce").dropna()
            a = float(a.iloc[0]) if len(a) else np.nan
        elif s == "KXGDP":
            cl = g.close.max().tz_localize(None).normalize()
            m = tr[(tr.rel - cl).abs() <= pd.Timedelta(days=3)]
            a = float(m.fc.iloc[0]) if len(m) else np.nan
        elif s == "KXPCECORE":
            f = pce[(pce.target == f"{y}-{mo:02d}") & (pce.date <= dec.normalize() - pd.Timedelta(days=1))].sort_values("date")
            a = float(f.value.iloc[-1]) if len(f) else np.nan
        out[ev] = a
    return out, sd


def prices(mk):
    got = {}
    for s, g in mk.groupby("series"):
        ev = g.groupby("event_ticker").decision_ts.min()
        urls = [f"{API}/series/{s}/events/{e}/candlesticks?start_ts={t - 12 * 3600}&end_ts={t}&period_interval=60" for e, t in ev.items()]
        missing = []
        for (e, t), d in zip(ev.items(), fetch_many(urls)):
            tks = d.get("market_tickers", [])
            if not tks:
                missing.append(e)
            for tk, cs in zip(tks, d.get("market_candlesticks", [])):
                got[tk] = _px(cs, t)
        leg = g[g.event_ticker.isin(missing)]
        urls = [f"{API}/historical/markets/{tk}/candlesticks?start_ts={t - 12 * 3600}&end_ts={t}&period_interval=60"
                for tk, t in zip(leg.ticker, leg.decision_ts)]
        for tk, d in zip(leg.ticker, fetch_many(urls)):
            got[tk] = _px(d.get("candlesticks", []), int(leg.loc[leg.ticker == tk, "decision_ts"].iloc[0]))
    out = mk.copy()
    out["yes_bid"] = out.ticker.map(lambda k: got.get(k, (np.nan, np.nan))[0])
    out["yes_ask"] = out.ticker.map(lambda k: got.get(k, (np.nan, np.nan))[1])
    return out


def cmd_build():
    mk = pd.concat([settled(s) for s in SERIES], ignore_index=True)
    anc, sd = anchors_and_sd(mk)
    mk["anchor"] = mk.event_ticker.map(anc)
    mk["sd"] = mk.series.map(lambda s: sd["claims"] if s in ("KXJOBLESS", "KXJOBLESSCLAIMS") else sd[s])
    for s, g in mk.groupby("series"):
        print(f"{s}: {g.event_ticker.nunique()} events ({g[g.dev].event_ticker.nunique()} dev), {len(g)} rungs, "
              f"anchor coverage {g.anchor.notna().mean():.2f}, sd {g.sd.iloc[0]:.4g}")
    dev = prices(mk[mk.dev].copy())   # SEALED events are never priced here
    dev.to_parquet(OUT, index=False)
    (ROOT / "data" / "econ_reject_sd.json").write_text(json.dumps(sd, indent=1))
    print(f"DEV priced: {dev.yes_ask.notna().mean():.2f} of {len(dev)} rungs")
    return 0


def trial(d, k, maker=False):
    d = d[d.anchor.notna() & d.yes_ask.notna()].copy()
    up, lo = d.anchor + k * d.sd, d.anchor - k * d.sd
    rows = []
    if maker:
        sel = (d.strike > up) & (d.yes_ask < 1) & (d.yes_ask > d.yes_bid)
        s = d[sel].copy()
        p = 1 - s.yes_ask
        s["cost"] = p + p.map(lambda x: fee(x, 0.0175))
        s["side"] = "no"
        rows.append(s)
    else:
        s = d[(d.strike > up) & (d.yes_bid >= 0.01)].copy()
        p = 1 - s.yes_bid
        s["cost"] = p + p.map(lambda x: fee(x, 0.07))
        s["side"] = "no"
        rows.append(s)
        s = d[(d.strike < lo) & (d.yes_ask > 0) & (d.yes_ask <= 0.99)].copy()
        s["cost"] = s.yes_ask + s.yes_ask.map(lambda x: fee(x, 0.07))
        s["side"] = "yes"
        rows.append(s)
    t = pd.concat(rows)
    t["pnl"] = np.where(t.result == t.side, 1 / t.cost - 1, -1.0)
    return t


def stats(t, seed=0):
    if t.empty:
        return {"bets": 0, "events": 0, "losses": 0, "units": 0.0, "roi": None, "ci_lo": None, "ci_hi": None, "p": None}
    ev = t.groupby("event_ticker").pnl.sum().to_numpy()
    n = t.groupby("event_ticker").size().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ev), (10000, len(ev)))
    roi_b = ev[idx].sum(1) / n[idx].sum(1)
    return {"bets": int(len(t)), "events": int(len(ev)), "losses": int((t.pnl < 0).sum()), "units": float(t.pnl.sum()),
            "roi": float(t.pnl.mean()), "ci_lo": float(np.quantile(roi_b, .025)), "ci_hi": float(np.quantile(roi_b, .975)),
            "p": float(np.mean(roi_b <= 0)), "avg_cost": float(t.cost.mean())}


def cmd_dev():
    d = pd.read_parquet(OUT)
    assert d.dev.all()
    sd = json.loads((ROOT / "data" / "econ_reject_sd.json").read_text())
    print("sd:", {k: round(v, 4) for k, v in sd.items()})
    q = d[d.yes_ask.notna() & d.anchor.notna()]
    print(f"DEV {d.event_ticker.nunique()} events, {len(d)} rungs; priced+anchored {len(q)}")
    # coverage descriptive: where do rungs sit vs the band, and are the outside ones quoted?
    z = (q.strike - q.anchor) / q.sd
    print("rung z = (strike-anchor)/sd quantiles:", np.round(np.quantile(z, [0, .1, .5, .9, 1]), 2).tolist())
    for k in (2, 3):
        o = q[z.abs() > k]
        print(f"  |z|>{k}: {len(o)} rungs; quoted tradeable side "
              f"{int(((o.strike > o.anchor) & (o.yes_bid >= .01)).sum() + ((o.strike < o.anchor) & (o.yes_ask <= .99)).sum())}")
    res = {}
    for name, k in TRIALS.items():
        t = trial(d, k)
        r = stats(t)
        t = t.sort_values("close")
        h = len(t) // 2
        r["half1"] = stats(t.iloc[:h])["roi"] if h else None
        r["half2"] = stats(t.iloc[h:])["roi"] if h else None
        r["by_series"] = {s: {"bets": int(len(g)), "losses": int((g.pnl < 0).sum()), "roi": float(g.pnl.mean())}
                          for s, g in t.groupby("series")}
        r["by_side"] = {s: {"bets": int(len(g)), "roi": float(g.pnl.mean())} for s, g in t.groupby("side")}
        r["maker_upper_bound"] = stats(trial(d, k, maker=True))
        # POST-HOC (added after first run): the registered event bootstrap is degenerate when a sample has 0 losses
        # (every resample is positive -> p = 0). Market-price null: each bet wins with prob = its cost (price+fee).
        if len(t):
            c = t.cost.to_numpy()
            sim = np.where(np.random.default_rng(1).random((200000, len(c))) < c, 1 / c - 1, -1).sum(1)
            r["null_p_market_price"] = float((sim >= t.pnl.sum()).mean())
            r["expected_losses_at_price"] = float((1 - c).sum())
        res[name] = r
        print(f"\n{name} (k={k}): bets {r['bets']} / events {r['events']}, losses {r['losses']}, ROI "
              f"{(r['roi'] or 0):+.2%}, CI [{(r['ci_lo'] or 0):+.2%}, {(r['ci_hi'] or 0):+.2%}], p {r['p']}, halves {r['half1']} / {r['half2']}")
        print("   by series:", r["by_series"])
        print("   by side:", r["by_side"])
        print("   maker NO upper bound:", {k2: r["maker_upper_bound"][k2] for k2 in ("bets", "losses", "roi", "p")})
        if len(t):
            print(t[["series", "ticker", "side", "anchor", "sd", "strike", "yes_bid", "yes_ask", "cost", "result", "pnl"]].to_string())
    if "--ledger" in sys.argv:
        prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
        for i, (name, r) in enumerate(res.items(), 1):
            with LEDGER.open("a") as f:
                f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"econ_reject_{name}",
                                    "sha": hashlib.sha256(f"econ_reject:{name}".encode()).hexdigest()[:12], "dataset": DATASET,
                                    "hypothesis": "pre-registered in scripts/econ_reject.py; NO on econ-ladder rungs outside anchor +- k*sd",
                                    "source": "weather consensus-reject transfer", "status": "ok", "split": "DEV", "trial_number": i,
                                    "study_trials": len(TRIALS), "bonferroni_bar": 0.05 / len(TRIALS), **r,
                                    "passed_registered_bootstrap": bool(r["p"] is not None and r["p"] < 0.05 / len(TRIALS) and r["roi"] > 0),
                                    "passed": False, "verdict": VERDICT[name]}, default=str) + "\n")
        print(f"appended {len(res)} ledger rows (prior rows {len(prior)})")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
