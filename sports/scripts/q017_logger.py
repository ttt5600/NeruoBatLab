"""Q017 forward paper logger: Kalshi KXTXERCOTPEAKD quotes + ERCOT's own 7-day load forecast vintages (no orders).

Why: the historical leg (notes/q017_ercot_load.md) has only 60 operating days and the far-tail rule fired on 7 DEV
days -- underpowered. This logger accrues the same frozen rules forward, with the forecast vintage captured live.

  tick   -- (run every 30 min; NOT installed in launchd) append
            data/q017/fwd/vintages.jsonl  every unseen ERCOT NP3-560-CD vintage: {vintage (CT stamp), day, peak}
            data/q017/fwd/quotes.jsonl    every open KXTXERCOTPEAKD market: {ts, ticker, opday, K, bid/ask, sizes, vol}
  report -- paper P&L of the frozen trials A06/A10/B06/B10 on settled op days.
            Frozen forecast = F_dm1: peak for D from the LAST ERCOT vintage stamped <= 14:30 CT on D-1 (the vintage EIA
            DF reproduces, audit 10-29) -- identical to the backtest input. F_latest (last vintage <= T) is reported
            as a descriptive variant only. Quote at T = last snapshot in [T-45 min, T]. Error model = frozen
            data/q017/error_sample.npy (pre-Kalshi EIA, n 366). Rules imported from q017_ercot_load (no copy).
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q017_ercot_load as q  # noqa: E402

FWD = q.D / "fwd"
FWD.mkdir(parents=True, exist_ok=True)
VINT, QUOTES = FWD / "vintages.jsonl", FWD / "quotes.jsonl"


def _seen_vintages() -> set[str]:
    if not VINT.exists():
        return set()
    return {json.loads(l)["vintage"] for l in VINT.open()}


def tick() -> int:
    now = pd.Timestamp.now(tz="UTC")
    # 1) ERCOT vintages
    html = q.get(q.ERCOT_LIST.format(12311), raw=True).decode()
    pairs = re.findall(r"<td class='labelOptional_ind'>([^<]+_csv\.zip)</td>.*?doclookupId=(\d+)", html, flags=re.S)
    seen = _seen_vintages()
    nv = 0
    with VINT.open("a") as f:
        for name, did in pairs:
            p = name.split(".")
            stamp = pd.Timestamp(p[3] + p[4][:6]).isoformat()
            if stamp in seen:
                continue
            z = zipfile.ZipFile(io.BytesIO(q.get(q.ERCOT_DL.format(did), raw=True, gap=0.3)))
            df = pd.read_csv(z.open(z.namelist()[0]))
            for d, v in df.groupby("DeliveryDate").SystemTotal.max().items():
                f.write(json.dumps({"vintage": stamp, "day": pd.to_datetime(d, format="%m/%d/%Y").date().isoformat(),
                                    "peak": float(v), "fetched": now.isoformat(timespec="seconds")}) + "\n")
            seen.add(stamp)
            nv += 1
    # 2) Kalshi open markets
    ms, cur = [], None
    while True:
        prm = {"series_ticker": q.SERIES, "status": "open", "limit": 1000}
        if cur:
            prm["cursor"] = cur
        r = q.get(f"{q.API}/markets", prm)
        ms += r.get("markets", [])
        cur = r.get("cursor")
        if not cur:
            break
    with QUOTES.open("a") as f:
        for m in ms:
            f.write(json.dumps({
                "ts": now.isoformat(timespec="seconds"), "ticker": m["ticker"],
                "opday": q.opday_from_rules(m["rules_primary"]).date().isoformat(), "K": float(m["floor_strike"]),
                "yes_bid": float(m.get("yes_bid_dollars") or 0), "yes_ask": float(m.get("yes_ask_dollars") or 0),
                "yes_bid_size": float(m.get("yes_bid_size_fp") or 0), "yes_ask_size": float(m.get("yes_ask_size_fp") or 0),
                "volume": float(m.get("volume_fp") or 0)}) + "\n")
    print(f"tick {now:%Y-%m-%d %H:%M}Z: {nv} new ERCOT vintages ({len(seen)} total), {len(ms)} open markets logged")
    return 0


def _forecasts(v: pd.DataFrame, day: pd.Timestamp, T_ct: pd.Timestamp):
    g = v[v.day == day]
    cut = day - pd.Timedelta(days=1) + pd.Timedelta(hours=14, minutes=30)
    a = g[g.vintage <= cut]
    b = g[g.vintage <= T_ct]
    f_dm1 = float(a.loc[a.vintage.idxmax(), "peak"]) if len(a) else None
    f_lat = float(b.loc[b.vintage.idxmax(), "peak"]) if len(b) else None
    return f_dm1, f_lat


def report() -> int:
    if not QUOTES.exists() or not VINT.exists():
        print("no forward data yet")
        return 0
    v = pd.read_json(VINT, lines=True)
    v["vintage"] = pd.to_datetime(v.vintage)
    v["day"] = pd.to_datetime(v.day)
    qt = pd.read_json(QUOTES, lines=True)
    qt["ts"] = pd.to_datetime(qt.ts, utc=True)
    qt["opday"] = pd.to_datetime(qt.opday)
    r = np.load(q.D / "error_sample.npy")
    print(f"forward: {qt.ts.min():%m-%d %H:%M}..{qt.ts.max():%m-%d %H:%M}Z, {qt.ts.nunique()} ticks, "
          f"{qt.opday.nunique()} op days quoted, {v.vintage.nunique()} ERCOT vintages")
    # settled outcomes (fetched now; never used by tick)
    res = {}
    for e in sorted(qt.ticker.str.rsplit("-", n=1).str[0].unique()):
        for m in q.get(f"{q.API}/markets", {"event_ticker": e}).get("markets", []):
            if m.get("result") in ("yes", "no"):
                res[m["ticker"]] = m["result"]
    rows = []
    for (tk, day), g in qt.groupby(["ticker", "opday"]):
        for h in q.TIMES:
            T_ct = day + pd.Timedelta(hours=h)
            T = T_ct.tz_localize(q.CT).tz_convert("UTC")
            s = g[(g.ts <= T) & (g.ts >= T - pd.Timedelta(minutes=45))]
            if s.empty:
                continue
            s = s.sort_values("ts").iloc[-1]
            f_dm1, f_lat = _forecasts(v, day, T_ct)
            for kind, F in (("dm1", f_dm1), ("latest", f_lat)):
                if F is None:
                    continue
                rows.append({"ticker": tk, "opday": day, "hour": h, "kind": kind, "K": s.K, "F": F,
                             "yes_bid": s.yes_bid, "yes_ask": s.yes_ask, "has_quote": True,
                             "no_ask_size": s.yes_bid_size, "yes_ask_size": s.yes_ask_size})
    if not rows:
        print("no decision-time snapshots yet (need ticks around 06:00 / 10:00 CT)")
        return 0
    dd = pd.DataFrame(rows)
    for kind in ("dm1", "latest"):
        x = q.decide(dd[dd.kind == kind], r, 1.0)
        tag = "FROZEN" if kind == "dm1" else "descriptive"
        for t in ("A06", "A10", "B06", "B10"):
            b = q.trial_bets(x, t)
            if b.empty:
                print(f"  {kind:6s} {t}: 0 signals")
                continue
            b["result"] = b.ticker.map(res)
            open_ = int(b.result.isna().sum())
            b = b.dropna(subset=["result"])
            if b.empty:
                print(f"  {kind:6s} {t}: {open_} signals, none settled")
                continue
            b["win"] = (b.result == b.side).astype(int)
            b["cost"] = b.ask + b.fee
            b["pnl"] = b.win - b.cost
            roi = b.pnl.sum() / b.cost.sum()
            ci = ""
            if b.opday.nunique() >= 5:
                lo, hi, p = q.boot(b, np.random.default_rng(q.SEED))
                ci = f" [{100 * lo:+.1f}, {100 * hi:+.1f}] p {p:.3f}"
            print(f"  {kind:6s} {t} ({tag}): settled {len(b)} bets / {b.opday.nunique()} days, win {b.win.mean():.3f}, "
                  f"avg ask {b.ask.mean():.3f}, units {b.pnl.sum():+.2f}, ROI {100 * roi:+.2f}%{ci}; open {open_}")
    return 0


if __name__ == "__main__":
    sys.exit({"tick": tick, "report": report}[sys.argv[1]]())
