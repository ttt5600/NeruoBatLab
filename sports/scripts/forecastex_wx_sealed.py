"""SEALED test FX1: ForecastEx daily-HIGH consensus-reject (frozen E1/E2 port), scored ONCE on 2026-08-15..2026-10-26.

    ../.venv_quant/bin/python scripts/forecastex_wx_sealed.py devcheck                 # DEV reproduction (safe, no sealed data)
    ../.venv_quant/bin/python scripts/forecastex_wx_sealed.py sealed --confirm-lockbox  # ONE shot (user: "yes ForecastEx" 2026-10-28)

Dev source: scripts/forecastex_wx.py (docstring = the original pre-registration, file sha256 8a6edbb5..., commit 24c6266),
notes/forecastex_wx.md. DEV 2026-03-01..08-14: T1 159 bets, +20.92u, ROI +13.2%, p .0004; T2 159 bets, +14.41u, p .0010.

=====================================================================================================================
PRE-REGISTRATION (written 2026-10-28, BEFORE any file dated >= 2026-08-15 with sealed contracts was downloaded or read.
data/forecastex/ holds DEV contracts only: cmd_pull dropped every row with event date >= 2026-08-15 before writing.)
=====================================================================================================================
CODE   The EXACT frozen DEV code path. This file imports scripts/forecastex_wx.py UNCHANGED (sha256 checked against
       FROZEN_SHA; abort before anything else if it differs) and calls its functions: cmd_build (contracts, D-1 trade
       prices, final settlement, monotonicity, Open-Meteo lead-1 forecasts at the WU station), candidates, picks, score.
       Only four module globals are re-pointed for the sealed window: DATA -> data/forecastex_sealed/, OUT ->
       data/forecastex_wx_sealed.parquet, FIRST_EVENT -> 2026-08-15, SPLIT -> 2026-10-27 (= one past END_EVENT, so
       cmd_build keeps event dates 08-15..10-26). MARGIN 3.0, FEE 0.01, models, hours, >=3 models: untouched (asserted).
RULE   unchanged: all >=3 models' max(hours<=18) + 3 < X + 1 -> buy NO of "exceed X"; min - 3 > X -> buy YES.
WINDOW event dates 2026-08-15 .. 2026-10-26 inclusive. Files downloaded: dated 2026-08-14 (D-1 prices of 08-15 events)
       .. 2026-10-27 (settlement of 10-26 events), mirroring DEV (`pull` ran D-1 of the first event .. one past the last).
       Rows kept on disk: UH* tickers that parse AND whose event date is in the window (nothing later is ever written).
SETTLED ONLY  a contract counts only if its YES settlement_price in the latest row dated >= its event day is exactly 0.00
       or 1.00 (cmd_build, unchanged). Others are dropped and COUNTED in the record. Non-monotone station-days dropped,
       counted (unchanged).
NEW SETTLEMENT WORDING (decided now, before seeing any sealed row). The Aug-2026 filing moved settlement from WU's
       Summary "Actual" to WU's "Daily Observations" max; the effective day is UNVERIFIED, so SEALED is mostly the new
       wording. Handling:
       (a) Ground truth is the exchange's OWN final settlement_price per contract, whatever wording governed it. We never
           re-derive an outcome from WU or NWS data, so the wording can change outcomes but cannot change the rule.
       (b) The rule is applied unchanged across the change: "exceed X" = strictly greater, same side mapping, NO WU-vs-NWS
           offset (DEV W1 found WU 1-2F below NWS on ~2/3 of days; adding a shift now would be a fitted change).
       (c) One pooled test over the whole window. No contract is excluded, re-labelled or split by wording era; because
           the effective day is unverified there is no era trial. By-month splits are descriptive only.
       (d) If the new wording in practice settles on >= rather than >, or on a different station value, that is part of
           what the sealed test measures; it is not grounds to re-score.
       (e) Schema gate (tickers/columns only, BEFORE any price, outcome or forecast is used): required columns present, and
           parsed in-window contracts cover >= 20 cities and >= 60 event dates. If a new ticker format or file layout
           breaks parsing, the run stops with status "schema_abort", records what it saw, computes NO outcome or P&L,
           and a human decides (the results file still blocks a silent re-run). Unparsed UH ticker count is recorded.
       (f) Descriptive flag "settlement_anomaly" if > 5% of sealed station-days are non-monotone or > 5% of in-window
           contracts lack a 0/1 settlement. It does not change the verdict; it is reported next to it.
PRIMARY  T1 (bought side's D-1 LAST trade + $0.01): one-sided bootstrap over station-days, 5,000 resamples, seed 0
       (forecastex_wx.score unchanged), p = P(resampled sum <= 0). Single primary test -> bar p < 0.05.
       LOSS-RATE CHECK (gate, defined now; DEV only reported it): Wilson 95% UPPER bound on the T1 loss rate must be
       below the T1 break-even loss rate (1 - mean cost). DEV: upper 8.0% vs break-even 12.5% -> would pass.
SECONDARY T2 (D-1 HIGH trade = worst price any buyer paid, + $0.01), same test, bar p < 0.05. Does not decide PASS/FAIL;
       T1 passing with T2 failing appends "-FRAGILE".
FLOOR  500 bets (the DEV pre-registration's resolvability floor).
VERDICT (written now):
       FAIL               T1 bets == 0, or T1 p >= 0.05, or T1 units <= 0, or the loss-rate check fails.
       PASS-UNDERPOWERED  T1 p < 0.05 AND units > 0 AND loss-rate check passes, but T1 bets < 500.
       PASS               as above with T1 bets >= 500.
       Suffix "-FRAGILE"  appended to either pass label when T2 fails (p >= 0.05 or units <= 0).
       A plain "PASS" is impossible below 500 bets. No outcome of this test authorises a real-money order: even PASS only
       buys the forward paper test on live IBKR bid/ask (notes/forecastex_wx.md, next steps). PASS-UNDERPOWERED is
       recorded as "consistent with DEV, not resolved".
DESCRIPTIVE (no pass/fail): halves by date, by month, by side (NO/YES), by city, by price bucket (price <= 0.80 vs > 0.80 --
       DEV earned +9.99u from the 27 bets priced <= 0.80 whose executability daily files cannot show), T1 units with
       the <= 0.80 bets removed, median D-1 pair_quantity on picks, candidates / traded / takeable counts, drop counts.
POWER  DEV Jul..Aug-14 gave 128 bets in ~45 days; 73 sealed days with more D-1 trading suggest ~200-400 bets, i.e.
       most likely below the floor: the best realistic label is PASS-UNDERPOWERED.
ONE SHOT `sealed` needs --confirm-lockbox; aborts if forecastex_wx.py's sha256 differs from FROZEN_SHA, if
       results/forecastex_wx_sealed.json exists, or if lab/lockbox.jsonl already has strategy forecastex_wx_FX1. It writes
       results/forecastex_wx_sealed.json {"status": "started"} BEFORE downloading or reading any sealed file, so a crash
       also blocks a re-run (a human decides; never delete it quietly). sha256 of every sealed CSV, the built parquet,
       forecastex_wx.py and this file go into the record.
DEV CHECK `devcheck` runs cmd_build on data/forecastex/ (DEV, asserted to contain no event >= 08-15) into a separate
       parquet, then the same picks/score/verdict code, and must reproduce T1 159 bets +20.92u and T2 159 bets +14.41u,
       with the identical pick set and P&L as the committed data/forecastex_wx.parquet.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecastex_wx as fx  # noqa: E402

ROOT = fx.ROOT
FROZEN_SHA = "8a6edbb5c86d1492778c41f4c9ed95d4da18aa9304fbb8ca6024a97182a88f92"  # scripts/forecastex_wx.py @ 24c6266
STRATEGY = "forecastex_wx_FX1"
DEV_SPLIT = date(2026, 8, 15)
END_EVENT = date(2026, 10, 26)
SEALED_DIR = ROOT / "data" / "forecastex_sealed"
SEALED_PARQ = ROOT / "data" / "forecastex_wx_sealed.parquet"
DEV_PARQ = ROOT / "data" / "forecastex_wx_devcheck.parquet"
OUT = ROOT / "results" / "forecastex_wx_sealed.json"
BETS = ROOT / "results" / "forecastex_wx_sealed_bets.csv"
DEVCHECK = ROOT / "results" / "forecastex_wx_devcheck.json"
LOCKBOX = ROOT / "lab" / "lockbox.jsonl"
FLOOR = 500
BAR = 0.05
CHEAP = 0.80
REQ_COLS = {"event_contract", "subtype", "date", "start_price", "high_price", "low_price", "end_price",
            "settlement_price", "pair_quantity", "vwap"}
DEV_WANT = {"T1_end": (159, 20.92), "T2_high": (159, 14.41)}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_frozen():
    h = sha(Path(fx.__file__))
    if h != FROZEN_SHA:
        raise SystemExit(f"ABORT: scripts/forecastex_wx.py sha256 {h[:12]} != frozen {FROZEN_SHA[:12]}")
    assert fx.MARGIN == 3.0 and fx.FEE == 0.01 and fx.TRIALS == {"T1_end": "end_price", "T2_high": "high_price"}
    assert fx.SPLIT == DEV_SPLIT and fx.FIRST_EVENT == date(2026, 3, 1)


def loss_check(r: dict) -> bool:
    return r["bets"] > 0 and r["loss_ci"][1] < r["breakeven_loss_rate"]


def verdict(r1: dict, r2: dict) -> str:
    if r1["bets"] == 0 or r1["p"] >= BAR or r1["units"] <= 0 or not loss_check(r1):
        return "FAIL"
    v = "PASS" if r1["bets"] >= FLOOR else "PASS-UNDERPOWERED"
    if r2["bets"] == 0 or r2["p"] >= BAR or r2["units"] <= 0:
        v += "-FRAGILE"
    return v


def evaluate(df: pd.DataFrame, lo: date, hi: date) -> tuple[dict, dict]:
    """Frozen picks/score on event dates lo..hi; returns (record, {trial: picks frame})."""
    df = df.copy()
    df["event_date"] = pd.to_datetime(df.event_date)
    df = df[(df.event_date >= pd.Timestamp(lo)) & (df.event_date <= pd.Timestamp(hi))]
    days = df.groupby(["code", "event_date"]).monotone.first()
    rec = {"contracts": int(len(df)), "cities": int(df.code.nunique()), "station_days": int(len(days)),
           "event_dates": int(df.event_date.nunique()),
           "contracts_no_01_settlement": int(df.yes_won.isna().sum()),
           "station_days_non_monotone": int((~days.astype(bool)).sum())}
    rec["settlement_anomaly"] = bool(rec["station_days_non_monotone"] > 0.05 * max(len(days), 1)
                                     or rec["contracts_no_01_settlement"] > 0.05 * max(len(df), 1))
    c = fx.candidates(df)
    rec["candidates"] = {"all": int(len(c)), "NO": int((c.side == "NO").sum()), "YES": int((c.side == "YES").sum())}
    out, tabs = {}, {}
    for name, field in fx.TRIALS.items():
        t = fx.picks(df, field)
        r = fx.score(t, np.random.default_rng(0))
        r["loss_check_pass"] = bool(loss_check(r)) if r["bets"] else False
        if len(t):
            r["by_month"] = {k: {"n": int(v["size"]), "units": round(float(v["sum"]), 4)}
                             for k, v in t.groupby(t.event_date.dt.strftime("%Y-%m")).pnl.agg(["size", "sum"]).iterrows()}
            r["by_side"] = {k: {"n": int(v["size"]), "units": round(float(v["sum"]), 4)}
                            for k, v in t.groupby("side").pnl.agg(["size", "sum"]).iterrows()}
            r["by_city"] = {k: {"n": int(v["size"]), "units": round(float(v["sum"]), 4)}
                            for k, v in t.groupby("code").pnl.agg(["size", "sum"]).iterrows()}
            cheap = t.price <= CHEAP
            r["by_price"] = {f"<={CHEAP}": {"n": int(cheap.sum()), "units": float(t.pnl[cheap].sum()),
                                           "losses": int((~t.won[cheap]).sum())},
                             f">{CHEAP}": {"n": int((~cheap).sum()), "units": float(t.pnl[~cheap].sum()),
                                            "losses": int((~t.won[~cheap]).sum())}}
            r["median_d1_pair_quantity"] = float(t.qty.median())
        out[name], tabs[name] = r, t
    rec["trials"] = out
    rec["n_vs_floor"] = f"{out['T1_end']['bets']} vs {FLOOR}"
    rec["verdict"] = verdict(out["T1_end"], out["T2_high"])
    return rec, tabs


def show(rec: dict, tag: str):
    print(f"\n{tag}: {rec['cities']} cities, {rec['station_days']} station-days, {rec['contracts']} contracts; "
          f"no 0/1 settlement {rec['contracts_no_01_settlement']}, non-monotone days {rec['station_days_non_monotone']}"
          f"{'  [SETTLEMENT ANOMALY]' if rec['settlement_anomaly'] else ''}; candidates {rec['candidates']}")
    for name, r in rec["trials"].items():
        if not r["bets"]:
            print(f"  {name}: 0 bets")
            continue
        print(f"  {name}: bets {r['bets']} station-days {r['station_days']} units {r['units']:+.4f} ROI {r['roi']:+.2%} "
              f"CI [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] p {r['p']:.4f}; losses {r['losses']} "
              f"({r['loss_rate']:.2%}, Wilson upper {r['loss_ci'][1]:.2%}) vs break-even {r['breakeven_loss_rate']:.2%} "
              f"-> loss check {'pass' if r['loss_check_pass'] else 'FAIL'}; mean cost {r['mean_cost']:.4f}")
        print(f"     halves H1 {r['units_H1']:+.2f}/{r['bets_H1']}  H2 {r['units_H2']:+.2f}/{r['bets_H2']}; "
              f"by side {r['by_side']}; by price {r['by_price']}; median qty {r['median_d1_pair_quantity']}")
        print(f"     by month {r['by_month']}")
    print(f"  n vs floor: {rec['n_vs_floor']}  ->  VERDICT {rec['verdict']}")


def cmd_devcheck() -> int:
    check_frozen()
    for f in sorted(fx.DATA.glob("uh_*.csv")):                       # DEV files must hold no sealed contract
        t = pd.read_csv(f, usecols=["event_contract"])
        m = fx.parse_ticker(t.event_contract.astype(str))
        assert not (m.event_date >= pd.Timestamp(DEV_SPLIT)).any(), f"{f.name} holds a sealed contract"
    fx.OUT = DEV_PARQ                                                 # same build, separate output
    fx.cmd_build()
    rec, tabs = evaluate(pd.read_parquet(DEV_PARQ), fx.FIRST_EVENT, DEV_SPLIT - timedelta(days=1))
    show(rec, "DEV CHECK (rebuilt)")
    ref, ref_tabs = evaluate(pd.read_parquet(ROOT / "data" / "forecastex_wx.parquet"), fx.FIRST_EVENT,
                             DEV_SPLIT - timedelta(days=1))
    ok = True
    for name, (n, u) in DEV_WANT.items():
        r = rec["trials"][name]
        a = tabs[name].set_index("event_contract")[["side", "price", "pnl"]].sort_index()
        b = ref_tabs[name].set_index("event_contract")[["side", "price", "pnl"]].sort_index()
        same = a.index.equals(b.index) and (a.side == b.side).all() and np.allclose(a[["price", "pnl"]], b[["price", "pnl"]])
        hit = r["bets"] == n and round(r["units"], 2) == u and same
        ok &= hit
        print(f"  {name}: want {n} bets {u:+.2f}u | rebuilt {r['bets']} {r['units']:+.4f}u p {r['p']:.4f} | "
              f"pick set+pnl identical to committed parquet: {same} -> {'MATCH' if hit else 'MISMATCH'}")
    DEVCHECK.write_text(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "match": bool(ok),
                                    "want": DEV_WANT, "rebuilt": rec, "fx_sha256": FROZEN_SHA,
                                    "this_sha256": sha(Path(__file__))}, indent=1, default=float))
    print(f"wrote {DEVCHECK}")
    return 0 if ok else 1


def pull_sealed() -> dict:
    """Files dated DEV_SPLIT-1 .. END_EVENT+1; keep parsed UH rows with event date in the sealed window only."""
    SEALED_DIR.mkdir(parents=True, exist_ok=True)
    d, last, info = DEV_SPLIT - timedelta(days=1), END_EVENT + timedelta(days=1), {"files": 0, "uh_unparsed_tickers": set()}
    while d <= last:
        f = SEALED_DIR / f"uh_{d:%Y%m%d}.csv"
        for attempt in range(5):
            try:
                r = requests.get(fx.URL, params={"type": "prices", "date": f"{d:%Y%m%d}"}, headers=fx.H, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(10 * (attempt + 1))
        t = pd.read_csv(io.StringIO(r.text))
        miss = REQ_COLS - set(t.columns)
        if miss:
            raise ValueError(f"schema: {d} missing columns {sorted(miss)}")
        t = t[t.event_contract.astype(str).str.startswith("UH")]
        meta = fx.parse_ticker(t.event_contract.astype(str))
        info["uh_unparsed_tickers"] |= set(t.event_contract[meta.event_date.isna()].astype(str))
        keep = meta.event_date.notna() & (meta.event_date >= pd.Timestamp(DEV_SPLIT)) & (meta.event_date <= pd.Timestamp(END_EVENT))
        t[keep].to_csv(f, index=False)
        info["files"] += 1
        print(f"{d}: {int(keep.sum())} in-window UH rows", flush=True)
        d += timedelta(days=1)
    info["uh_unparsed_tickers"] = sorted(info["uh_unparsed_tickers"])
    return info


def cmd_sealed(confirm: bool) -> int:
    if not confirm:
        print("sealed: needs --confirm-lockbox and the user's explicit yes (given 2026-10-28: 'yes ForecastEx')")
        return 1
    check_frozen()
    if OUT.exists():
        print(f"ABORT: {OUT} exists -- FX1 was already run (or crashed mid-run). Never re-score.")
        return 1
    if LOCKBOX.exists() and any(json.loads(l).get("strategy") == STRATEGY for l in LOCKBOX.read_text().splitlines() if l.strip()):
        print(f"ABORT: {STRATEGY} already in lab/lockbox.jsonl -- never re-score")
        return 1
    if SEALED_DIR.exists() and any(SEALED_DIR.iterdir()):
        print(f"ABORT: {SEALED_DIR} is not empty -- sealed files exist from an earlier attempt; a human decides")
        return 1
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {"ts": ts, "strategy": STRATEGY, "sha": hashlib.sha256(STRATEGY.encode()).hexdigest()[:12],
            "dataset": fx.DATASET, "window": f"{DEV_SPLIT}..{END_EVENT}", "fx_sha256": FROZEN_SHA,
            "this_sha256": sha(Path(__file__))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({**base, "status": "started"}))           # one-shot marker before any sealed download/read
    # ---- download + schema gate (tickers/columns only; no price, outcome or forecast used yet)
    try:
        info = pull_sealed()
    except Exception as e:  # noqa: BLE001
        OUT.write_text(json.dumps({**base, "status": "aborted_download", "error": repr(e)}, indent=1))
        print(f"ABORTED during download/schema: {e!r} -- nothing scored; a human decides")
        return 1
    files = sorted(SEALED_DIR.glob("uh_*.csv"))
    tick = pd.concat([pd.read_csv(f, usecols=["event_contract"]) for f in files]).event_contract.astype(str).unique()
    m = fx.parse_ticker(pd.Series(tick))
    gate = {"files": info["files"], "uh_unparsed_tickers": len(info["uh_unparsed_tickers"]),
            "uh_unparsed_examples": info["uh_unparsed_tickers"][:10], "cities": int(m.code.nunique()),
            "event_dates": int(m.event_date.nunique())}
    print(f"schema gate: {gate}")
    hashes = {f.name: sha(f) for f in files}
    if gate["cities"] < 20 or gate["event_dates"] < 60:
        OUT.write_text(json.dumps({**base, "status": "schema_abort", "gate": gate, "file_hashes": hashes}, indent=1))
        print("SCHEMA ABORT: coverage below gate -- no outcome or P&L computed; a human decides")
        return 1
    # ---- frozen build on the sealed window (only these four globals change)
    fx.DATA, fx.OUT, fx.FIRST_EVENT, fx.SPLIT = SEALED_DIR, SEALED_PARQ, DEV_SPLIT, END_EVENT + timedelta(days=1)
    fx.cmd_build()
    hashes[SEALED_PARQ.name] = sha(SEALED_PARQ)
    rec, tabs = evaluate(pd.read_parquet(SEALED_PARQ), DEV_SPLIT, END_EVENT)
    t1 = tabs["T1_end"]
    if len(t1):
        dear = t1[t1.price > CHEAP]
        rec["T1_units_excl_price_le_0.80"] = {"n": int(len(dear)), "units": float(dear.pnl.sum())}
    show(rec, f"SEALED FX1 {DEV_SPLIT}..{END_EVENT}")
    full = {**base, "status": "done", "gate": gate, "file_hashes": hashes, **rec}
    OUT.write_text(json.dumps(full, indent=1, default=float))
    with LOCKBOX.open("a") as f:
        f.write(json.dumps({k: v for k, v in full.items() if k not in ("file_hashes",)}, default=float) + "\n")
    cols = ["event_contract", "code", "event_date", "X", "side", "price", "cost", "qty", "won", "pnl"]
    pd.concat([tabs[k][cols].assign(trial=k) for k in tabs]).to_csv(BETS, index=False)
    print(f"wrote {OUT}, {BETS}; lockbox appended")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["devcheck", "sealed"])
    ap.add_argument("--confirm-lockbox", action="store_true")
    a = ap.parse_args()
    return cmd_devcheck() if a.cmd == "devcheck" else cmd_sealed(a.confirm_lockbox)


if __name__ == "__main__":
    raise SystemExit(main())
