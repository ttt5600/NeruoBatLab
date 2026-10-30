"""Q015: Kalshi approval-average markets vs the already-published RCP average. Pre-registered in notes/q015_approval_avg.md.

    python scripts/q015_approval_avg.py build      # RCP daily history JSON + settled Kalshi markets -> data/q015/
    python scripts/q015_approval_avg.py events     # rule check + eligible events per trial + DEV/SEALED split (no prices)
    python scripts/q015_approval_avg.py dev [--ledger]   # DEV trials only; SEALED event candles are never fetched

Info rule: an RCP history entry dated D is usable from 00:00 ET on D+1; a KXTRUMPAPPROVE settlement value is usable from
its 13:00 ET observation time. Model: empirical Delta_k pmf from the 2017-01-27..2021-09-30 history (pre-Kalshi).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports.http import fetch

API = "https://api.elections.kalshi.com/trade-api/v2"
D = ROOT / "data" / "q015"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "q015_approval_avg"
ET = "America/New_York"
RCP_IDS = (6179, 7320, 8656)  # Trump-1, Biden, Trump-2 ("President Trump 2nd Term Job Approval")
PMF_END = pd.Timestamp("2021-09-30")
TRANSITION = pd.Timestamp("2021-01-20")  # Trump-1 series ends, Biden series starts 01-27 (approval jumps ~14 pts)
PMF_KEEP_TRANSITION = "--orig-pmf" in sys.argv  # the first DEV run (ledger rows 'q015_T*') used the buggy pmf
TRIALS = {  # id: (series, decision rule, horizon days)
    "T1_wk24": ("KXAPRPOTUS", "close-24h", 2),
    "T2_wk72": ("KXAPRPOTUS", "close-72h", 4),
    "T3_day24": ("KXTRUMPAPPROVE", "close-24h+6m", 1),
    "T4_wk2": ("KXAPRPOTUS", "close-2h", 1),
}
EDGE = 0.05
BAR = 0.05 / len(TRIALS)
SEED = 1015
NBOOT = 10000


# ---------------------------------------------------------------- data
def _curl(url: str) -> str:
    r = subprocess.run(["/usr/bin/curl", "-s", "-m", "60", "-A", "Mozilla/5.0", url], capture_output=True, text=True)
    return r.stdout


def rcp_history() -> pd.DataFrame:
    rows = []
    for i in RCP_IDS:
        f = D / f"rcp_{i}_historical.js"
        t = f.read_text()
        p = json.loads(t[t.index("(") + 1: t.rindex(")")])["poll"]
        for a in p["rcp_avg"]:
            v = {c["name"]: c["value"] for c in a["candidate"]}
            dt = pd.to_datetime(a["date"], format="%a, %d %b %Y %H:%M:%S %z").tz_localize(None).normalize()
            if v.get("Approve") not in (None, ""):
                rows.append({"date": dt, "approve": float(v["Approve"]), "src": i})
    h = pd.DataFrame(rows).sort_values(["date", "src"]).drop_duplicates("date", keep="last").reset_index(drop=True)
    return h


def settled(series: str) -> pd.DataFrame:
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
    return pd.DataFrame(rows).drop_duplicates("ticker")


def cmd_build() -> int:
    D.mkdir(parents=True, exist_ok=True)
    for i in RCP_IDS:
        t = _curl(f"https://www.realclearpolitics.com/epolls/json/{i}_historical.js")
        if t.startswith("return_json("):
            (D / f"rcp_{i}_historical.js").write_text(t)
        else:
            print(f"WARN rcp {i}: not JSON ({len(t)} bytes); keeping cached copy")
    h = rcp_history()
    h.to_parquet(D / "rcp_history.parquet", index=False)
    print(f"rcp history {len(h)} days {h.date.min().date()}..{h.date.max().date()}")
    for s in ("KXAPRPOTUS", "KXTRUMPAPPROVE", "KXTRUMPVH"):
        df = settled(s)
        df.to_parquet(D / f"settled_{s}.parquet", index=False)
        print(f"{s}: {len(df)} markets, {df.event_ticker.nunique()} events")
    return 0


# ---------------------------------------------------------------- contract semantics (tenths of a point, integers)
def yes_set(row) -> tuple[int, int] | None:
    """Inclusive [lo, hi] in tenths for which the market is YES, from yes_sub_title / strike fields."""
    t = str(row.get("yes_sub_title") or "")
    n = lambda x: int(round(float(x) * 10))
    m = re.fullmatch(r"\s*([\d.]+)%?\s+to\s+([\d.]+)%?\s*", t)
    if m:
        return n(m.group(1)), n(m.group(2))
    m = re.fullmatch(r"\s*([\d.]+)%?\s+or (?:above|higher)\s*", t)
    if m:
        return n(m.group(1)), 10 ** 4
    m = re.fullmatch(r"\s*([\d.]+)%?\s+or (?:below|lower)\s*", t)
    if m:
        return -1, n(m.group(1))
    st, fl, cp = row.get("strike_type"), row.get("floor_strike"), row.get("cap_strike")
    if st == "between" and fl == fl and cp == cp:
        return n(fl), n(cp)
    if st == "greater" and fl == fl:
        return n(fl) + 1, 10 ** 4
    if st == "less" and cp == cp:
        return -1, n(cp) - 1
    return None


def markets() -> pd.DataFrame:
    out = []
    for s in ("KXAPRPOTUS", "KXTRUMPAPPROVE"):
        m = pd.read_parquet(D / f"settled_{s}.parquet")
        m["series"] = s
        out.append(m)
    m = pd.concat(out, ignore_index=True)
    m = m[m.result.isin(["yes", "no"])].copy()
    ys = [yes_set(r) for r in m.to_dict("records")]
    m["lo"] = [y[0] if y else np.nan for y in ys]
    m["hi"] = [y[1] if y else np.nan for y in ys]
    m["ev"] = pd.to_numeric(m.expiration_value, errors="coerce")
    m["close"] = pd.to_datetime(m.close_time, utc=True, format="ISO8601")
    m["close_ts"] = (m.close - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    m["y"] = (m.result == "yes").astype(int)
    return m


def rule_check(m: pd.DataFrame) -> tuple[int, int]:
    k = m[m.ev.notna() & m.lo.notna()]
    v = (k.ev * 10).round().astype(int)
    pred = ((v >= k.lo) & (v <= k.hi)).astype(int)
    bad = k[pred != k.y]
    if len(bad):
        print(bad[["ticker", "yes_sub_title", "ev", "result"]].head(10).to_string())
    return len(bad), len(k)


# ---------------------------------------------------------------- info + model
def info_sources(m: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    """All point-in-time observations of the RCP Approve value with the time they became public (UTC seconds)."""
    hist = pd.DataFrame({"val": h.approve.values, "src": "rcp_hist",
                         "avail": (h.date + pd.Timedelta(days=1)).dt.tz_localize(ET).dt.tz_convert("UTC")})
    k = m[(m.series == "KXTRUMPAPPROVE") & m.ev.notna()].drop_duplicates("event_ticker")
    kal = pd.DataFrame({"val": k.ev.values, "src": "kalshi_13et", "avail": k.close.values + pd.Timedelta(minutes=1)})
    kal["avail"] = pd.to_datetime(kal.avail, utc=True)
    s = pd.concat([hist, kal], ignore_index=True)
    s["avail_ts"] = (s.avail - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    return s.sort_values("avail_ts").reset_index(drop=True)


def decision_ts(close_ts: int, rule: str) -> int:
    return {"close-24h": close_ts - 24 * 3600, "close-72h": close_ts - 72 * 3600, "close-2h": close_ts - 2 * 3600,
            "close-24h+6m": close_ts - 24 * 3600 + 6 * 60}[rule]


def pmf(h: pd.DataFrame, k: int) -> dict[int, float]:
    s = h[h.date <= PMF_END].set_index("date").approve
    s = s.asfreq("D").ffill()
    d = ((s.shift(-k) - s) * 10).round()
    if not PMF_KEEP_TRANSITION:  # bug fix 2026-10-29: drop deltas spanning the Trump-1 -> Biden series switch
        d[(d.index > TRANSITION - pd.Timedelta(days=k + 1)) & (d.index < TRANSITION + pd.Timedelta(days=8))] = np.nan
    d = d.dropna().astype(int)
    emp = d.value_counts(normalize=True)
    sd = d.std() * 2
    grid = np.arange(-60, 61)
    nrm = np.exp(-0.5 * (grid / sd) ** 2)
    nrm /= nrm.sum()
    p = {int(g): 0.05 * w for g, w in zip(grid, nrm)}
    for g, w in emp.items():
        p[int(g)] = p.get(int(g), 0.0) + 0.95 * w
    return p


def p_yes(info: float, lo: int, hi: int, pm: dict[int, float]) -> float:
    i = int(round(info * 10))
    return float(sum(w for dlt, w in pm.items() if lo <= i + dlt <= hi))


def fee(p: float) -> float:
    return 0.07 * p * (1 - p)


def fee_ceil1(p: float) -> float:
    return np.ceil(round(fee(p) * 100, 9)) / 100


# ---------------------------------------------------------------- events + split
def eligible(m: pd.DataFrame, src: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for tid, (series, rule, k) in TRIALS.items():
        ev = m[m.series == series].groupby("event_ticker").close_ts.max().reset_index()
        ev["tau"] = ev.close_ts.map(lambda c: decision_ts(int(c), rule))
        idx = np.searchsorted(src.avail_ts.values, ev.tau.values, side="right") - 1
        ev["iv"] = [src.val.iloc[j] if j >= 0 else np.nan for j in idx]
        ev["info_src"] = [src.src.iloc[j] if j >= 0 else None for j in idx]
        ev["info_age_h"] = [(t - src.avail_ts.iloc[j]) / 3600 if j >= 0 else np.nan for t, j in zip(ev.tau, idx)]
        # stale guard: info must be < 4 days old at tau (history feed died 2026-02-25)
        ev = ev[ev.iv.notna() & (ev.info_age_h < 96)].sort_values("close_ts").reset_index(drop=True)
        cut = int(np.floor(0.7 * len(ev)))
        ev["split"] = np.where(np.arange(len(ev)) < cut, "DEV", "SEALED")
        out[tid] = ev
    return out


def cmd_events() -> int:
    h = pd.read_parquet(D / "rcp_history.parquet")
    m = markets()
    bad, n = rule_check(m)
    print(f"rule check: {bad} mismatches / {n} numeric-ev markets; unparsed strikes {int(m.lo.isna().sum())}")
    src = info_sources(m, h)
    el = eligible(m, src)
    sealed = {}
    for tid, ev in el.items():
        dv, sl = ev[ev.split == "DEV"], ev[ev.split == "SEALED"]
        to_d = lambda x: pd.to_datetime(x, unit="s", utc=True).strftime("%Y-%m-%d")
        print(f"{tid}: {len(ev)} events | DEV {len(dv)} {to_d(dv.close_ts.min())}..{to_d(dv.close_ts.max())} | "
              f"SEALED {len(sl)} {to_d(sl.close_ts.min())}..{to_d(sl.close_ts.max())} | info src {ev.info_src.value_counts().to_dict()} "
              f"| median info age {ev.info_age_h.median():.1f} h")
        sealed[tid] = sl.event_ticker.tolist()
    sf = D / "sealed_events.json"
    if not sf.exists():
        sf.write_text(json.dumps(sealed, indent=1))
        print("wrote", sf)
    for k in (1, 2, 4):
        pm = pmf(h, k)
        print(f"Delta_{k}: P(0) {pm.get(0, 0):.3f}, P(|d|<=0.2) {sum(w for d, w in pm.items() if abs(d) <= 2):.3f}")
    return 0


# ---------------------------------------------------------------- prices (DEV only)
def _fetch_safe(url):
    try:
        return fetch(url)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _px(candles, t):
    c = [x for x in candles if x["end_period_ts"] <= t]
    if not c:
        return np.nan, np.nan, np.nan
    last = c[-1]

    def g(side):
        v = last.get(side) or {}
        return pd.to_numeric(v.get("close_dollars", v.get("close")), errors="coerce")
    return g("yes_bid"), g("yes_ask"), last["end_period_ts"]


def prices(mk: pd.DataFrame) -> pd.DataFrame:
    got = {}
    from concurrent.futures import ThreadPoolExecutor
    urls = [(tk, t, f"{API}/historical/markets/{tk}/candlesticks?start_ts={t - 6 * 3600}&end_ts={t}&period_interval=60")
            for tk, t in zip(mk.ticker, mk.tau)]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda u: _fetch_safe(u[2]), urls))
    miss = []
    for (tk, t, _), d in zip(urls, res):
        cs = d.get("candlesticks") if isinstance(d, dict) else None
        if cs:
            got[tk] = _px(cs, t)
        else:
            miss.append((tk, t))
    if miss:  # recent markets: live per-market endpoint under the series
        sm = dict(zip(mk.ticker, mk.series))
        urls = [(tk, t, f"{API}/series/{sm[tk]}/markets/{tk}/candlesticks?start_ts={t - 6 * 3600}&end_ts={t}&period_interval=60")
                for tk, t in miss]
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(lambda u: _fetch_safe(u[2]), urls))
        for (tk, t, _), d in zip(urls, res):
            cs = d.get("candlesticks") if isinstance(d, dict) else None
            if cs:
                got[tk] = _px(cs, t)
    out = mk.copy()
    out["yes_bid"] = out.ticker.map(lambda k: got.get(k, (np.nan,) * 3)[0])
    out["yes_ask"] = out.ticker.map(lambda k: got.get(k, (np.nan,) * 3)[1])
    out["quote_ts"] = out.ticker.map(lambda k: got.get(k, (np.nan,) * 3)[2])
    return out


def bets(px: pd.DataFrame, pm: dict[int, float], fee_fn=fee) -> pd.DataFrame:
    rows = []
    for r in px.itertuples():
        b, a = r.yes_bid, r.yes_ask
        if not (b == b and a == a and 0 < b < a < 1):
            continue
        assert r.quote_ts <= r.tau, "quote after decision time"
        p = p_yes(r.iv, int(r.lo), int(r.hi), pm)
        ey = p - a - fee_fn(a)
        no_cost = 1 - b
        en = (1 - p) - no_cost - fee_fn(no_cost)
        if ey >= EDGE and ey >= en:
            cost = a + fee_fn(a)
            rows.append({**r._asdict(), "side": "yes", "p": p, "price": a, "cost": cost, "pnl": r.y - cost, "edge": ey})
        elif en >= EDGE:
            cost = no_cost + fee_fn(no_cost)
            rows.append({**r._asdict(), "side": "no", "p": 1 - p, "price": no_cost, "cost": cost,
                         "pnl": (1 - r.y) - cost, "edge": en})
    return pd.DataFrame(rows)


def boot(bt: pd.DataFrame, seed: int = SEED, n: int = NBOOT):
    g = bt.groupby("event_ticker").agg(pnl=("pnl", "sum"), cost=("cost", "sum"))
    P, C = g.pnl.values, g.cost.values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    r = P[idx].sum(1) / C[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), float((r <= 0).mean()), float(r.mean())


def cmd_dev(ledger: bool) -> int:
    h = pd.read_parquet(D / "rcp_history.parquet")
    m = markets()
    bad, n = rule_check(m)
    if bad > 0.01 * n:
        print(f"RULE CHECK FAILED {bad}/{n} -> no scoring")
        return 1
    src = info_sources(m, h)
    el = eligible(m, src)
    sealed = json.loads((D / "sealed_events.json").read_text())
    res = []
    for ti, (tid, (series, rule, k)) in enumerate(TRIALS.items(), 1):
        ev = el[tid]
        dv = ev[ev.split == "DEV"]
        assert not set(dv.event_ticker) & set(sealed[tid]), "sealed leak"
        mk = m[m.event_ticker.isin(dv.event_ticker) & m.lo.notna()].merge(
            dv[["event_ticker", "tau", "iv", "info_src", "info_age_h"]], on="event_ticker")
        pf = D / f"dev_prices_{tid}.parquet"
        if "--cached" in sys.argv and pf.exists():
            px = pd.read_parquet(pf)
        else:
            px = prices(mk)
            px.to_parquet(pf, index=False)
        pm = pmf(h, k)
        bt = bets(px, pm)
        quoted = int(((px.yes_bid > 0) & (px.yes_ask < 1) & (px.yes_ask > px.yes_bid)).sum())
        if bt.empty:
            print(f"{tid}: 0 bets ({len(mk)} markets, {quoted} quoted)")
            res.append({"tid": tid, "bets": 0})
            continue
        bt = bt.sort_values("close_ts")
        bt.to_parquet(D / f"dev_bets_{tid}.parquet", index=False)
        roi = bt.pnl.sum() / bt.cost.sum()
        lo, hi, p, bm = boot(bt)
        evs = bt.event_ticker.drop_duplicates().tolist()
        h1e = set(evs[: len(evs) // 2])
        b1, b2 = bt[bt.event_ticker.isin(h1e)], bt[~bt.event_ticker.isin(h1e)]
        r1, r2 = b1.pnl.sum() / b1.cost.sum(), b2.pnl.sum() / b2.cost.sum()
        bc = bets(px, pm, fee_ceil1)
        roi_c = bc.pnl.sum() / bc.cost.sum() if len(bc) else np.nan
        nev = bt.event_ticker.nunique()
        passed = bool(p < BAR and roi > 0 and r1 > 0 and r2 > 0 and nev >= 30)
        verdict = "PASS" if passed else ("UNDERPOWERED" if nev < 30 else "FAIL")
        print(f"{tid}: markets {len(mk)} quoted {quoted} bets {len(bt)} events {nev} | ROI {roi:+.4f} "
              f"[{lo:+.4f},{hi:+.4f}] p {p:.4f} (bar {BAR}) | halves {r1:+.4f} (n{len(b1)}) / {r2:+.4f} (n{len(b2)}) "
              f"| yes/no {bt.side.value_counts().to_dict()} | win {((bt.pnl > 0).mean()):.3f} vs mean model p {bt.p.mean():.3f} "
              f"| mean price {bt.price.mean():.3f} | ceil-fee ROI {roi_c:+.4f} | {verdict}")
        res.append({"tid": tid, "bets": len(bt), "roi": roi})
        if ledger:
            row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"q015_{tid}" + ("" if PMF_KEEP_TRANSITION else "_pmffix"),
                   "sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12], "dataset": DATASET,
                   "hypothesis": "pre-registered in notes/q015_approval_avg.md; Kalshi RCP approval brackets vs persistence "
                                 "of the already-published RCP average + pre-Kalshi Delta_k pmf; edge>=5c at ask+fee",
                   "source": "QUESTIONS Q015", "status": "ok" if PMF_KEEP_TRANSITION else "bugfix_rerun (pmf excluded Trump-1->Biden series jump; supersedes q015_T* rows)", "split": "DEV (first 70% of eligible events by close)",
                   "trial_number": ti, "study_trials": len(TRIALS), "bonferroni_bar": BAR, "bets": int(len(bt)),
                   "events": int(nev), "units": float(bt.pnl.sum()), "staked": float(bt.cost.sum()), "roi": float(roi),
                   "ci_lo": lo, "ci_hi": hi, "p": p, "half1": float(r1), "half2": float(r2), "half1_n": int(len(b1)),
                   "half2_n": int(len(b2)), "event_clustered": True, "passed": passed, "boot_mean": bm,
                   "win_rate": float((bt.pnl > 0).mean()), "mean_model_p": float(bt.p.mean()),
                   "roi_fee_ceil_1ct": float(roi_c), "series": series, "decision": rule, "horizon_days": k,
                   "verdict": verdict}
            with LEDGER.open("a") as f:
                f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "events"
    if cmd == "build":
        sys.exit(cmd_build())
    if cmd == "events":
        sys.exit(cmd_events())
    if cmd == "dev":
        sys.exit(cmd_dev("--ledger" in sys.argv))
    print(__doc__)
