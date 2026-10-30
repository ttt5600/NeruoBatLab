"""L11 earnings run-up straddle (pre-registered in notes/l11_earnings_straddle.md).

  ../.venv_quant/bin/python -I scripts/l11_earnings_straddle.py events      # yfinance + EDGAR -> event table (no prices)
  ../.venv_quant/bin/python -I scripts/l11_earnings_straddle.py fetch dev   # DoltHub chains for dev snapshot dates only
  ../.venv_quant/bin/python -I scripts/l11_earnings_straddle.py dev [--ledger]
  ../.venv_quant/bin/python -I scripts/l11_earnings_straddle.py fetch sealed; ... sealed [--ledger]   # only on a dev pass
"""
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "l11"
CHAINS = DATA / "chains"
RES = ROOT / "results" / "l11_earnings_straddle"
LEDGER = ROOT / "lab" / "ledger.jsonl"
API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"

UNIVERSE = ("AAPL MSFT NVDA AMZN META GOOGL TSLA NFLX AMD JPM BAC WMT DIS INTC BA CSCO ORCL CRM ADBE PYPL UBER "
            "COST PFE KO MU QCOM V GS").split()
START, END = "2020-01-06", "2026-10-28"     # DoltHub single-name coverage window (probed)
DEV_FRAC = 0.70
FEE = 0.03 / 100                            # $ per share-equivalent per contract per side
NBOOT = 10_000
K_TRIALS = 2
BAR = 0.05 / K_TRIALS


# ------------------------------------------------------------------------------------------------ calendar
def trading_days():
    """NYSE trading days from Yahoo SPY daily bars (reuses the q009 file; extended if needed)."""
    f = ROOT / "data" / "options" / "yahoo_spy" / "spy_daily.parquet"
    px = pd.read_parquet(f)
    return pd.DatetimeIndex(pd.to_datetime(px.index).tz_localize(None).normalize()).unique().sort_values()


# ------------------------------------------------------------------------------------------------ events
def edgar_202():
    """8-K item 2.02 acceptance times straight from data.sec.gov submissions JSON (raw acceptanceDateTime is true
    UTC: AAPL 20:30Z = 16:30 ET, JPM 10:30Z = 06:30 ET; the company_graph copy is shifted, so it is not reused)."""
    import requests
    f = DATA / "edgar_202.parquet"
    if f.exists():
        return pd.read_parquet(f)
    H = {"User-Agent": "quant-research-bot admin@quantresearch.local"}
    tick = json.loads((ROOT / "data" / "company_graph" / "company_tickers_2026-10-22.json").read_text())
    t2c = {v["ticker"]: int(v["cik_str"]) for v in tick.values()}
    rows = []
    for t in UNIVERSE:
        j = requests.get(f"https://data.sec.gov/submissions/CIK{t2c[t]:010d}.json", headers=H, timeout=30).json()
        time.sleep(0.5)
        blocks = [j["filings"]["recent"]]
        for ff in j["filings"].get("files", []):
            if ff.get("filingTo", "9999") < "2019-06-01":
                continue
            blocks.append(requests.get(f"https://data.sec.gov/submissions/{ff['name']}", headers=H, timeout=30).json())
            time.sleep(0.5)
        for b in blocks:
            for form, acc, items in zip(b["form"], b["acceptanceDateTime"], b["items"]):
                if form == "8-K" and "2.02" in str(items):
                    rows.append((t, acc))
    e = pd.DataFrame(rows, columns=["ticker", "accepted"]).drop_duplicates()
    e["ed_ts"] = pd.to_datetime(e.accepted, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    e.to_parquet(f)
    return e


def cmd_events():
    import yfinance as yf
    DATA.mkdir(parents=True, exist_ok=True)
    yf_rows = []
    for t in ([] if (DATA / "yf_earnings_dates.parquet").exists() else UNIVERSE):
        d = yf.Ticker(t).get_earnings_dates(limit=60)
        time.sleep(0.6)
        d = d.reset_index()
        d.columns = ["ts", "eps_est", "eps_rep", "surprise"][: len(d.columns)] if len(d.columns) == 4 else d.columns
        for _, r in d.iterrows():
            ts = pd.Timestamp(r["ts"]).tz_convert("America/New_York")
            yf_rows.append({"ticker": t, "yf_ts": ts.tz_localize(None), "eps_rep": r.get("eps_rep")})
    if yf_rows:
        pd.DataFrame(yf_rows).to_parquet(DATA / "yf_earnings_dates.parquet")
    y = pd.read_parquet(DATA / "yf_earnings_dates.parquet")

    e = edgar_202()

    td = trading_days()
    rows = []
    y = y[(y.yf_ts >= START) & (y.yf_ts <= END) & y.eps_rep.notna()]
    for _, r in y.iterrows():
        cand = e[(e.ticker == r.ticker)]
        dd = (cand.ed_ts.dt.normalize() - r.yf_ts.normalize()).dt.days
        ok = cand[(dd >= -3) & (dd <= 3)]
        ed = ok.ed_ts.min() if len(ok) else pd.NaT
        rows.append({"ticker": r.ticker, "yf_ts": r.yf_ts, "edgar_ts": ed})
    ev = pd.DataFrame(rows)

    def amc(ts):
        return ts.hour >= 16

    def bmo(ts):
        return ts.hour < 9 or (ts.hour == 9 and ts.minute < 30)

    out = []
    for _, r in ev.iterrows():
        rec = r.to_dict()
        if pd.isna(r.edgar_ts):
            rec["dropped"] = "no_edgar_match"
            out.append(rec)
            continue
        # latest allowed exit close date = the last close strictly before the earliest possible release.
        lim = []
        for ts, src in ((r.yf_ts, "yf"), (r.edgar_ts, "edgar")):
            dday = ts.normalize()
            if src == "yf" and ts.hour == 0 and ts.minute == 0:     # no time given -> unknown -> treat as BMO
                lim.append(dday - pd.Timedelta(days=1))
            elif amc(ts):
                lim.append(dday)                                    # release after D's close -> D's close is OK
            else:
                lim.append(dday - pd.Timedelta(days=1))             # BMO or intraday -> must exit by D-1 close
        xmax = min(lim)
        tmax = td[td <= xmax]
        if not len(tmax):
            rec["dropped"] = "calendar"
            out.append(rec)
            continue
        t0 = tmax[-1]                                               # last trading close before the release
        i0 = td.get_loc(t0)
        e_session = td[i0 + 1] if i0 + 1 < len(td) else t0 + pd.Timedelta(days=1)  # first session after release
        rec.update(t0=t0, e_session=e_session, timing="AMC" if t0 == min(r.yf_ts, r.edgar_ts).normalize() else "BMO",
                   entry_max=td[i0 + 1 - 5],                         # <= E - 5 trading days
                   disagree_days=abs((r.yf_ts.normalize() - r.edgar_ts.normalize()).days), dropped="")
        out.append(rec)
    ev = pd.DataFrame(out).sort_values(["e_session", "ticker"]).reset_index(drop=True)
    ev.to_parquet(DATA / "events.parquet")
    print(ev.dropped.value_counts().to_dict())
    k = ev[ev.dropped == ""]
    print("usable events", len(k), "tickers", k.ticker.nunique(), k.e_session.min(), k.e_session.max())
    print("timing", k.timing.value_counts().to_dict(), "yf/edgar date disagree>0:", int((k.disagree_days > 0).sum()))
    print(k.groupby("ticker").size().to_dict())


# ------------------------------------------------------------------------------------------------ chains
SEALED_FROM = pd.Timestamp("2024-10-23")


def load_events(split):
    ev = pd.read_parquet(DATA / "events.parquet")
    ev = ev[ev.dropped == ""].copy()
    return ev[ev.e_session < SEALED_FROM] if split == "dev" else ev[ev.e_session >= SEALED_FROM]


def mwf_days():
    td = trading_days()
    return td[td.weekday.isin([0, 2, 4])]


def snap_candidates(ev):
    """Entry/exit candidate dates per event (primary, one fallback), Mon/Wed/Fri trading days only."""
    m = mwf_days()
    out = []
    for i, r in ev.iterrows():
        en = m[m <= r.entry_max][-2:][::-1]
        ex = m[(m <= r.t0) & (m > en[0])][-2:][::-1]
        out.append({"i": i, "ticker": r.ticker, "en": list(en), "ex": list(ex)})
    return out


def _q(day, tickers, tries=4):
    import requests
    tl = ",".join(f"'{t}'" for t in sorted(tickers))
    q = (f"SELECT date, act_symbol, expiration, strike, call_put, bid, ask, vol, delta FROM option_chain "
         f"WHERE date='{day}' AND act_symbol IN ({tl})")
    for k in range(tries):
        try:
            r = requests.get(API, params={"q": q}, timeout=150)
            d = json.loads(r.text)
            if d.get("query_execution_status") == "Success":
                return d["rows"]
        except Exception:
            pass
        time.sleep(2 + 3 * k)
    return None


def fetch_dates(need):
    """need: {date_str: set(tickers)}; skips (date, ticker) already logged."""
    CHAINS.mkdir(parents=True, exist_ok=True)
    logf = DATA / "fetch_log.csv"
    done = set()
    if logf.exists():
        lg = pd.read_csv(logf)
        done = set(zip(lg.date, lg.ticker))
    todo = {d: {t for t in ts if (d, t) not in done} for d, ts in need.items()}
    todo = {d: ts for d, ts in todo.items() if ts}
    print("dates to fetch", len(todo), "pairs", sum(map(len, todo.values())), flush=True)

    def one(item):
        d, ts = item
        time.sleep(0.5)
        return d, ts, _q(d, ts)

    with ThreadPoolExecutor(max_workers=6) as ex:
        for k, (d, ts, rows) in enumerate(ex.map(one, sorted(todo.items()))):
            if rows is None:
                print("FAIL", d, flush=True)
                continue
            df = pd.DataFrame(rows)
            f = CHAINS / f"{d}.parquet"
            if f.exists() and len(df):
                df = pd.concat([pd.read_parquet(f), df], ignore_index=True)
            if len(df):
                df.to_parquet(f, index=False)
            n = df.groupby("act_symbol").size() if len(df) else pd.Series(dtype=int)
            pd.DataFrame([{"date": d, "ticker": t, "n": int(n.get(t, 0))} for t in ts]).to_csv(
                logf, mode="a", header=not logf.exists(), index=False)
            if k % 20 == 0:
                print(k, d, len(df), flush=True)


def cmd_fetch(split):
    ev = load_events(split)
    cands = snap_candidates(ev)
    for stage in (0, 1):
        need = {}
        logf = DATA / "fetch_log.csv"
        have = {}
        if logf.exists():
            lg = pd.read_csv(logf)
            have = {(d, t): n for d, t, n in zip(lg.date, lg.ticker, lg.n)}
        for c in cands:
            for key in ("en", "ex"):
                ds = [x.strftime("%Y-%m-%d") for x in c[key]]
                if not ds:
                    continue
                if stage == 0:
                    need.setdefault(ds[0], set()).add(c["ticker"])
                elif have.get((ds[0], c["ticker"]), 1) == 0 and len(ds) > 1:
                    need.setdefault(ds[1], set()).add(c["ticker"])
        if split == "dev":
            assert all(pd.Timestamp(d) < SEALED_FROM for d in need)
        fetch_dates(need)


# ------------------------------------------------------------------------------------------------ scoring
def build_trades(split):
    ev = load_events(split)
    cands = snap_candidates(ev)
    lg = pd.read_csv(DATA / "fetch_log.csv")
    have = {(d, t): n for d, t, n in zip(lg.date, lg.ticker, lg.n)}
    cache = {}

    def chain(d, t):
        if d not in cache:
            f = CHAINS / f"{d}.parquet"
            c = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["act_symbol"])
            for col in ("strike", "bid", "ask", "vol", "delta"):
                if col in c:
                    c[col] = pd.to_numeric(c[col], errors="coerce")
            cache[d] = c.drop_duplicates()
        c = cache[d]
        return c[c.act_symbol == t]

    def bump(k):
        drops[k] = drops.get(k, 0) + 1

    rows, drops = [], {}
    for c in cands:
        r = ev.loc[c["i"]]
        t = c["ticker"]
        en = next((x for x in c["en"] if have.get((x.strftime("%Y-%m-%d"), t), 0) > 0), None)
        ex = None if en is None else next(
            (x for x in c["ex"] if x > en and have.get((x.strftime("%Y-%m-%d"), t), 0) > 0), None)
        if en is None or ex is None:
            bump("no_snapshot")
            continue
        assert ex <= r.t0 < r.e_session and en < ex
        a, b = chain(en.strftime("%Y-%m-%d"), t), chain(ex.strftime("%Y-%m-%d"), t)
        exps = sorted(x for x in set(a.expiration) & set(b.expiration) if pd.Timestamp(x) > r.e_session)
        if not exps:
            bump("no_common_expiry")
            continue
        xp = exps[0]
        a, b = a[a.expiration == xp], b[b.expiration == xp]
        ac = a[a.call_put == "Call"].set_index("strike")
        ap = a[a.call_put == "Put"].set_index("strike")
        bc = b[b.call_put == "Call"].set_index("strike")
        bp = b[b.call_put == "Put"].set_index("strike")
        ok_en = [k for k in ac.index.intersection(ap.index)
                 if ac.bid[k] > 0 and ap.bid[k] > 0 and ac.ask[k] >= ac.bid[k] and ap.ask[k] >= ap.bid[k]]
        if not ok_en:
            bump("no_entry_quote")
            continue
        par = {k: abs((ac.bid[k] + ac.ask[k]) / 2 - (ap.bid[k] + ap.ask[k]) / 2) for k in ok_en}
        k_free = min(par, key=par.get)                              # entry-ATM ignoring exit availability
        ok = [k for k in ok_en if k in bc.index and k in bp.index]
        if not ok:
            bump("no_exit_strike")
            continue
        k = min(ok, key=par.get)
        cm0, pm0 = (ac.bid[k] + ac.ask[k]) / 2, (ap.bid[k] + ap.ask[k]) / 2
        F = k + cm0 - pm0
        if abs(k - F) / F > 0.03:
            bump("strike_far_from_forward")
            continue
        cost = ac.ask[k] + ap.ask[k]
        cb1, pb1 = max(bc.bid[k], 0), max(bp.bid[k], 0)
        cm1, pm1 = (bc.bid[k] + bc.ask[k]) / 2, (bp.bid[k] + bp.ask[k]) / 2
        rows.append({"ticker": t, "e_session": r.e_session, "t0": r.t0, "timing": r.timing, "entry": en, "exit": ex,
                     "expiry": xp, "strike": k, "k_free": k_free, "fwd": F, "k_dist": abs(k - F) / F,
                     "c_bid0": ac.bid[k], "c_ask0": ac.ask[k], "p_bid0": ap.bid[k], "p_ask0": ap.ask[k],
                     "c_bid1": bc.bid[k], "c_ask1": bc.ask[k], "p_bid1": bp.bid[k], "p_ask1": bp.ask[k],
                     "iv0": (ac.vol[k] + ap.vol[k]) / 2, "iv1": (bc.vol[k] + bp.vol[k]) / 2,
                     "cost": cost, "mid0": cm0 + pm0, "mid1": cm1 + pm1,
                     "roi": (cb1 + pb1 - cost - 4 * FEE) / cost,
                     "roi_mid": (cm1 + pm1 - cm0 - pm0) / (cm0 + pm0),
                     "rel_spread0": (cost - ac.bid[k] - ap.bid[k]) / (cm0 + pm0),
                     "rel_spread1": (bc.ask[k] + bp.ask[k] - bc.bid[k] - bp.bid[k]) / (cm1 + pm1),
                     "stale_exit": bool(bc.bid[k] == ac.bid[k] and bc.ask[k] == ac.ask[k]
                                        and bp.bid[k] == ap.bid[k] and bp.ask[k] == ap.ask[k])})
    tr = pd.DataFrame(rows).sort_values(["e_session", "ticker"]).reset_index(drop=True)
    # P2: entry IV below the name's trailing median over >= 4 previous events in this split
    tr["iv_med_prior"] = np.nan
    for t, g in tr.groupby("ticker"):
        iv = g.iv0.values
        tr.loc[g.index, "iv_med_prior"] = [np.median(iv[:j]) if j >= 4 else np.nan for j in range(len(iv))]
    tr["p2"] = tr.iv0 < tr.iv_med_prior
    tr["week"] = tr.e_session.dt.strftime("%G-%V")
    return tr, drops


def cluster_boot(x, g, seed=0):
    x = np.asarray(x, float)
    codes, _ = pd.factorize(pd.Series(list(g)))
    s = np.bincount(codes, weights=x)
    n = np.bincount(codes).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(s), size=(NBOOT, len(s)))
    return s[idx].sum(1) / n[idx].sum(1)


def summarize(tr):
    x = tr.roi.values
    m = cluster_boot(x, tr.week)
    mm = cluster_boot(tr.roi_mid.values, tr.week)
    h = len(tr) // 2
    wk = tr.groupby("week").roi.sum()
    r = {"n": int(len(tr)), "dates": f"{tr.e_session.min():%Y-%m-%d}..{tr.e_session.max():%Y-%m-%d}",
         "weeks": int(tr.week.nunique()), "roi": float(x.mean()),
         "roi_lo": float(np.percentile(m, 2.5)), "roi_hi": float(np.percentile(m, 97.5)),
         "p": float((m <= 0).mean()), "se_cluster": float(m.std()), "sd": float(x.std(ddof=1)),
         "median": float(np.median(x)), "win_rate": float((x > 0).mean()),
         "half1": float(x[:h].mean()), "half2": float(x[h:].mean()),
         "roi_mid": float(tr.roi_mid.mean()), "roi_mid_lo": float(np.percentile(mm, 2.5)),
         "roi_mid_hi": float(np.percentile(mm, 97.5)), "rel_spread_med": float(tr.rel_spread0.median()),
         "worst_week": float(wk.min()), "worst_week_at": str(wk.idxmin()), "worst_event": float(x.min()),
         "median_cost_usd": float(tr.cost.median() * 100), "max_events_per_week": int(tr.week.value_counts().max())}
    r["passed"] = bool(r["p"] < BAR and r["half1"] > 0 and r["half2"] > 0)
    r["too_good"] = bool(r["roi"] > 0.10 and r["n"] > 200)
    return r


def cmd_score(split):
    tr, drops = build_trades(split)
    RES.mkdir(parents=True, exist_ok=True)
    tr.to_csv(RES / f"trades_{split}.csv", index=False)
    print("drops", drops, "trades", len(tr), "stale exits", int(tr.stale_exit.sum()),
          "strike != unconstrained ATM", int((tr.strike != tr.k_free).sum()), "P2 n", int(tr.p2.sum()))
    print("k_dist pct 50/90/max", np.round(np.percentile(tr.k_dist, [50, 90, 100]), 4),
          "hold days median", (tr.exit - tr.entry).dt.days.median(),
          "DTE at entry median", (pd.to_datetime(tr.expiry) - tr.entry).dt.days.median())
    yr = tr.e_session.dt.year
    print("median rel spread entry/exit by year:", tr.groupby(yr).rel_spread0.median().round(3).to_dict(),
          tr.groupby(yr).rel_spread1.median().round(3).to_dict())
    print("by year ROI natural:", tr.groupby(yr).roi.mean().round(4).to_dict())
    print("by year ROI mid:    ", tr.groupby(yr).roi_mid.mean().round(4).to_dict())
    print("timing ROI nat/mid:", tr.groupby("timing")[["roi", "roi_mid"]].mean().round(4).to_dict())
    res = {}
    for k, sub in (("P1", tr), ("P2", tr[tr.p2])):
        r = summarize(sub)
        res[k] = r
        print(k, split, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()})
    (RES / f"results_{split}.json").write_text(json.dumps({"drops": drops, **res}, indent=1, default=str))
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for j, (k, r) in enumerate(res.items(), 1):
                f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    "strategy": f"l11_earnings_straddle_{k}",
                                    "sha": hashlib.sha256(f"l11:{k}".encode()).hexdigest()[:12],
                                    "dataset": "l11_dolthub_single_name_options", "split": split,
                                    "hypothesis": "pre-registered in notes/l11_earnings_straddle.md (commit 3e081d1)",
                                    "source": "DoltHub post-no-preference/options + yfinance/EDGAR 8-K 2.02 dates",
                                    "status": "ok", "trial_number": j, "study_trials": K_TRIALS,
                                    "bonferroni_bar": BAR, "price": "natural bid/ask, $0.03/contract/leg/side",
                                    **r}, default=str) + "\n")
        print("appended to ledger")


if __name__ == "__main__":
    globals()["cmd_" + sys.argv[1]](*sys.argv[2:3])
