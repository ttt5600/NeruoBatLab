"""Q009 SPY short-vol (defined risk) after natural bid/ask. Pre-registered in notes/q009_options_vrp.md (commit 2421dcf).

  ../.venv_quant/bin/python -I scripts/q009_options_vrp.py audit            # dev rows only
  ../.venv_quant/bin/python -I scripts/q009_options_vrp.py dev [--ledger]
  ../.venv_quant/bin/python -I scripts/q009_options_vrp.py sealed --trials T1 [--ledger]   # only if dev passed

T1 put credit spread, T2 iron condor; short |delta| closest to .16, expiry DTE in [21,70] closest to 45, wing ~2% of
S_prev, sell at bid / buy at ask, $0.03/contract/leg, held to expiry, settled on Yahoo unadjusted close.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOLT = ROOT / "data" / "options" / "dolthub"
SPYF = ROOT / "data" / "options" / "yahoo_spy" / "spy_daily.parquet"
OUT = ROOT / "results" / "q009_options_vrp"
LEDGER = ROOT / "lab" / "ledger.jsonl"
FEE = 0.03 / 100          # per share-equivalent per leg
TARGET_DELTA, DTE_LO, DTE_HI, DTE_TGT = 0.16, 21, 70, 45
WING, WMIN, WMAX = 0.02, 0.01, 0.04
BLOCK, NBOOT, ALPHA = 8, 10000, 0.05 / 2


def weeks_and_cutoff():
    log = pd.read_csv(DOLT / "fetch_log_weekly_SPY.csv")
    ok = log[log.n > 0].groupby("week").date.min().sort_index()
    fails = log[log.n < 0]
    weeks = list(ok.index)
    cut = weeks[int(np.ceil(0.7 * len(weeks)))]          # first sealed week
    return weeks, cut, ok, fails


def load(split):
    weeks, cut, ok, _ = weeks_and_cutoff()
    use = [w for w in weeks if (w < cut if split == "dev" else w >= cut)]
    df = pd.concat([pd.read_parquet(DOLT / f"SPY_wk_{w}.parquet") for w in use], ignore_index=True)
    for c in ["strike", "bid", "ask", "vol", "delta"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    return df, cut


def spy():
    s = pd.read_parquet(SPYF)
    s.index = pd.to_datetime(s.index)
    return s


def audit(df, px):
    n = len(df)
    print(f"rows {n}, snapshot dates {df.date.nunique()}, {df.date.min():%Y-%m-%d}..{df.date.max():%Y-%m-%d}")
    print("weekday of entry snapshots:", df.drop_duplicates("date").date.dt.day_name().value_counts().to_dict())
    print("expirations/date:", df.groupby("date").expiration.nunique().value_counts().sort_index().to_dict())
    print("strikes per (date,exp,type) median:", df.groupby(["date", "expiration", "call_put"]).size().median())
    print(f"crossed ask<bid {(df.ask < df.bid).sum()}, bid==0 {(df.bid == 0).sum()}, bid==ask {(df.bid == df.ask).sum()}, "
          f"NaN bid/ask {df[['bid','ask']].isna().any(axis=1).sum()}, exp<=date {(df.expiration <= df.date).sum()}, "
          f"dup keys {df.duplicated(['date','expiration','strike','call_put']).sum()}")
    bad_sign = ((df.call_put == "Put") & (df.delta > 0)) | ((df.call_put == "Call") & (df.delta < 0))
    print(f"delta sign wrong {bad_sign.sum()}, call_put values {df.call_put.value_counts().to_dict()}")
    rs = (df.ask - df.bid) / ((df.ask + df.bid) / 2)
    print("rel spread quantiles (bid>0):", rs[df.bid > 0].quantile([.1, .5, .9]).round(3).to_dict())
    # implied spot via put-call parity on the nearest expiry, strike closest to the money (r, q ignored)
    rows = []
    for d, g in df.groupby("date"):
        e = g.expiration.min()
        g = g[g.expiration == e]
        c = g[g.call_put == "Call"].set_index("strike")
        p = g[g.call_put == "Put"].set_index("strike")
        k = c.index.intersection(p.index)
        if not len(k):
            continue
        both = pd.DataFrame({"c": (c.loc[k].bid + c.loc[k].ask) / 2, "p": (p.loc[k].bid + p.loc[k].ask) / 2})
        kk = (both.c - both.p).abs().idxmin()
        s_imp = kk + both.c[kk] - both.p[kk]
        if d not in px.index:
            continue
        i = px.index.get_loc(d)
        rows.append({"date": d, "s_imp": s_imp, "close": px.Close.iloc[i], "open": px.Open.iloc[i],
                     "prev_close": px.Close.iloc[i - 1], "next_open": px.Open.iloc[i + 1] if i + 1 < len(px) else np.nan})
    a = pd.DataFrame(rows)
    for col in ["prev_close", "open", "close", "next_open"]:
        print(f"  |S_implied - {col}|/S median {((a.s_imp - a[col]).abs() / a[col]).median():.5f}  "
              f"share within 0.1% {(((a.s_imp - a[col]).abs() / a[col]) < 0.001).mean():.2f}")
    return a


def usable(g):
    return g[(g.bid > 0) & (g.ask >= g.bid)]


def side(g, typ, s_prev):
    """Short ~16-delta leg + wing ~2% further OTM. Returns dict or None."""
    g = usable(g[g.call_put == typ])
    if g.empty:
        return None
    sh = g.loc[(g.delta.abs() - TARGET_DELTA).abs().idxmin()]
    otm = g[g.strike < sh.strike] if typ == "Put" else g[g.strike > sh.strike]
    if otm.empty:
        return None
    lg = otm.loc[((otm.strike - sh.strike).abs() - WING * s_prev).abs().idxmin()]
    w = abs(lg.strike - sh.strike)
    if not (WMIN * s_prev <= w <= WMAX * s_prev):
        return None
    return {"ks": sh.strike, "kl": lg.strike, "w": w, "credit": sh.bid - lg.ask, "d_s": sh.delta,
            "bid_s": sh.bid, "ask_s": sh.ask, "bid_l": lg.bid, "ask_l": lg.ask}


def settle_px(px, exp):
    s = px.Close[px.index <= exp]
    return s.iloc[-1], s.index[-1]


def build(df, px):
    out = []
    for d, g in df.groupby("date"):
        if d not in px.index:                              # holiday snapshot = stale copy, not tradeable (amendment)
            continue
        dte = (g.expiration - d).dt.days
        el = g[(dte >= DTE_LO) & (dte <= DTE_HI)]
        if el.empty:
            continue
        exps = sorted(el.expiration.unique(), key=lambda e: (abs((e - d).days - DTE_TGT), e))
        e = exps[0]
        ge = g[g.expiration == e]
        prev = px.Close[px.index < d]
        s_prev = prev.iloc[-1]
        if e > px.index.max():
            continue                                       # not yet expired
        s_t, sdate = settle_px(px, e)
        p = side(ge, "Put", s_prev)
        c = side(ge, "Call", s_prev)
        base = {"date": d, "exp": e, "dte": (e - d).days, "s_prev": s_prev, "s_T": s_t, "settle_date": sdate}
        if p and p["credit"] > 0:
            pay = max(p["ks"] - s_t, 0) - max(p["kl"] - s_t, 0)
            ml = p["w"] - p["credit"] + 2 * FEE
            pnl = p["credit"] - 2 * FEE - pay
            out.append({**base, "trial": "T1", **{f"p_{k}": v for k, v in p.items()}, "credit": p["credit"],
                        "maxloss": ml, "pnl": pnl, "roi": pnl / ml})
        if p and c and p["credit"] + c["credit"] > 0:
            cr = p["credit"] + c["credit"]
            pay = (max(p["ks"] - s_t, 0) - max(p["kl"] - s_t, 0)) + (max(s_t - c["ks"], 0) - max(s_t - c["kl"], 0))
            ml = max(p["w"], c["w"]) - cr + 4 * FEE
            pnl = cr - 4 * FEE - pay
            out.append({**base, "trial": "T2", **{f"p_{k}": v for k, v in p.items()}, **{f"c_{k}": v for k, v in c.items()},
                        "credit": cr, "maxloss": ml, "pnl": pnl, "roi": pnl / ml})
    return pd.DataFrame(out)


def block_boot(x, rng, nboot=NBOOT, block=BLOCK):
    n = len(x)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(nboot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(nboot, -1)[:, :n]
    return x[idx].mean(axis=1)


def risk(t, rng, f):
    t = t.sort_values("date")
    # realized equity path: stake fixed at entry from realized equity, P&L lands at settlement
    eq, events, open_pos = 1.0, [], []
    for r in t.itertuples():
        for o in [o for o in open_pos if o[0] <= r.date]:
            eq += o[1]
            events.append((o[0], eq))
        open_pos = [o for o in open_pos if o[0] > r.date]
        open_pos.append((r.settle_date, f * eq * r.roi))
    for o in sorted(open_pos):
        eq += o[1]
        events.append((o[0], eq))
    e = pd.Series([v for _, v in events], index=[d for d, _ in events]).sort_index()
    peak = np.maximum.accumulate(np.r_[1.0, e.values])
    mdd = float(np.max(1 - np.r_[1.0, e.values] / peak))
    month = (t.assign(m=t.settle_date.dt.to_period("M")).groupby("m").roi.sum() * f)
    x = t.roi.values
    n, nb = len(x), int(np.ceil(52 / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=(NBOOT, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(NBOOT, -1)[:, :52]
    paths = np.cumprod(1 + f * x[idx], axis=1)
    return {"f": f, "final_equity": float(eq), "max_dd": mdd, "worst_month": float(month.min()),
            "worst_month_at": str(month.idxmin()), "p_halve_52": float((paths.min(axis=1) <= 0.5).mean()),
            "median_52_growth": float(np.median(paths[:, -1]) - 1)}


def score(t, rng):
    t = t.sort_values("date").reset_index(drop=True)
    x = t.roi.values
    bm = block_boot(x, rng)
    h = len(t) // 2
    r = {"n": int(len(t)), "dates": f"{t.date.min():%Y-%m-%d}..{t.date.max():%Y-%m-%d}", "roi": float(x.mean()),
         "roi_lo": float(np.quantile(bm, .025)), "roi_hi": float(np.quantile(bm, .975)), "p": float((bm <= 0).mean()),
         "se_block": float(bm.std()), "se_iid": float(x.std(ddof=1) / np.sqrt(len(x))), "sd": float(x.std(ddof=1)),
         "half1": float(x[:h].mean()), "half2": float(x[h:].mean()), "win_rate": float((t.pnl > 0).mean()),
         "full_loss_rate": float((t.roi <= -0.99).mean()), "worst": float(x.min()),
         "credit_over_width": float((t.credit / (t.maxloss + t.credit)).mean()), "mean_dte": float(t.dte.mean()),
         "sharpe_ann": float(x.mean() / x.std(ddof=1) * np.sqrt(52))}
    r["passed"] = bool(r["p"] < ALPHA and r["half1"] > 0 and r["half2"] > 0)
    r["too_good"] = bool(r["sharpe_ann"] > 2 or (r["roi"] > 0.10 and r["n"] > 200))
    r["risk"] = [risk(t, rng, f) for f in (0.02, 0.05)]
    return r


def main():
    mode = sys.argv[1]
    px = spy()
    split = "dev" if mode in ("dev", "audit") else "sealed"
    df, cut = load(split)
    print(f"split {split}: first sealed week {cut}")
    if mode == "audit":
        a = audit(df, px)
        OUT.mkdir(parents=True, exist_ok=True)
        a.to_csv(OUT / "audit_implied_spot_dev.csv", index=False)
        return
    trials = ["T1", "T2"]
    if "--trials" in sys.argv:
        trials = sys.argv[sys.argv.index("--trials") + 1].split(",")
    t = build(df, px)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / f"trades_{split}.csv", index=False)
    rng = np.random.default_rng(9)
    res = {}
    for k in trials:
        r = score(t[t.trial == k], rng)
        res[k] = r
        print(f"{k} {split}: n {r['n']} ROI {r['roi']:+.4f} [{r['roi_lo']:+.4f},{r['roi_hi']:+.4f}] p {r['p']:.4f} "
              f"halves {r['half1']:+.4f}/{r['half2']:+.4f} win {r['win_rate']:.3f} full-loss {r['full_loss_rate']:.3f} "
              f"sd {r['sd']:.3f} SE block {r['se_block']:.4f} iid {r['se_iid']:.4f} sharpe {r['sharpe_ann']:.2f} "
              f"PASS {r['passed']} too_good {r['too_good']}")
        for rk in r["risk"]:
            print("   risk", {a: (round(b, 4) if isinstance(b, float) else b) for a, b in rk.items()})
    (OUT / f"results_{split}.json").write_text(json.dumps(res, indent=1, default=str))
    if "--ledger" in sys.argv:
        prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
        shas = {x["sha"] for x in prior if x.get("status") == "ok" and x.get("dataset") == "q009_spy_options"}
        with LEDGER.open("a") as f:
            for k, r in res.items():
                sha = hashlib.sha256(f"q009:{k}".encode()).hexdigest()[:12]
                shas.add(sha)
                f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    "strategy": f"q009_spy_{'put_spread' if k == 'T1' else 'iron_condor'}_{k}",
                                    "sha": sha, "dataset": "q009_spy_options", "split": split,
                                    "hypothesis": "pre-registered in notes/q009_options_vrp.md (commit 2421dcf)",
                                    "source": "DoltHub post-no-preference/options + Yahoo SPY close", "status": "ok",
                                    "trial_number": len(shas), "study_trials": 2, "bonferroni_bar": ALPHA,
                                    "price": "natural bid/ask, $0.03/contract/leg", **r}, default=str) + "\n")
        print("appended to ledger")


if __name__ == "__main__":
    main()
