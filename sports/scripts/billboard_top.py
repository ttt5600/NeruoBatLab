"""TA: Kalshi KXTOPALBUM (Billboard 200 #1) vs HitsDailyDouble midweek-20 / hits-top-50 vintages.
Prereg: notes/billboard_top.md. Read-only; Kalshi via scripts/kx_throttle.py.

  billboard_top.py build    -> data/billboard_top/ta_frame.parquet (one row per market x stage; inputs + hidden result)
  billboard_top.py gate     -> HDD #1 == Kalshi winner on DEV weeks (reads winner)
  billboard_top.py candles  -> hourly candles around T for selected markets (dev)
  billboard_top.py trades   -> trade tape after T for TA-E-M (dev)
  billboard_top.py score SPLIT
"""
from __future__ import annotations

import json, math, os, re, sys, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kx_throttle import kget  # noqa: E402
from album_units import norm as _norm, unix, fee, boot, _px, _yp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "billboard_top"
MK = ROOT / "results" / "consensus_reject_census" / "mk_KXTOPALBUM.json"
R = ROOT / "results" / "billboard_top"
DEV = (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-08-31"))
SEALED = (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-11-08"))
REF = {"fc35fa78-5176-4b8f-bd5f-561735cab75d": "midweek-20", "7c58ec0a-d3e8-473c-a6ef-ec4259321ebc": "hits-top-50"}
M = {"early": 1.30, "late": 1.15}
ASK_MAX = 0.98


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[\u2019'`]", "", s.replace("$", "S"))   # AMENDMENT 1: MAN'S -> MANS, $OME -> SOME
    return _norm(s)


def same(a: str, t: str, k: str) -> bool:
    """AMENDMENT 1 match: HDD (artist a, title t) vs Kalshi title k (all normalized)."""
    return k in (t, f"{a} {t}") or (t in ("SOUNDTRACK", "SOUNDTRACK SOUNDTRACK", "ORIGINAL SOUNDTRACK") and k.startswith(a))


def near(t: str, k: str) -> bool:
    """possible alias (substring either way, >= 4 chars) -> never treat as absent."""
    return min(len(t), len(k)) >= 4 and (f" {t} " in f" {k} " or f" {k} " in f" {t} ")


def split_of(w):
    return "dev" if DEV[0] <= w <= DEV[1] else "sealed" if SEALED[0] <= w <= SEALED[1] else "fwd" if w > SEALED[1] else "pre"


def _num(s):
    s = s.replace(",", "").replace("$", "").strip()
    return float(s) if re.fullmatch(r"\d+(\.\d+)?", s) else None


def hdd_docs() -> pd.DataFrame:
    rows = []
    for c in json.load(open(D / "hdd_chart_docs_all.json")):
        typ = REF.get((c.get("chart_type") or {}).get("_ref"))
        if not typ:
            continue
        cr, up = pd.Timestamp(c["_createdAt"]), pd.Timestamp(c["_updatedAt"])
        crd = cr.tz_convert("America/New_York").normalize().tz_localize(None)
        if typ == "midweek-20":
            wend = crd + pd.Timedelta(days=(3 - crd.weekday()) % 7)
        else:
            wend = crd - pd.Timedelta(days=(crd.weekday() - 3) % 7 or 7)
        ents = []
        for ln in (c.get("chart_data") or "").splitlines():
            parts = [p.strip() for p in ln.split("\t")]
            i = next((j for j, p in enumerate(parts) if "|" in p), None)
            if i is None:
                continue
            u = next((_num(p) for p in parts[i + 1:] if _num(p) is not None), None)
            if u is None:
                continue
            a, t = [x.strip() for x in parts[i].split("|", 1)]
            ents.append((a, t, u))
        ents.sort(key=lambda e: -e[2])
        rows.append(dict(type=typ, cr=cr, up=up, wend=wend, n=len(ents), ents=ents, raw=norm(c.get("chart_data") or "")))
    d = pd.DataFrame(rows)
    # one doc per (type, week): if duplicates, keep the earliest-created (conservative); flag count
    d = d.sort_values("cr")
    print("docs", d.groupby("type").size().to_dict(), "dup weeks", int(d.duplicated(["type", "wend"]).sum()))
    return d.drop_duplicates(["type", "wend"], keep="first")


def markets() -> pd.DataFrame:
    m = pd.DataFrame(json.load(open(MK)))
    m["chart"] = pd.to_datetime(m.event_ticker.str.extract(r"-(\d{2}[A-Z]{3}\d{2})")[0], format="%y%b%d")
    m["wend"] = m.chart - pd.Timedelta(days=9)
    for c in ("close_time", "open_time"):
        m[c] = pd.to_datetime(m[c], utc=True, format="ISO8601")
    return m


def build() -> int:
    docs, m = hdd_docs(), markets()
    m = m[m.chart.notna()]
    out = []
    for ev, g in m.groupby("event_ticker"):
        wend = g.wend.iloc[0]
        for stage, typ, T, full in (("early", "midweek-20", (wend - pd.Timedelta(days=1, hours=-12)).tz_localize("UTC"), 20),
                                    ("late", "hits-top-50", (wend + pd.Timedelta(days=2, hours=12)).tz_localize("UTC"), 50)):
            c = docs[(docs.type == typ) & (docs.wend == wend)]
            base = dict(event_ticker=ev, wend=wend, stage=stage, T=T, doc_up=pd.NaT, FL=np.nan, F2=np.nan, leader="",
                        leader_artist="")
            st0 = "nodoc"
            if len(c):
                c = c.iloc[0]
                base.update(doc_up=c.up)
                st0 = "edited_after_T" if c.up >= T else ("short" if c.n < 2 else "ok")
                if st0 == "ok":
                    base.update(FL=c.ents[0][2], F2=c.ents[1][2], leader=norm(c.ents[0][1]),
                                leader_artist=norm(c.ents[0][0]))
            for _, r in g.iterrows():
                row = dict(base, ticker=r.ticker, title=r.yes_sub_title, open_time=r.open_time, close_time=r.close_time,
                           F=np.nan, U=np.nan, status=st0, result=r.result)
                if st0 == "ok":
                    nt = norm(r.yes_sub_title)
                    hit = [u for a, t, u in c.ents if same(norm(a), norm(t), nt)]
                    if hit:
                        row.update(F=max(hit), status="present")
                    elif c.n >= full and nt and f" {nt} " not in f" {c.raw} " and \
                            not any(near(norm(t), nt) or near(norm(a), nt) for a, t, u in c.ents):
                        row.update(U=c.ents[-1][2], status="absent")
                    else:
                        row["status"] = "ambiguous"
                out.append(row)
    f = pd.DataFrame(out)
    f["split"] = f.wend.map(split_of)
    f.to_parquet(D / "ta_frame.parquet")
    print(f.drop_duplicates(["event_ticker", "stage"]).groupby(["split", "stage", "status"]).size().unstack(fill_value=0)
          if False else f.groupby(["split", "stage", "status"]).size().unstack(fill_value=0))
    e = f.drop_duplicates(["event_ticker", "stage"])
    print("events with usable doc:", e[e.FL.notna()].groupby(["split", "stage"]).size().to_dict())
    return 0


def selected(f: pd.DataFrame) -> pd.DataFrame:
    f = f.assign(isL=norm_eq(f) & (f.status == "present"))
    f = f[(f.open_time < f["T"]) & (f.close_time > f["T"])]
    # AMENDMENT 1: no bets in a week whose HDD leader matches no Kalshi market open at T
    f = f[f.groupby(["event_ticker", "stage"]).isL.transform("any")]
    isL = f.isL
    mg = f.stage.map(M)
    yes = isL & (f.FL >= mg * f.F2)
    no = (~isL) & (((f.status == "present") & (f.F * mg <= f.FL)) | ((f.status == "absent") & (f.FL >= mg * f.U)))
    f = f.assign(side=np.where(yes, "yes", np.where(no, "no", "")))
    return f[(f.side != "") & (f.open_time < f["T"]) & (f.close_time > f["T"])].copy()


def norm_eq(f):
    return pd.Series([same(a, t, norm(k)) for a, t, k in zip(f.leader_artist, f.leader, f.title)], index=f.index)


def gate() -> int:
    f = pd.read_parquet(D / "ta_frame.parquet")
    f = f[(f.split == "dev") & f.FL.notna()]
    for st, need in (("early", .70), ("late", .85)):
        e = f[f.stage == st]
        w = e[e.result == "yes"].groupby("event_ticker").title.first().map(norm)
        L = e.groupby("event_ticker").leader.first()
        e = e.assign(isL=norm_eq(e).to_numpy())
        has_l = e.groupby("event_ticker").isL.any()
        wl = e[e.result == "yes"].groupby("event_ticker").isL.first()
        j = pd.DataFrame({"L": L, "W": w, "listed": has_l, "hit": wl}).dropna(subset=["W"])
        hit = j.hit.astype(bool)
        print(f"{st}: usable weeks {len(j)}; HDD #1 == winner {hit.mean():.3f} (need {need}); leader listed on Kalshi {j.listed.mean():.3f}")
        print("   misses:", j[~hit].head(12).to_dict("index"))
    return 0


def candles() -> int:
    f = selected(pd.read_parquet(D / "ta_frame.parquet"))
    f = f[f.split == "dev"]
    cache = D / "ta_candles.json"
    out = json.loads(cache.read_text()) if cache.exists() else {}
    for (tk, T), g in f.groupby(["ticker", "T"]):
        key = f"{tk}|{unix(T)}"
        if key in out:
            continue
        end = unix(g.close_time.iloc[0])
        p = {"start_ts": unix(T) - 2 * 86400, "end_ts": end, "period_interval": 60}
        cs = ((kget(f"/series/KXTOPALBUM/markets/{tk}/candlesticks", p) or {}).get("candlesticks", []))
        if not cs:
            cs = ((kget(f"/historical/markets/{tk}/candlesticks", p) or {}).get("candlesticks", []))
        out[key] = cs
        if len(out) % 25 == 0:
            cache.write_text(json.dumps(out))
    cache.write_text(json.dumps(out))
    print("keys", len(out), "with candles", sum(bool(v) for v in out.values()))
    return 0


def trades() -> int:
    f = selected(pd.read_parquet(D / "ta_frame.parquet"))
    f = f[(f.split == "dev") & (f.stage == "early")]
    cache = D / "ta_trades.json"
    out = json.loads(cache.read_text()) if cache.exists() else {}
    for tk, g in f.groupby("ticker"):
        if tk in out:
            continue
        tr = []
        for ep in ("/markets/trades", "/historical/trades"):
            cur = None
            while True:
                p = {"ticker": tk, "min_ts": unix(g["T"].min()), "limit": 1000}
                if cur:
                    p["cursor"] = cur
                r = kget(ep, p) or {}
                tr += r.get("trades", [])
                cur = r.get("cursor")
                if not cur or not r.get("trades"):
                    break
            if tr:
                break
        out[tk] = tr
    cache.write_text(json.dumps(out))
    print("tickers", len(out), "trades", sum(len(v) for v in out.values()))
    return 0


def quote(cs, ts):
    c = [k for k in cs if k["end_period_ts"] <= ts]
    return (_px(c[-1], "yes_bid"), _px(c[-1], "yes_ask")) if c else (np.nan, np.nan)


def priced(split: str) -> pd.DataFrame:
    f = selected(pd.read_parquet(D / "ta_frame.parquet"))
    f = f[f.split == split]
    cs = json.loads((D / "ta_candles.json").read_text())
    rows = []
    for _, r in f.iterrows():
        c = cs.get(f"{r.ticker}|{unix(r['T'])}", [])
        yb, ya = quote(c, unix(r["T"]))
        sun = (r.wend + pd.Timedelta(days=3, hours=12)).tz_localize("UTC")  # Sunday 12:00Z = event_start, before the reveal
        cb, ca = quote(c, unix(sun))
        if r.side == "no":
            ask, bid, cl = 1 - yb, 1 - ya, 1 - cb
        else:
            ask, bid, cl = ya, yb, ca
        rows.append(dict(r, ask=ask, bid=bid, close_ask=cl, won=(r.result == r.side)))
    return pd.DataFrame(rows)


def score(split: str) -> int:
    if split == "sealed" and os.environ.get("TA_SEALED_OK") != "1":
        print("sealed refused (user yes required)"); return 2
    x = priced(split)
    x.to_parquet(R / f"ta_scored_{split}.parquet")
    for st in ("early", "late"):
        s = x[x.stage == st]
        print(f"{st}: selected {len(s)} weeks {s.wend.nunique()} by side {s.side.value_counts().to_dict()} "
              f"losses {(~s.won).sum()} | ask quoted {s.ask.notna().sum()}, ask<=.98 {(s.ask <= ASK_MAX).sum()}, "
              f">.99 {(s.ask > .99).sum()}")
    for st, name in (("early", "TA-E-T"), ("late", "TA-L-T")):
        t = x[(x.stage == st) & (x.ask > 0) & (x.ask <= ASK_MAX)].copy()
        if t.empty:
            print(name, "n 0"); continue
        t["cost"] = t.ask + t.ask.map(fee)
        t["pnl"] = t.won.astype(float) - t.cost
        ci, p = boot(t.pnl, t.cost, t.wend)
        print(f"{name}: n {len(t)} weeks {t.wend.nunique()} sides {t.side.value_counts().to_dict()} losses {(~t.won).sum()} "
              f"ROI {t.pnl.sum() / t.cost.sum():+.4f} CI [{ci[0]:+.4f},{ci[1]:+.4f}] p_boot {p:.4f} mean ask {t.ask.mean():.3f} "
              f"CLV {(t.close_ask - t.ask).mean():+.4f}")
        t.to_parquet(R / f"ta_{name}_{split}.parquet")
    trd = json.loads((D / "ta_trades.json").read_text()) if (D / "ta_trades.json").exists() else {}
    t = x[(x.stage == "early") & x.ask.notna() & x.bid.notna()]
    mk = []
    for _, r in t.iterrows():
        q = math.floor(round(min((r.ask + r.bid) / 2, r.ask - 0.01) * 100, 6)) / 100
        if not (0.50 <= q <= 0.98):
            continue
        filled = False
        for k in sorted(trd.get(r.ticker, []), key=lambda k: k["created_time"]):
            ct = pd.Timestamp(k["created_time"])
            if ct <= r["T"] or ct > r.close_time:
                continue
            yp = _yp(k)
            if (r.side == "no" and k["taker_side"] == "yes" and yp > 1 - q + 1e-9) or \
               (r.side == "yes" and k["taker_side"] == "no" and yp < q - 1e-9):
                filled = True; break
        mk.append(dict(r, q=q, filled=filled))
    mk = pd.DataFrame(mk)
    if len(mk):
        fl = mk[mk.filled].copy()
        print(f"TA-E-M: posted {len(mk)} filled {len(fl)}; loss rate filled "
              f"{1 - fl.won.mean() if len(fl) else float('nan'):.3f} vs unfilled "
              f"{1 - mk[~mk.filled].won.mean() if (~mk.filled).any() else float('nan'):.3f}")
        if len(fl):
            fl["pnl"] = fl.won.astype(float) - fl.q
            ci, p = boot(fl.pnl, fl.q, fl.wend)
            print(f"   ROI {fl.pnl.sum() / fl.q.sum():+.4f} CI [{ci[0]:+.4f},{ci[1]:+.4f}] p_boot {p:.4f} n {len(fl)} weeks {fl.wend.nunique()}")
        mk.to_parquet(R / f"ta_TA-E-M_{split}.parquet")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(score(a[1]) if a[0] == "score" else {"build": build, "gate": gate, "candles": candles, "trades": trades}[a[0]]())
