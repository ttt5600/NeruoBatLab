"""AU: Kalshi KXALBUMEQUIV (first-week album units) vs HitsDailyDouble midweek-20 / hits-top-50 vintages.
Prereg: notes/consensus_reject_census.md (AMENDMENT 1). Read-only; Kalshi via scripts/kx_throttle.py.

  album_units.py build    -> data/consensus_reject_census/au_frame.parquet (one row per market x stage, inputs only + hidden result)
  album_units.py gate     -> data gate on DEV weeks (reads settlement)
  album_units.py candles  -> hourly candles around T for selected rungs
  album_units.py trades   -> trade tape after T for selected rungs (maker)
  album_units.py score SPLIT -> dev|sealed (sealed refused unless AU_SEALED_OK=1)
"""
from __future__ import annotations

import json, math, os, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kx_throttle import kget  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "consensus_reject_census"
R = ROOT / "results" / "consensus_reject_census"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
DEV = (pd.Timestamp("2026-05-01"), pd.Timestamp("2026-08-31"))
SEALED = (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-11-08"))
M_LATE, M_EARLY = 1.15, 1.30


def unix(t) -> int:
    return int((pd.Timestamp(t) - EPOCH) // pd.Timedelta(seconds=1))


def norm(s: str) -> str:
    s = s.upper().replace("&", " AND ")
    s = re.sub(r"\((DELUXE|EXPANDED|ANNIVERSARY|REISSUE|[^)]*EDITION)[^)]*\)", " ", s)
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def hdd_charts() -> pd.DataFrame:
    d = json.load(open(D / "hdd_charts_all.json"))
    rows = []
    for c in d:
        if c.get("type") not in ("midweek-20", "hits-top-50"):
            continue
        cr, up = pd.Timestamp(c["_createdAt"]), pd.Timestamp(c["_updatedAt"])
        crd = cr.tz_convert("America/New_York").normalize().tz_localize(None)
        if c["type"] == "midweek-20":
            wend = crd + pd.Timedelta(days=(3 - crd.weekday()) % 7)
        else:
            back = (crd.weekday() - 3) % 7 or 7
            wend = crd - pd.Timedelta(days=back)
        ents = []
        for ln in (c.get("chart_data") or "").splitlines():
            parts = [p.strip() for p in ln.split("\t")]
            if len(parts) < 5 or "|" not in parts[2]:
                continue
            artist, title = [x.strip() for x in parts[2].split("|", 1)]
            try:
                units = float(parts[4].replace(",", ""))
            except ValueError:
                continue
            ents.append((artist, title, units))
        rows.append(dict(type=c["type"], cr=cr, up=up, wend=wend, n=len(ents), ents=ents,
                         raw=norm(c.get("chart_data") or "")))
    return pd.DataFrame(rows)


def kalshi_markets() -> pd.DataFrame:
    m = pd.DataFrame(json.load(open(R / "mk_KXALBUMEQUIV.json")))
    src = set(pd.DataFrame(json.load(open(R / "mk_KXALBUMEQUIV.json")))["ticker"])  # noqa
    m["K"] = m["floor_strike"].astype(float)
    pr = m["rules_primary"].str.extract(r"^If (.*) by (.*?)\s+has above")
    m["title"], m["artist"] = pr[0], pr[1]
    m["wend"] = pd.to_datetime(m["rules_primary"].str.extract(r"-\s*([A-Za-z]+ \d{1,2}, \d{4})\s*,?\s*tracking week")[0],
                               format="%B %d, %Y")
    m["close_time"] = pd.to_datetime(m["close_time"], utc=True)
    m["open_time"] = pd.to_datetime(m["open_time"], utc=True)
    return m


def build() -> int:
    ch = hdd_charts()
    m = kalshi_markets()
    print("markets", len(m), "events", m.event_ticker.nunique(), "wend NaT", m.wend.isna().sum(),
          "title NaN", m.title.isna().sum())
    out = []
    for _, r in m.iterrows():
        sat = r.wend + pd.Timedelta(days=2, hours=12)       # Saturday 12:00Z after the Thursday week end
        wed = r.wend - pd.Timedelta(days=1) + pd.Timedelta(hours=12)  # Wednesday 12:00Z inside the week
        for stage, typ, T, full in (("late", "hits-top-50", sat.tz_localize("UTC"), 50),
                                    ("early", "midweek-20", wed.tz_localize("UTC"), 20)):
            c = ch[(ch.type == typ) & (ch.wend == r.wend)]
            row = dict(ticker=r.ticker, event_ticker=r.event_ticker, wend=r.wend, K=r.K, stage=stage, T=T,
                       close_time=r.close_time, open_time=r.open_time, title=r.title, artist=r.artist,
                       F=np.nan, U=np.nan, status="nodoc", doc_up=pd.NaT, result=r.result,
                       expiration_value=r.expiration_value)
            if len(c):
                c = c.iloc[0]
                row["doc_up"] = c.up
                if c.up >= T:
                    row["status"] = "edited_after_T"
                else:
                    nt, na = norm(r.title or ""), norm(r.artist or "")
                    hit = [u for a, t, u in c.ents if norm(t) == nt and (norm(a) == na or na in norm(a) or norm(a) in na)]
                    if not hit:
                        hit = [u for a, t, u in c.ents if norm(t) == nt]
                    if hit:
                        row["F"], row["status"] = max(hit), "present"
                    elif c.n >= full and nt and (f" {nt} " not in f" {c.raw} ") and (f" {na} " not in f" {c.raw} "):
                        row["U"], row["status"] = min(u for _, _, u in c.ents), "absent"
                    else:
                        row["status"] = "ambiguous"
            out.append(row)
    f = pd.DataFrame(out)
    f.to_parquet(D / "au_frame.parquet")
    print(f.groupby(["stage", "status"]).size())
    return 0


def units_of(v):
    s = str(v).replace(",", "").strip().upper()
    mult = 1000.0 if s.endswith("K") else 1e6 if s.endswith("M") else 1.0
    try:
        return float(s.rstrip("KM")) * mult
    except ValueError:
        return np.nan


def event_start(wend):
    """Friday 04:00Z after the Thursday week end (~midnight ET): the week's sales are complete from then."""
    return pd.Timestamp(wend).tz_localize("UTC") + pd.Timedelta(days=1, hours=4)


def split_of(w):
    return "dev" if DEV[0] <= w <= DEV[1] else "sealed" if SEALED[0] <= w <= SEALED[1] else "fwd" if w > SEALED[1] else "pre"


def gate() -> int:
    f = pd.read_parquet(D / "au_frame.parquet")
    f["split"] = f.wend.map(split_of)
    f = f[f.split == "dev"]
    f["units"] = f.expiration_value.map(units_of)
    e = f.groupby(["event_ticker", "stage"]).first().reset_index()
    for st, tol, need in (("late", .10, .90), ("early", .30, .80)):
        x = e[(e.stage == st) & (e.status == "present")]
        err = (x.F - x.units).abs() / x.units
        a = e[(e.stage == st) & (e.status == "absent")]
        viol = (a.units > a.U)
        print(f"{st}: events {len(e[e.stage == st])} present {len(x)} within {tol:.0%}: {(err <= tol).mean():.3f} "
              f"(need {need}); median |err| {err.median():.3f}; absent {len(a)} bound violated {viol.mean() if len(a) else float('nan'):.3f}")
        print("   status:", e[e.stage == st].status.value_counts().to_dict())
        print(x.assign(err=err)[["event_ticker", "title", "F", "units", "err"]].sort_values("err").tail(6).to_string())
        if len(a):
            print(a[["event_ticker", "title", "U", "units"]].sort_values("units").tail(5).to_string())
    return 0


def selected(f: pd.DataFrame) -> pd.DataFrame:
    mg = np.where(f.stage == "late", M_LATE, M_EARLY)
    no = ((f.status == "present") & (f.K >= mg * f.F)) | ((f.status == "absent") & (f.K >= mg * f.U))
    yes = (f.status == "present") & (f.K <= f.F / mg)
    f = f.assign(side=np.where(no, "no", np.where(yes, "yes", "")))
    return f[(f.side != "") & (f.open_time < f["T"]) & (f.close_time > f["T"])].copy()


def candles() -> int:
    f = selected(pd.read_parquet(D / "au_frame.parquet"))
    f = f[(f.stage == "early") & f.wend.map(lambda w: split_of(w) == "dev")]  # late trials died at the data gate; sealed unopened
    hist = set(m["ticker"] for m in json.load(open(R / "mk_KXALBUMEQUIV.json")) if m.get("_hist"))
    cache = D / "au_candles.json"
    out = json.loads(cache.read_text()) if cache.exists() else {}
    for (tk, T), _ in f.groupby(["ticker", "T"]):
        key = f"{tk}|{unix(T)}"
        if key in out:
            continue
        p = {"start_ts": unix(T) - 2 * 86400, "end_ts": unix(event_start(_["wend"].iloc[0])) + 3600, "period_interval": 60}
        r = kget(f"/series/KXALBUMEQUIV/markets/{tk}/candlesticks", p)
        cs = (r or {}).get("candlesticks", [])
        if not cs:
            r = kget(f"/historical/markets/{tk}/candlesticks", p)
            cs = (r or {}).get("candlesticks", [])
        out[key] = cs
        if len(out) % 25 == 0:
            cache.write_text(json.dumps(out))
    cache.write_text(json.dumps(out))
    print("keys", len(out), "with candles", sum(bool(v) for v in out.values()))
    return 0


def trades() -> int:
    f = selected(pd.read_parquet(D / "au_frame.parquet"))
    f = f[f.stage == "late"]
    cache = D / "au_trades.json"
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


def _px(c, side):
    s = c.get(side) or {}
    v = s.get("close_dollars", s.get("close"))
    return np.nan if v is None else float(v)


def fee(p):
    return math.ceil(round(0.07 * p * (1 - p) * 100, 6)) / 100


def _yp(t):
    v = float(t.get("yes_price_dollars", t.get("yes_price")))
    return v / 100 if v > 1.5 else v


def boot(pnl, cost, cl, n=5000, seed=1):
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"p": pnl, "c": cost, "k": cl}).groupby("k").sum()
    P, C = g.p.to_numpy(), g.c.to_numpy()
    idx = rng.integers(0, len(g), (n, len(g)))
    r = P[idx].sum(1) / C[idx].sum(1)
    return np.percentile(r, [2.5, 97.5]), (r <= 0).mean()


def score(split: str) -> int:
    if split == "sealed" and os.environ.get("AU_SEALED_OK") != "1":
        print("sealed refused (user yes required)"); return 2
    f = selected(pd.read_parquet(D / "au_frame.parquet"))
    f = f[f.wend.map(split_of) == split]
    cs = json.loads((D / "au_candles.json").read_text())
    trd = json.loads((D / "au_trades.json").read_text()) if (D / "au_trades.json").exists() else {}
    rows = []
    for _, r in f.iterrows():
        c = [k for k in cs.get(f"{r.ticker}|{unix(r['T'])}", []) if k["end_period_ts"] <= unix(r["T"])]
        yb, ya = (_px(c[-1], "yes_bid"), _px(c[-1], "yes_ask")) if c else (np.nan, np.nan)
        if r.side == "no":
            ask, bid = (1 - yb if yb == yb else np.nan), (1 - ya if ya == ya else np.nan)
        else:
            ask, bid = ya, yb
        won = (r.result == r.side)
        rows.append(dict(r, ask=ask, bid=bid, won=won))
    x = pd.DataFrame(rows)
    x.to_parquet(R / f"au_scored_{split}.parquet")
    print(f"{split}: selected rungs {len(x)} events {x.event_ticker.nunique()} weeks {x.wend.nunique()}; by stage/side:",
          x.groupby(["stage", "side"]).size().to_dict())
    print("  all selected rungs: loss rate", round(1 - x.won.mean(), 4), "losses", int((~x.won).sum()))
    for st, name in (("late", "AU-L-T"), ("early", "AU-E-T")):
        t = x[(x.stage == st) & (x.ask > 0) & (x.ask <= 0.98)].copy()
        if t.empty:
            print(name, "n 0"); continue
        t["cost"] = t.ask + t.ask.map(fee)
        t["pnl"] = t.won.astype(float) - t.cost
        ci, p = boot(t.pnl, t.cost, t.wend)
        print(f"{name} taker: n {len(t)} weeks {t.wend.nunique()} losses {(~t.won).sum()} ROI {t.pnl.sum() / t.cost.sum():+.4f} "
              f"CI [{ci[0]:+.4f},{ci[1]:+.4f}] p_boot {p:.4f} mean ask {t.ask.mean():.3f}")
        t.to_parquet(R / f"au_{name}_{split}.parquet")
    # maker (late)
    t = x[(x.stage == "late") & x.ask.notna() & x.bid.notna()].copy()
    mk = []
    for _, r in t.iterrows():
        q = math.floor(round(min((r.ask + r.bid) / 2, r.ask - 0.01) * 100, 6)) / 100
        if not (0.50 <= q <= 0.99):
            continue
        filled = False
        for k in sorted(trd.get(r.ticker, []), key=lambda k: k["created_time"]):
            ct = pd.Timestamp(k["created_time"])
            if ct <= r["T"] or ct > r.close_time:
                continue
            yp = _yp(k)
            if r.side == "no" and k["taker_side"] == "yes" and yp > 1 - q + 1e-9:
                filled = True; break
            if r.side == "yes" and k["taker_side"] == "no" and yp < q - 1e-9:
                filled = True; break
        mk.append(dict(r, q=q, filled=filled))
    mk = pd.DataFrame(mk)
    if len(mk):
        fl = mk[mk.filled].copy()
        print(f"AU-L-M maker: posted {len(mk)} filled {len(fl)}; loss rate filled {1 - fl.won.mean() if len(fl) else float('nan'):.3f} "
              f"vs unfilled {1 - mk[~mk.filled].won.mean() if (~mk.filled).any() else float('nan'):.3f}")
        if len(fl):
            fl["pnl"] = fl.won.astype(float) - fl.q
            ci, p = boot(fl.pnl, fl.q, fl.wend)
            print(f"   ROI {fl.pnl.sum() / fl.q.sum():+.4f} CI [{ci[0]:+.4f},{ci[1]:+.4f}] p_boot {p:.4f} n {len(fl)} weeks {fl.wend.nunique()}")
        mk.to_parquet(R / f"au_AU-L-M_{split}.parquet")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit({"build": build, "gate": gate, "candles": candles, "trades": trades}[a[0]]() if a[0] != "score" else score(a[1]))
