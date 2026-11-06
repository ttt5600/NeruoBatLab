"""Builder for edgelab dataset pm_maker_no (trial pm_h2a_maker_no, registered 2026-11-06 BEFORE this ran).

    python scripts/pm_maker_no.py enum      # gamma events -> mention/culture binary Yes/No markets (dev days + sealed days)
    python scripts/pm_maker_no.py prices    # CLOB prices-history -> p_t (S - 24 h), p_close (S - 1 h)
    python scripts/pm_maker_no.py trades    # data-api /v2/trades for candidate markets -> prints after t
    python scripts/pm_maker_no.py frame     # -> data/pm_maker_no/frame.parquet + trades.parquet (edgelab loader input)
    python scripts/pm_maker_no.py diag      # DEV ONLY: fill rate, adverse selection (YES rate filled vs unfilled), capacity
    python scripts/pm_maker_no.py diag_sealed  # only after the sealed unseal is spent: same diagnostics on sealed

Windows (vault.yaml pm_maker_no, split key = UTC date of S):
  dev    = the 92 days sampled by smallmarket_calibration.cmd_pm_enum (seed 20261027) inside 2025-05-01..2026-04-30
  sealed = every day 2026-05-01..2026-10-31 (never priced before; the 10-27 census only counted markets)
Outcome (`won` = NO resolved) is written for both, but this script never prints sealed outcomes; edgelab hides `won`.
Markets resolved 50/50 (outcomePrices not 0/1) are dropped (count printed).
Public read-only data only (gamma-api, clob, data-api); no keys, no orders.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smallmarket_calibration as sm  # noqa: E402  (cached get, pm_day, pm_category, rate limits)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pm_maker_no"
OUT.mkdir(parents=True, exist_ok=True)
DATA_API = "https://data-api.polymarket.com/v2/trades"
SEAL0, SEAL1 = pd.Timestamp("2026-05-01", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC")
CATS = ("mentions", "culture_tech")
sm.RATE["data-api.polymarket.com"] = 0.15


def dev_days():
    rng = random.Random(20261027)
    days = list(pd.date_range(sm.DEV0, sm.DEV1 - pd.Timedelta(days=1), freq="D"))
    return sorted(rng.sample(days, 92))


def flatten(evs):
    rows = []
    for e in evs:
        tags = [t.get("label", "") for t in (e.get("tags") or [])]
        cat = sm.pm_category(tags)
        if cat not in CATS:
            continue
        for m in e.get("markets") or []:
            try:
                outs = json.loads(m.get("outcomes") or "[]")
                toks = json.loads(m.get("clobTokenIds") or "[]")
                op = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
            except Exception:
                continue
            if len(toks) != 2 or len(op) != 2 or [o.lower() for o in outs] != ["yes", "no"]:
                continue
            rows.append({"event_id": e["id"], "ticker": str(m["id"]), "cond": m.get("conditionId"),
                         "question": m.get("question"), "cat": cat, "token_yes": toks[0], "token_no": toks[1],
                         "end": m.get("endDate") or e.get("endDate"), "closed_time": m.get("closedTime"),
                         "game_start": m.get("gameStartTime"), "volume": float(m.get("volumeNum") or 0),
                         "op_yes": op[0], "closed": bool(m.get("closed"))})
    return rows


def cmd_enum():
    days = [(d, "dev") for d in dev_days()] + [(d, "sealed") for d in pd.date_range(SEAL0, SEAL1 - pd.Timedelta(days=1))]
    rows = []
    for i, evs in enumerate(sm.pmap(sm.pm_day, [d for d, _ in days], workers=3)):
        for r in flatten(evs):
            r["window"] = days[i][1]
            rows.append(r)
        if i % 20 == 0:
            print(f"  {i}/{len(days)} {days[i][0].date()} total {len(rows)}", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce", format="ISO8601")
    gs = pd.to_datetime(df["game_start"], utc=True, errors="coerce", format="mixed")
    df["S"] = df["end"].where(gs.isna() | (df["end"] <= gs), gs)
    df["closed_time"] = pd.to_datetime(df["closed_time"], utc=True, errors="coerce", format="mixed")
    inwin = ((df.S >= sm.DEV0) & (df.S < sm.DEV1)) | ((df.S >= SEAL0) & (df.S < SEAL1))
    df = df[inwin & df.closed & df.closed_time.notna()]
    df["won"] = np.where(df.op_yes == 0, 1.0, np.where(df.op_yes == 1, 0.0, np.nan))
    n5050 = int(df.won.isna().sum())
    df = df[df.won.notna()].drop(columns=["op_yes"])
    df.to_parquet(OUT / "markets.parquet", index=False)
    print(f"dropped 50/50 or unresolved: {n5050}")
    df["win2"] = np.where(df.S < sm.DEV1, "dev", "sealed")
    print(df.groupby(["win2", "cat"]).agg(markets=("ticker", "size"), events=("event_id", "nunique")).to_string())


def hist(r):
    end = int(r.end.timestamp())
    d = sm.get(f"{sm.CLOB}/prices-history", {"market": r.token_yes, "startTs": end - 8 * 86400, "endTs": end + 3600,
                                              "fidelity": 60})
    out = {"ticker": r.ticker, "n_pts": 0}
    if not d or not d.get("history"):
        return out
    h = d["history"]
    ts = np.array([x["t"] for x in h]); ps = np.array([x["p"] for x in h], float)
    out["n_pts"] = len(h)
    S = int(r.S.timestamp())
    cts = r.closed_time.timestamp()
    for k, hh in (("t", 24), ("close", 1)):
        t = S - hh * 3600
        if t < end - 8 * 86400 or t >= cts:
            continue
        j = np.searchsorted(ts, t, side="right") - 1
        if j >= 0 and t - ts[j] <= 6 * 3600:
            out[f"p_{k}"] = ps[j]
            out[f"p_{k}_time"] = int(ts[j])
    return out


def cmd_prices():
    df = pd.read_parquet(OUT / "markets.parquet")
    rs = list(df.itertuples())
    rows = []
    for i in range(0, len(rs), 1000):
        rows += sm.pmap(hist, rs[i:i + 1000], workers=8)
        print(f"  {min(i + 1000, len(rs))}/{len(rs)}", flush=True)
    p = pd.DataFrame(rows)
    df.merge(p, on="ticker", how="left").to_parquet(OUT / "priced.parquet", index=False)
    print("with p_t:", int(p.get("p_t", pd.Series(dtype=float)).notna().sum()))


def trades_after(r):
    """Taker prints (each fill once) of this condition with timestamp > t, newest-first pages until older than t."""
    t = int(r.S.timestamp()) - 24 * 3600
    out, cursor = [], None
    for _ in range(200):
        # the cursor alone returns an empty page; it must be sent WITH the condition (checked 11-06)
        params = {"condition": r.cond, "limit": 1000} if cursor is None else {"condition": r.cond, "cursor": cursor}
        d = sm.get(DATA_API, params, cache=True)
        if not d or not d.get("data"):
            if cursor is not None:
                INCOMPLETE.append(r.ticker)
            break
        for x in d["data"]:
            if int(x["timestamp"]) > t:
                out.append({"ticker": r.ticker, "timestamp": int(x["timestamp"]), "price": float(x["price"]),
                            "size": float(x["size"]), "yes": str(x["token_id"]) == str(r.token_yes),
                            "token_ok": str(x["token_id"]) in (str(r.token_yes), str(r.token_no))})
        oldest = min(int(x["timestamp"]) for x in d["data"])
        cursor = (d.get("pagination") or {}).get("next_cursor")
        if oldest <= t or not cursor:
            break
    else:
        INCOMPLETE.append(r.ticker)
    return out


INCOMPLETE: list = []


def cmd_trades():
    p = pd.read_parquet(OUT / "priced.parquet")
    c = p[p.p_t.between(0.015, 0.105) & p.cond.notna()]   # superset of the rule's 0.02..0.10 filter
    print(f"candidate markets for the tape: {len(c)}", flush=True)
    rs = list(c.itertuples())
    rows = []
    for i in range(0, len(rs), 500):
        for x in sm.pmap(trades_after, rs[i:i + 500], workers=4):
            rows += x
        print(f"  {min(i + 500, len(rs))}/{len(rs)} prints {len(rows)}", flush=True)
    tr = pd.DataFrame(rows, columns=["ticker", "timestamp", "price", "size", "yes", "token_ok"])
    bad = int((~tr.token_ok).sum()) if len(tr) else 0
    print(f"prints {len(tr)}; token mismatch {bad}; markets with a truncated tape: {len(set(INCOMPLETE))}")
    tr[tr.token_ok].drop(columns=["token_ok"]).to_parquet(OUT / "trades.parquet", index=False)


def cmd_frame():
    p = pd.read_parquet(OUT / "priced.parquet")
    p["t"] = p["S"] - pd.Timedelta(hours=24)
    p["p_t_time"] = pd.to_datetime(p["p_t_time"], unit="s", utc=True)
    p["date"] = p["S"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    keep = ["ticker", "event_id", "cat", "question", "S", "t", "p_t", "p_t_time", "p_close", "closed_time", "volume",
            "date", "won"]
    p[keep].to_parquet(OUT / "frame.parquet", index=False)
    # traps: timestamps sane (seconds, not ms/us) and the decision quote is never after t
    tr = pd.read_parquet(OUT / "trades.parquet")
    assert tr.timestamp.between(1.6e9, 1.9e9).all(), "trade timestamps not epoch seconds"
    ok = p.p_t.notna()
    assert (p.loc[ok, "p_t_time"] <= p.loc[ok, "t"]).all(), "p_t after decision"
    print(f"frame rows {len(p)}, with p_t {int(ok.sum())}; trades {len(tr)}")


def _diag(split):
    sys.path.insert(0, str(ROOT))
    from edgelab.rules.pm_maker_no import candidates
    from edgelab import loaders
    x, _ = loaders.pm_maker_no()
    lo, hi = (sm.DEV0, sm.DEV1) if split == "dev" else (SEAL0, SEAL1)
    S = pd.to_datetime(x["S"], utc=True)
    x = x[(S >= lo) & (S < hi)]
    x.attrs = loaders.pm_maker_no()[0].attrs
    f = candidates(x)
    f["filled"] = f.fill_time.notna()
    f["yes_won"] = 1 - f["won"]
    tr = x.attrs["trades"]
    vol = []
    for tk, ti, cl, a, q, ft in zip(f.ticker, pd.to_datetime(f.t, utc=True), pd.to_datetime(f.closed_time, utc=True),
                                    f.a, f.q, f.fill_time):
        g = tr.get(tk)
        if g is None or pd.isna(ft):
            vol.append(0.0); continue
        ts = pd.to_datetime(g.timestamp, unit="s", utc=True)
        thr = (ts > ti) & (ts <= cl) & np.where(g.yes, g.price > a + 1e-9, g.price < q - 1e-9)
        vol.append(float(g["size"][thr].sum()) if "size" in g else np.nan)
    f["through_vol"] = vol
    out = {"split": split, "posted": int(len(f)), "filled": int(f.filled.sum()), "fill_rate": float(f.filled.mean()),
           "yes_rate_filled": float(f.loc[f.filled, "yes_won"].mean()),
           "yes_rate_unfilled": float(f.loc[~f.filled, "yes_won"].mean()),
           "mean_a_filled": float(f.loc[f.filled, "a"].mean()), "mean_a_unfilled": float(f.loc[~f.filled, "a"].mean()),
           "by_cat": {c: {"posted": int(len(g)), "filled": int(g.filled.sum()),
                          "yes_filled": float(g.loc[g.filled, "yes_won"].mean()) if g.filled.any() else None,
                          "yes_unfilled": float(g.loc[~g.filled, "yes_won"].mean()) if (~g.filled).any() else None}
                      for c, g in f.groupby("cat")},
           "fills_per_day": float(f.filled.sum() / max(1, pd.to_datetime(f.S).dt.date.nunique()))}
    print(json.dumps(out, indent=1))
    (OUT / f"diag_{split}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"enum": cmd_enum, "prices": cmd_prices, "trades": cmd_trades, "frame": cmd_frame,
     "diag": lambda: _diag("dev"), "diag_sealed": lambda: _diag("sealed")}[cmd]()
