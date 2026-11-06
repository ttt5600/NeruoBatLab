"""Cross-venue in-game lead-lag: Polymarket (international CLOB, public) vs Kalshi on the SAME game winner market.

    python scripts/xvenue_ingame.py list        # Polymarket closed moneyline games + Kalshi settled game markets
    python scripts/xvenue_ingame.py match       # same game across venues -> data/xvenue_ingame/match.parquet
    python scripts/xvenue_ingame.py minute      # 1-min series both venues for every matched game
    python scripts/xvenue_ingame.py ticks [N]   # trade-level tapes both venues for a sample of N games per league
    python scripts/xvenue_ingame.py lag         # descriptive lead-lag + gap/convergence tables

Descriptive only (no outcome-based selection; settlement is not read here). Polymarket side = the international
order book (gamma/clob/data-api); a US user trades the separate Polymarket US book (polymarket.us), which is not the
same order book -- the lead-lag here is about price discovery, not a fill we could get there.
Sources: gamma-api.polymarket.com/events (series ids from /sports), clob.polymarket.com/prices-history (fidelity=1),
data-api.polymarket.com/v2/trades (condition=, cursor; taker side only by default), Kalshi trade-api v2
markets (+ /historical), candlesticks period 1, /markets/trades.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports.http import fetch
from traps import check_epoch_seconds

D = ROOT / "data" / "xvenue_ingame"
D.mkdir(parents=True, exist_ok=True)
KAPI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com/events"
CLOB = "https://clob.polymarket.com/prices-history"
PTR = "https://data-api.polymarket.com/v2/trades"
START = pd.Timestamp("2025-01-01", tz="UTC")
# league -> (polymarket series id, kalshi series)
LG = {"nba": (10345, "KXNBAGAME"), "nfl": ("nfl", "KXNFLGAME"), "nhl": (10346, "KXNHLGAME"),
      "mlb": ("mlb", "KXMLBGAME"), "cbb": (10470, "KXNCAAMBGAME"), "cfb": ("cfb", "KXNCAAFGAME")}


def _ts(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    return pd.to_datetime(str(x).replace("+00", "+00:00") if str(x).endswith("+00") else str(x), utc=True, errors="coerce")


def poly_list(lg: str) -> pd.DataFrame:
    src = LG[lg][0]
    sel = {"tag_slug": src} if isinstance(src, str) else {"series_id": src}
    rows, seen = [], set()
    weeks = pd.date_range(START, pd.Timestamp.now(tz="UTC"), freq="7D")
    for w0 in weeks:
        w1 = w0 + pd.Timedelta(days=7)
        off = 0
        evs = []
        while True:
            try:
                ev = fetch(GAMMA, {**sel, "closed": "true", "limit": 100, "offset": off,
                                   "end_date_min": w0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                   "end_date_max": w1.strftime("%Y-%m-%dT%H:%M:%SZ")}, cache=w1 < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=14))
            except Exception as ex:
                print("  gamma fail", lg, w0.date(), off, ex, flush=True)
                break
            evs += ev or []
            if not ev or len(ev) < 100:
                break
            off += 100
        for e in evs:
            if e.get("id") in seen:
                continue
            seen.add(e.get("id"))
            for m in e.get("markets", []):
                if m.get("sportsMarketType") != "moneyline":
                    continue
                outs = json.loads(m["outcomes"]) if isinstance(m.get("outcomes"), str) else (m.get("outcomes") or [])
                toks = json.loads(m["clobTokenIds"]) if isinstance(m.get("clobTokenIds"), str) else (m.get("clobTokenIds") or [])
                if len(outs) != 2 or len(toks) != 2:
                    continue
                rows.append({"lg": lg, "slug": m.get("slug"), "title": e.get("title"), "cond": m.get("conditionId"),
                             "o0": outs[0], "o1": outs[1], "t0": toks[0], "t1": toks[1],
                             "start": _ts(m.get("gameStartTime") or e.get("startTime")),
                             "pvol": float(m.get("volume") or 0)})
    return pd.DataFrame(rows)


def kalshi_list(lg: str) -> pd.DataFrame:
    ser = LG[lg][1]
    rows = []
    for path, extra in ((f"{KAPI}/historical/markets", {}), (f"{KAPI}/markets", {"status": "settled"})):
        cursor = None
        while True:
            p = {"series_ticker": ser, "limit": 1000, **extra}
            if cursor:
                p["cursor"] = cursor
            d = fetch(path, p, cache=False)
            rows += d.get("markets", [])
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                break
    m = pd.DataFrame(rows).drop_duplicates("ticker")
    m["lg"] = lg
    m["code"] = m.ticker.str.split("-").str[-1]
    m["kdate"] = pd.to_datetime(m.event_ticker.str.split("-").str[1].str[:7], format="%y%b%d", errors="coerce")
    m["kvol"] = pd.to_numeric(m.get("volume_fp", m.get("volume")), errors="coerce")
    keep = ["lg", "ticker", "event_ticker", "code", "kdate", "kvol", "yes_sub_title", "open_time", "close_time",
            "expected_expiration_time"]
    return m[[c for c in keep if c in m.columns]]


def cmd_list() -> int:
    import time
    ps, ks = [], []
    for lg in LG:
        fp, fk = D / f"_poly_{lg}.parquet", D / f"_kalshi_{lg}.parquet"
        if not fp.exists():
            poly_list(lg).to_parquet(fp, index=False)
        if not fk.exists():
            for attempt in range(5):
                try:
                    kalshi_list(lg).astype({"open_time": str, "close_time": str}).to_parquet(fk, index=False)
                    break
                except Exception as ex:
                    print("  kalshi list retry", lg, ex, flush=True)
                    time.sleep(120)
        p, k = pd.read_parquet(fp), pd.read_parquet(fk)
        print(f"{lg}: polymarket {len(p)} moneyline games since {START.date()}, kalshi {k.event_ticker.nunique()} events",
              flush=True)
        ps.append(p)
        ks.append(k)
    pd.concat(ps).to_parquet(D / "poly_games.parquet", index=False)
    pd.concat(ks).astype({"open_time": str, "close_time": str}).to_parquet(D / "kalshi_markets.parquet", index=False)
    return 0


def _norm(s) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum() or ch == " ").strip()


from xvenue_sports_snapshot import NBA as _NBA, NFL as _NFL  # noqa: E402  (nickname -> Kalshi code)

_MLB = {"Diamondbacks": "ARI AZ", "Braves": "ATL", "Orioles": "BAL", "Red Sox": "BOS", "Cubs": "CHC", "White Sox": "CWS",
        "Reds": "CIN", "Guardians": "CLE", "Rockies": "COL", "Tigers": "DET", "Astros": "HOU", "Royals": "KC",
        "Angels": "LAA", "Dodgers": "LAD", "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN", "Mets": "NYM",
        "Yankees": "NYY", "Athletics": "ATH OAK", "Phillies": "PHI", "Philies": "PHI", "Pirates": "PIT", "Padres": "SD",
        "Giants": "SF", "Mariners": "SEA", "Cardinals": "STL", "Rays": "TB", "Rangers": "TEX", "Blue Jays": "TOR",
        "Nationals": "WSH WAS"}
NICK = {"nba": {k: {v} for k, v in _NBA.items()},
        "nfl": {k: {v, {"JAX": "JAC", "LA": "LAR"}.get(v, v)} for k, v in _NFL.items()},
        "mlb": {k: set(v.split()) for k, v in _MLB.items()}}


def _side_ok(lg: str, poly_name: str, k_code: str, k_sub: str) -> bool:
    """Polymarket outcome (team nickname / school) <-> Kalshi market (ticker code, yes_sub_title)."""
    pn, sub = _norm(poly_name), _norm(k_sub)
    if lg in NICK:
        for nick, codes in NICK[lg].items():
            if pn == _norm(nick) or pn.endswith(" " + _norm(nick)):
                return k_code.upper() in codes
        return False
    if not pn or not sub:
        return False
    if pn == sub or pn.startswith(sub + " ") or sub.endswith(" " + pn) or sub.startswith(pn + " "):
        return True
    return len(pn) >= 5 and pn in sub


def cmd_match() -> int:
    p = pd.read_parquet(D / "poly_games.parquet")
    k = pd.read_parquet(D / "kalshi_markets.parquet")
    p = p.dropna(subset=["start"]).drop_duplicates("cond")
    p["pdate"] = pd.to_datetime(p.slug.str[-10:], format="%Y-%m-%d", errors="coerce")
    p["et_date"] = p.start.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    parts = p.slug.str.split("-")
    p["c0"] = parts.str[1].fillna("")
    p["c1"] = parts.str[2].fillna("")
    ev = {(lg, d): x for (lg, d), x in k.groupby(["lg", "kdate"])}
    out, amb = [], 0
    for r in p.itertuples():
        cands = []
        for dd in (0, -1, 1):
            x = ev.get((r.lg, r.et_date + pd.Timedelta(days=dd)))
            if x is None:
                continue
            for et, e in x.groupby("event_ticker"):
                if len(e) != 2:
                    continue
                a, b = e.iloc[0], e.iloc[1]
                # which Kalshi market is Polymarket outcome 0?  slug codes are in the same order as outcomes
                for ka, kb in ((a, b), (b, a)):
                    if (_side_ok(r.lg, r.o0, ka.code, ka.yes_sub_title) and
                            _side_ok(r.lg, r.o1, kb.code, kb.yes_sub_title)):
                        cands.append((dd, et, ka, kb))
            if cands:
                break
        if len(cands) != 1:
            amb += len(cands) > 1
            continue
        dd, et, ka, kb = cands[0]
        out.append({"lg": r.lg, "slug": r.slug, "cond": r.cond, "tok": r.t0, "team": r.o0, "opp": r.o1,
                    "start": r.start, "pvol": r.pvol, "ticker": ka.ticker, "opp_ticker": kb.ticker,
                    "event_ticker": et, "kvol": float(ka.kvol or 0) + float(kb.kvol or 0), "date_off": dd})
    m = pd.DataFrame(out)
    m = m.drop_duplicates("ticker", keep=False).drop_duplicates("cond", keep=False)
    m["split"] = np.where(m.start < pd.Timestamp("2026-03-01", tz="UTC"), "dev",
                          np.where(m.start < pd.Timestamp("2026-11-06", tz="UTC"), "sealed", "forward"))
    m.to_parquet(D / "match.parquet", index=False)
    print(f"polymarket games {len(p)}; matched {len(m)} (ambiguous {amb})")
    print(m.groupby(["lg", "split"]).size().unstack(fill_value=0))
    print("kalshi events per league:", k.groupby("lg").event_ticker.nunique().to_dict())
    print("poly games per league:", p.groupby("lg").size().to_dict())
    return 0


DUR = {"nba": 3.0, "nfl": 4.0, "nhl": 3.0, "mlb": 4.0, "cbb": 2.5, "cfb": 4.5}   # hours after start to pull


def _win(r):
    s0 = int(r.start.timestamp()) - 1800
    return s0, s0 + 1800 + int(DUR[r.lg] * 3600)


def _kpx(x, k, f="close"):
    d = x.get(k) or {}
    v = d.get(f, d.get(f + "_dollars"))
    return np.nan if v is None else float(v)


def _one_minute(r):
    s0, e0 = _win(r)
    kc = []
    for path in (f"{KAPI}/historical/markets/{r.ticker}/candlesticks",
                 f"{KAPI}/series/{LG[r.lg][1]}/markets/{r.ticker}/candlesticks"):
        try:
            j = fetch(path, {"start_ts": s0, "end_ts": e0, "period_interval": 1}, cache=False)
        except Exception:
            continue
        if j.get("candlesticks"):
            kc = j["candlesticks"]
            break
    try:
        ph = fetch(CLOB, {"market": r.tok, "startTs": s0, "endTs": e0, "fidelity": 1}, cache=False).get("history", [])
    except Exception:
        ph = []
    k = pd.DataFrame([{"t": int(x["end_period_ts"]), "k_bid": _kpx(x, "yes_bid"), "k_ask": _kpx(x, "yes_ask"),
                       "k_px": _kpx(x, "price"), "k_vol": float(x.get("volume") or x.get("volume_fp") or 0)} for x in kc])
    p = pd.DataFrame(ph).rename(columns={"t": "pt", "p": "p_px"})
    return r.cond, k, p


def cmd_minute() -> int:
    m = pd.read_parquet(D / "match.parquet")
    m = m[m.split == "dev"]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 400      # random (seed 0) games per league; no outcome info used
    m = pd.concat([x.sample(min(cap, len(x)), random_state=0) for _, x in m.groupby("lg")])
    rows, praw = [], []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, (cond, k, p) in enumerate(ex.map(_one_minute, list(m.itertuples()))):
            if len(k):
                k["cond"] = cond
                rows.append(k)
            if len(p):
                p["cond"] = cond
                praw.append(p)
            if i % 250 == 0:
                print("minute", i, len(m), flush=True)
    k = pd.concat(rows, ignore_index=True)
    p = pd.concat(praw, ignore_index=True)
    check_epoch_seconds(k.t, "kalshi end_period_ts")
    check_epoch_seconds(p.pt, "poly t")
    k.to_parquet(D / "kalshi_minute_dev.parquet", index=False)
    p.to_parquet(D / "poly_points_dev.parquet", index=False)
    print(f"kalshi candles {len(k)} for {k.cond.nunique()} games; poly points {len(p)} for {p.cond.nunique()} games")
    return 0


def _ktrades(ticker, s0, e0):
    out = []
    for path in (f"{KAPI}/historical/trades", f"{KAPI}/markets/trades"):
        cursor = None
        while True:
            q = {"ticker": ticker, "min_ts": s0, "max_ts": e0, "limit": 1000}
            if cursor:
                q["cursor"] = cursor
            try:
                j = fetch(path, q, cache=False)
            except Exception:
                break
            out += j.get("trades", [])
            cursor = j.get("cursor")
            if not cursor or not j.get("trades"):
                break
        if out:
            break
    return out


def _ptrades(cond, s0, e0, max_pages=200):
    out, cursor = [], None
    for _ in range(max_pages):
        q = {"condition": cond, "cursor": cursor} if cursor else {"condition": cond, "limit": 1000}
        try:
            j = fetch(PTR, q, cache=False)
        except Exception:
            break
        d = j.get("data", [])
        out += [x for x in d if s0 <= x["timestamp"] <= e0]
        cursor = (j.get("pagination") or {}).get("next_cursor")
        if not cursor or not d or min(x["timestamp"] for x in d) < s0:   # newest-first feed: stop once before window
            break
    return out


def _one_tick(r):
    s0, e0 = _win(r)
    kt = _ktrades(r.ticker, s0, e0)
    pt = _ptrades(r.cond, s0, e0)
    k = pd.DataFrame([{"ts": pd.Timestamp(x["created_time"]), "px": float(x["yes_price_dollars"]),
                       "n": float(x.get("count_fp") or x.get("count") or 0), "taker": x.get("taker_side")} for x in kt])
    # Polymarket: price is for the traded token; convert to OUR token (outcome 0 = r.tok) probability
    p = pd.DataFrame([{"ts": int(x["timestamp"]), "px": float(x["price"]) if x["token_id"] == r.tok else 1 - float(x["price"]),
                       "n": float(x["size"]), "side": x["side"], "mine": x["token_id"] == r.tok} for x in pt])
    return r.cond, k, p


def cmd_ticks() -> int:
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    m = pd.read_parquet(D / "match.parquet")
    m = m[(m.split == "dev") & m.lg.isin(["nba", "nfl", "nhl", "cbb"])]
    samp = pd.concat([x.sample(min(n, len(x)), random_state=0) for _, x in m.groupby("lg")])
    ks, ps = [], []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for i, (cond, k, p) in enumerate(ex.map(_one_tick, list(samp.itertuples()))):
            if len(k):
                k["cond"] = cond
                ks.append(k)
            if len(p):
                p["cond"] = cond
                ps.append(p)
            print("ticks", i, len(samp), len(k), len(p), flush=True)
    k = pd.concat(ks, ignore_index=True)
    k["ts"] = pd.to_datetime(k.ts, utc=True, format="ISO8601")
    p = pd.concat(ps, ignore_index=True)
    check_epoch_seconds(p.ts, "poly trade ts")
    k.to_parquet(D / "kalshi_ticks_dev.parquet", index=False)
    p.to_parquet(D / "poly_ticks_dev.parquet", index=False)
    samp.to_parquet(D / "tick_sample.parquet", index=False)
    print(f"kalshi trades {len(k)} ({k.cond.nunique()} games); poly trades {len(p)} ({p.cond.nunique()} games)")
    return 0


# ------------------------------------------------------------------ analysis (dev only)
RNG = np.random.default_rng(0)


def kfee(p):
    return 0.07 * p * (1 - p)


def boot_ratio(df, num, den, B=2000, key="cond"):
    """game-clustered bootstrap of sum(num)/sum(den) via per-game sums (fast)."""
    s = df.assign(_n=num, _d=den).groupby(key)[["_n", "_d"]].sum().values
    idx = RNG.integers(0, len(s), (B, len(s)))
    v = s[idx].sum(1)
    r = v[:, 0] / v[:, 1]
    return float(s[:, 0].sum() / s[:, 1].sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def boot(df, stat, B=1000, key="cond"):
    """game-clustered bootstrap: stat(frame) -> float; returns (point, lo, hi)."""
    g = {c: x for c, x in df.groupby(key)}
    keys = np.array(list(g))
    pt = stat(df)
    vals = []
    for _ in range(B):
        s = RNG.choice(keys, len(keys))
        vals.append(stat(pd.concat([g[c] for c in s], ignore_index=True)))
    return pt, float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def minute_frame() -> pd.DataFrame:
    m = pd.read_parquet(D / "match.parquet")
    m = m[m.split == "dev"]
    k = pd.read_parquet(D / "kalshi_minute_dev.parquet")
    p = pd.read_parquet(D / "poly_points_dev.parquet").sort_values("pt")
    k = k.merge(m[["cond", "lg", "start"]], on="cond", how="inner", validate="many_to_one")
    k["t0"] = (k.start - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    k = k.sort_values("t")
    k = pd.merge_asof(k, p, left_on="t", right_on="pt", by="cond", direction="backward")
    k["p_age"] = k.t - k.pt
    k = k.sort_values(["cond", "t"]).reset_index(drop=True)
    k["k_mid"] = (k.k_bid + k.k_ask) / 2
    k["k_spr"] = k.k_ask - k.k_bid
    k["dur"] = k.lg.map(DUR) * 3600
    k["ingame"] = (k.t >= k.t0) & (k.t <= k.t0 + k.dur)
    return k


def cmd_lag() -> int:
    k = minute_frame()
    g = k[k.ingame].copy()
    print(f"dev in-game minutes {len(g)} in {g.cond.nunique()} games; poly point age (s) median {g.p_age.median():.0f}, "
          f"p90 {g.p_age.quantile(.9):.0f}")
    ok = g.k_mid.between(.03, .97) & g.p_px.between(.03, .97) & (g.k_spr <= .10) & (g.p_age <= 120) & g.k_bid.notna()
    g = g[ok].copy()
    # consecutive-minute diffs within game
    g["gap_t"] = g.groupby("cond").t.diff()
    for L in range(-5, 6):
        g[f"dP_{L}"] = g.groupby("cond").p_px.shift(-L) - g.groupby("cond").p_px.shift(-L + 1)
    g["dK"] = g.groupby("cond").k_mid.diff()
    print(f"usable minutes {len(g)} games {g.cond.nunique()}; |gap| median {(g.p_px - g.k_mid).abs().median():.4f} "
          f"p90 {(g.p_px - g.k_mid).abs().quantile(.9):.4f}; kalshi spread median {g.k_spr.median():.3f}")
    print("\nLEAD-LAG corr(dK_t, dP_{t+L}) by league (L>0: Kalshi move precedes Poly move):")
    tab = []
    for lg, x in g.groupby("lg"):
        row = {"lg": lg, "games": x.cond.nunique(), "min": len(x)}
        for L in (-3, -2, -1, 0, 1, 2, 3):
            y = x[["dK", f"dP_{L}"]].dropna()
            row[L] = round(float(np.corrcoef(y.dK, y[f"dP_{L}"])[0, 1]), 3)
        tab.append(row)
    print(pd.DataFrame(tab).to_string(index=False))

    # error correction: next-minute moves toward the other venue
    g["gap"] = g.p_px - g.k_mid
    for h in (1, 5):
        g[f"dK{h}"] = g.groupby("cond").k_mid.shift(-h) - g.k_mid
        g[f"dP{h}"] = g.groupby("cond").p_px.shift(-h) - g.p_px
    print("\nERROR CORRECTION: share of the gap closed by each venue over h min (|gap|>=g), game-bootstrap CI")
    for gth in (.02, .03, .05):
        x = g[g.gap.abs() >= gth].dropna(subset=["dK5", "dP5"])
        for h in (1, 5):
            fk = boot_ratio(x, x[f"dK{h}"] * np.sign(x.gap), x.gap.abs())
            fp = boot_ratio(x, -x[f"dP{h}"] * np.sign(x.gap), x.gap.abs())
            print(f"  |gap|>={gth:.2f} n={len(x)} games={x.cond.nunique()} h={h}: Kalshi closes {fk[0]:+.3f} "
                  f"[{fk[1]:+.3f},{fk[2]:+.3f}]  Poly closes {fp[0]:+.3f} [{fp[1]:+.3f},{fp[2]:+.3f}]")

    # DESCRIPTIVE GATE: buy Kalshi side at ask(t+1)+fee when Poly mid - (ask_t + fee) >= X; CLV at mid(t+1+5)
    k2 = k[k.ingame].copy()
    for c in ("k_ask", "k_bid", "k_mid"):
        k2[c + "_n1"] = k2.groupby("cond")[c].shift(-1)
        k2[c + "_n6"] = k2.groupby("cond")[c].shift(-6)
        k2[c + "_n5"] = k2.groupby("cond")[c].shift(-5)
    k2["tn1"] = k2.groupby("cond").t.shift(-1)
    k2["tn6"] = k2.groupby("cond").t.shift(-6)
    k2 = k2[ok.reindex(k2.index, fill_value=False) & (k2.tn1 - k2.t == 60) & (k2.tn6 - k2.t == 360)]
    print("\nDESCRIPTIVE GATE (dev): CLV_5min per contract, game-bootstrap 95% CI; delayed = ask one candle later")
    res = {}
    for X in (.03, .05):
        sy = k2.p_px - (k2.k_ask + kfee(k2.k_ask)) >= X
        sn = k2.k_bid - kfee(1 - k2.k_bid) - k2.p_px >= X
        s = k2[sy | sn].copy()
        s["yes"] = sy[sy | sn]
        # non-overlapping: drop a signal within 5 min of the previous kept one in the same game
        keep, last = [], {}
        for r in s.itertuples():
            if r.t - last.get(r.cond, -1e18) >= 300:
                keep.append(r.Index)
                last[r.cond] = r.t
        s = s.loc[keep]
        for mode in ("delayed", "same"):
            if mode == "delayed":
                ay, an = s.k_ask_n1, 1 - s.k_bid_n1
                fut = s.k_mid_n6
            else:
                ay, an = s.k_ask, 1 - s.k_bid
                fut = s.k_mid_n5
            s["clv"] = np.where(s.yes, fut - ay - kfee(ay), (1 - fut) - an - kfee(an))
            s["px"] = np.where(s.yes, ay, an)
            x = s.dropna(subset=["clv"])
            pt, lo, hi = boot_ratio(x, x.clv, np.ones(len(x)))
            res[(X, mode)] = (len(x), x.cond.nunique(), pt, lo, hi)
            print(f"  X={X:.2f} {mode:7s} n={len(x)} games={x.cond.nunique()} mean CLV {pt:+.4f} [{lo:+.4f},{hi:+.4f}]"
                  f"  per $ {pt / x.px.mean():+.3%}")
            if mode == "delayed":
                print("     by league:", {lg: (len(y), round(float(y.clv.mean()), 4)) for lg, y in x.groupby("lg")})
    s.to_parquet(D / "gate_signals_dev.parquet", index=False)
    return 0


def cmd_lag2() -> int:
    """ALIGNMENT FIX (found after `lag`): Polymarket prices-history points sit at ~:07 past each minute, Kalshi candles
    close at :00, so `lag`'s backward merge paired Kalshi(t) with Poly(t-53 s) -- a stale Poly read that manufactures a
    'Kalshi leads by 1 min' pattern. Here the grid is the Poly point time pt; Kalshi = last candle close <= pt
    (age ~7 s, Kalshi slightly OLDER than Poly). Decision at pt; delayed execution = next Kalshi candle close (~53 s
    later); CLV at Kalshi mid 5 candles after that."""
    m = pd.read_parquet(D / "match.parquet")
    m = m[m.split == "dev"]
    k = pd.read_parquet(D / "kalshi_minute_dev.parquet").sort_values("t")
    p = pd.read_parquet(D / "poly_points_dev.parquet")
    p = p.merge(m[["cond", "lg", "start"]], on="cond", how="inner", validate="many_to_one")
    p["t0"] = (p.start - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    p = p.sort_values("pt")
    k["tk"] = k.t
    x = pd.merge_asof(p, k[["cond", "t", "tk", "k_bid", "k_ask"]], left_on="pt", right_on="t", by="cond",
                      direction="backward", tolerance=59)
    x = x.sort_values(["cond", "pt"]).reset_index(drop=True)
    x["k_age"] = x.pt - x.tk
    x["k_mid"] = (x.k_bid + x.k_ask) / 2
    x["k_spr"] = x.k_ask - x.k_bid
    x["ingame"] = (x.pt >= x.t0) & (x.pt <= x.t0 + x.lg.map(DUR) * 3600)
    g = x[x.ingame & x.k_mid.between(.03, .97) & x.p_px.between(.03, .97) & (x.k_spr <= .10)].copy()
    print(f"aligned grid: {len(g)} points, {g.cond.nunique()} games; Kalshi age vs Poly point: median "
          f"{g.k_age.median():.0f} s")
    g["step"] = g.groupby("cond").pt.diff()
    g["dK"] = g.groupby("cond").k_mid.diff()
    g["dP"] = g.groupby("cond").p_px.diff()
    g = g[g.step.between(50, 70)]
    print("LEAD-LAG on aligned grid corr(dK_i, dP_{i+L}) (L>0: Kalshi move precedes Poly; Poly is ~7 s later at L=0):")
    tab = []
    for lg, y in g.groupby("lg"):
        row = {"lg": lg, "games": y.cond.nunique()}
        for L in (-2, -1, 0, 1, 2):
            z = pd.DataFrame({"a": y.dK, "b": y.groupby("cond").dP.shift(-L)}).dropna()
            row[L] = round(float(np.corrcoef(z.a, z.b)[0, 1]), 3)
        tab.append(row)
    print(pd.DataFrame(tab).to_string(index=False))
    g["gap"] = g.p_px - g.k_mid
    print(f"|gap| = |Poly mid - Kalshi mid| median {g.gap.abs().median():.4f}, p90 {g.gap.abs().quantile(.9):.4f}, "
          f"p99 {g.gap.abs().quantile(.99):.4f}; share >= 3c {(g.gap.abs() >= .03).mean():.3f}")
    for h in (1, 5):
        g[f"dK{h}"] = g.groupby("cond").k_mid.shift(-h) - g.k_mid
        g[f"dP{h}"] = g.groupby("cond").p_px.shift(-h) - g.p_px
    print("ERROR CORRECTION (aligned): share of gap closed over h steps (~min)")
    for gth in (.02, .03, .05):
        y = g[g.gap.abs() >= gth].dropna(subset=["dK5", "dP5"])
        for h in (1, 5):
            fk = boot_ratio(y, y[f"dK{h}"] * np.sign(y.gap), y.gap.abs())
            fp = boot_ratio(y, -y[f"dP{h}"] * np.sign(y.gap), y.gap.abs())
            print(f"  |gap|>={gth:.2f} n={len(y)} games={y.cond.nunique()} h={h}: Kalshi closes {fk[0]:+.3f} "
                  f"[{fk[1]:+.3f},{fk[2]:+.3f}]  Poly closes {fp[0]:+.3f} [{fp[1]:+.3f},{fp[2]:+.3f}]")
    # gate on aligned grid; Kalshi future quotes from the candle table
    kk = k.set_index(["cond", "t"])[["k_bid", "k_ask"]]
    g["t_ex"] = g.tk + 60
    g["t_cl"] = g.tk + 360
    ex = kk.reindex(pd.MultiIndex.from_arrays([g.cond, g.t_ex])).values
    cl = kk.reindex(pd.MultiIndex.from_arrays([g.cond, g.t_cl])).values
    g["ex_bid"], g["ex_ask"] = ex[:, 0], ex[:, 1]
    g["cl_mid"] = cl.mean(1)
    cl0 = kk.reindex(pd.MultiIndex.from_arrays([g.cond, g.tk + 300])).values
    g["cl0_mid"] = cl0.mean(1)
    print("DESCRIPTIVE GATE (aligned, dev): buy Kalshi side when Poly mid - (Kalshi ask + fee) >= X (mirror NO)")
    for X in (.03, .05):
        sy = g.p_px - (g.k_ask + kfee(g.k_ask)) >= X
        sn = g.k_bid - kfee(1 - g.k_bid) - g.p_px >= X
        s = g[sy | sn].copy()
        s["yes"] = sy[sy | sn]
        keep, last = [], {}
        for r in s.itertuples():
            if r.pt - last.get(r.cond, -1e18) >= 300:
                keep.append(r.Index)
                last[r.cond] = r.pt
        s = s.loc[keep]
        for mode in ("delayed", "same"):
            ay = s.ex_ask if mode == "delayed" else s.k_ask
            an = 1 - (s.ex_bid if mode == "delayed" else s.k_bid)
            fut = s.cl_mid if mode == "delayed" else s.cl0_mid
            s["clv"] = np.where(s.yes, fut - ay - kfee(ay), (1 - fut) - an - kfee(an))
            s["px"] = np.where(s.yes, ay, an)
            y = s.dropna(subset=["clv"])
            pt_, lo, hi = boot_ratio(y, y.clv, np.ones(len(y)))
            print(f"  X={X:.2f} {mode:7s} n={len(y)} games={y.cond.nunique()} mean CLV {pt_:+.4f} [{lo:+.4f},{hi:+.4f}]"
                  f"  per $ {pt_ / y.px.mean():+.3%}  by lg " +
                  str({lg: (len(z), round(float(z.clv.mean()), 4)) for lg, z in y.groupby("lg")}))
        s.to_parquet(D / f"gate2_signals_dev_X{int(X * 100)}.parquet", index=False)
    return 0


def cmd_tlag() -> int:
    """trade-level lead-lag on the tick sample (5-s last-trade grid; jump event study)."""
    m = pd.read_parquet(D / "tick_sample.parquet")
    kt = pd.read_parquet(D / "kalshi_ticks_dev.parquet")
    pt = pd.read_parquet(D / "poly_ticks_dev.parquet")
    kt["s"] = ((kt.ts - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1))
    check_epoch_seconds(kt.s, "kalshi trade s")
    pt["s"] = pt.ts.astype(float)
    STEP = 5
    rows_cc, ev = [], []
    for r in m.itertuples():
        a = kt[kt.cond == r.cond].sort_values("s")
        b = pt[pt.cond == r.cond].sort_values("s")
        if len(a) < 50 or len(b) < 50:
            continue
        t0 = r.start.timestamp()
        grid = np.arange(t0, t0 + DUR[r.lg] * 3600, STEP)
        ka = pd.merge_asof(pd.DataFrame({"s": grid}), a[["s", "px"]], on="s").px.values
        pb = pd.merge_asof(pd.DataFrame({"s": grid}), b[["s", "px"]], on="s").px.values
        okm = (ka > .03) & (ka < .97) & (pb > .03) & (pb < .97)
        ka = np.where(okm, ka, np.nan)
        pb = np.where(okm, pb, np.nan)
        dk, dp = np.diff(ka), np.diff(pb)
        rows_cc.append(pd.DataFrame({"cond": r.cond, "lg": r.lg, "dk": dk, "dp": dp}))
        # jumps: >= 4c over 10 s on one venue, persisting 15 s later (>= half), then time for the other to cover half
        for name, X, Y in (("kalshi", ka, pb), ("poly", pb, ka)):
            i = 2
            while i < len(X) - 3:
                j = X[i] - X[i - 2]
                if np.isfinite(j) and abs(j) >= .04 and np.isfinite(X[i + 3]) and np.sign(j) * (X[i + 3] - X[i - 2]) >= abs(j) / 2:
                    y0 = Y[i - 2]
                    lag = np.nan
                    if np.isfinite(y0):
                        for q in range(i - 1, min(i + 61, len(Y))):      # up to 300 s
                            if np.isfinite(Y[q]) and np.sign(j) * (Y[q] - y0) >= abs(j) / 2:
                                lag = (q - i) * STEP
                                break
                    ev.append({"cond": r.cond, "lg": r.lg, "leader": name, "jump": abs(j), "lag": lag})
                    i += 6
                else:
                    i += 1
    cc = pd.concat(rows_cc, ignore_index=True)
    print(f"tick sample games used {cc.cond.nunique()} ({cc.groupby('lg').cond.nunique().to_dict()})")
    print("5-s return cross-corr corr(dK_t, dP_{t+L}) (L>0 = Kalshi first), pooled:")
    out = {}
    for L in (-6, -3, -2, -1, 0, 1, 2, 3, 6):
        x = cc.copy()
        x["dpL"] = x.groupby("cond").dp.shift(-L)
        y = x[["dk", "dpL"]].dropna()
        out[L * STEP] = round(float(np.corrcoef(y.dk, y.dpL)[0, 1]), 3)
    print("  ", out)
    e = pd.DataFrame(ev)
    e.to_parquet(D / "tick_jumps_dev.parquet", index=False)
    print("jump event study (leader jumped >=4c in 10 s): other venue's lag to cover half, seconds (censored 300 s)")
    for (ld), x in e.groupby("leader"):
        lagv = x.lag
        med = boot(x.assign(l=lagv.fillna(9999)), lambda d: float(np.median(d.l)), B=500)
        print(f"  leader={ld}: events {len(x)} games {x.cond.nunique()} follower within 300 s {lagv.notna().mean():.2f}; "
              f"median lag {med[0]:.0f} s [{med[1]:.0f},{med[2]:.0f}]; lag<=0 (follower already moved) "
              f"{(lagv <= 0).mean():.2f}; by lg {x.groupby('lg').lag.median().to_dict()}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    sys.exit({"list": cmd_list, "match": cmd_match, "minute": cmd_minute, "ticks": cmd_ticks, "lag": cmd_lag,
              "tlag": cmd_tlag, "lag2": cmd_lag2}[cmd]())
