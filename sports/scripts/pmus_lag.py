"""Kalshi-leads-Polymarket-US in-game lag: does Polymarket US's REAL ask lag a Kalshi jump long enough to buy it?

    python scripts/pmus_lag.py match            # PM US closed game moneylines <-> Kalshi game markets (metadata only)
    python scripts/pmus_lag.py build dev        # sample games, pull Kalshi tape + PM US 1-min ask/bid -> frame_dev
    python scripts/pmus_lag.py build sealed     # ONLY after a registered trial passed dev (edgelab gate)
    python scripts/pmus_lag.py describe dev     # descriptive staleness table (no outcomes)

Pre-registration: notes/pmus_lag.md (written before any price was pulled). Trials pmus_lag_t1 / pmus_lag_t2
(family cross_venue). Vault dataset pmus_ingame_lag: dev 2026-05-01..2026-08-31, sealed 2026-09-01..2026-11-04 (UTC date
of game start); 2026-11-05 = plumbing probe day, in no split; forward = live logger (scripts/pmus_lag_logger.py).

Sources (public, no key): gateway.polymarket.us/v1/events (seriesId, closed) and /v1/price-history (custom range,
fidelity 1 -> one stored observation per minute ON the minute: longPrice = YES display price, normally the best ask;
shortPrice = NO display price, normally 1 - best bid). Kalshi trade-api v2 /markets/trades + /historical/trades.

Frame (one row per Kalshi jump; pre-decision columns only, `result` hidden by the vault):
  jump = a Kalshi trade on the PM-long team's ticker whose YES price differs by >= 4c from the last trade at or before
  10 s earlier; in-game (start <= t <= Kalshi close), 0.05 <= price <= 0.95; one jump per game per 60 s.
  attrs['pm'] = {slug: DataFrame[ts, long, short]} the PM US minute observations of that game.
"""
from __future__ import annotations

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

D = ROOT / "data" / "pmus_lag"
D.mkdir(parents=True, exist_ok=True)
KAPI = "https://api.elections.kalshi.com/trade-api/v2"
G = "https://gateway.polymarket.us/v1"
SERIES = {"nba": 4, "nhl": 6, "mlb": 15, "nfl": 2, "cfb": 225, "cbb": 7}
SPLITS = {"dev": ("2026-05-01", "2026-08-31"), "sealed": ("2026-09-01", "2026-11-04")}
N_PER_SPLIT = 700
SEED = {"dev": 20261106, "sealed": 20261107}
DUR = {"nba": 3.0, "nfl": 4.0, "nhl": 3.0, "mlb": 4.0, "cbb": 2.5, "cfb": 4.5}
JUMP_C, JUMP_S, DEDUP_S = 0.04, 10, 60
ALIAS = {"VEG": ["VGK"], "VGK": ["VEG"], "WAS": ["WSH"], "WSH": ["WAS"], "LA": ["LAR", "LAK", "LAD"], "NJ": ["NJD"],
         "TB": ["TBL"], "SJ": ["SJS"], "UTAH": ["UTA"], "UTA": ["UTAH"], "GS": ["GSW"], "NO": ["NOP"], "SA": ["SAS"],
         "NY": ["NYK"], "JAX": ["JAC"], "MON": ["MTL"], "NAS": ["NSH"], "CHW": ["CWS"], "CWS": ["CHW"], "AZ": ["ARI"],
         "ARI": ["AZ"], "OAK": ["ATH"], "ATH": ["OAK"], "PHO": ["PHX"], "PHX": ["PHO"]}


def _norm(s) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def pm_games() -> pd.DataFrame:
    rows = []
    for lg, ser in SERIES.items():
        off = 0
        while True:
            ev = (fetch(f"{G}/events", {"seriesId": ser, "closed": "true", "limit": 200, "offset": off,
                                        "startTimeMin": "2026-05-01T00:00:00Z"}, cache=False) or {}).get("events", [])
            for e in ev:
                for m in e.get("markets", []):
                    if m.get("marketType") != "moneyline" or not m.get("slug", "").startswith("aec-"):
                        continue
                    sides = m.get("marketSides") or []
                    L = [s for s in sides if s.get("long") is True]
                    S = [s for s in sides if s.get("long") is False]
                    if len(L) != 1 or len(S) != 1:
                        continue
                    t0, t1 = L[0].get("team") or {}, S[0].get("team") or {}
                    px = L[0].get("price")
                    rows.append({"lg": lg, "slug": m["slug"], "start": e.get("startTime") or m.get("gameStartTime"),
                                 "a_abbr": t0.get("abbreviation"), "a_disp": t0.get("displayAbbreviation"),
                                 "a_name": t0.get("safeName") or t0.get("name"), "a_nick": L[0].get("description"),
                                 "b_abbr": t1.get("abbreviation"), "b_disp": t1.get("displayAbbreviation"),
                                 "b_name": t1.get("safeName") or t1.get("name"),
                                 "result": "yes" if px == "1" else ("no" if px == "0" and S[0].get("price") == "1" else None)})
            if len(ev) < 200:
                break
            off += 200
    g = pd.DataFrame(rows).drop_duplicates("slug")
    g["start"] = pd.to_datetime(g["start"], utc=True)
    return g


def _codes(*xs) -> set[str]:
    out = set()
    for x in xs:
        if x:
            u = str(x).upper()
            out |= {u, *ALIAS.get(u, [])}
    return out


def cmd_match() -> int:
    g = pm_games()
    k = pd.read_parquet(ROOT / "data" / "xvenue_ingame" / "kalshi_markets.parquet")
    k = k[k.lg.isin(SERIES)]
    ev = {(lg, d): x for (lg, d), x in k.groupby(["lg", "kdate"])}
    out = []
    for r in g.itertuples():
        etd = r.start.tz_convert("America/New_York").normalize().tz_localize(None)
        ca, cb = _codes(r.a_abbr, r.a_disp), _codes(r.b_abbr, r.b_disp)
        cands = []
        for dd in (0, -1, 1):
            x = ev.get((r.lg, etd + pd.Timedelta(days=dd)))
            if x is None:
                continue
            for et, e in x.groupby("event_ticker"):
                if len(e) != 2:
                    continue
                for ka, kb in ((e.iloc[0], e.iloc[1]), (e.iloc[1], e.iloc[0])):
                    oka = ka.code in ca or _norm(ka.yes_sub_title) in (_norm(r.a_name), )
                    okb = kb.code in cb or _norm(kb.yes_sub_title) in (_norm(r.b_name), )
                    if oka and okb:
                        cands.append((et, ka, kb))
            if cands:
                break
        if len(cands) != 1:
            continue
        et, ka, kb = cands[0]
        # MLB doubleheaders: Kalshi ticker carries the local start time; keep only if close_time is after PM start
        out.append({"lg": r.lg, "slug": r.slug, "start": r.start, "ticker": ka.ticker, "opp_ticker": kb.ticker,
                    "event_ticker": et, "k_close": pd.to_datetime(ka.close_time, utc=True), "result": r.result})
    m = pd.DataFrame(out)
    m = m.drop_duplicates("ticker", keep=False).drop_duplicates("slug", keep=False)
    m = m[m.k_close > m.start]
    m["date"] = m.start.dt.tz_localize(None).dt.normalize()
    m["split"] = None
    for s, (a, b) in SPLITS.items():
        m.loc[m.date.between(pd.Timestamp(a), pd.Timestamp(b)), "split"] = s
    m.to_parquet(D / "match.parquet", index=False)
    print(f"PM US closed game moneylines since 2026-05-01: {len(g)}; matched unique: {len(m)}")
    print(m.groupby(["lg", "split"], dropna=False).size().unstack(fill_value=0))
    return 0


def _ktrades(ticker, s0, e0):
    out = []
    for path in (f"{KAPI}/markets/trades", f"{KAPI}/historical/trades"):
        cursor = None
        while True:
            q = {"ticker": ticker, "min_ts": s0, "max_ts": e0, "limit": 1000}
            if cursor:
                q["cursor"] = cursor
            try:
                j = fetch(path, q, cache=True)
            except Exception:
                break
            out += j.get("trades", [])
            cursor = j.get("cursor")
            if not cursor or not j.get("trades"):
                break
        if out:
            break
    return out


def _pm_hist(slug, s0, e0):
    h = (fetch(f"{G}/price-history", {"symbol": slug, "timestamp.startTimestamp": int(s0),
                                      "timestamp.endTimestamp": int(e0), "fidelity": 1}, cache=True) or {}).get("history", [])
    return pd.DataFrame([{"ts": int(x["timestamp"]), "long": float(x["longPrice"]), "short": float(x["shortPrice"])}
                         for x in h if x.get("longPrice") is not None and x.get("shortPrice") is not None])


def _jumps(tr: pd.DataFrame, s0: int, e0: int) -> pd.DataFrame:
    """tr: Kalshi trades [ts (float s), p (YES price)] sorted. One jump per DEDUP_S."""
    ts, p = tr.ts.to_numpy(float), tr.p.to_numpy(float)
    out, last = [], -1e18
    for i in range(len(ts)):
        t = ts[i]
        if t < s0 or t > e0 or t < last + DEDUP_S or not (0.05 <= p[i] <= 0.95):
            continue
        j = np.searchsorted(ts, t - JUMP_S, side="right") - 1
        if j < 0:
            continue
        d = p[i] - p[j]
        if abs(d) >= JUMP_C - 1e-9:
            out.append({"t_jump": t, "p_jump": p[i], "p_ref": p[j], "t_ref": ts[j], "sgn": int(np.sign(d))})
            last = t
    return pd.DataFrame(out)


def _one(r):
    s0 = int(r.start.timestamp())
    e0 = min(int(r.k_close.timestamp()), s0 + int((DUR[r.lg] + 1.0) * 3600))
    kt = _ktrades(r.ticker, s0 - 900, e0)
    if not kt:
        return None, None
    tr = pd.DataFrame({"ts": pd.to_datetime([x["created_time"] for x in kt], utc=True, format="ISO8601"),
                       "p": [float(x.get("yes_price_dollars") or x.get("yes_price", 0) / 100) for x in kt]})
    tr["ts"] = (tr.ts - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    check_epoch_seconds(tr.ts, "kalshi created_time")
    tr = tr.sort_values("ts", kind="stable").reset_index(drop=True)
    j = _jumps(tr, s0, e0)
    h = _pm_hist(r.slug, s0 - 900, e0 + 900)
    if len(h):
        check_epoch_seconds(h.ts, "pmus price-history ts")
    if not len(j):
        return None, h
    for c in ("lg", "slug", "ticker", "start", "k_close", "date", "result"):
        j[c] = getattr(r, c)
    j["n_ktrades"] = len(tr)
    return j, h


def cmd_build(split: str) -> int:
    if split == "sealed" and "--gate-ok" not in sys.argv:
        print("REFUSED: sealed is built only after edgelab reports a dev pass (re-run with --gate-ok)")
        return 2
    m = pd.read_parquet(D / "match.parquet")
    m = m[(m.split == split) & m.result.notna()]
    rng = np.random.default_rng(SEED[split])
    m = m.iloc[np.sort(rng.choice(len(m), size=min(N_PER_SPLIT, len(m)), replace=False))]
    print(f"{split}: {len(m)} sampled games", m.lg.value_counts().to_dict(), flush=True)
    js, hs = [], []
    with ThreadPoolExecutor(6) as ex:
        for i, (j, h) in enumerate(ex.map(_one, m.itertuples())):
            if j is not None:
                js.append(j)
            if h is not None and len(h):
                hs.append(h.assign(slug=m.iloc[i].slug))
            if i % 50 == 0:
                print(f"  {i}/{len(m)} jumps so far {sum(len(x) for x in js)}", flush=True)
    f = pd.concat(js, ignore_index=True)
    hh = pd.concat(hs, ignore_index=True)
    f.to_parquet(D / f"frame_{split}.parquet", index=False)
    hh.to_parquet(D / f"pm_{split}.parquet", index=False)
    print(f"frame_{split}: {len(f)} jumps in {f.slug.nunique()} games; pm obs {len(hh)} in {hh.slug.nunique()} games")
    return 0


def cmd_describe(split: str) -> int:
    """Staleness without outcomes: PM US side ask at the first minute obs in [t+3, t+10] vs the last obs <= t."""
    f = pd.read_parquet(D / f"frame_{split}.parquet")
    hh = pd.read_parquet(D / f"pm_{split}.parquet")
    H = {s: g.sort_values("ts") for s, g in hh.groupby("slug")}
    rows = []
    for r in f.itertuples():
        h = H.get(r.slug)
        if h is None:
            continue
        ts = h.ts.to_numpy()
        ask = (h.long if r.sgn > 0 else h.short).to_numpy()
        pre = np.searchsorted(ts, r.t_jump, side="right") - 1
        for lo, hi in ((3, 10), (10, 20), (20, 30), (30, 60), (60, 120)):
            k = np.flatnonzero((ts >= r.t_jump + lo) & (ts <= r.t_jump + hi))
            if len(k) and pre >= 0:
                kp = r.p_jump if r.sgn > 0 else 1 - r.p_jump
                rows.append({"win": f"{lo}-{hi}s", "d_ask": ask[k[0]] - ask[pre], "kal_minus_ask": kp - ask[k[0]],
                             "kal_jump": abs(r.p_jump - r.p_ref)})
    x = pd.DataFrame(rows)
    print(x.groupby("win").agg(n=("d_ask", "size"), ask_move=("d_ask", "mean"), kal_jump=("kal_jump", "mean"),
                               kal_minus_ask=("kal_minus_ask", "mean")).round(4))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "match"
    if cmd == "match":
        raise SystemExit(cmd_match())
    if cmd == "build":
        raise SystemExit(cmd_build(sys.argv[2]))
    if cmd == "describe":
        raise SystemExit(cmd_describe(sys.argv[2]))
    raise SystemExit(__doc__)
