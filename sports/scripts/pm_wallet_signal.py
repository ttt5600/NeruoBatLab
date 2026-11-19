"""Skilled-Polymarket-wallet trades as a Kalshi signal on the SAME game (notes/pm_wallet_signal.md has the pre-reg).

    python scripts/pm_wallet_signal.py pull SPLIT     # PM taker fills >= $100 for matched games (train|dev|sealed)
    python scripts/pm_wallet_signal.py results        # Kalshi settlement of every matched team ticker in pulled splits
    python scripts/pm_wallet_signal.py skill SPLIT    # skilled wallets from games before SPLIT start
    python scripts/pm_wallet_signal.py events SPLIT [CASH]  # events + Kalshi 1-min candles -> frame_SPLIT.parquet
    python scripts/pm_wallet_signal.py describe SPLIT # price path, kill rules K1/K2, CLV / ROI with game-bootstrap CIs

Read-only GETs only: Polymarket data-api v2 /trades (public), Kalshi via scripts/kx_throttle.py. No orders.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))

import numpy as np
import pandas as pd
import requests

from kx_throttle import kget

D = ROOT / "data" / "pm_wallet_signal"
D.mkdir(parents=True, exist_ok=True)
R = ROOT / "results" / "pm_wallet_signal"
R.mkdir(parents=True, exist_ok=True)
MATCH = ROOT / "data" / "xvenue_ingame" / "match.parquet"
KMK = ROOT / "data" / "xvenue_ingame" / "kalshi_markets.parquet"
PTR = "https://data-api.polymarket.com/v2/trades"
SPLITS = {"train": ("2025-04-01", "2025-12-31"), "dev": ("2026-01-01", "2026-04-30"),
          "sealed": ("2026-05-01", "2026-11-05")}
KSER = {"nba": "KXNBAGAME", "nfl": "KXNFLGAME", "nhl": "KXNHLGAME", "mlb": "KXMLBGAME", "cbb": "KXNCAAMBGAME",
        "cfb": "KXNCAAFGAME"}
MIN_CASH_PULL = 100
# skill definition (pre-registered)
MIN_GAMES, MIN_RISK, MAX_TWOSIDED, Z_SKILL = 20, 5000.0, 0.5, 2.5
EVENT_CASH = 500.0
DEDUPE_S = 600
_S = requests.Session()
_S.headers["User-Agent"] = "research-readonly"


def games(split: str | None = None) -> pd.DataFrame:
    m = pd.read_parquet(MATCH)
    m["gdate"] = pd.to_datetime(m.slug.str[-10:], format="%Y-%m-%d", errors="coerce")
    k = pd.read_parquet(KMK)[["ticker", "close_time"]]
    m = m.merge(k, on="ticker", how="left")
    m["kclose"] = pd.to_datetime(m.close_time, utc=True, format="ISO8601")
    m = m.dropna(subset=["gdate", "kclose"])
    if split:
        a, b = SPLITS[split]
        m = m[(m.gdate >= a) & (m.gdate <= b)]
    return m.reset_index(drop=True)


def _pget(params, tries=8):
    for i in range(tries):
        try:
            r = _S.get(PTR, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            print("  pm http", r.status_code, r.text[:120], flush=True)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 * (i + 1))
                continue
            return None
        except requests.RequestException as ex:
            print("  pm exc", type(ex).__name__, flush=True)
            time.sleep(2 * (i + 1))
    return None


def _pull_batch(cs):
    base = {"condition": ",".join(cs), "limit": 1000, "filter_type": "CASH", "filter_amount": MIN_CASH_PULL}
    q, rows, pages, ok = dict(base), [], 0, True
    while True:
        j = _pget(q)
        if j is None:
            ok = False
            break
        for x in j.get("data", []):
            rows.append({k: x.get(k) for k in ("proxy_wallet", "side", "token_id", "condition_id", "size",
                                                "price", "timestamp", "outcome_index", "transaction_hash")})
        pages += 1
        cur = (j.get("pagination") or {}).get("next_cursor")
        if not cur or not j.get("data"):
            break
        q = {**base, "cursor": cur}
    return cs, rows, pages, ok


def cmd_pull(split: str) -> int:
    from concurrent.futures import ThreadPoolExecutor
    g = games(split)
    out = D / f"trades_{split}.parquet"
    donef = D / f"done_{split}.json"
    done = set(json.loads(donef.read_text())) if donef.exists() else set()
    todo = [c for c in g.cond if c not in done]
    print(f"{split}: {len(g)} games, {len(todo)} to pull", flush=True)
    batches = [todo[i:i + 20] for i in range(0, len(todo), 20)]
    acc, t0 = [], time.time()

    def flush():
        if not acc:
            return
        t = pd.DataFrame(acc)
        if out.exists():
            t = pd.concat([pd.read_parquet(out), t], ignore_index=True)
        t = t.drop_duplicates(["transaction_hash", "proxy_wallet", "token_id", "size", "price", "side"])
        t.to_parquet(out, index=False)
        donef.write_text(json.dumps(sorted(done)))
        acc.clear()
    with ThreadPoolExecutor(max_workers=4) as ex:
        for k, (cs, rows, pages, ok) in enumerate(ex.map(_pull_batch, batches)):
            if ok:
                acc.extend(rows)
                done.update(cs)
            else:
                print("  batch failed (will retry next run)", flush=True)
            if k % 10 == 0:
                print(f"  batch {k + 1}/{len(batches)} pages {pages} rows {len(rows)} {time.time() - t0:.0f}s", flush=True)
            if k % 25 == 24:
                flush()
    flush()
    t = pd.read_parquet(out)
    print(f"saved {out} rows {len(t)} conditions {t.condition_id.nunique()} done {len(done)}/{len(g)}")
    return 0


def cmd_results() -> int:
    out = D / "results.parquet"
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["ticker", "result"])
    g = pd.concat([games(s) for s in SPLITS if (D / f"trades_{s}.parquet").exists()])
    todo = sorted(set(g.ticker) - set(have.ticker))
    rows = []
    for i in range(0, len(todo), 100):
        ts = todo[i:i + 100]
        got = {}
        for path in ("/historical/markets", "/markets"):
            left = [t for t in ts if t not in got]
            if not left:
                break
            j = kget(path, {"tickers": ",".join(left), "limit": 1000}) or {}
            for mk in j.get("markets", []):
                got[mk["ticker"]] = mk.get("result")
        rows += [{"ticker": t, "result": got.get(t)} for t in ts]
    r = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    r.to_parquet(out, index=False)
    print(r.result.value_counts(dropna=False))
    return 0


def fills(split: str) -> pd.DataFrame:
    """PM fills joined to the game; bull = toward the matched (Kalshi) team; v = team payout (1/0/0.5)."""
    t = pd.read_parquet(D / f"trades_{split}.parquet")
    g = games(split)
    t = t.merge(g[["cond", "tok", "ticker", "event_ticker", "lg", "gdate", "kclose", "slug"]],
                left_on="condition_id", right_on="cond")
    t["ts"] = t.timestamp.astype(np.int64)
    t["team_tok"] = t.token_id.astype(str) == t.tok.astype(str)
    t["bull"] = (t.team_tok & (t.side == "BUY")) | (~t.team_tok & (t.side == "SELL"))
    t["cash"] = t["size"] * t["price"]
    t["kclose_s"] = (t.kclose - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    return t


def _resolved(t: pd.DataFrame) -> pd.DataFrame:
    r = pd.read_parquet(D / "results.parquet")
    t = t.merge(r, on="ticker", how="left")
    t["v_team"] = t.result.map({"yes": 1.0, "no": 0.0})
    return t.dropna(subset=["v_team"])


def wallet_stats(t: pd.DataFrame) -> pd.DataFrame:
    t = t[(t.price >= 0.02) & (t.price <= 0.98) & (t.ts < t.kclose_s)].copy()
    v_tok = np.where(t.team_tok, t.v_team, 1 - t.v_team)
    buy = t.side == "BUY"
    t["pnl"] = np.where(buy, t["size"] * (v_tok - t.price), t["size"] * (t.price - v_tok))
    t["risk"] = np.where(buy, t["size"] * t.price, t["size"] * (1 - t.price))
    pg = t.groupby(["proxy_wallet", "cond"]).agg(pnl=("pnl", "sum"), risk=("risk", "sum"), nb=("bull", "sum"),
                                                 n=("bull", "size")).reset_index()
    pg["two"] = (pg.nb > 0) & (pg.nb < pg.n)
    w = pg.groupby("proxy_wallet").agg(games=("cond", "size"), fills=("n", "sum"), pnl=("pnl", "sum"),
                                       risk=("risk", "sum"), two=("two", "mean"))
    w["roi"] = w.pnl / w.risk
    pg = pg.join(w.roi, on="proxy_wallet")
    pg["e2"] = (pg.pnl - pg.roi * pg.risk) ** 2
    w["se"] = np.sqrt(pg.groupby("proxy_wallet").e2.sum()) / w.risk
    w["z"] = w.roi / w.se
    w["eligible"] = (w.games >= MIN_GAMES) & (w.risk >= MIN_RISK) & (w.two <= MAX_TWOSIDED)
    return w


def cmd_skill(split: str) -> int:
    src = {"dev": ["train"], "sealed": ["train", "dev"]}[split]
    cut = pd.Timestamp(SPLITS[split][0])
    t = pd.concat([fills(s) for s in src], ignore_index=True)
    t = _resolved(t[t.gdate < cut])
    w = wallet_stats(t)
    w.to_parquet(D / f"wallets_for_{split}.parquet")
    e = w[w.eligible]
    print(f"skill window games {t.cond.nunique()} fills {len(t)} wallets {len(w)} eligible {len(e)}")
    for z in (2.0, Z_SKILL, 3.0):
        s = e[(e.z >= z) & (e.roi > 0)]
        print(f"  z>={z}: skilled {len(s)}  (expected by chance one-sided ~{len(e) * {2.0: .0228, 2.5: .0062, 3.0: .00135}[z]:.0f})")
    print("eligible roi quantiles", e.roi.quantile([.1, .25, .5, .75, .9]).round(4).to_dict())
    print("pooled eligible roi (sum pnl/sum risk)", round(e.pnl.sum() / e.risk.sum(), 4))
    s = e[(e.z >= Z_SKILL) & (e.roi > 0)].sort_values("z", ascending=False)
    print(s.head(15).round(4).to_string())
    return 0


def skilled(split: str, z: float = Z_SKILL) -> set:
    w = pd.read_parquet(D / f"wallets_for_{split}.parquet")
    return set(w[w.eligible & (w.z >= z) & (w.roi > 0)].index)


def _candles(lg, ticker, s0, e0):
    cache = D / "candles" / f"{ticker}_{s0}_{e0}.json"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        return json.loads(cache.read_text())
    kc = []
    for path in (f"/historical/markets/{ticker}/candlesticks", f"/series/{KSER[lg]}/markets/{ticker}/candlesticks"):
        j = kget(path, {"start_ts": s0, "end_ts": e0, "period_interval": 1})
        if j and j.get("candlesticks"):
            kc = j["candlesticks"]
            break

    def px(x, k, f="close"):
        d = x.get(k) or {}
        v = d.get(f + "_dollars", d.get(f))
        return None if v is None else float(v)
    out = [{"t": int(x["end_period_ts"]), "bid": px(x, "yes_bid"), "ask": px(x, "yes_ask")} for x in kc]
    cache.write_text(json.dumps(out))
    return out


def cmd_events(split: str, cash: float = EVENT_CASH, z: float = Z_SKILL) -> int:
    sk = skilled(split, z)
    t = fills(split)
    t = t[t.proxy_wallet.isin(sk) & (t.cash >= cash) & (t.ts <= t.kclose_s - 120)].sort_values("ts")
    keep, last = [], {}
    for r in t.itertuples():
        k = (r.cond, r.bull)
        if k in last and r.ts - last[k] < DEDUPE_S:
            continue
        last[k] = r.ts
        keep.append(r.Index)
    ev = t.loc[keep].reset_index(drop=True)
    print(f"{split}: skilled {len(sk)}, fills >= ${cash:.0f} {len(t)}, events after dedupe {len(ev)} in {ev.cond.nunique()} games")
    rows = []
    for i, r in enumerate(ev.itertuples()):
        s0 = (r.ts // 60) * 60 - 31 * 60
        e0 = min(s0 + 160 * 60, int(r.kclose_s) + 60)
        c = _candles(r.lg, r.ticker, int(s0), int(e0))
        if not c:
            continue
        cd = pd.DataFrame(c)
        rows.append({"bet_key": f"{r.cond}|{r.ts}|{int(r.bull)}", "cond": r.cond, "ticker": r.ticker,
                     "event_ticker": r.event_ticker, "lg": r.lg, "date": r.gdate, "t": int(r.ts),
                     "wallet": r.proxy_wallet, "bull": bool(r.bull), "cash": float(r.cash), "pm_price": float(r.price),
                     "kclose_s": int(r.kclose_s), "candles": json.dumps(c)})
        if i % 100 == 0:
            print(f"  {i}/{len(ev)}", flush=True)
    f = pd.DataFrame(rows)
    tag = "" if (cash == EVENT_CASH and z == Z_SKILL) else f"_c{int(cash)}_z{z}"
    f.to_parquet(D / f"frame_{split}{tag}.parquet", index=False)
    print(f"saved frame_{split}{tag}.parquet rows {len(f)}")
    return 0


# ---------------------------------------------------------------- rule lives in edgelab.rules.pm_wallet
sys.path.insert(0, str(ROOT))
from edgelab.rules.pm_wallet import bet_rows  # noqa: E402


def _boot(df, f, B=2000, key="event_ticker", seed=0):
    rng = np.random.default_rng(seed)
    gs = {k: g for k, g in df.groupby(key)}
    ks = list(gs)
    vals = []
    for _ in range(B):
        s = rng.choice(len(ks), len(ks))
        vals.append(f(pd.concat([gs[ks[i]] for i in s])))
    return np.percentile(vals, [2.5, 97.5])


def cmd_describe(split: str, tag: str = "") -> int:
    f = pd.read_parquet(D / f"frame_{split}{tag}.parquet")
    b = bet_rows(f).merge(f[["bet_key", "event_ticker", "lg", "wallet", "cash"]], on="bet_key")
    r = pd.read_parquet(D / "results.parquet")
    b = b.merge(r, on="ticker", how="left")
    b["won"] = np.where(b.side == "yes", b.result == "yes", b.result == "no").astype(float)
    b.loc[~b.result.isin(["yes", "no"]), "won"] = np.nan
    fee = np.ceil(0.07 * b.price * (1 - b.price) * 100 - 1e-9) / 100
    b["cost"] = b.price + fee
    b["pnl"] = b.won - b.cost
    b["clv"] = b.close_side_mid - b.cost
    print(f"{split}{tag}: bets {len(b)} games {b.event_ticker.nunique()} days {b.date.nunique()}  by lg {b.lg.value_counts().to_dict()} wallets {b.wallet.nunique()}")
    cols = ["d-30", "d-5", "d-1", "d_entry", "d1", "d5", "d30", "d120"]
    print("mean side-mid change vs t (cents):", {c: round(100 * b[c].mean(), 2) for c in cols})
    pre, post5, ent = b["d-5"].mean(), b["d5"].mean(), b.d_entry.mean()
    print(f"K1 already moved: pre[-5,0] {100*pre:.2f}c vs post[0,+5] {100*post5:.2f}c -> {'KILL' if pre >= post5 else 'ok'}")
    print(f"K2 reprice <=60s: entry {100*ent:.2f}c vs 0.8*post5 {80*post5:.2f}c -> {'KILL' if ent >= 0.8 * post5 else 'ok'}")
    ci = _boot(b, lambda d: d.d5.mean(), B=1000)
    print(f"d5 CI [{100*ci[0]:.2f},{100*ci[1]:.2f}]c; d-5 CI {np.round(100*_boot(b, lambda d: d['d-5'].mean(), B=1000),2)}c")
    bb = b.dropna(subset=["won"])
    roi = bb.pnl.sum() / bb.cost.sum()
    print(f"ROI {100*roi:.2f}% CI {np.round(100*_boot(bb, lambda d: d.pnl.sum()/d.cost.sum()),2)} n {len(bb)}")
    print(f"CLV {100*b.clv.mean():.2f}c CI {np.round(100*_boot(b.dropna(subset=['clv']), lambda d: d.clv.mean()),2)}")
    b.to_parquet(D / f"bets_{split}{tag}.parquet", index=False)
    return 0


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "pull":
        sys.exit(cmd_pull(sys.argv[2]))
    if c == "results":
        sys.exit(cmd_results())
    if c == "skill":
        sys.exit(cmd_skill(sys.argv[2]))
    if c == "events":
        sys.exit(cmd_events(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else EVENT_CASH,
                            float(sys.argv[4]) if len(sys.argv) > 4 else Z_SKILL))
    if c == "describe":
        sys.exit(cmd_describe(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""))
