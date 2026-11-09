"""Lock-in study: Kalshi markets whose outcome is already determined by the public record but which still trade.

    python scripts/lock_in.py markets        # all KXNBATOTAL / KXNBAGAME markets, joined to ESPN games (no prices)
    python scripts/lock_in.py measure [N]    # descriptive on N dev games: how long / at what price / how many contracts
    python scripts/lock_in.py build dev|sealed      # frame + tape for edgelab (loader lock_in_nba_total)

Determination:
  NBA total over rung K (market "over K", K = x.5): running score hs + as from ESPN play-by-play. YES is locked once
  hs + as >= K + 0.5. The rule uses the safe version hs + as >= K + 3.5, so one reversed 3-pointer cannot undo it.
  NBA game winner: last ESPN play (final horn). ESPN play `wall` = wallclock of the play, so data latency is added on top.
Only period >= 4 states are in data/nba_late_states.parquet; a rung locked earlier gets the first Q4 state as its lock.
Outcome columns are written only by `build` and stay hidden in the vault.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))

import numpy as np
import pandas as pd

import traps
from sports import http as H
from sports.http import fetch

H.MIN_INTERVAL = 0.08
A = "https://api.elections.kalshi.com/trade-api/v2"
D = ROOT / "data" / "lock_in"
D.mkdir(parents=True, exist_ok=True)
DEV = (pd.Timestamp("2025-10-21"), pd.Timestamp("2026-01-31"))
SEALED = (pd.Timestamp("2026-02-01"), pd.Timestamp("2026-06-14"))
LAT = 60          # seconds from ESPN play wallclock to our decision
MARGIN = 3.5      # hs + as >= K + MARGIN
ALIAS = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
MON = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def k(x: str) -> str:
    return ALIAS.get(x, x)


def page(path: str, params: dict, key: str) -> list:
    out, cur = [], None
    while True:
        p = dict(params, limit=1000)
        if cur:
            p["cursor"] = cur
        r = fetch(A + path, p, cache=True)
        out += r.get(key, [])
        cur = r.get("cursor")
        if not cur or not r.get(key):
            return out


def markets() -> pd.DataFrame:
    rows = []
    for s in ("KXNBATOTAL", "KXNBAGAME"):
        for m in page("/historical/markets", {"series_ticker": s}, "markets"):
            rows.append(dict(series=s, ticker=m["ticker"], event=m["event_ticker"], K=m.get("floor_strike"),
                             strike=m.get("strike_type"), close=m["close_time"], settle=m.get("settlement_ts"),
                             result=m.get("result"), expiration_value=m.get("expiration_value"),
                             volume=float(m.get("volume_fp") or 0)))
    x = pd.DataFrame(rows).drop_duplicates("ticker")
    tok = x["event"].str.split("-").str[1]
    x["date"] = pd.to_datetime("20" + tok.str[:2] + "-" + tok.str[2:5].map(MON).astype(str) + "-" + tok.str[5:7])
    x["teams"] = tok.str[7:]
    x["close_ts"] = traps.epoch_seconds(pd.to_datetime(x["close"], utc=True, format="ISO8601"))
    x = x[(x["date"] >= DEV[0]) & (x["date"] <= SEALED[1])]
    x.to_parquet(D / "markets.parquet")
    print(x.groupby("series").size().to_dict(), x["date"].min(), x["date"].max())
    return x


def games() -> pd.DataFrame:
    s = pd.read_parquet(ROOT / "data" / "nba_late_states.parquet",
                        columns=["eid", "seq", "period", "clock_s", "hs", "as", "wall", "home", "away", "date"])
    s = s[(s["date"] >= DEV[0]) & (s["date"] <= SEALED[1])].sort_values(["eid", "seq"], kind="stable").copy()
    s["sum"] = s["hs"] + s["as"]
    s["wall_ts"] = traps.epoch_seconds(s["wall"])
    s["teams"] = s["away"].map(k) + s["home"].map(k)
    return s


def join(m: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    g = s.groupby("eid").agg(date=("date", "first"), teams=("teams", "first")).reset_index()
    # ESPN `date` may be the UTC date; match on teams within +/- 1 day of the ticker's ET date
    ev = m.drop_duplicates("event")[["event", "date", "teams"]].rename(columns={"date": "tdate"})
    j = ev.merge(g, on="teams", how="inner", validate="many_to_many")
    j = j[(j["date"] - j["tdate"]).abs() <= pd.Timedelta(days=1)]
    dup = j.groupby("event").size()
    print("events", len(ev), "joined", j["event"].nunique(), "ambiguous", int((dup > 1).sum()))
    return j[j["event"].map(dup) == 1][["event", "eid", "tdate"]]


def candles(t: str, a: int, b: int) -> pd.DataFrame:
    r = fetch(f"{A}/historical/markets/{t}/candlesticks", dict(start_ts=a, end_ts=b, period_interval=1), cache=True)
    rows = []
    for c in r.get("candlesticks", []):
        yb, ya = c.get("yes_bid") or {}, c.get("yes_ask") or {}
        rows.append((c["end_period_ts"], yb.get("close"), ya.get("close"), c.get("volume", c.get("volume_fp"))))
    d = pd.DataFrame(rows, columns=["end", "bid", "ask", "vol"])
    for col in ("bid", "ask", "vol"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def trades(t: str, a: int, b: int) -> pd.DataFrame:
    tr = page("/historical/trades", {"ticker": t, "min_ts": a, "max_ts": b}, "trades")
    if not tr:
        return pd.DataFrame(columns=["ts", "yes_price", "count", "side"])
    d = pd.DataFrame({"ts": pd.to_datetime([x["created_time"] for x in tr], utc=True, format="ISO8601"),
                      "yes_price": [float(x.get("yes_price_dollars", x.get("yes_price"))) for x in tr],
                      "count": [float(x.get("count_fp", x.get("count"))) for x in tr],
                      "side": [x.get("taker_side") for x in tr]})
    d["ts"] = traps.epoch_seconds(d["ts"])
    return d


def locks(mk: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """first state per over-rung with sum >= K + MARGIN (inputs only, no prices)."""
    by = {e: g[["sum", "wall_ts", "period", "clock_s"]].to_numpy(float) for e, g in s.groupby("eid")}
    rows = []
    for r in mk.itertuples(index=False):
        a = by[r.eid]
        hit = np.nonzero(a[:, 0] >= float(r.K) + MARGIN)[0]
        if not len(hit):
            continue
        i = hit[0]
        rows.append(dict(ticker=r.ticker, event=r.event, eid=r.eid, K=float(r.K), lock_wall=int(a[i, 1]),
                         first_state=bool(i == 0), period=int(a[i, 2]), clock_s=float(a[i, 3]),
                         game_end=int(a[-1, 1]), final_sum=float(a[-1, 0]), close_ts=int(r.close_ts), tdate=r.tdate))
    return pd.DataFrame(rows)


def measure(n: int) -> None:
    m = pd.read_parquet(D / "markets.parquet")
    s = games()
    j = join(m, s)
    dv = j[j["tdate"] <= DEV[1]].sort_values("event")
    pick = dv.iloc[np.linspace(0, len(dv) - 1, n).astype(int)]
    print("dev events", len(dv), "picked", len(pick))
    t = m[(m["series"] == "KXNBATOTAL") & (m["strike"] == "greater")].merge(pick, on="event")
    l = locks(t, s[s["eid"].isin(pick["eid"])])
    print("total rungs", len(t), "locked in Q4+ data", len(l), "already locked at first Q4 state", int(l["first_state"].sum()))
    rows = []
    for r in l.itertuples(index=False):
        T = r.lock_wall + LAT
        c = candles(r.ticker, T - 300, r.close_ts + 60)
        tr = trades(r.ticker, T, r.close_ts + 60)
        q = c[c["end"] <= T]
        ya = q["ask"].iloc[-1] if len(q) else np.nan
        yb = q["bid"].iloc[-1] if len(q) else np.nan
        at = tr[tr["yes_price"] < 1 - 1e-9]
        rows.append(dict(ticker=r.ticker, eid=r.eid, K=r.K, period=r.period, first=r.first_state, T=T, game_end=r.game_end,
                         close_ts=r.close_ts, ya=ya, yb=yb, n_tr=len(tr), vol=tr["count"].sum(),
                         vol_lt_1=at["count"].sum(), min_px=tr["yes_price"].min() if len(tr) else np.nan,
                         notional_gap=float(((1 - at["yes_price"]) * at["count"]).sum())))
    x = pd.DataFrame(rows)
    x.to_parquet(D / f"measure_totals_{n}.parquet")
    print(x.drop(columns=["ticker", "eid"]).describe().T.to_string())
    print("\nmargin to close (min):", ((x.close_ts - x["T"]) / 60).describe().round(1).to_dict())
    print("ask at T:", x.ya.round(2).value_counts().sort_index().tail(10).to_dict())
    print("per game: cents-left x contracts ($)", round(x.notional_gap.sum() / max(1, x.eid.nunique()), 2))
    # final-horn window: moneyline
    g = m[m["series"] == "KXNBAGAME"].merge(pick, on="event")
    ends = s.groupby("eid")["wall_ts"].max()
    rows = []
    for r in g.drop_duplicates("event").itertuples(index=False):
        T = int(ends[r.eid]) + LAT
        tr = trades(r.ticker, T - 300, int(r.close_ts) + 60)
        rows.append(dict(event=r.event, end_to_close=int(r.close_ts) - int(ends[r.eid]), after=tr[tr.ts >= T]["count"].sum() if len(tr) else 0))
    y = pd.DataFrame(rows)
    print("\nML: seconds from last ESPN play to Kalshi close:", y.end_to_close.describe().round(0).to_dict())
    print("ML contracts traded in [end+60s, close]:", y.after.describe().round(0).to_dict())


def measure_ml(n: int) -> None:
    """After the final horn: both moneyline tickers of each game; the winner comes from the public score only."""
    m = pd.read_parquet(D / "markets.parquet")
    s = games()
    j = join(m, s)
    dv = j[(j["tdate"] <= DEV[1]) & j["event"].str.startswith("KXNBAGAME")].sort_values("event")
    pick = dv.iloc[np.linspace(0, len(dv) - 1, n).astype(int)]
    last = s.groupby("eid").tail(1).set_index("eid")
    rows = []
    for ev, eid in zip(pick["event"], pick["eid"]):
        lst = last.loc[eid]
        if lst["hs"] == lst["as"]:
            continue
        win = k(lst["home"]) if lst["hs"] > lst["as"] else k(lst["away"])
        end = int(lst["wall_ts"])
        for r in m[m["event"] == ev].itertuples(index=False):
            T = end + LAT
            if not -600 <= int(r.close_ts) - end <= 7200:   # misjoined game (ESPN date vs ticker date): skip, counted below
                rows.append(dict(event=ev, ticker=r.ticker, winner=False, end=end, close_ts=int(r.close_ts)))
                continue
            tr = trades(r.ticker, T, int(r.close_ts) + 60)
            c = candles(r.ticker, T - 300, int(r.close_ts) + 60)
            q = c[c["end"] <= T]
            w = r.ticker.rsplit("-", 1)[1] == win
            tr = tr.assign(won_px=tr["yes_price"] if w else 1 - tr["yes_price"])
            rows.append(dict(event=ev, ticker=r.ticker, winner=w, end=end, close_ts=int(r.close_ts),
                             ask=q["ask"].iloc[-1] if len(q) else np.nan, bid=q["bid"].iloc[-1] if len(q) else np.nan,
                             vol=tr["count"].sum(), vol_lt99=tr.loc[tr["won_px"] < 0.985, "count"].sum(),
                             vol_99=tr.loc[(tr["won_px"] >= 0.985) & (tr["won_px"] < 0.999), "count"].sum(),
                             vol_100=tr.loc[tr["won_px"] >= 0.999, "count"].sum(),
                             gap=float(((1 - tr["won_px"]).clip(lower=0) * tr["count"]).sum()),
                             min_px=tr["won_px"].min() if len(tr) else np.nan, last_t=tr["ts"].max() if len(tr) else np.nan))
    x = pd.DataFrame(rows)
    x.to_parquet(D / f"measure_ml_{n}.parquet")
    x["to_close"] = x["close_ts"] - x["end"]
    print(x.drop(columns=["event", "ticker"]).describe().T.to_string())
    ok = x[(x["to_close"] > -60) & (x["to_close"] < 7200)]
    print("games usable", ok["event"].nunique(), "of", x["event"].nunique())
    print("winner ask at end+60:", ok[ok.winner].ask.round(2).value_counts().sort_index().to_dict())
    print("loser bid at end+60:", ok[~ok.winner].bid.round(2).value_counts().sort_index().to_dict())
    g = ok.groupby("event")[["vol", "vol_lt99", "vol_99", "vol_100", "gap"]].sum()
    print(g.describe().T.to_string())
    print("to_close s", ok["to_close"].describe().round(0).to_dict())


def _row(r: dict) -> tuple[dict, pd.DataFrame]:
    T = r["T"]
    c = candles(r["ticker"], T - 300, r["close_ts"] + 60)
    tr = trades(r["ticker"], T, r["close_ts"] + 60)
    q = c[c["end"] <= T].dropna(subset=["bid", "ask"])
    r = dict(r)
    if len(q):
        r.update(q_end=int(q["end"].iloc[-1]), yes_bid=float(q["bid"].iloc[-1]), yes_ask=float(q["ask"].iloc[-1]))
    else:
        r.update(q_end=-1, yes_bid=np.nan, yes_ask=np.nan)
    last = c.dropna(subset=["ask"])
    r["mark_ask"] = float(last["ask"].iloc[-1]) if len(last) else np.nan
    tr = tr.assign(ticker=r["ticker"])[["ticker", "ts", "yes_price", "count"]]
    return r, tr


def build(split: str) -> None:
    from concurrent.futures import ThreadPoolExecutor

    a, b = DEV if split == "dev" else SEALED
    m = pd.read_parquet(D / "markets.parquet")
    s = games()
    j = join(m, s)
    j = j[(j["tdate"] >= a) & (j["tdate"] <= b)]
    print(split, "events", len(j))
    t = m[(m["series"] == "KXNBATOTAL") & (m["strike"] == "greater")].merge(j, on="event")
    l = locks(t, s[s["eid"].isin(j["eid"])])
    rows = []
    for r in l.itertuples(index=False):
        rows.append(dict(ticker=r.ticker, kind="total", event=r.event, eid=r.eid, date=r.tdate, T=int(r.lock_wall) + LAT,
                         input_ts=int(r.lock_wall), close_ts=int(r.close_ts)))
    last = s.groupby("eid").tail(1).set_index("eid")
    bad = 0
    for ev, eid, td in zip(j["event"], j["eid"], j["tdate"]):
        if not ev.startswith("KXNBAGAME"):
            continue
        lst = last.loc[eid]
        if lst["clock_s"] != 0 or lst["hs"] == lst["as"]:
            bad += 1
            continue
        win = k(lst["home"]) if lst["hs"] > lst["as"] else k(lst["away"])
        for r in m[m["event"] == ev].itertuples(index=False):
            if r.ticker.rsplit("-", 1)[1] != win or not -600 <= int(r.close_ts) - int(lst["wall_ts"]) <= 7200:
                continue
            rows.append(dict(ticker=r.ticker, kind="ml", event=ev, eid=eid, date=td, T=int(lst["wall_ts"]) + LAT,
                             input_ts=int(lst["wall_ts"]), close_ts=int(r.close_ts)))
    rows = [r for r in rows if r["T"] < r["close_ts"]]
    print("rows", len(rows), "ml games skipped (no horn / tied)", bad)
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(_row, rows))
    f = pd.DataFrame([x[0] for x in res])
    tape = pd.concat([x[1] for x in res], ignore_index=True)
    outc = m[["ticker", "result", "expiration_value", "volume"]]
    f = traps.checked_merge(f, outc, on="ticker", how="left", validate="one_to_one")
    f.to_parquet(D / f"frame_{split}.parquet")
    tape[tape["yes_price"] <= 0.99 + 1e-9].to_parquet(D / f"tape_{split}.parquet")
    print(f.groupby("kind").size().to_dict(), "tape rows", len(tape))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2])
    if cmd == "ml":
        measure_ml(int(sys.argv[2]))
    if cmd == "markets":
        markets()
    elif cmd == "measure":
        measure(int(sys.argv[2]))
