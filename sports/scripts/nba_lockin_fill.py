"""nba_lockin_fill (agent, 2026-11-11): practical-lock moneyline grid -- comeback risk + first-in-queue 99c fills.
Pre-registration: notes/nba_lockin_fill.md. Never reads outcomes dated >= 2026-02-01 (sealed window).

    python scripts/nba_lockin_fill.py grid      # comeback rate per (L, t) cell, pre-Kalshi seasons + dev
    python scripts/nba_lockin_fill.py fills     # dev Kalshi candles/tape at the first state in each cell
    python scripts/nba_lockin_fill.py horn      # live logger: reachable locks at the horn
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint as pc

import traps

OUT = ROOT / "results" / "nba_lockin_fill"
OUT.mkdir(parents=True, exist_ok=True)
SEALED_START = pd.Timestamp("2026-02-01")
DEV = (pd.Timestamp("2025-10-21"), pd.Timestamp("2026-01-31"))
LS, TS = (5, 7, 9, 12), (120, 60, 30, 15)
if len(sys.argv) > 3:
    LS, TS = tuple(map(int, sys.argv[2].split(","))), tuple(map(int, sys.argv[3].split(",")))
TAG = "" if len(sys.argv) <= 3 else "_early"
LAT = 60


def states() -> pd.DataFrame:
    s = pd.read_parquet(ROOT / "data" / "nba_late_states.parquet")
    s = s[s["date"] < SEALED_START].sort_values(["eid", "seq"], kind="stable").copy()
    s["m"] = s["hs"] - s["as"]
    s["wall_ts"] = traps.epoch_seconds(s["wall"])
    return s


def first_in_cell(x: pd.DataFrame, L: int, t: int) -> pd.DataFrame:
    c = x[(x["period"] == 4) & (x["clock_s"] <= t) & (x["clock_s"] > 0) & (x["m"].abs() >= L)]
    f = c.groupby("eid").head(1).copy()
    f["lead_home"] = f["m"] > 0
    f["lose"] = (f["lead_home"] & (f["home_won"] == 0)) | (~f["lead_home"] & (f["home_won"] == 1))
    return f


def grid() -> None:
    s = states()
    last = s.groupby("eid").tail(1)
    agree = ((last["hs"] > last["as"]).astype(int) == last["home_won"]).mean()
    print(f"games {len(last)}; home_won agrees with last ESPN score {agree:.4f}; last clock 0 {(last['clock_s'] == 0).mean():.4f}")
    rows = []
    for per, (lo, hi) in {"pre": (pd.Timestamp("2021-01-01"), DEV[0]), "dev": (DEV[0], SEALED_START)}.items():
        x = s[(s["date"] >= lo) & (s["date"] < hi)]
        ng = x[x["period"] == 4]["eid"].nunique()
        for L in LS:
            for t in TS:
                f = first_in_cell(x, L, t)
                n, k = len(f), int(f["lose"].sum())
                wp = np.where(f["lead_home"], f["espn_wp"], 1 - f["espn_wp"])
                rows.append(dict(per=per, L=L, t=t, n=n, share_games=round(n / ng, 3), losses=k, rate=k / max(n, 1),
                                 wilson_ub=pc(k, n, method="wilson")[1] if n else np.nan,
                                 espn_wp_med=float(np.nanmedian(wp)) if n else np.nan))
    o = pd.DataFrame(rows)
    o.to_csv(OUT / f"comeback_grid{TAG}.csv", index=False)
    print(o.round(4).to_string())


def fills() -> None:
    import lock_in as LI
    s = states()
    s = s[(s["date"] >= DEV[0]) & (s["date"] <= DEV[1])]
    m = pd.read_parquet(ROOT / "data" / "lock_in" / "markets.parquet")
    s["teams"] = s["away"].map(LI.k) + s["home"].map(LI.k)
    j = LI.join(m[m["series"] == "KXNBAGAME"], s)
    j = j[j["event"].str.startswith("KXNBAGAME")]
    ml = m[m["series"] == "KXNBAGAME"].set_index("ticker")
    rows = []
    for L in LS:
        for t in TS:
            f = first_in_cell(s[s["eid"].isin(j["eid"])], L, t).merge(j, on="eid")
            for r in f.itertuples(index=False):
                lead = LI.k(r.home if r.lead_home else r.away)
                tk = f"{r.event}-{lead}"
                if tk not in ml.index:
                    continue
                close = int(ml.loc[tk, "close_ts"])
                T = int(r.wall_ts) + LAT
                if not 0 < close - T < 7200:
                    continue
                rows.append(dict(L=L, t=t, event=r.event, ticker=tk, T=T, close_ts=close, clock_s=r.clock_s, m=abs(r.m),
                                 kalshi_lose=ml.loc[tk, "result"] != "yes", espn_lose=bool(r.lose)))
    base = pd.DataFrame(rows)
    print("cell-entries", len(base), "unique tickers", base["ticker"].nunique())
    cache: dict = {}

    def one(tk_T):
        import time
        tk, T, close = tk_T
        for k in range(6):
            try:
                c = LI.candles(tk, T - 600, close + 60)
                tr = LI.trades(tk, T - 600, close + 60)
                return tk, T, c, tr
            except Exception:  # noqa: BLE001  (429 rate limit)
                time.sleep(5 * (k + 1))
        raise RuntimeError(f"fetch failed {tk}")

    from concurrent.futures import ThreadPoolExecutor
    keys = list({(r.ticker, r.T, r.close_ts) for r in base.itertuples()})
    with ThreadPoolExecutor(3) as ex:
        for tk, T, c, tr in ex.map(one, keys):
            cache[(tk, T)] = (c, tr)
    out = []
    for r in base.itertuples(index=False):
        c, tr = cache[(r.ticker, r.T)]
        q = c[c["end"] <= r.T].dropna(subset=["bid"])
        bid = float(q["bid"].iloc[-1]) if len(q) else np.nan
        qa = c[c["end"] <= r.T].dropna(subset=["ask"])
        ask = float(qa["ask"].iloc[-1]) if len(qa) else np.nan
        after = tr[tr["ts"] > r.T]
        v99s = after.loc[(after["yes_price"].round(2) == 0.99) & (after["side"] == "no"), "count"].sum()
        v99 = after.loc[after["yes_price"].round(2) == 0.99, "count"].sum()
        vlt = after.loc[after["yes_price"] < 0.985, "count"].sum()
        # first time the best bid reached 99 (candle close) after T
        b99 = c[(c["end"] > r.T) & (c["bid"] >= 0.99 - 1e-9)]
        out.append(dict(r._asdict(), bid=bid, ask=ask, v99_sell=v99s, v99_all=v99, v_below99=vlt,
                        min_px_after=after["yes_price"].min() if len(after) else np.nan,
                        s_to_bid99=(int(b99["end"].iloc[0]) - r.T) if len(b99) else np.nan,
                        s_to_close=r.close_ts - r.T))
    x = pd.DataFrame(out)
    x.to_parquet(OUT / f"fills_dev{TAG}.parquet")
    print("bid/ask units check:", x["bid"].describe().round(3).to_dict())
    x["first_q"] = (x["bid"] < 0.99 - 1e-9) & (x["ask"] > 0.99 + 1e-9)
    x["cross"] = x["ask"] <= 0.99 + 1e-9
    x["fill100"] = x["first_q"] & (x["v99_sell"] >= 100)
    g = x.groupby(["L", "t"]).agg(n=("ticker", "size"), kalshi_losses=("kalshi_lose", "sum"), espn_losses=("espn_lose", "sum"),
                                 bid_med=("bid", "median"), ask_med=("ask", "median"), first_q=("first_q", "mean"),
                                 cross=("cross", "mean"), fill_ok=("fill100", "mean"),
                                 v99_sell_med=("v99_sell", "median"), s_to_bid99_med=("s_to_bid99", "median"),
                                 s_to_close_med=("s_to_close", "median"))
    g.to_csv(OUT / f"fills_grid{TAG}.csv")
    print(g.round(3).to_string())
    fo = x[x["fill100"]]
    print("losses among fill_ok entries:", int(fo["kalshi_lose"].sum()), "/", len(fo),
          "| among not fill_ok:", int(x.loc[~x["fill100"], "kalshi_lose"].sum()), "/", int((~x["fill100"]).sum()))


def onset() -> None:
    """When does the 99c YES bid first appear on a KXNBAGAME ticker (1-min candle close), what was the game state then,
    and how did it end (Kalshi result)? Being FIRST at 99c = posting 99c at that moment or earlier."""
    import lock_in as LI
    from concurrent.futures import ThreadPoolExecutor
    s = states()
    s = s[(s["date"] >= DEV[0]) & (s["date"] <= DEV[1])]
    m = pd.read_parquet(ROOT / "data" / "lock_in" / "markets.parquet")
    s["teams"] = s["away"].map(LI.k) + s["home"].map(LI.k)
    j = LI.join(m[m["series"] == "KXNBAGAME"], s)
    j = j[j["event"].str.startswith("KXNBAGAME")]
    q4 = s[s["period"] >= 4].groupby("eid")["wall_ts"].min()
    ml = m[(m["series"] == "KXNBAGAME") & m["event"].isin(j["event"])].merge(j[["event", "eid"]], on="event")
    ml = ml[ml["eid"].isin(q4.index)]

    def one(r):
        a, close = int(q4[r.eid]) - 300, int(r.close_ts)
        if not 0 < close - a < 4 * 3600:
            return None
        c = LI.candles(r.ticker, a, close + 60)
        on = c[c["bid"] >= 0.99 - 1e-9]
        if not len(on):
            return dict(ticker=r.ticker, eid=r.eid, onset=np.nan, close_ts=close, result=r.result)
        t0 = int(on["end"].iloc[0]) - 60          # the candle in which the bid first closed at 99
        tr = LI.trades(r.ticker, t0, close + 60)
        a99 = tr[(tr["yes_price"].round(2) == 0.99) & (tr["side"] == "no")]
        v5 = a99.loc[a99["ts"] <= t0 + 300, "count"].sum()
        return dict(ticker=r.ticker, eid=r.eid, onset=t0, close_ts=close, result=r.result, v99_sell=a99["count"].sum(),
                    v99_sell_5min=v5, team=r.ticker.rsplit("-", 1)[1])

    with ThreadPoolExecutor(8) as ex:
        res = [x for x in ex.map(one, ml.itertuples(index=False)) if x]
    x = pd.DataFrame(res)
    # game state at onset (last ESPN state with wall <= onset)
    st = []
    for r in x.itertuples(index=False):
        g = s[s["eid"] == r.eid]
        if np.isnan(r.onset):
            st.append(dict(period=np.nan, clock_s=np.nan, lead=np.nan, wp=np.nan))
            continue
        g0 = g[g["wall_ts"] <= r.onset]
        if not len(g0):
            st.append(dict(period=3, clock_s=np.nan, lead=np.nan, wp=np.nan))
            continue
        z = g0.iloc[-1]
        home = LI.k(z["home"]) == r.team
        st.append(dict(period=int(z["period"]), clock_s=float(z["clock_s"]), lead=float(z["m"] if home else -z["m"]),
                       wp=float(z["espn_wp"] if home else 1 - z["espn_wp"])))
    x = pd.concat([x.reset_index(drop=True), pd.DataFrame(st)], axis=1)
    x["s_before_close"] = x["close_ts"] - x["onset"]
    x.to_parquet(OUT / "onset_dev.parquet")
    y = x[x["onset"].notna()]
    print("tickers", len(x), "with a 99c bid ever", len(y), "| lost (result no):", int((y["result"] != "yes").sum()))
    print("state at onset: period", y["period"].value_counts().to_dict())
    print(y[["clock_s", "lead", "wp", "s_before_close", "v99_sell", "v99_sell_5min"]].describe(
        percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
    print(y[y["result"] != "yes"][["ticker", "period", "clock_s", "lead", "wp", "v99_sell"]].to_string())


def horn() -> None:
    rows = [json.loads(line) for line in open(Path.home() / ".claude/quant-sports/data/live/lock_queue.jsonl")]
    sn = pd.DataFrame([r for r in rows if r["kind"] == "snapshot"])
    z = sn[sn["delay_s"] == 0]
    print(z.groupby(["series", "market_status"]).agg(n=("ticker", "size"), q_le5k=("yes_99", lambda v: int((v <= 5000).sum())),
                                                     q_med=("yes_99", "median")).to_string())
    act = z[z["market_status"] == "active"]
    print("active books with q <= 5,000:", int((act["yes_99"] <= 5000).sum()), "of", len(act))
    print(act.groupby("series")["yes_99"].describe().round(0).to_string())


if __name__ == "__main__":
    {"grid": grid, "fills": fills, "horn": horn, "onset": onset}[sys.argv[1]]()
