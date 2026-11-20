"""KXTSAD DEV descriptive: quotes at T = open + 2 h, book shape, per-rung calibration of mid / ask vs outcome. No rule P&L."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from census import load
from quotes import quote_at
E = pd.Timestamp("1970-01-01", tz="UTC")
def frame():
    t = load("KXTSAD"); t = t[t.status == "finalized"].copy()
    t["date"] = pd.to_datetime(t.event_ticker.str.split("-").str[1], format="%y%b%d")
    t["K"] = t.floor_strike.astype(float)
    t["y"] = (t.result == "yes").astype(int)
    t["T"] = ((t.ot + pd.Timedelta(hours=2) - E) // pd.Timedelta(seconds=1)).astype(int)
    c = {json.loads(l)["ticker"]: json.loads(l)["c"] for l in open(OUT / "candles_tsad_dev.jsonl")}
    t = t[t.ticker.isin(c)]
    q = pd.DataFrame([dict(ticker=k, **(quote_at(v, T) or {})) for k, T, v in ((r.ticker, r.T, c[r.ticker]) for r in t.itertuples())])
    t = t.merge(q, on="ticker", how="left", validate="one_to_one")
    t["traded_life"] = t.vol > 0
    return t
if __name__ == "__main__":
    t = frame()
    print("dev markets", len(t), "events", t.event_ticker.nunique(), "with candle <= T", t.end_ts.notna().sum())
    t["has_ask"] = t.ask.notna() & (t.ask < 1) & (t.ask > 0)
    t["has_bid"] = t.bid.notna() & (t.bid > 0)
    t["two"] = t.has_ask & t.has_bid
    t["spread"] = np.where(t.two, t.ask - t.bid, np.nan)
    print("at T: two-sided", t.two.mean().round(3), "ask only", (t.has_ask & ~t.has_bid).mean().round(3), "bid only", (~t.has_ask & t.has_bid).mean().round(3), "no quote", (~t.has_ask & ~t.has_bid).mean().round(3))
    print("median spread two-sided", np.nanmedian(t.spread), "lifetime vol>0 share", t.traded_life.mean().round(3))
    t["mid"] = np.where(t.two, (t.ask + t.bid) / 2, np.nan)
    b = pd.cut(t.mid, [0, .05, .15, .35, .65, .85, .95, 1])
    g = t[t.two].groupby(b, observed=True).agg(n=("y", "size"), ev=("event_ticker", "nunique"), mid=("mid", "mean"), ask=("ask", "mean"), bid=("bid", "mean"), yrate=("y", "mean"))
    print(g.round(3).to_string())
    # one-sided far rungs: YES ask with no bid (cheap YES) and bid with no ask
    a1 = t[t.has_ask & ~t.has_bid]; print("ask-only rungs n", len(a1), "mean ask", round(a1.ask.mean(), 3), "yes rate", round(a1.y.mean(), 3))
    b1 = t[~t.has_ask & t.has_bid]; print("bid-only rungs n", len(b1), "mean bid", round(b1.bid.mean(), 3), "yes rate", round(b1.y.mean(), 3))
    t.drop(columns=[c for c in t.columns if t[c].map(lambda v: isinstance(v, (list, dict))).any()]).to_parquet(OUT / "tsa_dev_quotes.parquet")
