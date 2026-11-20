"""UFC DEV descriptive calibration: Kalshi MOV / VICROUND mid and ask at T vs outcome, by rung type and price bin.
Card-clustered bootstrap CI on (outcome - mid). No model, no rule."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from quotes import ufc_quotes
def boot(df, col, B=2000, seed=1):
    rng = np.random.default_rng(seed); cards = df.card.unique()
    g = {c: x for c, x in df.groupby("card")}
    v = []
    for _ in range(B):
        s = pd.concat([g[c] for c in rng.choice(cards, len(cards))]); v.append(s[col].mean())
    return np.percentile(v, [2.5, 97.5])
if __name__ == "__main__":
    m = pd.read_parquet(OUT / "ufc_markets.parquet")
    q = ufc_quotes()
    d = m.merge(q, on="ticker", how="inner", validate="one_to_one")
    d = d[d.split == "dev"]
    d["y"] = (d.result == "yes").astype(int)
    sfx = d.ticker.str.split("-").str[2]
    d["rtype"] = np.select([d.series == "KXUFCFIGHT", sfx.str.startswith("DRAW"), sfx.str.endswith("KOTKODQ"), sfx.str.endswith("SUB"), sfx.str.endswith("DEC"), sfx == "OTHER"],
                           ["WIN", "DRAW", "KO", "SUB", "DEC", "VR_OTHER"], "VR_ROUND")
    d.loc[d.rtype == "VR_ROUND", "rtype"] = "VR_R" + sfx[d.rtype == "VR_ROUND"].str[-1]
    # drop events with no YES (cancel/NC) -- they are refunded/void-ish; report count
    anyy = d.groupby("event_ticker").y.transform("sum")
    print("dev markets with quote file", len(d), "events with no YES dropped", d[anyy == 0].event_ticker.nunique())
    d = d[anyy > 0]
    print("valid two-sided at T by series", d.groupby("series").valid.mean().round(3).to_dict(), "quote age min median", d.age_min.median())
    v = d[d.valid].copy(); v["spread"] = v.ask - v.bid; v["res"] = v.y - v.mid; v["res_ask"] = v.y - v.ask
    print("median spread by series", v.groupby("series").spread.median().to_dict())
    rows = []
    for rt, x in v.groupby("rtype"):
        lo, hi = boot(x, "res")
        rows.append(dict(rtype=rt, n=len(x), cards=x.card.nunique(), mid=x.mid.mean(), ask=x.ask.mean(), yrate=x.y.mean(), y_minus_mid=x.res.mean(), lo=lo, hi=hi, spread=x.spread.median()))
    print(pd.DataFrame(rows).round(4).to_string())
    # MOV by price bin
    mv = v[v.series == "KXUFCMOV"]
    b = pd.cut(mv.mid, [0, .05, .10, .20, .35, .55, 1])
    rows = []
    for k, x in mv.groupby(b, observed=True):
        lo, hi = boot(x, "res"); rows.append(dict(bin=str(k), n=len(x), mid=x.mid.mean(), yrate=x.y.mean(), y_minus_mid=x.res.mean(), lo=lo, hi=hi, y_minus_ask=x.res_ask.mean()))
    print(pd.DataFrame(rows).round(4).to_string())
    # ladder coherence: sum of MOV mids per fight; MOV-implied win vs FIGHT mid
    mv2 = mv.copy(); mv2["sidecode"] = mv2.ticker.str.split("-").str[2].str.replace(r"(KOTKODQ|SUB|DEC)$", "", regex=True)
    s = mv2.groupby("event_ticker").mid.sum(); print("MOV sum of mids per fight: median", s.median().round(3), "IQR", s.quantile([.25, .75]).round(3).tolist(), "n", len(s))
    v.to_parquet(OUT / "ufc_dev_quotes.parquet")
