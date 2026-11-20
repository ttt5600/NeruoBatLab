import json, pandas as pd
from pathlib import Path
OUT = Path(__file__).resolve().parent
d = json.load(open(OUT / "census_raw.json"))
def load(s):
    v = d[s]
    df = pd.DataFrame(v["markets"] + v["hist_markets"]).drop_duplicates("ticker")
    df["vol"] = df.volume_fp.astype(float)
    df["ot"] = pd.to_datetime(df.open_time, format="ISO8601")
    df["ct"] = pd.to_datetime(df.close_time, format="ISO8601")
    return df
if __name__ == "__main__":
    for s, v in d.items():
        df = load(s)
        fz = df[df.status == "finalized"]
        ev = fz.groupby("event_ticker").agg(n=("ticker", "size"), vol=("vol", "sum"), ot=("ot", "min"), res=("result", lambda x: (x == "yes").sum()))
        print("=====", s, "fee", v["series"]["series"].get("fee_type"), v["series"]["series"].get("fee_multiplier"))
        print("settled events", len(ev), "first", ev.ot.min(), "last", ev.ot.max())
        print("rungs/event", ev.n.value_counts().head(8).to_dict(), "yes/event", ev.res.value_counts().to_dict())
        print("event vol quantiles", ev.vol.quantile([.1, .25, .5, .75, .9]).round(0).to_dict())
        print("market vol median", fz.vol.median(), "zero-vol share", round((fz.vol == 0).mean(), 3))
        print({str(k): int(x) for k, x in ev.groupby(ev.ot.dt.to_period("M")).size().items()})
        if "MOV" in s:
            print(fz.yes_sub_title.str.extract(r" by (.*)$")[0].value_counts().head(8).to_dict())
