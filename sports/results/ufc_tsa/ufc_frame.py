"""UFC market frame: card date, decision time T, split. No prices/outcomes joined here."""
import json, sys
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from census import load, d as RAW
SPLIT = pd.Timestamp("2026-08-01")
def frame():
    parts = []
    for s in ["KXUFCMOV", "KXUFCVICROUND", "KXUFCFIGHT"]:
        df = load(s)
        df["series"] = s
        df["src_hist"] = df.ticker.isin({m["ticker"] for m in RAW[s]["hist_markets"]})
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df[df.status == "finalized"].copy()
    df["fight"] = df.event_ticker.str.split("-").str[1]          # e.g. 26NOV14GANHOK
    df["card"] = pd.to_datetime(df.fight.str[:7], format="%y%b%d")
    f = df[df.series == "KXUFCFIGHT"]
    # earliest fight-end on the card among fights that settled with a winner, within [card-1d, card+2d]
    ok = f[(f.result == "yes") & (f.ct >= f.card.dt.tz_localize("UTC") - pd.Timedelta(days=1)) & (f.ct <= f.card.dt.tz_localize("UTC") + pd.Timedelta(days=2))]
    first = ok.groupby("card").ct.min().rename("first_end")
    df = df.merge(first, left_on="card", right_index=True, how="left")
    df["Tdec"] = df.first_end - pd.Timedelta(minutes=60)
    df["split"] = (df.card >= SPLIT).map({True: "sealed", False: "dev"})
    return df
if __name__ == "__main__":
    df = frame()
    print(df.groupby(["series", "split"]).event_ticker.nunique())
    print("missing T", df.Tdec.isna().groupby(df.series).sum().to_dict())
    c = df.drop_duplicates("card")[["card", "first_end", "Tdec"]].sort_values("card")
    c["T_hour_utc"] = c["Tdec"].dt.hour
    print(c.T_hour_utc.value_counts().sort_index().to_dict())
    print(c.head(12).to_string())
    # any market whose own close is before T? (fight ended before T = impossible unless cancellation)
    bad = df[df.ct < df["Tdec"]]
    print("markets closing before T", len(bad), bad.groupby("series").size().to_dict())
    # MOV markets opened after T?
    late = df[df.ot > df["Tdec"]]
    print("opened after T", late.groupby("series").size().to_dict())
    df.drop(columns=[c for c in df.columns if df[c].map(lambda v: isinstance(v, (list, dict))).any()]).to_parquet(OUT / "ufc_markets.parquet")
