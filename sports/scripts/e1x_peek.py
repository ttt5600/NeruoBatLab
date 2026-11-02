"""Structure peek (no prices vs outcomes): open/close times, strike layout, expiration values per event.
Usage: python scripts/e1x_peek.py KXTSAW
"""
import sys
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / "data" / "e1x"
s = sys.argv[1]
df = pd.read_parquet(D / f"{s}_markets.parquet")
print(df.columns.tolist())
df["open"] = pd.to_datetime(df.open_time, format="ISO8601")
df["close"] = pd.to_datetime(df.close_time, format="ISO8601")
df["ev_val"] = pd.to_numeric(df.expiration_value, errors="coerce")
g = df.groupby("event_ticker").agg(open=("open", "min"), close=("close", "max"), n=("ticker", "size"),
                                   kmin=("floor_strike", "min"), kmax=("floor_strike", "max"),
                                   val=("ev_val", "median"), st=("strike_type", "first"))
g = g.sort_values("close")
g["open_lead_h"] = (g.close - g.open).dt.total_seconds() / 3600
pd.set_option("display.width", 220)
print(g.tail(40).to_string())
print(g.iloc[::25].to_string())
