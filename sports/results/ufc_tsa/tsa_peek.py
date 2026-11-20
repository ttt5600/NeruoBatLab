import sys, pandas as pd
sys.path.insert(0, "results/ufc_tsa")
from census import load
t = load("KXTSAD")
t["st"] = pd.to_datetime(t.settlement_ts, format="ISO8601")
e = t[t.status == "finalized"].groupby("event_ticker").agg(ot=("ot", "min"), ct=("ct", "max"), st=("st", "max"), ev=("expiration_value", "first"), vol=("vol", "sum"), n=("ticker", "size")).sort_values("ot")
e["st_minus_open_h"] = ((e.st - e.ot) / pd.Timedelta(hours=1)).round(1)
print(e.to_string())
