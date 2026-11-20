import json, pandas as pd, sys
sys.path.insert(0, "results/ufc_tsa")
from census import load
for s in ["KXUFCMOV", "KXUFCFIGHT", "KXUFCVICROUND"]:
    df = load(s)
    fz = df[df.status == "finalized"].sort_values("ot")
    print("==", s)
    print(fz[["ticker", "open_time", "close_time", "occurrence_datetime", "expected_expiration_time", "settlement_ts", "result"]].tail(8).to_string())
mov = load("KXUFCMOV")
ev = mov[mov.event_ticker == mov[mov.status == "finalized"].event_ticker.iloc[-1]]
print(ev[["ticker", "yes_sub_title", "result", "vol"]].to_string())
print(mov.rules_primary.iloc[0]); print(mov.rules_secondary.iloc[0])
vr = load("KXUFCVICROUND"); print(vr.rules_primary.iloc[0]); print(vr.rules_secondary.iloc[0])
ev = vr[vr.event_ticker == vr[vr.status == "finalized"].event_ticker.iloc[-1]]
print(ev[["ticker", "yes_sub_title", "result", "vol"]].to_string())
