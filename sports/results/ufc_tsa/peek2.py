import json, sys
from pathlib import Path
sys.path.insert(0, "../scripts"); sys.path.insert(0, "scripts")
from kx_throttle import kget
import pandas as pd
T = int(pd.Timestamp("2026-01-24T22:00Z").timestamp())
for path in ["/historical/markets/KXUFCMOV-26JAN24GAEPIM-GAEDEC/candlesticks", "/series/KXUFCMOV/markets/KXUFCMOV-26JAN24GAEPIM-GAEDEC/candlesticks"]:
    d = kget(path, {"start_ts": T - 6 * 3600, "end_ts": T, "period_interval": 60})
    print(path, json.dumps(d)[:1500])
