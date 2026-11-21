"""Pull all KXAAAGASD markets (historical + live settled + open/closed). Read-only. -> markets.parquet"""
import pandas as pd
from kx import paged
rows = []
rows += paged("/historical/markets", {"series_ticker": "KXAAAGASD", "limit": 1000}, "markets")
n_hist = len(rows)
for st in ("settled", "closed", "open"):
    rows += paged("/markets", {"series_ticker": "KXAAAGASD", "limit": 1000, "status": st}, "markets")
df = pd.DataFrame(rows)
print("hist", n_hist, "total", len(df))
df = df.drop_duplicates("ticker", keep="last")
keep = [c for c in ["ticker","event_ticker","open_time","close_time","expected_expiration_time","expiration_time","latest_expiration_time",
        "settlement_ts","strike_type","floor_strike","cap_strike","result","expiration_value","status","volume_fp","volume","open_interest_fp",
        "rules_primary","last_price_dollars","last_price","yes_sub_title","created_time","can_close_early","settlement_timer_seconds"] if c in df.columns]
df[keep].to_parquet("markets.parquet", index=False)
print(df[keep].shape)
print(df.status.value_counts())
print(df.strike_type.value_counts())
