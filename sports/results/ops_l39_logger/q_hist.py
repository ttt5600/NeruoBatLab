"""Read-only: settled-history rules for the 4 new snow series + ACIS id sanity (name, Nov-2025 snow total)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import kalshi_wx_winter as W  # noqa: E402

for s, sid in [("KXMKESNOWM", "MKE"), ("KXMSPSNOWM", "MSP"), ("KXPITSNOWM", "PIT"), ("KXTTNSNOWM", "TTN")]:
    df = W.settled(s)
    print("==", s, "settled events:", 0 if df.empty else df.event_ticker.nunique())
    if len(df):
        print("   rules:", df.rules_primary.iloc[-1][:200])
    ob = W.acis(sid, "2025-12-01", "2025-12-31")
    print("   ACIS", sid, "rows", len(ob), "Dec-2025 snow", None if ob.empty else round(float(ob.snow.sum()), 1),
          "missing", None if ob.empty else int(ob.snow.isna().sum()))
    try:
        print("   NWS meta", W.station_meta(sid))
    except Exception as ex:  # noqa: BLE001
        print("   NWS meta ERROR", ex)
