"""Read-only: full rules + settlement sources for given series."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from sports import wx  # noqa: E402
from sports.http import fetch  # noqa: E402

for s in sys.argv[1:]:
    se = fetch(f"{wx.API}/series/{s}", cache=False).get("series", {})
    m = fetch(f"{wx.API}/markets", {"series_ticker": s, "limit": 1}, cache=False).get("markets", [])
    print("==", s, se.get("settlement_sources"), se.get("contract_url"))
    if m:
        print("  ", m[0].get("event_ticker"), m[0].get("status"))
        print("  P:", m[0].get("rules_primary"))
        print("  S:", m[0].get("rules_secondary"))
