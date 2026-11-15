"""rules_quirks: how often do Kalshi cricket matches settle at $0.50 (tie/NR/abandoned) -- books VOID these.
Read-only GETs. Output results/rules_quirks/cricket_markets.parquet + printed rates."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kx_throttle import kget

D = Path(__file__).resolve().parents[1] / "results" / "rules_quirks"
SER = ["KXODIMATCH", "KXT20MATCH", "KXWT20MATCH", "KXTESTMATCH", "KXIPLGAME", "KXCRICKETT20IMATCH", "KXBBLGAME"]
rows = []
for s in SER:
    for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
        cur = None
        while True:
            p = {"series_ticker": s, "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            d = kget(path, p) or {}
            for m in d.get("markets", []):
                rows.append({"series": s, "ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"),
                             "result": m.get("result"), "sv": m.get("settlement_value_dollars", m.get("settlement_value")),
                             "close_time": m.get("close_time"), "sub": m.get("yes_sub_title"),
                             "vol": m.get("volume_fp", m.get("volume"))})
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    print(s, len(rows), flush=True)
m = pd.DataFrame(rows).drop_duplicates("ticker")
m.to_parquet(D / "cricket_markets.parquet", index=False)
m["sv"] = pd.to_numeric(m.sv, errors="coerce")
m["half"] = (m.result == "scalar") | m.sv.between(0.4, 0.6)
ev = m.groupby(["series", "event_ticker"]).agg(n=("ticker", "size"), half=("half", "all"), anyhalf=("half", "any"),
                                               res=("result", lambda s: ",".join(sorted(set(s)))))
print(ev.groupby("series").agg(events=("n", "size"), all_half=("half", "mean"), any_half=("anyhalf", "mean")))
print(ev.res.value_counts().head(10))
