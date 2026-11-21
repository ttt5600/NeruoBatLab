"""Federal Register EOs (free API) + data gate vs KXEOWEEK settled rungs."""
import json, requests, pandas as pd, numpy as np
from pathlib import Path
R = Path("results/count_nowcast")
f = R / "fr_eo.json"
if not f.exists():
    url = "https://www.federalregister.gov/api/v1/documents.json"
    p = {"conditions[presidential_document_type][]": "executive_order", "conditions[type][]": "PRESDOCU",
         "conditions[signing_date][gte]": "2025-08-01", "per_page": 200, "order": "oldest",
         "fields[]": ["executive_order_number", "signing_date", "publication_date", "title", "document_number"]}
    out, nxt = [], None
    while True:
        r = requests.get(nxt or url, params=None if nxt else p, timeout=60, headers={"User-Agent": "quant-research-readonly"}).json()
        out += r.get("results", []); nxt = r.get("next_page_url")
        if not nxt: break
    json.dump(out, open(f, "w"))
fr = pd.DataFrame(json.load(open(f)))
fr["sd"] = pd.to_datetime(fr.signing_date); fr["pd"] = pd.to_datetime(fr.publication_date)
fr["lag"] = (fr.pd - fr.sd).dt.days
print("FR EOs", len(fr), fr.sd.min().date(), fr.sd.max().date(), "lag days:", fr.lag.describe()[["50%","max"]].to_dict(),
      "lag>7:", (fr.lag > 7).sum(), "missing EO number:", fr.executive_order_number.isna().sum())
m = pd.DataFrame(json.load(open(R / "mk/KXEOWEEK.json"))).drop_duplicates("ticker")
pr = m.rules_primary.str.extract(r"during (\w+ \d+, \d{4}) to (\w+ \d+, \d{4})")
m["ws"] = pd.to_datetime(pr[0], format="%b %d, %Y", errors="coerce"); m["we"] = pd.to_datetime(pr[1], format="%b %d, %Y", errors="coerce")
m = m[m.we.notna() & m.result.isin(["yes", "no"])].copy()
m["k"] = m.floor_strike.astype(float)
m["n_fr"] = [((fr.sd >= a) & (fr.sd <= b)).sum() for a, b in zip(m.ws, m.we)]
m["pred"] = np.where(m.n_fr > m.k, "yes", "no")
m["ok"] = m.pred == m.result
print("settled rungs", len(m), "events", m.event_ticker.nunique(), "match", m.ok.mean().round(4))
print(m.loc[~m.ok, ["ticker", "ws", "we", "k", "n_fr", "result"]].to_string())
m[["ticker", "event_ticker", "ws", "we", "k", "n_fr", "result", "close_time", "open_time"]].to_parquet(R / "eo_rungs.parquet")
