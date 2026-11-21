"""Settlement audit: AAA homepage captures (Wayback id_ raw + today's live page) vs Kalshi expiration_value.
Each capture: 'Price as of MM/DD/YY' + Current / Yesterday / Week Ago / Month Ago regular averages."""
import re, sys, gzip
from pathlib import Path
import pandas as pd, numpy as np
W = Path("wayback")
aaa = pd.read_parquet("aaa_kalshi.parquet").set_index("day").v
def parse(h):
    m = re.search(r"Price as of\s*(\d{1,2}/\d{1,2}/\d{2})", h)
    if not m: return None
    asof = pd.to_datetime(m.group(1), format="%m/%d/%y")
    out = {"asof": asof}
    for lab, off in (("Current", 0), ("Yesterday", 1), ("Week Ago", 7), ("Month Ago", None)):
        mm = re.search(lab + r" Avg\.</td>\s*<td>\$([0-9.]+)</td>", h)
        if mm: out[lab] = (mm.group(1), off)
    return out
rows = []
files = sorted(W.glob("*.html")) + [Path("/tmp/gasdl/aaa/home.html")]
for f in files:
    b = f.read_bytes()
    if b[:2] == b"\x1f\x8b":
        try: b = gzip.decompress(b)
        except Exception: pass
    h = b.decode("utf-8", errors="ignore")
    p = parse(h)
    if not p: rows.append({"file": f.name, "ok": False}); continue
    cap = f.stem
    for lab, (val, off) in [(k, v) for k, v in p.items() if k != "asof"]:
        if off is None:   # month ago: AAA's definition of the date is not documented -> try same day previous month
            d = p["asof"] - pd.DateOffset(months=1)
        else:
            d = p["asof"] - pd.Timedelta(days=off)
        rows.append({"file": cap, "ok": True, "asof": p["asof"], "row": lab, "date": d, "aaa_str": val, "aaa": float(val),
                     "kalshi": aaa.get(d, np.nan)})
a = pd.DataFrame(rows)
bad = a[~a.ok]
a = a[a.ok].copy()
a["cap_ts"] = pd.to_datetime(a.file.where(a.file != "home", "20261121053000"), format="%Y%m%d%H%M%S", errors="coerce", utc=True)
a["has_k"] = a.kalshi.notna()
a["match"] = np.isclose(a.aaa, a.kalshi, atol=1e-9)
k = a[a.has_k]
uniq = k.drop_duplicates(["date", "aaa_str"])
print("captures parsed", a.file.nunique(), "unparsed", len(bad))
print("audit pairs with a Kalshi value", len(k), "distinct dates", k.date.nunique())
print("match by row:", k.groupby("row").match.agg(["sum", "size"]).to_string())
dd = k.groupby("date").agg(any_match=("match", "max"), all_match=("match", "min"), vals=("aaa_str", lambda x: sorted(set(x))))
print("distinct dates audited", len(dd), "all rows match", int(dd.all_match.sum()), "some mismatch", int((~dd.all_match).sum()))
print(k[~k.match][["file", "asof", "row", "date", "aaa_str", "kalshi"]].head(30).to_string())
# capture time vs as-of date (posting time of AAA's daily print)
c = a[a.row == "Current"].drop_duplicates("file").copy()
c["cap_et"] = c.cap_ts.dt.tz_convert("America/New_York")
c["lag_days"] = (c.cap_et.dt.tz_localize(None).dt.normalize() - c["asof"]).dt.days
c["hour_et"] = c.cap_et.dt.hour + c.cap_et.dt.minute / 60
print("capture-date minus as-of date (0 = today's print already up):", c.lag_days.value_counts().to_dict())
print("hours ET of captures with lag 0:", np.sort(c[c.lag_days == 0].hour_et.round(2).to_numpy())[:15], "...")
print("hours ET of captures with lag 1 (print not yet up):", np.sort(c[c.lag_days == 1].hour_et.round(2).to_numpy()))
a.to_parquet("audit_rows.parquet", index=False)
