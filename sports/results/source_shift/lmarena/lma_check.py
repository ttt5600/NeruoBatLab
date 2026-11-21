"""LMArena default switched to style control on 2025-05-16 (HF dataset card). For every settled KXLLM1 / KXTOPMODEL event, which
leaderboard view (text = no style control vs text_style_control = the new default) does Kalshi's YES market match?
Leaderboard on the settlement date = latest leaderboard_publish_date <= event date (as-of; no lookahead). Read-only."""
import json, re, time, urllib.request, urllib.parse
import pandas as pd
API = "https://api.elections.kalshi.com/trade-api/v2"
def g(path, p):
    for a in range(6):
        try: return json.load(urllib.request.urlopen(API + path + "?" + urllib.parse.urlencode(p), timeout=60))
        except Exception: time.sleep(3 * (a + 1))
    return {}
rows = []
for s in ("KXLLM1", "KXTOPMODEL"):
    for path, ex in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
        cur = None
        while True:
            p = dict(series_ticker=s, limit=1000, **ex)
            if cur: p["cursor"] = cur
            d = g(path, p); ms = d.get("markets", []); cur = d.get("cursor")
            for m in ms:
                rows.append(dict(series=s, e=m["event_ticker"], t=m["ticker"], res=m["result"], sub=m.get("yes_sub_title"), close=m["close_time"],
                                 last=m.get("last_price_dollars"), vol=m.get("volume_fp"), rp=m.get("rules_primary"), rs=m.get("rules_secondary") or ""))
            if not cur or not ms: break
km = pd.DataFrame(rows).drop_duplicates("t"); km.to_parquet("kalshi_lma_markets.parquet")
L = {}
for name, f in (("nosc", "/tmp/lma/text_full.parquet"), ("sc", "/tmp/lma_sc/text_style_control_full.parquet")):
    d = pd.read_parquet(f); d = d[d.category == "overall"].copy(); d["date"] = pd.to_datetime(d.leaderboard_publish_date); L[name] = d
def top(view, day):
    d = L[view]; dd = d[d.date <= day]
    if dd.empty: return None, None, None
    pub = dd.date.max(); x = dd[dd.date == pub].sort_values(["rank", "rating", "vote_count"], ascending=[True, False, False])
    r = x.iloc[0]; return r.model_name, r.organization, pub.date().isoformat()
out = []
for e, gdf in km.groupby("e"):
    w = gdf[gdf.res == "yes"]
    m = re.search(r" on (\w{3}) (\d{1,2}), (\d{4})", gdf.rp.iloc[0] or "")
    if not m: continue
    day = pd.Timestamp(f"{m.group(1)} {m.group(2)} {m.group(3)}")
    a, b = top("nosc", day), top("sc", day)
    out.append(dict(series=gdf.series.iloc[0], event=e, day=day.date(), kalshi_yes=";".join(w["sub"].astype(str)) or "(none)",
                    nosc_model=a[0], nosc_org=a[1], nosc_pub=a[2], sc_model=b[0], sc_org=b[1], sc_pub=b[2],
                    toggle_text="Remove Style Control" in gdf.rs.iloc[0]))
o = pd.DataFrame(out).sort_values("day"); o.to_csv("lma_vs_kalshi.csv", index=False)
o["views_differ"] = (o.nosc_org != o.sc_org) | (o.nosc_model != o.sc_model)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
print(o[o.day >= pd.Timestamp("2025-03-01").date()][["series", "event", "day", "kalshi_yes", "nosc_model", "nosc_org", "sc_model", "sc_org", "toggle_text"]].to_string())
