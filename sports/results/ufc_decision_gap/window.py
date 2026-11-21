"""Step 1 (DEV only): book in the final-horn -> announcement window. Inputs: dev_raw.jsonl (Kalshi tapes/candles),
fights_timed.parquet (UFC live horn / results / unofficial-winner timestamps). Output window.parquet (one row per fight).
Prices only; outcome columns are carried but not used here except in the printed jump check (winner side identity)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
F = pd.read_parquet(OUT / "fights_timed.parquet").set_index("event_ticker")
recs = {}
for cf in ("dev_raw.jsonl", "dev_raw_rev.jsonl", "dev_raw_v2.jsonl"):
    if (OUT / cf).exists():
        for l in open(OUT / cf):
            try: x = json.loads(l)
            except json.JSONDecodeError: print("skip partial line in", cf); continue
            recs.setdefault(x["event_ticker"], x)
recs = list(recs.values())
def tape_df(tr):
    if not tr: return pd.DataFrame(columns=["t", "px", "n", "side"])
    d = pd.DataFrame(tr)
    d["t"] = pd.to_datetime(d.created_time, format="ISO8601", utc=True); d["px"] = d.yes_price_dollars.astype(float)
    d["n"] = d.count_fp.astype(float); d["side"] = d.taker_side
    return d.sort_values("t")[["t", "px", "n", "side"]]
def cand_df(c):
    if not c: return pd.DataFrame(columns=["te", "bid", "ask", "vol"])
    d = pd.DataFrame({"te": pd.to_datetime([x["end_period_ts"] for x in c], unit="s", utc=True),
                      "bid": [float(x["yes_bid"]["close"]) if x["yes_bid"]["close"] is not None else np.nan for x in c],
                      "ask": [float(x["yes_ask"]["close"]) if x["yes_ask"]["close"] is not None else np.nan for x in c],
                      "vol": [float(x["volume"]) for x in c]})
    return d.sort_values("te")
def quote_at(cd, t):
    """Last candle whose period END is <= t: close bid/ask as of that minute boundary (no lookahead)."""
    x = cd[cd.te <= t]
    if not len(x): return np.nan, np.nan, pd.NaT
    r = x.iloc[-1]; return r.bid, r.ask, r.te
def last_trade(td, t):
    x = td[td.t <= t]
    return (x.px.iloc[-1], x.t.iloc[-1]) if len(x) else (np.nan, pd.NaT)
rows = []
for r in recs:
    ev = r["event_ticker"]
    if ev not in F.index: continue
    f = F.loc[ev]
    A, B = tape_df(r["tapeA"]), tape_df(r["tapeB"]); cA, cB = cand_df(r["candA"]), cand_df(r["candB"])
    row = {"event_ticker": ev, "card": f.card, "nA": len(A), "nB": len(B), "has_live": pd.notna(f.t_horn)}
    ms = r.get("milestone") or {}
    row["ms_start"] = pd.to_datetime(ms.get("start_date"), utc=True) if ms else pd.NaT
    if pd.notna(f.t_horn):
        h, res, uw = f.t_horn, f.t_results, f.t_uw
        tdec = (h.floor("min") + pd.Timedelta(minutes=1))          # first minute boundary after the horn: quote age 0-60 s after horn
        row.update(t_horn=h, t_results=res, t_uw=uw, t_dec=tdec, ct=f.ct)
        for k, cd in (("A", cA), ("B", cB)):
            b, a, te = quote_at(cd, tdec); row[f"bid{k}"], row[f"ask{k}"] = b, a; row[f"qte{k}"] = te
            b2, a2, _ = quote_at(cd, h - pd.Timedelta(seconds=1)); row[f"bid{k}_pre"], row[f"ask{k}_pre"] = b2, a2
        # trades: last trade before horn, at horn+30s, just before results (5 s), and window volume
        for k, td in (("A", A), ("B", B)):
            for tag, t in (("pre", h), ("h30", h + pd.Timedelta(seconds=30)), ("late", (res if pd.notna(res) else uw) - pd.Timedelta(seconds=5))):
                px, tt = last_trade(td, t); row[f"px{k}_{tag}"] = px; row[f"age{k}_{tag}"] = (t - tt).total_seconds() if pd.notna(tt) else np.nan
            end = res if pd.notna(res) else uw
            w = td[(td.t > h) & (td.t < end)]
            row[f"wvol{k}"] = w.n.sum(); row[f"wtr{k}"] = len(w)
            row[f"wvwap{k}"] = (w.px * w.n).sum() / w.n.sum() if w.n.sum() > 0 else np.nan
        # Kalshi jump: first trade (either ticker) after horn that prices one side >= .95, with every later trade on
        # that ticker >= .90 (A) / the other ticker <= .10
        jumps = []
        for k, td in (("A", A), ("B", B)):
            x = td[td.t > h - pd.Timedelta(seconds=60)]
            if not len(x): continue
            hi = x.px >= .95
            ok = hi[::-1].cummin()[::-1]    # True where this and all later trades are >= .95
            if ok.any(): jumps.append((x.t[ok].iloc[0], k))
            lo = x.px <= .05
            ok2 = lo[::-1].cummin()[::-1]
            if ok2.any(): jumps.append((x.t[ok2].iloc[0], "B" if k == "A" else "A"))
        if jumps:
            tj, side = min(jumps); row["t_jump"] = tj; row["jump_side"] = side
    rows.append(row)
W = pd.DataFrame(rows)
for k in "AB":
    W[f"mid{k}"] = (W[f"bid{k}"] + W[f"ask{k}"]) / 2; W[f"spr{k}"] = W[f"ask{k}"] - W[f"bid{k}"]
W.to_parquet(OUT / "window.parquet")
w = W[W.has_live]
print("DEV fights with tape", len(W), "with UFC live timing", len(w))
w = w.assign(h2res=(w.t_results - w.t_horn).dt.total_seconds(), h2uw=(w.t_uw - w.t_horn).dt.total_seconds(),
             jump_minus_uw=(w.t_jump - w.t_uw).dt.total_seconds(), jump_minus_res=(w.t_jump - w.t_results).dt.total_seconds(),
             jump_minus_horn=(w.t_jump - w.t_horn).dt.total_seconds(), close_minus_uw=(w.ct - w.t_uw).dt.total_seconds(),
             ms_vs_r1=np.nan)
print(w[["h2res", "h2uw", "jump_minus_horn", "jump_minus_res", "jump_minus_uw", "close_minus_uw"]].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
