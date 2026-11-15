"""Descriptive DEV diagnostics for OT1b (no outcomes loaded, no P&L). notes/ot1b_monotone.md."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(S / "scripts"), str(S)]
import options_implied_tails as oit  # noqa: E402
from edgelab.rules.opt_tails import taker  # noqa: E402
from edgelab.rules.opt_tails_mono import p_above_grid, p_above_mono  # noqa: E402

D = S / "data" / "opt_tails"
m = pd.read_parquet(D / "markets.parquet").drop(columns=["result"])
imp = pd.read_parquet(D / "implied.parquet")
m = m[m.series.isin(oit.CRYPTO) & (pd.to_datetime(m.date) <= oit.DEV_END)]
rows, ev_stats = [], []


def rev(p):
    p = p[np.isfinite(p)]
    return int((np.diff(p) > 1e-12).sum())


for (s, dd), g in m.groupby(["series", "date"]):
    cur = "BTC" if s == "KXBTCD" else "ETH"
    f = D / "deribit" / f"{cur}_{pd.Timestamp(dd).date()}.json"
    if not f.exists():
        continue
    T = oit.decision_ts(pd.Timestamp(dd))
    close = int(g.close_ts.iloc[0])
    sm = oit.smile(json.loads(f.read_text()), close, T)
    if sm is None:
        continue
    tauK = (close - T) / (365 * 86400)
    grid = p_above_grid(sm, tauK)
    for _, r in g.iterrows():
        rows.append({"ticker": r.ticker, "p_yes_b": oit.p_yes(r, lambda k: p_above_mono(k, grid))})
    gg = g[g.strike_type.isin(["greater", "greater_or_equal"])].sort_values("floor")
    po = np.array([oit.p_above_crypto(k, sm, tauK) for k in gg.floor])
    pn = np.array([p_above_mono(k, grid) for k in gg.floor])
    ev_stats.append({"event": g.event_ticker.iloc[0], "rev_old": rev(po), "rev_new": rev(pn)})

b = pd.DataFrame(rows)
x = m.merge(imp, on="ticker").merge(b, on="ticker")
e = pd.DataFrame(ev_stats)
both = x[np.isfinite(x.p_yes) & np.isfinite(x.p_yes_b)]
print("events with smile", len(e), "| events with >=1 reversal (greater rungs) old", int((e.rev_old > 0).sum()),
      "new", int((e.rev_new > 0).sum()))
print("rungs p finite old", int(np.isfinite(x.p_yes).sum()), "new", int(np.isfinite(x.p_yes_b).sum()), "both", len(both))
print("clipped exactly 0/1 old", int(((x.p_yes == 0) | (x.p_yes == 1)).sum()), "new",
      int(((x.p_yes_b == 0) | (x.p_yes_b == 1)).sum()))
d = (both.p_yes_b - both.p_yes).abs()
print("|p_b - p| median %.5f p90 %.5f p99 %.5f max %.4f" % (d.median(), d.quantile(.9), d.quantile(.99), d.max()))
fr = pd.read_parquet(D / "frame.parquet").drop(columns=["result"])  # quotes only; outcome dropped on read
fr = fr[(fr.asset_class == "crypto") & (pd.to_datetime(fr.date) <= oit.DEV_END)]
q = fr[["ticker", "yes_bid", "yes_ask", "quote_ts", "mark_yes_bid", "mark_yes_ask"]]
x["asset_class"] = "crypto"
old_c = set(oit.tail_candidates(x[np.isfinite(x.p_yes)])["ticker"])
xb = x.drop(columns=["p_yes"]).rename(columns={"p_yes_b": "p_yes"})
new_c = set(oit.tail_candidates(xb[np.isfinite(xb.p_yes)])["ticker"])
print("amend-2 candidates old", len(old_c), "new", len(new_c), "common", len(old_c & new_c))
print("new candidates without a DEV quote (never fetched; their picks unknowable)", len(new_c - set(q.ticker)))
fo = x[x.ticker.isin(old_c)].merge(q, on="ticker")
fn = xb[xb.ticker.isin(new_c)].merge(q, on="ticker")
po, pn = taker(fo, {"asset_class": "crypto"}), taker(fn, {"asset_class": "crypto"})
so, sn = set(po.bet_id), set(pn.bet_id)
print("rule picks old", len(so), "| new (quoted candidates)", len(sn), "| common", len(so & sn),
      "| dropped", len(so - sn), "| added", len(sn - so))
e.to_csv(Path(__file__).parent / "dev_reversals.csv", index=False)
