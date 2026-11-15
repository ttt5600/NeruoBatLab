"""Per posted order: trades after post on that ticker, relation to our price."""
import json
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
tape = json.load(open(HERE / "tape.json"))
posts = [json.loads(l) for l in open(Path.home() / ".claude/quant-sports/data/e1x/tsa_paper.jsonl")]
posts = [p for p in posts if p["kind"] == "post"]


def yp(k):
    v = float(k.get("yes_price_dollars", k.get("yes_price")))
    return v / 100 if v > 1.5 else v


def cnt(k):
    return float(k.get("count_fp", k.get("count", 0)))


for p in posts:
    tr = tape[p["event"]]["trades"][p["ticker"]]
    df = pd.DataFrame([{"t": pd.Timestamp(k["created_time"]), "side": k["taker_side"], "yp": yp(k), "n": cnt(k)} for k in tr])
    t0 = pd.Timestamp(p["ts"])
    m = next(x for x in tape[p["event"]]["markets"] if x["ticker"] == p["ticker"])
    if df.empty:
        print(p["ticker"], p["side"], p["q"], "NO TRADES AT ALL", m.get("result")); continue
    a = df[df.t > t0]
    # price in our side's terms: our side price = yp for yes, 1-yp for no
    if p["side"] == "yes":
        hit = a[(a.side == "no") & (a.yp <= p["q"] + 1e-9)]      # sellers of YES at/below our bid
        sells = a[a.side == "no"]; ourpx = sells.yp
    else:
        hit = a[(a.side == "yes") & (a.yp >= 1 - p["q"] - 1e-9)]
        sells = a[a.side == "yes"]; ourpx = 1 - sells.yp
    pre = df[df.t <= t0]
    print(f"{p['ticker']:24s} {p['side']:3s} q={p['q']:.2f} res={m.get('result')!s:4s} trades pre {len(pre)} post {len(a)} "
          f"(ct {a.n.sum():.0f}); sells-to-our-side {len(sells)} ct {sells.n.sum():.0f} px[min,max]="
          f"[{ourpx.min() if len(sells) else float('nan'):.2f},{ourpx.max() if len(sells) else float('nan'):.2f}]; "
          f"at/through q: {len(hit)} ct {hit.n.sum():.0f} first {hit.t.min() if len(hit) else '-'}")
    if len(sells):
        print("    sell px counts (our-side price):", ourpx.round(2).value_counts().sort_index().to_dict())
