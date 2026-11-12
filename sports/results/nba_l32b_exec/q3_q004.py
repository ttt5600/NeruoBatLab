"""Q3: why Q004 NBA pick_tight loses while L32b shows CLV. Decompose by side and by de-biasing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

L = json.load(open(Path.home() / ".claude/quant-kalshi/data/open_totals_state.json"))
T = json.load(open(Path.home() / ".claude/quant-kalshi/data/open_totals_tight.json"))
S = json.load(open(Path.home() / ".claude/quant-sports/data/l32b_state.json"))


def level(date_et):  # identical to l32b_logger.level
    xs = sorted((pd.Timestamp(g["start"]), g["decision"]["model"] - g["pin_open"]["line"]) for k, g in L.items()
                if (g.get("decision") or {}).get("model") is not None and g.get("pin_open") and k[:10] < date_et)[-100:]
    return float(np.mean([x for _, x in xs])) if len(xs) >= 30 else None


rows = []
for k, p in T["picks"].items():
    if p.get("model") is None:
        continue
    g = L.get(k, {})
    c = level(k[:10])
    rows.append(dict(key=k, t=p["t"], start=g.get("start"), strike=p["strike"], model=p["model"], gap=p["gap"],
                     pick=p.get("pick"), ask=p.get("ask"), fee=p.get("fee"), result=p.get("result"), roi=p.get("roi"),
                     clv=p.get("clv"), final=p.get("final_total"), c=c,
                     pin_open=(g.get("pin_open") or {}).get("line"), pin_close=(g.get("pin_last") or {}).get("line"),
                     l32b_pick=(S.get(k) or {}).get("pick"), h_before_tip=(pd.Timestamp(g["start"]) - pd.Timestamp(p["t"])).total_seconds() / 3600 if g.get("start") else np.nan))
d = pd.DataFrame(rows)
d.to_csv(Path(__file__).parent / "q3_q004_rows.csv", index=False)
print("Q004 NBA decided with model:", len(d), " picks:", d.pick.notna().sum(), " settled:", d.result.notna().sum())
print("mean raw gap (model - strike) over ALL decided: %+.2f  (sd %.2f)" % (d.gap.mean(), d.gap.std()))
print("h before tip at decision: median %.1f" % d.h_before_tip.median())
s = d[d.result.notna()].copy()


def summ(x, nm):
    if not len(x):
        print(nm, "n 0")
        return
    r = x.roi.values
    rng = np.random.default_rng(0)
    bs = [rng.choice(r, len(r)).mean() for _ in range(4000)]
    cl = x.clv.dropna()
    print(f"{nm}: n {len(x)} W {(x.result == 'win').sum()} ROI {r.mean():+.1%} [{np.percentile(bs, 2.5):+.1%},{np.percentile(bs, 97.5):+.1%}] "
          f"CLV {cl.mean():+.2f} pts  mean ask+fee {(x.ask + x.fee).mean():.3f}")


summ(s, "Q004 all settled")
summ(s[s.pick == "over"], "  overs")
summ(s[s.pick == "under"], "  unders")
s["adj"] = s.model - s.c - s.strike
summ(s[s.adj.abs() >= 3], "  subset that ALSO passes de-biased |model-c-strike|>=3")
summ(s[(s.adj.abs() >= 3) & (np.sign(s.adj) == np.sign(s.gap))], "    same side")
summ(s[s.adj.abs() < 3], "  subset de-biasing would have SKIPPED")
print("c at Q004 picks: mean %.2f" % s.c.mean())
print("strike - pin_open at Q004 decision: mean %+.2f  |.| %.2f" % ((s.strike - s.pin_open).mean(), (s.strike - s.pin_open).abs().mean()))
print("Q004 picks also L32b picks:", s.l32b_pick.notna().sum())
