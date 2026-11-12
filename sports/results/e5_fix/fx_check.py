"""FX-native checks (pre-registered in notes/e5_fix.md): F3 lower bid on the FX daily lows; F1 feasibility (D-1 prints).
Fill iff the day-D low is strictly below our bid (audit rule, validated 16/16 on live orders); D+1-file prints are also
counted (they are trades on the evening of D before the cutoff: 1 extra fill, a winner)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTD = Path(__file__).parent
FEE = 0.01
s = pd.read_csv(OUTD / "fx_picks_d1.csv")
s["lo"] = s[["low_d", "low_d1"]].min(axis=1)
rng = np.random.default_rng(0)


def boot(cl, pnl, bet, n=4000):
    ud, inv = np.unique(cl, return_inverse=True)
    sp = np.bincount(inv, weights=np.where(bet, pnl, 0), minlength=len(ud))
    sb = np.bincount(inv, weights=bet.astype(float), minlength=len(ud))
    idx = rng.integers(0, len(ud), (n, len(ud)))
    num, den = sp[idx].sum(1), sb[idx].sum(1)
    roi = np.where(den > 0, num / np.maximum(den, 1), 0)
    return [round(float(x), 4) for x in np.percentile(roi, [2.5, 97.5])], round(float((roi <= 0).mean()), 4)


out = {}
for lab, m in [("dev<11-01", s.event_date < "2026-11-01"), ("holdout>=11-01", s.event_date >= "2026-11-01"),
               ("all", s.event_date.notna())]:
    t = s[m]
    R = []
    for k in [0, 2, 5, 10, 15, 20, 30, 40]:
        q = t.side_bid.to_numpy() - k / 100
        filled = (t.lo.to_numpy() < q) & (q > 0)
        won = t.won.to_numpy().astype(bool)
        pnl = np.where(won, 1 / (q + FEE) - 1, -1.0)
        n = int(filled.sum())
        ci, p = boot(t.cl.to_numpy(), pnl, filled) if n else ([None, None], 1.0)
        R.append({"k_cents": k, "fills": n, "fill_rate": round(n / len(t), 3), "loss_filled": int((filled & ~won).sum()),
                  "roi": round(float(pnl[filled].mean()), 4) if n else None, "ci_stationday": ci, "p": p,
                  "units": round(float(pnl[filled].sum()), 2),
                  "breakeven_loss_rate": round(float(1 - (q[filled] + FEE).mean()), 3) if n else None})
    out[lab] = {"n_picks": int(len(t)), "rows": R}
    print(lab, len(t))
    print(pd.DataFrame(R).to_string(index=False))
out["f1_feasibility"] = {"picks_with_D-1_file_print": int(s.last_d1.notna().sum()), "n": int(len(s))}
json.dump(out, open(OUTD / "fx_check.json", "w"), indent=1)
print(out["f1_feasibility"])
