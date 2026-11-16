"""Descriptive what-if: if no M>=6.6 quake arrives and the 11-16 book stays as is, when does Q1 (NO taker) fire?"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model

yb = {6.6: .70, 7.0: .40, 7.4: .17, 7.6: .10, 7.8: .06, 8.0: .03, 8.2: .02}   # 11-16 book YES bids
for d in ["2026-11-16", "2026-11-20", "2026-11-24", "2026-11-27", "2026-11-29"]:
    r = model.p_model(pd.Timestamp(f"{d}T15:00Z"), pd.Timestamp("2026-11-01", tz="UTC"), pd.Timestamp("2026-12-01", tz="UTC"))
    out = []
    for K, b in yb.items():
        p = 1 - r["p"][K]; a = 1 - b; e = p - a - 0.07 * a * (1 - a)
        out.append(f"{K}:pNO={p:.3f} ask={a:.2f} e={e:+.3f}{'*' if (p >= .85 and e >= .03) else ''}")
    print(d, f"T={r['T_days']:.1f}", " | ".join(out))
