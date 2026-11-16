"""Sanity: flat smile -> MS == CB1 M0 (MAX and MIN); live smile shape; MS vs M1 on a grid (no outcomes)."""
import math
import sys
import time

import numpy as np
import requests

sys.path.insert(0, "scripts")
import crypto_barrier_model as M  # noqa: E402
import crypto_barrier_v2_model as V  # noqa: E402

sig = 0.5
flat = {T: (np.array([-1.0, 0, 1.0]), np.array([sig ** 2 * T] * 3)) for T in (0.02, 0.05, 0.2)}
worst = 0
for kind in ("max", "min"):
    for B in (55000, 60000, 62000, 66000, 70000, 80000):
        S, tau = 64000, 0.04
        d = M.dist(kind, S, B)
        if d <= 0:
            continue
        p0 = M.p_m0(d, sig, tau) if kind == "max" else M.p_m0_min(d, sig, tau)
        ps, _ = V.p_ms(kind, S, B, flat, tau)
        worst = max(worst, abs(p0 - ps))
        print(kind, B, round(p0, 5), round(ps, 5))
print("max |M0 - MS| flat:", worst)
now = time.time()
for cur in ("BTC", "ETH"):
    rows = requests.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
                        params={"currency": cur, "kind": "option"}, timeout=30).json()["result"]
    sm = V.smile_from_summary(rows, now)
    print(cur, "expiries d", [round(T * 365, 1) for T in sm], "points", [len(v[0]) for v in sm.values()])
    S = float(requests.get(f"https://api.exchange.coinbase.com/products/{cur}-USD/ticker", timeout=30).json()["price"])
    tau = 14.8 / 365
    for kind, mults in (("max", (1.05, 1.1, 1.2, 1.3)), ("min", (0.95, 0.9, 0.8, 0.7))):
        for m in mults:
            B = S * m
            ps, sb = V.p_ms(kind, S, B, sm, tau)
            atm = math.sqrt(V.w_at(sm, 0.0, tau) / tau)
            p1 = M.p_m1(M.dist(kind, S, B), atm, tau)
            print(f"  {kind} B/S {m}: sig_B {sb:.3f} atm {atm:.3f}  MS {ps:.4f}  M1(atm) {p1:.4f}")
