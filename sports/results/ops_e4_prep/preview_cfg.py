"""Offline preview: deployed .env + proposed lines -> per-cohort $ caps and contract caps. No network, prints no secrets.
usage: python preview_cfg.py [KEY=VAL ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import kalshi_exec as kx  # noqa: E402

kx.set_root(kx.DEPLOY)
e = kx.env()
for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    e[k] = v
cfg = kx.config(e)
print("cohorts", sorted(cfg["cohorts"]), "| x1 kill since", cfg["x1_kill_since"], "| bands", cfg["price_band_caps"])
for coh in ("orig7", "x1"):
    c = kx.caps(cfg, coh)
    row = []
    for q in (0.90, 0.94, 0.96, 0.97, 0.98):
        n = min(int(c["order"] // q), kx.contract_cap(cfg, coh, q, "live"))
        row.append(f"q{q:.2f}:{n}ct/${n * q:.2f}")
    print(f"{coh:5s} order cap ${c['order']:.2f} day ${c['day']:.2f} ->", " ".join(row))
