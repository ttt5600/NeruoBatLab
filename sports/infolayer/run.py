"""Run infolayer families in PAPER mode (no orders exist in this package).

    python -m infolayer.run canvass_margin            # one pass, print flags
    python -m infolayer.run canvass_margin --loop 900  # every 15 min
"""
from __future__ import annotations

import argparse
import importlib
import time
import traceback

from . import core

FAMILIES = ["canvass_margin"]


def once(name: str, min_ev: float) -> None:
    fam = importlib.import_module(f"infolayer.families.{name}")
    obs, flags, seen = fam.scan(min_ev=min_ev)
    stale = sum(not o.fresh for o in obs)
    print(f"{core.now().isoformat(timespec='seconds')} {name}: markets={seen} sources={len(obs)} stale={stale} "
          f"flags={len(flags)}", flush=True)
    for fl in sorted(flags, key=lambda x: -x.ev)[:25]:
        print(f"  FLAG {fl.ticker:32} {fl.side:3} fair {fl.fair:.3f} price {fl.price:.3f} ev {fl.ev:+.3f} {fl.extra}",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=FAMILIES)
    ap.add_argument("--min-ev", type=float, default=0.05)
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes (0 = once)")
    a = ap.parse_args()
    while True:
        try:
            once(a.family, a.min_ev)
        except Exception:
            traceback.print_exc()
        if not a.loop:
            return 0
        time.sleep(a.loop)


if __name__ == "__main__":
    raise SystemExit(main())
