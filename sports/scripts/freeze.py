"""Freeze a read-only, checksummed snapshot of the collected ESPN data.

    python scripts/freeze.py              # -> data/frozen/<YYYY-MM-DD>/ + MANIFEST.json

Everything mined later reads ONLY from a frozen snapshot, so every number can be
traced to exact bytes. The manifest also records the SEALED season per league,
chosen before any mining: the newest complete season with graded props. Mining
(pattern_mine.py discover) never reads sealed rows; `confirm` scores them once.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import stat
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "espn"
OUT = ROOT / "data" / "frozen"
KINDS = ("graded", "games", "lines")
# newest complete season with graded props, per league (decided 2026-10-14, before mining)
SEALED = {"nba": 2026, "nhl": 2026, "mlb": 2025, "nfl": 2025}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    dst = OUT / date.today().isoformat()
    if dst.exists():
        print(f"{dst} already exists -- snapshots are immutable; delete it by hand to re-freeze")
        return 1
    files = {}
    for lg in SEALED:
        for f in sorted((SRC / lg).glob("*.parquet")):
            if f.stem.split("_")[0] not in KINDS:
                continue
            to = dst / lg / f.name
            to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, to)
            to.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            files[f"{lg}/{f.name}"] = {"sha256": sha256(to), "bytes": to.stat().st_size}
    man = {"frozen": date.today().isoformat(), "source": "ESPN core/site APIs via scripts/collect.py",
           "sealed": SEALED, "files": files}
    (dst / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    (dst / "MANIFEST.json").chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(f"froze {len(files)} files -> {dst}")
    print("sealed seasons:", SEALED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
