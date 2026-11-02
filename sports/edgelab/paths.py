from __future__ import annotations

import os
import subprocess
from pathlib import Path

PKG = Path(__file__).resolve().parent
SPORTS = PKG.parent
LAB = SPORTS / "lab"
DATA = SPORTS / "data"
REGISTRY = Path(os.environ["EDGELAB_REGISTRY"]) if os.environ.get("EDGELAB_REGISTRY") else LAB / "edgelab_registry.jsonl"
VAULT_YAML = PKG / "vault.yaml"
GOAL = SPORTS / "GOAL.md"
# Forward logs written by the launchd harness (read-only here; never written by edgelab).
QS_DATA = Path.home() / ".claude" / "quant-sports" / "data"
KALSHI_KILL = Path.home() / ".claude" / "quant-kalshi" / "KILL"


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=SPORTS, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None
