"""SessionStart hook (compact|clear|resume): re-inject the fund's working state so a long-lived session never loses it.

Prints hook JSON whose additionalContext = top of sports/STATE.md + ACT_LIST pointer + a live-bot status line.
Read-only; never fails the session (any error -> a short note instead).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

SPORTS = Path(__file__).resolve().parents[1]
DEPLOY = Path.home() / ".claude" / "quant-kalshi"


def head(p: Path, n: int) -> str:
    try:
        return "\n".join(p.read_text().splitlines()[:n])
    except OSError as e:
        return f"(could not read {p}: {e})"


def bot_status() -> str:
    py = DEPLOY / ".venv" / "bin" / "python"
    out = []
    for script, keep in (("kalshi_exec.py", ("[live]", "exchange (prod)")), ("ingame_exec.py", ("ingame_exec:",)),
                         ("lead_exec.py", ("lead_exec:",))):
        try:
            r = subprocess.run([str(py), "-W", "ignore", f"scripts/{script}", "status"], cwd=DEPLOY,
                               capture_output=True, text=True, timeout=25)
            out += [l.strip() for l in r.stdout.splitlines() if any(k in l for k in keep)][:2]
        except Exception as e:  # noqa: BLE001 - a hook must never break the session
            out.append(f"{script} status unavailable ({type(e).__name__})")
    d = DEPLOY / "data"
    kills = [n for n in ("KILL", "LEAD_KILL") if (d / n).is_file()] + sorted(p.name for p in d.glob("LEAD_DISABLE_*"))
    out.append("kill/disable files: " + (", ".join(kills) if kills else "none"))
    return "\n".join(out)


def main() -> None:
    ctx = ("## Re-injected after compact/clear/resume (sports/scripts/ctx_inject.py)\n"
           "Quant-trader lead. Source of truth = files, not memory: sports/STATE.md (top = latest), "
           "notes/ACT_LIST.md (what is live), notes/FUND_BOOK.md (every edge), notes/EDGES_AND_AGENTS.md (log), "
           "`cd sports && ../.venv_quant/bin/python -m edgelab status`.\n\n"
           "### sports/STATE.md (top 40 lines)\n" + head(SPORTS / "STATE.md", 40) +
           "\n\n### Live bots now\n" + bot_status())
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}}))


if __name__ == "__main__":
    main()
