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


ODDLOT = Path.home() / ".claude" / "quant-sports" / "data" / "oddlot" / "filings.jsonl"


def oddlot_recent(days: int = 10, n: int = 6) -> str:
    """Odd-lot tender filings with a ticker: filed in the last `days` days OR still open by parsed expiry (newest first)."""
    import datetime as dt
    try:
        cut = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        rows = [json.loads(l) for l in ODDLOT.read_text().splitlines() if l.strip()]
    except (OSError, ValueError) as e:
        return f"(oddlot file unreadable: {type(e).__name__})"
    def open_offer(r: dict) -> bool:  # expiry parsed from the filing ("DECEMBER 10, 2026") is today or later
        try:
            return dt.datetime.strptime(str((r.get("terms") or {}).get("expires_date")).title(), "%B %d, %Y").date() >= dt.date.today()
        except ValueError:
            return False
    rows = [r for r in rows if r.get("tickers") and (str(r.get("file_date", "")) >= cut or open_offer(r))]
    if not rows:
        return f"none with a ticker filed since {cut}"
    out = []
    for r in sorted(rows, key=lambda r: str(r.get("file_date")), reverse=True)[:n]:
        x = r.get("terms") or {}
        px = x.get("price_fixed") or (f"{x.get('price_lo')}-{x.get('price_hi')}" if x.get("price_lo") else "?")
        out.append(f"- {r.get('file_date')} {r.get('form')} {','.join(r['tickers'])} {x.get('kind') or '?'} ${px} "
                   f"exp {x.get('expires_date') or '?'}{' UNDER100-EXCLUDED' if x.get('under100_excluded') else ''}")
    return "\n".join(out) + "\n(read the filing + check price vs floor before acting; notes/money_now.md section B)"


def main() -> None:
    ctx = ("## Re-injected after compact/clear/resume (sports/scripts/ctx_inject.py)\n"
           "Quant-trader lead. Source of truth = files, not memory: sports/STATE.md (top = latest), "
           "notes/ACT_LIST.md (what is live), notes/FUND_BOOK.md (every edge), notes/EDGES_AND_AGENTS.md (log), "
           "`cd sports && ../.venv_quant/bin/python -m edgelab status`.\n\n"
           "### sports/STATE.md (top 40 lines)\n" + head(SPORTS / "STATE.md", 40) +
           "\n\n### Live bots now\n" + bot_status() +
           "\n\n### Odd-lot tender filings: open or filed in the last 10 days (oddlot_watch, daily 07:15 PT)\n" + oddlot_recent())
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}}))


if __name__ == "__main__":
    main()
