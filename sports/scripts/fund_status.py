"""Deterministic fund status bundle for the headless digest (no LLM, no network writes, no orders).

    .venv/bin/python scripts/fund_status.py              # full bundle (markdown) -> stdout
    .venv/bin/python scripts/fund_status.py --digest     # <= 25-line digest built from the bundle alone (no-LLM fallback)
    .venv/bin/python scripts/fund_status.py --headline   # one line for the macOS notification
    .venv/bin/python scripts/fund_status.py --out digests/2026-11-02-am --tag am   # all three files, one pass

Runs from the deployed copy ~/.claude/quant-sports (launchd agent com.jonathanwang.quant-sports-digest) and reads the
other deployed copy ~/.claude/quant-kalshi read-only. Every section is a read-only report command or a file stat:
  kalshi_exec.py status (head), bet.py pnl (head), F1 early-form / L32b / L32 totals / prop-ladder / user_calls reports
  (summary lines), today's ticket, launchd health (log mtimes, last exit codes, `launchctl list`), recent errors,
  pending user asks (top of the notes/STATE.md copy that scripts/qs_sync.sh push drops here).
The output contains dollar amounts: it is written under ~/.claude only, never into the (public) repo.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QK = Path.home() / ".claude" / "quant-kalshi"
PY = str(ROOT / ".venv" / "bin" / "python")
QPY = str(QK / ".venv" / "bin" / "python")
JOBS = {  # job -> (max healthy age of logs/<job>.log in minutes, windowed 08:00-22:00 PT?)
    "early-form": (25, True), "l32b": (25, True), "props": (25, True), "ticket": (25, True), "q017": (45, False)}
QK_LOGS = ["data/kalshi_exec.jsonl", "data/kalshi_paper.jsonl", "data/open_totals_state.json", "data/bets.jsonl"]
ERR_RE = re.compile(r"Traceback|Error|Exception|exit [1-9]")


def sh(cmd: list[str], cwd: Path, timeout: int = 180) -> str:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + ("\n" + r.stderr if r.returncode else "")).strip()
        return out + (f"\n[exit {r.returncode}]" if r.returncode else "")
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout}s: {' '.join(cmd[-2:])}]"
    except Exception as e:  # noqa: BLE001
        return f"[failed: {e}]"


def age_min(p: Path) -> float | None:
    return (time.time() - p.stat().st_mtime) / 60 if p.exists() else None


def fmt_age(a: float | None) -> str:
    return "missing" if a is None else (f"{a:.0f} min" if a < 120 else f"{a / 60:.1f} h")


def in_window(now: datetime) -> bool:
    return 800 <= now.hour * 100 + now.minute < 2200


def last_exit(log: Path) -> str:
    if not log.exists():
        return "-"
    ex = [l for l in log.read_text(errors="replace").splitlines()[-60:] if l.startswith("exit ")]
    return ex[-1] if ex else "-"


def grep_lines(text: str, pats: list[str], n: int = 3) -> list[str]:
    out = [l.rstrip() for l in text.splitlines() if any(p in l for p in pats)]
    return out[:n]


def health(now: datetime) -> tuple[list[str], list[str]]:
    rows, problems = [], []
    for job, (max_age, windowed) in JOBS.items():
        log = ROOT / "logs" / f"{job}.log"
        a, ex = age_min(log), last_exit(log)
        ok = True
        if a is None:
            ok = False
        elif (in_window(now) or not windowed) and a > max_age:
            ok = False
        if ex not in ("-", "exit 0") and not ex.startswith("exit 0 "):
            ok = False
        rows.append(f"| {job} | {fmt_age(a)} | {ex} | {'ok' if ok else 'CHECK'} |")
        if not ok:
            problems.append(f"{job}: log age {fmt_age(a)}, last {ex}")
    dg = ROOT / "logs" / "digest.log"
    rows.append(f"| digest | {fmt_age(age_min(dg))} | - | - |")
    for rel in QK_LOGS:
        rows.append(f"| quant-kalshi {rel} | {fmt_age(age_min(QK / rel))} | - | - |")
    return rows, problems


def errors_24h() -> list[str]:
    out = []
    for log in sorted((ROOT / "logs").glob("*.log")):
        if (age_min(log) or 1e9) > 24 * 60:
            continue
        bad = [l for l in log.read_text(errors="replace").splitlines()[-400:] if ERR_RE.search(l)]
        if bad:
            out.append(f"{log.name}: {len(bad)} error lines in tail, last: {bad[-1][:160]}")
    for err in sorted(QK.glob("*.err")):
        if err.stat().st_size and (age_min(err) or 1e9) < 24 * 60:
            out.append(f"quant-kalshi/{err.name}: {err.read_text(errors='replace').strip().splitlines()[-1][:160]}")
    return out


def launchctl() -> tuple[list[str], list[str]]:
    txt = sh(["/bin/launchctl", "list"], ROOT, 30)
    rows = [l for l in txt.splitlines() if "quant" in l or "ibc" in l]
    bad = []
    for l in rows:
        p = l.split()
        if len(p) >= 3 and p[0] == "-" and p[1] not in ("0", "-"):   # not running AND last exit nonzero
            bad.append(f"{p[2]} last exit {p[1]}")
    return rows, bad


def pending_asks() -> list[str]:
    st = ROOT / "notes" / "STATE.md"
    if not st.exists():
        return ["(no notes/STATE.md copy; run scripts/qs_sync.sh push)"]
    lines = st.read_text(errors="replace").splitlines()[:150]
    asks = [l.strip() for l in lines if re.search(r"USER ASK|needs (a )?yes|awaits yes|WAITING ON USER", l)]
    return asks[:5] or ["(none found in the top of STATE.md)"]


def bundle() -> tuple[str, dict]:
    now = datetime.now()
    d = now.strftime("%Y-%m-%d")
    S: dict = {"date": d, "now": now.strftime("%Y-%m-%d %H:%M %Z")}
    exec_st = sh([QPY, "scripts/kalshi_exec.py", "status"], QK)
    pnl = sh([QPY, "scripts/bet.py", "pnl"], QK)
    ef = sh([PY, "scripts/nba_early_form_logger.py", "report"], ROOT)
    l32b = sh([PY, "scripts/l32b_logger.py", "report"], ROOT)
    props = sh([PY, "scripts/kalshi_prop_logger.py", "report"], ROOT)
    calls = sh([PY, "scripts/user_calls.py", "report"], ROOT)
    tot = sh([QPY, "scripts/nba_open_totals.py", "report"], QK)
    ticket_p = ROOT / "data" / "tickets" / f"{d}.md"
    ticket = ticket_p.read_text().strip() if ticket_p.exists() else "(no ticket file for today yet)"
    hrows, hprob = health(now)
    lrows, lbad = launchctl()
    errs = errors_24h()
    asks = pending_asks()

    S["exec_live"] = (grep_lines(exec_st, ["[live]"], 1) or ["?"])[0].strip()
    S["exec_mode"] = exec_st.splitlines()[0].strip() if exec_st else "?"
    S["ef"] = (grep_lines(ef, ["F1 primary"], 1) or ["F1: no summary line"])[0].strip()
    S["l32b"] = " / ".join(x.strip() for x in grep_lines(l32b, ["decisions", "[live]"], 2)) or "L32b: no summary"
    S["tot"] = next((l.strip() for l in tot.splitlines() if l.startswith("settled ")), "L32: no summary line")
    S["props"] = (props.strip().splitlines() or ["?"])[-1].strip() if "no graded" in props else \
        " / ".join(x.strip() for x in grep_lines(props, ["picks", "ROI", "CLV"], 3)) or props.strip()[:160]
    S["calls"] = " / ".join(x.strip() for x in grep_lines(calls, ["calls ", "CLV mean"], 2)) or "user_calls: none"
    S["ticket_picks"] = [l for l in ticket.splitlines() if l.startswith("| ") and "---" not in l][1:]
    S["problems"] = hprob + lbad + errs
    S["asks"] = asks

    md = [f"# Fund status bundle {S['now']}", "",
          "## Kalshi executor (kalshi_exec.py status, head)", "```", *exec_st.splitlines()[:20], "```",
          "## Bet journal (bet.py pnl, head)", "```", *pnl.splitlines()[:22], "```",
          "## Forward tests (one-liners)",
          f"- F1 early-form: {S['ef']}", f"- L32b de-biased totals: {S['l32b']}", f"- L32 open totals: {S['tot']}",
          f"- Kalshi prop ladders: {S['props']}", f"- User calls: {S['calls']}",
          f"## Today's ticket ({ticket_p.name})", ticket[:2500],
          "## launchd health (quant-sports logs; windowed jobs idle 22:00-08:00 PT)",
          "| job | log age | last exit | status |", "|---|---|---|---|", *hrows,
          "## launchctl list | grep quant", "```", *lrows, "```",
          "## Errors (last 24 h)", *(f"- {e}" for e in errs or ["none"]),
          "## Problems flagged", *(f"- {p}" for p in S["problems"] or ["none"]),
          "## Pending user asks (top of STATE.md copy; may be stale -- copied at last qs_sync push)",
          *(f"- {a}" for a in asks)]
    return "\n".join(md), S


def headline(S: dict) -> str:
    n = len(S["ticket_picks"])
    prob = f"{len(S['problems'])} problem(s)" if S["problems"] else "all agents ok"
    live = re.search(r"P&L \$[+-][0-9.]+", S["exec_live"])
    return f"{S['date']}: {n} ticket pick(s); Kalshi live {live.group(0) if live else '?'}; {prob}"


def digest(S: dict, tag: str) -> str:
    out = [f"# Fund digest {S['now']} ({tag}, no-LLM fallback)", "", f"**{headline(S)}**", "",
           f"- Kalshi: {S['exec_mode'][:110]}", f"- {S['exec_live'][:150]}",
           f"- F1: {S['ef'][:150]}", f"- L32b: {S['l32b'][:150]}", f"- L32: {S['tot'][:150]}",
           f"- Props: {S['props'][:150]}", f"- Calls: {S['calls'][:150]}", "", "Today's ticket:"]
    out += S["ticket_picks"][:5] or ["- no picks yet"]
    out += ["", "Broken / check:"] + ([f"- {p[:150]}" for p in S["problems"][:4]] or ["- none"])
    out += ["", "Waiting on user:"] + [f"- {a[:150]}" for a in S["asks"][:3]]
    return "\n".join(out[:25])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--digest", action="store_true")
    ap.add_argument("--headline", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", help="PREFIX: write PREFIX.status.md, PREFIX.fallback.md, PREFIX.headline.txt in one pass")
    a = ap.parse_args()
    md, S = bundle()
    if a.out:
        Path(a.out + ".status.md").write_text(md + "\n")
        Path(a.out + ".fallback.md").write_text(digest(S, a.tag or "auto") + "\n")
        Path(a.out + ".headline.txt").write_text(headline(S) + "\n")
    elif a.digest:
        print(digest(S, a.tag or "manual"))
    elif a.headline:
        print(headline(S))
    else:
        print(md)
        print(f"\nHEADLINE: {headline(S)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
