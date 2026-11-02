"""Daily bet ticket: every model pick for a date in one place, at the user's flat stake. PAPER/ADVICE ONLY, no orders.

    ../.venv_quant/bin/python scripts/daily_ticket.py [--date 2026-11-02] [--stake 2]

Sources (all already-running, pre-registered tests; nothing here is a new rule):
  early_form_F1   data/nba_early_form_log.jsonl rows with fires=True (logged 60-90 min pre-tip)
  open_totals_L32 ~/.claude/quant-kalshi/scripts/nba_open_totals.py report rows for the date
  user calls      data/user_calls.jsonl (the user's own reads, for CLV)
Writes data/tickets/<date>.md (gitignored) and prints it. Policy (STATE.md 10-31): flat $2-3, bet only flagged picks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QK = Path.home() / ".claude" / "quant-kalshi"


def early_form(date: str) -> list[str]:
    f = ROOT / "data" / "nba_early_form_log.jsonl"
    out = []
    for r in (json.loads(x) for x in f.read_text().splitlines() if x.strip()) if f.exists() else []:
        if r.get("date") == date and r.get("fires"):
            team = r["home"] if r["side"] == "home" else r["away"]
            k = r.get("kalshi") or {}
            ask = k.get(f"{r['side']}_ask") or k.get("ask")
            out.append(f"| {r['away']}@{r['home']} | **{team} ML** | {ask if ask is not None else '?'} | early-form F1 "
                       f"(R {r['R']:+.1f}, tip {r['start_utc'][11:16]}Z) |")
    return out


def totals(date: str) -> list[str]:
    """L32b de-biased totals (live decisions only). The raw L32 rule is NOT ticketed: its picks are ~all overs from the
    model's level offset (notes/l32b_debiased_totals.md); it stays a paper logger only."""
    f = ROOT / "data" / "l32b_log.jsonl"
    out = []
    for r in (json.loads(x) for x in f.read_text().splitlines() if x.strip()) if f.exists() else []:
        if r.get("type") == "decision" and r.get("kind") == "live" and r.get("pick") and r["key"].startswith(date + "|"):
            adj = r["model"] - r["c"]
            out.append(f"| {r['key'].split('|', 1)[1]} | **{r['pick'].upper()} {r['open_line']}** | -110 or better; "
                       f"skip if line moved within 3 of {adj:.1f} | de-biased totals L32b (gap {r['adj_gap']:+.1f}) |")
    return out


def totals_l32_raw(date: str) -> list[str]:   # kept for reference; not ticketed
    try:
        rep = subprocess.run([str(QK / ".venv/bin/python"), str(QK / "scripts/nba_open_totals.py"), "report"],
                             capture_output=True, text=True, timeout=300).stdout
    except Exception as e:
        return [f"| (totals report failed: {e}) | | | |"]
    out = []
    for ln in rep.splitlines():
        if ln.strip().startswith(date + "|"):
            p = ln.split()
            out.append(f"| {p[0].split('|', 1)[1]} | **{p[1].upper()} {p[2]}** | -110 or better | open-totals L32 "
                       f"(model {p[3]}, gap {p[4]}) |")
    return out


def calls(date: str) -> list[str]:
    f = ROOT / "data" / "user_calls.jsonl"
    rs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    return [f"| {r['game']} | {r['pick']} ML | {r['price']:.2f} | your call ({r.get('result') or 'open'}) |"
            for r in rs if r["date"] == date]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--stake", type=float, default=2.0)
    a = ap.parse_args()
    rows = early_form(a.date) + totals(a.date)
    md = [f"# Ticket {a.date}  (flat ${a.stake:.0f}/bet; bet only these; log fills -> bet.py log-external)", "",
          "| game | pick | price | source |", "|---|---|---|---|"] + (rows or ["| -- | no model picks yet | | |"])
    c = calls(a.date)
    if c:
        md += ["", "Your calls (tracked for CLV):", "", "| game | pick | price | |", "|---|---|---|---|"] + c
    md += ["", "Early-form picks appear 60-90 min before each tip; re-run this before betting."]
    txt = "\n".join(md)
    d = ROOT / "data" / "tickets"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.date}.md").write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
