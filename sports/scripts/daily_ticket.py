"""Daily bet ticket: every model pick for a date in one place, at the user's flat stake. PAPER/ADVICE ONLY, no orders.

    ../.venv_quant/bin/python scripts/daily_ticket.py [--date 2026-11-02] [--stake 2]

Sources (all already-running, pre-registered tests; nothing here is a new rule):
  early_form_F1   data/nba_early_form_log.jsonl rows with fires=True (logged 60-90 min pre-tip)
  open_totals_L32 ~/.claude/quant-kalshi/scripts/nba_open_totals.py report rows for the date
  user calls      data/user_calls.jsonl (the user's own reads, for CLV)
Writes data/tickets/<date>.md (gitignored) and prints it. Policy (STATE.md 10-31): flat $2-3, bet only flagged picks.

GOAL.md rule 6 filter (2026-11-02): a strategy's picks are listed ONLY if edgelab says its state is >= forward_paper
(labelled PAPER for forward_paper). Every other running logger gets one line "not ticketed: <strategy> (<state>)".
State comes from edgelab's read-only status API (Registry.strategy_states, stdlib only) over lab/edgelab_registry.jsonl
(in the launchd copy: the snapshot pushed by scripts/qs_sync.sh push). If edgelab cannot be read, NOTHING is ticketed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QK = Path.home() / ".claude" / "quant-kalshi"
TICKET_OK = {"forward_paper": "PAPER", "live_small": "LIVE_SMALL", "live": "LIVE"}
# every running logger / executor and the edgelab strategy (spec id) it feeds. Ticket sources have a pick function.
RUNNING = [("f1_nba_early_form", "early-form F1 logger"), ("l32b_totals_open", "L32b totals logger"),
           ("kalshi_prop_ladders", "Kalshi prop-ladder logger"), ("q017_b_far_bracket", "Q017 ERCOT logger"),
           ("tsa_t2_maker", "TSA T2 maker paper logger"), ("e4_maker_e1", "quant-kalshi live maker (orig-7)"),
           ("e4_maker_x1", "quant-kalshi live maker (X1)")]


def edgelab_states() -> tuple[dict | None, str]:
    """-> ({spec id: state}, one-line provenance) or (None, why). Read-only over the registry."""
    try:
        sys.path.insert(0, str(ROOT))
        from edgelab.registry import Registry
        reg = Registry(ROOT / "lab" / "edgelab_registry.jsonl")
        if not reg.path.exists():
            return None, f"edgelab registry missing ({reg.path})"
        v = reg.verify()
        if not v["ok"]:
            return None, f"edgelab registry hash chain BROKEN at line {v['first_break']}"
        st = {k: x["state"] for k, x in reg.strategy_states().items()}
        age = datetime.now(timezone.utc).timestamp() - reg.path.stat().st_mtime
        return st, f"edgelab registry {v['lines']} lines, chain OK, snapshot {age / 3600:.1f} h old"
    except Exception as e:  # noqa: BLE001 -- fail closed: no state, no ticket
        return None, f"edgelab status unavailable ({type(e).__name__}: {e})"


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
    states, prov = edgelab_states()
    sources = {"f1_nba_early_form": early_form, "l32b_totals_open": totals}
    rows, notes = [], []
    for sid, what in RUNNING:
        st = states.get(sid, "unregistered") if states is not None else "unknown"
        label = TICKET_OK.get(st)
        if label and sid in sources:
            rows += [r.rstrip()[:-1].rstrip() + f" **{label}** |" for r in sources[sid](a.date)]
        elif label:
            notes.append(f"executor-run, no manual picks: {sid} ({st}, {label}) -- {what}")
        else:
            notes.append(f"not ticketed: {sid} ({st})")
    md = [f"# Ticket {a.date}  (flat ${a.stake:.0f}/bet; bet only these; log fills -> bet.py log-external)", "",
          "| game | pick | price | source |", "|---|---|---|---|"] + (rows or ["| -- | no picks from a strategy at >= forward_paper | | |"])
    c = calls(a.date)
    if c:
        md += ["", "Your calls (tracked for CLV):", "", "| game | pick | price | |", "|---|---|---|---|"] + c
    md += ["", f"Listed only if edgelab state >= forward_paper (GOAL.md rule 6); {prov}."] + notes
    md += ["", "Early-form picks appear 60-90 min before each tip; re-run this before betting."]
    txt = "\n".join(md)
    d = ROOT / "data" / "tickets"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.date}.md").write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
