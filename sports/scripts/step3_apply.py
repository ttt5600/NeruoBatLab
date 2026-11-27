"""step3_apply: apply the PRE-REGISTERED E4 orig-7 step 3 (24 -> 48 ct at c >= .97) once the bankroll gate is met.

Pre-registration: sports/notes/SIZING_RULE.md "PRE-REGISTERED 2026-11-25". Criteria 1 (record), 2 (depth 13/14) and
4 (no kill) were MET on 11-25; this script checks criterion 3 live (exchange balance + open exposure >= $370) and only then
rewrites ~/.claude/quant-kalshi/.env (backup first). The user asked for it on 11-25 ("time to up the weather ... be aggressive").

  cd ~/.claude/quant-kalshi && .venv/bin/python <repo>/sports/scripts/step3_apply.py          # check only, prints the diff
  cd ~/.claude/quant-kalshi && .venv/bin/python <repo>/sports/scripts/step3_apply.py --go     # back up + apply
  ... step3_apply.py --wait 960   # poll every 5 min for up to 960 min; exit 0 as soon as the gate is met (no apply)

Revert rule (pre-registered): back to 24 ct if orig-7 loses >= 10% of the step bankroll after the step; the baseline is
written to data/step3.json at apply time.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEPLOY = Path.home() / ".claude" / "quant-kalshi"
sys.path.insert(0, str(DEPLOY / "scripts"))
import kalshi_exec as kx  # noqa: E402

GATE = 370.0
NEW = {"KALSHI_PRICE_BAND_CAPS": "0.97:48,0:19", "KALSHI_MAX_CONTRACTS": "48", "KALSHI_MAX_ORDER_USD": "49.60"}


def bankroll(c) -> tuple[float, float]:
    bal = float(c.request("GET", "/portfolio/balance")[1]["balance_dollars"])
    pos = c._pages("/portfolio/positions", "market_positions", {"count_filter": "position"})
    return bal, sum(float(p.get("market_exposure_dollars") or 0) for p in pos)


def orig7_live_settles() -> list[float]:
    """P&L of live orig7 settles. Settle rows carry no cohort, so the cohort comes from the live order rows (by ticker)."""
    rows = [json.loads(l) for l in (DEPLOY / "data" / "kalshi_exec.jsonl").read_text().splitlines() if l.strip()]
    cohort = {r["ticker"]: r.get("cohort") for r in rows if r.get("event") == "order" and r.get("mode") == "live"}
    return [float(r.get("pnl") or 0) for r in rows
            if r.get("event") == "settle" and r.get("mode") == "live" and cohort.get(r.get("ticker")) == "orig7"]


def new_env(text: str, br: float) -> str:
    want = dict(NEW, KALSHI_BANKROLL_USD=str(math.floor(br)), KALSHI_ORIG7_FRAC=f"{49.60 / math.floor(br):.4f}")
    out, seen = [], set()
    for line in text.splitlines():
        k = line.split("=", 1)[0].strip()
        if k in want and not line.lstrip().startswith("#"):
            out.append(f"{k}={want[k]}")
            seen.add(k)
        else:
            out.append(line)
    missing = set(want) - seen
    if missing:
        raise SystemExit(f"keys not found in .env (refusing to append blindly): {sorted(missing)}")
    return "\n".join(out) + "\n"


def main() -> int:
    e = kx.env()
    c = kx.Kalshi(e, kx.config(e)["env"])
    if "--wait" in sys.argv:
        mins = int(sys.argv[sys.argv.index("--wait") + 1])
        end = time.time() + 60 * mins
        while True:
            try:
                bal, exp = bankroll(c)
                if bal + exp >= GATE:
                    print(f"GATE MET {datetime.now(timezone.utc):%FT%H:%MZ}: balance {bal:.2f} + open {exp:.2f} = {bal + exp:.2f}")
                    return 0
            except Exception as ex:  # noqa: BLE001 - network blips: keep waiting
                print("poll error", type(ex).__name__, ex, flush=True)
            if time.time() > end:
                print("WAIT TIMED OUT: gate not met")
                return 1
            time.sleep(300)
    bal, exp = bankroll(c)
    br = bal + exp
    print(f"balance {bal:.2f} + open exposure {exp:.2f} = bankroll {br:.2f}  (gate {GATE:.0f})")
    if br < GATE:
        print(f"NOT MET: short by {GATE - br:.2f}; nothing changed")
        return 1
    envp = DEPLOY / ".env"
    old = envp.read_text()
    new = new_env(old, br)
    for a, b in zip(old.splitlines(), new.splitlines()):
        if a != b:
            print(f"  {a.split('=')[0]}: {a.split('=', 1)[1]} -> {b.split('=', 1)[1]}")
    if "--go" not in sys.argv:
        print("check only (pass --go to back up and apply)")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bak = DEPLOY / f".env.bak-{stamp}-preStep3"
    shutil.copy2(envp, bak)
    tmp = envp.with_suffix(".tmp")
    tmp.write_text(new)
    tmp.replace(envp)
    settles = orig7_live_settles()
    (DEPLOY / "data" / "step3.json").write_text(json.dumps(dict(
        applied_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), bankroll=round(br, 2),
        revert_if_orig7_loss_usd=round(0.10 * br, 2), orig7_live_pnl_at_step=round(sum(settles), 2),
        orig7_live_settled_at_step=len(settles), backup=str(bak)), indent=1))
    print(f"APPLIED; backup {bak}; revert baseline data/step3.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
