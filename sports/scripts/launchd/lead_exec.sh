#!/bin/bash
# launchd wrapper TEMPLATE for lead_exec.py (deployed path ~/.claude/quant-kalshi/lead_exec.sh). DRY: add --live only
# when the user decides; .env must ALSO have EXEC_MODE=live and LEAD_EXEC_MODE=live before anything is sent.
DST="$HOME/.claude/quant-kalshi"
cd "$DST"
out=$("$DST/.venv/bin/python" scripts/lead_exec.py run 2>&1 | grep -v Warning)
if [ -n "$out" ]; then printf '=== %s\n%s\n' "$(date '+%F %T')" "$out" >> lead_exec.log; fi
