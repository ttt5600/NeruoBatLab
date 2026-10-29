#!/bin/bash
# Install (or refresh) the E5 ForecastEx EXECUTOR (scripts/forecastex_exec.py) as a launchd agent that runs
# `forecastex_exec.py run --live` every 5 min: orders the FX1-fwd picks the paper agent logs at 22:00 station-local
# (maker BUY at the side's best bid, sized under FX_MAX_ORDER_USD / FX_MAX_CONTRACTS / FX_MAX_DAILY_USD and tier B),
# reconciles fills, cancels orders at cutoff (last trade - 15 min), settles, honours KILL and the -$10/24 h loss stop.
# LIVE MONEY when ~/.claude/quant-kalshi/.env has IBKR_MODE=live + FX_ALLOW_LIVE_ACCOUNT=1. Run only on the user's yes.
# Needs: the paper agent (install_forecastex_paper_agent.sh) and an authenticated Client Portal Gateway session
#   (daily: ~/.claude/quant-kalshi/ibkr/start_gateway.sh, then .venv/bin/python ibkr/login.py --live-session).
# Logs: ~/.claude/quant-kalshi/forecastex_exec.{log,err}; journals data/forecastex_exec.jsonl, data/bets.jsonl.
# Status: cd ~/.claude/quant-kalshi && .venv/bin/python scripts/forecastex_exec.py status
# Stop:   touch ~/.claude/quant-kalshi/KILL   (or launchctl bootout gui/$(id -u)/com.jonathanwang.quant-forecastex-exec)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-forecastex-exec
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if launchctl list | grep -q "com.jonathanwang.quant-forecastex-live"; then
  echo "com.jonathanwang.quant-forecastex-live is loaded: two executors would double-order the same picks." >&2
  echo "bootout it first: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-forecastex-live" >&2; exit 1
fi
for f in bet.py kalshi_exec.py; do
  if ! cmp -s "$SRC/scripts/$f" "$DST/scripts/$f"; then echo "deployed $f differs from the worktree's; sync it first" >&2; exit 1; fi
done
mkdir -p "$DST/scripts/venues" "$DST/data"
cp "$SRC/scripts/forecastex_exec.py" "$DST/scripts/"
cp "$SRC/scripts/venues/__init__.py" "$SRC/scripts/venues/base.py" "$SRC/scripts/venues/forecastex_client.py" "$DST/scripts/venues/"

cat > "$DST/forecastex_exec.sh" <<EOS
#!/bin/bash
cd "$DST"
out=\$("$DST/.venv/bin/python" -W ignore scripts/forecastex_exec.py run --live 2>&1)
if echo "\$out" | grep -qE "PLACED|FILL|CANCEL|SETTLE|PANIC|HALT|REJECTED|REFUSED|NOT TRADING|AMBIGUOUS"; then
  printf '=== %s\n%s\n' "\$(date '+%F %T')" "\$out" >> forecastex_exec.log
fi
EOS
chmod +x "$DST/forecastex_exec.sh"

cat > "$PLIST" <<EOS
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/forecastex_exec.sh</string></array>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardErrorPath</key><string>$DST/forecastex_exec.err</string>
</dict></plist>
EOS

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (every 300 s, LIVE)"
