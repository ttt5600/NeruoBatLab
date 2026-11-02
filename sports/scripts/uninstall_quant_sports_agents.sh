#!/bin/bash
# Stop and remove every com.jonathanwang.quant-sports-* launchd agent. Keeps ~/.claude/quant-sports (code, venv,
# logs, digests) so nothing recorded is lost; run scripts/qs_sync.sh pull afterwards to bring the logs home.
# Never touches the quant-kalshi / forecastex agents.
set -u
LA="$HOME/Library/LaunchAgents"
for p in "$LA"/com.jonathanwang.quant-sports-*.plist; do
  [ -e "$p" ] || continue
  label=$(basename "$p" .plist)
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null && echo "stopped $label" || echo "$label was not loaded"
  rm -f "$p"
done
launchctl list | grep quant-sports || echo "no quant-sports agents loaded"
