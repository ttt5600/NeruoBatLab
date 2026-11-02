#!/bin/bash
# One-command pause/resume for every quant-sports job (agents stay loaded; each run no-ops while STOP exists).
#   scripts/qs_stop.sh          pause all  (touch ~/.claude/quant-sports/STOP)
#   scripts/qs_stop.sh resume   resume     (rm STOP)
# To remove the agents entirely: scripts/uninstall_quant_sports_agents.sh
F="$HOME/.claude/quant-sports/STOP"
if [ "${1:-}" = resume ]; then rm -f "$F"; echo "resumed (STOP removed)"; else date > "$F"; echo "paused: $F"; fi
