#!/bin/bash
# Install (or refresh) the READ-ONLY KX*EVENT macro-event ladder book logger (notes/ops_econ_event_logger.md). launchd
# cannot read ~/Desktop (TCC) -> runs from the deployed copy ~/.claude/quant-sports via scripts/qs_job.sh.
#   com.jonathanwang.quant-sports-econ-events   fires every 60 s; snapshots every open KX*EVENT event at release-5,
#   release+5, release+30 min (+ Deribit BTC/ETH option trades at +5) -> data/kx_forward/econ_events/*.jsonl.gz
# Public GETs only (kx_throttle; ~16 per 6-h schedule refresh, ~2-4 per event per slot). No orders, no keys, no .env.
# Stop: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-econ-events
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
. "$SRC/scripts/atomic_deploy.sh"   # acp / awrite = tmp + mv deploys; never cp onto a live script (ops_robustness_1118)
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
[ -x "$DST/.venv/bin/python" ] || { echo "deployed venv missing: run scripts/install_quant_sports_agents.sh first" >&2; exit 1; }
mkdir -p "$DST/scripts" "$DST/logs" "$DST/state" "$DST/data/kx_forward/econ_events"
for f in qs_job.sh kx_throttle.py econ_event_books.py; do acp "$SRC/scripts/$f" "$DST/scripts/$f"; done
chmod +x "$DST/scripts/qs_job.sh"
[ -f "$DST/scripts/ot1_forward.py" ] || echo "WARN: deployed ot1_forward.py missing (install_ot1_agent.sh) -> Deribit rows will carry an import error" >&2
label="com.jonathanwang.quant-sports-econ-events"
cat > "$LA/$label.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/scripts/qs_job.sh</string><string>econ-events</string></array>
  <key>WorkingDirectory</key><string>$DST</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  <key>StartInterval</key><integer>60</integer>
  <key>StandardOutPath</key><string>$DST/logs/launchd-econ-events.out</string>
  <key>StandardErrorPath</key><string>$DST/logs/launchd-econ-events.err</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
echo "installed $label (every 60 s; slots release-5/+5/+30 min)"
echo "verify: tail $DST/logs/econ-events.log ; cat $DST/state/econ_events_heartbeat ; $DST/.venv/bin/python $DST/scripts/econ_event_books.py check ; $DST/.venv/bin/python $DST/scripts/econ_event_books.py schedule"
