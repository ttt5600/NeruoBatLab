#!/bin/bash
# Install (or refresh) the READ-ONLY ESP1 Kalshi esports vs Pinnacle logger (notes/esports_pinnacle.md 'Ops').
# launchd cannot read ~/Desktop (TCC) -> runs from the deployed copy ~/.claude/quant-sports via scripts/qs_job.sh.
#   com.jonathanwang.quant-sports-esports-pin   launchd 120 s, 24 h: Pinnacle E Sports matchups + markets (2 cached
#                                               GETs) + Kalshi KX{CS2,LOL,VALORANT,DOTA2}GAME open markets (~4-6 GETs via
#                                               kx_throttle) + orderbooks of matched matches starting within 6 h.
# No orders, no keys, no .env. Copies ONLY the files this job needs. Never overwrites deployed outputs (data/esports_pin/).
# Stop: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-esports-pin   (all jobs: touch ~/.claude/quant-sports/STOP)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
[ -x "$DST/.venv/bin/python" ] || { echo "deployed venv missing: run scripts/install_quant_sports_agents.sh first" >&2; exit 1; }
mkdir -p "$DST/scripts" "$DST/data/esports_pin" "$DST/logs" "$DST/state"
for f in qs_job.sh kx_throttle.py esports_pin_logger.py; do cp "$SRC/scripts/$f" "$DST/scripts/$f"; done
chmod +x "$DST/scripts/qs_job.sh"
label="com.jonathanwang.quant-sports-esports-pin"
cat > "$LA/$label.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/scripts/qs_job.sh</string><string>esports-pin</string></array>
  <key>WorkingDirectory</key><string>$DST</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  <key>StartInterval</key><integer>120</integer>
  <key>StandardOutPath</key><string>$DST/logs/launchd-esports-pin.out</string>
  <key>StandardErrorPath</key><string>$DST/logs/launchd-esports-pin.err</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
echo "installed $label (every 120 s)"
echo "verify: cat $DST/state/esports_pin_heartbeat ; tail -c 600 $DST/data/esports_pin/fetch.jsonl ; $DST/.venv/bin/python $DST/scripts/esports_pin_logger.py report"
