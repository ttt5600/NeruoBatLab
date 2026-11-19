#!/bin/bash
# Install (or refresh) the READ-ONLY OT1 forward paper logger (notes/ops_ot1_forward.md). launchd cannot read ~/Desktop
# (TCC) -> runs from the deployed copy ~/.claude/quant-sports via scripts/qs_job.sh.
#   com.jonathanwang.quant-sports-ot1   launchd 120 s; decides once/day at the first run in [11:00, 12:00) ET, settles
#                                       after 17:05 ET; idle runs only write state/ot1_heartbeat
# Copies ONLY the files this job needs. No orders, no keys, no .env. Never overwrites deployed outputs.
# Stop: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-ot1   (all quant-sports jobs: touch ~/.claude/quant-sports/STOP)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
. "$SRC/scripts/atomic_deploy.sh"   # acp / awrite = tmp + mv deploys; never cp onto a live script (ops_robustness_1118)
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
[ -x "$DST/.venv/bin/python" ] || { echo "deployed venv missing: run scripts/install_quant_sports_agents.sh first" >&2; exit 1; }
mkdir -p "$DST/scripts" "$DST/edgelab/rules" "$DST/data/kx_forward" "$DST/logs" "$DST/state"
for f in qs_job.sh kx_throttle.py options_implied_tails.py ot1_forward.py; do acp "$SRC/scripts/$f" "$DST/scripts/$f"; done
acp "$SRC/edgelab/rules/opt_tails.py" "$DST/edgelab/rules/opt_tails.py"     # frozen rule (sha pinned in the registry)
acp "$SRC/edgelab/rules/opt_tails_mono.py" "$DST/edgelab/rules/opt_tails_mono.py"   # OT1b frozen (notes/ot1b_monotone.md)
chmod +x "$DST/scripts/qs_job.sh"
label="com.jonathanwang.quant-sports-ot1"
cat > "$LA/$label.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/scripts/qs_job.sh</string><string>ot1</string></array>
  <key>WorkingDirectory</key><string>$DST</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  <key>StartInterval</key><integer>120</integer>
  <key>StandardOutPath</key><string>$DST/logs/launchd-ot1.out</string>
  <key>StandardErrorPath</key><string>$DST/logs/launchd-ot1.err</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
echo "installed $label (every 120 s)"
echo "verify: tail $DST/logs/ot1.log ; cat $DST/state/ot1_heartbeat ; $DST/.venv/bin/python $DST/scripts/ot1_forward.py report"
