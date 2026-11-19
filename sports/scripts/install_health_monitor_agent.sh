#!/bin/bash
# Install (or refresh) the READ-ONLY hourly job-health monitor (notes/ops_health_monitor.md). launchd cannot read ~/Desktop
# (TCC) -> runs from the deployed copy ~/.claude/quant-sports.
#   com.jonathanwang.quant-sports-health   launchd every 3600 s (+ at load): scripts/health_monitor.py
# Checks every quant launchd job (config scripts/health_jobs.json): launchd load/exit, log errors, output freshness,
# last-row key fields, 0-pick streaks. Writes digests/health-<date>.md, state/health_status.json, data/health/history.jsonl;
# macOS notification at most once per job per day. Never edits, restarts or kills any job. Stdlib only.
# Files are copied via tmp + mv (atomic rename): a running bash job never sees a half-written file.
# Stop: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-health
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
[ -x "$DST/.venv/bin/python" ] || { echo "deployed venv missing: run scripts/install_quant_sports_agents.sh first" >&2; exit 1; }
mkdir -p "$DST/scripts" "$DST/logs" "$DST/state" "$DST/digests" "$DST/data/health"
for f in health_monitor.py health_jobs.json; do cp "$SRC/scripts/$f" "$DST/scripts/.$f.tmp"; mv -f "$DST/scripts/.$f.tmp" "$DST/scripts/$f"; done
label="com.jonathanwang.quant-sports-health"
cat > "$LA/$label.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>$DST/.venv/bin/python</string><string>$DST/scripts/health_monitor.py</string></array>
  <key>WorkingDirectory</key><string>$DST</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
    <key>TZ</key><string>America/Los_Angeles</string>
  </dict>
  <key>StartInterval</key><integer>3600</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$DST/logs/launchd-health.out</string>
  <key>StandardErrorPath</key><string>$DST/logs/health.log</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
echo "installed $label (hourly)"
echo "verify: cat $DST/digests/health-\$(date +%F).md ; tail $DST/logs/health.log"
