#!/bin/bash
# Install (or refresh) the READ-ONLY Kalshi new-series watcher (notes/ops_series_watch.md). launchd cannot read ~/Desktop
# (TCC) -> runs from the deployed copy ~/.claude/quant-sports via scripts/qs_job.sh.
#   com.jonathanwang.quant-sports-series-watch   fires 07:00 PT and every hour; qs_job.sh runs it once per day from 07:00
# Public GETs only (kx_throttle). No orders, no keys, no .env. Never overwrites deployed snapshots or digests.
# Stop: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-series-watch
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
[ -x "$DST/.venv/bin/python" ] || { echo "deployed venv missing: run scripts/install_quant_sports_agents.sh first" >&2; exit 1; }
mkdir -p "$DST/scripts" "$DST/logs" "$DST/state/series_watch" "$DST/digests" "$DST/notes/series_watch"
for f in qs_job.sh kx_throttle.py series_watch.py; do cp "$SRC/scripts/$f" "$DST/scripts/$f"; done
chmod +x "$DST/scripts/qs_job.sh"
label="com.jonathanwang.quant-sports-series-watch"
cat > "$LA/$label.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/scripts/qs_job.sh</string><string>series-watch</string></array>
  <key>WorkingDirectory</key><string>$DST</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
  <key>StartInterval</key><integer>3600</integer>
  <key>StandardOutPath</key><string>$DST/logs/launchd-series-watch.out</string>
  <key>StandardErrorPath</key><string>$DST/logs/launchd-series-watch.err</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
echo "installed $label (07:00 PT + hourly retry; once/day)"
echo "verify: tail $DST/logs/series-watch.log ; ls $DST/digests/series_watch-*.md"
echo "worktree sync (from a session): ../.venv_quant/bin/python scripts/series_watch.py sync"
