#!/bin/bash
# Install (or refresh) the game-day paper loggers + headless digest as launchd agents (notes/HARNESS.md).
# launchd cannot read ~/Desktop (TCC), so everything runs from the deployed copy ~/.claude/quant-sports.
#   com.jonathanwang.quant-sports-early-form   F1 early-form tick        every 600 s (wrapper: 08:00-22:00 PT)
#   com.jonathanwang.quant-sports-l32b         L32b de-biased totals     every 600 s (08:00-22:00 PT)
#   com.jonathanwang.quant-sports-props        Kalshi prop ladders       every 600 s (08:00-22:00 PT; grade 1st tick/day)
#   com.jonathanwang.quant-sports-ticket       daily ticket writer       every 600 s (08:00-22:00 PT)
#   com.jonathanwang.quant-sports-q017         Q017 ERCOT paper logger   every 600 s (wrapper: >= 25 min apart, 24 h)
#   com.jonathanwang.quant-sports-digest       fund digest (claude -p)   09:00 and 23:30 PT
# All paper / read-only: no orders, no keys. Re-running is safe: code is re-pushed, deployed logs are never touched.
# The live executors (quant-kalshi-paper / forecastex-exec) and ~/.claude/quant-kalshi/.env are NOT touched.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
PYBIN=/opt/homebrew/opt/python@3.14/bin/python3.14

if [ ! -x "$DST/.venv/bin/python" ]; then
  mkdir -p "$DST"
  "$PYBIN" -m venv "$DST/.venv"
  # pinned to the worktree's ../.venv_quant so cached pickles / parquet behave identically
  "$DST/.venv/bin/pip" install -q numpy==2.5.3 pandas==3.0.6 pyarrow==25.0.1 requests==2.34.2 scipy==1.18.1 \
      statsmodels==0.15.0 scikit-learn==1.9.1
fi
"$SRC/scripts/qs_sync.sh" seed
chmod +x "$DST/scripts/qs_job.sh" "$DST/scripts/qs_digest.sh"
mkdir -p "$DST/logs" "$DST/state" "$DST/digests" "$LA"

plist() {  # plist <job> <schedule-xml>
  local label="com.jonathanwang.quant-sports-$1"
  cat > "$LA/$label.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/scripts/qs_job.sh</string><string>$1</string></array>
  <key>WorkingDirectory</key><string>$DST</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  $2
  <key>StandardOutPath</key><string>$DST/logs/launchd-$1.out</string>
  <key>StandardErrorPath</key><string>$DST/logs/launchd-$1.err</string>
</dict></plist>
EOF
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
  echo "installed $label"
}

EVERY='<key>StartInterval</key><integer>600</integer>'
for j in early-form l32b props ticket q017; do plist "$j" "$EVERY"; done
plist digest '<key>StartCalendarInterval</key><array>
    <dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>23</integer><key>Minute</key><integer>30</integer></dict>
  </array>'
rm -f "$DST/STOP"
echo "deployed -> $DST ; verify: launchctl kickstart -k gui/\$(id -u)/com.jonathanwang.quant-sports-<job> && tail $DST/logs/<job>.log"
