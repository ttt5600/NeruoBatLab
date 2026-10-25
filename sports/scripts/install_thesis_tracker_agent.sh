#!/bin/bash
# Install (or refresh) the thesis tracker as a launchd agent: `thesis_tracker.py tick` weekdays 17:30 local (after the
# US close). One tick logs BOTH v1 (inference vs training, sha b745e0d3aa0e -> data/thesis/tracker.jsonl) and v2
# ("HPC for AI agents" sleeves, sha 7e822f587214 -> data/thesis/tracker_v2.jsonl). Measurement only: no orders.
# Re-run after any thesis_tracker.py change so the launchd copy is not stale. NOT installed by default (user decides).
# Shares ~/.claude/quant-kalshi (code copy + venv) with the other agents (launchd cannot read ~/Desktop, TCC).
# Logs: ~/.claude/quant-kalshi/data/thesis/tracker.jsonl + tracker_v2.jsonl ; report: cd ~/.claude/quant-kalshi && .venv/bin/python scripts/thesis_tracker.py report
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-thesis-tracker
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data/thesis"
cp "$SRC/scripts/thesis_tracker.py" "$DST/scripts/"
"$DST/.venv/bin/python" -c "import yfinance" 2>/dev/null || "$DST/.venv/bin/pip" install -q yfinance

cat > "$DST/thesis_tracker.sh" <<EOF
#!/bin/bash
cd "$DST"
out=\$("$DST/.venv/bin/python" scripts/thesis_tracker.py tick 2>&1 | grep -v Warning)
printf '=== %s\n%s\n' "\$(date '+%F %T')" "\$out" >> thesis_tracker.log
EOF
chmod +x "$DST/thesis_tracker.sh"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/thesis_tracker.sh</string></array>
  <key>StartCalendarInterval</key><array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>30</integer></dict>
    <!-- 08:00 catch-up Tue-Sat: Yahoo's daily bar was not final at 17:30 PT on 2026-10-23 (tick is idempotent) -->
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>6</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>StandardErrorPath</key><string>$DST/thesis_tracker.err</string>
</dict></plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (weekdays 17:30)"
