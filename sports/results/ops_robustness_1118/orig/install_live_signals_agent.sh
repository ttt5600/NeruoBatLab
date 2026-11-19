#!/bin/bash
# Install (or refresh) the live signal engine as a launchd agent: `live_signals.py tick` every 15 min.
# Shares ~/.claude/quant-kalshi (code copy + venv) with the Kalshi temp paper trader, because
# launchd cannot read ~/Desktop (TCC). Re-run after editing live_signals.py or sports/.
# Paper only: no orders are placed. Logs: ~/.claude/quant-kalshi/data/live_signals*.jsonl, live_signals.log
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-live-signals
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data"
rsync -a --delete --exclude __pycache__ "$SRC/sports/" "$DST/sports/"
cp "$SRC/scripts/live_signals.py" "$SRC/scripts/nfl_game_weather.py" "$DST/scripts/"
if [ ! -x "$DST/.venv/bin/python" ]; then
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m venv "$DST/.venv"
  "$DST/.venv/bin/pip" install -q numpy pandas requests pyarrow
fi

cat > "$DST/live_signals.sh" <<EOF
#!/bin/bash
cd "$DST"
out=\$("$DST/.venv/bin/python" scripts/live_signals.py tick 2>&1); rc=\$?
# keep the log quiet: only ticks that did something or failed
if [ -n "\$out" ] || [ \$rc -ne 0 ]; then printf '=== %s exit %s\n%s\n' "\$(date '+%F %T')" "\$rc" "\$out" >> live_signals.log; fi
EOF
chmod +x "$DST/live_signals.sh"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/live_signals.sh</string></array>
  <key>StartInterval</key><integer>900</integer>
  <key>StandardErrorPath</key><string>$DST/live_signals.err</string>
</dict></plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (every 15 min)"
