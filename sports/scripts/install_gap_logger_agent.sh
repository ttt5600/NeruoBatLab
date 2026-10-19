#!/bin/bash
# Install (or refresh) the Kalshi-vs-Pinnacle gap logger as a launchd agent: `gap_logger.py tick` every 2 min.
# Shares ~/.claude/quant-kalshi (code copy + venv) with the other paper agents (launchd cannot read ~/Desktop, TCC).
# Measurement only: no orders. Logs: ~/.claude/quant-kalshi/data/gap_log.jsonl, gap_ticks.jsonl, gap_logger.log
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-gap-logger
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data"
rsync -a --delete --exclude __pycache__ "$SRC/sports/" "$DST/sports/"
cp "$SRC/scripts/gap_logger.py" "$DST/scripts/"
"$DST/.venv/bin/python" -c "import scipy" 2>/dev/null || "$DST/.venv/bin/pip" install -q scipy

cat > "$DST/gap_logger.sh" <<EOF
#!/bin/bash
cd "$DST"
out=\$("$DST/.venv/bin/python" scripts/gap_logger.py tick 2>&1 | grep -v Warning); rc=\${PIPESTATUS[0]}
if [ -n "\$out" ]; then printf '=== %s\n%s\n' "\$(date '+%F %T')" "\$out" >> gap_logger.log; fi
EOF
chmod +x "$DST/gap_logger.sh"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/gap_logger.sh</string></array>
  <key>StartInterval</key><integer>120</integer>
  <key>StandardErrorPath</key><string>$DST/gap_logger.err</string>
</dict></plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (every 2 min)"
