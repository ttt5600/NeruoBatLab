#!/bin/bash
# Install (or refresh) the L28 NBA big-lead forward-paper logger as a launchd agent: `nba_bigleads_logger.py tick` every 2 min.
# Shares ~/.claude/quant-kalshi (code copy + venv) with the other paper agents (launchd cannot read ~/Desktop, TCC).
# Paper only: no orders. Cheap exit 03:00-11:00 ET when nothing is open.
# Logs: ~/.claude/quant-kalshi/data/nba_bigleads_{entries,toohigh,settled}.jsonl, nba_bigleads.log
# Report: cd ~/.claude/quant-kalshi && .venv/bin/python scripts/nba_bigleads_logger.py report
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-nba-bigleads
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data"
cp "$SRC/scripts/nba_bigleads_logger.py" "$DST/scripts/"

cat > "$DST/nba_bigleads.sh" <<EOF
#!/bin/bash
cd "$DST"
out=\$("$DST/.venv/bin/python" scripts/nba_bigleads_logger.py tick 2>&1 | grep -v Warning)
if [ -n "\$out" ]; then printf '=== %s\n%s\n' "\$(date '+%F %T')" "\$out" >> nba_bigleads.log; fi
printf '%s tick\n' "\$(date '+%F %T')" > nba_bigleads.heartbeat
EOF
chmod +x "$DST/nba_bigleads.sh"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/nba_bigleads.sh</string></array>
  <key>StartInterval</key><integer>120</integer>
  <key>StandardErrorPath</key><string>$DST/nba_bigleads.err</string>
</dict></plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (every 2 min)"
