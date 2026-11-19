#!/bin/bash
# Install (or refresh) the pick'em paper logger as a launchd agent: `pickem.py log` every 30 min.
# Logs every Sleeper Picks leg matched to a Pinnacle player prop (power + multiplicative fair, break-even 1/m)
# to ~/.claude/quant-kalshi/data/pickem/log.jsonl. Measurement only: no entries, no logins (Sleeper ToS).
# NOT installed by default -- run this yourself:  bash sports/scripts/install_pickem_logger_agent.sh
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-pickem-logger
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data/pickem"
rsync -a --delete --exclude __pycache__ "$SRC/sports/" "$DST/sports/"
cp "$SRC/scripts/pickem.py" "$DST/scripts/"

cat > "$DST/pickem_logger.sh" <<EOF
#!/bin/bash
cd "$DST"
out=\$("$DST/.venv/bin/python" scripts/pickem.py log 2>&1 | grep -v Warning)
if [ -n "\$out" ]; then printf '=== %s\n%s\n' "\$(date '+%F %T')" "\$out" >> pickem_logger.log; fi
EOF
chmod +x "$DST/pickem_logger.sh"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/pickem_logger.sh</string></array>
  <key>StartInterval</key><integer>1800</integer>
  <key>StandardErrorPath</key><string>$DST/pickem_logger.err</string>
</dict></plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (every 30 min)"
