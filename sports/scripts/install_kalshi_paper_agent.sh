#!/bin/bash
# Install (or refresh) the durable Kalshi paper trader: launchd cannot read ~/Desktop (TCC),
# so code + venv + log live in ~/.claude/quant-kalshi. Re-run after editing kalshi_paper.py or sports/.
#   tick hourly at :03 = pick every city in its decision window (22:00-23:59 CITY-local, once per
#   city-night) + settle. No wall-clock dispatch: launchd runs late after sleep (10-16 lost every pick).
# Paper only: kalshi_paper.py places no orders. Log: ~/.claude/quant-kalshi/data/kalshi_paper.jsonl
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-kalshi-paper
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data"
rsync -a --delete --exclude __pycache__ "$SRC/sports/" "$DST/sports/"
cp "$SRC/scripts/kalshi_paper.py" "$DST/scripts/"
if [ ! -x "$DST/.venv/bin/python" ]; then
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m venv "$DST/.venv"
  "$DST/.venv/bin/pip" install -q numpy pandas requests pyarrow
fi

cat > "$DST/run.sh" <<EOF
#!/bin/bash
# \$1 = tick | pick | settle. Appends a timestamped run record to run.log.
cd "$DST"
echo "=== \$(date '+%F %T') \$1" >> run.log
"$DST/.venv/bin/python" scripts/kalshi_paper.py "\$1" >> run.log 2>&1
echo "exit \$?" >> run.log
EOF
chmod +x "$DST/run.sh"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/dispatch.sh</string></array>
  <key>StartCalendarInterval</key><array>
    <dict><key>Minute</key><integer>3</integer></dict>
  </array>
  <key>StandardErrorPath</key><string>$DST/launchd.err</string>
</dict></plist>
EOF

cat > "$DST/dispatch.sh" <<EOF
#!/bin/bash
exec "$DST/run.sh" tick
EOF
chmod +x "$DST/dispatch.sh"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST"
