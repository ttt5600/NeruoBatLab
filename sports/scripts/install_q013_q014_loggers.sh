#!/bin/bash
# Install (or refresh) the Q013 (RT/Metacritic) and Q014 (Truth Social) FORWARD paper loggers as launchd agents,
# every 900 s. Pre-registrations: notes/q013_rt_scores.md, notes/q014_truth_social.md.
# Shares ~/.claude/quant-kalshi (code copy + venv) with the other paper agents (launchd cannot read ~/Desktop, TCC).
# q014_logger.py imports q013_logger.py, so both are always copied.
# Paper only: no orders, public GET endpoints only, no API key, no .env read.
# Usage: install_q013_q014_loggers.sh [q013|q014|both]   (default both)
# Logs:   ~/.claude/quant-kalshi/data/q01{3,4}/*.jsonl, q013_logger.log/.err, q014_logger.log/.err
# Report: cd ~/.claude/quant-kalshi && .venv/bin/python scripts/q013_logger.py report   (or q014_logger.py report)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
WHICH="${1:-both}"

mkdir -p "$DST/scripts" "$DST/data"
cp "$SRC/scripts/q013_logger.py" "$SRC/scripts/q014_logger.py" "$DST/scripts/"
"$DST/.venv/bin/python" -c "import requests, numpy, scipy" 2>/dev/null || "$DST/.venv/bin/pip" install -q requests numpy scipy

install_one() {
  local Q="$1"
  local LABEL="com.jonathanwang.quant-$Q"
  local PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
  cat > "$DST/${Q}_logger.sh" <<EOF
#!/bin/bash
cd "$DST"
out=\$("$DST/.venv/bin/python" scripts/${Q}_logger.py tick 2>&1 | grep -v Warning)
if [ -n "\$out" ]; then printf '=== %s\n%s\n' "\$(date '+%F %T')" "\$out" >> ${Q}_logger.log; fi
EOF
  chmod +x "$DST/${Q}_logger.sh"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/${Q}_logger.sh</string></array>
  <key>StartInterval</key><integer>900</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardErrorPath</key><string>$DST/${Q}_logger.err</string>
</dict></plist>
EOF
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  echo "installed $LABEL -> $DST (every 900 s)"
}

case "$WHICH" in
  q013) install_one q013 ;;
  q014) install_one q014 ;;
  both) install_one q013; install_one q014 ;;
  *) echo "usage: $0 [q013|q014|both]"; exit 2 ;;
esac
