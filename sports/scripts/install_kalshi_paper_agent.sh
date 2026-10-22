#!/bin/bash
# Install (or refresh) the durable Kalshi paper trader: launchd cannot read ~/Desktop (TCC),
# so code + venv + log live in ~/.claude/quant-kalshi. Re-run after editing kalshi_paper.py or sports/.
#   tick hourly at :03 = pick every city in its decision window (22:00-23:59 CITY-local, once per
#   city-night) + settle. No wall-clock dispatch: launchd runs late after sleep (10-16 lost every pick).
# kalshi_paper.py places no orders; kalshi_exec.py runs after each tick in DRY mode (log data/kalshi_exec.jsonl).
# Real orders need the USER to put KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH, KALSHI_BANKROLL_USD and EXEC_MODE=live
# in $DST/.env (EXEC_ENV=demo sends them to Kalshi's demo exchange instead). See sports/notes/go_live.md.
# This script never creates or reads $DST/.env; it only tightens its permissions if it exists.
# Log: ~/.claude/quant-kalshi/data/kalshi_paper.jsonl
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-kalshi-paper
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data"
rsync -a --delete --exclude __pycache__ "$SRC/sports/" "$DST/sports/"
cp "$SRC/scripts/kalshi_paper.py" "$SRC/scripts/kalshi_exec.py" "$DST/scripts/"
if [ -f "$DST/.env" ]; then chmod 600 "$DST/.env"; fi
"$DST/.venv/bin/python" -c "import cryptography" 2>/dev/null || "$DST/.venv/bin/pip" install -q cryptography
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
# shadow execution: size each fresh pick against the live book. DRY unless $DST/.env has EXEC_MODE=live
# (user-set) -- and even then only with --live, which this line passes; KILL file in $DST stops it.
if [ "\$1" = tick ]; then "$DST/.venv/bin/python" scripts/kalshi_exec.py run --live >> exec.log 2>&1; fi
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
