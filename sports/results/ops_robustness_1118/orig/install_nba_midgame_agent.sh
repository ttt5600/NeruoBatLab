#!/bin/bash
# Install (or refresh) the L37 NBA mid-game comeback logger (forward paper) as a launchd agent: tick every 90 s.
# Shares ~/.claude/quant-kalshi (code copy + venv) with the other paper agents (launchd cannot read ~/Desktop, TCC).
# Paper only: no orders. Cheap exit 03:00-11:00 ET when nothing is open.
# Logs: ~/.claude/quant-kalshi/data/nba_midgame_{ticks,entries,results}.jsonl, nba_midgame_pregame.json, nba_midgame.log
# Live table: cd ~/.claude/quant-kalshi && .venv/bin/python scripts/nba_midgame_scan.py show
# Report:     cd ~/.claude/quant-kalshi && .venv/bin/python scripts/nba_midgame_scan.py report
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-nba-midgame
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data" "$DST/results"
cp "$SRC/scripts/nba_midgame_scan.py" "$DST/scripts/"
cp "$SRC/results/nba_midgame_fit.json" "$DST/results/"
cp "$SRC/data/nba_midgame_states.parquet" "$DST/data/"   # dev seasons 2015-16..2024-25 only
"$DST/.venv/bin/python" -c "import scipy, pyarrow" 2>/dev/null || "$DST/.venv/bin/pip" install -q scipy pyarrow

cat > "$DST/nba_midgame.sh" <<EOF
#!/bin/bash
cd "$DST"
out=\$("$DST/.venv/bin/python" scripts/nba_midgame_scan.py tick 2>&1 | grep -v Warning)
if [ -n "\$out" ]; then printf '=== %s\n%s\n' "\$(date '+%F %T')" "\$out" >> nba_midgame.log; fi
printf '%s tick\n' "\$(date '+%F %T')" > nba_midgame.heartbeat
EOF
chmod +x "$DST/nba_midgame.sh"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/nba_midgame.sh</string></array>
  <key>StartInterval</key><integer>90</integer>
  <key>StandardErrorPath</key><string>$DST/nba_midgame.err</string>
</dict></plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (every 90 s)"
