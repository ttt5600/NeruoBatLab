#!/bin/bash
# Install (or refresh) the NBA opening-totals logger (L32 forward paper) as a launchd agent: tick every 10 min.
# Shares ~/.claude/quant-kalshi (code copy + venv) with the other paper agents (launchd cannot read ~/Desktop, TCC).
# Paper only: no orders. Logs: ~/.claude/quant-kalshi/data/open_totals_{state.json,picks.jsonl}, nba_open_totals.log
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-nba-open-totals
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data/espn" "$DST/data/nba_comebacks" "$DST/data/nba_simple_model" "$DST/results"
rsync -a --delete --exclude __pycache__ "$SRC/sports/" "$DST/sports/"
for f in nba_open_totals.py kalshi_tight.py nba_simple_live.py nba_simple_model.py nba_player_sim.py collect.py; do cp "$SRC/scripts/$f" "$DST/scripts/"; done
cp "$SRC/results/nba_simple_model_frozen.json" "$DST/results/"
# model inputs: ESPN NBA seasons (the live path refreshes the current season in place) + paint team logs
rsync -a --update "$SRC/data/espn/nba" "$DST/data/espn/"
rsync -a "$SRC/data/nba_comebacks/teamlogs.parquet" "$DST/data/nba_comebacks/"
"$DST/.venv/bin/python" -c "import statsmodels, scipy" 2>/dev/null || "$DST/.venv/bin/pip" install -q statsmodels scipy

cat > "$DST/nba_open_totals.sh" <<EOF
#!/bin/bash
cd "$DST"
out=\$("$DST/.venv/bin/python" scripts/nba_open_totals.py tick 2>&1 | grep -v Warning)
if [ -n "\$out" ]; then printf '=== %s\n%s\n' "\$(date '+%F %T')" "\$out" >> nba_open_totals.log; fi
EOF
chmod +x "$DST/nba_open_totals.sh"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/nba_open_totals.sh</string></array>
  <key>StartInterval</key><integer>600</integer>
  <key>StandardErrorPath</key><string>$DST/nba_open_totals.err</string>
</dict></plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (every 10 min)"
