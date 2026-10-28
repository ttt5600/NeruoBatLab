#!/bin/bash
# Install (or refresh) the NHL + NFL opening-line logger (L34 / L33 forward paper) as a launchd agent: every 15 min,
# one tick per league, separate state/picks files per league:
#   ~/.claude/quant-kalshi/data/open_lines_{nhl,nfl}_{state.json,picks.jsonl}, log open_lines.log
# Shares ~/.claude/quant-kalshi (code copy + venv) with the other paper agents (launchd cannot read ~/Desktop, TCC).
# Paper only: no orders. Does not touch the NBA open-totals agent or its files.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-open-lines
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data/pbp" "$DST/data/nhl_simple_model/live" "$DST/results"
rsync -a --exclude __pycache__ "$SRC/sports/" "$DST/sports/"          # shared package (no --delete)
for f in open_lines_logger.py kalshi_tight.py nhl_simple_live.py nfl_simple_model.py; do cp "$SRC/scripts/$f" "$DST/scripts/"; done
cp "$SRC/results/nhl_simple_live_fit.json" "$DST/results/"
# NHL: parsed NHL-API history (2024-25 + current season; the live path appends new games in place)
rsync -a --update "$SRC/data/nhl_simple_model/live/" "$DST/data/nhl_simple_model/live/"
# NFL: nflverse pbp 2006-2026 + schedule + depth charts + closes (the live path refreshes 2026/schedule/depth in place)
rsync -a --update --include 'play_by_play_*.parquet' --exclude '*' "$SRC/data/pbp/" "$DST/data/pbp/"
rsync -a --update "$SRC/data/nfl_schedule.csv" "$SRC/data/nfl_depth_charts_2026.parquet" "$SRC/data/nfl_games.parquet" "$DST/data/"
"$DST/.venv/bin/python" -c "import statsmodels, scipy, pyarrow" 2>/dev/null || "$DST/.venv/bin/pip" install -q statsmodels scipy pyarrow

cat > "$DST/open_lines.sh" <<EOF
#!/bin/bash
cd "$DST"
for lg in nhl nfl; do
  out=\$("$DST/.venv/bin/python" scripts/open_lines_logger.py tick --league \$lg 2>&1 | grep -v Warning)
  if [ -n "\$out" ]; then printf '=== %s %s\n%s\n' "\$(date '+%F %T')" "\$lg" "\$out" >> open_lines.log; fi
done
EOF
chmod +x "$DST/open_lines.sh"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/open_lines.sh</string></array>
  <key>StartInterval</key><integer>900</integer>
  <key>StandardErrorPath</key><string>$DST/open_lines.err</string>
</dict></plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (every 15 min, nhl + nfl)"
