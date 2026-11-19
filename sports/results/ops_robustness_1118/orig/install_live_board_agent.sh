#!/bin/bash
# Install (or refresh) the UNIFIED LIVE BOARD recorder (NBA / NFL / NHL) as a launchd agent: tick every 30 s.
# Shares ~/.claude/quant-kalshi (code copy + venv) with the other paper agents (launchd cannot read ~/Desktop, TCC).
# Decision support + calibration recorder only: NO orders. Cheap exit when nothing is live (3 scoreboard calls, no Kalshi).
# Logs: ~/.claude/quant-kalshi/data/live_board.jsonl (ticks), live_board_results.jsonl (finals), live_board_pregame.json,
#       live_board_open.json, ~/.claude/quant-kalshi/live_board.log
# Live table: cd ~/.claude/quant-kalshi && .venv/bin/python scripts/live_board.py show
# Report:     cd ~/.claude/quant-kalshi && .venv/bin/python scripts/live_board.py report
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-live-board
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data/live_board" "$DST/results" "$DST/sports"
cp "$SRC/scripts/live_board.py" "$SRC/scripts/live_board_models.py" "$SRC/scripts/live_board_view.py" "$DST/scripts/"
# notify() + read-only orders client come from the LIVE executor copy; never overwrite it from here
[ -f "$DST/scripts/kalshi_exec.py" ] || cp "$SRC/scripts/kalshi_exec.py" "$DST/scripts/"
# the L37 module is imported read-only (model + ESPN parser); refresh only if missing so the frozen logger is untouched
[ -f "$DST/scripts/nba_midgame_scan.py" ] || cp "$SRC/scripts/nba_midgame_scan.py" "$DST/scripts/"
[ -f "$DST/results/nba_midgame_fit.json" ] || cp "$SRC/results/nba_midgame_fit.json" "$DST/results/"
[ -f "$DST/data/nba_midgame_states.parquet" ] || cp "$SRC/data/nba_midgame_states.parquet" "$DST/data/"
cp "$SRC/results/nfl_ingame_fit.json" "$SRC/results/nhl_ingame_fit.json" "$DST/results/"
cp "$SRC/data/live_board/nfl_dev_states.parquet" "$SRC/data/live_board/nhl_dev_hist.parquet" "$DST/data/live_board/"  # dev only
"$DST/.venv/bin/python" -c "import scipy, pyarrow, pandas" 2>/dev/null || "$DST/.venv/bin/pip" install -q scipy pyarrow pandas

cat > "$DST/live_board.sh" <<EOF
#!/bin/bash
cd "$DST"
out=\$("$DST/.venv/bin/python" scripts/live_board.py tick 2>&1 | grep -v Warning)
if [ -n "\$out" ]; then printf '=== %s\n%s\n' "\$(date '+%F %T')" "\$out" >> live_board.log; fi
printf '%s tick\n' "\$(date '+%F %T')" > live_board.heartbeat
EOF
chmod +x "$DST/live_board.sh"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/live_board.sh</string></array>
  <key>StartInterval</key><integer>30</integer>
  <key>StandardErrorPath</key><string>$DST/live_board.err</string>
</dict></plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (every 30 s)"
# one-word lookup: lb [team...] [--fresh]
BIN="$HOME/.local/bin"; [ -w /usr/local/bin ] && BIN=/usr/local/bin
mkdir -p "$BIN"
printf '#!/bin/bash\ncd "%s" && exec .venv/bin/python scripts/live_board.py game "$@"\n' "$DST" > "$BIN/lb"
chmod +x "$BIN/lb"
echo "lb -> $BIN/lb"
