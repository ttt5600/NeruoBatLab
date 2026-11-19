#!/bin/bash
# Install (or refresh) the F4 Polymarket US weather PAPER logger (scripts/pmus_weather_paper.py; pre-registration in its
# docstring; notes/venue_expansion.md) as a launchd agent: tick every 5 min (decides each of the 6 matched cities once in
# its 22:00-23:59 local window, polls paper maker fills from the public book stats, settles hourly).
# Shares ~/.claude/quant-kalshi (code copy + venv) with the other paper agents (launchd cannot read ~/Desktop, TCC).
# Uses the deployed kalshi_paper.py (frozen rule + forecasts) and sports/wx.py; copies polymarket_us.py (key-free reader).
# Paper only: no orders, public GET endpoints only (Polymarket US gateway, Kalshi, Open-Meteo), no API key.
# Logs: ~/.claude/quant-kalshi/data/pmus_weather_paper{.jsonl,_state.json,_done.json}, pmus_weather_paper.{log,err}
# Report: cd ~/.claude/quant-kalshi && .venv/bin/python scripts/pmus_weather_paper.py report
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-pmus-weather-paper
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data"
# the rule must be the SAME code the E1/E2 paper agent runs: refuse to install over a diverged copy
if ! cmp -s "$SRC/scripts/kalshi_paper.py" "$DST/scripts/kalshi_paper.py"; then
  echo "deployed kalshi_paper.py differs from the worktree's; run install_kalshi_paper_agent.sh first" >&2; exit 1
fi
cp "$SRC/scripts/pmus_weather_paper.py" "$SRC/scripts/polymarket_us.py" "$DST/scripts/"
"$DST/.venv/bin/python" -c "import requests, numpy" 2>/dev/null || "$DST/.venv/bin/pip" install -q requests numpy

cat > "$DST/pmus_weather_paper.sh" <<EOS
#!/bin/bash
cd "$DST"
out=\$("$DST/.venv/bin/python" scripts/pmus_weather_paper.py tick 2>&1 | grep -v Warning)
if [ -n "\$out" ]; then printf '=== %s\n%s\n' "\$(date '+%F %T')" "\$out" >> pmus_weather_paper.log; fi
EOS
chmod +x "$DST/pmus_weather_paper.sh"

cat > "$PLIST" <<EOS
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/pmus_weather_paper.sh</string></array>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardErrorPath</key><string>$DST/pmus_weather_paper.err</string>
</dict></plist>
EOS

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (every 300 s)"
