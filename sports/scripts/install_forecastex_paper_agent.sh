#!/bin/bash
# Install (or refresh) the FX1-fwd ForecastEx weather PAPER logger (scripts/forecastex_paper.py; pre-registration in its
# docstring; notes/forecastex_setup.md) as a launchd agent: tick every 5 min (decides each of the 24 UH* stations once in
# its 22:00-23:59 local window at the LIVE public IBKR ForecastTrader quote, settles hourly from forecastex.com files).
# Shares ~/.claude/quant-kalshi (code copy + venv) with the other paper agents (launchd cannot read ~/Desktop, TCC).
# Paper only: no orders, no login, public GET endpoints only.
# Logs: ~/.claude/quant-kalshi/data/forecastex_paper{.jsonl,_state.json}, forecastex_paper.{log,err}
# Report: cd ~/.claude/quant-kalshi && .venv/bin/python scripts/forecastex_paper.py report
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-forecastex-paper
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts/venues" "$DST/data" "$DST/results"
if ! cmp -s "$SRC/scripts/kalshi_paper.py" "$DST/scripts/kalshi_paper.py"; then
  echo "deployed kalshi_paper.py differs from the worktree's; run install_kalshi_paper_agent.sh first" >&2; exit 1
fi
cp "$SRC/scripts/forecastex_paper.py" "$SRC/scripts/forecastex_wx.py" "$DST/scripts/"
cp "$SRC/scripts/venues/__init__.py" "$SRC/scripts/venues/base.py" "$SRC/scripts/venues/forecastex_client.py" "$DST/scripts/venues/"
cp "$SRC/results/forecastex_wx_sealed.json" "$DST/results/"
"$DST/.venv/bin/python" -c "import requests, numpy, pandas" 2>/dev/null || "$DST/.venv/bin/pip" install -q requests numpy pandas

cat > "$DST/forecastex_paper.sh" <<EOS
#!/bin/bash
cd "$DST"
out=\$("$DST/.venv/bin/python" scripts/forecastex_paper.py tick 2>&1 | grep -v Warning)
if [ -n "\$out" ]; then printf '=== %s\n%s\n' "\$(date '+%F %T')" "\$out" >> forecastex_paper.log; fi
EOS
chmod +x "$DST/forecastex_paper.sh"

cat > "$PLIST" <<EOS
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/forecastex_paper.sh</string></array>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardErrorPath</key><string>$DST/forecastex_paper.err</string>
</dict></plist>
EOS

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (every 300 s)"
