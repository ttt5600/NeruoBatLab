#!/bin/bash
# Install (or refresh) the L39 Kalshi monthly rain/snow logger (forward paper, W1/W1m + S1/S1m) as a launchd agent.
# Fires every hour at :05; each city is decided once per day at its first tick in 22:00-23:59 station-local during a
# month's last 7 days (idle otherwise, apart from settling past-close picks).
# Shares ~/.claude/quant-kalshi (code copy + venv) with the other paper agents (launchd cannot read ~/Desktop, TCC).
# Paper only: no orders. Logs: ~/.claude/quant-kalshi/data/kalshi_wx_monthly_{picks,evals}.jsonl, _state.json,
# kalshi_wx_monthly.log
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-kalshi-wx-monthly
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data"
rsync -a --delete --exclude __pycache__ "$SRC/sports/" "$DST/sports/"
for f in kalshi_wx_monthly_logger.py kalshi_wx_winter.py; do cp "$SRC/scripts/$f" "$DST/scripts/"; done
"$DST/.venv/bin/python" -c "import scipy, pandas, pyarrow" 2>/dev/null || "$DST/.venv/bin/pip" install -q scipy pandas pyarrow

cat > "$DST/kalshi_wx_monthly.sh" <<EOF
#!/bin/bash
cd "$DST"
out=\$("$DST/.venv/bin/python" scripts/kalshi_wx_monthly_logger.py tick 2>&1 | grep -v Warning)
if [ -n "\$out" ]; then printf '=== %s\n%s\n' "\$(date '+%F %T')" "\$out" >> kalshi_wx_monthly.log; fi
EOF
chmod +x "$DST/kalshi_wx_monthly.sh"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/kalshi_wx_monthly.sh</string></array>
  <key>StartCalendarInterval</key><dict><key>Minute</key><integer>5</integer></dict>
  <key>StandardErrorPath</key><string>$DST/kalshi_wx_monthly.err</string>
</dict></plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (hourly at :05; decides 22:00-23:59 station-local, last 7 days of month)"
