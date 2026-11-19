#!/bin/bash
# Install (or refresh) the READ-ONLY QM1/QM2 KXBIGGESTQUAKEMONTH paper logger (notes/quake_monthly.md 'Ops').
# launchd cannot read ~/Desktop (TCC) -> runs from the deployed copy ~/.claude/quant-sports via scripts/qs_job.sh.
#   com.jonathanwang.quant-sports-quake-monthly   launchd 1800 s; USGS ComCat GET every run; Kalshi public GETs only on
#                                                 snapshot slots (daily 15:00 UTC, month-end 23:00 UTC, new M>=6.5)
# No orders, no keys, no .env. Copies ONLY the files this job needs (+ the frozen model and its base catalog). Never
# overwrites deployed outputs (data/kx_forward/quake_monthly*.jsonl).
# Stop: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-quake-monthly   (all jobs: touch ~/.claude/quant-sports/STOP)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
[ -x "$DST/.venv/bin/python" ] || { echo "deployed venv missing: run scripts/install_quant_sports_agents.sh first" >&2; exit 1; }
mkdir -p "$DST/scripts" "$DST/data/kx_forward" "$DST/logs" "$DST/state" "$DST/results/quake_monthly/scripts"
for f in qs_job.sh kx_throttle.py quake_monthly_logger.py; do cp "$SRC/scripts/$f" "$DST/scripts/$f"; done
cp "$SRC/results/quake_monthly/scripts/model.py" "$DST/results/quake_monthly/scripts/model.py"
cp "$SRC/results/quake_monthly/usgs_m6.parquet" "$DST/results/quake_monthly/usgs_m6.parquet"
chmod +x "$DST/scripts/qs_job.sh"
label="com.jonathanwang.quant-sports-quake-monthly"
cat > "$LA/$label.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/scripts/qs_job.sh</string><string>quake-monthly</string></array>
  <key>WorkingDirectory</key><string>$DST</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  <key>StartInterval</key><integer>1800</integer>
  <key>StandardOutPath</key><string>$DST/logs/launchd-quake-monthly.out</string>
  <key>StandardErrorPath</key><string>$DST/logs/launchd-quake-monthly.err</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
echo "installed $label (every 1800 s)"
echo "verify: cat $DST/state/quake_monthly_heartbeat ; tail $DST/logs/quake-monthly.log ; $DST/.venv/bin/python $DST/scripts/quake_monthly_logger.py status"
