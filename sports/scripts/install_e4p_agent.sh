#!/bin/bash
# Install (or refresh) the READ-ONLY E4 probabilistic rule C / C-agree maker paper logger (notes/ops_e4p_forward.md). launchd cannot read
# ~/Desktop (TCC) -> runs from the deployed copy ~/.claude/quant-sports via scripts/qs_job.sh.
#   com.jonathanwang.quant-sports-e4p   launchd 300 s; decides per city (24 KXHIGH series) in [22:00, 22:50) station-local
#                                       D-1, settles at most hourly; idle runs only write state/e4p_heartbeat
# Reads ~/.claude/quant-kalshi/data/kalshi_paper.jsonl read-only (live E4 picks, for the NEW-only exclusion). No orders,
# no keys, no .env. Public GETs: Kalshi (kx_throttle), IEM MOS archive (NBS), Open-Meteo forecast.
# Copies ONLY the files this job needs + the dev training station-days (results/ops_e4p_forward/e4p_train_dev.parquet, only
# if absent: it is the frozen training base). Never overwrites deployed outputs (data/kx_forward/e4p.jsonl, e4p_fits.json).
# Stop: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-e4p   (all quant-sports jobs: touch ~/.claude/quant-sports/STOP)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
. "$SRC/scripts/atomic_deploy.sh"   # acp / awrite = tmp + mv deploys; never cp onto a live script (ops_robustness_1118)
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
[ -x "$DST/.venv/bin/python" ] || { echo "deployed venv missing: run scripts/install_quant_sports_agents.sh first" >&2; exit 1; }
mkdir -p "$DST/scripts" "$DST/data/kx_forward" "$DST/logs" "$DST/state"
for f in qs_job.sh kx_throttle.py e4p_paper.py; do acp "$SRC/scripts/$f" "$DST/scripts/$f"; done
chmod +x "$DST/scripts/qs_job.sh"
[ -f "$DST/data/kx_forward/e4p_train_dev.parquet" ] || acp "$SRC/results/ops_e4p_forward/e4p_train_dev.parquet" "$DST/data/kx_forward/e4p_train_dev.parquet"
label="com.jonathanwang.quant-sports-e4p"
cat > "$LA/$label.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/scripts/qs_job.sh</string><string>e4p</string></array>
  <key>WorkingDirectory</key><string>$DST</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  <key>StartInterval</key><integer>300</integer>
  <key>StandardOutPath</key><string>$DST/logs/launchd-e4p.out</string>
  <key>StandardErrorPath</key><string>$DST/logs/launchd-e4p.err</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
echo "installed $label (every 300 s)"
echo "verify: tail $DST/logs/e4p.log ; cat $DST/state/e4p_heartbeat ; $DST/.venv/bin/python $DST/scripts/e4p_paper.py status"
