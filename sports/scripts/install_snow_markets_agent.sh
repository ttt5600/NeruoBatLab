#!/bin/bash
# Install (or refresh) the READ-ONLY snow-markets paper logger (TG1 Thanksgiving / SD1 + SD2 KXSNOWDAYS scalar;
# prereg notes/snow_markets.md, ops section 'Ops'). launchd cannot read ~/Desktop (TCC) -> runs from the deployed copy
# ~/.claude/quant-sports via scripts/qs_job.sh snow-markets.
#   com.jonathanwang.quant-sports-snow-markets   launchd 1800 s; `log` once per UTC day at the first run >= 15:30 UTC
#                                                through 2027-04-01; `settle` once per UTC day from 2026-11-27
# No orders, no keys, no .env. Public GETs: Kalshi (kx_throttle), ACIS StnData, Open-Meteo ensemble.
# Copies ONLY the files this job needs + the static inputs (census.json, ACIS climatology cache). The snapshot log is
# seeded from the repo copy ONLY if the deployed one is absent (it carries the first snapshot 2026-11-16 11:07Z with the
# 5 SD1 fires); deployed outputs (snow_log.jsonl, snow_settle.jsonl) are never overwritten.
# Stop: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-snow-markets   (all quant-sports jobs: touch ~/.claude/quant-sports/STOP)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
. "$SRC/scripts/atomic_deploy.sh"   # acp / awrite = tmp + mv deploys; never cp onto a live script (ops_robustness_1118)
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
[ -x "$DST/.venv/bin/python" ] || { echo "deployed venv missing: run scripts/install_quant_sports_agents.sh first" >&2; exit 1; }
"$DST/.venv/bin/python" -c "import requests" || { echo "deployed venv lacks requests" >&2; exit 1; }
R=results/snow_markets
mkdir -p "$DST/scripts" "$DST/$R" "$DST/logs" "$DST/state"
for f in qs_job.sh kx_throttle.py snow_markets.py; do acp "$SRC/scripts/$f" "$DST/scripts/$f"; done
chmod +x "$DST/scripts/qs_job.sh"
acp "$SRC/$R/census.json" "$SRC"/$R/acis_*.json "$DST/$R/"
[ -f "$DST/$R/snow_log.jsonl" ] || acp "$SRC/$R/snow_log.jsonl" "$DST/$R/snow_log.jsonl"
label="com.jonathanwang.quant-sports-snow-markets"
cat > "$LA/$label.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/scripts/qs_job.sh</string><string>snow-markets</string></array>
  <key>WorkingDirectory</key><string>$DST</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  <key>StartInterval</key><integer>1800</integer>
  <key>StandardOutPath</key><string>$DST/logs/launchd-snow-markets.out</string>
  <key>StandardErrorPath</key><string>$DST/logs/launchd-snow-markets.err</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
echo "installed $label (every 1800 s; log >= 15:30 UTC once/day through 2027-04-01; settle daily from 2026-11-27)"
echo "verify: tail $DST/logs/snow-markets.log ; wc -l $DST/$R/snow_log.jsonl ; cat $DST/state/snow_log_day"
