#!/bin/bash
# Install (or refresh) the READ-ONLY CB1 crypto barrier paper logger (notes/crypto_barrier.md). launchd cannot read
# ~/Desktop (TCC) -> runs from the deployed copy ~/.claude/quant-sports via scripts/qs_job.sh.
#   com.jonathanwang.quant-sports-crypto-barrier   launchd 300 s; decides once per UTC day from 16:00 UTC (Deribit DVOL,
#                                                  Coinbase public candles/ticker, Kalshi public GETs), settles hourly
# No orders, no keys, no .env. Copies ONLY the files this job needs. Never overwrites deployed outputs
# (data/kx_forward/crypto_barrier*.jsonl).
# Stop: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-crypto-barrier   (all jobs: touch ~/.claude/quant-sports/STOP)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
. "$SRC/scripts/atomic_deploy.sh"   # acp / awrite = tmp + mv deploys; never cp onto a live script (ops_robustness_1118)
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
[ -x "$DST/.venv/bin/python" ] || { echo "deployed venv missing: run scripts/install_quant_sports_agents.sh first" >&2; exit 1; }
mkdir -p "$DST/scripts" "$DST/data/kx_forward" "$DST/logs" "$DST/state"
for f in qs_job.sh kx_throttle.py crypto_barrier_model.py crypto_barrier_v2_model.py crypto_barrier.py crypto_barrier_forward.py; do acp "$SRC/scripts/$f" "$DST/scripts/$f"; done
chmod +x "$DST/scripts/qs_job.sh"
label="com.jonathanwang.quant-sports-crypto-barrier"
cat > "$LA/$label.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/scripts/qs_job.sh</string><string>crypto-barrier</string></array>
  <key>WorkingDirectory</key><string>$DST</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  <key>StartInterval</key><integer>300</integer>
  <key>StandardOutPath</key><string>$DST/logs/launchd-crypto-barrier.out</string>
  <key>StandardErrorPath</key><string>$DST/logs/launchd-crypto-barrier.err</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
echo "installed $label (every 300 s)"
echo "verify: cat $DST/state/crypto_barrier_heartbeat ; tail $DST/logs/crypto-barrier.log ; $DST/.venv/bin/python $DST/scripts/crypto_barrier_forward.py status"
