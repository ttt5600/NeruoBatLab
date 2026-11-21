#!/bin/bash
# Install (or refresh) the READ-ONLY KXAAAGASCD* state gas order-book logger (notes/ops_state_gas_book.md). launchd
# cannot read ~/Desktop (TCC) -> runs from the deployed copy ~/.claude/quant-sports via scripts/qs_job.sh.
#   com.jonathanwang.quant-sports-source-shift   fires every 3600 s; source_shift_watch hourly (notes/source_shift.md), READ-ONLY
#   (slot am) and once in [02:15,03:05) ET (slot close) per day -> data/kx_forward/state_gas_books.jsonl.gz
# Public GETs only (kx_throttle, ~100 per snapshot). No orders, no keys, no .env. Append-only output.
# Stop: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-source-shift
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
. "$SRC/scripts/atomic_deploy.sh"   # acp / awrite = tmp + mv deploys; never cp onto a live script (ops_robustness_1118)
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
[ -x "$DST/.venv/bin/python" ] || { echo "deployed venv missing: run scripts/install_quant_sports_agents.sh first" >&2; exit 1; }
mkdir -p "$DST/scripts" "$DST/logs" "$DST/state" "$DST/data/source_shift"
for f in qs_job.sh kx_throttle.py source_shift_watch.py; do acp "$SRC/scripts/$f" "$DST/scripts/$f"; done
chmod +x "$DST/scripts/qs_job.sh"
label="com.jonathanwang.quant-sports-source-shift"
cat > "$LA/$label.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/scripts/qs_job.sh</string><string>source-shift</string></array>
  <key>WorkingDirectory</key><string>$DST</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  <key>StartInterval</key><integer>3600</integer>
  <key>StandardOutPath</key><string>$DST/logs/launchd-source-shift.out</string>
  <key>StandardErrorPath</key><string>$DST/logs/launchd-source-shift.err</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
echo "installed $label (hourly; source_shift_watch, READ-ONLY)"
echo "verify: tail $DST/logs/source-shift.log; tail -1 $DST/data/source_shift/snapshots.jsonl"
