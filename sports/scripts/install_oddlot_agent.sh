#!/bin/bash
# Install (or refresh) the READ-ONLY odd-lot tender watcher (notes/money_now.md section C). launchd cannot read
# ~/Desktop (TCC) -> runs from the deployed copy ~/.claude/quant-sports via scripts/qs_job.sh oddlot.
#   com.jonathanwang.quant-sports-oddlot   launchd 1800 s; scripts/oddlot_watch.py once per PT day at the first run >= 07:15 PT
# No orders, no keys, no .env. Public GETs to efts.sec.gov / www.sec.gov only (stdlib). Output:
# ~/.claude/quant-sports/data/oddlot/filings.jsonl (append-only, deduped by accession; never overwritten here).
# New filings surface in the fund SessionStart hook (scripts/ctx_inject.py).
# Stop: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-oddlot   (all quant-sports jobs: touch ~/.claude/quant-sports/STOP)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
. "$SRC/scripts/atomic_deploy.sh"   # acp = tmp + mv deploy; never cp onto a live script (ops_robustness_1118)
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
[ -x "$DST/.venv/bin/python" ] || { echo "deployed venv missing: run scripts/install_quant_sports_agents.sh first" >&2; exit 1; }
mkdir -p "$DST/scripts" "$DST/logs" "$DST/state" "$DST/data/oddlot"
for f in qs_job.sh oddlot_watch.py; do acp "$SRC/scripts/$f" "$DST/scripts/$f"; done
chmod +x "$DST/scripts/qs_job.sh"
label="com.jonathanwang.quant-sports-oddlot"
cat > "$LA/$label.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/scripts/qs_job.sh</string><string>oddlot</string></array>
  <key>WorkingDirectory</key><string>$DST</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  <key>StartInterval</key><integer>1800</integer>
  <key>StandardOutPath</key><string>$DST/logs/launchd-oddlot.out</string>
  <key>StandardErrorPath</key><string>$DST/logs/launchd-oddlot.err</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
echo "installed $label (every 1800 s; oddlot_watch.py once per PT day at the first run >= 07:15 PT)"
echo "verify: tail $DST/logs/oddlot.log ; cat $DST/state/oddlot_day ; wc -l $DST/data/oddlot/filings.jsonl"
