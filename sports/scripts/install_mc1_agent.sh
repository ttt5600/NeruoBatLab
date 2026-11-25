#!/bin/bash
# Install (or refresh) the READ-ONLY MC1 scanner (notes/small_capacity_census.md 'MC1 scanner'; pre-registered T-MTG
# paper week). launchd cannot read ~/Desktop (TCC) -> runs from the deployed copy ~/.claude/quant-sports via
# scripts/qs_job.sh mc1.
#   com.jonathanwang.quant-sports-mc1   launchd 1800 s; scripts/mc1_buylist_scan.py once per PT day at the first run
#                                       >= 06:30 PT (exit 2 = network failure -> retried next run, <= 6 tries/day)
# No orders, no accounts, no keys, no .env. Public downloads only (stdlib): mtgjson.com AllPricesToday.json.xz (~4.4 MB),
# api.cardkingdom.com/api/v2/pricelist gzip (~10 MB), api.scryfall.com/cards/collection (~12 POSTs, 0.6 s apart).
# Raw downloads live in a temp dir that is deleted on exit. Output: ~/.claude/quant-sports/data/mc1/<date>.json + .md
# (spot-check table) and daily.jsonl (one summary line per run; judge on the last line per date).
# Stop: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-mc1   (all quant-sports jobs: touch ~/.claude/quant-sports/STOP)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
. "$SRC/scripts/atomic_deploy.sh"   # acp = tmp + mv deploy; never cp onto a live script (ops_robustness_1118)
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
[ -x "$DST/.venv/bin/python" ] || { echo "deployed venv missing: run scripts/install_quant_sports_agents.sh first" >&2; exit 1; }
mkdir -p "$DST/scripts" "$DST/logs" "$DST/state" "$DST/data/mc1"
for f in qs_job.sh mc1_buylist_scan.py; do acp "$SRC/scripts/$f" "$DST/scripts/$f"; done
chmod +x "$DST/scripts/qs_job.sh"
label="com.jonathanwang.quant-sports-mc1"
cat > "$LA/$label.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/scripts/qs_job.sh</string><string>mc1</string></array>
  <key>WorkingDirectory</key><string>$DST</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  <key>StartInterval</key><integer>1800</integer>
  <key>StandardOutPath</key><string>$DST/logs/launchd-mc1.out</string>
  <key>StandardErrorPath</key><string>$DST/logs/launchd-mc1.err</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
echo "installed $label (every 1800 s; mc1_buylist_scan.py once per PT day at the first run >= 06:30 PT)"
echo "verify: tail $DST/logs/mc1.log ; cat $DST/state/mc1_day $DST/state/mc1_tries ; tail -3 $DST/data/mc1/daily.jsonl"
