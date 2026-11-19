#!/bin/bash
# Install (or refresh) the READ-ONLY NHL paper jobs (notes/ops_nhl_goalie.md). launchd cannot read ~/Desktop (TCC) ->
# runs from the deployed copy ~/.claude/quant-sports via scripts/qs_job.sh.
#   com.jonathanwang.quant-sports-nhl-om-a    NHL-om-A forward paper (nhl_om_a_kft-754cfc46)  launchd 1800 s, 24 h;
#                                             MoneyPuck refresh once a day from 05:00 PT
#   com.jonathanwang.quant-sports-nhl-goalie  goalie-confirmation timestamps + Kalshi NHL ladder snapshots, 120 s
#                                             (idle exit when no NHL game within 14 h)
# Copies ONLY the files these jobs need. No orders, no keys, no .env. Never overwrites deployed outputs.
# Stop one: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-<job>   (all quant-sports jobs: touch ~/.claude/quant-sports/STOP)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
. "$SRC/scripts/atomic_deploy.sh"   # acp / awrite = tmp + mv deploys; never cp onto a live script (ops_robustness_1118)
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
[ -x "$DST/.venv/bin/python" ] || { echo "deployed venv missing: run scripts/install_quant_sports_agents.sh first" >&2; exit 1; }
RS=(rsync -a --exclude __pycache__ --exclude '*.pyc')
OM=results/nhl_opener_model
mkdir -p "$DST/scripts" "$DST/$OM" "$DST/results/ops_nhl_goalie" "$DST/results/kalshi_openers_xleague" \
         "$DST/data/kx_forward" "$DST/data/nhl_goalie" "$DST/logs" "$DST/state"
for f in qs_job.sh kx_throttle.py kx_forward.py nhl_om_forward.py nhl_goalie_logger.py; do acp "$SRC/scripts/$f" "$DST/scripts/$f"; done
acp "$SRC/results/kalshi_openers_xleague/kx.py" "$DST/results/kalshi_openers_xleague/kx.py"
for f in model.py fetch_mp.py; do acp "$SRC/$OM/$f" "$DST/$OM/$f"; done
acp "$SRC/results/ops_nhl_goalie/c_seed_A.csv" "$DST/results/ops_nhl_goalie/c_seed_A.csv"
"${RS[@]}" --ignore-existing "$SRC/$OM/mp" "$DST/$OM/"      # seed; the job refreshes it daily
chmod +x "$DST/scripts/qs_job.sh"

plist() {  # plist <job> <interval-s>
  local label="com.jonathanwang.quant-sports-$1"
  cat > "$LA/$label.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/scripts/qs_job.sh</string><string>$1</string></array>
  <key>WorkingDirectory</key><string>$DST</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
  </dict>
  <key>StartInterval</key><integer>$2</integer>
  <key>StandardOutPath</key><string>$DST/logs/launchd-$1.out</string>
  <key>StandardErrorPath</key><string>$DST/logs/launchd-$1.err</string>
</dict></plist>
EOF
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
  echo "installed $label (every $2 s)"
}
plist nhl-om-a 1800
plist nhl-goalie 120
echo "verify: tail $DST/logs/nhl-om-a.log $DST/logs/nhl-goalie.log ; $DST/.venv/bin/python $DST/scripts/nhl_om_forward.py status"
