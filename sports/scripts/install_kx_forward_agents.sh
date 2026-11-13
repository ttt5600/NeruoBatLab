#!/bin/bash
# Install (or refresh) the READ-ONLY Kalshi-execution forward paper loggers + LX1 scanner (notes/ops_forward_wiring.md).
# launchd cannot read ~/Desktop (TCC) -> runs from the deployed copy ~/.claude/quant-sports via scripts/qs_job.sh.
#   com.jonathanwang.quant-sports-kx-k1       L32b-K1 pricing/settle   launchd 600 s, acts every >= 25 min (24 h)
#   com.jonathanwang.quant-sports-kx-knfl     K-NFL + K50-NFL          launchd 600 s, acts every >= 2 h (24 h)
#   com.jonathanwang.quant-sports-kx-kcfb     K-CFB forward pipeline   launchd 1800 s, acts once/day from 07:00 PT
#   com.jonathanwang.quant-sports-kx-openers  openers timing logger    launchd 600 s, 09:00-24:00 PT
#   com.jonathanwang.quant-sports-lx1         LX1 arb scanner          launchd 900 s; scan every >= 2 h until 11-20 08:00Z,
#                                                                      persist+report daily from 06:00 PT
# Copies ONLY the files these jobs need (no full qs_sync push, so the other deployed jobs keep their code).
# No orders, no keys, no .env; the live executors are not touched. Never overwrites a deployed log.
# Stop one: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-<job>   (all quant-sports jobs: touch ~/.claude/quant-sports/STOP)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-sports"
LA="$HOME/Library/LaunchAgents"
[ -x "$DST/.venv/bin/python" ] || { echo "deployed venv missing: run scripts/install_quant_sports_agents.sh first" >&2; exit 1; }
RS=(rsync -a --exclude __pycache__ --exclude '*.pyc')
# K-CFB's frozen L36 model parses the SBR html archive (pandas.read_html -> lxml); pinned to ../.venv_quant's versions
"$DST/.venv/bin/python" -c "import lxml, bs4" 2>/dev/null || "$DST/.venv/bin/pip" install -q lxml==6.1.3 beautifulsoup4==4.15.0
X=results/kalshi_openers_xleague
mkdir -p "$DST/scripts" "$DST/$X" "$DST/results/lx1_logical_arb" "$DST/data/lab" "$DST/data/kx_forward" "$DST/logs" "$DST/state"

# code (overwrite = refresh)
for f in qs_job.sh kx_throttle.py kx_forward.py lx1_arb_scan.py l36_cfb_sealed.py cfb_simple_model.py; do
  cp "$SRC/scripts/$f" "$DST/scripts/$f"
done
for f in kx.py q1_fetch.py q1_timing.py q2_cfb.py q2_cfb_preds.py tonight_logger.py; do cp "$SRC/$X/$f" "$DST/$X/$f"; done
chmod +x "$DST/scripts/qs_job.sh"
# data the jobs read (never clobber what the deployed jobs wrote)
"${RS[@]}" --ignore-existing "$SRC/data/sbr_cfb" "$DST/data/"
"${RS[@]}" --ignore-existing "$SRC/data/lab/l36_espn_to_sbr_names.json" "$DST/data/lab/"
"${RS[@]}" --ignore-existing "$SRC/$X/cfb_espn" "$DST/$X/"
# LX1: carry the 11-12 scan history so `report` counts the whole 11-12..11-19 window
for f in violations.jsonl scans.jsonl candidates.jsonl yes_dutch_unverified.jsonl persistence.csv; do
  [ -e "$SRC/results/lx1_logical_arb/$f" ] && "${RS[@]}" --ignore-existing "$SRC/results/lx1_logical_arb/$f" "$DST/results/lx1_logical_arb/"
done

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
plist kx-k1 600
plist kx-knfl 600
plist kx-kcfb 1800
plist kx-openers 600
plist lx1 900
echo "verify: tail $DST/logs/<job>.log ; $DST/.venv/bin/python $DST/scripts/kx_forward.py status"
