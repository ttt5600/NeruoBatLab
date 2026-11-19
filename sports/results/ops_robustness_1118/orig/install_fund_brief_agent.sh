#!/bin/bash
# Install (or refresh) the fund DAILY PM BRIEF as a launchd agent: every day at 07:30 local.
# READ-ONLY: runs scripts/fund_brief.py, which calls the executor's `status` and the paper loggers' `report` commands
# (no orders) and writes ~/.claude/quant-kalshi/notes/briefs/brief_YYYY-MM-DD.md. Account balances are redacted.
# Shares ~/.claude/quant-kalshi (code copy + venv) with the other agents (launchd cannot read ~/Desktop, TCC), so the
# ledger + lockbox are SNAPSHOTS copied at install time: re-run this script after new trials to refresh the gate.
# Log: ~/.claude/quant-kalshi/fund_brief.log. Remove: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-fund-brief
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-fund-brief
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/lab" "$DST/notes/briefs"
for f in fund_brief.py fund_stats.py; do cp "$SRC/scripts/$f" "$DST/scripts/"; done
cp "$SRC/lab/ledger.jsonl" "$SRC/lab/lockbox.jsonl" "$DST/lab/"
cp "$SRC/QUESTIONS.md" "$DST/QUESTIONS.md"   # Q003/Q004 status sections of the brief
for f in kalshi_exec.py nba_open_totals.py open_lines_logger.py kalshi_wx_monthly_logger.py nba_midgame_scan.py manual_bets.py forecastex_paper.py; do
  [ -f "$DST/scripts/$f" ] || echo "WARNING: $DST/scripts/$f missing (deploy its own agent first); the brief will show it as missing"
done

cat > "$DST/fund_brief.sh" <<EOS
#!/bin/bash
cd "$DST"
printf '=== %s\n' "\$(date '+%F %T')" >> fund_brief.log
"$DST/.venv/bin/python" scripts/fund_brief.py >> fund_brief.log 2>&1
EOS
chmod +x "$DST/fund_brief.sh"

cat > "$PLIST" <<EOS
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/fund_brief.sh</string></array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>7</integer><key>Minute</key><integer>30</integer></dict>
  <key>StandardErrorPath</key><string>$DST/fund_brief.err</string>
</dict></plist>
EOS

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (daily 07:30 local; briefs in $DST/notes/briefs/)"
echo "verify now: launchctl kickstart gui/\$(id -u)/$LABEL && tail $DST/fund_brief.log"
