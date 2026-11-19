#!/bin/bash
# Install (or refresh) the L40b Kalshi LIP NON-weather paper logger (notes/kalshi_lip_b.md) as a launchd agent:
# tick every 60 s. Shares ~/.claude/quant-kalshi (code copy + venv) with the other paper agents (launchd cannot read
# ~/Desktop, TCC). lip_paper_b.py imports lip_paper.py, so both are copied.
# Paper only: no orders, public GET endpoints only, no API key.
# FIRST RUN: the sample must exist before the first tick:
#   cd ~/.claude/quant-kalshi && .venv/bin/python scripts/lip_paper_b.py sample     (refuses to redraw)
# Logs: ~/.claude/quant-kalshi/data/lip_b_{sample.json,state.json,ticks.jsonl,events.jsonl}, lip_paper_b.log/.err
# Report: cd ~/.claude/quant-kalshi && .venv/bin/python scripts/lip_paper_b.py report
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-lip-paper-b
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data"
cp "$SRC/scripts/lip_paper.py" "$SRC/scripts/lip_paper_b.py" "$DST/scripts/"
"$DST/.venv/bin/python" -c "import requests" 2>/dev/null || "$DST/.venv/bin/pip" install -q requests

cat > "$DST/lip_paper_b.sh" <<EOF
#!/bin/bash
cd "$DST"
[ -f data/lip_b_sample.json ] || exit 0
out=\$("$DST/.venv/bin/python" scripts/lip_paper_b.py tick 2>&1 | grep -v Warning)
if [ -n "\$out" ]; then printf '=== %s\n%s\n' "\$(date '+%F %T')" "\$out" >> lip_paper_b.log; fi
EOF
chmod +x "$DST/lip_paper_b.sh"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/lip_paper_b.sh</string></array>
  <key>StartInterval</key><integer>60</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardErrorPath</key><string>$DST/lip_paper_b.err</string>
</dict></plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST (every 60 s)"
