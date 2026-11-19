#!/bin/bash
# Install (or refresh) the durable Kalshi paper trader: launchd cannot read ~/Desktop (TCC),
# so code + venv + log live in ~/.claude/quant-kalshi. Re-run after editing kalshi_paper.py or sports/.
#   tick hourly at :03 = pick every city in its decision window (22:00-23:59 CITY-local, once per
#   city-night) + settle. No wall-clock dispatch: launchd runs late after sleep (10-16 lost every pick).
# kalshi_paper.py places no orders; kalshi_exec.py runs after each tick in DRY mode (log data/kalshi_exec.jsonl).
# Real orders need the USER to put KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH, KALSHI_BANKROLL_USD and EXEC_MODE=live
# in $DST/.env (EXEC_ENV=demo sends them to Kalshi's demo exchange instead). See sports/notes/go_live.md.
# This script never creates or reads $DST/.env; it only tightens its permissions if it exists.
# Log: ~/.claude/quant-kalshi/data/kalshi_paper.jsonl
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-kalshi"
LABEL=com.jonathanwang.quant-kalshi-paper
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$DST/scripts" "$DST/data"
# atomic deploys (ops_robustness_1118): rsync already writes tmp + rename; single files go tmp-in-same-dir + mv,
# so a job running at that moment keeps reading the old inode (bash reads scripts lazily; cp over it corrupts it).
. "$SRC/scripts/atomic_deploy.sh"
rsync -a --delete --exclude __pycache__ "$SRC/sports/" "$DST/sports/"
for f in kalshi_paper.py kalshi_exec.py om_fetch.py; do acp "$SRC/scripts/$f" "$DST/scripts/$f"; done
if [ -f "$DST/.env" ]; then chmod 600 "$DST/.env"; fi
"$DST/.venv/bin/python" -c "import cryptography" 2>/dev/null || "$DST/.venv/bin/pip" install -q cryptography
if [ ! -x "$DST/.venv/bin/python" ]; then
  /opt/homebrew/opt/python@3.14/bin/python3.14 -m venv "$DST/.venv"
  "$DST/.venv/bin/pip" install -q numpy pandas requests pyarrow
fi

RUN_TMP="$(mktemp "$DST/run.sh.tmp.XXXXXX")"
cat > "$RUN_TMP" <<EOF
#!/bin/bash
# \$1 = tick | pick | settle. Appends a timestamped run record to run.log.
cd "$DST"
echo "=== \$(date '+%F %T') \$1" >> run.log
"$DST/.venv/bin/python" scripts/kalshi_paper.py "\$1" >> run.log 2>&1
rc=\$?
echo "exit \$rc" >> run.log
# shadow execution: size each fresh pick against the live book. DRY unless $DST/.env has EXEC_MODE=live
# (user-set) -- and even then only with --live, which this line passes; KILL file in $DST stops it.
if [ "\$1" = tick ]; then "$DST/.venv/bin/python" scripts/kalshi_exec.py run --live >> exec.log 2>&1; fi
# exit 3 = a city in its window could not be fetched. AFTER exec has acted on the cities that worked, re-try the
# failed ones every 90 s for up to 20 min (same source + params), then exec again for any new pick; a city still
# unfetched at the end of its window is logged to data/kalshi_paper_lost.jsonl + macOS alert (ops_robustness_1118).
if [ "\$1" = tick ] && [ "\$rc" = 3 ]; then
  echo "=== \$(date '+%F %T') retry" >> run.log
  "$DST/.venv/bin/python" scripts/kalshi_paper.py retry >> run.log 2>&1
  echo "exit \$? (retry)" >> run.log
  "$DST/.venv/bin/python" scripts/kalshi_exec.py run --live >> exec.log 2>&1
fi
EOF
chmod +x "$RUN_TMP"; mv -f "$RUN_TMP" "$DST/run.sh"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DST/dispatch.sh</string></array>
  <key>StartCalendarInterval</key><array>
    <dict><key>Minute</key><integer>3</integer></dict>
  </array>
  <key>StandardErrorPath</key><string>$DST/launchd.err</string>
</dict></plist>
EOF

D_TMP="$(mktemp "$DST/dispatch.sh.tmp.XXXXXX")"
cat > "$D_TMP" <<EOF
#!/bin/bash
exec "$DST/run.sh" tick
EOF
chmod +x "$D_TMP"; mv -f "$D_TMP" "$DST/dispatch.sh"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL -> $DST"
