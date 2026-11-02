#!/bin/bash
# Headless fund digest (launchd com.jonathanwang.quant-sports-digest, 09:00 + 23:30 PT; or by hand).
#   1. scripts/fund_status.py --out digests/<date>-<tag>  -> .status.md (bundle), .fallback.md, .headline.txt
#   2. claude -p (sonnet, max 15 turns, read-only tools, no MCP, no user hooks) on digest_prompt.md + bundle
#      -> digests/<date>-<tag>.md ; usage/cost appended to digests/usage.jsonl
#   3. if claude fails or returns nothing: the no-LLM fallback digest becomes digests/<date>-<tag>.md
#   4. macOS notification with the headline.
# $1 = am | pm (default: am before 12:00 PT, else pm). Writes only under ~/.claude/quant-sports/digests.
set -u
QS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$QS" || exit 1
export TZ=America/Los_Angeles PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
CLAUDE=/opt/homebrew/bin/claude
tag="${1:-}"; [ -n "$tag" ] || { [ "$(date +%H)" -lt 12 ] && tag=am || tag=pm; }
d=$(date +%F); P="digests/$d-$tag"; OUT="$P.md"
mkdir -p digests
echo "=== $(date '+%F %T %Z') digest $tag"
.venv/bin/python scripts/fund_status.py --out "$P" --tag "$tag" || echo "fund_status.py failed (exit $?)"

mode=fallback
if [ -x "$CLAUDE" ] && [ -s "$P.status.md" ] && [ "${QS_DIGEST_NO_LLM:-0}" != 1 ]; then
  cat digest_prompt.md "$P.status.md" | "$CLAUDE" -p --model sonnet --max-turns 15 --output-format json \
      --setting-sources project --strict-mcp-config --no-session-persistence \
      --allowedTools "Read" "Bash(tail:*)" "Bash(grep:*)" "Bash(ls:*)" "Bash(launchctl list)" \
      --disallowedTools "Edit" "Write" "NotebookEdit" "WebFetch" "WebSearch" \
      > "$P.raw.json" 2> "$P.claude.err"
  rc=$?
  .venv/bin/python - "$P" "$rc" <<'EOF'
import json, sys, time
p, rc = sys.argv[1], int(sys.argv[2])
try:
    j = json.load(open(p + ".raw.json"))
except Exception as e:
    print(f"claude output unreadable (rc {rc}): {e}"); sys.exit(1)
res = (j.get("result") or "").strip()
u = j.get("usage") or {}
rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "digest": p, "rc": rc, "is_error": j.get("is_error"),
       "num_turns": j.get("num_turns"), "duration_ms": j.get("duration_ms"), "total_cost_usd": j.get("total_cost_usd"),
       "input_tokens": u.get("input_tokens"), "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
       "cache_read_input_tokens": u.get("cache_read_input_tokens"), "output_tokens": u.get("output_tokens"),
       "models": list((j.get("modelUsage") or {}).keys())}
open("digests/usage.jsonl", "a").write(json.dumps(rec) + "\n")
print("usage", json.dumps(rec))
if rc != 0 or j.get("is_error") or "HEADLINE:" not in res:
    print(f"claude digest rejected (rc {rc}, is_error {j.get('is_error')}): {res[:200]}"); sys.exit(1)
lines = res.splitlines()
i = next(k for k, l in enumerate(lines) if l.startswith("HEADLINE:"))
open(p + ".headline.txt", "w").write(lines[i][len("HEADLINE:"):].strip() + "\n")
open(p + ".md", "w").write("\n".join(lines[i:i + 26]) + "\n")
EOF
  [ $? -eq 0 ] && mode=llm
fi
if [ "$mode" = fallback ]; then cp "$P.fallback.md" "$OUT"; fi
head_line=$(head -1 "$P.headline.txt" 2>/dev/null | tr -d '"\\' | cut -c1-200)
/usr/bin/osascript -e "display notification \"${head_line:-digest written}\" with title \"Fund digest $tag ($mode)\"" \
  || echo "osascript failed"
echo "digest $OUT written ($mode): $head_line"
