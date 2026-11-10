#!/bin/bash
# PreCompact hook: commit any uncommitted fund notes (STATE/ACT_LIST/FUND_BOOK/EDGES) so compaction can never lose them.
# Only these note files; never code, data or the edgelab registry. Local commit only (no push). Never blocks compaction.
W="$(cd "$(dirname "$0")/../.." && pwd)"
FILES="sports/STATE.md sports/notes/ACT_LIST.md sports/notes/FUND_BOOK.md sports/notes/EDGES_AND_AGENTS.md"
changed=$(git -C "$W" status --porcelain -- $FILES)
if [ -n "$changed" ]; then
  git -C "$W" add -- $FILES && git -C "$W" commit -q -m "sports: auto pre-compact snapshot of fund notes" -- $FILES \
    && echo '{"systemMessage": "pre-compact: fund notes committed"}'
else
  echo '{"systemMessage": "pre-compact: fund notes already committed"}'
fi
exit 0
