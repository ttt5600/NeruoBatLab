#!/bin/bash
# Sync between the worktree (sports/) and the launchd-deployed copy ~/.claude/quant-sports (launchd cannot read
# ~/Desktop: macOS TCC). From 2026-11-01 the DEPLOYED copy is the source of truth for the game-day logs.
#
#   scripts/qs_sync.sh push   code only -> deployed (scripts/, sports/ pkg, lab/*.py, results/*.json params,
#                             notes/STATE.md copy for the digest). Never touches deployed logs. Also copies
#                             data/user_calls.jsonl worktree -> deployed (the worktree is that file's writer).
#   scripts/qs_sync.sh pull   deployed logs -> worktree (for analysis / reports). Overwrites the worktree copies.
#   scripts/qs_sync.sh seed   first deploy only: push + copy the data the jobs read, never overwriting a deployed file.
#   scripts/qs_sync.sh diff   show which log files differ between the two copies.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/.claude/quant-sports"
RS=(rsync -a --exclude __pycache__ --exclude '*.pyc')

# logs the deployed jobs write (relative to sports/data). Deployed = truth.
LOGS=(nba_early_form_log.jsonl l32b_log.jsonl l32b_state.json kalshi_prop_log.jsonl kalshi_prop_grades.jsonl
      tickets/ q017/fwd/ e1x/tsa_paper.jsonl)

push() {
  mkdir -p "$DST/scripts" "$DST/sports" "$DST/lab" "$DST/results" "$DST/notes" "$DST/data"
  "${RS[@]}" "$SRC/scripts/" "$DST/scripts/"
  "${RS[@]}" --delete "$SRC/sports/" "$DST/sports/"
  "${RS[@]}" --include '*.py' --exclude '*' "$SRC/lab/" "$DST/lab/"
  "${RS[@]}" --include '*.json' --exclude '*' "$SRC/results/" "$DST/results/"
  cp "$SRC/scripts/qs_digest_prompt.md" "$DST/digest_prompt.md"
  [ -f "$SRC/STATE.md" ] && cp "$SRC/STATE.md" "$DST/notes/STATE.md"
  [ -f "$SRC/data/user_calls.jsonl" ] && "${RS[@]}" --update "$SRC/data/user_calls.jsonl" "$DST/data/"
  echo "pushed code -> $DST (logs untouched)"
}

pull() {
  for f in "${LOGS[@]}"; do
    if [ -e "$DST/data/$f" ]; then
      mkdir -p "$(dirname "$SRC/data/$f")"
      "${RS[@]}" "$DST/data/$f" "$SRC/data/$f"
    fi
  done
  mkdir -p "$SRC/data/quant_sports_logs"
  "${RS[@]}" "$DST/logs/" "$SRC/data/quant_sports_logs/" 2>/dev/null || true
  echo "pulled deployed logs -> $SRC/data (job run logs -> data/quant_sports_logs/)"
}

seed() {
  push
  cd "$SRC"
  # data the jobs read; --ignore-existing so a re-seed never clobbers what the deployed jobs wrote
  for f in espn/nba nba_sim nba_simple_model nba_pbp/game_terms.parquet nba_comebacks/teamlogs.parquet q017 e1x \
           "${LOGS[@]}" user_calls.jsonl; do
    f="${f%/}"
    [ -e "data/$f" ] || continue
    mkdir -p "$(dirname "$DST/data/$f")"
    "${RS[@]}" --ignore-existing "data/$f" "$DST/data/$(dirname "$f")/"
  done
  "${RS[@]}" --ignore-existing --exclude '*.csv' "results/" "$DST/results/"
  echo "seeded data -> $DST/data"
}

difflogs() {
  for f in "${LOGS[@]}" user_calls.jsonl; do
    a="$SRC/data/$f"; b="$DST/data/$f"
    if [ ! -e "$a" ] && [ ! -e "$b" ]; then s="absent in both"
    elif [ -d "$b" ] || [ -d "$a" ]; then diff -rq "$a" "$b" >/dev/null 2>&1 && s=same || s=DIFF
    else cmp -s "$a" "$b" && s=same || s=DIFF; fi
    printf '%-28s %s\n' "$f" "$s"
  done
}

case "${1:-}" in
  push) push ;; pull) pull ;; seed) seed ;; diff) difflogs ;;
  *) echo "usage: $0 push|pull|seed|diff" >&2; exit 2 ;;
esac
