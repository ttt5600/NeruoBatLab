#!/bin/bash
# Fetch meta.yaml (title, tournament slug, type, score) for every question logged by the public
# m-ad/claude_forecasting_auto bot repo. Read-only GitHub API; output one JSON line per question.
OUT=/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports/results/metaculus_bot_tourney/raw/mad_meta.jsonl
SL=/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports/results/metaculus_bot_tourney/raw/mad_question_slugs.txt
: > "$OUT"
grep -v gitkeep "$SL" | while read s; do
  y=$(gh api "repos/m-ad/claude_forecasting_auto/contents/questions/$s/meta.yaml" --jq .content 2>/dev/null | base64 -d 2>/dev/null)
  python3 -I -c '
import sys,json,yaml
d=yaml.safe_load(sys.stdin.read()) or {}
t=d.get("tournament") or {}
r=d.get("resolution") or {}
print(json.dumps({"slug":sys.argv[1],"title":d.get("title"),"type":d.get("type"),"tourn":t.get("slug"),"score":t.get("official_score"),"missed":t.get("missed_window"),"opened":t.get("opened"),"outcome":r.get("outcome")}))' "$s" <<< "$y" >> "$OUT" 2>/dev/null || echo "{\"slug\":\"$s\",\"err\":1}" >> "$OUT"
done
wc -l "$OUT"
