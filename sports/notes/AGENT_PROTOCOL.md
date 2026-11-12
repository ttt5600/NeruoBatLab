# AGENT PROTOCOL -- how research subagents save what they find (written 2026-11-09, user: "update context and memory")

Every subagent launched from the quant-trader lead follows this. Findings that live only in a chat reply are lost at the
next /clear; the files below are what survives.

## Write as you go, not at the end
1. **On start**: append a `started` row to `sports/notes/findings_inbox.jsonl` (format below).
2. **Own notes file**: `sports/notes/<agent_topic>.md`. Create it in your first few minutes with the question, the
   pre-registered success/kill rule (before you look at any outcome), data sources and plan. Update it after every
   meaningful step: numbers with the file/command that produced them, wrong turns, what you ruled out.
3. **On any result** (pass, fail, partial, blocked): append a `result` row to the inbox AND update your notes file.
   Do this BEFORE writing your final reply.
4. **Final reply** to the lead = a short summary + paths. The lead merges inbox rows into EDGES_AND_AGENTS.md,
   FUND_BOOK.md, ACT_LIST.md, STATE.md and the memory dir, then commits.

5. **Learnings for the memory vault**: end your notes file with a `## Learnings` section, ONE `### <short claim>` heading
   per learning (claim, the number + source path, why it matters). The vault chunks at headings; a fact buried in a
   long paragraph is stored but never retrieved. The lead runs `vault ingest --only markdown` and messages the vault
   session (user 11-11: every agent's learnings must reach the vault).

## Inbox row (one JSON object per line, append with `>>`, never rewrite the file)
{"ts":"<UTC ISO>","agent":"<topic>","kind":"started|progress|result|blocked","edge":"<ID e.g. E4, L39, TSA, NHL-late, new>",
 "verdict":"confirmed|holds|weakened|dead|lead|blocked|n/a","headline":"<one line with the key number + interval>",
 "numbers":{...},"files":["sports/notes/x.md","sports/results/x/..."],"lesson":"<one line, or empty>",
 "memory":"<one-line fact worth remembering across sessions, or empty>"}

## Hard rules (inherit sports/STATE.md + ~/.claude/CLAUDE.md)
- NO orders, NO cancels, NO .env / launchd / kill-file changes, NO deploys to ~/.claude/quant-kalshi. Read-only GETs to
  Kalshi/IBKR APIs are fine. Anything that would spend money or change a live bot -> write it as a proposal in your notes.
- NO git commands (several agents share this worktree; the lead commits).
- Sealed / lockbox windows are opened only on the user's explicit yes. Dev-only unless told otherwise.
- Pre-register (question, success rule, kill rule) in your notes file before computing the outcome.
- Every number cites the file or command it came from. Intervals with every effect; a CI that crosses 0 = "not
  distinguishable". Timestamps through `~/agent-harness/lib/traps.py` checks (datetime64[us] // 1e9 trap, lookahead).
- Too-good results are bugs until the rows are inspected.
- Register any new trial in edgelab (`cd sports && ../.venv_quant/bin/python -m edgelab ...`) so the family bar counts it.
- Write scratch files to sports/results/<agent_topic>/, not the repo root.
