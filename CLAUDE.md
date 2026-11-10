# quant-trader worktree (branch worktree-quant-trader)

This worktree is the sports-betting / quant project. Start every session by reading
`sports/STATE.md`; it says where things stand, what is waiting on the user, and the hard rules.
Then read `sports/notes/EDGES_AND_AGENTS.md`: every edge (passed / live / sealed ask / lead / dead), every research agent
(ID, brief, findings, output files, how to reach or re-launch) and the peer sessions (Robinhood lives in
`connect-robinhood-mcp`). Update it whenever an agent finishes or a trial is scored.

Every research subagent gets `sports/notes/AGENT_PROTOCOL.md` in its brief: it writes its own notes file and appends
rows to `sports/notes/findings_inbox.jsonl` AS IT GOES (no git, no orders). The lead merges unmerged inbox rows into
EDGES_AND_AGENTS / FUND_BOOK / ACT_LIST / STATE + memory and commits. After a /clear, check the inbox for rows newer
than the last merge (`"kind":"merged"` marker rows) before anything else.
