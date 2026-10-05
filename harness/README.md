# Research harness

Agents that keep the search for a better encoder moving without someone at the keyboard. Each
finished experiment is scored the same way and written into the knowledge base before the next one
starts.

```
Savio job watcher (launchd, every 5 min)          ~/.claude/savio-watch
   └─ each poll: queue + jobs that ended ──▶ harness/tick.py   (bookkeeping, no model)
                                                │ something to act on?
                                                ▼
                                         harness/run_agent.sh  (one at a time, ≤6/day)
                                                │  claude -p  with harness/AGENT.md
                                                ▼
         score_run.sh ─▶ analysis JSON ─▶ knowledge/findings/0NN ─▶ next experiment ─▶ sv train
                                                │
                                                ▼
                                    harness/reports/<time>.md  + a Mac notification
```

| file | what it is |
|---|---|
| `experiments.yaml` | **the search**: one entry per experiment, one change against a baseline, question and success criterion written before it runs |
| `AGENT.md` | the agent's standing orders and hard rules |
| `tick.py` | decides whether to start an agent: a run finished, failed, or an approved experiment has a free slot |
| `run_agent.sh` | starts one headless iteration; lock + daily cap |
| `score_run.sh` | export → copy → call type, detection, holdouts. `DRY=1` checks inputs only |
| `reports/` | one short report per iteration |
| `logs/` | full agent transcripts and scoring logs |

## Steering it

- **Approve** an experiment: set its `status: approved`. It starts when a slot frees (`max_concurrent`).
- **Stop** one: set `status: rejected` (and `sv cancel JOBID` if it is running).
- **Pause everything**: set `"enabled": false` under `harness` in `~/.claude/savio-watch/jobs.json`.
- **Run an iteration now**: `bash harness/run_agent.sh "why"`.
- `needs-human` in the queue or a report means the agent stopped on purpose; the report says why.

## What it will not do

It doesn't spend allocation hours, delete anything, touch other people's files, change the
evaluation protocol, or edit a past finding's numbers. Every number it writes comes from a scorer's
JSON. It approves its own proposals only when they're cheap and change one flag or path against an
existing script; anything bigger waits for you. The full rules are in `AGENT.md`.

It needs the Savio SSH certificate (`sv login`, 12 h). When that expires, the agent stops and says so.
