# Tournament protocol — the rules, and how the bot meets each

Sources: the FutureEval Bot Tournament Resources Page [1] ("Competition Rules" and "What Do
the Tournaments Look Like?") and the Fall 2026 announcement [2], both read through the
Metaculus API on 2026-10-17, plus the MiniBench project record [3]. Re-read [1] at the start of
each season: Metaculus changes details without much notice, and the comments section of [1]
is where announcements land.

## The tournaments

| | MiniBench | Fall 2026 FutureEval |
|---|---|---|
| Slug / id | `minibench` (always the current round; 33132 for the 2026-10-19 round) | `fall-futureeval-2026` (33121) |
| Cadence | 2-week rounds starting Monday every other week; most questions open in the first days, the second week is usually empty [1][2] | continuous until a few weeks before 2027-01-01; 300–400 questions [2] |
| Next round | starts **2026-10-19 00:00 UTC**, forecasting ends 2026-10-24 17:20 UTC, 60 questions [3] | running since 2026-09-28 |
| Question window | **3 hours**, random launch times; usually ≤ 5 open at once, rarely ~10 [1] | same |
| Question source | created and resolved by an LLM from varied sources (since 2026-06-01) [1] | custom + main-site questions |
| Types | binary, numeric, discrete, multiple choice (no date questions) [2] | same |
| Scoring | spot peer score: only the last forecast before close counts [1] | same |
| Prize | $1k per round, paid with the seasonal prizes | $50k |

The 3-hour window replaced the 1.5 hours the September plan assumed. It is what lets the sweep
run every 30 minutes instead of every 20.

## The rules → the mechanism

| Rule [1] | How this repository complies |
|---|---|
| **No human in the loop.** "A bot maker cannot preview how their bot forecasts on open or upcoming questions and then update their bot based on that preview"; "cannot decide that they don't like their bot's forecast and rerun the bot"; no copy-pasting forecasts — "the bot must directly forecast and comment via API". | The sweep runs unattended; nothing waits for approval (the upstream workbench's merge-to-submit workflow was removed for this reason). The maintainer should read forecasts on open tournament questions only after close, and must not change prompts or code in response to them. |
| **One forecast per question** in the bot-only tournaments. | Four guards before any POST: `meta.tournament.submitted`; a `submission` record in the ledger (and a test that no slug has two); the bot's live forecast on Metaculus (checked by the gate and again right before the POST); `log_forecast.py` refuses a second forecast on a one-shot question. Sweeps are serialised so two jobs never hold the same question. |
| **Comments required** on every forecast, private (`is_private=true`); they "should accurately represent the reasoning process and final decision". Required for prizes. | `submit_metaculus.py` posts the newest `forecast.md` version — digest, outside view, evidence, reasoning, the final numbers, sources — as a private comment, extracted mechanically. A failed comment is recorded (`comment_posted: false`) and retried by the daily sync. |
| **Bot account**; one prize-eligible (primary) bot per participant; secondary bots allowed but excluded. | The token in `METACULUS_API_KEY` belongs to the bot account `my-claude-forecasting` (`is_bot: true`, verified 2026-10-17). Whether it is the account's *primary* bot is visible in Metaculus settings — confirm it. |
| **Code or description + inspection** to win a prize; a **survey** each season; the participation form before submitting. | This repository is the description. Survey and form are maintainer tasks (see the summary in `bot-migration-progress.md`). |
| A bot may use public information, including other platforms' and Metaculus' public forecasts; it must not create markets or questions to help itself. | The forecaster uses web search and prediction markets; it creates nothing. |
| Test on `bot-testing-area` or closed questions, not on open tournament questions. | `bot-testing-area` is registered (inactive); the sweep's manual run takes `tournament: bot-testing-area`. |
| Autonomous bots: a human creates the human account, answers prize emails, and handles Cloudflare blocks without asking Metaculus for workarounds; do not let the AI email Metaculus. | The bot only calls the official API. It never contacts Metaculus staff. |

## Data the bot token can see [1]

Open question text, its own forecasts and comments, staff comments, and resolutions of
questions it forecast. **Not** the community prediction or other aggregates (except on ~50
questions). The bot therefore does not look for a CP; the "Other forecasters" table records
"hidden — bot tournament" for Metaculus and uses prediction markets where they exist.

## What happens when

| Event | Outcome |
|---|---|
| Question opens | Seen by the next gate (≤ 30 min + scheduler delay). Skipped if < 20 min remain. |
| Forecast session overruns its cap | The step is killed; whatever was logged is still submitted; nothing logged → no submission, and the next sweep retries while the window is open. |
| Window closes with nothing submitted | `missed_window: true`, `status: dropped` (by the submit step or the daily sync). Peer score 0. |
| Forecast accepted, comment fails | Recorded; the daily sync retries the comment. |
| Question resolves | Daily sync: ledger resolution, official spot peer score, a `## Resolution` note. Annulled → dropped. |

## Sources

1. FutureEval Bot Tournament Resources Page, https://www.metaculus.com/notebooks/38928/futureeval-resources-page/ (accessed 2026-10-17, via `GET /api/posts/38928/`)
2. Announcing Fall 2026 FutureEval Bot Tournament, https://www.metaculus.com/notebooks/45615/ (accessed 2026-10-17, via `GET /api/posts/45615/`)
3. MiniBench project record, `GET https://www.metaculus.com/api/projects/tournaments/minibench/` (accessed 2026-10-17)
4. Metaculus bot template (cron `7,27,47 * * * *`, private comments), https://github.com/Metaculus/metac-bot-template (accessed 2026-10-17)
