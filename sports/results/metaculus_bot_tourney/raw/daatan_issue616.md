Metaculus FutureEval bot (Daatan-v1 / Daatan-v2): constraints, decisions, entry plan (UMBRELLA)
2026-08-23T16:21:41Z
Umbrella for entering Metaculus's FutureEval bot tournament with two Daatan bots. All facts below verified 2026-08-23 against the Metaculus API and the FutureEval resources notebook (post 38928), not search summaries.

## Current state

- Two bot accounts already created under Mark's Metaculus account (Settings → "My Forecasting Bots"): **`Daatan-v1`** and **`Daatan-v2`**.
- API tokens stored in AWS Secrets Manager (`eu-central-1`): `daatan-bots/metaculus-v1-token`, `daatan-bots/metaculus-v2-token`. Both verified working against the API.
- ~~Nothing built yet. No service, no repo directory.~~ **Stale — corrected 2026-08-29.** `retro/metaculus/` (client, Oracle relay, sync loop, 18 tests, workflow) was written 2026-08-22 but existed **only as untracked files** in a shared checkout until it was committed in retro PR#723. It polls open binary questions, asks Oracle for a probability + rationale, and submits the forecast with the mandatory private comment; defaults to the unscored `bot-testing-area` with no `schedule:` trigger.

## Decisions taken

- Service lives in **`retro`** (Python/uv), alongside the Oracle pipeline — not in `daatan`, whose `docs/bots.md` bot system is an unrelated concept (personas staking on Daatan's own markets).
- **`Daatan-v1` = Oracle-powered**, and is the prize-eligible slot. **`Daatan-v2` = baseline/challenger**, secondary.
- v1 calls Oracle over its API/MCP rather than importing pipeline internals.

## Tournament rules that constrain the design

Source: <https://www.metaculus.com/notebooks/38928/futureeval-resources-page/>

- **Scoring is spot peer score**: *"Since it's a spot score, only the last forecast made before the question closes counts."* Log-based, scored relative to how hard the question was for others — so a "good" score is partly a function of field difficulty, not raw accuracy.
- **One prize-winning bot per participant/team.** Secondary bots are explicitly permitted and *"must be labelled as such"*, by *"add 'v2' (or whatever version it is) to the bot's username and email."* `Daatan-v2` already conforms. There is no sybil/collusion risk here — this is the sanctioned mechanism.
- **Comments are mandatory** on every question forecast, and *"should only leave private comments"*; Metaculus makes *"all comments public after a question closes."* v2 needs a rationale generator too, not just a number.
- **No human in the loop.** No previewing the bot's forecast on open/upcoming tournament questions and then tuning; no re-running because you dislike the output; no copy/paste submission. Test only on closed tournament questions or the main site.
- **Prize winners must submit code or an architecture description and accept a Metaculus inspection.** Relevant given v1 wraps proprietary Oracle internals.
- Using *"publicly available forecasts on questions found on other platforms or on Metaculus itself"* is **allowed** — so a market/community-median baseline for v2 is legal. Creating a question elsewhere to feed the bot is not.

## The constraint that drives the architecture

**Tournament questions are open for 1.5 hours** (temporarily raised to 3h while GitHub Actions reliability is degraded — see the pinned admin comments on post 38928). Verified empirically: post 45182 has `open_time` 2026-08-19T15:00Z and `actual_close_time` 2026-08-19T18:00Z, with `scheduled_resolve_time` three weeks later.

Consequences:

- There is **no multi-day evidence-driven update loop**. An earlier design in this workstream assumed "push a new forecast whenever Oracle's estimate moves ≥5pts" — that is not applicable to this tournament.
- The binding constraint is **latency**: detect a newly opened question and produce a forecast + rationale inside a ~1.5–3h window. See the Oracle latency issue linked below.
- Discovery is by **polling** — no webhook is documented, and the bot templates/community assume roughly 20-minute poll cadence. (Note: absence of a documented webhook is not proof one does not exist; not exhaustively verified.)
- Questions launch *"at random hours during the week"*, up to 5 at a time, so polling must run continuously, not on a business-hours schedule.

## Season timing — act on this

The Summer 2026 season is **effectively over**: the announcement states the tournament *"runs continuously until questions stop opening a few weeks before September 1, 2026."* Observed launch counts in `summer-futureeval-2026`: 7 on Aug 13, tapering to 1 on Aug 20, and **zero since**. Entering now would yield almost no scored questions.

Target instead:
- The **Fall season** (primary, $50k pool, 300–500 questions, 3 seasons/year aligned to the Metaculus Cup).
- The **bi-weekly $1k MiniBench** (~60 questions per two-week round) as a faster feedback loop and a legitimate practice ground.

## Cost note

Metaculus sponsors bot-maker costs: free LLM credits (Anthropic/Google/OpenAI, distributed via tournament-provided OpenRouter keys) and **AskNews** search access (1k calls/month, 4k total per tournament, `/news` and `/deepnews`). Both require per-season application/renewal. This could materially offset Oracle inference cost for the bot — worth applying before Fall. Note the admin comments say the Summer LLM credit budget is largely exhausted; apply early for Fall.

## Open sub-issues

- #617 — Oracle `/forecast` latency vs. the 1.5h window (blocks the deployment shape)
- #618 — `insufficient_data` gate too permissive
- #619 — calibration backtest over Metaculus resolved questions (gates entry)
- #620 — Polymarket dry-run paper-trading bot

## Related

- #615 — Oracle high-bias vs Metaculus community (the finding that started this)
- #403 — existing backtest harness over the replayed claim layer (different corpus; blocked until ~Dec 2026)


