# ops_econ_event_logger -- read-only KX*EVENT order-book logger around releases (2026-11-15)

Agent: ops_econ_event_logger (ops subagent). Proposal 6 of notes/econ_event_ladders.md. No orders, no git, no .env.
User approved read-only loggers under launchd.

## What it does
- Script: `scripts/econ_event_books.py` (deployed copy `~/.claude/quant-sports/scripts/`), job `econ-events` in
  `scripts/qs_job.sh`, installer `scripts/install_econ_event_books_agent.sh`.
- launchd `com.jonathanwang.quant-sports-econ-events`, every 60 s, 24 h. Idle ticks: no network, only
  `state/econ_events_heartbeat`. Schedule (GET /events per series, status=open, 16 series) re-pulled at most every 6 h.
- Per open KX*EVENT event, three slots: pre = release - 5 min, p5 = release + 5, p30 = release + 30; a slot is taken by
  the first tick in [slot, slot + 10 min) and only if slot < obs (strike_date). Rates (obs = release + 15) get pre/p5 only.
- Per slot per event: GET /markets?event_ticker= (top of book, sizes, volume, OI) + GET /markets/orderbooks batches of 25
  (full depth; rows carry yes_top3/no_top3 best-first + full book). All via kx_throttle.
- At p5, if a KXBTCEVENT/KXETHEVENT event is in the slot: Deribit BTC/ETH option trades in [t - 60 min, t) (deribit_trades
  imported from the deployed scripts/ot1_forward.py) + Deribit index price, fetched BEFORE the Kalshi quotes (EV1 rule:
  inputs strictly older than the quote).
- Output, append-only gzip members: `~/.claude/quant-sports/data/kx_forward/econ_events/{books,deribit,schedule}.jsonl.gz`;
  manual tests go to `*_manual.jsonl.gz`. Done/missed state: `state/econ_events_done.json` (a slot whose window passed
  while the Mac slept is marked "missed" and logged).

## Release-time parsing (obs = strike_date; logged with every row so it can be recomputed)
explicit 'N hour(s)/min after' in title/sub_title -> that; non-rates 'auction' -> 30 min (1:00pm print); RATEEVENT -> 15
min; Fed/FOMC 14:00 ET, ECB 08:15, BoE 07:00, BoC 09:45, Banxico 14:00 (fixed clock on obs date); US data keywords -> 60
min; else 'default' 60 min. Schedule on 2026-11-15: 228 open events; 'default' = BOJ (4), RBA decision/minutes (4),
NVIDIA earnings (3). Known approximations: BOJ has no fixed time (default is wrong -- the decision is usually the ET
evening before); 'Banxico minutes' maps to 14:00 but minutes print ~10:00 ET. Analysts should filter on release_src.

## Verification (2026-11-15, Sunday 11:45 ET)
- Dry run (manual, from the deployed copy): KXBTCEVENT-17NOV260930 + Deribit -> 17 markets, 0 two-sided, BTC 272 option
  trades in the prior hour, index 64,726.05; KXUSDJPYEVENT-15NOV261950 -> 21 markets, 15 "two-sided" (2c/99c placeholder,
  e.g. T15900 yes bid .02 / ask .99). Files: data/kx_forward/econ_events/{books,deribit}_manual.jsonl.gz.
- Tick path simulated (fake clock, output to /tmp, deleted): pre/p5/p30 for the USDJPY event each fired once, idle before
  and after.
- launchd installed; first real tick 2026-11-15 08:46:24 PST rc=0 (heartbeat). First REAL scheduled snapshot due tonight:
  KXUSDJPYEVENT-15NOV261950 (Japan GDP) pre 18:45 ET, p5 18:55, p30 19:20. Next big cluster: 11-16 08:25 ET (USDCAD), 11-17
  08:25 ET (US retail sales: metals, FX, BTC/ETH/SOL, rates).
- Load: largest cluster (~31 events sharing an 08:30 release) ~100 GETs per slot (~30 s at the shared 3.3 req/s slot).

## Stop
`launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-econ-events` (or `touch ~/.claude/quant-sports/STOP` for all jobs).

## Learnings
### A KX*EVENT event's settlement time is its strike_date; the release time must be parsed from the title
Event strike_date = observation; sub_titles say '1 hour after X', '90 min after', '30 minutes after', or only name the
release (crypto 'US CPI', rates '8:45am ET' = 15 min after 8:30). 3 of 228 open events (BOJ) have no recoverable release
time. Source: scripts/econ_event_books.py release_of(), `schedule` output 2026-11-15.

### Launchd StartInterval jobs do not fire at bootstrap
The first tick came ~1 min after `launchctl bootstrap` (no RunAtLoad); an immediate heartbeat check reads "no file" and
is not a failure.
