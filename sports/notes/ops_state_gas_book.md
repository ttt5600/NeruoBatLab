# ops_state_gas_book -- read-only KXAAAGASCD* order-book logger (ops agent, 2026-11-15)

Brief (lead): order-book depth for the 50 state AAA gas-change series is the only non-retroactive data
(notes/state_gas.md). Add a daily read-only snapshot (~14:00 UTC + ~1 h before close) of all open books to the
deployed quant-sports launchd setup. No orders, no keys, no .env/kill changes; user approved read-only loggers.
No outcome is computed here, so no pre-registration beyond the GS1 pre-reg already in notes/state_gas.md.

## Plan
- scripts/state_gas_books.py snap --slot am|close: 50 GET /events (series KXAAAGASCD<ST>, status=open, nested markets)
  + batched GET /markets/orderbooks (25 tickers/call) -> ~60 GETs via kx_throttle. Full book per bucket stored.
- Output (deployed, append-only, one gzip member per snapshot): ~/.claude/quant-sports/data/kx_forward/state_gas_books.jsonl.gz
- qs_job.sh job `state-gas-books`, launchd every 900 s; slots in UTC.

## Done (2026-11-15)
- Files: scripts/state_gas_books.py, scripts/install_state_gas_books_agent.sh, qs_job.sh case `state-gas-books`,
  qs_sync.sh LOGS += kx_forward/state_gas_books.jsonl.gz, results/ops_state_gas_book/verify.py. Deployed + loaded.
- launchd kickstart 16:32:37Z: `states=50/50 markets=250 events=50 no_book=0 bytes+=12202 59s`, exit 0
  (logs/state-gas-books.log); state/state_gas_am = 2026-11-15. Manual slot=verify 16:34:10Z: 50/50, 250, 80 s.
- Storage: ~12 kB gz per snapshot -> ~9 MB/year at 2/day.
- Cross-check (verify.py): book-derived best YES bid / (1 - best NO bid) vs the /events market fields: 17/250 mismatched
  in v1 (books fetched up to ~50 s after the events pass; AK LD ask 0.21 had been lifted -> 0.54, re-fetch agreed),
  5/250 in v2 (book fetched right after each state's event). Treat yes_book/no_book as authoritative; the market fields
  are a slightly earlier quote.
- Stop: `launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-state-gas-books` (or scripts/qs_stop.sh pauses all jobs).
- Open risk: slot times are fixed; if the close moves, the `close` slot may miss (check `slot=close` rows exist on 11-16).

## Learnings
### Kalshi batch orderbook endpoint takes repeated tickers params, not a comma list
GET /markets/orderbooks?tickers=A&tickers=B returns both books; tickers=A,B returns one empty book keyed "A,B" (curl
test 2026-11-15). Silent empty books, not an error, so count book_ok / empty books.
### Fetch the order book in the same breath as the market quote
Quotes moved in 17/250 state-gas markets during a ~50 s gap between the /events pass and the book pass; per-state
fetching cut it to 5/250 (verify.py on state_gas_books.jsonl.gz). Any join of market fields to a book must allow for this.
