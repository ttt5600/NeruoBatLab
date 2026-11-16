# ops_series_watch -- daily read-only Kalshi new-series watcher (agent started 2026-11-15)

## Question
Which Kalshi series are new or newly active each day, and which of them look like the E4 pattern (data/physically
resolved, public timestamped settlement source, independent public forecast, ladder with likely-stale far brackets,
enough events per month)? Output = leads for the research queue, never a tested edge.

## Pre-registered rule (written before the first diff was computed)
Score 0-10 = S1 settlement source verifiable (0-2) + S2 independent public forecast exists (0-2, domain map) + S3
ladder (max markets/open event >= 6 -> 2, >= 3 -> 1) + S4 stale far brackets (share of ladder markets quoted at
yes_ask 2-15c with zero 24h volume >= 30% -> 2, >= 10% -> 1) + S5 cadence (>= 4 events/month -> 2, >= 1 -> 1),
-3 penalty for category Sports / Politics / Elections or mention/say markets with no data domain.
PROMISING = score >= 6 AND >= 2 open markets AND no penalty. Promising series get a ### heading in the digest and a line
in notes/RESEARCH_QUEUE.md under "From series watch" (deduped by ticker). No outcome or settlement data is looked at.
Known limits: S1/S2 are keyword maps on title + settlement sources + rules text (false positives likely, e.g. "gold"
or "rain" in an unrelated title); S4 is a one-snapshot staleness proxy, not a quote-age measurement.

## Design
- scripts/series_watch.py (run | seed | sync); GETs only via scripts/kx_throttle.kget (/series, /events?status=open&
  with_nested_markets=true, ~100 pages). Snapshot = compact per-series summary (gz JSON) in state/series_watch/.
- Diff vs the most recent earlier snapshot: new listed series, new series with open events, previously listed series
  that gained open events ("newly active").
- Deployed: ~/.claude/quant-sports (launchd cannot read/write ~/Desktop under TCC). Digest written to
  ~/.claude/quant-sports/digests/series_watch-<date>.md (vault DIGESTS glob indexes it) and
  ~/.claude/quant-sports/notes/series_watch/<date>.md; promising lines go to notes/series_watch/queue_pending.md there.
  A mirror into this worktree is attempted on every run; if TCC blocks it, run
  `../.venv_quant/bin/python scripts/series_watch.py sync` from a session (copies digests to notes/series_watch/,
  appends pending lines to RESEARCH_QUEUE.md).
- Seed "yesterday": the 2026-10-29 raw census (data/census/raw_2026-10-29: kalshi_series.json + 82 kalshi_events pages).
  So the FIRST diff spans 10-29 -> 11-15 (17 days), not one day; later diffs are daily.

## Log
- 11-15 08:04 seed: data/census/raw_2026-10-29 -> ~/.claude/quant-sports/state/series_watch/snap_2026-10-29.json.gz
  (15501 series, 5021 with open events).
- 11-15 08:05 first launchd run HUNG 15 min: crawl + snapshot done, then open() on the ~/Desktop worktree blocked on
  TCC (stack sample: _io_FileIO___init__ -> open). Killed (exit 143). Fix: qs_job.sh sets SW_NO_MIRROR=1; the deployed
  job never touches ~/Desktop; `series_watch.py sync` from a session mirrors + flushes the queue.
- 11-15 08:20 second launchd run exit 0, but the done-flag `state/series_watch` collided with the snapshot DIRECTORY of
  the same name (echo failed silently -> would re-run hourly). Renamed to state/series_watch_done.
- 11-15 first digest had keyword false positives: "ice" matched "voice" (The Voice scored 9/10), "bloomberg" in
  news-outlet source lists put CEO/college-admission markets in markets/crypto, "traffic" caught retail foot-traffic.
  Fixed: word-bounded regex, news outlets removed from domain map. Per-state families (50 x KXAAAGASCD<ST>) collapsed to
  one heading. Re-scored with `rerender` (no network).
- 11-15 08:22 third launchd run: exit 0, done flag written, GET /series 16448 series, 101 open-event pages, 4800 active
  (logs/series-watch.log). Final digest (rerender after the regex fix): 947 new series since 10-29 (630 already open),
  197 newly active, 150 promising -> 101 families; top 30 headed + queued in RESEARCH_QUEUE.md "From series watch".
  Top: metals/FX/crypto/treasury "after economic events" ladders (KX*EVENT, new), KXAAAGASCD* 50 state gas prices
  (AAA, new), ERCOT hub monthly power, Dutch TTF monthly, KXTEMPNYCHS hourly NYC temp, KXBIGGESTQUAKEMONTH,
  KXARCTICICEMAX. Remaining known false positive: KXPERFORMROLE. Span is 17 days (10-29 seed), daily from tomorrow.
- 11-16 BUG (agent ops_series_watch_fix, reported in notes/nyc_hourly_dir.md): KXTEMPNYCHS was queued as "newly active"
  although it has traded since 09-10 (1,319 events). Cause: "new / newly active" = "no open event in the previous
  snapshot", and the 10-29 census seed simply had no open KXTEMPNYCHS event; nothing checked history.
  FIX (scripts/series_watch.py, deployed to ~/.claude/quant-sports/scripts, plist unchanged): for every candidate,
  GET /markets?series_ticker=T&max_close_ts=<prev snapshot day 00:00Z>&limit=1 (any settled/closed/finalized market
  closing before the previous snapshot) and, if none, GET ...&max_created_ts=<same> (market already listed then, closes
  later -- KXCOINBASE created 08-04). Kinds: NEW (neither, not listed before) / newly active (listed before, neither) /
  reactivated (last pre-snapshot close > 3 days before it) / old (snapshot gap) (last close within 3 days) / old (open
  at prev snapshot) / unverified (GET failed). Only NEW, newly active, reactivated, unverified can be queued. Results
  cached in state/series_watch/hist_<date>_vs_<prev>.json (rerender needs no second crawl). Checked the API first:
  KXTEMPNYCHS max_close_ts<=10-29 -> KXTEMPNYCHS-26OCT2820 closed 10-29T00:00Z; KXAAAGASCDCA -> none (truly new);
  max_created_ts on KXCOINBASE -> created 2026-08-04 (results/ops_series_watch_fix/probe2.py, probe3.py).
  Also: TESTED / RELATED registry in the script (compiled from FUND_BOOK 3d, LEARNINGS_2026-11, EDGES_AND_AGENTS:
  KXHIGH*, KXLOWT*, KXRAIN*, KXTEMP*, KXTSAW, KXAAAGAS*, KX(TX)ERCOT*, KX*EVENT, KXHURCAT*, album/Billboard,
  YouTube/Netflix, crypto KXBTC/ETH/SOL D/E/Y/W + MAX/MIN, KXINX/NASDAQ100, Q016 WTI/BRENT/GOLD/NATGAS + KXTTFMON,
  KXBIGGESTQUAKE, PortWatch/EIA crude, phys-ladder census, KX30YMORTW, resort snow). TESTED = annotated, never queued;
  RELATED (sibling tested: KXBIGGESTQUAKEMONTH, KXSILVER/COPPER, *ICEMAX, *SNOW*, other KXBTC/ETH/SOL) = queued with
  the note. The deployed job cannot read worktree notes, so the registry is code: ADD A LINE WHEN A SERIES IS TESTED.
- 11-16 re-run of the 11-15 digest (rerender, 827 candidates, 1,544 read-only GETs; logs results/ops_series_watch_fix/
  rerender_1115b.log): NEW 630, newly active 69, reactivated 70, old (snapshot gap) 40, old (open at prev snapshot) 18,
  unverified 0. Promising 150 -> leads 55 (was 150 before: 95 were tested or not new). KXTEMPNYCHS = old (snapshot
  gap) + TESTED; KXCOINBASE = old (open since 08-04). RESEARCH_QUEUE "From series watch": 27 of the 30 11-15 lines
  removed (26 tested + KXCOINBASE), replaced by the top 30 untested leads (backup results/ops_series_watch_fix/
  RESEARCH_QUEUE.before.md). Digest synced to notes/series_watch/2026-11-15.md.
- 11-16 03:0x dry run of the NEXT digest with the deployed script into a scratch root (SW_ROOT=results/
  ops_series_watch_fix/dryroot, so the real 07:00 slot is untouched; log dryrun_1116.log): 16,454 series, 95 pages,
  71 candidates vs 11-15 -> NEW 5, reactivated 22, old (snapshot gap) 44; 11 promising, 0 leads. The old logic would
  have queued KX2YRRATEH/KX10YRRATEH/KXGOLDH/KXWTIH/KXTTFD... as "newly active" (hourly markets closed over the weekend
  at the Sunday snapshot; last close 11-13). The real 07:00 PT 11-16 launchd run had not happened when this agent ended.
- Remaining known scoring weaknesses (not fixed here): keyword false positives KXTHEVOICE / KXPERFORMROLE /
  KXROLEINPRODUCTIONLI / KXOSCARS still score 6-8; per-city families with different ticker lengths (KXHOUHOME,
  KXSEAHOME, ...) are not collapsed by the "ticker minus 2 chars" rule.

## Ops
- launchd com.jonathanwang.quant-sports-series-watch: 07:00 PT + hourly retry, once/day (state/series_watch_done).
  Installer scripts/install_series_watch_agent.sh. Logs ~/.claude/quant-sports/logs/series-watch.log.
- Digest: ~/.claude/quant-sports/digests/series_watch-<date>.md (vault-indexed) + notes/series_watch/<date>.md there.
  Worktree copy + RESEARCH_QUEUE append: `cd sports && ../.venv_quant/bin/python scripts/series_watch.py sync`.
- Vault: sources.py globs `notes/*.md` (not subdirs), so worktree notes/series_watch/<date>.md is NOT indexed; the
  digests copy is. Proposal for the vault session: add ("quant", QUANT, "notes/series_watch/*.md", "note").

## Learnings
### A launchd job that opens a file under ~/Desktop hangs instead of failing
Under TCC the open() blocks (15 min stuck, stack in _io_FileIO___init__ -> open; ~/.claude/quant-sports/logs/
series-watch.log exit 143 after kill). Deployed jobs must never touch the worktree; sync from a session instead.
### Keyword domain maps need word boundaries and must ignore news-outlet source lists
"ice" matched "voice" and "bloomberg" (a settlement source on CEO/college markets) scored them 8-10/10 as markets
data. Notes: notes/ops_series_watch.md log 11-15.
### Kalshi listed ~947 new series in 17 days (10-29 -> 11-15), 630 already trading
Source: digests/series_watch-2026-11-15.md. Manual census cannot keep up; the daily diff is the replacement. New
untouched data-resolved ladders: KX*EVENT (metals/FX/crypto/yields after econ releases), KXAAAGASCD* (50 states).
### "Not open in the previous snapshot" is not "new": check the series' market history
KXTEMPNYCHS (traded since 09-10, 1,319 events) was queued as newly active because the 10-29 seed had no open event for
it. With a history check (GET /markets max_close_ts / max_created_ts <= prev snapshot), the 11-15 diff's 150 promising
series shrink to 55 untested leads; the 11-16 dry run shows hourly Treasury/gold/WTI series reappearing every Monday
as "new" from weekend gaps. Source: notes/ops_series_watch.md log 11-16, results/ops_series_watch_fix/dryrun_1116.log.
### A lead queue needs the do-not-redo list wired into the generator, not checked by hand
26 of the first 30 queued series (KX*EVENT, KXAAAGASCD*, KXERCOT*M, Q016 commodities, KXTEMPNYCHS) were already tested
and dead. The watcher now carries a TESTED registry (scripts/series_watch.py) and never queues those; update it when a
series is tested.
