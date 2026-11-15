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
