# ops_gap_logger -- fix gap_logger's Kalshi<->Pinnacle team matcher, log raw Pinnacle odds + age (2026-11-15)

Ops task from the lead (source: notes/kalshi_soccer_sharp.md "gap_logger fixes (proposal)"). No orders, no git,
no deploy. Worktree code + tests only; deploy is a proposal at the bottom.

## Questions / acceptance rules (written before looking at outcomes)
1. Which Kalshi<->Pinnacle soccer events did the old matcher (`_first()` = first token after stop-words, then
   SUBSTRING test) mis-side or miss? Acceptance for the fix: 0 mis-sided events on every collision found, 0 event
   disagreements with the old matcher where both match, coverage >= old.
2. Log raw Pinnacle odds + quote age every tick so 3-way Shin and Pinnacle-close CLV are computable.
3. Silence mlb/wnba off-season KeyErrors.
4. Does the same matcher pattern live in other loggers?

## How the audit was done
- `results/ops_gap_logger/names.json`: every Kalshi soccer market (all statuses, close >= 2025-10-09) for the 9
  gap_logger series (927 events; all 81 soccer events in the deployed gap_log.jsonl are covered) + current Pinnacle
  participants per league. Per-event Kalshi GETs hit HTTP 429, so it is one paginated GET per series.
- `results/ops_gap_logger/audit_matcher.py`: replays the OLD rule (verbatim copy of `_first`) on every event, with
  the Pinnacle name of each Kalshi team found by the new matcher. Orientation: Kalshi soccer event tickers list the
  HOME code first -- checked on live data, 138/138 matched games (command in this file's log, 2026-11-15).
  Output: `results/ops_gap_logger/missided_events.csv`, `audit.json` (37 cross-team old-rule pairs).

## Results
### Collisions (old rule, home-first orientation) -- missided_events.csv
| event | pin home vs away | what went wrong | gap_log rows |
|---|---|---|---|
| KXEFLCHAMPIONSHIPGAME-26OCT31WHUWBA | West Ham v West Brom | "west" in "west": WBA got West Ham's fair | 3172 (3169 WBA yes, max EV +252%) |
| KXLIGUE1GAME-26NOV28MANLIL | Le Mans v Lille | "le" is a substring of "lille": Lille got Le Mans' fair | 774 (LIL no, EV +28..+47%, LIVE until fix; started 11-14 so NOT in the 11-13 KS12 census) |
| KXMLSGAME-26OCT28NENYRB | New England v NY Red Bulls | "new" in "new": NYRB labelled H | 0 (wrong-sided quotes in the `seen` denominator only) |
| KXMLSGAME-26OCT25LAFCLAG | LAFC v LA Galaxy | "los" in "los": Galaxy labelled H | 0 |
| pre-logger (before 10-19, no rows) | MUN-MCI 09-13, RMA-VIL 10-10 ("real" in "villarreal"), MAN-RCL 09-13 ("le" in "lens"), NYC-NYRB, SD-SJ, NYC-NE | | 0 |
Latent pairs that would mis-side the next time they meet (audit.json): Le Havre/Le Mans vs Lens, Lille, Marseille;
Real Madrid/Betis/Sociedad vs Villarreal; St Louis vs Houston/Austin ("st" in "houston"); RB Leipzig vs Paderborn and
Fenerbahce ("rb" in "paderborn"); Real Salt Lake vs Montreal; San Jose vs San Diego; Manchester derby.

### Silent misses (coverage)
On live data (143 Pinnacle soccer games, 2026-11-15): old matcher matched 96, new 138, 0 event disagreements where
both matched. Old misses were systematic: Athletic Bilbao, Rayo Vallecano, Racing Santander, Internazionale, AC Milan,
RB Leipzig, FC Koln, Bayer Leverkusen, Borussia Dortmund, PSG, ... (first token "athletic"/"ac"/"rb"/"bayer" not in the
Kalshi short name). So KS12 history EXCLUDES those clubs -- a selection effect, not just noise. Dry-run tick: soccer
quotes seen laliga 39->57, bund 18->54, ered 9->27, ligue1 36->54 (results/ops_gap_logger/dry/gap_ticks.jsonl vs
deployed gap_ticks.jsonl tail).

### Pinnacle guest API is a 2-15 minute old cache
Every response is Cloudflare `HIT`, `Cache-Control: max-age=905`. Dry-run tick ages (Date - last-modified) per league:
nfl 710 s, nhl 878 s, ered 596 s, efl 520 s, epl 494 s, nba 122 s, ucl 189 s (results/ops_gap_logger/dry/gap_pin_odds.jsonl
`fetch` rows). Gap "episodes" shorter than ~15 min can be Pinnacle staleness, not Kalshi; age_s is now logged so the
report can filter on it.

### US matcher (live.match_kalshi, startswith on _norm)
NFL/NBA/NHL: 0 Kalshi-name prefix pairs, live matches all consistent. CFB: 49 Kalshi-name prefix pairs ("georgia" /
"georgia tech", "florida" / "florida state", "penn" / "penn state" ...). Within a matched game the {home, away} set
check prevents mis-siding (a rivalry where the shorter name is AWAY is dropped, not mis-sided); a wrong-game match needs
both names to prefix-collide on the same date with the true game absent. Not fixed (live.py is shared with live bots).

### Other loggers (task 4)
- kalshi_soccer_sharp.py `shin` used `gl._first` -> switched to `gl.match_soccer_event` (would otherwise break).
- live_lag_logger.py: reuses live.match_kalshi + recomputes side with the same startswith -> inherits the CFB caveat,
  no first-token bug.
- xvenue_ingame.py `_side_ok`: nickname->code tables for nba/nfl/mlb (safe); other leagues fall back to
  `len(pn) >= 5 and pn in sub` -- a raw SUBSTRING test, same bug class (e.g. "miami" in "miami oh"). Flag, not fixed.
- pmus_vs_pin.py `team_hit`: `p.endswith(k)` on normalized names -- suffix match, safe for pro nicknames; CFB-style
  "state" suffixes would collide. pmus_lag / pmus_lag_logger: exact code or exact normalized name (safe).
- No other copy of `_first`.

## Fixes (scripts/gap_logger.py)
- `team_tokens` (accent strip incl. M´gladbach, punctuation, saint->st, ALIAS table, STOP words, digits dropped) +
  `same_team` (token-set equal or subset; never token substrings) + `match_soccer_event`: date (UTC/ET/UK) + exactly 2
  team markets + TIE + home matches exactly one Kalshi team and away the other + exactly one event.
- Raw odds: `data/gap_pin_odds.jsonl` -- a `fetch` row per league per tick (age_s, cf_age, cf, last_modified, open
  pin_ids) and an `odds` row per game whenever American prices / market version / Kalshi mapping change (lossless,
  ~63 KB first tick then a few KB/tick); state in `data/gap_pin_last.json`. US 2-way prices logged too.
- `us()`: empty Pinnacle/Kalshi -> empty frame (mlb/wnba now `0`, were `err KeyError` on 10194 ticks / `err HTTPError`
  on 8620 of 13872 ticks); HTTP errors now record the status code (`err HTTPError 429`).
- Tests: tests/test_gap_logger.py (58 cases incl. West Ham/West Brom both orientations, Le Mans/Lille, derbies,
  wrong date, ambiguity, missing TIE, odds dedupe/age, off-season). Full suite: 487 passed, 2 failed -- both in
  tests/test_bet.py (order body price format), pre-existing and unrelated to these files.

## Proposal (lead decides; NOT done)
Deploy: `cp sports/scripts/gap_logger.py ~/.claude/quant-kalshi/scripts/gap_logger.py` (deployed copy == old worktree
copy, diff-checked; live.py unchanged so nothing else needs copying). Optionally copy kalshi_soccer_sharp.py if a
deployed copy exists. Historical analysis: drop the WHUWBA and MANLIL events (missided_events.csv) from any gap_log
study; note KS12 history under-covers the clubs listed above.

## Learnings
### First-token substring matchers mis-side real fixtures
"West Ham"/"West Bromwich" -> "west" and "le" (Le Mans) inside "lille" gave fake +252% and +47% EV rows (3172 and 774
gap_log rows). Match on whole-token sets + alias table and require a one-to-one home/away assignment per event.
Source: results/ops_gap_logger/missided_events.csv.
### A matcher that misses silently biases the sample
The old soccer matcher matched 96/143 live games vs 138/143 for the strict one; misses were whole clubs (Inter, Milan,
Bilbao, Leipzig, Leverkusen, Dortmund, PSG), so KS12 history is a club-selected subset.
### Pinnacle guest API quotes are 2-15 minutes old
Cloudflare HIT, max-age 905 s; observed age 122-878 s per league in one tick. Short "gaps" vs Pinnacle can be cache
staleness. Source: results/ops_gap_logger/dry/gap_pin_odds.jsonl.
### Kalshi soccer event tickers put the home team first
138/138 live matched games (2026-11-15). Lets history be oriented without stored Pinnacle names.
