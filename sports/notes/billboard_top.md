# billboard_top -- Kalshi Billboard #1 markets (KXTOPALBUM / KXTOPSONG) vs public midweek projections (agent, 2026-11-15)

Question: the album-units ladder (KXALBUMEQUIV) already prices HitsDailyDouble's (HDD) midweek projection
(notes/consensus_reject_census.md, AU-E-T dead: 300/319 rungs > 0.99 at Wed). Are the winner-take-all Billboard #1
markets different -- is the #1 race decided by HDD's numbers while Kalshi still prices rivals (or a close race priced
as a lock)?

Do-not-redo: AU-* trials (consensus_reject_census.md); FUND_BOOK 3d list.

## Data (inputs only so far)
- Kalshi markets: results/consensus_reject_census/mk_KXTOPALBUM.json (1,523 mkts, 103 events, weekly since 2024-12),
  mk_KXTOPSONG.json (weekly since 2023-06). One market per album/song; YES iff it is #1 on the chart "Week of <date>".
  Chart date = Saturday; tracking week = Fri..Thu ending chart date - 9 days (wend); Billboard posts the top 10 on the
  Sunday after wend; Kalshi close = that Sunday 23:59 ET.
- HDD Sanity CMS (results/consensus_reject_census/hdd_q.py; full dump data/billboard_top/hdd_chart_docs_all.json, 273
  chart docs): midweek-20 91 docs 2025-01-28..2026-11-10; hits-top-50 97 docs 2025-01-17..2026-11-14. Song charts:
  song-revenue 26 docs (2025-01..2025-10) and song-streams 8 (2025-02..06), both FINAL post-week numbers (created the
  Thursday AFTER the week, i.e. after Billboard's Hot 100 reveal) -> not a midweek forecast.
- Songs: no public PIT midweek source reachable. Wayback (kworb vintages) SSL-blocked from this machine
  (python requests SSLEOFError, 11-15); Spotify charts API needs auth; kworb has no vintages; Billboard "chart race"
  posts not scraped/timestamp-auditable. => KXTOPSONG NOT TESTED (blocked at the data gate, no trial registered).

## PRE-REGISTRATION (written 2026-11-15 BEFORE any HDD number was joined to a Kalshi price or result)
Dataset kalshi_top_album, family kalshi_data_ladders. Unit = one Kalshi market (album) per chart week.
Split by wend (tracking-week end Thursday): dev 2025-01-01..2026-08-31; sealed 2026-09-01..2026-11-08 (NOT opened);
forward 2026-11-09+. (Dev extends earlier than AU because KXTOPALBUM and HDD both exist from 2025-01.)
Doc usable at T only if `_updatedAt` < T. Doc->week mapping as album_units.py (midweek: next Thursday on/after
created date (ET); top-50: Thursday before created date). Match: normalized title (album_units.norm + accent strip)
Kalshi yes_sub_title == HDD title. Absent album: only if the doc has its full 20/50 rows and the normalized title
appears nowhere on the doc; bound U = smallest units on the doc. Leader L = largest units row on the doc, F2 = second.
Selection with margin M:
  YES on L's market if F_L >= M * F2;
  NO on any other market whose album is present with F <= F_L / M, or absent with F_L >= M * U.
  Market must be open at T (open_time < T < close_time).
Trials (k = 3):
- TA-E-T taker: T = Wednesday 12:00Z of the tracking week (wend - 1 d), doc = midweek-20, M = 1.30. Buy side at the
  ask from the last hourly candle ending <= T, + quadratic taker fee; ask <= 0.98.
- TA-L-T taker: T = Saturday 12:00Z after wend, doc = hits-top-50, M = 1.15. Same pricing.
- TA-E-M maker: TA-E-T selection, post q = floor_cent(min(mid, ask - 1c)) on the side, 0.50 <= q <= 0.98, filled iff a
  later tape trade prints THROUGH q before Kalshi close; fee 0. Report filled-vs-unfilled loss rates.
Success (dev): ROI > 0 AND week-clustered bootstrap 95% LCB > 0 AND p < family bar AND n >= 30.
Kill: otherwise; or the data gate fails.
Data gate (dev weeks only, reads the winner only here): HDD doc #1 == Kalshi winning album on >= 70% of usable weeks
(midweek-20) and >= 85% (hits-top-50). Fail -> that stage's trials stop unscored.
CLV: our side's ask at the last candle <= Sunday 12:00Z (= edgelab event_start, before Billboard's reveal) vs entry.

## AMENDMENT 1 (2026-11-15, after the data gate, BEFORE any Kalshi price was joined)
The first gate run exposed title-matching bugs that would have put NO bets on the true #1 (classed "absent"):
HDD "MAN'S BEST FRIEND" -> "MAN S" vs Kalshi "Mans"; HDD "$OME $EXY $ONG$ 4 U" -> "OME EXY ONGS"; HDD
"KPOP DEMON HUNTERS | SOUNDTRACK"; Playboi Carti HDD "MUSIC" vs Kalshi "I AM MUSIC". Fixes (inputs only):
(a) norm drops apostrophes and maps $ -> S; (b) match = title, or "artist title", or HDD title SOUNDTRACK and Kalshi
title starts with the artist; (c) never "absent" if any HDD title/artist is a >=4-char substring of the Kalshi title
or vice versa (-> ambiguous); (d) skip a whole week if the HDD leader matches no Kalshi market open at T.
event_start for edgelab = Sunday 12:00Z after the week (late T is Saturday, after sales close; the information
race is until Billboard's Sunday reveal). Gate re-run is reported below with both versions.

## Log
- 11-15: started; HDD chart-type census (above). Songs blocked.
- 11-15 build (`scripts/billboard_top.py build` -> data/billboard_top/ta_frame.parquet): dev weeks with a usable doc
  (_updatedAt < T): early 72, late 66 (96/240 dev market-rows dropped as edited_after_T).
- 11-15 DATA GATE (`billboard_top.py gate`, dev only): v0 matching early 0.912 / late 0.887; after AMENDMENT 1
  early **0.941** of 68 weeks (need 0.70), late **0.952** of 62 (need 0.85) -> PASS. Every early miss is a week whose
  HDD leader is not listed on Kalshi (MUSIC/I AM MUSIC alias, Drake TAKE CARE catalog, DEBI TIRAR not listed that week);
  in every week the leader WAS listed, it won. Late real misses: 25JAN25 (WHAM led HDD, Bad Bunny won), 26AUG08.
- 11-15 edgelab registered: ta_e_taker-8fb53d43, ta_l_taker-a3bdf85f, ta_e_maker-0a4d846e (family
  kalshi_data_ladders -> 44 trials, bar p < 1.14e-03). Code: edgelab/rules/billboard_top.py, loader kalshi_top_album,
  vault dataset kalshi_top_album, specs edgelab/specs/ta_*.yaml.
- 11-15 candles 1,580 keys (1,482 with candles), tape 802 tickers / 3,271 trades (data/billboard_top/ta_candles.json,
  ta_trades.json). `billboard_top.py score dev` + `python -m edgelab evaluate ta_* --split dev`:
  * selected dev rows: early 802 (44 YES-leader, 758 NO-rival; 59 weeks) with **0 losses**; late 778 with 2 losses.
  * Book at Wed 12:00Z: HDD-leader YES ask median 1.00 (no offer) / bid median 0.98; rival NO ask 1.00 on 702/754
    quoted rows (no YES bid at all). Leader margin F_L/F2 median 1.98 (results/billboard_top/ta_scored_dev.parquet).
  * **TA-E-T DEAD (n < min)**: n 7 / 5 weeks, ROI +2.88% [+1.90,+4.13], p 2.5e-05, CLV +2.7c, 0 losses; n 7 << 30.
  * **TA-L-T DEAD**: n 7 / 4 weeks, ROI -10.2% [-52.8,+5.2] (the 2 late misses were the only cheap YES), p 0.69.
  * **TA-E-M DEAD (n < min, flagged too-good->audit)**: 279 posts (q 0.95-0.98), 22 filled / 16 weeks, ROI +3.00%
    [+2.47,+3.72], p 2.5e-05; loss rate filled 0.000 vs unfilled 0.000 (no adverse-selection signal measurable: nothing
    lost). Capacity bound: 45,880 contracts printed through q on the 22 fills over 20 dev months -> <= $1,695 total
    even with 100% queue priority (results/billboard_top/ta_TA-E-M_dev.parquet).
  * Why not tradeable: the #1 race is decided by Tuesday's HDD midweek and the Kalshi book already shows it: leader bid
    0.98, rivals bid 0. The only left-over is 1-3c on a few thousand contracts -- same verdict as AU (KXALBUMEQUIV).
- KXTOPSONG: not tested (no PIT midweek source; see Data). Sealed (weeks 2026-09-01..11-08) NOT opened.

## Learnings
### HDD midweek leader wins Billboard 200 #1 whenever Kalshi lists it
Dev 2025-01..2026-08: HDD midweek-20 #1 == Billboard 200 #1 in 0.941 of 68 weeks; every miss was a week where the
HDD leader had no Kalshi market (alias/catalog). Friday top-50 #1: 0.952 of 62. Source notes/billboard_top.md gate.
### Kalshi KXTOPALBUM prices the HDD projection fully by Wednesday
At Wed 12:00Z the HDD leader's YES has bid ~0.98 and no ask, and rivals have no YES bid on 702/754 rows; 0 losses on
802 selected rows but only 7 taker bets at ask <= 0.98. Winner-take-all structure does not create the stale-rival
edge; the book is made by people reading HDD.
### Maker leftovers on decided Billboard races are real but tiny
TA-E-M: 22 fills, +3.0% [+2.5,+3.7], 0 losses; at most ~$1.7k total profit over 20 months even at full queue
priority (45.9k contracts through q). Not worth a bot.
### HDD's public Sanity CMS has 2025 history and song charts that are post-week only
273 chart docs: midweek-20 back to 2025-01-28, hits-top-50 to 2025-01-17 (dump data/billboard_top/hdd_chart_docs_all.json).
Song charts (song-revenue / song-streams) are FINAL numbers created after Billboard's Hot 100 reveal and stop 2025-10:
no song forecast. Wayback is SSL-blocked from this Mac (python requests), so kworb vintages are unavailable.
### Title-matching bugs can put NO bets on the true winner
HDD "MAN'S" -> "MAN S", "$OME $EXY $ONG$" -> "OME EXY ONGS", "KPOP DEMON HUNTERS | SOUNDTRACK", "MUSIC" vs Kalshi
"I AM MUSIC": a naive normalizer classes the real #1 as "absent" and the rule buys NO on it. Fix: apostrophe/$ handling,
artist+title matching, substring guard, skip weeks whose leader is unmatched (AMENDMENT 1).
