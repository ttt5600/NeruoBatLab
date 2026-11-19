# esports_pinnacle -- Kalshi esports match winners vs Pinnacle (agent started 2026-11-18)

Lead: notes/forum_strategy_scout.md idea #1. Prior = no edge (Kalshi soccer KS12 dead; the "gaps" vs free Pinnacle were
stale Pinnacle, notes/kalshi_soccer_sharp.md). This file: census, pre-registration, forward logger ops.

## Question
On Kalshi esports match-winner markets (KXCS2GAME, KXLOLGAME, KXVALORANTGAME, KXDOTA2GAME), does buying at the Kalshi
ask when a FRESH Pinnacle de-vigged price is >= 3c above ask + fee make money and beat Pinnacle's close?

## Pre-registration (written 2026-11-18 ~01:40Z, BEFORE the logger's first tick; no outcome has been seen)
Trial ESP1 (edgelab id esp1_kalshi_esports_pin_fresh, family cross_venue, forward only -- no dev data exists, see Data).
- Universe: Kalshi KXCS2GAME / KXLOLGAME / KXVALORANTGAME / KXDOTA2GAME events matched to a Pinnacle E Sports (sport 12)
  parent matchup with a 2-way full-match moneyline (period 0). Map-winner series are logged context only, not bets.
- Matching (all required): same game (Pinnacle league prefix "CS2 -" / "League of Legends" or "LoL" / "Valorant" /
  "Dota 2" -> the series); Kalshi start time (ET, parsed from the event ticker 26NOV191200 = Nov 19 12:00 ET) within
  +-90 min of Pinnacle startTime; the two Pinnacle teams map ONE-TO-ONE onto the two Kalshi yes_sub_titles by the
  normalized token-set matcher (gap_logger's fixed whole-word rule: equal sets or one inside the other, never substring
  of a token, never first-word), and exactly one Kalshi event satisfies this. Ambiguous -> no match.
- Pinnacle fair: conservative 2-way de-vig per side = min(multiplicative, power) of the side bought.
- Quote age: age_s = Date - last-modified of the Pinnacle markets/straight response (Cloudflare cache, max-age 905 s).
  Fresh = age_s <= 120 s. (Probe 11-18: a cache-busting query param returns HTTP 204 empty -> no way around the cache.)
- Signal: pregame (tick time < Pinnacle start - 2 min), fresh, Kalshi YES ask in [0.03, 0.97] with ask size >= 1, and
  fair - (ask + fee) >= 0.03 where fee = ceil(100 * 0.07 * 10 * p * (1-p)) / 1000 per contract (a 10-lot taker order,
  rounded up to the cent). Either team's YES may fire (= buying that team). FIRST qualifying tick per match only.
- Outcome: Kalshi settlement; a "scalar" (fair-market-price) settlement pays settlement_value per contract (walkover /
  cancel / postponed > 48 h) -- these count as bets at that value (they are part of the product's real P&L).
- CLV: Pinnacle close = the last Pinnacle snapshot (any age) before start, de-vigged the same way; CLV = close fair - ask
  (prob points). Also reported: CLV vs fresh-only close.
- Stats: ROI = sum pnl / sum (ask + fee); 95% CI by bootstrap over matches (event = cluster), 10k resamples.
- SUCCESS (-> propose a small forward money trial to the user, nothing automatic): n >= 300 bets AND ROI CI lower > 0
  AND both chronological halves ROI > 0 AND mean Pinnacle-close CLV > 0 AND p < cross_venue family bar.
- KILL (any): (a) after 28 logger days, 0 qualifying signals on >= 70% of days; (b) ROI point <= 0 at n = 300;
  (c) feasibility: after 14 days, fewer than 5 matched Kalshi-Pinnacle matches per day on average (300 bets then
  needs > 2 months even if every match fired) -> park as "thin", not test further.
- Descriptive only (not trials): thresholds 1/2/5c, stale-quote (age > 120 s) version of the same rule (expected to
  look better and be fake -- this is the control for the cache artefact), by game, by Kalshi spread.

## Data
- Local history: NONE. gap_logger (data/gap_pin_odds.jsonl) covers US leagues + 9 soccer leagues only; no esports
  league id in it. -> forward only.
- Pinnacle guest API: /sports/12/matchups, /sports/12/markets/straight (public web-app key, same as sports/live.py).
- Kalshi public GETs via scripts/kx_throttle.py.

## Census (2026-11-18, results/esports_pinnacle/census.py -> census.json; last 200 settled per series)
| series | open mkts / events now | settled med vol (ct) | scalar (fair-price) settles of 200 |
|---|---|---|---|
| KXCS2GAME | 152 / 76 | 5,622 | 22 |
| KXLOLGAME | 0 / 0 (off-season) | 313,891 | 2 |
| KXVALORANTGAME | 46 / 23 | 2,244 | 18 |
| KXDOTA2GAME | 40 / 20 | 6,355 | 16 |
| KXCS2MAP | 202 / 101 | 2,176 | 24 |
| KXVALORANTMAP | 94 / 47 | 447 | 16 |
| KXDOTA2MAP | 80 / 40 | 2,964 | 8 |
| KXR6GAME / KXCODGAME | 0 / 0 | 6,241 / 22,062 | 8 / 0 |
Rules (rules_secondary, identical across the 4 GAME series): market = FULL MATCH winner (map winner is a separate
series, KX*MAP, one event per map). Postponed and not started within 48 h, cancelled before play, or forfeited before
any play -> resolves to "the fair market price" (scalar; values seen 0.22..0.78, often 0.50). Forfeit after play
starts -> official result. Sources: KXCS2GAME BO3/HLTV/Gamers World; LoL Sofascore/BO3/GW; Valorant Valorant/GW/BO3;
Dota BO3/DLTV/GW. Timing: markets open ~1-3 days before (open_time), close_time = start + 48 h, can_close_early when a
winner is declared, settlement timer ~2-5 min. Markets keep trading IN-PLAY (no close at start) -> "pregame" must be
cut at the start time ourselves. Fee: quadratic taker (0.07 p(1-p)).
Books: many events are tier-2/3 (ESEA Advanced, CCT, Predator League, European Pro League) with very wide spreads at
listing (e.g. 0.62/0.82, 0.08/0.92, 0.01/0.99 on map markets) and 0 volume until near start.
Pinnacle at 01:24Z 11-19: only 8 parent esports matchups (CS2 Elisa Open Suomi, European Pro League, STK Ranked; Dota
European Pro League). Overlap is the open question the logger answers first (kill c).

## Log
### 2026-11-18 01:24-01:30Z probes (no outcomes)
- Pinnacle cache: /sports/12/markets/straight Age 550-858 s at read (cf HIT, max-age 905). Cache-busting query param
  (?_=random) -> HTTP 204 with an empty body (results/esports_pinnacle/probe_cache.py): the cache cannot be bypassed;
  fresh reads happen only right after a refill (first deployed tick 01:30:02Z read age_s 0). With 120 s ticks expect
  ~1 fresh read per 15-min cycle per match.
- Match dry-run with a 72 h window: Pinnacle listed 8 esports parent matchups (all starting within ~17 h), Kalshi 115
  events in 72 h; 6/8 Pinnacle matchups matched (5 exact, 1 subset: Pinnacle "WBT" = Kalshi "WBT Academy" vs MOUZ NXT,
  same start). Kalshi ticker times equal Pinnacle startTime exactly (e.g. 26NOV191030 ET = 15:30Z). Unmatched:
  BIG-NiP, GamerLegion-Inner Circle (STK Ranked; not listed on Kalshi or alias -- audit from fetch.jsonl unmatched_*).
- Pinnacle maxRiskStake on these matchups is often 100 (tier-3: 4 of 6 matched at 100-125; STK Ranked 3000). A $100 limit
  line is not a sharp reference -- descriptive split by max_risk planned at scoring.
- Overlap is the binding constraint: Pinnacle lists far fewer esports matches than Kalshi and only ~<1 day ahead.
  At ~6 matched/day, kill (c) (< 5/day over 14 days) is live.

## Ops
- Logger: scripts/esports_pin_logger.py (tick / settle / report). Job `esports-pin` in scripts/qs_job.sh; installer
  scripts/install_esports_pin_agent.sh; launchd com.jonathanwang.quant-sports-esports-pin, StartInterval 120 s, 24 h.
  Deployed copy ~/.claude/quant-sports (qs_job.sh was identical to deployed before the copy; only the new case added).
- Outputs: ~/.claude/quant-sports/data/esports_pin/{fetch,snaps,bets,settle}.jsonl, state decided.json; heartbeat
  state/esports_pin_heartbeat; settle hourly (state/esports_pin_settle_last); log logs/esports-pin.log.
- Verified 2026-11-19 01:30:03Z (kickstart): heartbeat rc=0, fetch row written (n_pin 0 in 6 h window, n_kal 3,
  Pinnacle markets age_s 0), settle ran exit 0. ~6-26 Kalshi GETs per tick via kx_throttle (<= ~8 s of the shared slot).
- Stop: launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-esports-pin. Edgelab: esp1_kalshi_esports_pin_fresh-91b54d37
  (cross_venue 11 trials, bar p < 4.55e-03). Check pace at day 14 (kill c) and score at n 300 or day 28.

## Learnings
### Pinnacle guest API cache cannot be bypassed with a query string
A cache-busting param on guest.api.arcadia.pinnacle.com returns HTTP 204 empty (probe 11-18, results/esports_pinnacle/
probe_cache.py). Freshness must come from timing reads after a refill (age_s = Date - last-modified), not from busting.
### Kalshi esports GAME markets are full-match winners that settle to "fair price" on pre-play forfeits
rules_secondary (census.json): cancel / pre-play forfeit / postponed > 48 h -> fair market price (scalar). 22/200
KXCS2GAME, 18/200 Valorant, 16/200 Dota recent markets settled scalar (values 0.22-0.78). Markets trade in-play
(close = start + 48 h), so any pregame rule must cut at the start time itself.
### Kalshi lists ~10x more esports matches than Pinnacle's guest feed
11-19 01:24Z: Pinnacle 8 esports parent matchups (all < 17 h out) vs Kalshi 115 GAME events in 72 h; 6/8 matched.
Many Pinnacle tier-3 lines carry a $100 max-risk limit, so "Pinnacle = sharp" is weak exactly where Kalshi is thin.
