# RESEARCH QUEUE -- keep agents running (user 2026-11-15: "you should have been trying to confirm and find more
# edges in the market through continually running agents")

Rule for the lead: keep 3-4 research agents running at all times while the user is active. When one finishes: merge
its inbox rows (AGENT_PROTOCOL.md), ingest LEARNINGS into the vault, then launch the next item below. Re-prioritise by
evidence: anything with CLV/ROI signal jumps the queue; game-day prep comes first on game days. Agents never place
orders; money changes stay with the user.

## Running (2026-11-19, lead launched)
- forward_review_1119 -- score every forward/live trial vs its own pre-reg since 11-15 (notes/forward_review_2026-11-19.md)  [DONE 11-19: none promote; LX1 kill pending 11-20]
- cs2_model -- public-results CS2 rating model vs Kalshi KXCS2GAME after taker fee, time split (notes/cs2_model.md)  [DONE 11-19: dead; FLB favourite post-hoc lead]
- pm_full_census -- every active Polymarket (global + US) by category, tested vs untested, dev tests on best untested (notes/pm_full_census.md)  [DONE 11-19: T1 dead; PM US fee .095]
- kalshi_fg_markets -- user idea "kicker stats": KXNFLFG ladders vs points model  [DONE 11-19: lead, dev +23% n56 n.d.; forward paper next]
- strategy_synthesis -- taxonomy of ~100 tests -> model-building lessons, candidate rubric, top-15 untested, 30-day plan (notes/STRATEGY_v2.md)  [DONE 11-20: notes/STRATEGY_v2.md; nowcast-of-free-official-source = 5/6 sealed passes, models/bias/maker 0/12]
- derived_ladders -- generalise KFG1: census + shape-calibration of all Kalshi secondary sports ladders, dev test best 1-2 (notes/derived_ladders.md)  [DONE 11-20: dead; ladders calibrated]
- ufc_tsa -- UFC method of victory + daily TSA  [DONE 11-20: both dead]
- trading_ways_survey -- all non-PM retail trading strategies ranked for CA + $1-2k, pre-reg top 1-2 (notes/trading_ways_survey.md)  [DONE 11-19: nothing doubles; promos/RSA/odd-lot = small one-offs]
- altdata_kpi -- census + pre-reg of JBLU / foot traffic / card spend / TikTok Shop / SPR series (notes/altdata_kpi.md)  [DONE 11-19: dead, Carbon Arc paywalled]
## Done 11-16..19: e4_timing, ot1_extension, tropical_nhc, forward_review_1115, fed_vs_futures, mortgage_rates (see LEARNINGS_2026-11.md)

## Next up (priority order)
1. K-NFL/K50-NFL/K1 forward scoring after Sunday's games (judge on CLV vs Kalshi's own strike, not ROI).
2. Information edges (the only lever left given execution cost): late news timing (news_lag), goalie confirmations
   (NHL), weather updates for NFL totals between model runs, official injury designations.
3. Live in-game Kalshi vs a real-time sharp feed (depends on sharp_feed_research cost).
4. Hurricane-landfall NO (LX2 climate +3.0%, 36 events, 0 losses) -- sealed 2025 needs the user's yes.
5. X1 maker forward paper (17 cities; user call to go live).
6. Fix gap_logger team matcher (West Ham/West Brom) + log raw Pinnacle odds/age (KS12 proposal).
7. NFLF re-arm prerequisites: deploy order_guard fix (cec89a0, user gate) + replay test on 11-08/09 fills.
8. Wire scripts/kx_maker_paper.py as a free shadow judged on filled CLV (low priority).

## From series watch (auto, scripts/series_watch.py; read-only census leads, untested; 11-16: NEW = no market closed or created before the previous snapshot; already-tested series (FUND_BOOK 3d, LEARNINGS) are never queued -- pruned 27 of 30 11-15 lines (26 tested + KXCOINBASE listed since 08-04): KX*EVENT, KXAAAGASCD*, KXERCOT*M, Q016 commodities, KXTEMPNYCHS, KXCOINBASE)
- 2026-11-15 **KXARCTICICEMAX** Arctic sea ice max extent -- newly active, score 8/10, weather, custom, 11 mkts, vol24 1041 [related tested: KXSEAICE too few events (phys_ladders.md)] (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXBIGGESTQUAKEMONTH** Biggest earthquake in a given month -- NEW, score 8/10, geo/space, monthly, 9 mkts, vol24 5 [related tested: daily KXBIGGESTQUAKE dead (e1x_quake.md)] (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXPERFORMROLE** will person perform role -- newly active, score 8/10, charts/media, one_off, 15 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXRARETTS** Rare Beauty TikTok Shop revenue -- NEW, score 8/10, charts/media, monthly, 7 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXSNOWDAYS** Cumulative snow days in cities -- NEW, score 7/10, weather, custom, 22 mkts, vol24 3859 [related tested: snow: resort/city monthly snow too few settled events, city monthly snow = L39 (consensus_reject_census.md)] (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXJBLU** JetBlue Airways KPI -- NEW, score 7/10, other, one_off, 31 mkts, vol24 1597 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXOSCARS** who will win oscar -- NEW, score 7/10, other, one_off, 72 mkts, vol24 1151 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXSNOWTHANKSGIVING** Snow on Thanksgiving -- NEW, score 7/10, weather, annual, 21 mkts, vol24 29 [related tested: snow: resort/city monthly snow too few settled events, city monthly snow = L39 (consensus_reject_census.md)] (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXHOUHOME** Houston home values -- NEW, score 7/10, macro, monthly, 14 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXSANTAMONICAHOME** Santa Monica home values -- NEW, score 7/10, macro, monthly, 5 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXSEAHOME** Seattle home values -- NEW, score 7/10, macro, monthly, 5 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXRHODETTS** Rhode TikTok Shop revenue -- NEW, score 6/10, charts/media, monthly, 10 mkts, vol24 4335 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXTHEVOICE** WHO WILL WIN THE VOICE? -- newly active, score 6/10, other, one_off, 28 mkts, vol24 2002 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXSPRLVLDATE** When will the US SPR level be above a threshold? -- NEW, score 6/10, transport/energy, one_off, 3 mkts, vol24 1596 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXREMTTS** r.e.m. beauty TikTok Shop revenue -- NEW, score 6/10, charts/media, monthly, 10 mkts, vol24 533 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXCELEBTTSARRAY** Celebrity beauty TikTok Shop array -- NEW, score 6/10, charts/media, monthly, 13 mkts, vol24 401 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXSNOWCITIES** Snowfall by city -- NEW, score 6/10, weather, custom, 5 mkts, vol24 357 [related tested: snow: resort/city monthly snow too few settled events, city monthly snow = L39 (consensus_reject_census.md)] (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXDDSFT** Dillard's Foot Traffic -- NEW, score 6/10, other, one_off, 8 mkts, vol24 238 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXDGFT** Dollar General Foot Traffic -- NEW, score 6/10, other, one_off, 8 mkts, vol24 234 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXFIVEFT** Five Below Foot Traffic -- NEW, score 6/10, other, one_off, 10 mkts, vol24 212 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXADMITLOWGROUP** College with Lowest Admissions Rate -- NEW, score 6/10, other, custom, 70 mkts, vol24 152 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXBTCHITBY** Bitcoin above a price by a date -- NEW, score 6/10, markets/crypto, custom, 5 mkts, vol24 94 [related tested: crypto ladders tested (OT1/CB1)] (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXBJFT** BJ's Wholesale Club Foot Traffic -- NEW, score 6/10, other, one_off, 10 mkts, vol24 58 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXROLEINPRODUCTIONLI** will person perfom role in production -- newly active, score 6/10, other, one_off, 11 mkts, vol24 38 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXDALHOME** Dallas home values -- NEW, score 6/10, macro, monthly, 5 mkts, vol24 7 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXALDICC** ALDI Credit Card Spend -- NEW, score 6/10, other, one_off, 10 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXAPPLEBEESCC** Applebee's Credit Card Spend -- NEW, score 6/10, other, one_off, 10 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXAUSHOME** Austin home values -- NEW, score 6/10, macro, monthly, 5 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXCHIHOME** Chicago home values -- NEW, score 6/10, macro, monthly, 5 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXCHILISCC** Chili's Credit Card Spend -- NEW, score 6/10, other, one_off, 9 mkts, vol24 0 (notes/series_watch/2026-11-15.md)


- NEXT (after ops_edgelab_v2_apply finishes): implement GOAL rules 6/7 amendment (user-approved 11-18) WITH the critic's
  edits a-g (inbox row critic_amendment_v2; notes/bar_calibration_v2.md): code + tests in edgelab, sandbox re-check, user re-lock of fund.yaml sha.

## Dated re-checks (do not lose)
- 2026-11-24..30  L39 live window (monthly rain); 11-27 TG1 Thanksgiving snow settles.
- ~2026-12-14     State gas: 30-date freeze -> fit lag model, score GS1 (notes/state_gas.md).
- ~2026-12-15     Home values: does Zillow's page show the Nov value before Kalshi closes 12-17 15:00Z? Browser check only
                  (script got 403-blocked) (notes/home_values.md).
- 2026-12-20     Econ-event ladders: makers arrived? EV1 forward (notes/econ_event_ladders.md).
- 2026-11-20 08:00Z  LX1 confirm kill + retire scan (forward_review_1119: 1 of 5 violations).
- 2026-12-02     NFL fade F1 one-shot score; 2026-12-06 NHL goalie-lag test eligible (>= 3 wks of logger data).
- 2027-04-01     Snow days SD1/SD2 settle (needs edgelab SCORING_VERSION 2 for scalar payouts).
