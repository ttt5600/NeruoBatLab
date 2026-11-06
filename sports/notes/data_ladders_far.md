# DL-PW -- Kalshi weekly chokepoint transits (IMF PortWatch) far-bracket maker; G-EIA screened out

PRE-REGISTERED 2026-11-06, written BEFORE any Kalshi candle, trade or settlement was joined to a forecast.
Scripts: `scripts/data_ladders_fetch.py` (market lists), `scripts/portwatch_fetch.py` (source history),
`scripts/dl_portwatch.py` (audit -> build -> candles -> trades). Rule `edgelab/rules/portwatch.py`, dataset
`kalshi_portwatch` (vault.yaml), family `kalshi_data_ladders`. Data `data/dl/` (gitignored).

## Disclosed before writing (no outcomes looked at)
- Market lists (structure, volume only): KXHORMUZWEEKLY 28 events (26APR26..26NOV08, 9-14 `greater` rungs, 24h-era
  volume 55k-544k contracts/event); KXPANAMAWEEKLY / KXSUEZWEEKLY / KXBABELMANDEBWEEKLY 13-15 events each from
  26AUG02 (5-12 rungs; volume < 20k/event until late Sep, 60k-260k from Oct). Fee_type quadratic (maker 0).
- Contract: YES iff the sum of PortWatch daily transit calls Mon..Sun of the week ending on the ticker date is above K.
  Lists the Friday ~16:00Z before the week (some earlier), closes the Tuesday after (13:00-14:00Z).
- PortWatch layer `Daily_Chokepoints_Data` lastEditDate = 2026-11-03 14:13Z (a Tuesday): the source is updated
  WEEKLY on Tuesdays, so NOTHING of the target week is observable from the settlement source before close. This is not
  a partial-observation market like TSA/ERCOT; the only information is last week's published total. The layer keeps
  only the current revision (no vintages) -> the audit gate below measures revision drift against settlements.
- G-EIA KXEIACRUDEW: 11 events, all from 2026-09-02, total volume 0-24k contracts/event (5 of 10 settled events < 100
  contracts). Too few events and too thin for a trial -> NOT registered (no trial spent). Other census ladders
  not on the dead list are not weekly data series (RT, Truth, YouTube, charts, Netflix -> different mechanism).

## Audit gate (must pass before scoring; settlements are read only here)
A_W = sum of PortWatch n_total over Mon..Sun of week W. On >= 90% of settled events every rung's result must agree
with A_W vs K. If Mon..Sun fails and Sun..Sat passes, use Sun..Sat. Fail -> stop and report (no scoring).

## Forecast and selection (frozen)
Decision T = Wednesday 00:00Z of week W (Tuesday evening ET, ~10 h after the Tuesday publication of week W-1).
inputs_as_of = Tuesday 15:00Z. F = A_{W-1} (last published full week). Residuals r = ln((A_V + 1)/(A_{V-1} + 1)) for
every earlier week V of the SAME chokepoint since 2019 whose week ended >= 9 days before T (i.e. published by the
Tuesday before last). Rung REJECTED-HIGH (bet NO) if K + 1 >= (F + 1) e^(r_max + 0.05); REJECTED-LOW (bet YES) if
K + 1 <= (F + 1) e^(r_min - 0.05). Only events open at T with a quote at T.
Amended before any edgelab data access (2026-11-06): bet only if week W-1 was PUBLISHED before T, publication time =
earliest Kalshi settlement_ts over the 4 series for that week end (PortWatch was a day late on 10-13 and 10-27;
settlements land 14:20-14:50Z). Fetch note: the ArcGIS layer 429s on full queries (shared 6,000 units/min); queries
with cacheHint=true and 3 fields go through. As fetched 11-06 the layer ends 2026-10-25 (week 11-01 unpublished).

## Trial (k = 1): pw_t1_maker
Post on the far side at T: NO bid q = min(NO mid, NO ask - 0.01) (NO ask = 1 - yes_bid, NO mid = 1 - (yb+ya)/2);
mirror for YES (q = min(YES mid, YES ask - 0.01)). Quote = last hourly candle ending <= T; yes_bid missing/0 ->
NO ask 1.00. Requires (NO side) yes_ask <= 0.99 / (YES side) yes_bid >= 0.01 and spread >= 2c; q floored to the cent. Filled iff a later tape trade (before
close) prints strictly THROUGH q (NO: taker_side yes at yes_price > 1 - q; YES: taker_side no at yes_price < q).
One contract, maker fee 0, held to settlement, no cancel. No taker trial (far rungs one-sided in every ladder seen so
far; would be the TSA T1 zero).

## Split (vault, by week end; set before outcomes): dev 2026-04-26..2026-09-20, sealed 2026-09-27..2026-11-01,
forward 2026-11-08... Dev is mostly Hormuz (Panama/Suez/Bab dev weeks are thin).
min_n 30 fills. Bar = edgelab family bar for kalshi_data_ladders at registration.

## Power / honesty
Far NO fills cost ~.90-.98, profit a few cents; with ~40 dev events and maybe 1-3 fills each, n ~ 30-100. A pass needs
nearly zero losses. Hormuz is in a regime break (transits ~5-100/week vs ~700 normal) -> r_min/r_max from 2026 crash
weeks are extreme, so few rungs will be rejected. Capacity: maker fills limited by the tape; reported with the result.
Kill: dev ROI <= 0 or fail -> DEAD for this family, no re-tweak.

## RESULT 2026-11-06 -- AUDIT GATE FAILED, trial never scored -> pw_t1_maker-49f5009f retired (dead)
`dl_portwatch.py audit`: Mon..Sun sums of the CURRENT PortWatch layer agree with Kalshi settlements on 49/63 settled
events (0.778; Sun..Sat 0.492) vs the 0.90 gate. 13 of 14 mismatches are Hormuz and they are revisions, not window
errors: MAY03 settled in (30,40] but the layer now says 86; MAY17 (10,15] vs 38; JUL19 (75,100] vs 44; JUL05
(200,225] vs 162. Suez/Panama/Bab agree on all but 1 event (Suez AUG02 291 vs (280,290]).
Mechanism: Kalshi settles on PortWatch's FIRST print; in the Hormuz crisis (AIS-dark tankers) PortWatch later revises
weekly counts by up to ~2x. The layer keeps no vintages, so any backtest built on it uses revised W-1 inputs and
revised history -> not point-in-time. Lesson: audit settlement-vs-source BEFORE trusting a public series as history.
Other findings: publication is weekly (Tuesday ~14:13Z; 10-13 and 10-27 a day late; week 11-01 still unpublished
11-06), so nothing of the target week is observable before close -- these are forecast markets, not
partial-observation markets. G-EIA KXEIACRUDEW: 10 settled events, mostly < 100 contracts -> no trial.
Way forward (no trial spent): a Tuesday vintage logger (snapshot the layer each week, cacheHint=true) would give
first-print history; after ~20 weeks a forward-only Suez/Panama/Bab/Hormuz far-bracket test is possible.
