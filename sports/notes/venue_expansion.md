# Venue expansion: Polymarket US client, cross-venue map, ForecastEx feasibility (2026-10-28)

Builds on notes/venues.md (2026-10-22). Nothing here placed an order, opened an account, or touched a key.
Tags: [F] = fetched/measured by us; [S] = a search snippet or third party we did not confirm; UNVERIFIED = no source.

## Verdict, ranked by edge or capacity added

| # | Venue | What it adds | Proven edge there? | Do next |
|---|---|---|---|---|
| 1 | **Polymarket US** | A second order book for the same E1/E4 weather bet on **6 of our 24 cities**: same NWS CLI stations, same 2F brackets, matched 48/48 brackets 1:1 [F]. Every NBA/NHL/NFL game is listed with a moneyline plus totals and spreads ladders. NBA moneyline books are much deeper than Kalshi's (median top-of-book ~119k vs ~4k contracts [F]). Maker fee 0; taker 0.0695·p(1-p) [F]. Automated API orders are allowed [F]. | **No.** The dev port scored -0.14% on 283 bets (p .53) using display quotes (venues.md), which cannot resolve the question either way. These bets are the same bets as on Kalshi (correlated), so this adds capacity, not a new edge. | Open the account and create keys (steps below). Install the F4 forward paper logger. Go live only on forward evidence, at tier B. |
| 2 | **ForecastEx (IBKR)** | The only other liquid weather book, and the best chance of an **independent** weather edge. **18 of our 24 stations are the same airports**, but settlement uses a different source (Weather Underground daily-observations max, not the CLI). Contracts are above-X thresholds in 1F steps. Daily-high volume is about 3.9M pairs in 14 days; LAX alone is about 106k pairs/day [F price files]. | **Unknown.** The **$0.01/contract fee is about 1.0% of cost on a 97c NO**, close to E1's whole +1.45% edge, so it would have to price the tails worse than Kalshi does. | Run a pre-registered **proxy backtest from the public daily price files now** (no account needed). Open an IBKR account only if that passes. |
| -- | Others (Robinhood, Coinbase, Webull, sports-only books) | Unchanged from venues.md: they route to Kalshi or ForecastEx, or are sports-only with no tested edge. | -- | Skip. |

In dollars: at today's sizes ($50 unit, tier A ≤ 1u/night) neither venue adds evidence-backed $/month yet.
- **Polymarket US** is cheap optionality: about 1 hour of your time, and it doubles the book on 6 cities and adds a deep sports book. Our open-totals and NHL timing leads need low-fee execution, and that sports book could provide it.
- **ForecastEx** is the only venue that could hold a *new* weather edge with capacity. That matters for the $5k→$10k capital gate ("a 2nd edge passes sealed + forward").

## Your action list

1. **Polymarket US keys (about 15 min)** [F docs.polymarket.us/api-reference/authentication.md]:
   1. Install the Polymarket US iPhone app (https://apps.apple.com/us/app/polymarket/id6648798962). Create the account with Apple, Google or email, then finish identity verification (name, DOB, address, SSN; sometimes ID + selfie). Wait for "Approved to Start Trading". For an invite code or help, email support@polymarket.us.
   2. Fund it if you want to (debit card, ACH, Apple Pay, PayPal, Venmo, USDC, or wire with a $5k minimum). No funding is needed for the key step.
   3. Open https://polymarket.us/developer and sign in with **the same method** you used in the app. Switching sign-in methods can break key access.
   4. Click create API key. Copy the **Key ID** and the **Secret Key**. The secret is shown once.
   5. Save the secret as a one-line file, for example `~/.claude/quant-kalshi/keys/pmus.secret`, then `chmod 600` it. Add these lines to `~/.claude/quant-kalshi/.env` yourself:
      `POLYMARKET_US_KEY_ID=<key id>` and `POLYMARKET_US_SECRET_PATH=/Users/jonathanwang/.claude/quant-kalshi/keys/pmus.secret`.
      Optional caps, with these defaults: `PMUS_MAX_ORDER_USD=5`, `PMUS_MAX_CONTRACTS=10`, `PMUS_MAX_DAILY_USD=25`.
   6. Run `../.venv_quant/bin/python scripts/venues/polymarket_us_client.py preflight`. It should say "signed GET ... OK".
   7. Once, and only with your yes: `... polymarket_us_client.py smoke <weather-slug> --live`.
      - It rests 1 NO contract about 10c below the market, post-only, so it cannot fill. It reads the order back and cancels it.
      - Check in the app that the order showed as "Buy No" at the printed price. If not, set `PMUS_PRICE_CONVENTION=nominal` (see "Open questions").
   8. Revoke the key at the same page if it is ever exposed.
2. **Say yes or no to installing the F4 paper logger.** It is `scripts/venues_weather.py paper tick`, built 10-22 and not installed. It is the clean forward test of E1 on the live Polymarket book. I would add a maker variant to it, mirroring E4.
3. **Say yes or no to the ForecastEx proxy backtest** (section 3). It needs no account, no money and no keys.
4. Only after step 3 passes: open an IBKR Pro account, request **ForecastEx / event-contract trading permission** in Client Portal, and check whether California is eligible (UNVERIFIED). Also ask about a paper account.

## 1. Polymarket US client (built)

Files:
- `scripts/venues/base.py`: the `Venue` protocol and `Book`/`OrderResult` types.
- `scripts/venues/__init__.py`: `get_venue("polymarket_us", live=...)`.
- `scripts/venues/polymarket_us_client.py`: the client.
- `tests/test_polymarket_us_client.py`: 13 tests, mocked HTTP, all pass.

What it does:
- **Read-only, no keys, working now** [F tested live 10-28 06Z]: `search`, `markets`, `market`, `event`, `book`, `bbo`, `fee` (uses each market's `feeCoefficient`). CLI: `search`, `book`, `preflight`.
- **Account reads (key needed, no `--live`):** `balance` (/v1/account/balances), `positions` (/v1/portfolio/positions), `open_orders` (/v1/orders/open), `fills` (/v1/portfolio/activities?types=ACTIVITY_TYPE_TRADE).
- **Trading (key, `--live`, chmod-600 secret, no KILL file):**
  - `place_limit` builds a LIMIT order with intent BUY_LONG or BUY_SHORT, `participateDontInitiate`=true (post-only), `manualOrderIndicator`=AUTOMATIC, and GTC or GTD (`goodTillTime`).
  - Before sending it checks: post-only must not cross; tick and minimum-quantity multiples; per-order, per-contract and per-UTC-day caps.
  - `preview`, `cancel`, `cancel_all`.
  - Without `--live` it prints the exact body it would send and sends nothing. The verified CLI refusal: "REFUSED: POLYMARKET_US_KEY_ID + POLYMARKET_US_SECRET_PATH missing from .env".
- **Signing** [F docs + official SDK client.py/auth.py]:
  - Ed25519 over `ts_ms + METHOD + path`, where path includes `/v1` but not the query string.
  - The secret is base64; its first 32 bytes are the seed.
  - Headers are X-PM-Access-Key, X-PM-Timestamp and X-PM-Signature. The timestamp must be within 30 s.
  - The test checks the signature against the public key.
- **Idempotency:** Polymarket US has **no client order id** [F create-order schema], so the client handles it:
  - Each order is written to `~/.claude/quant-kalshi/data/pmus_orders.jsonl` *before* the POST.
  - Each tag gets one decision.
  - Order POSTs are never retried; the official SDK doesn't retry them either.
  - A POST with no answer, or a 5xx answer, is marked `ambiguous`. You then reconcile with `open_orders(slug)`.
- **Rate limits** [F]: 25 req/s per IP, shared. On a 429 the client waits at least 1 s and backs off exponentially, for GETs and cancels only. A "Global Rate Limit Exceeded" reject on an order is a 5-second latency stopgap: the order did not go in.
- **Router plug-in** (bet.py is being built in parallel; I did not edit it): `v = get_venue("polymarket_us", live=args.live); v.place_limit(slug, side, price, n, post_only=True, expire=start)`. Prices are always in the side's own terms (NO at 0.96 means 0.96), and the client translates them to the wire. The router keeps the tier caps and loss stops. The client keeps the venue caps and the KILL file, which it shares with kalshi_exec.

## 2. Cross-venue map: Kalshi vs Polymarket US (live snapshot 2026-10-28 06:02Z)

Re-run with `python scripts/venues/xvenue_map.py [weather|sports]`. Rows go to `data/xvenue_map_<stamp>.json` (gitignored).
The latest runs are 20261028T0602Z (weather + sports) and 0604Z (sports after a team-matching fix).

**Weather, daily highs (E1/E2/E4):**
- **Overlap: 6 of our 24 series** [F series list + weather FAQ + market text]:

  | Kalshi series | Station |
  |---|---|
  | KXHIGHNY | KNYC |
  | KXHIGHCHI | KMDW |
  | KXHIGHMIA | KMIA |
  | KXHIGHLAX | KLAX |
  | KXHIGHPHIL | KPHL |
  | KXHIGHTSFO | KSFO |

- Missing on Polymarket US: AUS, DEN and 16 of the 17 X1 cities.
- **Settlement:**
  - Both venues use the NWS Daily Climate Report (CLI) for the same station, in whole °F.
  - Polymarket settles at 08:00 ET the next day. If the CLI disagrees with the 24-h METAR it may wait until 11:00 ET; if nothing is published within a week, it settles at the last fair price [F weather-faqs].
  - Kalshi settles via The Weather Company, which matched the CLI on 3,080 of 3,081 days (venues_weather.py). The 10-22 build found 2 disputed brackets on 1 day (MIA 2026-08-29).
  - So the basis risk is small but not zero.
  - The FAQ table still lists only 5 cities; PHL appears only in the market text.
- **Brackets:** identical 2F bins plus two tails. All 48 live Polymarket brackets matched a Kalshi bracket 1:1.

Spread and depth on the 48 matched brackets:

| Brackets | Median spread, PM / Kalshi | Median qty at best YES bid (= what a NO taker can hit), PM / Kalshi | At best YES ask, PM / Kalshi |
|---|---|---|---|
| all 48 | 1c / 1c | 3 / 1 | 103 / 123 |
| 33 tails (PM YES ask ≤ .10) | 1c / 1c | 0 / 0 | 1,300 / 613 |

- Books with an empty side: PM 22/48, Kalshi 25/48. On the tails the YES bid is usually **empty on both venues**, so the NO ask is $1 and a taker gets nothing; that is the case for maker E4.
- **LAX and SFO were inside their 22:00 decision window for 10-28 at snapshot time.** Contracts at the YES bid, bracket by bracket:

  | Venue | Contracts at YES bid per bracket |
  |---|---|
  | PM | 0, 0, 164, 203, 16, 7499, 1, 6, 16, 80, 28, 84 |
  | Kalshi | 0, 0, 386, 1, 70, 2571, 0, 2, 20, 8, 11, 35 |

  Polymarket's book is at least as deep, so on these 6 cities it roughly **doubles the size available per pick**. That matters because E4 maker capacity is capped by adverse selection (~$75/pick per FUND_ARCHITECTURE).
- **Can E1 run there? Yes, as an exact port.**
  - The code exists already: `venues_weather.py` reuses the frozen rule, Polymarket fees, and the 22:00 local decision.
  - The quote must be the live book, not price-history: display quotes invent bids when the book is empty (the 10-22 trap).
  - The P1 dev result does not resolve the question. The clean test is the F4 forward logger, plus a maker variant (E4 port).
  - The sealed P1 (2026-08-15..) is still unscored and needs your yes. It runs on display quotes, so it is indicative only.
- **Lows:** Polymarket lists daily lows for NYC and MDW (6 brackets each). Series also exist for MIA, LAX and SFO, but they had no live events. We don't trade lows (X2 failed dev).
- **Rain:** Polymarket has **daily** "Where will it rain" markets at KLAX, KMIA, KNYC, KORD, KPHL and KSFO. Each resolves YES if any METAR or SPECI observation that day carries a rain code; there is no minimum amount. Median spread is 7c.
  - Our Kalshi rain work (L39 logger) is **monthly CLI totals**, a different product, so there is no port.
  - The daily rain markets are a new hypothesis (forecast PoP vs price on wide-spread markets), not part of this pass.

**Sports (game winners + full-game totals, games within 60 h):**

| League | Market | PM games | Matched to Kalshi | Median spread PM / K | Median top qty bid/ask, PM | Median top qty bid/ask, Kalshi |
|---|---|---|---|---|---|---|
| NBA | winner | 15 | 15 | 1c / 1c | 119k / 132k | 4.4k / 42k |
| NBA | total (line nearest 50c) | 12 | 12 | 1.5c / 1c | 4.1k / 4.1k | 5.2k / 8.9k |
| NHL | winner | 15 | 15 | 2c / 2c | 1.5k / 400 | 431 / 1.2k |
| NHL | total | 15 | 15 | 8c / 18c | 622 / 38 | 300 / 40 |
| NFL | winner | 1 | 1 | 0.3c / 1c | 459k / 530k | 86k / 3.9M |

- NFL: only 1 game was in the window, and no full-game Polymarket total was found for it (only period totals).
- **Next-day NHL totals are still placeholder ladders on both venues** (Kalshi 1c/98c), which matches the 10-25 finding that the first tight Kalshi ladder comes about 15 h after listing.
- Polymarket also lists spreads ladders (~34 per NBA game) and props.
- **Prior result still stands:** Polymarket US game winners vs Pinnacle had 0/116 +EV quotes, all within 0.72c (soft_venues.md). Polymarket is a good *execution* venue for our timing leads (NBA/NHL open totals) but **not a source of soft prices** by itself.
- Not measured yet: whether Polymarket lists lines earlier or staler than Kalshi. That is the useful follow-up for L35/L34-type opener edges.

## 3. ForecastEx (Interactive Brokers) feasibility

Research agent, 2026-10-28. Raw price files and CFTC filing extracts are in `/Users/jonathanwang/.claude/jobs/588c8b26/tmp/forecastex/` (scratch, untracked).

**Contracts** [F daily price files + CFTC filings]:
- **24 cities in °F**, each with a daily high (`UH`) and daily low (`UL`) product:

  | Group | Station codes |
  |---|---|
  | **Same station as our Kalshi series (18)** | MDW, MIA, AUS, LAX, PHL, ATL, BOS, DFW, DCA, HOU, LAS, MSP, MSY, OKC, PHX, SAT, SEA, SFO |
  | Different station | NYC = **LGA** (Kalshi uses Central Park KNYC); Denver = **BKF** (Kalshi uses KDEN) |
  | ForecastEx only | BNA, CLT, DTW, JAX |
  | Kalshi only | EWR, SAN, SDF, TTN |

- Also listed:
  - Celsius highs (Canadian cities).
  - Hourly temperature (`HRU*`) for LAS, LAX, LGA, MIA and SFO.
  - Possibly rain and city-climate products (codes not decoded, UNVERIFIED).
  - Climate contracts: CO2, global temperature, emissions, sea level, AMOC, some expiring in 2030.
- **Ticker example:** `UHLGA_101926_63` means "LGA high on 10/19/26 above 63F". Strikes are 1F apart.
- **Pays $1/$0; tick $0.01.** YES and NO are **separate instruments**: YES is modelled as a call, NO as a put. You exit by buying the opposite side; there is no selling.
- Position accountability is 250,000 contracts.

**Settlement** [F CFTC ptc08062614351, eff. Aug 2026]:
- The value is the max (or min) of the **Weather Underground History → Daily "Daily Observations" table**. Results are final once the next day's first reading posts. Last trade is 23:59 local on the contract date.
- The January 2026 wording used WU's Summary-table "Actual" value instead, so pre-August and post-August rows are separate regimes.
- **What this means for us** (my inference, UNVERIFIED in magnitude):
  - A max taken from the table of hourly and special METARs can only be ≤ the CLI's continuous max, so the settlement can sit 0-1F lower than the CLI.
  - With E1's 3F margin this mostly matters for the low-tail and "above X" strikes near the margin. It is a reason to pre-register the port as a new trial, not to assume it.
- Whether "day" means local standard or daylight time is not stated (UNVERIFIED).

**Fees and carry:**
- IBKR commission $0.00 plus the exchange fee of **$0.01 per contract** [F IBKR pricing page; ForecastEx fee filing].
- At a 97c NO that is about 1.03% of cost, against Kalshi's 0.07·p(1-p) ≈ 0.2c (taker) or 0 (maker).
- IBKR advertises a 3.38% APY "coupon" on positions [F]. Since 2026-08-14 brokers no longer have to pass it on [F]. It is irrelevant for overnight weather.

**API** [F IBKR docs]:
- TWS API / IB Gateway: event contracts are `secType="OPT"`, `exchange="FORECASTX"`, `right` C = YES and P = NO.
- Web API: `GET /forecast/contract/market?underlyingConid=…&exchange=FORECASTX` plus `secdef/search` and `secdef/info`.
- Market data gives the bid and ask but no historical bars and no last price.
- Orders are **limit only**, DAY/GTC/IOC [S IBKR campus page].
- **IBKR also routes Kalshi** (`exchange="KALSHI"`, IBKR's own example uses `KXHIGHLAX`). That is a second access path to our existing markets, not a new book.
- Not established (UNVERIFIED): paper-account support for ForecastEx, whether market data is free, and rate limits.

**California** (UNVERIFIED):
- IBKR offers these "to eligible customers"; the only stated restriction is on US election contracts. defirate (third party) lists ForecastEx as live in CA with no sports.
- Only the account's permission request settles it.

**History for backtesting** [F]:
- `https://forecastex.com/api/download?type=prices&date=YYYYMMDD` returns about 26.7k rows per day.
- Fields: `event_contract`, `subtype` (YES/NO), `expiration_date`, `date`, `start/high/low/end_price`, `settlement_price`, `pair_quantity`, `open_interest`, `vwap`. There is **no bid or ask**.
- Coverage of the daily-high products by date:

  | Date | Daily-high coverage |
  |---|---|
  | 2025-06 | none |
  | 2026-02 | 1 city |
  | 2026-03 | 23 cities |

- So about 7 months of history (Mar-Oct 2026) exist.
- **Proxy backtest that needs no account:**
  1. Decide on D-1 with the frozen E1 rule (same 5 models at the same airports; LGA and BKF need their own forecast points).
  2. Price each pick at the D-1 `end_price` (the last trade of the day before) or the VWAP. Both are trade prices, not quotes, so the test is indicative, like P1 on Polymarket.
  3. Settle on `settlement_price`.
  4. Pre-register before pulling any outcomes:
     - dev = Mar-Jul 2026, sealed = Aug-Oct (the new settlement wording).
     - Fee $0.01/contract.
     - Bar: p < .05 with units > 0.
     - Power: not computed yet. The files show 579 traded strike-days with YES VWAP ≤ 5c in 14 days (~1,200/month). Rule picks are a subset of those, but the count should sit well above Polymarket's ~80/month, so unlike P1 this test can plausibly resolve. The pre-registration must check power before scoring.
- **Liquidity:** daily-high pairs per day over 10-13..10-26 [F]:

  | City | Pairs/day |
  |---|---|
  | LAX | 105.7k |
  | SEA | 19.8k |
  | SFO | 15.5k |
  | MDW | 12.1k |
  | MIA | 11.5k |
  | AUS | 10.3k |
  | PHL | 7.1k |
  | LGA | 5.4k |
  | lows | 170-970 |

  579 of 5,537 strike-days traded a YES VWAP ≤ 5c, which is where consensus-reject lives. Weather is about 64% of all ForecastEx volume. (Whether `pair_quantity` counts pairs or contracts is UNVERIFIED.)
- **Overlap with consensus-reject:** structurally yes.
  - The same forecast inputs serve 18 stations directly.
  - Thresholds instead of brackets make the rule simpler (one side of each strike).
  - What is different is the settlement source, which is exactly why it might be **independent** of our Kalshi P&L. It might also be a *more* retail-mispriced book: its flow comes through IBKR ForecastTrader and Robinhood.

## Open questions

- **Polymarket US NO-order price convention**:
  - docs concepts/orders.md says "the price always refers to the YES side"; the create-order schema does not say what `price` means with BUY_SHORT.
  - The client defaults to YES-side pricing and exposes `PMUS_PRICE_CONVENTION`.
  - A post-only order with the wrong convention either rests far from the market or is rejected for crossing; either way it cannot fill. That is why the `smoke` test is safe and settles the question.
- The Polymarket US state list is not published; KYC decides. A third party (defirate, 10-04) says CA is available, including sports.
- ForecastEx: CA eligibility, paper account support, data fees, and the day boundary.
