# Rumor / news-flow signals (2026-10-22)

The user asked (10-22): "Rumors on X and Reddit are a huge place to start, if we can pull and distill X information
accurately." This note covers:
1. what the 2023-2026 research says
2. which data we can get legally, and what it costs
3. what was built: `scripts/rumor_signals.py`
4. a pre-registered event study: GDELT news spikes vs Kalshi prices, 2025 dev only
5. whether paid X access is worth it

Every number below came from a source I read or a script I ran. Where I could not verify something, I say so.

## 1. Research: does social or news flow lead market prices?

Bottom line: **no paper shows public social or news volume leading prediction-market prices by a tradable lag
(minutes to hours) after fees.**

The evidence splits into three grades:
- **Market-tested after costs (1):** a trading result that survived costs.
- **Correlation or lead-lag only (2):** a statistical relationship, with no trading test.
- **Anecdote (3):** individual cases, or claims with no primary source.

| Work | What it tested | Grade |
|---|---|---|
| Brown, Rambaccussing, Reade & Rossi, *Economic Inquiry* 2018 ([PDF](https://ueaeprints.uea.ac.uk/id/eprint/64244/1/BetfairTwitterEI_revisedv2.pdf)) | 13.8M tweets vs per-second Betfair prices, 372 EPL matches in 2013/14. Tweet tone holds information that is not yet in the price, mostly right after goals or red cards. Most of it is priced within about 10 s. A back-of-envelope rule ("back when tone is positive") made +2.28% at 5% commission, but the 903k overlapping per-second bets come from only 372 matches. The study has no liquidity model and has not been replicated. | (1) with caveats. The horizon is SECONDS. |
| Croxson & Reade, *Economic Journal* 2014 | Betfair prices around goals adjust "quickly and completely". | (2) efficiency |
| Snowberg, Wolfers & Zitzewitz, *QJE* 2007 | On 2004 election night, prediction-market prices moved intraday together with futures. | (2) |
| Ng, Peng, Tao & Zhou 2026 (SSRN 5331995); Aktug & Torul | Polymarket leads Kalshi/PredictIt in price discovery, about 5-min lead. Large-trade order imbalance predicts later returns. Full texts were blocked; I read search summaries only. | (2), cross-venue rather than news |
| Tsang & Yang (arXiv 2603.03152) | Three shocks in 2024 on Polymarket's on-chain record. The debate jump mostly reverted. The assassination-attempt jump persisted. Biden's exit drew heavy two-sided trading with a small price change. | (2) |
| "Polymarket takes ~80 min to fully reprice" (Crypto Briefing, 2026-08) | No primary source found. | (3), unverified |
| Insider cases: Maduro (Jan 2026), Nobel (Oct 2025), Google Year-in-Search (CFTC/SDNY charges, May 2026); Mitts & Ofir 2026 (210,718 flagged wallet-market pairs, about $143M anomalous profit; first trade 71 min before the news of the Iran strike) | In each, informed money moved the price **before** the public news existed. | (3) individually, (2) systematically. This points the **opposite** way to "public news leads price". |
| Twitter vs elections: Gayo-Avello 2011; Bollen 2010 "Twitter mood predicts the DJIA" | Volume and sentiment methods were inadequate. Bollen is widely regarded as not replicating; I did not check that myself. | (2), skeptical |
| Injury news (Woj/Shams) vs NBA/NFL lines | No peer-reviewed latency study exists. There is only practitioner material ("lines overreact, then partly revert"). | (3) |
| Lopez-Lira & Tang (arXiv 2304.07619 v6) | ChatGPT headline scores predict next-day stock returns. The authors' own v6 says returns **decline as LLM adoption rises**. | (1) in equities, decaying |
| Halawi et al. 2024; ForecastBench; Prophet Arena (ICLR 2026) | Retrieval-based LLM forecasters approach the crowd but have not beaten superforecasters. Prophet Arena finds LLMs aggregate information **more slowly than the markets** near resolution. | (2), evidence against |
| GDELT studies (tone vs Chinese stocks; media tone vs FX at 1-6 month horizons) | No GDELT study uses prediction-market prices. | (2) |

What this means for us:
- Public news is priced within seconds to minutes when it is a clear event (a goal, a resignation).
- The documented "lead" in prediction markets comes from **private** information, and it runs ahead of the news.
- A free public feed polled every 15 minutes is the slowest participant in this chain. Any edge from it would have to come
  from an **under-reaction** lasting hours. That is what section 4 tests.

## 2. Data access: legal routes and cost (checked 10-22)

| Source | Cost | Auth | History | Latency | Notes |
|---|---|---|---|---|---|
| **X API, pay-per-use** ([pricing](https://docs.x.com/x-api/getting-started/pricing)) | $0.005 per post read. Counts endpoint $0.005-0.010 per request. Capped at 3M reads/month. | The user creates a dev account and app, then adds a bearer token | Recent search covers 7 days. **Full-archive search back to 2006 is included in pay-per-use** (500 posts/request; 300 requests per 15 min, 1/s) | Near real time | The only legal X route. The X free tier has no reads. Scraping and logged-out mirrors are out. |
| Reddit Data API | Free tier for non-commercial use. Commercial use needs an agreement. | OAuth app. Since the Responsible Builder Policy (Nov 2025), new tokens **need Reddit's approval**. The 100 QPM figure comes from a secondary source and is unverified. | Listings only | Minutes | The user would have to apply. |
| Reddit public RSS (`/r/<sub>/new/.rss`) | Free | none | Newest ~25-100 posts | Minutes | **Probed: RSS gives 200, `/new.json` gives 403.** Used at 1 request per 2 s. |
| GDELT 2.0 raw files (`data.gdeltproject.org/gdeltv2/`) | Free | none | 2015 onward | 15-min files | Per file: mentions ~85 KB, events ~75 KB, GKG ~3.8 MB (zipped). About 340 unique article URLs per 15 min in 2025. |
| GDELT DOC 2.0 API | Free | none | Rolling 3 months | 15 min | **Returned 429 ("one request every 5 seconds") on EVERY request from this Mac on 10-22, including the first one after a 40 s pause.** Probably the shared VPN exit IP. Not used. |
| GDELT on BigQuery | 1 TiB of queries/month free, then $6.25/TiB | Google Cloud account | Full archive | 15 min | Optional. Raw files were enough. |
| Google News RSS | Free | none | ~100 recent items | Minutes | Unofficial and undocumented endpoint. Light use. |
| Bluesky `api.bsky.app/xrpc/app.bsky.feed.searchPosts` | Free | **none (works)** | Searchable archive | Seconds | `public.api.bsky.app` returned 403 from here. Jetstream (websocket firehose) is free and live. |
| Kalshi candles | Free | none | `/historical/*` holds markets settled before about 2026-08-23. Intervals are 1 min, 1 h or 1 d. | Real time | **The live `/markets` and `/events` returned 503 `service_unavailable` all session on 10-22. `/historical` worked.** |
| Polymarket CLOB `/prices-history` | Free | none | `fidelity` in minutes; settled markets may be coarser (unverified) | Real time | Read-only; US product only for trading. |

## 3. Built: `scripts/rumor_signals.py`

The script uses only credential-free sources and no LLM.

| Command | What it does |
|---|---|
| `markets` | Indexes open Kalshi events (Politics, World, Companies, Economics, Elections, Mentions, plus NBA/NFL/NHL/MLB/NCAAF game markets) and the top ~1,000 Polymarket events by 24-hour volume. Each market gets a keyword: the person's last name plus their full name. Generic words such as "election", "reserve" and "league" are dropped. On 10-22 the Kalshi live API was down, so the index held only Polymarket: 9,256 markets and 1,202 keywords. |
| `ingest` | Runs one pass over five sources: the latest GDELT 15-minute mentions file, 13 Reddit subreddit RSS feeds, and, for the top 30 keywords, Google News RSS and Bluesky search. It keeps only items that match an open market and appends them to `data/rumor/timeline.jsonl`, deduplicated by id. |
| `report` | Shows per-market item counts for the last N hours against earlier windows. |

How matching works:
- The last-name token is the candidate filter.
- The **full name** must then also appear. This stops "Theo James" from matching LeBron James markets.
- Sports game markets also need an injury word, such as "ruled out" or "questionable".

Test tick on 10-22:
- 3,243 items fetched: Bluesky 1,473, Google News 1,121, GDELT 349, Reddit 300.
- 1,687 items matched a market.

Precision problems seen in that tick:
- The entity matches are correct, but **relevance is weak**. For example, Kardashian gossip matched "Kim Kardashian 2028 nomination", and a Lewis Hamilton tabloid story matched the F1 title market.
- A volume spike says nothing about direction.
- An LLM or classifier pass would be needed to separate rumor from noise. It is not built.

## 4. Event study (pre-registered in the script docstring before any count or price was pulled; DEV = 2025 only)

Setup:
- **Universe:** 134 settled Kalshi rumor-type markets in 2025: 28 events, 98 person keywords. Examples are Fed Chair nominee,
  cabinet departures, Trump and Biden pardons, administration departures, and the Rubio/Vance Senate seats.
  - Caveat: Kalshi's live `/markets` returned 503 all session, so the universe comes from `/historical` only. It includes only
    markets settled before about 2026-08-23; 22 series could not be checked for later settlements.
- **News volume:** GDELT 2.0 mentions files at :00 and :30 of every hour of 2025.
  - 17,520 files were requested and 841 failed (4.8%). About 1.4 GB was streamed; only hourly keyword counts were stored (28 MB on disk).
  - A hit means the keyword appears as a token in the article URL path.
- **Spike onsets:** 3,762 onsets across 157 keywords. 826 spike-market pairs had a live Kalshi price.

### Results (descriptive)

| Measure | Value |
|---|---|
| Mean \|dmid\| per hour, same markets, random hours (placebo) | 0.24c |
| Mean \|dmid\| 3 h and 2 h **before** the spike hour | 0.35c / 0.32c (elevated) |
| Mean \|dmid\| during the spike hour | 0.27c |
| Mean \|dmid\| in the first tradable hour after the signal (h+1 to h+2) | **0.12c (BELOW placebo)** |
| First cumulative >= 3c move within +-24 h (115 of 826 pairs) | **68% happen at or before the spike hour**. Lag quantiles (h): 10% -20, 25% -13.5, median **-4**, 75% +4, 90% +17.6 |
| \|move\| in the 6 h before the spike vs the 6 h after the signal | 0.73c vs 0.32c. Post > pre in only 14% of pairs |
| "News leads" (post >= 3c and pre < 3c) vs "price leads" (pre >= 3c and post < 3c) | 1.3% vs 3.4% |
| Pooled corr(log news count at hour h, \|dmid\| at hour h+L) | Peaks at L = -2/-3 (0.035), then 0.022 at L = +1 and 0.010 at L = +12. **Price activity leads news volume.** |

Clear examples from `study_pairs_gdelt.parquet`:
- **Giuliani pardon (Nov 2025):** 0.05 -> 0.995 before the GDELT spike hour.
- **Bongino leaving (Dec 15):** 0.35 -> 0.87 before the spike hour.
- **Blagojevich pardon (Feb 10):** 0.18 -> 0.965 before the spike hour.
- **Counter-example, Giuliani (May 30):** 0.16 -> 0.535 within 7 h *after* the spike, then 0.26 at 24 h. This is the kind of rumor
  that over-shoots and fades.

### Trials (ledger `rumor_gdelt_kalshi`, bar p < 0.0125)

| Rule | Bets | ROI | 95% CI | p | Win rate |
|---|---|---|---|---|---|
| M6 (follow the move, exit at 6 h) | 19 | -7.1% | [-10.4, -4.7] | 1.00 | 0.05 |
| M24 | 20 | -8.1% | [-12.6, -4.5] | 1.00 | 0.05 |
| F6 (fade the move) | 19 | -13.4% | [-23.2, -7.4] | 1.00 | 0.11 |
| F24 | 20 | -12.0% | [-23.0, -4.9] | 1.00 | 0.15 |

All four FAIL.
- Few bets qualify, because by the signal hour the 2c "move since h-1" rule rarely fires. The price has already settled.
- The rules lose roughly the spread plus fees, because nothing moves afterwards.
- The sample is small, so read this as "no sign of an edge", not as a precise estimate.

**Caveat on GDELT timing.** GDELT stamps an article when it is crawled. This can be minutes to hours after publication, and a
volume *spike* builds up as outlets pile on. So part of "price leads news" is GDELT being slow.
- That does not rescue the trade: GDELT is still what we would trade on.
- It does mean this study cannot tell us whether a **fast** source (X, wires) leads Kalshi. That is the X follow-up below.

Sports injuries were not studied. GDELT barely covers injury news, and Kalshi game markets move on play. The live matcher logs
injury items so they can be tested going forward.

## 5. Is paid X access worth it? A cheap test first, not a subscription

- What the literature and our GDELT result show: public news is priced fast, and on Kalshi rumor markets the price moved **before** the
  news-volume spike in about 2 of every 3 big moves. X is where many of these stories plausibly break first (not verified per episode here),
  so X is the one source that *could* lead. Whether it does has not been measured.
- **The cheap test:** `rumor_signals.py xcounts --confirm-spend` pulls X full-archive **hourly counts** for the same 98 people over 2025.
  - The counts endpoint costs $0.010 per request (docs.x.com pricing page, read 10-22).
  - The plan is about 1,274 requests, roughly **$13**. The page size per request is unverified; the script follows `next_token`, so the real cost may differ.
  - Then `study --source x --ledger` runs the identical pre-registered rules as new trials XM6/XM24/XF6/XF24.
- **Pass criterion:** the share of first >= 3c moves that come after the X onset rises well above GDELT's 0.32, and a rule clears p < 0.0125.
- Only then would post reads ($0.005/post) for live monitoring be worth paying for. Rough cost: about $5 per 1,000 posts. Watching
  100 names at about 20 posts/hour each would be about 1.4M posts/month, roughly $7k/month, so live X would have to be narrowly targeted.
- **What the user must provide:**
  1. Create an X developer account and app (pay-per-use).
  2. Add about $15-20 of credit.
  3. Put `X_BEARER_TOKEN=...` in `sports/.env`. Never paste it in chat.
  4. Say "yes, spend ~$13 on X counts".
- Reddit's OAuth API is not needed: public RSS covers new posts. Under the Nov-2025 policy a Reddit app needs Reddit's approval anyway.
