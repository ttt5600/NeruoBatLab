# RotoBot pick tracker (2026-10-21)

User ask: "Rotobot is an AI thing with an app; if it's free, worth checking out. Scrape X for their tweets
and see how well they're doing." No record has been graded yet. This note covers what the product is,
the legitimate ways to read its posts on X, the tracker that is now built, and what we need from the user.

## 1. The product
- **RotoBot AI** (rotobot.ai). App Store id 6502530085, developer "Rotobot.ai", seller "RotoBot Inc.".
  It is rated 4.6 from about 2K ratings and was updated about 2 days before 10-21 (v3.0.39).
- **X account: [@rotobot_ai](https://x.com/rotobot_ai)**, linked from rotobot.ai. It also has Instagram
  `rotobot.ai` and a LinkedIn company page.
- **What it is:** an AI *research assistant*, not a pick service. Features:
  - a chat for prop and matchup questions
  - a prop analyzer with "L10 HIT" hit-rate cards
  - a daily "insights" feed of "curated picks"/"edges"
  - a parlay generator
  - fantasy tools (Sleeper/ESPN/Yahoo sync, start/sit, trade analyzer)
  
  Sports: NFL, NCAAF, NBA, MLB, NCAAB, with NHL "coming soon". Its own disclaimer says it offers
  "insights and analysis, not gambling, betting, or financial advice".
- **Price:** the download is free, but it uses in-app subscriptions:
  - Full Access: $179.99/yr, $39.99/mo or $14.99/wk (3-day trial)
  - Fantasy Pass: $79.99/yr or $9.99/wk
  - The store also lists a "Monthly Subscription" at $34.99, which conflicts with the $39.99 above.
  
  The free stat explorer on the web needs no signup. **So the picks are not free.** The free part is the
  research tools.
- **No public audited track record** appears on the site or the store page. The store page shows only
  "L10 hit" cards, which are player hit rates, not a record of its own picks.
- **UNVERIFIED: what @rotobot_ai actually posts.** x.com returned HTTP 402 to an anonymous fetch. Web search
  surfaced no indexed posts. I did not try to get past the login wall. So I cannot say:
  - whether the account posts picks at all, or marketing only
  - whether picks come with a line and price
  - which sports it covers, or how often it posts
  
  The user should glance at the profile; that one look settles whether there is anything to grade.

## 2. Ways to read X legitimately
| Option | Cost | What it gets | Verdict |
|---|---|---|---|
| Official X API, pay-per-use (the default for new developers since 2026-02) | $0.005 per post read, $0.010 per user lookup; capped at 3M post reads/month (docs.x.com pricing page, read 10-21) | The user-timeline endpoint returns @rotobot_ai's posts with exact `created_at`. 1,000 posts is about $5. Needs a developer account and app keys, which the **user** creates | **Recommended** if the account posts real picks |
| Official X API, free tier | The pricing page lists no free tier. Third-party 2026 write-ups say it was retired for new signups, or was write-only (no reads) | No timeline reads | Not usable |
| Legacy Basic $200/mo, Pro $5,000/mo | Existing subscribers only (secondary sources) | Timeline reads | Not available or not worth it |
| Official X MCP server [xdevplatform/xmcp](https://github.com/xdevplatform/xmcp) ([docs.x.com/tools/mcp](https://docs.x.com/tools/mcp)) | Same pay-per-use API charges. It wraps the API (OAuth 1.0a, runs locally, tools limited by `X_API_TOOL_ALLOWLIST`) | The same data as the API, callable from Claude | Fine as a convenience layer. It is **not** a free route. The third-party "X MCP Server" listings on mcp.so are not X's own |
| User's own X data export | Free | The user's **own** account data only, not other accounts' timelines | Does not answer this question |
| Manual paste | Free | The user copies post text **plus the status URL**. The tracker reads the exact post time from the status id (snowflake), so a hand-typed time is not trusted | **Fine for about 50-300 posts** |
| Scrapers, logged-out mirrors, scraping-API resellers | Varies | Violates X's ToS or bypasses the login wall | Not used, per the hard rule |

I pulled no posts. No free, ToS-compliant read route exists.

## 3. The tracker: `scripts/rotobot_tracker.py` (tests: `tests/test_rotobot_tracker.py`, 17 pass; full suite 55 pass)
- `parse POSTS -o picks.csv`: free text goes into the pick schema
  (`posted_at, sport, game, market, side, line, price, source_url` [+ `player`]).
  - Input: `.jsonl`/`.json` of `{posted_at|created_at, text, url}`, which is also the shape the X API returns.
    Or a `.txt` of blocks separated by `---`; the first line of each block is an ISO time or a status URL.
  - Formats handled: player props (`LeBron James o24.5 pts (-115)`, `X rebounds under 8.5 +105`), game
    totals, spreads and moneylines, with American, decimal or Kalshi-cent prices. Sport tags are stripped.
  - Posts with no parsable pick go to `*_unparsed.txt` for hand entry. Nothing is silently dropped.
- `grade picks.csv`: matches each pick to an ESPN game. Teams come from ESPN's keyless teams endpoint;
  players from box scores. Then it applies these rules:
  - **Excluded if `posted_at >= start`.** We take the earliest game in the window [posted-6h, posted+5d]
    and never slide a late pick onto the team's next game.
  - Unmatched picks are excluded, never guessed.
  - **Closing line**, in priority order:
    1. hand-entered `close_line`/`close_price`
    2. Kalshi candle mid before the start (via `kalshi_ticker`; checked live on KXNBAGAME-26OCT03MIATOR-MIA:
       0.455, which matches our stored 0.45/0.46)
    3. ESPN game-line `*_close`
    4. ESPN prop last price, flagged `close_quality=approx`. It is reported separately because its
       `last_updated` is post-kickoff.
  - **Metrics:**
    - ROI at the posted price, flat 1u
    - hit rate, and the breakeven hit rate at the posted prices
    - price CLV = `posted_dec × de-vigged close prob − 1`, only when the close is at the same line
    - line CLV in points otherwise
    - 95% CIs from a bootstrap over posting **days** (one card's picks are correlated); no CI with fewer than 2 days
  - A hand-graded `result` on an unmatched pick also needs a hand-entered `start_utc` before the post.
- `power`: the n-needed table below, computed and not hand-typed.
- Smoke-tested end to end on real ESPN NBA data: 4 made-up picks on 2025-11-04 CHA@NO in
  `data/rotobot/smoke/`. They are test inputs, not RotoBot picks. The late ML was excluded, and the
  total, spread and prop were settled against the real box score with real closes.
- **Data gap:** ESPN data on disk ends with NBA season 2026 (June 2026) and NFL 2025. Grading the current
  seasons needs `collect.py all --league nba --seasons 2027` and `--league nfl --seasons 2026` after the
  games finish. Props from ESPN have no true close.

## 4. What it takes to judge the record (`rotobot_tracker.py power`)
**On ROI** (two-sided 5%, 80% power, flat stakes):

| Price | 2% edge | 3% edge |
|---|---|---|
| -110 | 17,775 bets | 7,888 bets |
| +150 | 29,589 bets | 13,190 bets |
| +300 (parlay-ish) | 59,577 bets | 26,649 bets |

No tout's X feed will ever reach this. A good-looking ROI over a few hundred picks is noise.

**On CLV** the bar is far lower. CLV spread was measured from ESPN NBA 2025-26 opening-ML vs de-vigged
close (n=2,818): SD 0.137, robust SD 0.084. That gives about **140-370 picks to resolve a 2% mean CLV** and
**about 60-165 for 3%**. Those counts are an upper bound, because open-to-close is the widest horizon and
same-day posts move less.

So the test is: **are RotoBot's posted prices beaten by the close?** If not, no ROI over any sample size
means anything.

Two caveats:
- Props from ESPN only have an approximate close, so a props-heavy feed needs the Odds API or Kalshi for
  clean CLV.
- Parlays cannot be CLV-graded leg by leg without each leg's price, and their variance is huge. Grade
  straight picks only.

## 5. The ask for the user
1. **Look at https://x.com/rotobot_ai (about 1 minute).** Does it post concrete picks with a **line and a
   price**, and how many per week? If it is marketing or parlay screenshots without prices, stop: there is
   no record to grade.
2. If it does post priced picks, choose one:
   - **(a) X API pay-per-use** (recommended for more than about 200 posts). Create an X developer account
     and app yourself, add about $5-20 of credit, and put the bearer token in `sports/.env` as
     `X_BEARER_TOKEN=...`. Never paste it in chat. I would then add a read-only timeline pull for
     @rotobot_ai. Optionally run X's official xmcp server on the same key.
   - **(b) Manual paste** (free). For each post, paste the **status URL** on one line and the post text
     below it, with blocks separated by `---`, into `sports/data/rotobot/posts.txt`. The URL alone fixes
     the exact post time.
3. If the in-app "curated picks" are the real product, that feed is behind the $14.99/wk-$179.99/yr
   paywall. Grading it means logging each pick at the moment it appears, with line and price, which is a
   forward-only exercise. Backfilled app picks cannot be trusted for timing.
