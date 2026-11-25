"""Hand-written keyword classifier for the tournament questions logged by a public bot (m-ad/claude_forecasting_auto).
Classes:
  OURS   = resolvable from an official/free series the fund ALREADY pulls (NWS/NOAA weather+precip+tides/buoys, AAA gas,
           US macro releases BEA/BLS/EIA/USDA, PortWatch/AIS Hormuz, crypto/stock/commodity closes we log, TSA, Fed Reg)
  SERIES = resolvable from some official data series or market price, but not a pipeline we run (foreign CPI, disease
           surveillance counts, foreign FX refs, election-commission tallies, YouTube counts ...)
  EVENT  = needs news judgment (politics, conflict, courts, approvals, sports results, company announcements)
Rules are keyword-based on title (or slug); every row is printed so a human can audit misclassifications."""
import json, sys, collections
BASE = "/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports/results/metaculus_bot_tourney/"
rows = [json.loads(l) for l in open(BASE + "raw/mad_meta.jsonl")]
OURS = ["aaa-national-average", "gasoline", "precipitation-san-francisco", "northern-sierra", "water-level", "mllw",
        "ndbc-buoy", "wave-height", "real-gdp-growth-bea", "us-ppi", "producer-prices", "core-pce", "eia-report",
        "brent", "bitcoin", "btc", "hormuz", "ais-visible", "nasdaq", "nvda", "stock-close", "closing-share-price",
        "closing-price", "close-friday", "kospi", "ibovespa", "mstr", "crwv", "amazon-s-stock", "spcx", "anthropic-ant",
        "crypto-fear-greed", "net-outlays-us-government", "usda", "turkey-poults", "wheat-net-export", "hurricane",
        "named-storms", "storms-uk", "respiratory-specimens-tested-influenza-u-s", "measles-cases-u-s", "u-s-measles",
        "seasonally-adjusted-month-month", "eu-aggregate-gas-storage", "egg", "graded-loose-white-large", "won-close",
        "eur-pln", "bcra", "usd-ars", "tsa", "federal-register"]
SERIES = ["cpi", "inflation", "gdp", "rosstat", "dengue", "measles", "ebola", "influenza", "infogripe", "srag",
          "screwworm", "west-nile", "cases", "advance-votes", "postal-ballots", "subscribers", "viewers", "rating",
          "posts-donald-trump", "drones", "country-risk", "revenue", "net-profit", "guidance", "domestic-corporate-goods",
          "oil-market-report", "approval", "dairy-herds", "new-forecasters", "applied", "flash-estimate", "deposits",
          "uk-gdp", "percentage-points-okta", "highest", "bond-close", "corporate-goods", "first-place-votes",
          "patients", "federal-units", "brazil-s-27-state-capitals", "japanese-educational", "prefectures", "dabom",
          "fifth-circuit-news", "storms", "black-friday", "deepest-percentage", "financial-stability-report",
          "math-repository", "un-member-states-vote", "meps-vote", "how-many", "how-much", "which-band", "value-"]
def cls(s):
    k = (s.get("slug") or "").lower()
    if any(w in k for w in OURS): return "OURS"
    if any(w in k for w in SERIES): return "SERIES"
    return "EVENT"
c = collections.Counter(); ct = collections.Counter()
for r in rows:
    r["cls"] = cls(r); c[r["cls"]] += 1; ct[(r.get("tourn"), r["cls"])] += 1
    print(f'{r["cls"]:6s} {str(r.get("tourn")):22s} {str(r.get("type")):9s} {r["slug"][11:80]}')
n = len(rows)
print("\nTOTAL", n, {k: f"{v} ({v/n:.0%})" for k, v in c.items()})
print("by tournament:", dict(ct))
# scores by class (bot's own official spot peer score; descriptive only)
for k in ("OURS", "SERIES", "EVENT"):
    sc = [r["score"] for r in rows if r["cls"] == k and isinstance(r.get("score"), (int, float))]
    if sc: print(k, "n_scored", len(sc), "mean peer score", round(sum(sc)/len(sc), 1))
