"""pm_full_census: assign one category per event (priority order), global Gamma."""
import pandas as pd, re
OUT = "results/pm_full_census/"
RULES = [  # (category, tag slugs, title regex)
 ("crypto_updown", {"up-or-down"}, r"up or down"),
 ("weather", {"daily-temperature","highest-temperature","lowest-temperature","hurricanes","precipitation"}, r"temperature|rainfall|hurricane|hottest|snow"),
 ("mentions", {"mention-markets","mentions"}, r"\bsay\b|mention"),
 ("tweets", {"tweets-markets"}, r"tweets?"),
 ("earnings", {"earnings-calls-x"}, r"earnings|beat .*estimate|quarterly (revenue|eps)"),
 ("fed_macro", {"fed-rates","fed","fomc","Global-Rates","macro-indicators","cpi-release","inflation","economy","economic-policy","jobs-report","gdp"}, r"\bfed\b|interest rate|cpi|inflation|unemployment|gdp|payroll"),
 ("crypto_price", {"crypto-prices","hit-price","bitcoin","ethereum","solana","crypto"}, r"bitcoin|ethereum|solana|xrp|\bbtc\b|\beth\b"),
 ("equities_commod", {"stocks","equities","pre-market","gold","oil","commodities","daily-close","ipos","ipo","finance"}, r"close above|stock|s&p|nasdaq|crude|gold|ipo"),
 ("esports", {"esports","counter-strike-2","league-of-legends","dota-2","valorant"}, r"counter-strike|lol:|dota|valorant"),
 ("sports_game", set(), None),  # filled below: sports with a game-like title
 ("sports_future", {"sports","games","soccer","basketball","nba","nfl","nhl","mlb","tennis","ufc","f1","cricket","ncaa","golf","boxing","table-tennis","cfb","chess"}, None),
 ("elections", {"elections","world-elections","global-elections","us-presidential-election","2028-election","primaries","main-election"}, r"election|primary|nominee"),
 ("geopolitics", {"geopolitics","world","iran","israel","ukraine","middle-east","taiwan","venezuela","diplomacy-ceasefire","peace-deal"}, r"ceasefire|invade|strike|war\b"),
 ("tech_ai", {"tech","ai","big-tech","anthropic","openai","claude","spacex","space","science","nvidia"}, r"\bai\b|openai|gpt|gemini|model"),
 ("culture", {"pop-culture","movies","tv","netflix","music","awards","oscars","box-office","celebrities","game-awards","youtube","spotify"}, r"oscar|grammy|box office|album|song|billboard"),
 ("politics_other", {"politics","trump","united-states","us-politics"}, None),
]
def cat(row):
    t = set(row.tslugs.split("|")); ti = (row.title or "").lower()
    if re.search(r"crude|wti|brent|gold \(|silver|natural gas", ti): return "equities_commod"
    if re.search(r"nominee|presidential election|election winner", ti): return "elections"
    for c, tags, rx in RULES:
        if c == "sports_game":
            if t & RULES[10][1] and re.search(r"\bvs\.?\b| @ |\bv\b", ti): return "sports_game"
            continue
        if t & tags or (rx and c not in ("sports_future",) and re.search(rx, ti) and c in ("crypto_updown","weather","mentions","earnings")):
            return c
    return "other"
E = pd.read_parquet(OUT + "gamma_events.parquet")
E["cat"] = E.apply(cat, axis=1)
E.to_parquet(OUT + "gamma_events_cat.parquet")
g = E.groupby("cat").agg(events=("eid","size"), mkts=("n_mkts","sum"), vol24=("vol24","sum"), vol1wk=("vol1wk","sum"), liq=("liq","sum")).sort_values("vol1wk", ascending=False)
print(g.round(0).to_string())
for c in ["other", "politics_other", "mentions", "tweets"]:
    print("==", c); print(E[E.cat == c].nlargest(12, "vol1wk")[["title","vol1wk","tslugs"]].to_string()[:2500])
