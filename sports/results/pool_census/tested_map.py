"""TESTED / UNTESTED label per series family, with the citation (file:line) that carries the test. Hand-built from
FUND_BOOK 3d, LEARNINGS_2026-11, EDGES_AND_AGENTS, STRATEGY_v2 and notes files (grep index notes_ticker_index.json).
Order matters: first matching rule wins."""
import re
RULES = [
    (r"^KX(BTC|ETH|XRP|SOL|DOGE|HYPE|ZEC|NEAR|BNB)15M$", "SIBLING-DEAD", "STRATEGY_v2 l.103 KX*15M excluded (Q012/Q016/OT1-ext); latency-arb version untested, low prior (forum_strategy_scout l.43/80)"),
    (r"^KX(GOLD|SILVER|WTI|NATGAS|COPPER|BRENT)15M$", "SIBLING-DEAD", "Q016 commodities DEAD (EDGES_AND_AGENTS l.319); STRATEGY_v2 l.103"),
    (r"^KX(EURUSD|GBPUSD|USDJPY|USDCAD|AUDUSD)15M$", "UNTESTED", "no notes hit; 15-min up/down on Pyth FX"),
    (r"^KXINX15M$|^KXNASDAQ10015M$", "UNTESTED", "no notes hit (daily index tested: options_implied_tails, LRN l.269 index arm failed)"),
    (r"^KX(BTC|ETH|SOL|XRP|DOGE)D$|^KX(BTC|ETH|SOL)$", "TESTED", "Q012 hourly crypto DEAD (EA l.317); OT1 daily lead/weakened (LRN l.268-276); range ext dead (LRN l.235)"),
    (r"^KXINXU$|^KXNASDAQ100U$|^KXINX$|^KXNASDAQ100$", "TESTED", "options_implied_tails index arm failed (LRN l.272); K-FAV dead (FB l.328)"),
    (r"^KX(NBA|NFL|NCAAF|NCAAMB|NCAAWB|NHL|MLB|WNBA)GAME$", "TESTED", "in-game N0/M4/CBB-IG (FB l.231-238), longshots G2/G3 (FB l.331), xvenue M2 (FB l.346), openers (LRN l.155/334)"),
    (r"^KX(NBA|NFL|NCAAF|NCAAMB|NCAAWB|NHL|MLB|WNBA)(SPREAD|TOTAL)$", "TESTED", "Q004 / NBA live totals / early NFL totals (FB l.165, l.249, l.338); openers (LRN l.334)"),
    (r"^KX(NBA|NFL|NCAAF|NCAAMB)(1H|2H)", "TESTED", "derived_ladders calibrated at T-60 (LRN l.406, EA l.431)"),
    (r"^KXNFL(TD|RECYDS|RSHYDS|PASSYDS|REC|FIRSTTD|PASSTDS)$|^KXNBA(PTS|REB|AST|3PT|STL|BLK)", "TESTED", "props: Kalshi vs DK gap 0c (FB l.295); NFLUP DEAD (EA l.431-435); prop ladders paper"),
    (r"^KXMLB", "TESTED", "MLB ML/totals/bullpen rows (FB 3d 'Sports models'); season ended (60d window)"),
    (r"^KX(ATP|WTA|ITF|ITFW|ATPCHALLENGER|WTACHALLENGER)MATCH$|^KXATPSETWINNER$", "UNTESTED", "only first-listing staleness (kalshi_first_listing l.66-70) + rules flag (rules_quirks l.74)"),
    (r"^KX(UEFANL|INTLFRIENDLY|CONCACAFNL|EPL|UCL|UCLW|SERIEA|LALIGA|LALIGA2|BUNDESLIGA|LIGUE1|BRASILEIRO|MLS|LIGAMX|AFCON|WC|ARGPREMDIV|CLUBF)(GAME|TOTAL|ADVANCE|SPREAD|BTTS)?$", "TESTED", "soccer 1X2 S1-S3 (FB l.333), SOC-IG (FB l.239), vs Pinnacle (LRN l.149)"),
    (r"^KXUFC(FIGHT)$", "UNTESTED", "winner itself untested; U1 MOV used it as input -> DEAD (EA l.426)"),
    (r"^KXUFC", "TESTED", "U1 UFC MOV DEAD (EA l.426)"),
    (r"^KX(CS2|LOL|VALORANT|DOTA2)(GAME|MAP)$", "TESTED", "cs2_model DEAD (EA l.405); esports_pinnacle ESP1 forward"),
    (r"^KXE(SOCCER|BASKETBALL)GAME$", "UNTESTED", "no notes hit"),
    (r"^KX(ODI|T20|WT20|TEST)MATCH$|^KXIPL", "UNTESTED", "only rules quirk (cricket NR at $0.50 priced; rules_quirks l.128)"),
    (r"^KXPGATOUR$|^KXPGA", "TESTED", "golf tie quirk priced, -18% at ask (rules_quirks l.128)"),
    (r"^KXHIGH|^KXLOWT", "TESTED", "E1/E2/E4 live (FB 3a)"),
    (r"^KXTEMP", "TESTED", "HT1 DEAD (FB l.245), 8-station index (LRN l.214, nyc_hourly_dir.md)"),
    (r"^KXAAAGAS", "TESTED", "N6 DEAD (FB l.335); gas_daily_v2 in flight (pool is MM-collected)"),
    (r"^KXRT$", "TESTED", "RT reprices within minutes (LRN l.175)"),
    (r"^KXTSA", "TESTED", "TSAW lead / TSAD DEAD (EA l.428)"),
    (r"^KX(GOOG|ANTH|OPENAI|XAI|META)SHARE$|^KXTOKENUSE$|^KXOPENSOURCESHARE$", "TESTED", "ai_nowcast TOK1/2 DEAD (EA l.453); vercel_share in flight"),
    (r"^SENATE|^KXMIDTERM|^KX.*SENATE|^CONTROL|^KXCAELECTION", "ONE-OFF", "election one-off (election_needle_2026.md)"),
    (r"MENTION", "TESTED", "M3 mention NO sealed FAIL (FB l.329)"),
    (r"^KXIPO", "ONE-OFF", "one-off IPO"),
    (r"^KX(BALANCEPOWER|MAYOR|GOV|HOUSE|SENATE|PRES)|^(PRES|GOV|HOUSE)", "ONE-OFF", "election one-off (election_needle_2026.md)"),
]
def label(series):
    for pat, st, cite in RULES:
        if re.search(pat, series): return st, cite
    return "UNTESTED?", "no rule matched; check notes_ticker_index.json"
