"""Census of Kalshi derived/secondary game-level sports series. Read-only public GETs. -> census.json"""
import sys, json, re, statistics as st
from pathlib import Path
sys.path.insert(0, "scripts")
from kx_throttle import kget
OUT = Path("results/derived_ladders")
S = json.load(open(OUT / "series_sports.json"))
LEAG = r"^KX(NFL|NCAAF|NBA|NCAAMB|NCAAWB|WNBA|NHL|MLB|EPL|LALIGA|SERIEA|BUNDESLIGA|LIGUE1|MLS|UCL)"
MAIN = {f"KX{l}{k}" for l in "NFL NCAAF NBA NCAAMB NCAAWB WNBA NHL MLB EPL LALIGA SERIEA BUNDESLIGA LIGUE1 MLS UCL".split() for k in ["GAME", "SPREAD", "TOTAL"]}
DONE = {"KXNFLFG", "KXNBAPTS", "KXNBAREB", "KXNBAAST"}
EXCL = re.compile(r"WINS|AWARD|POTY|POY|OTY|MVP|DRAFT|SEASON|COMBINE|CAREER|NEXT|RECORD|STREAK|PLAYOFF|QUAL|REG$|TOP|RANK|POLL|LAST$|LOSE|ALLPRO|PROBOWL|VIEWER|SELLOUT|DELAY|LOCATION|EXACTWINS|1STTEAM|2NDTEAM|3RDTEAM|AWARD|CHAMP|FINALIST|ROUND|RO\d|MATCHUP|COMPETE|PREPACK|ESCALATOR|ALLGAMES|SUMMER|ASGAME|ASG|HRDERBY|CONTEST|DEBUT|RETURN|FRANCHISE|GOVERNOR|DUNK|SLAMDUNK|SKILLS|RULE|SIGN|TEAMMATE|PLAYTOGETHER|STOCK|EXEC|LAWSUIT|RAISE|OWNER|FF|WEEK|H2HWINS|MOSTWINS|BEST|WORST|TROPH|MANAGER|UNDEFEATED|CUP|NIT|BIT|CONF|BIGTEN|BIG12|BIGEAST|ACC|SEC|PAC12|WAC|NEC|MEAC|SOCON|HOR|BSKY|BWEST|HL$|IVY|COTY|MOP|KENPOM|APCHURN|BOWL|OPENING|SS$|TRIPLE|WORLD|SEED|500|100LOSS|CBA|JOIN|LEAVE|PORTNOY|7FIGS|ALLTEAM|ALLDEFENSE|FIRSTTEAM|SECONDTEAM|SSTARS|4NATIONS|PRES$|HART|NORRIS|VEZINA|CALDER|RICHARD|ROSS|LINDSAY|OPENER|ARENA|GM$|XMAS|LASVEGAS|SEATTLE|RELOC|LOTTERY|FIRSTPICK|POTM|PLAYEROTM|PITCHEROTM|PLAYEROTW|STAGEOFELIM|53MAN|ROLE|DEPTH|FIRSTSTART|START$|STARTER|STARTINGLINEUP|CELEB|REDZONE|T100|TESTIMAGE|ENDSTREAK|WHATTEND|APOLOGY|PARTNERSHIP|PLAYERDEAL|TOJOIN|STATAVG|PTSRECORD|PLAYCRON|PLAYMESSI|TREBLE|H2HFINISH|BOTTOM|1SEED|MIMP|SIXTH|CLUTCH|DPOTM|POTM|EOTY|OPOY|DPOY|CPOTY|PROOTY|WPMOTY|FIRSTWIN|LEASTPENAL|BOTH$|TIES?$|BLOWOUT|LARGELEAD|LARGESTLEAD|LEADCHANGE|30PTSTREAK|ERA$|GG$|SB$|WA$|HA$|SEEDWIN|BBREG|BBFINAL")
cands = sorted({x["ticker"] for x in S if re.match(LEAG, x["ticker"]) and x["ticker"] not in MAIN | DONE and not EXCL.search(x["ticker"][2:]) and x.get("frequency") != "annual"})
meta = {x["ticker"]: x for x in S}
print("candidates", len(cands), flush=True)
def pages(path, s, cap):
    out, cur = [], None
    for _ in range(cap):
        p = {"series_ticker": s, "limit": 1000}
        if path == "/markets": p["status"] = "settled"
        if cur: p["cursor"] = cur
        j = kget(path, p) or {}
        out += j.get("markets") or []
        cur = j.get("cursor")
        if not cur or not j.get("markets"): break
    return out, bool(cur)
res = {}
done = json.load(open(OUT / "census.json")) if (OUT / "census.json").exists() else {}
for i, s in enumerate(cands):
    if s in done: res[s] = done[s]; continue
    a, ca = pages("/markets", s, 8)
    h, ch = pages("/historical/markets", s, 8)
    ms = {m["ticker"]: m for m in h + a}
    ms = [m for m in ms.values() if m.get("result") in ("yes", "no")]
    def vol(m):
        for k in ("volume_fp", "volume"):
            if m.get(k) is not None:
                try: return float(m[k])
                except Exception: pass
        return 0.0
    v = [vol(m) for m in ms]
    ev = {m["event_ticker"] for m in ms}
    ct = sorted(m.get("close_time") or "" for m in ms)
    res[s] = {"title": meta[s].get("title"), "fee": f'{meta[s].get("fee_type")} {meta[s].get("fee_multiplier")}', "n_mk": len(ms), "n_ev": len(ev),
              "vol_tot": sum(v), "vol_med": st.median(v) if v else 0, "mk_per_ev": len(ms) / max(1, len(ev)), "first": ct[0][:10] if ct else "", "last": ct[-1][:10] if ct else "",
              "capped": ca or ch, "sample_title": (ms[0].get("title") if ms else ""), "sample_sub": (ms[0].get("yes_sub_title") if ms else "")}
    if i % 10 == 0 or len(ms): print(i, s, res[s]["n_ev"], res[s]["n_mk"], int(res[s]["vol_tot"]), flush=True)
    json.dump(res, open(OUT / "census.json", "w"))
print("done")
