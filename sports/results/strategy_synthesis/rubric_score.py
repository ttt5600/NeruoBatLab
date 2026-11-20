"""STRATEGY_v2 rubric applied to the Kalshi open-series universe (read-only; input = series_watch live snapshot).
Auto components only (R1 source, R2 sharper input, R3 neglect, R4 events/month, R5 fee type, R6 info order, penalties);
the top rows are then judged by hand in notes/STRATEGY_v2.md. Usage: python rubric_score.py SNAP.json.gz > out.csv"""
import gzip, json, re, sys, csv, datetime as _dt
NOW = _dt.datetime(2026, 11, 19, 21, 43, tzinfo=_dt.timezone.utc)   # snapshot time
sys.path.insert(0, "/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports/scripts")
from series_watch import DOMAINS, tested_note, FREQ_PER_MONTH, NEWS_SRC_RE, MENTION_RE

EXTRA_DEAD = [  # tested after series_watch's TESTED list was compiled, or covered by a FUND_BOOK 3d row
    r"KXNFLFG", r"KXCS2(GAME|MAP)", r"KXRT.*", r"KXFED.*", r"KXSNOWDAYS.*", r".*HOME$", r"KX30YMORT.*",
    r"KXMBA.*MORT.*", r"KXJBLU.*", r"KXSPRLVLDATE", r"KXTRUTHSOCIAL.*", r"KXAPPROVAL.*", r".*MENTION.*",
    r"KXCPI.*", r"KXGAS.*", r"KXPAYROLL.*", r"KXU3.*", r"KXJOBLESS.*", r"KXGDP$", r"KXPCE.*", r"KXECONSTAT.*",
    r"KXNBATOTAL|KXNBASPREAD|KXNFLTOTAL|KXNFLSPREAD|KXNCAAFTOTAL|KXNCAAFSPREAD|KXNCAAMBTOTAL|KXNCAAMBSPREAD",
    r"KXNHLTOTAL|KXNHLSPREAD", r"KXGOLF.*|KXPGA.*|KXLPGA.*", r"KXCRICKET.*|KXTESTMATCH|KXT20.*",
]
PAID_RE = re.compile(r"carbon arc|carbonarc|fiscal\.ai", re.I)
HEADLINE_SPORTS = re.compile(r"(GAME|MATCH|SPREAD|TOTAL|WINS?|CHAMP.*|MVP|PLAYOFF.*|TITLE|NEXTTEAM|ROY|COTY)$")
PLAYER_PROP = re.compile(r"KX(NFL|NBA|NHL|MLB)(TD|FIRSTTD|RSHYDS|RECYDS|REC|PASSYDS|PASSTDS|PTS|AST|REB|3PT|PRA|GOAL|"
                         r"FIRSTGOAL|STL|2D|3D|PASSATT|PASSCOMP|RSHATT|PASSINT|LONGREC|FFPTS)(PROT)?$")


def score(t, m, a):
    srcs = " ".join(f"{n} {u}" for n, u in m.get("sources") or [])
    text = f"{m.get('title','')} {a.get('event_title','')} {srcs} {a.get('rules_primary','')}".lower()
    cat = (m.get("category") or "").lower()
    st, note = tested_note(t)
    if st == "tested" or any(re.fullmatch(rx, t) for rx in EXTRA_DEAD):
        return None
    dom = "other"
    for name, rx, v1, v2 in DOMAINS:
        if re.search(rf"\b(?:{rx})\b", text):
            dom = name
            break
    freq = (m.get("frequency") or "").lower()
    import datetime as dt
    nc = a.get("next_close") or "2099-01-01T00:00:00Z"
    days_to_close = (dt.datetime.fromisoformat(nc.replace("Z", "+00:00")) - NOW).days
    long_dated = days_to_close > 30              # season-settled / long-dated: one correlated draw (LRN l.188)
    epm = FREQ_PER_MONTH.get(freq)
    if epm is None:                              # custom/one_off: open events closing within ~a week ~ weekly cadence
        epm = 0.5 if long_dated else min(a.get("n_ev", 0) * 4, 60)
    sports = cat == "sports"
    headline = sports and bool(HEADLINE_SPORTS.search(t))
    prop = sports and bool(PLAYER_PROP.fullmatch(t))
    # R1 free settlement source the value derives from (0-2)
    if PAID_RE.search(srcs):
        r1 = 0
    elif sports or dom in ("weather", "geo/space", "transport/energy", "macro", "health", "markets/crypto"):
        r1 = 2
    elif dom == "charts/media" or (srcs and not NEWS_SRC_RE.search(srcs)):
        r1 = 1
    else:
        r1 = 0
    # R2 independent sharper input that the price may not fully use (0-2)
    if dom in ("weather", "geo/space"):
        r2 = 2
    elif sports and not headline and not prop and not long_dated and epm >= 4:
        r2 = 2          # derived/secondary ladder of a sharp sportsbook main line (KFG1 shape)
    elif dom in ("macro", "markets/crypto", "transport/energy") or prop:
        r2 = 1
    else:
        r2 = 0
    # R3 neglect: share of ladder rungs quoted 2-15c with zero 24h volume (0-2)
    lad = a.get("n_ladder_mk", 0)
    sf = a.get("n_tail_stale", 0) / lad if lad else 0.0
    r3 = 2 if sf >= 0.30 else 1 if sf >= 0.10 else 0
    # R4 power: independent events per month (0-2)
    r4 = 2 if epm >= 20 else 1 if epm >= 4 else 0
    # R5 cost: maker-free fee schedule (0-1)
    r5 = 1 if m.get("fee_type") == "quadratic" else 0
    # R6 who sees settlement data first: public, simultaneous release (1) vs paid / insiders / mentions / politics (0)
    r6 = 0 if (PAID_RE.search(srcs) or cat in ("politics", "elections", "mentions") or MENTION_RE.search(text)) else 1
    pen = 0
    if headline or cat in ("politics", "elections", "mentions"):
        pen -= 3        # crowded: every headline-line test died 'priced' (FUND_BOOK 3d)
    if prop:
        pen -= 2        # Kalshi vs DK player props median gap 0c (FUND_BOOK l.295)
    if st == "related":
        pen -= 1
    tot = r1 + r2 + r3 + r4 + r5 + r6 + pen
    return dict(ticker=t, title=m.get("title", "")[:60], cat=cat, dom=dom, epm=round(epm, 1), R1=r1, R2=r2, R3=r3,
                R4=r4, R5=r5, R6=r6, pen=pen, score=tot, n_ev=a.get("n_ev"), max_mk=a.get("max_mk"),
                vol=int(a.get("vol", 0)), vol24=int(a.get("vol24", 0)), stale=f"{a.get('n_tail_stale',0)}/{lad}",
                related=note, src=srcs[:80])


def main():
    d = json.load(gzip.open(sys.argv[1]))
    rows = [r for t, a in d["active"].items() if (r := score(t, d["series"].get(t, {}), a))]
    rows.sort(key=lambda r: (-r["score"], -r["vol"]))
    w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    print(f"# scored {len(rows)} of {len(d['active'])} active series (rest = tested/dead)", file=sys.stderr)


if __name__ == "__main__":
    main()
