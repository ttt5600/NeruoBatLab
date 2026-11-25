"""EV model for a first-season FutureEval entry. Inputs are cited; outputs are scenario arithmetic, not data.
Prize share ~ (sum of peer scores)^2 (Metaculus scores FAQ). At full coverage, prize ~= k * avg^2, where k is read off
the API-derived 'dollar value of an average score' table (Bowen1314/metaculus-bot RESEARCH.md, raw/bowen_research.md):
avg 10 = $1,679 / $1,287 / $979 (Fall25 / Spring26 / Summer26)  ->  k = 16.8 / 12.9 / 9.8 $ per avg^2.
"""
k_hist = {"fall25": 16.79, "spring26": 12.87, "summer26": 9.79}
# check the S^2 rule against the same table's avg-7 and avg-5 rows (should be k*49, k*25)
rows = {"fall25": (837, 431), "spring26": (639, 328), "summer26": (484, 248)}
for s, (a7, a5) in rows.items():
    print(f"{s}: k={k_hist[s]:.2f}  pred avg7 {k_hist[s]*49:.0f} vs table {a7};  pred avg5 {k_hist[s]*25:.0f} vs table {a5}")
# decline per season
r1 = k_hist["spring26"] / k_hist["fall25"]; r2 = k_hist["summer26"] / k_hist["spring26"]
print(f"k ratio per season: {r1:.3f}, {r2:.3f}")
# Spring 2027 is two seasons after Summer 2026. Scenarios for k: field keeps growing at the same rate (low),
# growth stalls + commercial-bot exclusion returns their share to hobbyists (high).
k_low = k_hist["summer26"] * r2 * r2
k_high = k_hist["summer26"] * 1.0
print(f"Spring-2027 k range: {k_low:.2f} .. {k_high:.2f}")
# avg-score scenarios for 'template + multi-source research + clipping + cross-lab median', with coverage.
# Anchors: Summer26 official-template bots with frontier models scored avg 5.9-9.8 (best 9.8, ranks 24-41 of 284);
# Spring26 5 runs/median template 10.0 best. nostreambot (open-source, 3 models median, 4+ sources) ~12.7/scored q.
scen = [("bad: outages/old model", 0.20, 4.0, 0.70),
        ("template-level", 0.40, 7.5, 0.90),
        ("good template+", 0.30, 10.0, 0.95),
        ("nostreambot-level", 0.10, 12.5, 0.95)]
for k, lab in [(k_low, "low k"), (k_high, "high k")]:
    ev = 0
    for name, p, avg, cov in scen:
        prize = k * (avg * cov) ** 2
        prize = prize if prize >= 50 else 0.0
        ev += p * prize
        print(f"  [{lab}] {name:22s} p={p:.2f} avg={avg:5.1f} cov={cov:.2f} -> ${prize:7.0f}")
    print(f"[{lab}] seasonal EV = ${ev:.0f}")
# MiniBench: mean take $18-32/round for >=90%-coverage bots (Jun-Sep 2026, 6 rounds); ~8 rounds per 4-month season
print("MiniBench per season: $%d-$%d" % (18 * 8, 32 * 8))
# API cost: top-15 winners ~$1.40/question, bottom half ~$0.50 (Fall25 survey via LW a82q6yd8zKpYk56cF);
# nostreambot $2.60/question. Questions: 300-400 seasonal + ~8x60 MiniBench.
for c in (0.5, 1.4):
    print(f"API cost at ${c}/q: seasonal 350q ${350*c:.0f}; MiniBench 480q ${480*c:.0f}")
