"""Rebuilt RC counts (new endpoint) vs Kalshi settlement for every settled KXTRUTHSOCIAL/KXTRUTHSOCIALD event.
Check 1: integer expiration_value -> exact equality.  Check 2 (all events): the rebuilt count satisfies the bracket of
every market that settled YES and none that settled NO (settlement-equivalent)."""
import json, re, sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
sys.path.insert(0, "scripts")
import q013_logger as C
NY = ZoneInfo("America/New_York")
_W = re.compile(r"begins at 12:00(?::00)? AM ET on (\w{3} \d{1,2}, \d{4}) and ends at 11:59(?::59)? PM ET on (\w{3} \d{1,2}, \d{4})")
ev = json.load(open("results/ops_q013_q014/kalshi_settled.json"))
rules = json.load(open("results/ops_q013_q014/kalshi_rules.json"))
posts = {}
for l in open("results/ops_q013_q014/rc_new_posts.jsonl"):
    p = json.loads(l); posts[p["id"]] = p
print("unique posts", len(posts), "deleted", sum(p["deleted"] for p in posts.values()),
      "platforms", {p["platform"] for p in posts.values()})
lo = min(p["post_ts"] for p in posts.values())
n_int = d_int = n_br = d_br = 0
rows = []
for e_t in sorted(ev):
    e = ev[e_t]
    g = _W.search(rules[e_t])
    d0 = datetime.strptime(g.group(1), "%b %d, %Y").replace(tzinfo=NY)
    d1 = datetime.strptime(g.group(2), "%b %d, %Y").replace(tzinfo=NY) + timedelta(days=1)
    ws, we = int(d0.timestamp()), int(d1.timestamp())
    assert ws >= lo, (e_t, ws, lo)
    close = min(C.pts(m["close_time"]) for m in e["markets"])
    end = min(we, close)           # early-closed events: count up to the close
    inw = [p for p in posts.values() if ws <= p["post_ts"] < end]
    c = sum(1 for p in inw if not p["deleted"]); cd = len(inw)
    yes = [m for m in e["markets"] if m["result"] == "yes"]
    no = [m for m in e["markets"] if m["result"] == "no"]
    ok_br = all(C.yes_set(m)(c) for m in yes) and not any(C.yes_set(m)(c) for m in no)
    ok_br_d = all(C.yes_set(m)(cd) for m in yes) and not any(C.yes_set(m)(cd) for m in no)
    xv = sorted({str(m["expiration_value"]).strip() for m in e["markets"]})
    eq = None
    if len(xv) == 1 and xv[0].isdigit():
        eq = int(xv[0]) == c; n_int += 1; d_int += (not eq)
    n_br += 1; d_br += (not ok_br)
    early = close < we - 3600
    rows.append((e_t, xv, c, cd, "EQ" if eq else ("NE" if eq is False else "-"), "br_ok" if ok_br else "BR_FAIL",
                 "br_ok_incl_del" if ok_br_d else "br_fail_incl_del", "early_close" if early else ""))
for r in rows:
    print(r)
print(f"integer expiration_value: {n_int} events, {d_int} disagree; bracket check: {n_br} events, {d_br} fail")
