import json, requests
API = "https://api.elections.kalshi.com/trade-api/v2"
def m(t):
    r = requests.get(f"{API}/markets/{t}", timeout=20).json().get("market", {})
    return {k: r.get(k) for k in ("ticker", "status", "result", "expiration_value", "last_price_dollars", "settlement_value_dollars")}
print(m("KXTOKENUSE-26NOV23-T238"))
log = [json.loads(l) for l in open("/Users/jonathanwang/.claude/quant-kalshi/data/exp_budget.jsonl")]
sub = {r["cid"] if "cid" in r else r["body"]["client_order_id"]: r for r in log if r["event"] == "submit"}
for r in log:
    if r["event"] == "response" and r.get("strategy") == "KFG1":
        s = sub.get(r["cid"], {}); b = s.get("body", {})
        o = r["out"]; mm = m(b["ticker"])
        yes = b["side"] == "bid"
        n = float(o["fill_count"]); px = float(o["average_fill_price"]); fee = float(o["average_fee_paid"])
        cost_side = px if yes else 1 - px
        win = (mm["result"] == "yes") == yes if mm["result"] in ("yes", "no") else None
        pnl = None if win is None else n * ((1 if win else 0) - cost_side) - n * fee
        print(b["ticker"], "YES" if yes else "NO", n, "px", px, "fee/ct", fee, "edge", s.get("edge"), "->", mm["status"], mm["result"], mm["expiration_value"], "pnl", None if pnl is None else round(pnl, 2))
