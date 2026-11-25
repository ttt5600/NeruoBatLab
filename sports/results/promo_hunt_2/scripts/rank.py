"""promo_hunt_2 ranking arithmetic. Every input is a quoted term (see notes/promo_hunt_2.md) or a labelled ESTIMATE."""
import json, sys
# Kraken official BTC draw odds (kraken.com/legal/promotions s.5)
draw = {5: .33, 10: .32, 15: .15, 25: .12, 50: .06, 100: .02}
kraken_draw_ev = sum(k * p for k, p in draw.items())
rows = []
def add(name, reward, costs, max_loss, capital, minutes, src, note=""):
    net = reward - sum(costs.values())
    rows.append(dict(offer=name, reward=reward, costs=costs, net=round(net, 2), max_loss=round(max_loss, 2),
                     capital=capital, minutes=minutes, net_per_hour=round(net / (minutes / 60), 1), src=src, note=note))
# 1 Kraken via Rakuten: $150 cash back; ONE trade >= $200 non-stablecoin. Buy instant (1% fee, OFF) + est 0.5% spread,
#   sell on Kraken Pro maker 0.40% (OFF Tier 1). Max loss if cash back never tracks: fees + 3% adverse move (ESTIMATE, sell same hour).
cap = 205.0
k_fees = {"instant_buy_1pct": 0.01 * cap, "spread_est_0.5pct": 0.005 * cap, "pro_maker_sell_0.4pct": 0.004 * cap}
add("Kraken via Rakuten ($150)", 150.0, k_fees, sum(k_fees.values()) + 0.03 * cap, cap, 35,
    "rakuten.com/shop/kraken; kraken.com/legal/promotions; kraken.com/features/fee-schedule (snip)",
    f"possible extra BTC draw EV {kraken_draw_ev:.2f} if Rakuten counts as 'participating partner' (unverified, not counted)")
# 2 eToro via Rakuten: $50; trade >= $100; ETF commission 0 (OFF); withdrawal $5 worst case (OFF global page); ETF spread est $0.02
e_fees = {"withdrawal_fee_worst": 5.0, "etf_spread_est": 0.02}
add("eToro via Rakuten ($50)", 50.0, e_fees, 5.0 + 0.02 + 0.005 * 101, 101.0, 30,
    "rakuten.com/shop/etoro; etoro.com/en-us/trading/live-states; etoro.com/trading/fees")
# 3 Coinbase: Rakuten $12.50 (first trade > $50) + Coinbase new-user reward floor $10 (range $10-$2,000, snippet). Round-trip cost est 4% of $51.
c_fees = {"round_trip_est_4pct": 0.04 * 51}
add("Coinbase new-user + Rakuten ($22.50 floor)", 22.5, c_fees, 0.04 * 51 + 0.05 * 51, 51.0, 25,
    "rakuten.com/shop/coinbase; help.coinbase.com new-customer-incentive (snippet)", "ONLY if user has never had Coinbase")
# 4 Acorns ACORNS20: $20 Reward Shares; Bronze $4/mo (snip, sign-ups after 2026-08-16); assume 2 billed months.
a_fees = {"subscription_2mo": 8.0}
add("Acorns ACORNS20 ($20 shares)", 20.0, a_fees, 8.0 + 4.0, 25.0, 30,
    "acorns.com/acorns-20-bonus-investment-promotion-terms; acorns.com program agreement (snip)", "max loss assumes bonus denied + a 3rd month billed")
# 5 MoneyGram via Rakuten: $15, first online transfer >= $50; first online transfer fee $0 for new customers (snip).
add("MoneyGram via Rakuten ($15)", 15.0, {"fee_first_transfer": 0.0}, 5.0, 50.0, 20,
    "rakuten.com/shop/moneygram; moneygram.com send-money-online FAQ (snip)", "needs a real recipient; max loss = a fee if $0 promo not applied (est $5)")
# 6 Western Union via Rakuten: up to $10, first transfer >= $100; fee est $5.
add("Western Union via Rakuten (<= $10)", 10.0, {"fee_est": 5.0}, 5.0, 100.0, 20, "rakuten.com/shop/westernunion", "fee UNVERIFIED")
rows.sort(key=lambda r: -r["net_per_hour"])
out = dict(kraken_draw_ev=round(kraken_draw_ev, 2), rows=rows,
           assumptions="ESTIMATES: Kraken spread 0.5%, 3% adverse move cap, Coinbase 4% round trip, ETF spread $0.02, WU fee $5, MoneyGram fallback fee $5, Acorns 2 billed months")
json.dump(out, open(sys.argv[1], "w"), indent=1)
for r in rows:
    print(f"{r['offer']:<45} net {r['net']:>7.2f}  maxloss {r['max_loss']:>6.2f}  cap {r['capital']:>6.0f}  min {r['minutes']:>3}  $/h {r['net_per_hour']:>6.1f}")
print("kraken draw EV", round(kraken_draw_ev, 2))
