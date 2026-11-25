"""boost_hedge part A valuations (arithmetic only). Quotes = sampler tops at 02:57-02:59Z 11-25
(results/boost_hedge/xv/samples_20261125.jsonl, latest s2 row per game). Recompute at execution."""
import json, math
from boost_math import kfee, profit_boost_lock

out = {}
def gfee(N, p):  # Gemini taker 0.07*C*P*(1-P), ceil to cent (rounding unverified -> conservative)
    return math.ceil(round(0.07 * N * p * (1 - p) * 100, 6)) / 100

# (1) Gemini College & Pro Football Trade Match, tier 1: $250 eligible volume at 0.20-0.80 -> $30 Predict Cash (SR)
#     PHI-DAL (Thu 11-26): Gemini DAL ask .80 (bid ~.79), Kalshi PHI ask .21.  KC-BUF: Gemini BUF .59 / Kalshi KC .42.
def trade_match(vol, pg, maker, ak):
    N = math.ceil(vol / pg)                      # Gemini contracts so that volume >= vol
    g_cost = N * pg + (0 if maker else gfee(N, pg))
    rebate = 0.30 * 0.07 * N * pg * (1 - pg) if maker and 0.20 <= pg <= 0.80 else 0.0
    k_cost = N * ak + kfee(N, ak)
    lock_cost = g_cost + k_cost - rebate - N       # pays N either way
    return {"N": N, "gemini_cash": round(g_cost, 2), "kalshi_cash": round(k_cost, 2), "net_cost": round(lock_cost, 2),
            "rebate": round(rebate, 2)}

def sr_credit_lock(B, q, a):                     # SR credit B at venue price q, hedge opposite on Kalshi at ask a
    N = B / q
    M = math.floor(N)
    f = kfee(M, a)
    return round(min(N - M * a - f, M - M * a - f), 2), round(M * a + f, 2)

pc_lock, pc_cap = sr_credit_lock(30, 0.80, 0.21)
pc_lock = round(pc_lock - gfee(30 / 0.80, 0.80), 2)   # critic 03:25Z: Gemini taker fee on the Predict Cash buy (~$0.42)
for name, pg, maker, ak in (("DAL@.79 maker / PHI K .21", 0.79, True, 0.21), ("DAL@.80 taker / PHI K .21", 0.80, False, 0.21),
                            ("BUF@.58 maker / KC K .42", 0.58, True, 0.42), ("BUF@.59 taker / KC K .42", 0.59, False, 0.42)):
    t = trade_match(250, pg, maker, ak)
    t["net_locked_tier1"] = round(pc_lock - t["net_cost"], 2)
    t["capital"] = round(t["gemini_cash"] + t["kalshi_cash"] + pc_cap, 2)
    out[f"gemini_trade_match_t1 {name}"] = t
t2 = trade_match(1000, 0.79, True, 0.21)
pc2, cap2 = sr_credit_lock(110, 0.80, 0.21)
pc2 = round(pc2 - gfee(110 / 0.80, 0.80), 2)
out["gemini_trade_match_t2 (cum $1,000 -> $110) DAL@.79 maker"] = {**t2, "net_locked": round(pc2 - t2["net_cost"], 2)}
t2t = trade_match(1000, 0.80, False, 0.21)
out["gemini_trade_match_t2 (cum $1,000 -> $110) DAL@.80 taker"] = {**t2t, "net_locked": round(pc2 - t2t["net_cost"], 2)}
# unhedged alternative: genuine positions at fair prices, maker (fee 0): EV = PC value (hedged conversion) - 0; max loss = stake
out["gemini_trade_match_t1_unhedged_maker"] = {"EV": pc_lock, "max_loss": 250.0}
out["predict_cash_30_lock (DAL .80 Gemini, PHI .21 Kalshi)"] = {"lock": pc_lock, "kalshi_capital": pc_cap}

# (2) PM US Tradesgiving Daily Trader: $X credit/day (new users $5 with $10 min; existing amount in-app). Credit is a loss
#     absorber (withdraw-promo.md) -> unhedged EV ~= credit; hedged lock ratio from promo_verify_1124 ($25 -> $17.06 = 0.68).
#     Qualifier hedged: NYG-IND (Sun) PM NYG ask .745 + Kalshi IND .25, 14 contracts.
N = 14
q_cost = N * 0.745 + 0.095 * N * 0.745 * 0.255 + N * 0.25 + kfee(N, 0.25) - N
B, q = 5.0, 0.16
Nh = math.floor(B * (1 - q) / q)                 # Kalshi contracts on the favourite
pm_fee = 0.095 * (B / q) * q * (1 - q)
harvest = round(Nh * (q - 0.01) - kfee(Nh, 1 - q + 0.01) - pm_fee, 2)
out["pmus_tradesgiving_per_day"] = {"credit_assumed": 5.0, "qualifier_hedged_cost": round(q_cost, 2),
    "credit_hedged_lock": harvest, "kalshi_capital_for_harvest": round(Nh * (1 - q + 0.01), 2),
    "unhedged_EV_per_day": 5.0, "hedged_lock_per_day": round(harvest - q_cost, 2), "days_left_11_25_to_11_30": 6}

# (3) OG Weekly Trading Streak week 1 (50% profit boost, extra capped $10, contracts < .80), boost on q=.79 vs Kalshi .22
S = 75.0
r = profit_boost_lock(S, 0.79, 0.5, 0.22, venue_fee=round(0.07 * (S / 0.79) * 0.79 * 0.21, 2))  # OG taker fee
out["og_streak_w1_boost_S75_q.79"] = r
r2 = profit_boost_lock(150.0, 0.79, 0.5, 0.22, venue_fee=round(0.07 * (150 / 0.79) * 0.79 * 0.21, 2))
out["og_streak_w2_boost_S150_q.79 (cap $20)"] = r2
out["og_streak_qualifiers_per_week"] = {"two_trades_usd": 20, "max_loss": 20.0,
    "EV": round(-2 * (0.07 * (10 / 0.5) * 0.25), 2), "note": "unhedged (bona fide market risk clause)"}
print(json.dumps(out, indent=1))
json.dump(out, open("value_items_out.json", "w"), indent=1)
