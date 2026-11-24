"""Promo net-$ and max-loss arithmetic (promo_verify_1124, 2026-11-24).
Assumptions (stated, not market quotes): venue price for a side = Kalshi price of that side; Kalshi hedge of the
opposite side costs (1 - p + 0.01) per contract (1c spread); Kalshi taker fee = ceil_cent(0.07*C*x*(1-x)).
Fees per venue from official pages read 2026-11-24 (see notes/promo_verify_1124.md)."""
import math
def kfee(C, x): return math.ceil(round(0.07*C*x*(1-x)*100, 6))/100
def hedge(C, p):  # buy C opposite contracts on Kalshi when venue side is priced p
    x = round(1-p+0.01, 4); return C*x + kfee(C, x)
out = {}
# Fanatics Offer 1: daily first cash buy <=0.90, FanCash = $ volume up to $10; fee table $0.016 @0.90, ~$0.0144 @0.94
q_n = 12; q_cost = q_n*0.90 + q_n*0.016                     # $10.80 volume -> $10 FanCash
q_lock = q_n - q_cost - hedge(q_n, 0.90)                     # hedged qualifier
fc_n = math.floor(10/(0.94+0.0144)); fc_lock = fc_n - hedge(fc_n, 0.94)   # FanCash on .94 fav, SR (payouts -> cash)
out['fanatics_day'] = dict(qual_hedged=round(q_lock,2), qual_unhedged_EV=round(-q_n*0.016-q_n*0.005,2),
    qual_maxloss_unhedged=round(q_cost,2), fancash_lock=round(fc_lock,2), fc_contracts=fc_n,
    day_hedged=round(q_lock+fc_lock,2), x10_hedged=round(10*(q_lock+fc_lock),2),
    x10_unhedged_EV=round(10*(-q_n*0.021 + fc_n*0.94),2), kalshi_cap_per_day=round(hedge(q_n,0.90)+hedge(fc_n,0.94),2))
# Gemini: $20 maker buy at .80 unhedged (fee 0 to 12/31) -> $40 Predict Cash SR, 30d; use on .94 fav taker, hedge Kalshi
g_n = math.floor(40/(0.94+0.07*0.94*0.06)); g_lock = g_n - hedge(g_n, 0.94) - math.ceil(0.07*g_n*0.94*0.06*100)/100*0
out['gemini'] = dict(credit_contracts=g_n, credit_lock=round(g_lock,2), qual_EV=0.0, qual_maxloss=20.0,
    total_EV=round(g_lock,2), worst_case_if_credit_paid=round(g_lock-20,2))
# ProphetX: $40 own funds opening trades, hedged at .80 fav; 2% of net winnings per market at settlement
px_n = 50; px_k = hedge(px_n, 0.80)
px_q = min(px_n - 0.02*(px_n-40) - 40 - px_k, px_n - 40 - px_k)
c_n = math.floor(10/0.85); c_k = hedge(c_n, 0.85)
c_lock = min(c_n - 0.02*(c_n-10) - c_k, c_n - c_k)           # per $10 incentive (collateral, SR implied)
out['prophetx'] = dict(qual_hedged=round(px_q,2), per10_lock=round(c_lock,2), total=round(px_q+4*c_lock,2),
    qual_maxloss_unhedged=40.0, kalshi_cap_qual=round(px_k,2))
# Novig: 5 x $10 Deposit Credits at <=0.60, SR explicit; Novig fee assumed $0 pregame (affiliate only)
n_n = math.floor(10/0.60); n_lock = n_n - hedge(n_n, 0.60)
out['novig'] = dict(per_credit_lock=round(n_lock,2), total=round(5*n_lock,2), kalshi_cap_per_credit=round(hedge(n_n,0.60),2))
# Polymarket US: $25 Bonus absorbs first $25 of losses on ANY trade; win -> profit to cash, bonus recycles.
# Free roll: $25 cash on side priced q as maker (fee 0). EV/round = 25(1-q); continues w.p. q -> total EV ~ 25 if no expiry.
for q in (0.20, 0.35, 0.50):
    out[f'pmus_freeroll_q{q}'] = dict(EV_one_round=round(25*(1-q),2), EV_unlimited_rounds=25.0,
        p_profit_round1=q, profit_if_win=round(25/q-25,2), maxloss_if_docs_hold=0.0)
q = 0.16; N = 25*(1-q)/q; pm_fee = 0.0695*(25/q)*q*(1-q)
out['pmus_hedged_q0.16'] = dict(kalshi_N=round(N,1), lock=round(N*(q-0.01) - kfee(round(N), 1-q+0.01) - pm_fee,2),
    kalshi_cap=round(N*(1-q+0.01),2))
# OG.com: $25 qual (<.80, settle <=7d), $25 Bonus (lowers cost -> SR implied) used at .79, boost EV-only
og_qn = 32; og_fee = 0.07*og_qn*0.79*0.21; og_qcost = og_qn*0.79 + og_fee
b_n = math.floor(25/(0.79+0.07*0.79*0.21)); b_lock = b_n - hedge(b_n, 0.79)
out['og'] = dict(qual_cost=round(og_qcost,2), qual_unhedged_EV=round(-og_fee-og_qn*0.005,2), qual_maxloss=round(og_qcost,2),
    qual_hedged=round(og_qn - og_qcost - hedge(og_qn,0.79),2), bonus_lock=round(b_lock,2), boost_EV_at_p50=12.5,
    boost_maxloss=25.0)
import json; print(json.dumps(out, indent=1))
