#!/usr/bin/env python
"""How sharp can the call-type test get? Variance decomposition, power, alternative statistics, label ceiling.

Inputs: out-of-fold probe probabilities cached by cache_oof.py (board probe, each model at its CPU best layer, split 0
asserted to reproduce calltype_cpu_rescore.json acc11_cpu; splits 0-9 for the seed-group models). Nothing is trained.

Sections (all written to power_analysis.json beside this file):
  1. decomposition  Variance of an arm-mean difference (arm = mean of per-clip results over its seeds), split into
                    birds (2,000-draw bird bootstrap, seed 0, the tool's), clips within birds (birds fixed, clips
                    resampled inside each bird), seeds (pooled within-recipe variance over 12 seed groups), and the
                    probe's fold split (variance of the arm difference over StratifiedGroupKFold random_state 0-9).
  2. power          SE^2(B, n, K) = V_bird * 48/B + s^2 * 2/n + V_split/K ; power at true 0.01 / 0.02, and the
                    1/B scaling of V_bird checked by refitting the probe on bird subsets (B = 24, 36).
  3. statistics     The same components for alternative statistics (balanced accuracy, mean log p(true), mean p(true),
                    bird-mean accuracy, Tet recall, Tet+Thuk+Tuck recall, 10-split mean accuracy); recipe ICC
                    (one-way ANOVA over the 12 seed groups, 10-split-averaged values) and how many of the 66
                    recipe-pair contrasts each statistic resolves at the full SE (birds + seeds + split).
  4. ceiling        Clips wrong for every model (split 0, best layers), characterised by type, bird, age, duration,
                    and whether they stay wrong under a leaky random split (the bird's other clips in training).
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
import json, sys, warnings, itertools
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")
import cache_oof as C                                                      # noqa: E402

rows, y, birds, classes = C.data()
N = len(y); ub = np.unique(birds); B0 = len(ub)
bird_ix = np.searchsorted(ub, birds)
CPU = C.CPU
GROUPS, SINGLES = C.GROUPS, C.SINGLES
TE, TH, TU = classes.index("Te"), classes.index("Th"), classes.index("Tu")


def P(tag, s=0):
    return np.load(C.CACHE / f"{tag}_s{s}.npy")


def clipstats(tags, s=0):
    """per-clip arrays averaged over an arm's seeds: correct, log p(true), p(true)"""
    cs, lp, pt = [], [], []
    for t in tags:
        p = P(t, s); pt_ = p[np.arange(N), y]
        cs.append((p.argmax(1) == y).astype(float)); lp.append(np.log(np.clip(pt_, 1e-6, 1))); pt.append(pt_)
    return dict(c=np.mean(cs, 0), lp=np.mean(lp, 0), pt=np.mean(pt, 0))


Y1 = np.eye(11)[y]                                                          # N x 11
B1 = np.eye(B0)[bird_ix]                                                    # N x 48
NB_ORIG = B1.sum(0)
HARD = np.isin(y, [TE, TH, TU]).astype(float)
STATS = ["acc", "bacc", "logp", "ptrue", "birdacc", "tet", "hard"]


def metrics(W, st):
    """W: (D, N) clip weights (bootstrap multiplicities); returns dict stat -> (D,) values"""
    c, lp, pt = st["c"], st["lp"], st["pt"]
    sw = W.sum(1)
    out = dict(acc=W @ c / sw, logp=W @ lp / sw, ptrue=W @ pt / sw)
    ncls = W @ Y1; ccls = W @ (Y1 * c[:, None])
    rec = ccls / np.maximum(ncls, 1e-12)
    out["bacc"] = rec.mean(1)
    out["tet"] = rec[:, TE]
    out["hard"] = (W @ (HARD * c)) / (W @ HARD)
    nb = W @ B1; cb = W @ (B1 * c[:, None])
    have = nb > 0
    # bird-mean accuracy: each drawn bird counts once per draw (multiplicity = how often it was drawn)
    mult = nb / NB_ORIG                     # bird multiplicity (exactly k for the bird bootstrap, 1 within-bird)
    out["birdacc"] = (np.where(have, cb / np.maximum(nb, 1e-12), 0) * mult).sum(1) / mult.sum(1)
    return out


def bird_boot_W(D=2000, seed=0, birds_sel=None):
    """the tool's resampling: rng.choice over the unique birds, 2000 draws, seed 0 -> clip multiplicities"""
    rng = np.random.default_rng(seed)
    U = ub if birds_sel is None else birds_sel
    W = np.zeros((D, N))
    pos = {b: i for i, b in enumerate(ub)}
    for d in range(D):
        cnt = Counter(rng.choice(U, len(U)))
        w = np.zeros(B0)
        for b, k in cnt.items():
            w[pos[b]] = k
        W[d] = w[bird_ix]
    return W


def within_W(D=1000, seed=1):
    rng = np.random.default_rng(seed)
    W = np.zeros((D, N))
    for b in range(B0):
        ix = np.where(bird_ix == b)[0]
        W[:, ix] = rng.multinomial(len(ix), np.ones(len(ix)) / len(ix), size=D)
    return W


def iid_W(D=1000, seed=2):
    rng = np.random.default_rng(seed)
    return rng.multinomial(N, np.ones(N) / N, size=D).astype(float)


W_BIRD, W_WITHIN, W_IID = bird_boot_W(), within_W(), iid_W()
ONE = np.ones((1, N))


def full(st):
    return {k: float(v[0]) for k, v in metrics(ONE, st).items()}


# ---------- per-model values (split 0 and 10-split) ----------
group_tags = [t for g in GROUPS.values() for t in g]
split_tags = group_tags + SINGLES
val = {t: [full(clipstats([t], s)) for s in range(10)] for t in split_tags}       # t -> list over splits
for t in split_tags:
    assert abs(val[t][0]["acc"] - CPU[t]["acc11_cpu"]) < 1e-9


def seed_var(stat, splitavg=False):
    """pooled within-recipe variance of single-model values over the 12 seed groups"""
    ss, df = 0.0, 0
    for g, tags in GROUPS.items():
        v = np.array([np.mean([val[t][s][stat] for s in range(10)]) if splitavg else val[t][0][stat] for t in tags])
        ss += float(np.var(v, ddof=1)) * (len(v) - 1); df += len(v) - 1
    return ss / df, df


def contrast(ta, tb):
    """variance components of the arm-mean difference A - B, every statistic"""
    sa, sb = clipstats(ta), clipstats(tb)
    res = {}
    mb, mw, mi = [{k: metrics(W, sa)[k] - metrics(W, sb)[k] for k in STATS} for W in (W_BIRD, W_WITHIN, W_IID)]
    d0 = {k: full(sa)[k] - full(sb)[k] for k in STATS}
    dsplit = {k: [] for k in STATS}
    for s in range(10):
        a_, b_ = full(clipstats(ta, s)), full(clipstats(tb, s))
        for k in STATS:
            dsplit[k].append(a_[k] - b_[k])
    v11 = {k: [] for k in STATS}
    for x in ta:
        for z in tb:
            sx, sz = clipstats([x]), clipstats([z])
            mx, mz = metrics(W_BIRD, sx), metrics(W_BIRD, sz)
            for k in STATS:
                v11[k].append(float(np.var(mx[k] - mz[k], ddof=1)))
    for k in STATS:
        s2, df = seed_var(k)
        vb = float(np.var(mb[k], ddof=1)); vw = float(np.var(mw[k], ddof=1)); vi = float(np.var(mi[k], ddof=1))
        vs = float(np.var(dsplit[k], ddof=1)); vseed = s2 * (1 / len(ta) + 1 / len(tb))
        tot_tool = vb + vseed; tot_full = vb + vseed + vs
        res[k] = dict(diff_split0=d0[k], diff_10split_mean=float(np.mean(dsplit[k])),
                      lo_birds=float(d0[k] - 1.96 * np.sqrt(vb)), hi_birds=float(d0[k] + 1.96 * np.sqrt(vb)),
                      V_bird=vb, V_bird_1v1_mean=float(np.mean(v11[k])), n_a=len(ta), n_b=len(tb),
                      V_clip_within_birds=vw, V_between_birds_approx=max(vb - vw, 0.0), V_clip_iid=vi,
                      design_effect_bird_vs_iid=vb / vi, V_seed_term=vseed, seed_sd_pooled=float(np.sqrt(s2)),
                      seed_df=df, V_split=vs,
                      halfwidth_tool=float(1.96 * np.sqrt(tot_tool)), halfwidth_full=float(1.96 * np.sqrt(tot_full)),
                      share_full=dict(birds=vb / tot_full, seeds=vseed / tot_full, split=vs / tot_full),
                      share_birds_within_vs_between=dict(within=min(vw / vb, 1.0), between=max(1 - vw / vb, 0.0)))
    return res


def main():
    out = dict(protocol=("CPU embeddings, each model at its calltype_cpu_rescore.json best layer; board probe; split 0 "
                         "= board split (asserted); splits = StratifiedGroupKFold random_state 0-9; bird bootstrap "
                         "2000 draws seed 0 (the tool's draws); within-bird clip bootstrap 1000; iid clip bootstrap "
                         "1000; seeds pooled over 11 recipe groups"), groups=GROUPS)

    # ---- 1. decomposition on named arm contrasts ----
    named = {"run19 | run16": (GROUPS["run19"], GROUPS["run16"]),
             "e13 | run19": (GROUPS["e13"], GROUPS["run19"]),
             "run16 | vanilla_lr1e4": (GROUPS["run16"], GROUPS["vanilla"]),
             "run19 | vanilla_lr1e4": (GROUPS["run19"], GROUPS["vanilla"]),
             "e13 | vanilla_lr1e4": (GROUPS["e13"], GROUPS["vanilla"]),
             "run16_compute4x | run16_seed2 (single models)": (["run16_compute4x"], ["run16_seed2"])}
    out["decomposition"] = {}
    for name, (a, b) in named.items():
        out["decomposition"][name] = contrast(a, b)
        r = out["decomposition"][name]["acc"]
        print(f"[1] {name}: diff {r['diff_split0']:+.4f}  hw tool {r['halfwidth_tool']:.4f} full {r['halfwidth_full']:.4f}"
              f"  shares {r['share_full']}  within-bird share of V_bird {r['share_birds_within_vs_between']['within']:.2f}"
              f"  deff {r['design_effect_bird_vs_iid']:.2f}", flush=True)
    # cross-check against the tool's JSON for run16 | vanilla? (vanilla has 1 seed in the shared JSON too)
    smp = json.loads((C.ANA / "seed_mean_pairs_cpu.json").read_text())["pairs"]
    out["tool_crosscheck"] = {k: dict(diff=v["diff"], lo_birds=v["lo_birds"], hi_birds=v["hi_birds"], lo=v["lo"], hi=v["hi"],
                                      seed_sd_pooled=v["seed_sd_pooled"]) for k, v in smp.items()}

    # ---- all 55 recipe-pair contrasts: typical components ----
    gnames = list(GROUPS)
    pairs = {}
    for ga, gb in itertools.combinations(gnames, 2):
        pairs[f"{ga} | {gb}"] = contrast(GROUPS[ga], GROUPS[gb])
    out["recipe_pairs"] = pairs
    typical = {}
    for k in STATS:
        typical[k] = {c: float(np.median([pairs[p][k][c] for p in pairs])) for c in
                      ["V_bird", "V_clip_within_birds", "V_clip_iid", "V_seed_term", "V_split", "halfwidth_tool",
                       "halfwidth_full", "design_effect_bird_vs_iid"]}
        cs, as_ = [], []
        for p in pairs:
            r = pairs[p][k]; h = 1 / r["n_a"] + 1 / r["n_b"]
            c = (r["V_bird_1v1_mean"] - r["V_bird"]) / (2 - h); cs.append(c); as_.append(r["V_bird"] - c * h)
        typical[k]["V_bird_recipe_part_a"] = float(np.median(as_))
        typical[k]["V_bird_seedclip_part_c"] = float(np.median(cs))
        typical[k]["seed_var_single_model"] = seed_var(k)[0]
        typical[k]["seed_var_single_model_10splitavg"] = seed_var(k, True)[0]
    out["typical_components_median_over_recipe_pairs"] = typical

    # ---- 2. power (accuracy and every statistic, in its own units, scaled to an equivalent effect below) ----
    def se2(k, B, n, K):
        # birds part: recipe-level clip discordance (a) + seed-specific clip noise that averages out with seeds (c);
        # both are per-bird sampling terms and scale as 48/B. Split term measured on 2-seed arms (kept fixed: conservative
        # for n>2, since part of it is seed-by-split noise).
        t = typical[k]
        vb = max(t["V_bird_recipe_part_a"], 0) + max(t["V_bird_seedclip_part_c"], 0) * 2 / n
        s2 = t["seed_var_single_model"] if K == 1 else t["seed_var_single_model_10splitavg"]
        return vb * 48 / B + s2 * 2 / n + t["V_split"] / K

    def power(delta, se):
        z = delta / se
        return float(1 - norm.cdf(1.96 - z) + norm.cdf(-1.96 - z))

    # the typical pair has n=2 seeds per arm; V_split is measured on 2-seed arms; s2 from single models
    grid = {}
    for B in [48, 72, 96, 144, 192, 288, 480]:
        for n in [1, 2, 3, 4, 5, 6, 8, 10, 12, 20]:
            for K in [1, 10]:
                se = float(np.sqrt(se2("acc", B, n, K)))
                grid[f"B{B}_n{n}_K{K}"] = dict(se=se, halfwidth=1.96 * se, power_d01=power(0.01, se), power_d02=power(0.02, se))
    out["power_acc_grid"] = grid

    def need(target_se, vary, k="acc"):
        if vary == "n":   # birds fixed at 48, K=10
            for n in range(1, 2001):
                if np.sqrt(se2(k, 48, n, 10)) <= target_se:
                    return n
            return None
        for B in range(48, 100001, 4):   # seeds fixed
            if np.sqrt(se2(k, B, vary, 10)) <= target_se:
                return B
        return None

    def requirements(k, scale):
        req = {}
        for d in (0.01, 0.02):
            dd = d * scale
            for lab, z in (("halfwidth_le_delta", 1.96), ("power80", 1.96 + 0.8416)):
                req[f"d{d}_{lab}"] = dict(seeds_per_arm_at_48_birds_K10=need(dd / z, "n", k),
                                          birds_at_2_seeds_K10=need(dd / z, 2, k), birds_at_5_seeds_K10=need(dd / z, 5, k),
                                          birds_at_10_seeds_K10=need(dd / z, 10, k),
                                          power_now_2v2_48_K1=power(dd, float(np.sqrt(se2(k, 48, 2, 1)))),
                                          power_2v2_48_K10=power(dd, float(np.sqrt(se2(k, 48, 2, 10)))),
                                          power_5v5_48_K10=power(dd, float(np.sqrt(se2(k, 48, 5, 10)))),
                                          floor_se_infinite_seeds_48_birds_K10=float(np.sqrt(se2(k, 48, 10**9, 10))),
                                          se_now_2v2_48_K1=float(np.sqrt(se2(k, 48, 2, 1))))
        return req
    out["power_requirements_acc"] = requirements("acc", 1.0)
    print("[2]", json.dumps(out["power_requirements_acc"], indent=0), flush=True)
    # mean p(true): express a true accuracy difference in p(true) units by the slope (through the origin) of
    # p(true) differences on accuracy differences over the recipe pairs (10-split means)
    da = np.array([pairs[p]["acc"]["diff_10split_mean"] for p in pairs])
    dp = np.array([pairs[p]["ptrue"]["diff_10split_mean"] for p in pairs])
    slope = float((da * dp).sum() / (da * da).sum())
    out["ptrue_per_acc_slope_10split"] = slope
    out["power_requirements_ptrue_acc_equivalent"] = requirements("ptrue", slope)
    print("[2p] slope", slope, json.dumps(out["power_requirements_ptrue_acc_equivalent"], indent=0), flush=True)

    # ---- 2b. does V_bird scale as 1/B when the probe is refit on fewer birds? ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold
    pa, pb = "run19_seed2", "run16_compute4x"
    Xa = np.ascontiguousarray(np.load(C.FEAT / f"ct11_{pa}_cpuemb.npy", mmap_mode="r")[:, CPU[pa]["best_layer_cpu"]])
    Xb = np.ascontiguousarray(np.load(C.FEAT / f"ct11_{pb}_cpuemb.npy", mmap_mode="r")[:, CPU[pb]["best_layer_cpu"]])
    rng = np.random.default_rng(7)
    scal = {}
    for Bs in (24, 36, 48):
        vals = []
        for rep in range(6 if Bs < 48 else 1):
            sel = ub if Bs == 48 else rng.choice(ub, Bs, replace=False)
            m = np.isin(birds, sel); ix = np.where(m)[0]
            hits = []
            for X in (Xa, Xb):
                Pp = np.zeros((len(ix), 11))
                for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X[ix], y[ix], birds[ix]):
                    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(X[ix][tr], y[ix][tr])
                    pr = est.predict_proba(X[ix][te])
                    full_pr = np.zeros((len(te), 11)); full_pr[:, est.classes_] = pr
                    Pp[te] = full_pr
                hits.append((Pp.argmax(1) == y[ix]).astype(float))
            dh = hits[0] - hits[1]
            bsel = birds[ix]; ubs = np.unique(bsel); bi = {b: np.where(bsel == b)[0] for b in ubs}
            r2 = np.random.default_rng(0)
            bs = [np.concatenate([bi[b] for b in r2.choice(ubs, len(ubs))]) for _ in range(1000)]
            vb = float(np.var([dh[s].mean() for s in bs], ddof=1))
            vals.append(dict(diff=float(dh.mean()), V_bird=vb, V_bird_x_B_over_48=vb * Bs / 48,
                             acc_a=float(hits[0].mean()), acc_b=float(hits[1].mean())))
        scal[f"B{Bs}"] = dict(reps=vals, mean_V_bird_x_B_over_48=float(np.mean([v["V_bird_x_B_over_48"] for v in vals])),
                             mean_acc_a=float(np.mean([v["acc_a"] for v in vals])),
                             mean_acc_b=float(np.mean([v["acc_b"] for v in vals])))
        print(f"[2b] B={Bs}: V_bird*B/48 {scal[f'B{Bs}']['mean_V_bird_x_B_over_48']:.2e}  acc {scal[f'B{Bs}']['mean_acc_a']:.4f}/{scal[f'B{Bs}']['mean_acc_b']:.4f}", flush=True)
    out["bird_scaling_check"] = dict(pair=f"{pa} - {pb}", **scal)

    # ---- null calibration: seed-vs-seed pairs inside one recipe (true recipe difference = 0) ----
    nullp = {}
    for g, tags in GROUPS.items():
        for x, z in itertools.combinations(tags, 2):
            nullp[f"{x} | {z}"] = contrast([x], [z])
    out["null_seed_pairs"] = nullp
    out["seed_sd_by_group_acc"] = {g: dict(split0=float(np.std([val[t][0]["acc"] for t in tags], ddof=1)),
                                           avg10=float(np.std([np.mean([val[t][s]["acc"] for s in range(10)]) for t in tags], ddof=1)))
                                   for g, tags in GROUPS.items()}

    # ---- 3. alternative statistics ----
    stat_cmp = {}
    gm = {g: {k: [float(np.mean([val[t][s][k] for s in range(10)])) for t in tags] for k in STATS} for g, tags in GROUPS.items()}
    S2AVG = {k: seed_var(k, True)[0] for k in STATS}

    def zval(r, kk, ten):
        """z of a contrast: split-0 statistic with birds + seeds(split-0 s^2) + split; or the 10-split mean with
        birds + seeds(10-split-averaged s^2) + split/10"""
        h = 1 / r["n_a"] + 1 / r["n_b"]
        if ten:
            return r["diff_10split_mean"] / np.sqrt(r["V_bird"] + S2AVG[kk] * h + r["V_split"] / 10)
        return r["diff_split0"] / np.sqrt(r["V_bird"] + r["V_seed_term"] + r["V_split"])

    for k in STATS + [x + "10" for x in STATS]:
        ten = k.endswith("10"); kk = k[:-2] if ten else k
        # ICC(1) from one-way ANOVA over groups, 10-split-averaged single-model values
        vals_g = [np.array(gm[g][kk]) for g in GROUPS]
        ns = np.array([len(v) for v in vals_g]); grand = np.concatenate(vals_g).mean()
        msb = sum(len(v) * (v.mean() - grand) ** 2 for v in vals_g) / (len(vals_g) - 1)
        msw = sum(((v - v.mean()) ** 2).sum() for v in vals_g) / (ns.sum() - len(vals_g))
        n0 = (ns.sum() - (ns ** 2).sum() / ns.sum()) / (len(vals_g) - 1)
        icc = (msb - msw) / (msb + (n0 - 1) * msw)
        zsign = np.array([zval(pairs[p][kk], kk, ten) for p in pairs]); zs = np.abs(zsign)
        zacc = np.array([pairs[p]["acc"]["diff_split0"] for p in pairs])
        dk = np.array([pairs[p][kk]["diff_10split_mean" if ten else "diff_split0"] for p in pairs])
        zn = np.abs([zval(nullp[p][kk], kk, ten) for p in nullp])
        zn_tool = np.abs([nullp[p][kk]["diff_split0"] / np.sqrt(nullp[p][kk]["V_bird"]) for p in nullp])
        t = typical[kk]
        stat_cmp[k] = dict(resolved_pairs=[p for p, z in zip(pairs, zsign) if abs(z) > 1.96],
                           sign_agreement_with_acc_split0=float(np.mean(np.sign(dk) == np.sign(zacc))),
                           spearman_with_acc_split0_diffs=float(__import__("scipy.stats").stats.spearmanr(dk, zacc)[0]),
                           null_seed_pairs_resolved_full_se=f"{int((zn > 1.96).sum())}/{len(zn)}",
                           null_seed_pairs_resolved_birds_only_split0=f"{int((zn_tool > 1.96).sum())}/{len(zn_tool)}",
                           seed_sd_single_model=float(np.sqrt(S2AVG[kk] if ten else seed_var(kk)[0])),
                           icc_recipe_vs_seed_10splitavg=float(icc), MS_between=float(msb), MS_within=float(msw),
                           n_pairs_resolved=int((zs > 1.96).sum()), n_pairs=len(pairs), median_abs_z=float(np.median(zs)),
                           typical_variances=dict(birds=t["V_bird"],
                                                  seeds_2v2=(S2AVG[kk] if ten else t["seed_var_single_model"]) * 2 / 2,
                                                  split=t["V_split"] / (10 if ten else 1)),
                           z_named={n: float(zval(out["decomposition"][n][kk], kk, ten)) for n in out["decomposition"]})
        print(f"[3] {k:8s} ICC {icc:+.3f}  resolved {stat_cmp[k]['n_pairs_resolved']}/{len(pairs)} null {stat_cmp[k]['null_seed_pairs_resolved_full_se']}  median|z| {np.median(zs):.2f}", flush=True)
    out["statistics"] = stat_cmp

    # ---- 4. label ceiling ----
    import soundfile as sf
    tags_all = [t for t in CPU if (C.CACHE / f"{t}_s0.npy").exists()]
    tags_pre = [t for t in tags_all if t != "run37_shuffle"]
    Cm = np.array([(P(t).argmax(1) == y) for t in tags_all])               # models x N
    pred = np.array([P(t).argmax(1) for t in tags_all])
    ipre = [tags_all.index(t) for t in tags_pre]
    nwrong = (~Cm[ipre]).sum(0)
    allw = np.where(nwrong == len(ipre))[0]
    allw69 = np.where((~Cm).sum(0) == len(tags_all))[0]
    allr = np.where(nwrong == 0)[0]
    dur = np.array([sf.info(str(r[0])).duration for r in rows])
    age = np.array(["chick" if "Chick" in r[4] else "adult" for r in rows])
    cls = np.array(classes)[y]
    base = Counter(cls)
    def char(ix):
        cc = Counter(cls[ix]); bc = Counter(birds[ix])
        modal = []
        for i in ix:
            pr = Counter(pred[ipre, i]); m, k = pr.most_common(1)[0]
            modal.append((classes[m], k / len(ipre)))
        mc = Counter(m for m, _ in modal)
        return dict(n=int(len(ix)),
                    by_type={c: dict(n=int(cc.get(c, 0)), frac_of_type=cc.get(c, 0) / base[c]) for c in classes},
                    by_age={a: int((age[ix] == a).sum()) for a in ("adult", "chick")},
                    n_birds=int(len(bc)), top_birds=[(b, int(k), int((birds == b).sum())) for b, k in bc.most_common(8)],
                    top3_birds_share=float(sum(k for _, k in bc.most_common(3)) / max(len(ix), 1)),
                    duration_s=dict(median=float(np.median(dur[ix])), q25=float(np.percentile(dur[ix], 25)),
                                    q75=float(np.percentile(dur[ix], 75))),
                    modal_wrong_prediction=dict(mc.most_common()),
                    modal_agreement_median=float(np.median([a for _, a in modal])),
                    frac_unanimous_same_wrong_label=float(np.mean([a == 1.0 for _, a in modal])))
    leaky = {t: (np.load(C.CACHE / f"{t}_leaky.npy").argmax(1) == y) for t in C.LEAKY}
    leaky_allwrong = np.all([~leaky[t] for t in C.LEAKY], 0)
    hist = np.bincount(nwrong, minlength=len(ipre) + 1)
    # pair discordance: clips where two models disagree in correctness (only these move a difference)
    disc = [int((Cm[i] != Cm[j]).sum()) for i, j in itertools.combinations(ipre, 2)]
    out["ceiling"] = dict(
        n_models_all=len(tags_all), n_models_excl_shuffle=len(tags_pre),
        n_wrong_for_every_model_excl_shuffle=int(len(allw)), n_wrong_for_every_model_incl_shuffle=int(len(allw69)),
        n_right_for_every_model_excl_shuffle=int(len(allr)),
        hist_n_models_wrong=hist.tolist(),
        accuracy_ceiling_if_allwrong_unfixable=float(1 - len(allw) / N),
        all_wrong=char(allw), all_clips_reference=dict(
            duration_s=dict(median=float(np.median(dur)), q25=float(np.percentile(dur, 25)), q75=float(np.percentile(dur, 75))),
            by_age={a: int((age == a).sum()) for a in ("adult", "chick")}, by_type=dict(base)),
        leaky_split_models=C.LEAKY,
        leaky_acc={t: float(leaky[t].mean()) for t in C.LEAKY},
        allwrong_still_wrong_under_leaky_all3=int(leaky_allwrong[allw].sum()),
        leaky_all3_wrong_rate_overall=float(leaky_allwrong.mean()),
        leaky_all3_wrong_rate_contested=float(leaky_allwrong[(nwrong > 0) & (nwrong < len(ipre))].mean()),
        pair_discordant_clips=dict(median=float(np.median(disc)), q10=float(np.percentile(disc, 10)), q90=float(np.percentile(disc, 90))),
        allwrong_clip_files=[str(rows[i][0].name) for i in allw])
    print(f"[4] all-wrong (68 models) {len(allw)}, incl shuffle {len(allw69)}, all-right {len(allr)}; still wrong under leaky "
          f"{int(leaky_allwrong[allw].sum())}/{len(allw)}", flush=True)
    (HERE / "power_analysis.json").write_text(json.dumps(out, indent=1, default=float))
    print("->", HERE / "power_analysis.json")


if __name__ == "__main__":
    main()
