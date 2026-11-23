"""All results of the Kelmarsh study (O1-O3). CPU only.

    python tools/analyze_kelmarsh.py results/kelmarsh_shard0.jsonl results/kelmarsh_shard1.jsonl --out analysis_kelmarsh
    python tools/analyze_kelmarsh.py --baselines-only --out analysis_baselines

O1: operating shares and the UGF against observed plant availability (needs no LLM).
O2: does the LLM rating of one month predict the operating states of the next month better than persistence, the mean of earlier
    months and numerical models of the same facts? The link (a1, a2, b) is refitted before each window on earlier windows only.
O3: Shapley values against the LLM's own explanation (faithfulness, irrelevant share U, stability), and the audit gate against matched
    random dropping, scored on the band for next-month full-output share and on plant availability, at the pre-specified 70th percentile
    and at neighbouring percentiles.

Effect sizes come with every test: the mean difference with its interval, and the matched-pairs rank-biserial correlation. p-values are
Holm-adjusted within each family of comparisons. Turbine intervals resample whole turbines, which keeps each turbine's months together;
plant intervals use a moving-block bootstrap, because neighbouring months are not independent.
"""
import argparse, itertools, json, math, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from build_kelmarsh import next_period
from xfmss import benchmark as bm, metrics as mt, ugf
from xfmss.it2 import ALPHAS, IT2, TERM_CENTERS, dist_summary, interval_score, propagate_monotone, t1_from_dist

KAPPA = math.sqrt(6)  # the normalised triangle has the same standard deviation as the letter distribution, Lemma 1
GAMMA, CUT = 0.5, 0.5
EXPLAINERS = ["kernelshap_exact", "self_rationale"]
AUDIT_Q = 0.7  # pre-specified
AUDIT_QS = (0.5, 0.6, 0.7, 0.8, 0.9)  # robustness: the same ratings at neighbouring percentiles
TEST_FROM = "2019-01"  # the first two years only train the link
DRAWS = 200
FACTS = ["forced_stops", "stopped_h", "derated_pct", "gear_oil_t", "gen_bear_t", "transf_t"]
LONG_STOP = 0.10  # a rare event: a turbine stopped for more than a tenth of the next month
BLOCK = 6  # months, moving-block bootstrap at plant level


def sig(z):
    return 1 / (1 + np.exp(-z))


def link_probs(par, e):
    a1, a2, b = par
    ge1, ge2 = sig(a1 + b * e), sig(a2 + b * e)
    return np.clip(np.stack([1 - ge1, ge1 - ge2, ge2], -1), 1e-9, 1)


def fit_link(e, shares, weight=1000.0):
    """Maximum likelihood on state shares (each turbine-period counts equally). Theorem 1 needs b >= 0 and a1 >= a2,
    so a2 is written as a1 minus a positive gap and b is bounded below by zero."""
    par = lambda x: (x[0], x[0] - np.exp(x[1]), x[2])
    nll = lambda x: -float((weight * shares * np.log(link_probs(par(x), e))).sum())
    best = min((minimize(nll, [s, 0.0, 1.0], method="L-BFGS-B", bounds=[(-10, 15), (-5, 4), (0.0, 20.0)]) for s in (2.0, 4.0)), key=lambda r: r.fun)
    return par(best.x)


def ordinal_probs(par, X):
    a1, a2, beta = par
    eta = X @ beta
    ge1, ge2 = sig(a1 + eta), sig(a2 + eta)
    return np.clip(np.stack([1 - ge1, ge1 - ge2, ge2], -1), 1e-9, 1)


def fit_ordinal(X, shares, lam):
    """The link's likelihood on numbers in place of a rating, with a ridge penalty on the slopes."""
    par = lambda z: (z[0], z[0] - np.exp(z[1]), z[2:])
    nll = lambda z: -float((shares * np.log(ordinal_probs(par(z), X))).sum()) + 0.5 * lam * float(z[2:] @ z[2:])
    r = minimize(nll, np.r_[4.0, 0.0, np.zeros(X.shape[1])], method="L-BFGS-B", bounds=[(-10, 15), (-5, 4)] + [(None, None)] * X.shape[1])
    return par(r.x)


def numeric_forecasts(t, fe, test):
    """The six facts of the reports, each against that month's fleet median, and no text: does the LLM add anything to structured data?
    Earlier months only, as for the link. The ridge penalty is chosen on the last twelve training months; boosting settings are fixed."""
    x = fe[["period", "wt"]].copy()
    for c in FACTS:
        x[c] = fe[c] - fe.groupby("period")[c].transform("median")
    d = t.merge(x, on=["period", "wt"])
    Y = ["stop_next", "der_next", "full_next"]
    out, lam = [], 1.0
    for i, p in enumerate(test):
        tr, te = d[d.period < p], d[d.period == p].copy()
        mu, sd = tr[FACTS].mean(), tr[FACTS].std().replace(0, 1)
        A, B = ((tr[FACTS] - mu) / sd).fillna(0).values, ((te[FACTS] - mu) / sd).fillna(0).values
        if i % 12 == 0:
            cut = sorted(tr.period.unique())[-12]
            old, new = (tr.period < cut).values, (tr.period >= cut).values
            err = {l: float(np.abs(ordinal_probs(fit_ordinal(A[old], tr[Y].values[old], l), A[new])[:, 2] - tr.full_next.values[new]).mean()) for l in (0.1, 1.0, 10.0, 100.0)}
            lam = min(err, key=err.get)
        q = ordinal_probs(fit_ordinal(A, tr[Y].values, lam), B)
        te["ord_stop"], te["ord_der"], te["ord_full"] = q[:, 0], q[:, 1], q[:, 2]
        g = {c: np.clip(HistGradientBoostingRegressor(max_depth=2, max_iter=100, learning_rate=0.05, min_samples_leaf=10, random_state=0).fit(tr[FACTS].values, tr[c].values).predict(te[FACTS].values), 0, 1)
             for c in ("stop_next", "full_next")}
        te["gbm_stop"], te["gbm_full"] = g["stop_next"], g["full_next"]
        te["gbm_der"] = np.clip(1 - g["stop_next"] - g["full_next"], 0, 1)
        out.append(te[["period", "wt", "ord_stop", "ord_der", "ord_full", "gbm_stop", "gbm_der", "gbm_full"]])
    return pd.concat(out)


def boot_turbine(df, fn, n=2000, seed=0):
    """95 % bootstrap interval resampling whole turbines: a turbine's months stay together, so dependence within a turbine is kept.
    With six turbines the interval is coarse."""
    rng = np.random.default_rng(seed)
    wts = df.wt.unique()
    groups = {w: df[df.wt == w] for w in wts}
    out = [fn(pd.concat([groups[w] for w in rng.choice(wts, len(wts))])) for _ in range(n)]
    return tuple(np.nanquantile(out, [0.025, 0.975]))


def block_boot(x, n=2000, seed=0):
    """95 % interval for the mean of a monthly series, moving blocks of BLOCK months (neighbouring months are not independent)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = len(x) - BLOCK + 1
    k = int(np.ceil(len(x) / BLOCK))
    means = [np.mean(np.concatenate([x[s:s + BLOCK] for s in rng.integers(0, m, k)])[:len(x)]) for _ in range(n)]
    return tuple(np.quantile(means, [0.025, 0.975]))


def paired(d):
    """Wilcoxon p and matched-pairs rank-biserial correlation for paired differences."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) < 6:
        return np.nan, np.nan
    r = rankdata(np.abs(d))
    return float(wilcoxon(d).pvalue), float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj, top = np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        top = max(top, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = top
    return adj


def u_of(r, m):
    d = r["methods"].get(m, {})
    if "scores" not in d:
        return np.nan
    pl = mt.plausibility(d["scores"], r["kinds"])
    return pl["mass_distractor"] if np.isfinite(pl["mass_distractor"]) else np.nan


def turbine_table(sh):
    """this period's shares (persistence) next to the next period's shares (truth)"""
    s = sh.rename(columns={"p_stop": "stop_now", "p_der": "der_now", "p_full": "full_now"})
    s["next"] = s.period.map(next_period)
    n = sh[["period", "wt", "p_stop", "p_der", "p_full"]].rename(columns={"period": "next", "p_stop": "stop_next", "p_der": "der_next", "p_full": "full_next"})
    return s.merge(n, on=["next", "wt"])


def asset_it2(ps):
    return IT2.from_t1_list([t1_from_dist(p, U=0.0, lam=0.0, kappa=KAPPA) for p in ps])


def plant_band(plant, fz, par, level):
    f = lambda h: ugf.availability_from_probs(plant, {k: link_probs(par, np.array(v)) for k, v in h.items()}, demand=6 * level * 2.05)
    out = propagate_monotone(fz, f)
    lo, hi = out.umf[int(round(CUT * (len(ALPHAS) - 1)))]
    return out.defuzz(), lo, hi


def window(rows, p, m, q, within=False):
    """letter distributions of a window, and the wordings the gate admits at percentile q. within=True takes the percentile inside each
    wording, so that a wording with a higher U as such is not dropped for it (exploratory, not the pre-specified gate)."""
    win = [r for r in rows if r["period"] == p]
    us_all = {v: [u_of(r, m) for r in win if within is False or r["variant"] == v] for v in {r["variant"] for r in win}}
    thr = {v: np.nanquantile(x, q) for v, x in us_all.items()}
    ps, kept = {}, {}
    for wt in sorted({r["wt"] for r in win}):
        rs = sorted([r for r in win if r["wt"] == wt], key=lambda x: x["variant"])
        ps[wt] = [np.asarray(r["p"], float) / np.sum(r["p"]) for r in rs]
        us = np.nan_to_num(np.array([u_of(r, m) for r in rs], float))
        kept[wt] = [j for j, (u, r) in enumerate(zip(us, rs)) if u <= thr[r["variant"]]] or [int(np.argmin(us))]
    return ps, kept


def turbine_score(fz, par, truth, k_cut):
    ones = []
    for wt, f in fz.items():
        lo, hi = f.umf[k_cut]
        ql, qh = float(link_probs(par, np.array(lo))[2]), float(link_probs(par, np.array(hi))[2])
        ones.append({"wt": wt, "score": interval_score(ql, qh, truth[wt], alpha=GAMMA), "width": qh - ql, "covered": float(ql <= truth[wt] <= qh)})
    return ones


def gate_row(rows, test, links, t, m, q, within=False, draws=50):
    """audited against matched random removal at turbine level, for one gate (50 draws, seed 1; the 70 % row therefore differs slightly
    from the main test, which uses 200 draws)"""
    k_cut = int(round(CUT * (len(ALPHAS) - 1)))
    rng = np.random.default_rng(1)
    one = []
    for p in test:
        ps, kept = window(rows, p, m, q, within)
        truth = t[t.period == p].set_index("wt").full_next
        a = turbine_score({w: asset_it2([ps[w][j] for j in kept[w]]) for w in ps}, links[p], truth, k_cut)
        dr = [turbine_score({w: asset_it2([ps[w][j] for j in rng.choice(len(ps[w]), len(kept[w]), replace=False)]) for w in ps}, links[p], truth, k_cut) for _ in range(draws)]
        for i, row in enumerate(a):
            one.append({"period": p, "wt": row["wt"], "audited": row["score"], "random": float(np.mean([d[i]["score"] for d in dr])), "kept": len(kept[row["wt"]]) / len(ps[row["wt"]])})
    d = pd.DataFrame(one)
    d["diff"] = d.audited - d.random
    lo, hi = boot_turbine(d, lambda x: x["diff"].mean(), n=1000)
    p_w, rb = paired(d["diff"])
    return {"explainer": m, "percentile": q, "within wording": within, "wordings kept": d.kept.mean(), "audited": d.audited.mean(), "random": d.random.mean(), "difference": d["diff"].mean(), "CI low": lo, "CI high": hi, "p": p_w, "rank-biserial": rb}


def thresholds(rows, test, links, t, md, out):
    """the same ratings and the same control at neighbouring percentiles: one threshold guarantees nothing"""
    res = pd.DataFrame([gate_row(rows, test, links, t, m, q) for m in EXPLAINERS for q in AUDIT_QS])
    res.to_csv(f"{out}/o3_threshold_sensitivity.csv", index=False)
    md += ["## O3: sensitivity to the audit percentile (turbine level, audited minus matched random)", res.round(4).to_markdown(index=False), ""]
    # which wordings does the gate drop? (the gate should screen judgments, not a writing style)
    g = []
    for p in test:
        ps, kept = window(rows, p, EXPLAINERS[0], AUDIT_Q)
        for wt in ps:
            rs = sorted([r for r in rows if r["period"] == p and r["wt"] == wt], key=lambda x: x["variant"])
            g += [{"wording": r["variant"], "U": u_of(r, EXPLAINERS[0]), "E": r["E"], "admitted": float(i in kept[wt])} for i, r in enumerate(rs)]
    g = pd.DataFrame(g)
    by = g.groupby("wording").agg(mean_U=("U", "mean"), mean_E=("E", "mean"), admitted=("admitted", "mean"), n=("E", "size")).round(3)
    by.to_csv(f"{out}/o3_gate_by_wording.csv")
    within = pd.DataFrame([gate_row(rows, test, links, t, EXPLAINERS[0], q, within=True) for q in (0.6, 0.7, 0.8)])
    within.to_csv(f"{out}/o3_within_wording.csv", index=False)
    md += ["## O3: what the Shapley gate drops", by.to_markdown(), f"Mean E of admitted wordings {g[g.admitted == 1].E.mean():.4f}, of dropped wordings {g[g.admitted == 0].E.mean():.4f}.", "",
           "## O3: exploratory, percentile taken within each wording", within.round(4).to_markdown(index=False), ""]


def llm_checks(rows, md):
    """Output quality: the reading of next-token probabilities has no text to parse, so the checks are letter mass, degenerate ratings and failed rationales."""
    cov = np.array([r["coverage"] for r in rows], float)
    p = np.array([r["p"] for r in rows], float)
    tops = pd.Series([r["answer"] for r in rows]).value_counts(normalize=True).round(3).to_dict()
    spread = [dist_summary(x) for x in p]
    clipped = np.mean([c - KAPPA * max(s, 1 / 12) < 0 or c + KAPPA * max(s, 1 / 12) > 1 for c, s in spread])
    exp = [r for r in rows if r["methods"]]
    empty = {m: int(sum("scores" in r["methods"].get(m, {}) and not np.any(np.abs(r["methods"][m]["scores"]) > 0) for r in exp)) for m in EXPLAINERS}
    errors = {m: int(sum("error" in r["methods"].get(m, {}) for r in exp)) for m in EXPLAINERS}
    e = pd.DataFrame([{"k": f"{r['period']}|{r['wt']}", "v": r["variant"], "E": r["E"]} for r in rows]).pivot(index="k", columns="v", values="E").dropna()
    fr = friedmanchisquare(*[e[c].values for c in e.columns]).pvalue
    md += ["## LLM output checks", f"Ratings {len(rows)}. Letter mass mean {cov.mean():.3f}, minimum {cov.min():.3f}, ratings under 0.5: {(cov < 0.5).sum()}. Letters chosen: {tops}.",
           f"Ratings whose triangle is clipped at 0 or 1: {clipped:.1%}. Explained ratings {len(exp)}; failures {errors}; rationales naming no valid sentence {empty}.",
           f"Wording effect: mean E by wording {e.mean().round(4).to_dict()}, largest difference {e.mean().max() - e.mean().min():.4f}, Friedman p = {fr:.3g}, {len(e)} turbine-months.", ""]


def baselines(sh, pc, md, out):
    pc["err"], pc["err_power"] = (pc.A_obs - pc.A_ugf).abs(), (pc.A_power - pc.A_ugf).abs()
    o1 = pc.groupby("level").agg(mean_observed=("A_obs", "mean"), mean_power=("A_power", "mean"), mean_ugf=("A_ugf", "mean"), mae=("err", "mean"), worst=("err", "max"),
                                 mae_power=("err_power", "mean"), worst_power=("err_power", "max")).round(4)
    o1["corr_power"] = [round(float(g.A_power.corr(g.A_ugf)), 3) for _, g in pc.groupby("level")]
    o1.to_csv(f"{out}/o1_ugf_check.csv")
    md += ["## O1: UGF against observed plant availability", f"{pc.period.nunique()} months, {int(pc.groupby('period').n.first().sum()):,} plant-valid intervals.", o1.to_markdown(), ""]
    t = turbine_table(sh)
    dev = lambda c: t[c] - t.groupby("period")[c].transform("mean")
    rows = [{"state": s, "spearman": spearmanr(t[f"{s}_now"], t[f"{s}_next"]).statistic, "within_period": spearmanr(dev(f"{s}_now"), dev(f"{s}_next")).statistic} for s in ("stop", "der", "full")]
    md += ["## Persistence of each state share", f"{len(t)} turbine-periods.", pd.DataFrame(rows).round(3).to_markdown(index=False), ""]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--bench", default="data/kelmarsh.json")
    ap.add_argument("--shares", default="data/real/kelmarsh_shares.csv")
    ap.add_argument("--plant", default="data/real/kelmarsh_plant.csv")
    ap.add_argument("--features", default="data/real/kelmarsh_features.csv")
    ap.add_argument("--level", type=float, default=0.9)
    ap.add_argument("--out", default="analysis_kelmarsh")
    ap.add_argument("--baselines-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sh, pc, fe = pd.read_csv(a.shares), pd.read_csv(a.plant), pd.read_csv(a.features)
    md = ["# Kelmarsh results", ""]
    t = baselines(sh, pc, md, a.out)
    periods = sorted(t.period.unique())
    test = [p for p in periods if p >= TEST_FROM]
    te = t[t.period.isin(test)].copy()
    te["fleet_full"] = [t[t.period < p].full_next.mean() for p in te.period]
    te["fleet_stop"] = [t[t.period < p].stop_next.mean() for p in te.period]
    te["fleet_der"] = [t[t.period < p].der_next.mean() for p in te.period]
    te = te.merge(numeric_forecasts(t, fe, test), on=["period", "wt"])
    mae = lambda d, f, y="full_next": float((d[f] - d[y]).abs().mean())
    rho = lambda d, f, y: spearmanr(d[f], d[y]).statistic if d[f].nunique() > 1 else np.nan

    def row(name, d, f, s, g):
        lo, hi = boot_turbine(d, lambda x: mae(x, f))
        ev = (d.stop_next > LONG_STOP).astype(int)
        return {"predictor": name, "MAE full share": mae(d, f), "CI low": lo, "CI high": hi, "MAE stopped share": mae(d, s, "stop_next"), "MAE derated share": mae(d, g, "der_next"),
                "Spearman full": rho(d, f, "full_next"), "Spearman derated": rho(d, g, "der_next"), "AUC long stop": roc_auc_score(ev, d[s]) if d[s].nunique() > 1 else 0.5}

    def versus(d, f, g, name):
        """mean change in the absolute error of the full-output share, f against reference g"""
        x = d.assign(diff=(d[f] - d.full_next).abs() - (d[g] - d.full_next).abs())
        lo, hi = boot_turbine(x, lambda s: s["diff"].mean())
        p, rb = paired(x["diff"])
        return {"comparison": name, "difference": x["diff"].mean(), "CI low": lo, "CI high": hi, "relative": x["diff"].mean() / mae(d, g), "p": p, "rank-biserial": rb}

    REF = [("Persistence", "full_now", "stop_now", "der_now"), ("Earlier-periods mean", "fleet_full", "fleet_stop", "fleet_der"),
           ("Ordinal logit on the facts", "ord_full", "ord_stop", "ord_der"), ("Gradient boosting on the facts", "gbm_full", "gbm_stop", "gbm_der")]
    o2 = [row(n, te, f, s, g) for n, f, s, g in REF]
    ev = int((te.stop_next > LONG_STOP).sum())
    head = f"Forward test: {len(te)} turbine-periods from {test[0]}; {ev} of them with a stopped share above {LONG_STOP:.0%} in the next month."
    if a.baselines_only:
        pairs = [versus(te, f, g, f"{n} minus {m}") for (n, f) in [("Ordinal logit", "ord_full"), ("Gradient boosting", "gbm_full")] for (m, g) in [("persistence", "full_now"), ("earlier-periods mean", "fleet_full")]]
        q = pd.DataFrame(pairs)
        q["Holm p"] = holm(q.p.values)
        q.to_csv(f"{a.out}/o2_pairs.csv", index=False)
        pd.DataFrame(o2).to_csv(f"{a.out}/o2_prediction.csv", index=False)
        md += ["## O2 baselines", head, pd.DataFrame(o2).round(4).to_markdown(index=False), "", q.round(4).to_markdown(index=False), ""]
        open(f"{a.out}/summary.md", "w", encoding="utf-8").write("\n".join(md))
        print("\n".join(md))
        return

    rows = [json.loads(l) for p in a.files for l in open(p, encoding="utf-8") if l.strip()]
    for r in rows:
        r["period"], r["wt"], r["E"] = r["system"].split("_")[1], int(r["comp"][-1]), float(np.dot(r["p"], TERM_CENTERS))
    bench = {p["id"]: p for p in bm.load(a.bench)}
    md.insert(1, f"Ratings: {len(rows)}; prompt {sorted({r.get('prompt') for r in rows})}\n")
    llm_checks(rows, md)
    r_t = pd.DataFrame([{"period": r["period"], "wt": r["wt"], "E": r["E"]} for r in rows]).groupby(["period", "wt"]).E.agg(["mean", "std"]).reset_index().rename(columns={"mean": "E", "std": "E_sd"})
    t = t.merge(r_t, on=["period", "wt"])

    # O2: forward-chained predictions
    pred, links = [], {}
    for p in test:
        tr = t[t.period < p]
        par = fit_link(tr.E.values, tr[["stop_next", "der_next", "full_next"]].values)
        links[p] = par
        d = t[t.period == p][["period", "wt", "E"]].copy()
        q = link_probs(par, d.E.values)
        d["llm_full"], d["llm_stop"], d["llm_der"] = q[:, 2], q[:, 0], q[:, 1]
        pred.append(d)
    pred = te.merge(pd.concat(pred), on=["period", "wt"])
    o2.insert(0, row("LLM rating + link", pred, "llm_full", "llm_stop", "llm_der"))
    o2 = pd.DataFrame(o2)
    o2.to_csv(f"{a.out}/o2_prediction.csv", index=False)
    follow = {c: float(spearmanr(t.E, t[c]).statistic) for c in ("stop_now", "der_now", "full_now")}
    rng_auc = np.random.default_rng(0)
    groups = {w: pred[pred.wt == w] for w in pred.wt.unique()}
    diffs = []
    for _ in range(2000):
        b = pd.concat([groups[w] for w in rng_auc.choice(list(groups), len(groups))])
        y = b.stop_next > LONG_STOP
        if y.sum() >= 2:
            diffs.append(roc_auc_score(y, b.llm_stop) - roc_auc_score(y, b.stop_now))
    auc_ci = np.quantile(diffs, [0.025, 0.975])
    links_tab = pd.DataFrame([{"period": k, "a1": v[0], "a2": v[1], "b": v[2]} for k, v in links.items()])
    links_tab.to_csv(f"{a.out}/o2_links.csv", index=False)
    pairs = pd.DataFrame([versus(pred, "llm_full", f, f"LLM minus {n}") for n, f, _, _ in REF])
    pairs["Holm p"] = holm(pairs.p.fillna(1).values)
    pairs.to_csv(f"{a.out}/o2_pairs.csv", index=False)
    md += ["## O2: prediction of the next month", head + f" Spearman(E, next full share) on all {len(t)}: {spearmanr(t.E, t.full_next).statistic:.3f}. Mean E {t.E.mean():.3f}, mean spread across wordings {t.E_sd.mean():.3f}.",
           f"Slope b: above zero in {(links_tab.b > 1e-6).sum()} of {len(links_tab)} fits; mean {links_tab.b.mean():.3f}, sd {links_tab.b.std():.3f}, range {links_tab.b.min():.3f} to {links_tab.b.max():.3f}.",
           f"Spearman of the rating with this month's shares: stopped {follow['stop_now']:.3f}, derated {follow['der_now']:.3f}, full {follow['full_now']:.3f}. AUC, LLM minus persistence, 95% interval {auc_ci[0]:+.3f} to {auc_ci[1]:+.3f}.",
           o2.round(4).to_markdown(index=False), "", pairs.round(4).to_markdown(index=False), ""]

    # O3a: the explanations themselves
    ex, gain = [], {}
    for m in EXPLAINERS:
        g = [r for r in rows if "scores" in r["methods"].get(m, {})]
        gain[m] = {r["key"]: r["methods"][m].get("comp_aopc", np.nan) - np.mean([x["comp_aopc"] for x in r["random"]]) for r in g}
        taus = []
        for _, rs in itertools.groupby(sorted(g, key=lambda r: (r["period"], r["wt"], r["variant"])), key=lambda r: (r["period"], r["wt"])):
            rs = list(rs)
            vec = [mt.align_by_cue(x["methods"][m]["scores"], x["cue_ids"], rs[0]["cue_ids"]) for x in rs]
            taus += [mt.agreement(u, v)["kendall"] for u, v in itertools.combinations(vec, 2)]
        ex.append({"explainer": m, "n": len(g), "faithfulness gain": float(np.nanmean(list(gain[m].values()))), "irrelevant share U": float(np.nanmean([u_of(r, m) for r in g])),
                   "stability tau": float(np.nanmean(taus)), "seconds": float(np.mean([r["methods"][m]["time"] for r in g])), "total hours": float(np.sum([r["methods"][m]["time"] for r in g]) / 3600)})
    ex = pd.DataFrame(ex)
    ex.to_csv(f"{a.out}/o3_explainers.csv", index=False)
    keys = sorted(set(gain[EXPLAINERS[0]]) & set(gain[EXPLAINERS[1]]))
    x, y = np.array([gain[EXPLAINERS[0]][k] for k in keys]), np.array([gain[EXPLAINERS[1]][k] for k in keys])
    p_f, rb_f = paired(x - y)
    md += ["## O3: explanations", ex.round(4).to_markdown(index=False), f"Shapley minus self-rationale, faithfulness gain: {np.nanmean(x - y):+.4f}, Wilcoxon p = {p_f:.3g}, rank-biserial {rb_f:.3f}, n = {len(keys)}.", ""]

    # O3b: audit gate against matched random dropping, forward-chained
    res, plant_rows = [], []
    rng = np.random.default_rng(0)
    k_cut = int(round(CUT * (len(ALPHAS) - 1)))
    for m in EXPLAINERS:
        for p in test:
            par = links[p]
            ps, kept = window(rows, p, m, AUDIT_Q)
            wts = sorted(ps)
            truth = t[t.period == p].set_index("wt").full_next
            obs = pc[(pc.period == next_period(p)) & np.isclose(pc.level, a.level)].A_obs.iloc[0]
            plant = bench[f"kelmarsh_{p}"]
            score = lambda fz, name: [{**s, "explainer": m, "period": p, "est": name} for s in turbine_score(fz, par, truth, k_cut)]

            def plant_row(fz, name):
                hat, lo, hi = plant_band(plant, {f"KWF{w}": f for w, f in fz.items()}, par, a.level)
                return {"explainer": m, "period": p, "est": name, "A_hat": hat, "lo": lo, "hi": hi, "A_obs": obs, "abs_err": abs(hat - obs), "score": interval_score(lo, hi, obs, alpha=GAMMA), "covered": float(lo <= obs <= hi), "width": hi - lo}

            plain = {wt: asset_it2(ps[wt]) for wt in wts}
            audited = {wt: asset_it2([ps[wt][j] for j in kept[wt]]) for wt in wts}
            res += score(plain, "plain") + score(audited, "audited")
            plant_rows += [plant_row(plain, "plain"), plant_row(audited, "audited")]
            draws, pr = [], []
            for d in range(DRAWS):
                fz = {wt: asset_it2([ps[wt][j] for j in rng.choice(len(ps[wt]), len(kept[wt]), replace=False)]) for wt in wts}
                draws += score(fz, "random")
                if d < 20:  # a plant evaluation is slower; 20 draws are enough for its mean
                    pr.append(plant_row(fz, "random"))
            res += pd.DataFrame(draws).groupby(["explainer", "period", "wt", "est"]).mean(numeric_only=True).reset_index().to_dict("records")
            q = pd.DataFrame(pr)
            plant_rows.append({**pr[0], **q.mean(numeric_only=True).to_dict()})
    res, pr = pd.DataFrame(res), pd.DataFrame(plant_rows)
    res.to_csv(f"{a.out}/o3_audit_turbines.csv", index=False)
    pr.to_csv(f"{a.out}/o3_audit_plant.csv", index=False)
    md += ["## O3: audit gate (band for next-month full share, turbine level)", res.groupby(["explainer", "est"]).agg(score=("score", "mean"), width=("width", "mean"), hit_rate=("covered", "mean"), n=("score", "size")).round(4).to_markdown(), ""]
    family = []
    for m in EXPLAINERS:
        w = res[res.explainer == m].pivot_table(index=["period", "wt"], columns="est", values="score").dropna().reset_index()
        w["d"] = w.audited - w.random
        lo, hi = boot_turbine(w, lambda s: s.d.mean())
        pv, rb = paired(w.d)
        family.append({"test": f"{m}, turbine", "difference": w.d.mean(), "relative": w.d.mean() / w.random.mean(), "CI low": lo, "CI high": hi, "p": pv, "rank-biserial": rb, "n": len(w)})
        w = pr[pr.explainer == m].pivot_table(index="period", columns="est", values="score").dropna().sort_index()
        d = (w.audited - w.random).values
        lo, hi = block_boot(d)
        pv, rb = paired(d)
        family.append({"test": f"{m}, plant", "difference": d.mean(), "relative": d.mean() / w.random.mean(), "CI low": lo, "CI high": hi, "p": pv, "rank-biserial": rb, "n": len(d)})
    family = pd.DataFrame(family)
    family["Holm p"] = holm(family.p.fillna(1).values)
    family.to_csv(f"{a.out}/o3_audit_tests.csv", index=False)
    md += ["## O3: audited minus matched random (interval score; turbine intervals by turbine, plant intervals by moving blocks)", family.round(4).to_markdown(index=False), ""]
    md += [f"## Plant availability at demand {a.level:.0%}", pr.groupby(["explainer", "est"]).agg(score=("score", "mean"), abs_err=("abs_err", "mean"), width=("width", "mean"), hit_rate=("covered", "mean"), n=("score", "size")).round(4).to_markdown(), ""]
    thresholds(rows, test, links, t, md, a.out)
    open(f"{a.out}/summary.md", "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
