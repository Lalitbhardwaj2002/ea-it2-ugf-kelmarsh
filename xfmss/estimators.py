"""Asset- and plant-level estimators built from LLM judgment rows (RQ2b and its baselines).

Every estimator turns the rows of one asset (one row per paraphrase variant) into an uncertain health value,
then propagates it through the plant UGF:

    crisp      point health = mean expected health E over variants (plug-in baseline)
    t1         one type-1 TFN from the variant-pooled letter distribution
    prob_mc    probabilistic baseline: Monte Carlo over the letter distributions (variant drawn uniformly,
               letter drawn from its distribution, health uniform within the letter's half-step)
    it2        IT2 FOU from one TFN per variant ("subjects"); U = 0 (plain) or U from an explainer
    it2_const  width-matched control: every TFN widened by the explainer's MEAN U (same average width as it2,
               but the widening carries no per-judgment information)
    it2_screen subject screening: variants whose U is in the explainer's top 30 % are dropped before the FOU is
               built (no widening); if all are dropped, the lowest-U variant is kept. Analogue of the bad-data
               processing of Mendel's Interval Approach (Liu & Mendel 2008), with explanations as the test.
    it2_screen_rand  control: the same number of variants dropped per asset, chosen at random

it2 vs it2_const (and it2_screen vs it2_screen_rand) test whether explanations carry per-judgment information,
rather than just adding width or removing subjects. All fuzzy estimators are IT2 objects (crisp and t1 are
degenerate cases), so they share propagate_monotone() and the Karnik-Mendel centroid.
"""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np

from . import ugf
from . import metrics as mt
from .it2 import IT2, ALPHAS, TERM_CENTERS, t1_from_dist, propagate_monotone, interval_score

PRIOR_SMOOTH = 0.1  # share of a uniform distribution mixed into the evidence-free prior before dividing it out
HALF_STEP = float(TERM_CENTERS[1] - TERM_CENTERS[0]) / 2
BASE_CFG = {"lam": 2.0, "kappa": 1.5, "n_var": 3, "prob": "raw", "cut": 0.5, "link": None,
            "iv_alpha": 0.5, "mc_n": 400}


def term_probs(row: dict, mode: str = "raw") -> np.ndarray:
    """Letter distribution of one judgment; 'calibrated' divides out the evidence-free prior (Zhao et al. 2021).

    The prior is smoothed towards uniform first: Phi-3.5's prior is ~0.95 on 'D' and ~0 on 'A'/'G', and dividing by
    it unsmoothed multiplies P(A) by about 10,000, which makes ratings more pessimistic instead of less.
    """
    p = np.asarray(row["p"], float)
    if mode == "calibrated" and row.get("p_prior") is not None:
        p0 = np.asarray(row["p_prior"], float)
        p = p / ((1 - PRIOR_SMOOTH) * p0 / p0.sum() + PRIOR_SMOOTH / len(p0))
    return p / p.sum()


def nonrelevant_mass(row: dict, method: str | None) -> float:
    """U in [0, 1]: share of |attribution| on distractor + spurious evidence; 0 if the method gave no scores."""
    d = row["methods"].get(method, {}) if method else {}
    if "scores" not in d:
        return 0.0
    pl = mt.plausibility(d["scores"], row["kinds"])
    u = pl["mass_distractor"] + pl["mass_spurious"]  # nan when all scores are zero
    return float(u) if np.isfinite(u) else 0.0


@contextmanager
def link(params):
    """Temporarily replace the health->state link (A1, A2, B) used by xfmss.ugf (sensitivity analysis)."""
    old = (ugf.A1, ugf.A2, ugf.B)
    if params is not None:
        ugf.A1, ugf.A2, ugf.B = params
    try:
        yield
    finally:
        ugf.A1, ugf.A2, ugf.B = old


def index_rows(rows: list[dict]) -> dict:
    """{(system, asset): [rows sorted by variant]} for one model."""
    out: dict = {}
    for r in rows:
        out.setdefault((r["system"], r["comp"]), []).append(r)
    return {k: sorted(v, key=lambda r: r["variant"]) for k, v in out.items()}


def U_stats(rows: list[dict], methods: list[str], screen_q: float = 0.7) -> dict:
    """Per explainer: mean U (width-matched control) and the screening threshold (screen_q quantile of U)."""
    out = {}
    for m in methods:
        us = [nonrelevant_mass(r, m) for r in rows if "scores" in r["methods"].get(m, {})]
        out[m] = {"mean": float(np.mean(us)) if us else 0.0,
                  "thr": float(np.quantile(us, screen_q)) if us else np.inf}
    return out


def specs(methods: list[str]) -> list[dict]:
    """All estimators compared in RQ2b."""
    out = [{"label": "crisp", "est": "crisp", "method": None, "family": "baseline"},
           {"label": "t1", "est": "t1", "method": None, "family": "baseline"},
           {"label": "prob_mc", "est": "prob_mc", "method": None, "family": "baseline"},
           {"label": "it2_plain", "est": "it2", "method": None, "family": "baseline"}]
    for m in methods:
        out.append({"label": f"it2_xai:{m}", "est": "it2", "method": m, "family": "xai"})
        out.append({"label": f"it2_const:{m}", "est": "it2_const", "method": m, "family": "control"})
        out.append({"label": f"it2_screen:{m}", "est": "it2_screen", "method": m, "family": "screen"})
        out.append({"label": f"it2_screen_rand:{m}", "est": "it2_screen_rand", "method": m,
                    "family": "screen_control"})
    return out


def _degenerate(h: float) -> IT2:
    cut = [(h, h)] * len(ALPHAS)
    return IT2(list(cut), list(cut))


def asset_it2(rs: list[dict], spec: dict, cfg: dict, Ust: dict, rng) -> IT2:
    P = [term_probs(r, cfg["prob"]) for r in rs]
    if spec["est"] == "crisp":
        return _degenerate(float(np.mean([p @ TERM_CENTERS for p in P])))
    if spec["est"] == "t1":
        return IT2.from_t1_list([t1_from_dist(np.mean(P, 0), U=0.0, kappa=cfg["kappa"])])
    m = spec["method"]
    Us = [nonrelevant_mass(r, m) for r in rs]
    if spec["est"] in ("it2_screen", "it2_screen_rand"):
        keep = [i for i, u in enumerate(Us) if u <= Ust[m]["thr"]] or [int(np.argmin(Us))]
        if spec["est"] == "it2_screen_rand":
            keep = sorted(rng.choice(len(rs), size=len(keep), replace=False).tolist())
        return IT2.from_t1_list([t1_from_dist(P[i], kappa=cfg["kappa"]) for i in keep])
    if spec["est"] == "it2_const":
        Us = [Ust[m]["mean"]] * len(rs)
    return IT2.from_t1_list([t1_from_dist(p, U=u, lam=cfg["lam"], kappa=cfg["kappa"]) for p, u in zip(P, Us)])


def _mc_health(rs: list[dict], prob: str, n: int, rng) -> np.ndarray:
    P = np.array([term_probs(r, prob) for r in rs])
    v = rng.integers(len(rs), size=n)
    t = np.minimum((rng.random(n)[:, None] > P[v].cumsum(1)).sum(1), len(TERM_CENTERS) - 1)
    return np.clip(TERM_CENTERS[t] + rng.uniform(-HALF_STEP, HALF_STEP, n), 0.0, 1.0)


def truth(system: dict, link_params) -> tuple[float, dict, str]:
    """True availability and overhaul gains; recomputed from h_true when the link is changed."""
    if link_params is None:
        return system["A_true"], system["gain_true"], system["best_true"]
    h = {c["id"]: c["h_true"] for s in system["subsystems"] for c in s["components"]}
    gains = ugf.improvement_gains(system, h)
    return ugf.availability(system, h), gains, max(gains, key=gains.get)


def _priority(system: dict, h_point: dict, gain_true: dict, best_true: str) -> dict:
    gains = ugf.improvement_gains(system, h_point)
    chosen = max(gains, key=gains.get)
    return {"chosen": chosen, "hit": chosen == best_true, "regret": gain_true[best_true] - gain_true[chosen]}


def evaluate_plant(system: dict, by_asset: dict, spec: dict, cfg: dict, Ust: dict, rng) -> tuple[dict, list]:
    """One estimator on one plant -> (plant record, asset records). Requires all assets to have rows."""
    comps = [c for s in system["subsystems"] for c in s["components"]]
    rs_of = {c["id"]: by_asset[(system["id"], c["id"])][: cfg["n_var"]] for c in comps}
    k = int(round(cfg["cut"] * (len(ALPHAS) - 1)))
    q = (cfg["iv_alpha"] / 2, 1 - cfg["iv_alpha"] / 2)
    with link(cfg["link"]):
        A_true, gain_true, best_true = truth(system, cfg["link"])
        assets = []
        if spec["est"] == "prob_mc":
            H = {cid: _mc_health(rs, cfg["prob"], cfg["mc_n"], rng) for cid, rs in rs_of.items()}
            A = np.array([ugf.availability(system, {cid: H[cid][i] for cid in H}) for i in range(cfg["mc_n"])])
            A_hat, (lo, hi) = float(A.mean()), np.quantile(A, q)
            h_point = {cid: float(h.mean()) for cid, h in H.items()}
            for c in comps:
                alo, ahi = np.quantile(H[c["id"]], q)
                assets.append((c, h_point[c["id"]], float(alo), float(ahi)))
        else:
            fz = {cid: asset_it2(rs, spec, cfg, Ust, rng) for cid, rs in rs_of.items()}
            out = propagate_monotone(fz, lambda h: ugf.availability(system, h))
            A_hat, (lo, hi) = out.defuzz(), out.umf[k]
            h_point = {cid: f.defuzz() for cid, f in fz.items()}
            for c in comps:
                alo, ahi = fz[c["id"]].umf[k]
                assets.append((c, h_point[c["id"]], alo, ahi))
        pr = _priority(system, h_point, gain_true, best_true)
    base = {"system": system["id"], "estimator": spec["label"], "est": spec["est"], "method": spec["method"],
            "family": spec["family"]}
    plant = {**base, "A_true": A_true, "A_hat": A_hat, "abs_err": abs(A_hat - A_true), "lo": float(lo),
             "hi": float(hi), "covered": bool(lo - 1e-9 <= A_true <= hi + 1e-9), "width": float(hi - lo),
             "interval_score": interval_score(lo, hi, A_true, alpha=cfg["iv_alpha"]), **pr}
    arecs = [{**base, "asset": c["id"], "h_true": c["h_true"], "h_hat": h, "abs_err": abs(h - c["h_true"]),
              "covered": bool(alo - 1e-9 <= c["h_true"] <= ahi + 1e-9), "width": ahi - alo,
              "interval_score": interval_score(alo, ahi, c["h_true"], alpha=cfg["iv_alpha"])}
             for c, h, alo, ahi in assets]
    return plant, arecs


def complete_plants(bench: dict, by_asset: dict, n_var: int) -> list[str]:
    """Plants whose every asset has at least n_var variants (only those can be propagated)."""
    out = []
    for sid, s in bench.items():
        if all(len(by_asset.get((sid, c["id"]), [])) >= n_var for sub in s["subsystems"] for c in sub["components"]):
            out.append(sid)
    return out
