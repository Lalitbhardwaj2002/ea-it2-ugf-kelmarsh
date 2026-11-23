"""Explanation-quality metrics (segment level).

faithfulness  - comprehensiveness / sufficiency AOPC with segment removal
plausibility  - do top-ranked segments match the causal (should-matter) evidence?
spuriousness  - share of |attribution| on spurious / distractor segments
agreement     - Kendall tau and top-k overlap between two explainers
stability     - agreement of the same explainer across paraphrase variants (by cue id)
"""
from __future__ import annotations
import numpy as np
from scipy.stats import kendalltau


def ranking(scores) -> np.ndarray:
    s = np.abs(np.asarray(scores, float))
    return np.argsort(-s, kind="stable")


def faithfulness_masks(scores, ks=(1, 2, 3)):
    """Masks needed for comprehensiveness (remove top-k) and sufficiency (keep only top-k)."""
    r = ranking(scores)
    M = len(r)
    out = []
    for k in ks:
        k = min(k, M)
        rem = np.ones(M, bool); rem[r[:k]] = False
        keep = np.zeros(M, bool); keep[r[:k]] = True
        out.append((k, rem, keep))
    return out


def faithfulness(E_full, E_rem: dict, E_keep: dict, ks=(1, 2, 3)):
    comp = [abs(E_full - E_rem[k]) for k in ks]
    suff = [abs(E_full - E_keep[k]) for k in ks]
    return {"comp_aopc": float(np.mean(comp)), "suff_aopc": float(np.mean(suff)),
            "comp_k": [float(x) for x in comp], "suff_k": [float(x) for x in suff]}


def plausibility(scores, kinds) -> dict:
    kinds = np.asarray(kinds)
    causal = kinds == "causal"
    n_c = int(causal.sum())
    r = ranking(scores)
    prec = float(causal[r[:n_c]].mean()) if n_c else np.nan
    hits, ap = 0, 0.0
    for i, j in enumerate(r, 1):
        if causal[j]:
            hits += 1
            ap += hits / i
    ap = ap / n_c if n_c else np.nan
    a = np.abs(np.asarray(scores, float))
    tot = a.sum()
    share = lambda m: float(a[m].sum() / tot) if tot > 0 else np.nan
    return {"prec_at_c": prec, "avg_prec": float(ap), "mass_causal": share(causal),
            "mass_distractor": share(kinds == "distractor"), "mass_spurious": share(kinds == "spurious"),
            "has_spurious": bool((kinds == "spurious").any())}


def agreement(s1, s2, k: int = 3) -> dict:
    a1, a2 = np.abs(np.asarray(s1, float)), np.abs(np.asarray(s2, float))
    if np.allclose(a1, a1[0]) or np.allclose(a2, a2[0]):
        tau = np.nan
    else:
        tau = float(kendalltau(a1, a2).statistic)
    t1, t2 = set(ranking(s1)[:k]), set(ranking(s2)[:k])
    return {"kendall": tau, "topk_jaccard": len(t1 & t2) / len(t1 | t2)}


def align_by_cue(scores, cue_ids, ref_cue_ids):
    d = dict(zip(cue_ids, scores))
    return np.array([d.get(c, 0.0) for c in ref_cue_ids])
