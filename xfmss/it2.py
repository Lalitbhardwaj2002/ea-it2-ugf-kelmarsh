"""Interval type-2 (IT2) fuzzy numbers for LLM-elicited health judgments.

Pipeline
  1. Each LLM "subject" (one paraphrase / prompt variant) gives a probability
     distribution p over a 7-term linguistic health scale.
  2. That distribution becomes a type-1 triangular fuzzy number (T1 TFN)
     centred on the expected term value, with spread driven by the
     distribution's dispersion and (optionally) by an explanation-unreliability
     score U in [0,1]  ->  "explanation-aware" spread.
  3. The T1 TFNs of all subjects form an IT2 fuzzy number:
        UMF = envelope (union) of the subjects' TFNs,
        LMF = intersection of the subjects' TFNs
     (in the spirit of Mendel's Interval Approach: words from many subjects -> FOU).
  4. Alpha-cuts propagate exactly through any monotone non-decreasing system
     function (e.g. MSS availability) by evaluating at cut end-points.
  5. Type reduction: exact Karnik-Mendel centroid on a grid.
"""
from __future__ import annotations
import numpy as np

TERMS = ["A", "B", "C", "D", "E", "F", "G"]
TERM_NAMES = ["very poor", "poor", "fairly poor", "moderate", "fairly good", "good", "excellent"]
TERM_CENTERS = np.linspace(0.0, 1.0, len(TERMS))
ALPHAS = np.linspace(0.0, 1.0, 21)


def health_to_term(h: float) -> int:
    return int(np.argmin(np.abs(TERM_CENTERS - h)))


def dist_summary(p) -> tuple[float, float]:
    p = np.asarray(p, float)
    p = p / p.sum()
    c = float(p @ TERM_CENTERS)
    s = float(np.sqrt(p @ (TERM_CENTERS - c) ** 2))
    return c, s


def t1_from_dist(p, U: float = 0.0, lam: float = 1.0, kappa: float = 1.5,
                 delta_min: float = 1 / 12) -> tuple[float, float, float]:
    """T1 TFN (a, c, b) from a term distribution; spread grows with dispersion and U."""
    c, s = dist_summary(p)
    d = kappa * max(s, delta_min) * (1.0 + lam * float(U))
    return (max(0.0, c - d), c, min(1.0, c + d))


def tfn_cut(t, a):
    l, m, r = t
    return (l + a * (m - l), r - a * (r - m))


class IT2:
    """IT2 fuzzy number stored as alpha-cuts of its UMF and LMF.

    umf[k] = (lo, hi) at ALPHAS[k];  lmf[k] = (lo, hi) or None where the LMF cut is empty.
    """

    def __init__(self, umf, lmf):
        self.umf, self.lmf = umf, lmf

    @classmethod
    def from_t1_list(cls, tfns):
        umf, lmf = [], []
        for a in ALPHAS:
            cuts = [tfn_cut(t, a) for t in tfns]
            umf.append((min(c[0] for c in cuts), max(c[1] for c in cuts)))
            lo, hi = max(c[0] for c in cuts), min(c[1] for c in cuts)
            lmf.append((lo, hi) if lo <= hi + 1e-12 else None)
        return cls(umf, lmf)

    @property
    def lmf_height(self) -> float:
        hs = [a for a, c in zip(ALPHAS, self.lmf) if c is not None]
        return float(max(hs)) if hs else 0.0

    def support(self):
        return self.umf[0]

    def membership_grid(self, x):
        mu_u = np.zeros_like(x)
        mu_l = np.zeros_like(x)
        for a, cu, cl in zip(ALPHAS, self.umf, self.lmf):
            if a == 0:
                continue
            mu_u = np.where((x >= cu[0] - 1e-12) & (x <= cu[1] + 1e-12), np.maximum(mu_u, a), mu_u)
            if cl is not None:
                mu_l = np.where((x >= cl[0] - 1e-12) & (x <= cl[1] + 1e-12), np.maximum(mu_l, a), mu_l)
        return mu_u, np.minimum(mu_l, mu_u)

    def centroid(self, n: int = 801):
        """Exact Karnik-Mendel centroid interval [c_l, c_r] (brute-force switch point on a grid)."""
        lo, hi = self.support()
        if hi - lo < 1e-9:
            return lo, hi
        x = np.linspace(lo, hi, n)
        mu_u, mu_l = self.membership_grid(x)
        if mu_u.sum() == 0:
            m = (lo + hi) / 2
            return m, m
        cs_xu, cs_u = np.cumsum(x * mu_u), np.cumsum(mu_u)
        cs_xl, cs_l = np.cumsum(x * mu_l), np.cumsum(mu_l)
        tot_xu, tot_u, tot_xl, tot_l = cs_xu[-1], cs_u[-1], cs_xl[-1], cs_l[-1]
        # left: UMF on the left of switch k, LMF on the right
        num_l = cs_xu + (tot_xl - cs_xl)
        den_l = cs_u + (tot_l - cs_l)
        num_r = cs_xl + (tot_xu - cs_xu)
        den_r = cs_l + (tot_u - cs_u)
        with np.errstate(invalid="ignore", divide="ignore"):
            cl = np.nanmin(np.where(den_l > 0, num_l / den_l, np.nan))
            cr = np.nanmax(np.where(den_r > 0, num_r / den_r, np.nan))
        return float(cl), float(cr)

    def defuzz(self) -> float:
        cl, cr = self.centroid()
        return (cl + cr) / 2


def propagate_monotone(inputs: dict, f) -> IT2:
    """Propagate IT2 inputs {id: IT2} through f(dict_of_values) that is non-decreasing in each input."""
    keys = list(inputs)
    umf, lmf = [], []
    for k in range(len(ALPHAS)):
        lo = {i: inputs[i].umf[k][0] for i in keys}
        hi = {i: inputs[i].umf[k][1] for i in keys}
        umf.append((f(lo), f(hi)))
        if all(inputs[i].lmf[k] is not None for i in keys):
            lo = {i: inputs[i].lmf[k][0] for i in keys}
            hi = {i: inputs[i].lmf[k][1] for i in keys}
            lmf.append((f(lo), f(hi)))
        else:
            lmf.append(None)
    return IT2(umf, lmf)


def interval_score(lo: float, hi: float, y: float, alpha: float = 0.1) -> float:
    """Winkler/Gneiting-Raftery interval score (lower is better): width + penalty for missing y."""
    s = hi - lo
    if y < lo:
        s += 2 / alpha * (lo - y)
    elif y > hi:
        s += 2 / alpha * (y - hi)
    return float(s)
