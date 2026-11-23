"""Multi-state system (MSS) reliability via the Universal Generating Function (UGF).

A plant is a tree of nodes (system["structure"]):
    {"type": "series"|"parallel", "children": [...]}   series -> minimum (bottleneck), parallel -> sum
    {"asset": "WT01"}                                    a multi-state component
    {"cap": 10.0}                                        a perfectly reliable capacity limit (cable, export line)
Assets are looked up in system["subsystems"][*]["components"] (id, cap, degraded_fraction).

Each asset has 3 states {stopped, derated, full} with output {0, degraded_fraction*cap, cap}. In the synthetic
benchmark the state probabilities are a monotone function of a health index h in [0, 1] (ordinal-logit link), so
availability is non-decreasing in every asset's health; that lets fuzzy alpha-cuts of health map exactly onto
alpha-cuts of availability. With real data the state probabilities are measured from SCADA.
"""
from __future__ import annotations
import numpy as np

# ordinal-logit link parameters: P(X>=derated)=sig(A1+B h), P(X=full)=sig(A2+B h)
A1, A2, B = -0.5, -3.5, 6.0
DEGRADED_FRACTION = 0.5


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def state_probs(h: float) -> np.ndarray:
    """[P(stopped), P(derated), P(full)] for health h in [0,1]."""
    h = float(np.clip(h, 0.0, 1.0))
    ge1, ge2 = _sig(A1 + B * h), _sig(A2 + B * h)
    return np.array([1.0 - ge1, ge1 - ge2, ge2])


def health_from_probs(probs) -> float:
    """Health-equivalent of an observed state distribution (least-squares inverse of state_probs)."""
    grid = np.linspace(0, 1, 1001)
    P = np.array([state_probs(h) for h in grid])
    return float(grid[np.argmin(((P - np.asarray(probs, float)) ** 2).sum(1))])


def _compose(u1: dict, u2: dict, op) -> dict:
    out: dict = {}
    for g1, p1 in u1.items():
        for g2, p2 in u2.items():
            g = op(g1, g2)
            out[g] = out.get(g, 0.0) + p1 * p2
    return out


def _asset_index(system: dict) -> dict:
    return {c["id"]: c for s in system["subsystems"] for c in s["components"]}


def asset_ugf_from_probs(cap: float, probs, degraded_fraction: float = DEGRADED_FRACTION) -> dict:
    u: dict = {}
    for g, q in zip([0.0, degraded_fraction * cap, float(cap)], probs):
        u[round(g, 6)] = u.get(round(g, 6), 0.0) + float(q)
    return u


def _node_ugf(node: dict, assets: dict, probs: dict) -> dict:
    if "asset" in node:
        a = assets[node["asset"]]
        return asset_ugf_from_probs(a["cap"], probs[node["asset"]], a.get("degraded_fraction", DEGRADED_FRACTION))
    if "cap" in node:
        return {float(node["cap"]): 1.0}
    op = min if node["type"] == "series" else (lambda x, y: round(x + y, 6))
    out = None
    for ch in node["children"]:
        u = _node_ugf(ch, assets, probs)
        out = u if out is None else _compose(out, u, op)
    return out


def availability_from_probs(system: dict, probs: dict, demand: float | None = None) -> float:
    """P(plant output >= demand) when each asset's [P(stopped), P(derated), P(full)] is given directly
    (e.g. measured from SCADA). Used for the true availability."""
    w = system["demand"] if demand is None else demand
    u = _node_ugf(system["structure"], _asset_index(system), probs)
    return float(sum(p for g, p in u.items() if g >= w - 1e-9))


def availability(system: dict, health: dict, demand: float | None = None) -> float:
    """Availability from health indices (monotone in each h)."""
    return availability_from_probs(system, {k: state_probs(h) for k, h in health.items()}, demand)


def component_ids(system: dict) -> list[str]:
    return [c["id"] for s in system["subsystems"] for c in s["components"]]


def improvement_gains(system: dict, health: dict, overhaul_to: float = 0.95) -> dict:
    """Availability gain from overhauling each component (h -> max(h, overhaul_to))."""
    base = availability(system, health)
    gains = {}
    for cid in component_ids(system):
        h2 = dict(health)
        h2[cid] = max(h2[cid], overhaul_to)
        gains[cid] = availability(system, h2) - base
    return gains


def birnbaum_importance(system: dict, health: dict, eps: float = 1e-3) -> dict:
    """Numerical sensitivity dA/dh_j (a continuous Birnbaum-type importance)."""
    out = {}
    for cid in component_ids(system):
        hp, hm = dict(health), dict(health)
        hp[cid] = min(1.0, health[cid] + eps)
        hm[cid] = max(0.0, health[cid] - eps)
        out[cid] = (availability(system, hp) - availability(system, hm)) / (hp[cid] - hm[cid])
    return out
