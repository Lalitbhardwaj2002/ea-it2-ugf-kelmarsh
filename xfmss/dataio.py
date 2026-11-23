"""Excel data layer: the synthetic benchmark and REAL plant data use the same workbook format.

    export_excel(bench, "wind_farm_data_template.xlsx")   # writes the synthetic data in template form
    bench = load_excel("my_real_plant.xlsx")              # same structure, ready for run_experiment.py

Sheets
  Plants       plant_id, name, unit, demand, data_source
  Topology     plant_id, node_id, parent_id, node_type (series|parallel|asset|cap), asset_id, capacity
  Assets       plant_id, asset_id, asset_type, group, rated_capacity, unit, degraded_fraction
  Evidence     plant_id, asset_id, cue_id, kind (causal|distractor|spurious|unlabelled), value,
               template_1, template_2, template_3   ({v} is replaced by value; or write a literal sentence)
  GroundTruth  plant_id, asset_id, P_stopped, P_derated, P_full, period, source
               (for real data: fraction of time each asset spent stopped / derated / at full output,
                e.g. from 10-min SCADA power + status codes over the assessment period)
"""
from __future__ import annotations
import itertools
import pandas as pd

from .ugf import availability_from_probs, health_from_probs, improvement_gains, birnbaum_importance

README = [
    ["How to replace the synthetic data with real plant data"],
    ["1. Plants: one row per plant; demand = contracted / required export (same unit as capacities)."],
    ["2. Topology: describe the plant as a tree. series = bottleneck (min), parallel = capacities add, "
     "asset = a turbine/inverter/transformer (asset_id), cap = fixed limit such as a cable or export line (capacity)."],
    ["3. Assets: every asset with its rated capacity; degraded_fraction = output share in the derated state (e.g. 0.5)."],
    ["4. Evidence: one row per sentence of the condition report (SCADA alarms, CMS readings, maintenance-log facts). "
     "Put a reference value in the sentence where possible (alarm limit, fleet average, design life)."],
    ["   kind = causal (relevant to health), distractor (irrelevant), spurious (cosmetic but suggestive); "
     "use 'unlabelled' if unknown and let the expert sheet decide."],
    ["5. GroundTruth: observed state probabilities per asset over the period (they must sum to 1)."],
    ["   Suggested SCADA rule: full = power >= 90% of the power-curve expectation; derated = 10-90%; "
     "stopped = <10% or unavailable status, using only times with wind above cut-in."],
    ["6. Run: python -c \"from xfmss.dataio import load_excel; from xfmss.benchmark import save; "
     "save(load_excel('my_plant.xlsx'), 'data/benchmark.json')\""],
]


def export_excel(bench: list[dict], path: str):
    plants, topo, assets, ev, gt = [], [], [], [], []
    for p in bench:
        plants.append({"plant_id": p["id"], "name": p.get("name", p["id"]), "unit": p.get("unit", "units"),
                       "demand": p["demand"], "data_source": p.get("data_source", "")})
        cnt = itertools.count(1)

        def walk(node, parent):
            nid = f"N{next(cnt)}"
            if "asset" in node:
                topo.append({"plant_id": p["id"], "node_id": nid, "parent_id": parent, "node_type": "asset",
                             "asset_id": node["asset"], "capacity": None})
            elif "cap" in node:
                topo.append({"plant_id": p["id"], "node_id": nid, "parent_id": parent, "node_type": "cap",
                             "asset_id": None, "capacity": node["cap"]})
            else:
                topo.append({"plant_id": p["id"], "node_id": nid, "parent_id": parent, "node_type": node["type"],
                             "asset_id": None, "capacity": None})
                for ch in node["children"]:
                    walk(ch, nid)
        walk(p["structure"], None)
        for s in p["subsystems"]:
            for c in s["components"]:
                assets.append({"plant_id": p["id"], "asset_id": c["id"], "asset_type": c["type"], "group": s["name"],
                               "rated_capacity": c["cap"], "unit": c.get("unit", ""),
                               "degraded_fraction": c.get("degraded_fraction", 0.5)})
                for cue in c["cues"]:
                    tpl = cue.get("templates") or [cue.get("text", "")]
                    ev.append({"plant_id": p["id"], "asset_id": c["id"], "cue_id": cue["cue_id"], "kind": cue["kind"],
                               "value": None if cue.get("value") is None else str(cue.get("value")),
                               **{f"template_{i + 1}": (tpl[i] if i < len(tpl) else None)
                                                             for i in range(3)}})
                sp = c["state_probs_true"]
                gt.append({"plant_id": p["id"], "asset_id": c["id"], "P_stopped": sp[0], "P_derated": sp[1],
                           "P_full": sp[2], "period": "synthetic", "source": p.get("data_source", "")})
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame(README, columns=["Instructions"]).to_excel(xw, sheet_name="README", index=False)
        for name, rows in [("Plants", plants), ("Topology", topo), ("Assets", assets), ("Evidence", ev),
                           ("GroundTruth", gt)]:
            pd.DataFrame(rows).to_excel(xw, sheet_name=name, index=False)


def load_excel(path: str) -> list[dict]:
    x = pd.read_excel(path, sheet_name=None)
    plants = []
    for _, pr in x["Plants"].iterrows():
        pid = pr["plant_id"]
        A = x["Assets"][x["Assets"].plant_id == pid]
        E = x["Evidence"][x["Evidence"].plant_id == pid]
        G = x["GroundTruth"][x["GroundTruth"].plant_id == pid].set_index("asset_id")
        T = x["Topology"][x["Topology"].plant_id == pid]
        groups: dict = {}
        for _, a in A.iterrows():
            aid = str(a["asset_id"])
            if aid not in G.index:
                raise ValueError(f"GroundTruth is missing asset {aid} of plant {pid}")
            g = G.loc[aid]
            probs = [float(g["P_stopped"]), float(g["P_derated"]), float(g["P_full"])]
            tot = sum(probs)
            if abs(tot - 1) > 0.02:
                raise ValueError(f"State probabilities of {aid} sum to {tot:.3f}, not 1")
            probs = [q / tot for q in probs]
            cues = []
            for _, e in E[E.asset_id == a["asset_id"]].iterrows():
                tpl = [str(e[c]) for c in ("template_1", "template_2", "template_3")
                       if c in e and isinstance(e[c], str) and e[c].strip()]
                val = e["value"]
                if isinstance(val, float) and val.is_integer():
                    val = int(val)
                cues.append({"cue_id": str(e["cue_id"]), "kind": str(e.get("kind", "unlabelled")),
                             "value": None if pd.isna(val) else val, "templates": tpl})
            comp = {"id": aid, "type": str(a["asset_type"]), "cap": float(a["rated_capacity"]),
                    "unit": str(a.get("unit", "")), "degraded_fraction": float(a.get("degraded_fraction", 0.5)),
                    "state_probs_true": probs, "h_true": round(health_from_probs(probs), 3), "cues": cues}
            groups.setdefault(str(a.get("group", "All assets")), []).append(comp)

        nodes = {r["node_id"]: r for _, r in T.iterrows()}
        kids: dict = {}
        for _, r in T.iterrows():
            kids.setdefault(r["parent_id"] if pd.notna(r["parent_id"]) else None, []).append(r["node_id"])

        def build(nid):
            r = nodes[nid]
            if r["node_type"] == "asset":
                return {"asset": str(r["asset_id"])}
            if r["node_type"] == "cap":
                return {"cap": float(r["capacity"])}
            return {"type": r["node_type"], "children": [build(c) for c in kids.get(nid, [])]}
        roots = kids.get(None, [])
        if len(roots) != 1:
            raise ValueError(f"Topology of {pid} must have exactly one root node (parent_id empty)")
        plant = {"id": str(pid), "name": str(pr.get("name", pid)), "unit": str(pr.get("unit", "")),
                 "demand": float(pr["demand"]), "data_source": str(pr.get("data_source", "")),
                 "subsystems": [{"name": k, "components": v} for k, v in groups.items()],
                 "structure": build(roots[0])}
        probs = {c["id"]: c["state_probs_true"] for v in groups.values() for c in v}
        health = {c["id"]: c["h_true"] for v in groups.values() for c in v}
        plant["installed_mw"] = sum(c["cap"] for v in groups.values() for c in v)
        plant["A_true"] = availability_from_probs(plant, probs)
        gains = improvement_gains(plant, health)
        plant["gain_true"] = {k: round(v, 5) for k, v in gains.items()}
        plant["best_true"] = max(gains, key=gains.get)
        plant["birnbaum_true"] = {k: round(v, 5) for k, v in birnbaum_importance(plant, health).items()}
        plants.append(plant)
    return plants
