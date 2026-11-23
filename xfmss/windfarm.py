"""Wind-farm benchmark (synthetic placeholder for real plant data).

Plant model (multi-state system):
    Plant output = min( sum over feeders [ min( sum of turbines on feeder, feeder cable rating ) ],
                        sum of substation transformers )
    Each turbine / transformer has 3 states: stopped (0), derated (50 %), full (100 % of rating).
    Plant availability = P(plant output >= contracted export demand), computed with the UGF.

Every asset gets a condition dossier (SCADA / CMS / maintenance-log style sentences):
    causal      - generated from the asset's true health; each has an engineering reference value
    distractor  - irrelevant facts (OEM name, tower colour, asset tag, location)
    spurious    - cosmetic cues that point the WRONG way (e.g. "repainted, looks new" on a degraded unit)

ALL of this is a placeholder. The same JSON structure is produced from real data by
xfmss.dataio.load_excel() (fill templates/wind_farm_data_template.xlsx with SCADA/CMMS data).
"""
from __future__ import annotations
import numpy as np

from .ugf import state_probs, availability, availability_from_probs, improvement_gains, birnbaum_importance

TURBINE_CAUSAL = {
    "age": (lambda d, r: int(np.clip(round(1 + 22 * d + r.normal(0, 1.5)), 1, 28)),
            ["The turbine has been operating for {v} years (design life: 25 years).",
             "It was commissioned {v} years ago; its design life is 25 years.",
             "Its operating age is {v} years against a 25-year design life."]),
    "gearbox_vibration": (lambda d, r: round(float(np.clip(1.0 + 6.5 * d + r.normal(0, 0.3), 0.4, 10)), 1),
            ["Gearbox vibration velocity is {v} mm/s (site alarm limit: 6.0 mm/s).",
             "The condition-monitoring system reports {v} mm/s RMS gearbox vibration against a 6.0 mm/s alarm limit.",
             "Gearbox vibration was measured at {v} mm/s; the alarm threshold is 6.0 mm/s."]),
    "oil_debris": (lambda d, r: int(np.clip(round(1 + 18 * d + r.normal(0, 1.5)), 0, 30)),
            ["The gearbox oil debris sensor counts {v} ferrous particles per hour (alarm: 12 per hour).",
             "Online oil-debris monitoring shows {v} metal particles per hour; the alarm level is 12.",
             "Ferrous debris in the gearbox oil is {v} particles per hour versus an alarm level of 12."]),
    "generator_bearing_temp": (lambda d, r: int(np.clip(round(60 + 40 * d + r.normal(0, 2)), 45, 110)),
            ["The generator bearing runs at {v} °C (alarm: 95 °C).",
             "SCADA shows a generator bearing temperature of {v} °C against a 95 °C alarm.",
             "Generator bearing temperature averages {v} °C; the alarm setting is 95 °C."]),
    "fault_stops": (lambda d, r: int(r.poisson(0.5 + 9 * d)),
            ["Converter and pitch faults stopped the turbine {v} times in the last 12 months (fleet average: 3).",
             "The fault log records {v} converter/pitch trips over the past year; similar turbines average 3.",
             "There were {v} fault-related stops in the last 12 months, compared with a fleet average of 3."]),
    "availability_12m": (lambda d, r: round(float(np.clip(99.2 - 10 * d + r.normal(0, 0.6), 80, 99.8)), 1),
            ["Its time-based availability over the last 12 months was {v}% (fleet average: 97%).",
             "The turbine was available {v}% of the time last year; the fleet averages 97%.",
             "Twelve-month availability stands at {v}% against a 97% fleet average."]),
    "service": (lambda d, r: int(np.clip(round(2 + 22 * d + r.normal(0, 2)), 1, 36)),
            ["The last scheduled service was {v} months ago (recommended interval: 12 months).",
             "It has run {v} months since its last scheduled service; the recommended interval is 12 months.",
             "Scheduled maintenance was last done {v} months ago, against a 12-month interval."]),
}
TRANSFORMER_CAUSAL = {
    "age": (lambda d, r: int(np.clip(round(3 + 37 * d + r.normal(0, 2)), 1, 45)),
            ["The transformer has been in service for {v} years (design life: 40 years).",
             "It was energised {v} years ago; its design life is 40 years.",
             "Its service age is {v} years against a 40-year design life."]),
    "dga_tdcg": (lambda d, r: int(np.clip(round(150 + 1800 * d + r.normal(0, 60)), 50, 2500)),
            ["Dissolved-gas analysis shows {v} ppm total combustible gas (site caution limit: 720 ppm).",
             "The latest oil DGA reports {v} ppm of combustible gases against a 720 ppm caution limit.",
             "Total dissolved combustible gas is {v} ppm; the caution threshold is 720 ppm."]),
    "top_oil_temp": (lambda d, r: int(np.clip(round(55 + 35 * d + r.normal(0, 2)), 40, 105)),
            ["Top-oil temperature peaks at {v} °C (alarm: 95 °C).",
             "The top-oil temperature reached {v} °C against a 95 °C alarm.",
             "Peak top-oil temperature is {v} °C; the alarm setting is 95 °C."]),
    "loading": (lambda d, r: int(np.clip(round(60 + 45 * d + r.normal(0, 3)), 40, 120)),
            ["It typically carries {v}% of its rated load (recommended maximum: 90%).",
             "Average loading is {v}% of nameplate rating; the recommended maximum is 90%.",
             "The unit runs at {v}% of rated capacity versus a recommended 90% maximum."]),
    "oil_moisture": (lambda d, r: int(np.clip(round(8 + 30 * d + r.normal(0, 2)), 3, 50)),
            ["Water content in the insulating oil is {v} ppm (limit: 25 ppm).",
             "The oil moisture test shows {v} ppm water against a 25 ppm limit.",
             "Insulating-oil moisture is {v} ppm; the limit is 25 ppm."]),
}
# Distractor and spurious sentences are written to match the causal ones in length (about 11-16 words) and all
# contain a number, so neither sentence length nor "has a number" reveals which sentences matter (causal
# sentences average 13 words and always carry a number).
DISTRACTORS = {
    "colour": ([], ["The tower is painted {v}, the colour set out in the 2012 planning consent.",
                    "Its tower finish is {v}, as specified in the 2012 planning consent for the site.",
                    "Under the 2012 planning consent, the tower structure has a {v} coating."],
               ["light grey", "off-white", "pale green", "white"]),
    "oem": ([], ["It was supplied by {v} under a framework contract signed in 2014.",
                 "The original equipment manufacturer is {v}, which delivered it under a 2014 contract.",
                 "{v} delivered the unit in 2014 under the site's framework supply contract."],
            ["Northwind Turbines", "Aerovane Energy", "Halden Power Systems", "Voltara Electric"]),
    "tag": ([], ["Its asset tag is {v}, recorded under cost centre 3300 in the maintenance system.",
                 "It is registered in the asset database as {v}, with cost centre 3300.",
                 "The plant tag number is {v}; its cost centre in the maintenance system is 3300."],
            ["WF-1041", "AS-2290", "PT-7713", "GN-5521", "EQ-3306"]),
    "location": ([], ["It stands at {v}, about 350 m from the site office.",
                      "The unit is located at {v}, roughly 350 m from the site office.",
                      "Its position is {v}, around 350 m from the site office."],
                 ["row 2, position 4 of the array", "the north-east corner of the site",
                  "the ridge line near the access road"]),
}
TX_DISTRACTOR_OVERRIDES = {
    "colour": (["The tank is painted {v}, matching the 2015 substation design specification.",
                "Its tank finish is {v}, as required by the 2015 substation design specification.",
                "Following the 2015 substation design specification, the enclosure has a {v} coating."],
               ["light grey", "dark green", "white"]),
    "location": (["It is installed in {v}, which is 40 m from the control building.",
                  "The unit sits in {v}, about 40 m from the control building.",
                  "Its position is {v}, roughly 40 m from the control building."],
                 ["bay 1 of the substation compound", "bay 2 of the substation compound"]),
}
SPURIOUS_GOOD = ["The blades were washed and the tower repainted 3 weeks ago, so it looks almost new.",
                 "It featured in 4 photographs of the company's 2024 annual report on the site.",
                 "A new LED aviation warning light was fitted on its nacelle 2 months ago.",
                 "The 1.2 km access road to the unit was resurfaced with fresh asphalt recently."]
TX_SPURIOUS_GOOD = ["The tank and the substation fence were repainted 3 weeks ago, so it looks almost new.",
                    "It featured in 4 photographs of the company's 2024 annual report on the substation.",
                    "New nameplates and 12 safety signs were fitted around the unit this year.",
                    "The substation control room was refurbished in 2025 with modern furniture and lighting."]
TX_SPURIOUS_BAD = ["The tank paint is faded and stained after 2 winters without cleaning.",
                   "It is an older design that the manufacturer stopped marketing in 2016.",
                   "The gravel in the substation compound is overgrown, with weeds about 30 cm high.",
                   "Its nameplate is scratched, and 2 of its rating lines are hard to read."]
SPURIOUS_BAD = ["The tower paint is faded and streaked with dirt after 2 winters without cleaning.",
                "It is an older model that the manufacturer stopped marketing in 2016.",
                "The signage on its access door is worn, and 3 of the labels are unreadable.",
                "Visitors on the 2023 open day described the unit as old-fashioned in appearance."]


def _cues(bank: dict, h: float, rng, n_causal: int, p_spurious: float, transformer: bool) -> list[dict]:
    d = 1.0 - h
    keys = [str(k) for k in rng.choice(list(bank), size=min(n_causal, len(bank)), replace=False)]
    cues = []
    for k in keys:
        gen, templates = bank[k]
        v = gen(d, rng)
        tpl = list(templates)
        if k == "fault_stops" and v == 1:
            tpl = [t.replace("{v} times", "{v} time").replace("trips", "trip").replace("stops", "stop") for t in tpl]
        cues.append({"cue_id": k, "kind": "causal", "value": v, "templates": tpl})
    dkeys = [str(k) for k in rng.choice(["colour", "oem", "tag", "location"], size=2, replace=False)]
    for k in dkeys:
        _, templates, values = DISTRACTORS[k]
        if transformer and k in TX_DISTRACTOR_OVERRIDES:
            templates, values = TX_DISTRACTOR_OVERRIDES[k]
        cues.append({"cue_id": k, "kind": "distractor", "value": str(rng.choice(values)), "templates": templates})
    if rng.random() < p_spurious:
        if transformer:
            pool = TX_SPURIOUS_GOOD if h < 0.5 else TX_SPURIOUS_BAD
        else:
            pool = SPURIOUS_GOOD if h < 0.5 else SPURIOUS_BAD
        j = int(rng.integers(len(pool)))
        cues.append({"cue_id": f"spurious_{'good' if h < 0.5 else 'bad'}_{j}", "kind": "spurious", "text": pool[j]})
    return cues


def make_plant(pid: int, rng, n_feeders=(2, 3), per_feeder=(3, 5), turbine_mw=3.0, tx_mw=None,
               p_spurious: float = 0.5) -> dict:
    nf = int(rng.integers(n_feeders[0], n_feeders[1] + 1))
    feeders, feeder_nodes, k = [], [], 1
    for f in range(nf):
        comps = []
        nt = int(rng.integers(per_feeder[0], per_feeder[1] + 1))
        for _ in range(nt):
            h = float(rng.uniform(0.05, 0.95))
            comps.append({"id": f"WT{k:02d}", "type": "wind turbine", "cap": turbine_mw, "unit": "MW",
                          "h_true": round(h, 3), "cues": _cues(TURBINE_CAUSAL, h, rng, 5, p_spurious, False)})
            k += 1
        cable = round(0.8 * nt * turbine_mw, 1)  # feeder cable rated below the sum of turbine ratings
        feeders.append({"name": f"Feeder {f + 1}", "components": comps, "cable_mw": cable})
        feeder_nodes.append({"type": "series", "children": [
            {"type": "parallel", "children": [{"asset": c["id"]} for c in comps]}, {"cap": cable}]})
    installed = sum(c["cap"] for f in feeders for c in f["components"])
    tx_cap = tx_mw or round(0.6 * installed)  # two transformers, each ~60 % of installed capacity
    txs = []
    for j in range(2):
        h = float(rng.uniform(0.05, 0.95))
        txs.append({"id": f"TX{j + 1}", "type": "substation transformer", "cap": float(tx_cap), "unit": "MW",
                    "h_true": round(h, 3), "cues": _cues(TRANSFORMER_CAUSAL, h, rng, 4, p_spurious, True)})
    subsystems = feeders + [{"name": "Substation", "components": txs}]
    structure = {"type": "series", "children": [
        {"type": "parallel", "children": feeder_nodes},
        {"type": "parallel", "children": [{"asset": t["id"]} for t in txs]}]}
    demand = round(float(rng.uniform(0.45, 0.65)) * installed / 1.5) * 1.5
    plant = {"id": f"wf{pid:02d}", "name": f"Synthetic wind farm {pid + 1}", "unit": "MW",
             "installed_mw": installed, "subsystems": subsystems, "structure": structure, "demand": demand,
             "data_source": "synthetic placeholder (replace with real SCADA/CMMS data)"}
    health = {c["id"]: c["h_true"] for s in subsystems for c in s["components"]}
    for s in subsystems:
        for c in s["components"]:
            c["state_probs_true"] = [round(float(x), 5) for x in state_probs(c["h_true"])]
    plant["A_true"] = availability_from_probs(plant, {c["id"]: c["state_probs_true"] for s in subsystems
                                                      for c in s["components"]})
    gains = improvement_gains(plant, health)
    plant["gain_true"] = {kk: round(v, 5) for kk, v in gains.items()}
    plant["best_true"] = max(gains, key=gains.get)
    plant["birnbaum_true"] = {kk: round(v, 5) for kk, v in birnbaum_importance(plant, health).items()}
    return plant


def make_windfarm_benchmark(n_plants: int = 6, seed: int = 11) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [make_plant(i, rng) for i in range(n_plants)]
