"""Benchmark helpers shared by the synthetic wind farms (windfarm.py) and real plant data (dataio.py).

Every asset carries a list of cues (evidence facts). A cue has a stable cue_id, a kind (causal, distractor,
spurious or unlabelled) and either up to 3 wording templates with a {v} placeholder or a literal text.
component_segments() turns an asset into the ordered sentences of one paraphrase variant; explanations are
compared across variants through the stable cue ids.
"""
from __future__ import annotations
import json
import zlib
import numpy as np


def render_cue(cue: dict, variant: int) -> str:
    """Sentence for one cue in one variant: 'templates' = up to 3 wordings with {v}, or a literal 'text'."""
    if cue.get("templates"):
        t = cue["templates"][variant % len(cue["templates"])]
        return t.format(v=cue.get("value", "")) if "{v}" in t else t
    return cue["text"]


def system_seed(system_id: str) -> int:
    """Stable integer seed from any plant id (works for real ids like 'kelmarsh_2021')."""
    return zlib.crc32(str(system_id).encode()) % 100000


def iter_components(system: dict):
    for sub in system["subsystems"]:
        for c in sub["components"]:
            yield c


def component_segments(comp: dict, variant: int, seed: int = 0) -> list[dict]:
    """Ordered evidence segments for one paraphrase variant.

    Every asset and every variant gets its own random order (seeded by plant, asset id and variant), so a
    sentence's kind cannot be inferred from its position: with one fixed order, position-biased explainers
    would look plausible for free.
    """
    rng = np.random.default_rng([int(seed), zlib.crc32(str(comp["id"]).encode()), int(variant)])
    order = rng.permutation(len(comp["cues"]))
    return [{"text": render_cue(comp["cues"][i], variant), "cue_id": comp["cues"][i]["cue_id"],
             "kind": comp["cues"][i]["kind"]} for i in order]


def save(bench, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bench, f, indent=1)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
