"""Paper A main experiment: elicitation + explanations + faithfulness, one JSONL row per judgment.

Usage (Kaggle GPU; one process per GPU with --shard 0/2 and --shard 1/2):
    python run_experiment.py --model microsoft/Phi-3.5-mini-instruct --dtype float16 --variants 3 \
        --bench data/benchmark.json --out results/Phi-3.5-mini-instruct_shard0.jsonl --shard 0/2
Resumable: rows already in --out are skipped, so you can rerun after a session timeout.
"""
from __future__ import annotations
import argparse, json, os, time, warnings
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

warnings.filterwarnings("ignore")
from xfmss import benchmark as bm
from xfmss.llm import LLM, PROMPT_VERSION
from xfmss.explain import Explainer, INSEQ_METHODS, SHAP_METHODS
from xfmss import metrics as mt
from xfmss.it2 import TERM_CENTERS

DEFAULT_METHODS = ["saliency", "input_x_gradient", "integrated_gradients", "gradient_shap", "deeplift",
                   "attention", "occlusion", "lime", "reagent", "kernelshap_exact", "kernelshap_budget",
                   "self_rationale"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", default="data/benchmark.json")
    ap.add_argument("--systems", type=int, default=100)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--methods", default="auto",
                    help="'auto' = methods that passed check_methods.py for this model (else all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_rows", type=int, default=10**9)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--shard", default="0/1",
                    help="k/n: run only every n-th judgment starting at k (one process per GPU, e.g. 0/2 and 1/2)")
    ap.add_argument("--explain_from", default="",
                    help="plant id: earlier plants (by id order) get the rating only, no explanations (months that only train a link)")
    a = ap.parse_args()
    shard_k, shard_n = (int(x) for x in a.shard.split("/"))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if not os.path.exists(a.bench):
        os.makedirs(os.path.dirname(a.bench) or ".", exist_ok=True)
        bm.save(__import__("xfmss.windfarm", fromlist=["x"]).make_windfarm_benchmark(6, seed=11), a.bench)
    bench = bm.load(a.bench)[: a.systems]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}

    import torch
    llm = LLM(a.model, dtype=None if a.dtype == "auto" else getattr(torch, a.dtype))
    ex = Explainer(llm)
    if a.methods == "auto":
        okf = f"data/methods_ok_{a.model.rstrip('/').split('/')[-1]}.json"
        if os.path.exists(okf):
            methods = json.load(open(okf))["ok"]
            print(f"Using {len(methods)} methods that passed the pre-flight check: {', '.join(methods)}")
        else:
            methods = DEFAULT_METHODS
            print("No pre-flight file found (run check_methods.py first); trying all methods.")
    else:
        methods = [m for m in a.methods.split(",") if m]
    fails = {m: 0 for m in methods}
    n_rows = 0
    job = -1
    for s in bench:
        for sub in s["subsystems"]:
            for comp in sub["components"]:
                for v in range(a.variants):
                    job += 1
                    key = f"{s['id']}|{comp['id']}|{v}"
                    if job % shard_n != shard_k or key in done or n_rows >= a.max_rows:
                        continue
                    segs = bm.component_segments(comp, v, seed=bm.system_seed(s["id"]))
                    kinds = [g["kind"] for g in segs]
                    prompt, _ = llm.prompt(comp, segs)
                    p, cov = llm.term_probs([prompt])
                    E_full = float(p[0] @ TERM_CENTERS)
                    p0, _ = llm.term_probs([llm.prompt(comp, segs, keep=[False] * len(segs))[0]])
                    answer = "ABCDEFG"[int(np.argmax(p[0]))]
                    row = {"key": key, "model": a.model, "prompt": PROMPT_VERSION, "system": s["id"],
                           "comp": comp["id"], "variant": v,
                           "h_true": comp["h_true"], "kinds": kinds, "cue_ids": [g["cue_id"] for g in segs],
                           "p": p[0].round(5).tolist(), "p_prior": p0[0].round(5).tolist(), "coverage": float(cov[0]), "E": E_full,
                           "answer": answer, "methods": {}}
                    explain = s["id"] >= a.explain_from
                    # explanations
                    for m in [m for m in methods if fails[m] < 2 and explain]:
                        try:
                            if m in SHAP_METHODS or m == "self_rationale":
                                sc, dt, extra = ex.explain(comp, segs, m)
                            else:
                                pr, sp = llm.prompt(comp, segs)
                                sc, dt = ex.inseq_attribute(pr, sp, m, answer=answer)
                                extra = None
                            fails[m] = 0
                            row["methods"][m] = {"scores": [float(x) for x in sc], "time": dt,
                                                 **({"text": extra} if extra else {})}
                        except Exception as e:  # keep going; record the failure
                            fails[m] += 1
                            row["methods"][m] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                            if fails[m] >= 2:
                                print(f"  ! {m} failed twice in a row and is now skipped: {row['methods'][m]['error']}")
                    # faithfulness (one batched pass for all methods)
                    jobs, idx = [], []
                    for m, r in row["methods"].items():
                        if "scores" not in r or not np.any(np.abs(r["scores"]) > 0):
                            continue
                        for k, rem, keep in mt.faithfulness_masks(r["scores"]):
                            jobs += [llm.prompt(comp, segs, keep=rem)[0], llm.prompt(comp, segs, keep=keep)[0]]
                            idx += [(m, "rem", k), (m, "keep", k)]
                    # random-ranking baseline (5 draws)
                    rng = np.random.default_rng(0)
                    for d in range(5 if explain else 0):
                        rs = rng.random(len(segs))
                        for k, rem, keep in mt.faithfulness_masks(rs):
                            jobs += [llm.prompt(comp, segs, keep=rem)[0], llm.prompt(comp, segs, keep=keep)[0]]
                            idx += [(f"random{d}", "rem", k), (f"random{d}", "keep", k)]
                    Es = llm.expected_health(jobs) if jobs else []
                    acc: dict = {}
                    for (m, t, k), e in zip(idx, Es):
                        acc.setdefault(m, {"rem": {}, "keep": {}})[t][k] = float(e)
                    for m, d in acc.items():
                        f = mt.faithfulness(E_full, d["rem"], d["keep"])
                        if m.startswith("random"):
                            row.setdefault("random", []).append(f)
                        else:
                            row["methods"][m].update(f)
                    with open(a.out, "a") as fh:
                        fh.write(json.dumps(row) + "\n")
                    n_rows += 1
                    print(f"{key}  h*={comp['h_true']:.2f}  E={E_full:.2f}  ans={answer}  cov={cov[0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
