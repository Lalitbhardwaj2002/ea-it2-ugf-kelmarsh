"""Post-hoc explainers, all mapped to the SAME target (expected health E) and the SAME
features (evidence segments), so they can be compared like-for-like.

Gradient / perturbation family (run through Inseq):
    saliency, input_x_gradient, integrated_gradients, gradient_shap, deeplift,
    attention, occlusion, lime, reagent, value_zeroing
Shapley family (run through the SHAP library, black-box, segment-level):
    kernelshap_exact  - all 2^M coalitions (exact Shapley values)
    kernelshap_budget - sampling KernelSHAP with a small, cost-matched budget
Self-explanation:
    self_rationale    - the LLM names the sentences that drove its rating
"""
from __future__ import annotations
import re
import time
import numpy as np
import torch
import inseq
import shap

from .it2 import TERM_CENTERS


# --- Bug fix for Inseq 0.7.1 ReAGent with chat models -------------------------------
# POSTagTokenSampler sizes its id->POS table with tokenizer.vocab_size, which EXCLUDES
# added special tokens (Qwen <|im_start|>/<|im_end|>, Llama <|eot_id|>, ...). Chat prompts
# contain those ids, so indexing goes out of bounds (a CUDA device-side assert on GPU).
# We extend the table to len(tokenizer); special tokens get POS group 0.
try:
    from inseq.attr.feat.ops.reagent_core import token_sampler as _ts

    _orig_sampler_init = _ts.POSTagTokenSampler.__init__

    def _patched_sampler_init(self, *a, **k):
        _orig_sampler_init(self, *a, **k)
        n_old, n_new = self.id2pos.shape[0], len(self.tokenizer)
        if n_new > n_old:
            pad = torch.zeros(n_new - n_old, dtype=self.id2pos.dtype, device=self.id2pos.device)
            self.id2pos = torch.cat([self.id2pos, pad])

    _ts.POSTagTokenSampler.__init__ = _patched_sampler_init
except Exception:  # pragma: no cover - older/newer inseq layouts
    pass

INSEQ_METHODS = ["saliency", "input_x_gradient", "integrated_gradients", "gradient_shap",
                 "deeplift", "attention", "occlusion", "lime", "reagent"]  # value_zeroing: incompatible, see README
SHAP_METHODS = ["kernelshap_exact", "kernelshap_budget"]
SIGNED = {"input_x_gradient", "integrated_gradients", "gradient_shap", "deeplift", "lime",
          "kernelshap_exact", "kernelshap_budget"}
DEFAULT_KW = {"integrated_gradients": {"n_steps": 16, "internal_batch_size": 2},
              "gradient_shap": {"n_samples": 4},
              "deeplift": {},
              "lime": {"n_samples": 64}}
# ReAGent options are CONSTRUCTOR arguments (not attribute() arguments), so it gets its own Inseq wrapper.
# Note: Inseq's ReAGent ignores custom targets and explains p(predicted answer letter), not E.
REAGENT_INIT = {"keep_top_n": 5, "stopping_condition_top_k": 3, "replacing_ratio": 0.3,
                "max_probe_steps": 300, "num_probes": 8}


class Explainer:
    def __init__(self, llm):
        self.llm = llm
        self._inseq = None
        self._inseq_reagent = None
        centers = torch.tensor(TERM_CENTERS, dtype=torch.float32)
        groups = llm.term_ids

        def expected_health_fn(args):
            logits = args.attribution_model.output2logits(args.forward_output)
            p = logits.float().softmax(-1)
            mass = torch.stack([p[:, g].sum(-1) for g in groups], -1)
            mass = mass / mass.sum(-1, keepdim=True)
            return (mass * centers.to(mass.device)).sum(-1)

        inseq.register_step_function(fn=expected_health_fn, identifier="expected_health", overwrite=True)

    # ------------------------------------------------------------------ Inseq
    def _get_inseq(self, method: str = "saliency"):
        if method == "reagent":
            if self._inseq_reagent is None:
                self._inseq_reagent = inseq.load_model(self.llm.model, "reagent", tokenizer=self.llm.tok,
                                                       **REAGENT_INIT)
            return self._inseq_reagent
        if self._inseq is None:
            self._inseq = inseq.load_model(self.llm.model, "saliency", tokenizer=self.llm.tok)
        return self._inseq

    def _token_char_offsets(self, text: str, target_ids: list[int]):
        """Char offsets for the prompt tokens Inseq attributed (aligned by token ids)."""
        for add in (False, True):
            enc = self.llm.tok(text, return_offsets_mapping=True, add_special_tokens=add)
            ids = enc["input_ids"]
            if ids == target_ids[:len(ids)]:
                return enc["offset_mapping"], len(ids)
        # fallback: progressive decode
        offs, pos = [], 0
        for tid in target_ids:
            piece = self.llm.tok.decode([tid])
            j = text.find(piece, pos) if piece else -1
            if j < 0:
                offs.append((pos, pos))
            else:
                offs.append((j, j + len(piece)))
                pos = j + len(piece)
        return offs, len(target_ids)

    @staticmethod
    def _reduce(t: torch.Tensor, method: str) -> np.ndarray:
        """[n_tokens, n_generated(=1), *extra] -> [n_tokens] scores."""
        t = t.float()
        while t.dim() > 1:
            t = t.mean(-1) if method in ("attention", "value_zeroing") else t.sum(-1)
        return t.cpu().numpy()

    def inseq_attribute(self, prompt: str, spans: list, method: str, answer: str = "D", **kw):
        m = self._get_inseq(method)
        args = dict(DEFAULT_KW.get(method, {}))
        args.update(kw)
        t0 = time.time()
        out = m.attribute(input_texts=prompt, generated_texts=prompt + answer, method=method,
                          attributed_fn="expected_health", show_progress=False, skip_special_tokens=False,
                          **args)
        dt = time.time() - t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sa = out.sequence_attributions[0]
        attr = sa.target_attributions if sa.target_attributions is not None else sa.source_attributions
        n_prompt = attr.shape[0] - 1 if attr.shape[0] > 1 else attr.shape[0]
        tok_scores = self._reduce(attr[:n_prompt], method)
        tok_scores = np.nan_to_num(tok_scores)
        target_ids = [t.id for t in sa.target][:n_prompt]
        offs, n = self._token_char_offsets(prompt, target_ids)
        seg = np.zeros(len(spans))
        for (a, b), s in zip(offs[:n], tok_scores[:n]):
            if b <= a:
                continue
            mid = (a + b) / 2
            for j, sp in enumerate(spans):
                if sp is not None and sp[0] - 1 <= mid < sp[1] + 1:
                    seg[j] += s
                    break
        return seg, dt

    # ------------------------------------------------------------------ SHAP
    def shap_attribute(self, comp, segments, method: str, budget: int = 32, batch_size: int = 16):
        M = len(segments)

        def f(Z):
            texts = [self.llm.prompt(comp, segments, keep=z.astype(bool))[0] for z in Z]
            return self.llm.expected_health(texts, batch_size=batch_size)

        t0 = time.time()
        ex = shap.KernelExplainer(f, np.zeros((1, M)))
        ns = 2 ** M + 2 if method == "kernelshap_exact" else budget
        sv = ex.shap_values(np.ones((1, M)), nsamples=ns, silent=True)
        dt = time.time() - t0
        return np.asarray(sv).reshape(-1)[:M], dt

    # ------------------------------------------------------------------ self-rationale
    def self_rationale(self, comp, segments, k: int = 2):
        from .llm import user_message
        user, _ = user_message(comp, segments)
        user = user.replace("Answer with a single letter only.",
                            f"Do not give the rating. Instead list the IDs of the {k} evidence sentences that most "
                            f"influence your rating, most important first, in the format: S<number>, S<number>")
        t0 = time.time()
        txt = self.llm.generate(self.llm.chat(user), max_new_tokens=24)
        dt = time.time() - t0
        ids = [int(x) - 1 for x in re.findall(r"S\s*(\d+)", txt)]
        seg = np.zeros(len(segments))
        rank = 0
        for i in ids:
            if 0 <= i < len(segments) and seg[i] == 0:
                seg[i] = k - rank
                rank += 1
                if rank >= k:
                    break
        return seg, dt, txt

    # ------------------------------------------------------------------ dispatcher
    def explain(self, comp, segments, method: str, **kw):
        if method in SHAP_METHODS:
            s, dt = self.shap_attribute(comp, segments, method, **kw)
            return s, dt, None
        if method == "self_rationale":
            return self.self_rationale(comp, segments)
        prompt, spans = self.llm.prompt(comp, segments)
        s, dt = self.inseq_attribute(prompt, spans, method, **kw)
        return s, dt, None
