"""LLM elicitation: prompts, term probabilities and expected health score.

The LLM answers with ONE letter A..G (7-term health scale). We never parse free
text: we read the next-token probability mass on the 7 letter tokens, which gives
a probabilistic linguistic judgment and a differentiable target for attribution:
    E(prompt) = sum_i p_i * c_i   (c_i = term centres on [0,1])
"""
from __future__ import annotations
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .it2 import TERMS, TERM_NAMES, TERM_CENTERS

SYSTEM_MSG = "You are an experienced reliability engineer assessing equipment in power plants."
SCALE_TXT = ", ".join(f"{t} = {n}" for t, n in zip(TERMS, TERM_NAMES))
# Scale guide (health-index rubric). Without it the LLM cannot know where "moderate" sits on our scale: Phi-3.5
# ranked assets moderately well but rated them far too low. Real health indices (e.g. transformer HI
# schemes) define their grades the same way, by readings against limits and references. XFMSS_RUBRIC selects it:
# 1 = limits (synthetic reports), fleet = fleet medians (real data), 0 = none. Every result row records the prompt.
RUBRIC_MODE = os.environ.get("XFMSS_RUBRIC", "1")
RUBRIC = ("Scale guide: G = all readings well within their limits and better than the fleet averages and intervals. "
          "F = all readings within limits and close to the fleet averages and intervals. "
          "E = no limit exceeded; at most one reading somewhat worse than its fleet average, interval or design life. "
          "D = no limit exceeded, but several readings worse than their fleet averages, intervals or design life. "
          "C = one reading beyond its alarm, caution or stated limit, or several readings close to their limits. "
          "B = two readings beyond their alarm, caution or stated limits. "
          "A = three or more readings beyond their limits.")
RUBRIC_FLEET = ("Scale guide: G = every indicator clearly better than the fleet median. "
                "F = most indicators better than the fleet median. E = indicators close to the fleet median. "
                "D = a mix of indicators better and worse than the fleet median. "
                "C = several indicators worse than the fleet median. "
                "B = most indicators clearly worse than the fleet median. "
                "A = most indicators far worse than the fleet median, with repeated stops or long downtime.")
GUIDE = {"1": RUBRIC, "fleet": RUBRIC_FLEET}.get(RUBRIC_MODE, "")
PROMPT_VERSION = {"1": "rubric", "fleet": "rubric_fleet"}.get(RUBRIC_MODE, "plain")


def user_message(comp: dict, segments: list[dict], keep=None) -> tuple[str, list]:
    """Build the user message; returns (text, local char spans per segment or None if removed)."""
    head = (f"Component {comp['id']} is a {comp['type']} (rated capacity {comp['cap']} {comp.get('unit', 'units')}).\n"
            f"Evidence about its current condition:\n")
    body, spans, pos = "", [], len(head)
    for j, seg in enumerate(segments):
        if keep is not None and not keep[j]:
            spans.append(None)
            continue
        line = f"[S{j + 1}] {seg['text']}\n"
        start = pos + len(body) + len(f"[S{j + 1}] ")
        spans.append((start, start + len(seg["text"])))
        body += line
    tail = (f"\nRate the current health of component {comp['id']} on this scale: {SCALE_TXT}.\n"
            + (GUIDE + "\n" if GUIDE else "")
            + "Answer with a single letter only.")
    return head + body + tail, spans


class LLM:
    def __init__(self, name_or_path: str, device: str | None = None, dtype=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else (
                torch.float16 if self.device == "cuda" else torch.float32)
        self.name = name_or_path
        self.tok = AutoTokenizer.from_pretrained(name_or_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=dtype, attn_implementation="eager").to(self.device).eval()
        self.term_ids = self._letter_ids(TERMS)

    # ---------- prompts
    def chat(self, user: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # models without a system role (e.g. Gemma)
            msgs = [{"role": "user", "content": SYSTEM_MSG + "\n\n" + user}]
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def prompt(self, comp, segments, keep=None):
        """Full chat prompt + absolute char spans of each kept segment."""
        user, spans = user_message(comp, segments, keep)
        text = self.chat(user)
        off = text.find(user)
        assert off >= 0, "user message not found verbatim in chat template output"
        return text, [None if s is None else (s[0] + off, s[1] + off) for s in spans]

    # ---------- scoring
    def _letter_ids(self, letters):
        out = []
        for L in letters:
            ids = set()
            for v in (L, " " + L):
                t = self.tok.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    ids.add(t[0])
            if not ids:
                raise ValueError(f"letter {L!r} is not a single token for this tokenizer")
            out.append(sorted(ids))
        return out

    def letter_mass(self, logits: torch.Tensor, groups) -> torch.Tensor:
        """[batch, vocab] logits -> [batch, len(groups)] probability mass per letter group."""
        p = logits.float().softmax(-1)
        return torch.stack([p[:, g].sum(-1) for g in groups], -1)

    @torch.no_grad()
    def next_logits(self, texts: list[str], batch_size: int = 8) -> torch.Tensor:
        outs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            try:  # only the last position is needed; full logits over a 150k vocabulary can exhaust a T4
                logits = self.model(**enc, logits_to_keep=1, use_cache=False).logits
            except TypeError:
                logits = self.model(**enc, use_cache=False).logits
            outs.append(logits[:, -1, :].float().cpu())
        return torch.cat(outs)

    def term_probs(self, texts: list[str], batch_size: int = 8):
        """Returns (p [n,7] renormalised over the letters, coverage [n] = raw letter mass)."""
        mass = self.letter_mass(self.next_logits(texts, batch_size), self.term_ids)
        cov = mass.sum(-1)
        return (mass / cov.unsqueeze(-1)).numpy(), cov.numpy()

    def expected_health(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        p, _ = self.term_probs(texts, batch_size)
        return p @ TERM_CENTERS

    @torch.no_grad()
    def generate(self, text: str, max_new_tokens: int = 40) -> str:
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
