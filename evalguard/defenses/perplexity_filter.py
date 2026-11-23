"""Perplexity filtering (cf. Jain et al., 2023; Alon & Kamfonas, 2023).

Adversarial suffixes are usually unnatural text, so a small language model finds them surprising
(high perplexity). The filter scores the external content of each trajectory and drops it before
the judge sees it when the score exceeds a threshold tau.

A whole-text average dilutes a short suffix inside long benign content, so by default the score is
the *maximum* perplexity over sliding windows.

Scorers:
    HFPerplexityScorer  GPT-2 family model (e.g. distilgpt2) via transformers; token-level
    CharBigramScorer    offline character-bigram model; a rough proxy used in mock mode
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence

from evalguard.core.models import AgentTrajectory
from evalguard.defenses.base import Defense

logger = logging.getLogger(__name__)


def reduce_nlls(nlls: Sequence[float], window: int, mode: str) -> float:
    """Turn per-unit negative log-likelihoods into one perplexity: whole-text mean or windowed max."""
    if not nlls:
        return 1.0
    if mode == "mean" or len(nlls) <= window:
        return math.exp(sum(nlls) / len(nlls))
    if mode != "windowed_max":
        raise ValueError(f"Unknown perplexity mode {mode!r}; use 'mean' or 'windowed_max'")
    stride = max(1, window // 2)
    best = 0.0
    for start in range(0, len(nlls) - window + 1, stride):
        chunk = nlls[start:start + window]
        best = max(best, math.exp(sum(chunk) / len(chunk)))
    tail = nlls[-window:]  # make sure the end of the text, where suffixes sit, is always covered
    return max(best, math.exp(sum(tail) / len(tail)))


class PerplexityScorer(ABC):
    name: str = "base"

    @abstractmethod
    def score(self, text: str) -> float:
        """Perplexity of the text (higher = more surprising)."""


_CORPUS = """
Hi team, thanks for sending the report. I reviewed the numbers and everything looks good to me.
Could you please send the final version by Friday? Let me know if you have any questions.
The meeting is on Thursday at 2pm in room 4B. We will review the new design and plan the next steps.
Your order has shipped and should arrive in three to five business days. Track it on our website.
The price of the headphones is $348.00 with free shipping on orders over $35.
Account SCL-002318, amount due $84.20 by December 1. Payment received on Nov 12, thank you.
Contact us at support@example.com or call (206) 555-0142. Visit https://www.example.com/help for details.
{"status": "ok", "resource": "orders/A-77120", "data": "Order total $42.99, delivered Nov 14."}
<html><head><title>Product page</title></head><body><p>Battery life up to 30 hours.</p></body></html>
The build passed: 214 tests passed, 0 failed, coverage 87 percent. Please review pull request 212.
Unused time off carries over up to 5 days into the next year. The balance is 11.5 days.
Storage prices dropped again this week, and a new region opened in Madrid. Here are three tips.
From: priya.raman@acme-corp.com Subject: Q3 report draft. The finance numbers are final now.
Volume /dev/disk3 is 71 percent full. The largest folders are ./project, ./cache and ./logs.
We appreciate your patience while our servers are updated tonight between 11pm and 1am.
"""


class CharBigramScorer(PerplexityScorer):
    """Character-bigram language model with add-k smoothing, trained on a small built-in English corpus.

    It runs offline in milliseconds. It is only a proxy for a real language model: it catches text with
    unusual character sequences, but has no notion of words or meaning.
    """

    name = "char_bigram"
    OOV = "\x00"

    def __init__(self, corpus: str | None = None, window: int = 48, mode: str = "windowed_max", k: float = 0.5) -> None:
        self.window = window
        self.mode = mode
        self.k = k
        text = self._normalize(corpus or _CORPUS)
        self.vocab = {chr(c) for c in range(32, 127)} | {"\n", self.OOV}
        self.pairs = Counter(zip(text, text[1:]))
        self.contexts = Counter(text[:-1])

    @classmethod
    def _normalize(cls, text: str) -> str:
        return "".join(c if (32 <= ord(c) < 127 or c == "\n") else cls.OOV for c in text)

    def nlls(self, text: str) -> list[float]:
        t = self._normalize(text)
        v = len(self.vocab)
        return [
            -math.log((self.pairs[(a, b)] + self.k) / (self.contexts[a] + self.k * v))
            for a, b in zip(t, t[1:])
        ]

    def score(self, text: str) -> float:
        return reduce_nlls(self.nlls(text), self.window, self.mode)


class HFPerplexityScorer(PerplexityScorer):
    """Token-level perplexity from a small causal LM (default: distilgpt2)."""

    def __init__(self, model_name: str = "distilgpt2", window_tokens: int = 32, mode: str = "windowed_max") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise RuntimeError("HFPerplexityScorer needs torch and transformers: pip install torch transformers") from e
        self._torch = torch
        self.name = f"hf:{model_name}"
        self.window = window_tokens
        self.mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading perplexity model %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = int(getattr(self.model.config, "n_positions", 1024))

    def nlls(self, text: str) -> list[float]:
        torch = self._torch
        ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
        if ids.numel() < 2:
            return []
        out: list[float] = []
        # Score in chunks that fit the model's context window.
        for start in range(0, ids.numel() - 1, self.max_length - 1):
            chunk = ids[start:start + self.max_length].unsqueeze(0).to(self.device)
            if chunk.shape[1] < 2:
                break
            with torch.no_grad():
                logits = self.model(chunk).logits
            log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            nll = -log_probs.gather(-1, chunk[:, 1:].unsqueeze(-1)).squeeze(-1).squeeze(0)
            out.extend(nll.tolist())
        return out

    def score(self, text: str) -> float:
        return reduce_nlls(self.nlls(text), self.window, self.mode)


class PerplexityFilter(Defense):
    """Drops trajectories whose external content has perplexity above tau."""

    name = "perplexity"

    def __init__(
        self,
        scorer: PerplexityScorer,
        tau: float | None = None,
        calibrate_percentile: float = 95.0,
        margin: float = 1.05,
    ) -> None:
        self.scorer = scorer
        self.tau = tau
        self.calibrate_percentile = calibrate_percentile
        self.margin = margin
        self._cache: dict[str, float] = {}

    def score_text(self, text: str) -> float:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in self._cache:
            self._cache[key] = self.scorer.score(text)
        return self._cache[key]

    def fit(self, clean_texts: Sequence[str]) -> None:
        """If tau was not given, set it to a percentile of clean-content scores times a safety margin."""
        if self.tau is not None:
            return
        scores = sorted(self.score_text(t) for t in clean_texts if t.strip())
        if not scores:
            raise ValueError("Perplexity calibration needs at least one non-empty clean text")
        rank = (self.calibrate_percentile / 100.0) * (len(scores) - 1)
        lo, hi = math.floor(rank), math.ceil(rank)
        value = scores[lo] + (scores[hi] - scores[lo]) * (rank - lo)
        self.tau = value * self.margin
        logger.info("Perplexity tau calibrated to %.2f from %d clean texts", self.tau, len(scores))

    def screen(self, trajectory: AgentTrajectory) -> str | None:
        if self.tau is None:
            raise RuntimeError("PerplexityFilter has no tau: pass one or call fit() on clean texts first")
        value = self.score_text(trajectory.untrusted_input)
        if value > self.tau:
            return f"perplexity {value:.1f} > tau {self.tau:.1f}"
        return None
