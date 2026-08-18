"""Tier-1 natural-language-inference gate.

The gate asks whether an observed agent action is contradicted by the user's
task. It is local and requires no API key.
"""
from __future__ import annotations

import os
from functools import lru_cache

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import CrossEncoder

MODEL_NAME = os.environ.get("SHADOWTRACE_NLI_MODEL", "cross-encoder/nli-MiniLM2-L6-H768")


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(MODEL_NAME, max_length=256)


def contradiction_score(task: str, action_sentence: str) -> float:
    """Return the model probability assigned to contradiction.

    For the MiniLM NLI model family, the conventional score order is
    contradiction, entailment, neutral. We prefer model label metadata when
    available and otherwise use index 0.
    """
    model = _model()
    scores = model.predict([(task, action_sentence)], apply_softmax=True)[0]
    labels = getattr(model, "labels", None)
    if labels:
        for i, label in enumerate(labels):
            if "contrad" in str(label).lower():
                return float(scores[i])
    return float(scores[0])
