"""Tier-1 natural-language-inference gate.

The gate asks whether an observed agent action is entailed by the user's task.
It is deliberately local and deterministic at inference time: no API key and
no cloud dependency are required.
"""
from __future__ import annotations

import os
from functools import lru_cache

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from cross_encoder import CrossEncoder

MODEL_NAME = os.environ.get("SHADOWTRACE_NLI_MODEL", "cross-encoder/nli-MiniLM2-L6-H768")


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(MODEL_NAME, max_length=256)


def contradiction_score(task: str, action_sentence: str) -> float:
    """Return P(contradiction) for (task, action).

    The model's NLI labels are normally [contradiction, entailment, neutral].
    We inspect labels exposed by the model when available and otherwise use
    the conventional index 0 used by this model family.
    """
    model = _model()
    scores = model.predict([(task, action_sentence)], apply_softmax=True)[0]
    labels = getattr(model, "labels", None)
    if labels:
        labels = [str(x).lower() for x in labels]
        for i, label in enumerate(labels):
            if "contrad" in label:
                return float(scores[i])
    return float(scores[0])
