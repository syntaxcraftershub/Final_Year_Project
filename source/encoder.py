"""Stage 1: embedding fingerprint of intent vs. trajectory.

encode_intent(instruction)  -> vector for what the user asked for
encode_trace(steps)         -> vector for what the agent actually did
distance(v1, v2)            -> cosine distance in [0, 2]

Steps are converted to a plain-English summary before embedding, because
sentence-transformers are trained on natural language, not JSON tool calls.
"""

import json
import os

# Model is cached locally after first download. Force offline so we don't stall
# on HuggingFace HEAD update-checks (slow/blocked networks) — matters for the
# live demo. Set BEFORE importing sentence_transformers. To fetch the model the
# first time, unset these or run with network once.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-mpnet-base-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def step_to_sentence(step):
    """Render one {action, params, result} step as plain English."""
    action = step.get("action", "unknown")
    params = step.get("params", {})
    if action == "respond":
        text = str(params.get("text", ""))[:300]
        return f"The agent responded to the user: {text}"
    parts = []
    for k, v in list(params.items())[:6]:
        v_str = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
        parts.append(f"{k}={v_str[:120]}")
    arg_text = ", ".join(parts)
    sentence = f"The agent called the tool '{action}' with {arg_text or 'no arguments'}."
    result = str(step.get("result", "")).strip()
    if result:
        sentence += f" The tool returned: {result[:200]}"
    return sentence


def trace_to_text(steps):
    return " ".join(step_to_sentence(s) for s in steps)


def encode_intent(instruction: str) -> np.ndarray:
    return _get_model().encode(instruction, normalize_embeddings=True)


def encode_trace(steps: list) -> np.ndarray:
    return _get_model().encode(trace_to_text(steps), normalize_embeddings=True)


def encode_texts(texts: list) -> np.ndarray:
    """Batch encode — used by the drift tracker for per-step prefixes."""
    return _get_model().encode(texts, normalize_embeddings=True, batch_size=32)


def distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine distance: 0 = identical direction, 1 = orthogonal."""
    return float(1.0 - np.dot(v1, v2))
