"""Stage 2: cumulative drift line over a trajectory.

Instead of one distance for the whole trajectory, compute the distance
between the user's intent and the trajectory-so-far after EVERY step.
A trajectory pursuing the stated task stays flat/noisy; one drifting to a
different goal climbs as off-task actions accumulate.

drift_line(steps, intent)      -> one distance per step (prefix-based)
step_line(steps, intent)       -> per-step distance, each step alone (diagnostic)
trend_slope(line, window)      -> least-squares slope of the last `window` points
drift_features(line)           -> dict of summary features for classification
"""

import numpy as np

from encoder import distance, encode_intent, encode_texts, step_to_sentence


def drift_line(steps: list, intent: str) -> list[float]:
    """Cumulative drift: distance(intent, steps[0..i]) for each i."""
    intent_vec = encode_intent(intent)
    prefixes = []
    running = []
    for step in steps:
        running.append(step_to_sentence(step))
        prefixes.append(" ".join(running))
    prefix_vecs = encode_texts(prefixes)
    return [distance(intent_vec, v) for v in prefix_vecs]


def step_line(steps: list, intent: str) -> list[float]:
    """Diagnostic: distance of each step in isolation from the intent."""
    intent_vec = encode_intent(intent)
    vecs = encode_texts([step_to_sentence(s) for s in steps])
    return [distance(intent_vec, v) for v in vecs]


def delta_line(steps: list) -> list[float]:
    """Step-to-step novelty: cosine distance between consecutive step vectors.

    Candidate encoding #1 (Phase 3 rethink). A trajectory pivoting to a new
    goal produces a spike here that the cumulative prefix line smears out.
    First step has no predecessor -> 0.0.
    """
    if len(steps) < 2:
        return [0.0] * len(steps)
    vecs = encode_texts([step_to_sentence(s) for s in steps])
    out = [0.0]
    for i in range(1, len(vecs)):
        out.append(distance(vecs[i - 1], vecs[i]))
    return out


def trend_slope(line: list[float], window: int | None = None) -> float:
    """Least-squares slope over the last `window` points (all if None).

    Positive slope = drift climbing. Lines shorter than 2 points get 0.
    """
    y = np.asarray(line[-window:] if window else line, dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def drift_features(line: list[float]) -> dict:
    """Summary features of a drift line for thresholding/classification."""
    y = np.asarray(line, dtype=float)
    return {
        "final": float(y[-1]),
        "mean": float(y.mean()),
        "max": float(y.max()),
        "slope_all": trend_slope(line),
        "slope_last3": trend_slope(line, window=3),
        "rise": float(y[-1] - y[0]),  # net climb start -> end
        "n_steps": len(line),
    }
