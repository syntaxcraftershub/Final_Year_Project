"""Tier 0: embedding features. Effectively free, runs on every step.

Signal is the maximum step-to-step delta seen so far, which measures how
abruptly the trajectory pivots. Phase 3 (see docs/research/CURRENT_STATE.md)
showed cumulative drift alone does not separate the classes (ROC-AUC ~0.5), so
drift is carried for reporting but is not the decision signal.
"""

import time

from drift_tracker import delta_line, drift_line
from tiers.base import TierResult, band_verdict


class Tier0Embedding:
    def __init__(self, low: float = 0.35, high: float = 0.55):
        self.low = low
        self.high = high

    def judge(self, task: str, steps: list) -> TierResult:
        t0 = time.perf_counter()
        deltas = delta_line(steps)
        drifts = drift_line(steps, task)
        # Decision signal: sharpest pivot anywhere in the trajectory.
        score = max(deltas) if deltas else 0.0
        score = float(min(1.0, max(0.0, score)))
        cost_ms = (time.perf_counter() - t0) * 1000.0
        return TierResult(
            verdict=band_verdict(score, self.low, self.high),
            score=score,
            tier="tier0",
            cost_ms=cost_ms,
        )

    def features(self, task: str, steps: list) -> dict:
        """All Tier 0 features, for experiment E1."""
        deltas = delta_line(steps)
        drifts = drift_line(steps, task)
        return {
            "max_delta": max(deltas) if deltas else 0.0,
            "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
            "final_drift": drifts[-1] if drifts else 0.0,
            "mean_drift": sum(drifts) / len(drifts) if drifts else 0.0,
            "drift_rise": (drifts[-1] - drifts[0]) if drifts else 0.0,
            "n_steps": len(steps),
        }
