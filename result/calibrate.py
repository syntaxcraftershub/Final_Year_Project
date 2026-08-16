"""Choose Tier 0 and Tier 1 abstention bands on the CALIBRATION split.

Bands are picked so that decisions made by a tier are high-precision: the low
threshold is the score below which almost no malicious trajectory falls, and
the high threshold is the score above which almost no benign one does.
Everything between escalates.

The chosen values are frozen to thresholds.json before any test-split run, so
they cannot be retro-fitted to the final numbers.

PERFORMANCE NOTE: reuses the same batched scoring approach as E1/E2 (see
those scripts) rather than the per-trajectory Tier0Embedding/Tier1NLI.judge()
API, purely for wall-clock reasons on this 2-core CPU host. The feature/score
math is identical.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

from splits import load_records  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_tier0_signals import batched_features  # noqa: E402
from e2_nli_signal import batched_nli_scores  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "database" / "processed" / "thresholds.json"
TOLERANCE = 0.02  # allow 2% error on each confident side


def bands(scores, labels, tolerance=TOLERANCE):
    """low = tolerance-quantile of malicious; high = (1-tolerance)-quantile of benign."""
    mal = np.array([s for s, l in zip(scores, labels) if l == 1])
    ben = np.array([s for s, l in zip(scores, labels) if l == 0])
    low = float(np.quantile(mal, tolerance)) if len(mal) else 0.0
    high = float(np.quantile(ben, 1 - tolerance)) if len(ben) else 1.0
    if high < low:                      # distributions overlap badly
        low = high = float(np.median(np.concatenate([mal, ben])))
    return round(low, 4), round(high, 4)


def main():
    records = load_records(split="calibration")
    labels = [1 if r["label"] == "malicious" else 0 for r in records]

    print("computing tier0 (max_delta) scores for calibration...")
    t0_rows = batched_features(records)
    s0 = [min(1.0, max(0.0, r["max_delta"])) for r in t0_rows]
    low0, high0 = bands(s0, labels)

    print("computing tier1 (NLI) scores for calibration...")
    s1 = batched_nli_scores(records)
    low1, high1 = bands(s1, labels)

    cfg = {
        "tier0": {"low": low0, "high": high0},
        "tier1": {"low": low1, "high": high1},
        "tolerance": TOLERANCE,
        "calibrated_on": "calibration split only",
        "n_calibration": len(records),
        "seed": 42,
        "tier0_signal": "max_delta (step-to-step embedding distance)",
        "tier1_model": "cross-encoder/nli-deberta-v3-base",
        "tier2_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "encoder_model": "all-mpnet-base-v2",
    }
    OUT.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps(cfg, indent=2))
    print(f"\nFROZEN to {OUT}. Do not change these after running on the test split.")


if __name__ == "__main__":
    main()
