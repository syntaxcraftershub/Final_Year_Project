"""Deterministic, pair-aware splitting of the trajectory corpus.

A trajectory and its benign twin share a pair_id and MUST land in the same
split. Splitting them would leak near-identical text across the boundary and
inflate every reported number.
"""

import json
from collections import Counter
from pathlib import Path
import random

DATA = Path(__file__).resolve().parent.parent / "database" / "processed" / "tracesafe.jsonl"
SPLIT_PATH = Path(__file__).resolve().parent.parent / "database" / "processed" / "splits.json"

SEED = 42
CALIBRATION_FRAC = 0.4


def assign_splits(records, calibration_frac=CALIBRATION_FRAC, seed=SEED):
    """Tag each record with split="calibration" or "test", grouped by pair_id."""
    pair_ids = sorted({r["pair_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(pair_ids)
    n_cal = int(len(pair_ids) * calibration_frac)
    cal_pairs = set(pair_ids[:n_cal])
    return [
        {**r, "split": "calibration" if r["pair_id"] in cal_pairs else "test"}
        for r in records
    ]


def load_records(split=None):
    """Load parsed records, optionally filtered to one split."""
    records = [json.loads(line) for line in open(DATA, encoding="utf-8")]
    tagged = assign_splits(records)
    if split:
        tagged = [r for r in tagged if r["split"] == split]
    return tagged


def main():
    records = [json.loads(line) for line in open(DATA, encoding="utf-8")]
    tagged = assign_splits(records)

    by_split_label = Counter(f"{r['split']}/{r['label']}" for r in tagged)
    by_split_category = Counter(f"{r['split']}/{r['category']}" for r in tagged)
    n_pairs = len({r["pair_id"] for r in tagged})
    n_cal_pairs = len({r["pair_id"] for r in tagged if r["split"] == "calibration"})
    n_test_pairs = len({r["pair_id"] for r in tagged if r["split"] == "test"})

    summary = {
        "seed": SEED,
        "calibration_fraction_target": CALIBRATION_FRAC,
        "n_records": len(tagged),
        "n_pairs": n_pairs,
        "n_calibration_pairs": n_cal_pairs,
        "n_test_pairs": n_test_pairs,
        "counts_by_split_and_label": dict(by_split_label),
        "counts_by_split_and_category": dict(by_split_category),
    }
    SPLIT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
