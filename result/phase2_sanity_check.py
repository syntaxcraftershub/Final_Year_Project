"""Phase 2 gate test: distances on 10 benign vs 10 malicious trajectories.

Expectation at this stage: malicious ones TEND to score higher, not a clean
separation. Prints every distance plus group means.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

from encoder import distance, encode_intent, encode_trace  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "database" / "processed" / "tracesafe.jsonl"

# Categories where the trajectory semantically diverges from the task —
# the case this monitor targets. Interface/hallucination bugs are kept in
# the dataset but sampled separately below for an honest first look.
DRIFT_CATEGORIES = {"PROMPT_INJECTION", "PRIVACY_LEAKAGE"}


def main():
    random.seed(42)
    recs = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    benign = [r for r in recs if r["label"] == "benign"]
    drift_mal = [r for r in recs if r["category"] in DRIFT_CATEGORIES]
    other_mal = [r for r in recs if r["label"] == "malicious" and r["category"] not in DRIFT_CATEGORIES]

    samples = (
        [("benign", r) for r in random.sample(benign, 10)]
        + [("malicious/drift", r) for r in random.sample(drift_mal, 10)]
        + [("malicious/other", r) for r in random.sample(other_mal, 10)]
    )

    groups = {}
    for tag, r in samples:
        d = distance(encode_intent(r["user_instruction"]), encode_trace(r["steps"]))
        groups.setdefault(tag, []).append(d)
        print(f"{tag:16s} {r['task_id']:28s} steps={len(r['steps']):3d} distance={d:.4f}")

    print("\n--- group means ---")
    for tag, ds in groups.items():
        print(f"{tag:16s} mean={sum(ds)/len(ds):.4f}  min={min(ds):.4f}  max={max(ds):.4f}")


if __name__ == "__main__":
    main()
