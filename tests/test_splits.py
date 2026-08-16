import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

from splits import assign_splits  # noqa: E402


def _records():
    recs = []
    for i in range(100):
        pid = f"p{i}"
        recs.append({"task_id": pid, "pair_id": pid, "label": "malicious"})
        recs.append({"task_id": pid + "__twin", "pair_id": pid, "label": "benign"})
    return recs


def test_pairs_never_straddle_the_split():
    tagged = assign_splits(_records(), calibration_frac=0.4, seed=42)
    by_pair = {}
    for r in tagged:
        by_pair.setdefault(r["pair_id"], set()).add(r["split"])
    for pair_id, splits in by_pair.items():
        assert len(splits) == 1, f"pair {pair_id} leaked across splits: {splits}"


def test_split_proportions_are_approximately_right():
    tagged = assign_splits(_records(), calibration_frac=0.4, seed=42)
    cal = sum(r["split"] == "calibration" for r in tagged)
    assert 0.3 < cal / len(tagged) < 0.5


def test_split_is_deterministic():
    a = assign_splits(_records(), calibration_frac=0.4, seed=42)
    b = assign_splits(_records(), calibration_frac=0.4, seed=42)
    assert [r["split"] for r in a] == [r["split"] for r in b]
