import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

from atbench_loader import parse_record  # noqa: E402

RAW_PATH = Path(__file__).resolve().parent.parent / "database" / "raw" / "atbench" / "ATBench" / "test.json"


def _sample_record(label=1):
    return {
        "id": 42,
        "contents": [[
            {"role": "user", "content": "Summarise notes.txt for me."},
            {"role": "agent", "thought": "", "action": '{"name": "cat", "arguments": {"file": "notes.txt"}}'},
            {"role": "environment", "content": "hello world"},
        ]],
        "label": label,
        "risk_source": "test_source",
        "failure_mode": "test_mode",
        "reason": "because",
        "real_world_harm": "none",
    }


def test_parse_record_maps_label_correctly():
    mal = parse_record(_sample_record(label=1))
    ben = parse_record(_sample_record(label=0))
    assert mal["label"] == "malicious"
    assert ben["label"] == "benign"
    assert mal["category"] == "test_source"
    assert ben["category"] == "SAFE"


def test_parse_record_extracts_steps_and_task():
    rec = parse_record(_sample_record())
    assert rec["user_instruction"] == "Summarise notes.txt for me."
    assert rec["steps"][0]["action"] == "cat"
    assert rec["steps"][0]["params"] == {"file": "notes.txt"}
    assert rec["steps"][0]["result"] == "hello world"
    assert rec["source"] == "atbench"
    # ATBench has no matched-twin structure: pair_id is just its own task_id.
    assert rec["pair_id"] == rec["task_id"]


def test_full_dataset_matches_published_counts():
    if not RAW_PATH.exists():
        return  # external data not downloaded in this environment; skip
    import json
    from atbench_loader import load_all
    records, skipped = load_all()
    labels = {"benign": 0, "malicious": 0}
    for r in records:
        labels[r["label"]] += 1
    # Published: 1,000 cases, 503 safe / 497 unsafe (README.md, checked live)
    assert len(records) == 1000
    assert labels == {"benign": 503, "malicious": 497}
