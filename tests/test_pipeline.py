"""Tests for the data parser and (as stages land) the pipeline."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

from data_parser import _as_tool_call, parse_record  # noqa: E402
from drift_tracker import drift_features, trend_slope  # noqa: E402

PROCESSED = Path(__file__).resolve().parent.parent / "database" / "processed" / "tracesafe.jsonl"


def _sample_raw_record(category="BENIGN"):
    return {
        "mutation_category": category,
        "new_trace": {
            "trace": [
                {"role": "user", "content": "Convert 10 liters to gallons."},
                {"role": "agent", "content": {"name": "liter_to_gallon", "arguments": {"liter": 10}}},
                {"role": "tool", "content": '{"gallon": 2.64}'},
                {"role": "agent", "content": "10 liters is about 2.64 gallons."},
            ]
        },
    }


def test_as_tool_call_handles_dict_and_string():
    assert _as_tool_call({"name": "ls", "arguments": {}})["name"] == "ls"
    assert _as_tool_call("{'name': 'ls', 'arguments': {}}")["name"] == "ls"
    assert _as_tool_call('{"name": "ls", "arguments": {}}')["name"] == "ls"
    assert _as_tool_call("plain text answer") is None


def test_parse_record_structure():
    parsed = parse_record(_sample_raw_record(), task_id="t1")
    assert parsed["label"] == "benign"
    assert parsed["user_instruction"] == "Convert 10 liters to gallons."
    assert len(parsed["steps"]) == 2
    assert parsed["steps"][0]["action"] == "liter_to_gallon"
    assert parsed["steps"][0]["result"] == '{"gallon": 2.64}'
    assert parsed["steps"][1]["action"] == "respond"


def test_parse_record_malicious_label():
    parsed = parse_record(_sample_raw_record("PROMPT_INJECTION"), task_id="t2")
    assert parsed["label"] == "malicious"


def test_trend_slope_directions():
    assert trend_slope([0.1, 0.2, 0.3, 0.4]) > 0.09
    assert trend_slope([0.4, 0.3, 0.2, 0.1]) < -0.09
    assert abs(trend_slope([0.3, 0.3, 0.3, 0.3])) < 1e-9
    assert trend_slope([0.5]) == 0.0
    # window: flat then climbing tail
    assert trend_slope([0.3, 0.3, 0.3, 0.3, 0.5, 0.7], window=3) > 0.15


def test_drift_features_keys_and_rise():
    f = drift_features([0.2, 0.3, 0.5])
    assert f["n_steps"] == 3
    assert abs(f["rise"] - 0.3) < 1e-9
    assert f["final"] == 0.5
    assert f["max"] == 0.5
    assert set(f) == {"final", "mean", "max", "slope_all", "slope_last3", "rise", "n_steps"}


def test_processed_file_valid():
    assert PROCESSED.exists(), "run src/data_parser.py first"
    labels = set()
    n = 0
    for line in open(PROCESSED, encoding="utf-8"):
        rec = json.loads(line)
        assert rec["user_instruction"]
        assert rec["steps"]
        assert rec["label"] in ("benign", "malicious")
        labels.add(rec["label"])
        n += 1
    # The corpus legitimately grew once benign twins were added (Phase 2):
    # every malicious record now also emits its unmutated benign twin.
    assert n > 2000
    assert labels == {"benign", "malicious"}
