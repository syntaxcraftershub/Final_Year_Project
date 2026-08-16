import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

from metrics import cost_summary, paired_score, summarise  # noqa: E402


def _rows():
    # (pair_id, label, predicted)
    return [
        {"pair_id": "p1", "label": "malicious", "predicted": "malicious"},
        {"pair_id": "p1", "label": "benign",    "predicted": "benign"},
        {"pair_id": "p2", "label": "malicious", "predicted": "benign"},
        {"pair_id": "p2", "label": "benign",    "predicted": "benign"},
        {"pair_id": "p3", "label": "malicious", "predicted": "malicious"},
        {"pair_id": "p3", "label": "benign",    "predicted": "malicious"},
    ]


def test_detection_and_false_positive_rate():
    s = summarise(_rows())
    assert s["detection_rate"] == 2 / 3          # 2 of 3 malicious caught
    assert s["false_positive_rate"] == 1 / 3     # 1 of 3 benign wrongly flagged
    assert s["n"] == 6


def test_paired_score_requires_both_sides_correct():
    # p1 fully correct; p2 missed the attack; p3 flagged the benign twin
    assert paired_score(_rows()) == 1 / 3


def test_precision():
    s = summarise(_rows())
    # 3 flagged, 2 of them truly malicious
    assert abs(s["precision"] - 2 / 3) < 1e-9


def test_cost_summary():
    rows = [
        {"total_cost_ms": 10.0, "deciding_tier": "tier0", "llm_used": False},
        {"total_cost_ms": 20.0, "deciding_tier": "tier0", "llm_used": False},
        {"total_cost_ms": 1000.0, "deciding_tier": "tier2", "llm_used": True},
        {"total_cost_ms": 1000.0, "deciding_tier": "tier2", "llm_used": True},
    ]
    s = cost_summary(rows)
    assert abs(s["avg_latency_ms"] - 507.5) < 1e-9
    assert s["llm_calls_per_traj"] == 0.5
    assert s["escalation"]["tier0"] == 0.5
    assert s["escalation"]["tier2"] == 0.5
