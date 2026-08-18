import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from pipeline import Thresholds, _tier0


def test_tier0_low_is_benign():
    verdict, _, score = _tier0(0.10, 0.20, Thresholds())
    assert verdict == "benign"
    assert score == 0.20


def test_tier0_high_is_flagged():
    verdict, _, score = _tier0(0.70, 0.20, Thresholds())
    assert verdict == "flagged"
    assert score == 0.70


def test_tier0_gray_is_ambiguous():
    verdict, _, score = _tier0(0.50, 0.40, Thresholds())
    assert verdict == "gray"
    assert score == 0.50
