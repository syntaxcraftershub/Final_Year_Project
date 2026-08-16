import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

from cascade import Cascade, CascadeResult  # noqa: E402
from tiers.base import TierResult  # noqa: E402


class StubTier:
    """Returns a scripted verdict, records that it was called."""

    def __init__(self, name, verdict, score=0.5, cost_ms=1.0):
        self.name, self.verdict, self.score, self.cost_ms = name, verdict, score, cost_ms
        self.calls = 0

    def judge(self, task, steps):
        self.calls += 1
        return TierResult(self.verdict, self.score, self.name, self.cost_ms)


def test_confident_tier0_stops_the_cascade():
    t0 = StubTier("tier0", "benign")
    t1 = StubTier("tier1", "malicious")
    t2 = StubTier("tier2", "malicious")
    res = Cascade(t0, t1, t2).run("task", [{"action": "cat"}])

    assert res.verdict == "benign"
    assert res.deciding_tier == "tier0"
    assert t1.calls == 0 and t2.calls == 0, "expensive tiers must not run"


def test_abstention_escalates_to_the_next_tier():
    t0 = StubTier("tier0", "abstain")
    t1 = StubTier("tier1", "malicious")
    t2 = StubTier("tier2", "benign")
    res = Cascade(t0, t1, t2).run("task", [{"action": "cat"}])

    assert res.verdict == "malicious"
    assert res.deciding_tier == "tier1"
    assert t2.calls == 0


def test_full_escalation_reaches_tier2():
    t0 = StubTier("tier0", "abstain")
    t1 = StubTier("tier1", "abstain")
    t2 = StubTier("tier2", "malicious")
    res = Cascade(t0, t1, t2).run("task", [{"action": "cat"}])

    assert res.deciding_tier == "tier2"
    assert res.llm_used is True


def test_cost_accumulates_across_tiers():
    t0 = StubTier("tier0", "abstain", cost_ms=1.0)
    t1 = StubTier("tier1", "abstain", cost_ms=10.0)
    t2 = StubTier("tier2", "benign", cost_ms=1000.0)
    res = Cascade(t0, t1, t2).run("task", [{"action": "cat"}])

    assert res.total_cost_ms == 1011.0
    assert res.llm_used is True


def test_llm_not_used_when_cheap_tier_decides():
    res = Cascade(StubTier("tier0", "malicious"),
                  StubTier("tier1", "benign"),
                  StubTier("tier2", "benign")).run("task", [{"action": "cat"}])
    assert res.llm_used is False


def test_terminal_tier_abstaining_raises():
    t0 = StubTier("tier0", "abstain")
    t1 = StubTier("tier1", "abstain")
    t2 = StubTier("tier2", "abstain")
    try:
        Cascade(t0, t1, t2).run("task", [{"action": "cat"}])
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
