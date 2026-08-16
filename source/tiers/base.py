"""Shared tier contract.

CONTRACT: every tier returns `score` oriented so that HIGHER MEANS MORE LIKELY
MALICIOUS. Models with the opposite native direction (NLI entailment, for
example) must invert internally before returning. tests/test_tiers.py enforces
this for each tier.
"""

from dataclasses import dataclass


@dataclass
class TierResult:
    verdict: str      # "benign" | "malicious" | "abstain"
    score: float      # higher = more likely malicious
    tier: str
    cost_ms: float


def band_verdict(score: float, low: float, high: float) -> str:
    """Route a score through an abstention band.

    <= low  -> decide benign
    >= high -> decide malicious
    between -> abstain, escalate to the next tier
    """
    if score <= low:
        return "benign"
    if score >= high:
        return "malicious"
    return "abstain"
