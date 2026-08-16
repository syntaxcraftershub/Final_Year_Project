"""Three-tier cascade routing.

Each tier decides or abstains. Abstention escalates. Tier 2 is terminal.
The result records which tier decided and the accumulated cost, which is what
produces the accuracy-versus-cost trade-off curve.

Trajectory aggregation rule (fixed before evaluation, never tuned): a
trajectory is malicious if any tier decides malicious. Tiers 0 and 1 already
reduce their per-step signals to a trajectory-level score internally by taking
the worst step, which implements first-flag-wins.
"""

from dataclasses import dataclass, field


@dataclass
class CascadeResult:
    verdict: str                 # "benign" | "malicious"
    deciding_tier: str
    total_cost_ms: float
    llm_used: bool
    tier_results: list = field(default_factory=list)

    @property
    def scores(self) -> dict:
        return {r.tier: r.score for r in self.tier_results}


class Cascade:
    def __init__(self, tier0, tier1, tier2):
        self.tiers = [tier0, tier1, tier2]

    def run(self, task: str, steps: list) -> CascadeResult:
        total_cost = 0.0
        results = []
        for tier in self.tiers:
            r = tier.judge(task, steps)
            results.append(r)
            total_cost += r.cost_ms
            if r.verdict != "abstain":
                return CascadeResult(
                    verdict=r.verdict,
                    deciding_tier=r.tier,
                    total_cost_ms=total_cost,
                    llm_used=(r.tier == "tier2"),
                    tier_results=results,
                )
        # Tier 2 must decide; reaching here means it violated its contract.
        raise RuntimeError("terminal tier abstained, which violates the tier contract")
