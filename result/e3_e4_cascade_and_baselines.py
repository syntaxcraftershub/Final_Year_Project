"""E3: cascade trade-off curve. E4: reimplemented baselines.

Every configuration below runs through the SAME Cascade code path (source/
cascade.py), so the comparison is exactly apples-to-apples:

  tier0_only        tier0 decides everything (band collapsed to a point)
  tier0_1           tier0 + tier1, tier2 only as terminal fallback
  full_cascade      frozen calibrated bands
  per_step_llm      BASELINE: tier2 forced on every trajectory
  majority_floor    BASELINE: always predict the majority class

This is the single run that touches the TEST split. Run it once.

PERFORMANCE NOTE: Tier0/Tier1 scores are precomputed once for the whole test
split via the same batched approach as E1/E2 (source models re-invoked
hundreds of times individually is intractable on this 2-core host), then
wrapped in PrecomputedTier adapters that satisfy the same `judge(task,
steps) -> TierResult` contract as the real tiers, using band_verdict with the
frozen thresholds -- identical decision logic, just without re-running
inference that was already run once. Tier2 (real Qwen2.5-1.5B-Instruct) is
NOT precomputed: it is invoked lazily, on demand, exactly when a
configuration's cascade actually reaches it -- this is deliberate, because
per_step_llm forcing Tier2 on every trajectory (and its resulting latency) IS
the measurement this experiment exists to produce.
"""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade import Cascade  # noqa: E402
from metrics import cost_summary, paired_score, summarise  # noqa: E402
from splits import load_records  # noqa: E402
from tiers.base import TierResult, band_verdict  # noqa: E402
from tiers.tier2_llm import Tier2LLM  # noqa: E402

from e1_tier0_signals import batched_features  # noqa: E402
from e2_nli_signal import batched_nli_scores  # noqa: E402

THRESHOLDS = Path(__file__).resolve().parent.parent / "database" / "processed" / "thresholds.json"
OUT = Path(__file__).resolve().parent / "e3_e4_results.csv"


class PrecomputedTier:
    """Same judge() contract as a real tier, but reads a precomputed score."""

    def __init__(self, name, scores_by_index, low, high, cost_ms_by_index=None):
        self.name = name
        self.scores = scores_by_index
        self.low = low
        self.high = high
        self.cost_ms = cost_ms_by_index or {}
        self._idx = None  # set externally per-call via bind()

    def bind(self, idx):
        self._idx = idx
        return self

    def judge(self, task, steps):
        score = self.scores[self._idx]
        return TierResult(
            verdict=band_verdict(score, self.low, self.high),
            score=score,
            tier=self.name,
            cost_ms=self.cost_ms.get(self._idx, 0.1),  # embedding/NLI cost is ~free vs tier2
        )


class AlwaysAbstain:
    def __init__(self, name):
        self.name = name

    def judge(self, task, steps):
        return TierResult("abstain", 0.5, self.name, 0.0)


class AlwaysBenign:
    """Majority-class floor: reports the corpus's majority label."""

    def judge(self, task, steps):
        return TierResult("benign", 0.0, "floor", 0.0)


def build_configs(th, t0_scores, t1_scores):
    t0 = th["tier0"]
    t1 = th["tier1"]
    tier2_fake = Tier2LLM(fake=True)
    tier2_real = Tier2LLM(fake=False)

    def pc_tier0(low, high):
        return PrecomputedTier("tier0", t0_scores, low, high)

    def pc_tier1(low, high):
        return PrecomputedTier("tier1", t1_scores, low, high)

    return {
        # Tier 0 alone: collapse its band to a single point so it always decides.
        "tier0_only": Cascade(pc_tier0(t0["high"], t0["high"]),
                               AlwaysAbstain("tier1"), tier2_fake),
        "tier0_1": Cascade(pc_tier0(t0["low"], t0["high"]),
                            pc_tier1(t1["high"], t1["high"]), tier2_fake),
        "full_cascade": Cascade(pc_tier0(t0["low"], t0["high"]),
                                 pc_tier1(t1["low"], t1["high"]), tier2_real),
        "per_step_llm": Cascade(AlwaysAbstain("tier0"), AlwaysAbstain("tier1"), tier2_real),
        "majority_floor": Cascade(AlwaysBenign(), AlwaysAbstain("tier1"), tier2_fake),
    }


def evaluate(name, cascade, records):
    rows = []
    t_start = time.time()
    for i, r in enumerate(records):
        for tier in cascade.tiers:
            if isinstance(tier, PrecomputedTier):
                tier.bind(i)
        res = cascade.run(r["user_instruction"], r["steps"])
        rows.append({
            "pair_id": r["pair_id"],
            "label": r["label"],
            "predicted": res.verdict,
            "total_cost_ms": res.total_cost_ms,
            "deciding_tier": res.deciding_tier,
            "llm_used": res.llm_used,
            "category": r["category"],
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(records) - (i + 1)) / rate / 60 if rate > 0 else float("nan")
            print(f"  [{name}] {i + 1}/{len(records)}  ({rate:.2f}/s, eta {eta:.1f} min)")

    out = {"config": name}
    out.update(summarise(rows))
    out.update(cost_summary(rows))
    out["paired_score"] = paired_score(rows)
    out["escalation"] = json.dumps(out["escalation"])
    return out, rows


def main():
    th = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    records = load_records(split="test")
    print(f"TEST split: {len(records)} trajectories. This is the single scored run.\n")

    print("precomputing tier0 (max_delta) scores for the whole test split...")
    t0_rows = batched_features(records)
    t0_scores = [min(1.0, max(0.0, r["max_delta"])) for r in t0_rows]

    print("precomputing tier1 (NLI) scores for the whole test split...")
    t1_scores = batched_nli_scores(records)

    results = []
    all_rows = {}
    for name, cascade in build_configs(th, t0_scores, t1_scores).items():
        print(f"\nrunning {name} ...")
        summary, rows = evaluate(name, cascade, records)
        results.append(summary)
        all_rows[name] = rows
        print(f"  detection={summary['detection_rate']:.3f} "
              f"fpr={summary['false_positive_rate']:.3f} "
              f"paired={summary['paired_score']:.3f} "
              f"llm/traj={summary['llm_calls_per_traj']:.3f} "
              f"latency={summary['avg_latency_ms']:.1f}ms\n")

    fields = list(results[0].keys())
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"wrote {OUT}")

    # Save per-trajectory rows too, for category/length breakdowns later.
    detail_out = Path(__file__).resolve().parent / "e3_e4_per_trajectory.json"
    detail = {name: rows for name, rows in all_rows.items()}
    detail_out.write_text(json.dumps(detail), encoding="utf-8")
    print(f"wrote {detail_out}")


if __name__ == "__main__":
    main()
