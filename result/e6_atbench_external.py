"""E6: external, zero-retuning generalization check on ATBench.

Not part of the original approved design/plan -- added per user direction as
a follow-up to the competitor analysis (docs/research/COMPETITOR_ANALYSIS.md),
because "you only evaluated on one benchmark you calibrated on" is the
single biggest credibility gap that analysis identified.

METHODOLOGY GUARDRAIL: thresholds are loaded from
database/processed/thresholds.json, which was frozen from TraceSafe
CALIBRATION data (result/calibrate.py) BEFORE this script ever ran. Nothing
here retunes on ATBench. This is evaluation only, and it is run exactly
once, same as the TraceSafe E3/E4 test-split run.

ATBench is a genuinely different benchmark: different source models,
different tool universe, different mutation/construction methodology
(LLM-based safety-relevant trajectory construction with human audit, vs
TraceSafe's systematic single-mutation-per-trace design). A large drop from
TraceSafe-test performance to ATBench performance is evidence of
overfitting to TraceSafe-specific artifacts; a small drop is evidence the
cascade's signal is more general. Either result is reported as-is.

ATBench also has no matched-twin pair structure (source/atbench_loader.py
sets pair_id = task_id for every record), so paired_score is not meaningful
here and is not reported for this experiment.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atbench_loader import load_all  # noqa: E402
from cascade import Cascade  # noqa: E402
from metrics import cost_summary, summarise  # noqa: E402
from tiers.base import TierResult, band_verdict  # noqa: E402
from tiers.tier2_llm import Tier2LLM  # noqa: E402

from e1_tier0_signals import batched_features  # noqa: E402
from e2_nli_signal import batched_nli_scores  # noqa: E402

THRESHOLDS = Path(__file__).resolve().parent.parent / "database" / "processed" / "thresholds.json"
OUT = Path(__file__).resolve().parent / "e6_atbench_results.csv"
OUT_BY_CATEGORY = Path(__file__).resolve().parent / "e6_atbench_by_category.csv"


class PrecomputedTier:
    def __init__(self, name, scores_by_index, low, high):
        self.name = name
        self.scores = scores_by_index
        self.low = low
        self.high = high
        self._idx = None

    def bind(self, idx):
        self._idx = idx
        return self

    def judge(self, task, steps):
        score = self.scores[self._idx]
        return TierResult(band_verdict(score, self.low, self.high), score, self.name, 0.1)


def main():
    th = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    records, skipped = load_all()
    print(f"ATBench: {len(records)} trajectories loaded, {skipped} skipped (empty)")

    print("computing tier0 (max_delta) scores...")
    t0_rows = batched_features(records)
    t0_scores = [min(1.0, max(0.0, r["max_delta"])) for r in t0_rows]

    print("computing tier1 (NLI) scores...")
    t1_scores = batched_nli_scores(records)

    t0 = th["tier0"]
    t1 = th["tier1"]
    tier0 = PrecomputedTier("tier0", t0_scores, t0["low"], t0["high"])
    tier1 = PrecomputedTier("tier1", t1_scores, t1["low"], t1["high"])
    tier2 = Tier2LLM(fake=False)
    cascade = Cascade(tier0, tier1, tier2)

    rows = []
    for i, r in enumerate(records):
        tier0.bind(i)
        tier1.bind(i)
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
            print(f"  {i + 1}/{len(records)}")

    summary = {"config": "full_cascade_on_atbench_zero_retune"}
    summary.update(summarise(rows))
    summary.update(cost_summary(rows))
    summary["escalation"] = json.dumps(summary["escalation"])
    print("\nATBench external result (zero retuning, TraceSafe-frozen thresholds):")
    print(json.dumps(summary, indent=2))

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    print(f"wrote {OUT}")

    # Per risk_source category (malicious side only has a category; benign is "SAFE")
    benign = [r for r in rows if r["label"] == "benign"]
    cats = sorted({r["category"] for r in rows if r["label"] == "malicious"})
    cat_rows = []
    for cat in cats:
        subset = benign + [r for r in rows if r["category"] == cat]
        s = summarise(subset)
        s["category"] = cat
        cat_rows.append(s)
    with open(OUT_BY_CATEGORY, "w", newline="", encoding="utf-8") as fh:
        if cat_rows:
            w = csv.DictWriter(fh, fieldnames=list(cat_rows[0].keys()))
            w.writeheader()
            w.writerows(cat_rows)
    print(f"wrote {OUT_BY_CATEGORY}")


if __name__ == "__main__":
    main()
