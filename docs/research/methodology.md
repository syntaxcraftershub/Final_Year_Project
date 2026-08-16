# Methodology

## Dataset

TraceSafe-Bench (arXiv:2604.07223, Apache-2.0), downloaded from
`github.com/yenshan0530/TraceSafe` (mirrors `CyCraftAI/TraceSafe` on
Hugging Face). 1,170 raw records: 90 pure-benign baselines
(`golden_0_benign.jsonl`) plus 1,080 mutated records across 12 risk
categories under 4 root vectors (`PROMPT_INJECTION`, `PRIVACY_LEAKAGE`,
`HALLUCINATION`, `INTERFACE_INCONSISTENCIES`).

## Benign twin extraction

Every mutated record carries `original_trace`, the unmutated trace before
the injected mutation. `source/data_parser.py`'s `parse_record` reads either
`new_trace` (mutated) or `original_trace` (twin) via a `trace_key`
parameter, tagging both with a shared `pair_id` (the raw record's task_id)
so they can be kept together through splitting. Pure-benign records
(`mutation_category == "BENIGN"`) are not duplicated — their `new_trace`
already equals `original_trace`, so a "twin" would be a byte-identical
record, not a control; this is a deliberate, counted skip, not silent data
loss. Result: 2,250 processed trajectories, 1,170 benign (90 pure +
1,080 twins) vs 1,080 malicious — ratio 1.08, resolving the original 12:1
imbalance.

## Splitting

`source/splits.py` assigns each `pair_id` (not each record) to calibration
or test, seed 42, target 40% calibration / 60% test by pair count. A
malicious record and its benign twin always land in the same split — this
is enforced by grouping on `pair_id` before the random shuffle, and
verified by `tests/test_splits.py::test_pairs_never_straddle_the_split`.
Result: 468 calibration pairs (902 records) / 702 test pairs (1,348
records).

## Tiered cascade

Three tiers, shared contract `judge(task, steps) -> TierResult(verdict,
score, tier, cost_ms)`, score always oriented higher = more malicious
(`source/tiers/base.py`, enforced by `tests/test_tiers.py`).

- **Tier 0** (`source/tiers/tier0_embedding.py`): `all-mpnet-base-v2`
  sentence embeddings. Decision signal is `max_delta`, the largest
  step-to-step cosine distance anywhere in the trajectory (cumulative drift
  is computed and reported but is not the decision signal, given its known
  ROC-AUC ~0.5 failure from the prior Phase 3 gate).
- **Tier 1** (`source/tiers/tier1_nli.py`): `cross-encoder/nli-deberta-v3-base`.
  Premise = task instruction, hypothesis = each step rendered as English via
  `encoder.step_to_sentence`. Score = `1 - P(entailment | not-neutral)`,
  where the "not-neutral" renormalization (softmax over
  {contradiction, entailment} only) was adopted after diagnosing that the
  naive 3-way softmax is dominated (>98%) by the neutral class for these
  premise/hypothesis pairs — a fix decided from model-behavior diagnostics
  before any calibration or evaluation run, not from labeled outcomes.
  `max_seq_length` capped at 128 tokens for CPU tractability (see
  `docs/research/limitations.md` item 6).
- **Tier 2** (`source/tiers/tier2_llm.py`): `Qwen/Qwen2.5-1.5B-Instruct` via
  HuggingFace `transformers`, CPU (`device=-1`), greedy decoding
  (`do_sample=False`), single-word BENIGN/FLAGGED prompt. Terminal — must
  return a decision, never abstains (enforced by
  `tests/test_cascade.py::test_terminal_tier_abstaining_raises`).

## Routing

`source/cascade.py`: each tier's score is compared against a low/high band
(`tiers/base.py::band_verdict`). Below low → benign, above high →
malicious, in between → abstain and escalate to the next tier. Tier 2's
band collapses to a point (it always decides). Cost accumulates across every
tier actually invoked, whether or not it decided.

## Calibration

`result/calibrate.py`: bands are chosen on the calibration split only, using
a tolerance-quantile rule (default 2%) — the low threshold is the score
below which at most 2% of malicious calibration records fall, the high
threshold is the score above which at most 2% of benign calibration records
fall. Frozen to `database/processed/thresholds.json` (including seed, model
identifiers, and the calibration record count) before any test-split
evaluation.

## Trajectory aggregation

Fixed before evaluation, never tuned: a trajectory is malicious if any tier
decides malicious (first-flag-wins). It is benign only if every invoked tier
decided benign for it. This matches the existing streaming dashboard's
behavior and the design spec's explicit rationale — a monitor that catches
an attack at step 3 has succeeded even if the rest of the trajectory looks
calm.

## Metrics

`source/metrics.py`: detection rate (recall on malicious), false-positive
rate (on benign), precision, paired score (both the malicious trajectory
AND its benign twin judged correctly — the strictest, most
confound-resistant number, since a matched pair differs only by the
injected mutation), escalation rate per tier, average LLM calls per
trajectory, average latency.

## Experiments

E1 (Tier 0 feature ROC-AUC), E2 (Tier 1 NLI ROC-AUC) — calibration split
only, pre-registered interpretation thresholds written into the scripts
before running. E3+E4 (cascade configurations vs. reimplemented baselines)
— the single scored run on the test split. E5 (published-numbers secondary
context, explicitly not apples-to-apples). E6 (ATBench external,
zero-retuning validation) — added after the fact per user direction
following competitor analysis; not part of the originally approved design,
disclosed as such throughout.

## Performance adaptations (disclosed, not silent)

This host has 2 CPU cores and no GPU. Two changes were made purely for
wall-clock tractability, both decided from diagnostics rather than from
labeled results, and both documented at the point they were introduced:
batched (rather than per-record) embedding/NLI calls in
`result/e1_tier0_signals.py` / `result/e2_nli_signal.py` /
`result/calibrate.py` / `result/e3_e4_cascade_and_baselines.py` (same
feature math, different call pattern), and the 128-token NLI truncation
noted above. Neither changes what is measured, only how fast it is computed.
