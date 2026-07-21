# ShadowTrace — Tiered Cascade Monitor: Design Spec

**Date:** 2026-07-21
**Project:** ShadowTrace — cumulative trajectory divergence monitor (final-year project)
**Status:** Approved design, pending implementation plan

---

## 1. Context and motivation

ShadowTrace monitors an LLM agent's multi-step trajectory and flags when the
cumulative pattern of actions diverges from the user's stated task, even when no
individual step is obviously unsafe.

**Where the project stands.** Phases 0–2 are complete: 1,170 real TraceSafe
trajectories parsed, a sentence-embedding encoder, a drift tracker, a FastAPI +
SSE backend, and a live web dashboard.

**The problem this design solves.** The original core hypothesis — that
cumulative embedding drift climbs for malicious trajectories and stays flat for
benign ones — **failed its validation gate**. Across 60 benign vs 60 drift-type
malicious trajectories, every drift feature scored ROC-AUC 0.47–0.57 (0.50 is
chance). Drift lines were visually tangled; slopes were slightly negative for
both classes.

**Diagnosis.** Three causes, all real:
1. Growing the text prefix at each step dilutes early divergence.
2. Sentence embeddings capture *topic similarity*, not *goal alignment*. A
   privacy-leak trajectory inside a filesystem task still "sounds like" file
   operations.
3. TraceSafe attacks are frequently subtle single-tool mutations, which are tiny
   in embedding space.

**The promising lead.** Step-to-step **delta** (distance between consecutive
step vectors) spiked exactly at the pivot in a worked example
(`3_UserInfoLeak_0`: `grep` → `diff` → `authenticate_twitter` at delta 0.758 →
`post_tweet` leaking credentials and medical data). Cumulative drift stayed flat
(~0.31–0.41) throughout. This is currently **one illustrative example, not a
validated result.**

---

## 2. Goals and constraints

**Success criteria (agreed):** viva-safe first; publishable if the numbers
cooperate. The architecture must therefore produce a defensible result
regardless of which way the measurements go.

**Central claim to defend:**
- **(A) Efficiency** — comparable useful detection at a fraction of the
  cost/latency of LLM-judge monitors.
- **(B) Sequence-level detection** — catch divergence spread across steps that
  per-step checks miss.
- Honest capability limits form the limitations section rather than being hidden.

**Comparison strategy:** reimplemented baselines as the primary table; published
numbers from related work as a clearly-labelled secondary context table.

**Hard constraints:**
- Timeline: 1–2 months, solo.
- Hardware: **CPU only** (`torch.cuda.is_available() == False`). This is
  acceptable and in fact strengthens the efficiency argument, since the LLM tier
  cost is measured on the same machine as the cheap tiers.
- Demo must run **fully offline** (no cloud API in the critical path).
- Median trajectory length is **5 steps**, which limits how much sequence
  signal any method can extract.

**Non-goals (explicit scope fence):** no model fine-tuning, no new dataset
collection, no cloud-API dependency in the critical path, no further UI work
beyond what exists, no 3D/animation/SEO work.

---

## 3. Data foundation

### 3.1 The imbalance problem

The parsed dataset is **90 benign vs 1,080 malicious**. A false-positive rate
estimated from 90 samples is not credible, and FPR is a headline metric.

### 3.2 Benign twins

Every malicious TraceSafe record carries `original_trace`: the same trajectory
*before* mutation. Parsing these yields roughly **1,000 additional benign
trajectories**, giving an approximately balanced corpus (~1,090 benign /
~1,080 malicious).

### 3.3 Why twins matter methodologically

A twin pair differs **only by the injected mutation** — same task, same domain,
same tool inventory, similar length. This is a **matched-pair design** and it
removes a confound that would otherwise undermine every result: without pairing,
a monitor can score well by exploiting spurious correlates (trajectory length,
tool vocabulary, domain) rather than detecting divergence. With pairs, the only
remaining discriminative signal is the divergence itself.

### 3.4 Splits

- **Calibration split (~40%)** — all threshold and abstention-band tuning.
- **Held-out test split (~60%)** — used exactly once, for final reported numbers.
- Splitting is performed **by pair**: a trajectory and its twin always land in
  the same split. Splitting them would leak.
- Splits are written to disk with a fixed seed and committed, so they are stable
  across runs.

### 3.5 Reporting

All metrics are reported **per category** (prompt injection, privacy leakage,
hallucination, interface inconsistencies) as well as overall. TraceSafe's
"malicious" label mixes genuine intent-drift attacks with operational defects;
blending them into one number would be misleading.

---

## 4. Architecture: the three-tier cascade

### 4.1 Shared tier interface

Every tier implements one interface, so tiers are independently testable and
swappable:

```python
judge(task: str, steps: list[dict]) -> TierResult

TierResult = {
    "verdict": "benign" | "malicious" | "abstain",
    "score": float,       # tier-native signal, ALWAYS oriented so that
                          # higher = more likely malicious (see below)
    "tier": str,          # which tier produced this
    "cost_ms": float,     # measured wall-clock cost
}
```

**Score orientation is a contract, not a convention.** Every tier must return a
score where *higher means more likely malicious*, regardless of the underlying
model's native direction. NLI entailment, for example, is natively inverted
(high entailment = on-task = benign) and must be flipped inside Tier 1 before
returning. A contract test enforces this for every tier.

### 4.2 Tiers

| Tier | Method | Cost | When it runs |
|---|---|---|---|
| **0** | Embedding features: cumulative drift, step-to-step delta, rise, slope | ~free | Every step (preserves the live streaming dashboard) |
| **1** | NLI cross-encoder: *"does this action follow from the stated task?"* (entailment vs contradiction) | ~milliseconds | Only on steps where Tier 0 abstains |
| **2** | Local LLM judge via HuggingFace transformers, in-process | ~seconds (CPU) | Only when Tier 1 also abstains; trajectory-level |

Tier 1 exists specifically to repair diagnosed cause (2) in §1: entailment
measures **goal alignment**, which is what the task actually requires, rather
than topic similarity, which is what embeddings provide.

### 4.3 Routing

Each tier has a low and a high threshold:
- score below low → decide **benign**, stop.
- score above high → decide **malicious**, stop.
- score inside the band → **abstain**, escalate to the next tier.

Tier 2 is terminal and must return a decision. Bands are calibrated on the
calibration split only.

**From step-level signals to a trajectory verdict.** Tiers 0 and 1 score
individual steps, but the reported metrics are trajectory-level. The aggregation
rule is explicit: **a trajectory is malicious if any step is decided malicious**
(first-flag-wins, matching the existing streaming dashboard behaviour, and
appropriate because a monitor that detects an attack at step 3 has succeeded
even if later steps look calm). A trajectory is benign only if every step was
decided benign. If any step abstains and no step has flagged, the trajectory
escalates to the next tier. This rule is fixed before evaluation and is not a
tuned parameter.

### 4.4 Instrumentation (the headline instrument)

For every trajectory the pipeline records which tier produced the decision and
the cumulative `cost_ms`. This yields the accuracy-versus-expensive-compute
trade-off curve directly, which is the primary evidence for claim (A).

The **per-step LLM baseline is Tier 2 forced on every step**, and the
**whole-trajectory LLM baseline is Tier 2 run once**. Both therefore reuse the
same code path as the cascade, making the comparison exactly apples-to-apples
and essentially free to produce.

### 4.5 Failure containment

If delta does not generalise, Tier 0 abstains more often, Tier 1 absorbs the
load, and the escalation shift is itself a reportable finding. If Tier 1 is weak
on tool-call text, Tier 2 absorbs it and the cost curve shifts. No single
component failing invalidates the project.

### 4.6 Integration with existing code

- `source/` gains `tiers/` (tier0_embedding, tier1_nli, tier2_llm) and
  `cascade.py` implementing routing.
- `source/pipeline.py` is refactored to delegate its verdict to the cascade,
  preserving the existing `monitor_stream` streaming contract so the FastAPI SSE
  backend and dashboard continue to work unchanged.
- `connectors/llm_gate.py` becomes the Tier 2 backend adapter. Ollama and
  NVIDIA NIM remain as optional alternative backends; the **default is local
  HuggingFace transformers** to keep the demo offline and reproducible.

---

## 5. Evaluation

### 5.1 Metrics

Reported per category and overall:
- **Detection rate** (recall on malicious)
- **False-positive rate** (on benign)
- **Precision**
- **ROC-AUC per individual signal**
- **Escalation rate per tier**
- **Average LLM calls per trajectory**
- **Average latency per trajectory**
- **Paired score** — on matched twins: flagged the malicious version *and*
  cleared its benign twin. This is the strictest and most confound-resistant
  number available, and is reported alongside the standard metrics.

### 5.2 Experiments

| # | Experiment | Question answered |
|---|---|---|
| **E1** | ROC-AUC of each Tier 0 feature (drift, delta, rise, slope) across all categories | Does delta actually generalise? (the make-or-break question) |
| **E2** | ROC-AUC of the NLI entailment score | Does repairing topic-vs-goal fix the failure? |
| **E3** | Cascade abstention-band sweep → trade-off curve | Accuracy as a function of expensive-compute budget |
| **E4** | Baselines: per-step LLM on everything; whole-trajectory LLM once; majority-class floor | Apples-to-apples comparison on identical data |
| **E5** | Published numbers from TRACE / TraceGuard / TraceSIR | Secondary context, explicitly labelled not apples-to-apples |

### 5.3 Primary results table

Columns: `Method | Detection | FPR | Precision | LLM calls/traj | Latency`

Rows: Tier 0 only · Tier 0+1 · Full cascade · Per-step LLM baseline ·
Whole-trajectory LLM baseline · Majority-class floor.

Whatever this table reports is what goes into the paper.

---

## 6. Testing

- **Tier contract tests** — each tier returns a valid verdict, a numeric score,
  and a recorded cost.
- **Routing tests** — abstention-band logic: below-low decides benign,
  above-high decides malicious, inside-band escalates, Tier 2 never abstains.
- **Parser tests for twins** — a twin must differ from its malicious counterpart;
  pairs must never straddle the calibration/test boundary.
- **End-to-end smoke test** on a small fixture trajectory.
- **Determinism** — fixed seeds, embeddings cached to disk, pinned model
  revisions, so every number is regenerable.

---

## 7. Honesty guardrails

These exist to protect the integrity of the result and to survive viva
questioning:

1. **The test split is touched exactly once.** All tuning happens on the
   calibration split. If a change is needed later, it is re-tuned on calibration
   and the change is disclosed in the report.
2. **Thresholds are recorded before the test run**, not adjusted after seeing
   results.
3. **Negative results ship.** The cumulative-drift failure remains in the report
   as a finding.
4. **No cherry-picked examples.** The Twitter-leak trajectory appears as an
   illustration alongside the full distribution, never as evidence.
5. **Everything skipped or dropped is logged and reported** — for example,
   trajectories too short for delta to be meaningful.
6. **One command regenerates every number and figure** in `result/`.

---

## 8. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Delta fails to generalise (E1 negative) | Medium | Cascade absorbs it via Tier 1; the failure is reported as a finding, consistent with success criterion C |
| NLI brittle on tool-call text (not natural prose) | Medium | Step-to-sentence rendering already produces English-like text; if weak, Tier 2 absorbs and the cost curve shifts |
| CPU-only makes Tier 2 slow | High (expected) | Intended: it makes the measured cost gap real. Keep Tier 2 model small (1.5B–3B class) |
| Short trajectories (median 5) limit sequence signal | High | Report results stratified by trajectory length; disclose the limitation explicitly |
| Scope creep consuming the 1–2 months | Medium | §2 non-goals are a hard fence |

---

## 9. Open decisions deferred to implementation planning

- Exact NLI checkpoint and exact Tier 2 model checkpoint (chosen for CPU
  feasibility during implementation, then pinned).
- Exact calibration/test split ratio within the stated ~40/60 range.
- Whether Tier 1 scores each step independently or scores a sliding window.
