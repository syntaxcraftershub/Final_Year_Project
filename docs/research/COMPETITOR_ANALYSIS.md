# Competitor / Related-Work Analysis

Every entry below verified live against its primary source (arXiv abstract page,
official blog, or GitHub repo) on 2026-08-16, before writing anything — not
copied blind from a third-party summary. Numbers are quoted from the source;
none are recomputed or reproduced by ShadowTrace. Per the project's own
honesty rules, none of these numbers are comparable to ShadowTrace's own
measurements unless explicitly stated (different datasets, different setups,
different threat models) — this file is **secondary context**, same status
as `result/e5_published_context.csv`.

| # | System | Source | Verified? |
|---|---|---|---|
| 1 | SafetyDrift | arXiv:2603.27148 | Confirmed live (arxiv.org/abs/2603.27148) |
| 2 | TRACE | arXiv:2606.07054 | Confirmed live, abstract read directly |
| 3 | TraceGuard | arXiv:2604.03968 | Confirmed live, abstract read directly |
| 4 | AgentDoG | arXiv:2601.18491 | Confirmed live (arxiv.org, HF papers, GitHub AI45Lab/AgentDoG) |
| 5 | ATBench | github.com/LiYu0524/ATbench | Confirmed live, README read directly |
| 6 | SHADE-Arena | arXiv:2506.15740 | Confirmed live, abstract read directly |
| 7 | SLEIGHT-Bench | alignment.anthropic.com/2026/sleight-bench/ + arXiv:2605.16626 | Confirmed live, page read directly |
| 8 | MonitoringBench | arXiv:2605.09684 | Confirmed live (arxiv.org, OpenReview mirror) |
| 9 | TraceSafe-Bench | arXiv:2604.07223 | Already ShadowTrace's dataset (see README) |

---

## 1. SafetyDrift (arXiv:2603.27148)

**Method:** models an agent's safety trajectory as an absorbing Markov chain
and computes the probability of reaching a violation state within N steps —
explicitly *predictive*, not just post-hoc classification.

**Reported numbers (their setup, not comparable to ours):** 94.7% violation
detection, ~3.7 steps of average advance warning, large claimed speedup over
their compared baselines, evaluated on 357 traces across 40 tasks; violation
risk varies sharply by task type (e.g. ~85% within 5 steps for communication
tasks vs <5% for technical tasks).

**Limitation (from the paper's own framing):** depends on a predefined
monotonic safety-state model and task/category-specific dynamics — a strong
modeling assumption, not a general-purpose, model-agnostic monitor.

**Relevance to ShadowTrace:** closest conceptual competitor — same core
phenomenon (individually-acceptable actions accumulating into an unsafe
trajectory). ShadowTrace does **not** claim to beat SafetyDrift's numbers
(different dataset, different formulation); the honest positioning is that
ShadowTrace is model-agnostic (no predefined safety-state model) at the cost
of not being explicitly predictive the way SafetyDrift is. An early-warning /
detection-delay metric (does the cascade flag before the harmful step, and
how many steps early) is a natural, low-cost addition given TraceSafe already
labels the pivot step for drift-type categories — logged as a candidate
future-work metric in `docs/research/limitations.md`, not implemented in this
pass (would require step-level "which step is the pivot" ground truth beyond
what the current parser extracts).

## 2. TRACE (arXiv:2606.07054)

**Method:** Triage → Inspect → Judge. Identifies high-signal regions of a
trajectory, inspects them while retaining accumulated evidence from earlier
steps, then issues a trajectory-level verdict.

**Reported numbers (SHADE-Arena, their setup):** F1 0.713, recall 0.844,
tested across ten domains, particularly strong on tasks needing
long-distance evidence connection.

**Limitation (from the paper/related summaries):** evaluated only on
SHADE-Arena; limited generalization evidence; fixed threshold/inspection
budget; narrower threat-class coverage (primarily sabotage).

**Relevance to ShadowTrace:** TRACE's "retain evidence from earlier steps"
is a genuinely different architecture from ShadowTrace's stateless
per-step-then-aggregate tiers. ShadowTrace's three-tier cascade already
implements a cheap-to-expensive escalation, which is the same instinct as
TRACE's Triage step, but ShadowTrace's tiers do not currently carry explicit
structured evidence forward between steps (a "cross-step evidence ledger").
This is a legitimate architectural gap, noted here and in
`docs/research/limitations.md` as future work — not retrofitted into the
already-approved and partially-implemented three-tier design mid-build,
per the standing instruction not to redesign methodology without strong
evidence AND without disrupting an implementation that is mid-flight and
has passing tests against the original spec.

## 3. TraceGuard (arXiv:2604.03968)

**Method:** evaluates five dimensions (goal alignment, constraint adherence,
reasoning coherence, safety awareness, action-trace consistency) via
parallel independent LLM evaluations plus heuristic detectors, specifically
to resist **monitor collusion** (a monitor built from the same model family
as the agent it watches).

**Reported numbers (519 samples, their setup):** 0.616 vs 0.206 mean
separation between attack and honest behavior; 95% safety maintained under
collusion conditions vs 0% for a conventional single-score monitor.

**Limitation:** heavily LLM-call-based (multiple parallel judges), which is
expensive — the opposite trade-off ShadowTrace is built around.

**Relevance to ShadowTrace:** directly validates ShadowTrace's central
efficiency thesis by contrast — TraceGuard buys robustness with more
expensive inference; ShadowTrace's cascade tries to buy comparable
robustness signal with mostly-cheap inference. TraceGuard's collusion point
is real and ShadowTrace's design does NOT currently test for it (Tier 2 and
the TraceSafe trajectories may share model lineage in ways not controlled
for). This is disclosed as a limitation, not silently ignored — see
`docs/research/limitations.md`.

## 4. AgentDoG (arXiv:2601.18491)

**Method:** fine-grained diagnostic guardrail — not just safe/unsafe, but a
three-dimensional risk taxonomy, with smaller (4B-8B class) models.

**Relevance to ShadowTrace:** demonstrates that fine-grained diagnosis
(risk *type*, not just a verdict) is achievable at moderate model scale.
ShadowTrace's category-level reporting (Phase 9 of the research plan) is a
coarser version of the same idea (reporting per TraceSafe category rather
than per-trajectory diagnosis). A per-trajectory diagnostic breakdown
(which of the four root vectors is implicated, with evidence) is a
reasonable extension, not built in this pass.

## 5. ATBench (github.com/LiYu0524/ATbench)

**What it is:** 1,000 trajectories (503 safe / 497 unsafe), ~9.01 turns
average, human-audited, three-dimensional diagnosis (risk source, failure
mode, real-world harm), broad tool coverage (2,084 available tools, 1,954
actually invoked).

**Relevance to ShadowTrace — this is the single most important gap
identified in this analysis.** ShadowTrace calibrates AND tests entirely
within TraceSafe. A reviewer's obvious challenge: "how do we know the
cascade learned general drift signal rather than TraceSafe-specific
artifacts (its particular mutation styles, its particular tool-call
rendering)?" ATBench is a plausible external validation set — same task
type (trajectory-level agent safety), different construction. **Not run in
this pass** (see `docs/research/limitations.md` — flagged as the highest-value
follow-up experiment, requires adapting the data loader to ATBench's schema
and would need to be a strict zero-retuning evaluation using thresholds
already frozen from TraceSafe calibration, to be methodologically valid).

## 6. SHADE-Arena (arXiv:2506.15740)

**Method:** tests whether models can pursue a hidden, malicious side-goal
while completing a legitimate main task, undetected by a monitor.

**Reported numbers (their setup):** frontier agent sabotage success 27%
(Claude), 15% (Gemini); best monitor AUC ~0.87; the paper notes that
reaching 80% sabotage detection can require >20% false positives (called
impractical), and that monitors often flag isolated suspicious steps
without correctly connecting a multi-step attack.

**Relevance to ShadowTrace:** the "80% detection needs >20% FPR" finding is
exactly why ShadowTrace should not report a single accuracy number and stop
— it is why `source/metrics.py`'s `summarise()` already reports detection
rate and false-positive rate as separate numbers rather than one blended
accuracy figure, and why the calibration bands are tuned toward
high-precision decisions (§ design spec 4.3) with abstention as the
release valve, rather than one operating point.

## 7. SLEIGHT-Bench (alignment.anthropic.com/2026/sleight-bench/, arXiv:2605.16626)

**Method:** 40 synthetic transcripts specifically constructed around known
monitor blind spots (coding agents pursuing hidden goals like credential
theft or sabotage).

**Reported numbers (Claude Opus 4.6 as monitor, their setup):** 50% of
attacks never caught across any of 10 trials; only 8/40 reliably detected.
Blind-spot catch rates: file/object reuse 15%, counting 13%, system-state
tracking 46%, model priors 13%, authorization confusion 55%. Targeted
prompt fixes for one blind spot degrade at least one other category.

**Relevance to ShadowTrace:** this is a strong, sobering finding that
applies to LLM-judge monitors in general, including ShadowTrace's own Tier
2. ShadowTrace has **not** been tested against this class of adversarially
constructed blind-spot transcript. Given the scope and CPU-only timeline
constraints already documented in the design spec (§2), building a
comparable blind-spot stress suite from scratch is out of scope for this
pass — but pretending robustness against these known failure modes without
testing for them would misrepresent the system. Flagged explicitly in
`docs/research/limitations.md`.

## 8. MonitoringBench (arXiv:2605.09684)

**Method:** semi-automated red-teaming — 2,644 attack trajectories built to
specifically evade monitors, rather than naturally-occurring "elicited"
attacks.

**Reported numbers (their setup):** monitor catch rate falls from 94.9% on
ordinary elicited attacks to 60.3% on refined, monitor-aware attacks.

**Relevance to ShadowTrace:** TraceSafe's mutations are not adversarially
optimized against ShadowTrace specifically (they are fixed, pre-generated
mutations, not an adaptive red-team loop). This means ShadowTrace's
measured detection rate should be read as an upper bound relative to what a
determined, monitor-aware attacker could achieve — a caveat that belongs in
`docs/research/limitations.md`, not something the current evaluation can
correct for within scope.

## 9. TraceSafe-Bench (arXiv:2604.07223) — ShadowTrace's own dataset

Already cited in `README.md` and `docs/research/CURRENT_STATE.md`. Its
headline finding — guardrail performance correlates with structural
data-parsing competence (ρ=0.79 with structured-to-text benchmarks) more
than with jailbreak robustness — is the direct motivation for
`step_to_sentence`'s plain-English rendering already in `source/encoder.py`,
and for treating "hallucination"/"interface inconsistency" categories as
qualitatively different from "prompt injection"/"privacy leakage" categories
throughout this project's reporting.

---

## What this means for ShadowTrace, honestly

ShadowTrace's approved design (embeddings → NLI → local LLM cascade,
evaluated on TraceSafe with matched twins and a leakage-safe split) remains
sound and is being implemented as specified — this analysis does not
overturn it. What it does surface, honestly, or which this document commits
to as newly-identified, disclosed limitations rather than silent gaps:

1. No external-dataset validation (ATBench) — biggest single credibility
   gap for a reviewer.
2. No adversarial/blind-spot stress testing (SLEIGHT-Bench-style,
   MonitoringBench-style) — ShadowTrace's measured numbers should be read
   as best-case, not adversarially-robust.
3. No collusion/monitor-independence check (TraceGuard's point) — Tier 2's
   model lineage relative to TraceSafe's trace-generation models is not
   controlled for.
4. No cross-step evidence memory (TRACE's architecture) — each tier scores
   steps independently (worst-step-wins), not with carried state.
5. No early-warning / detection-delay metric (SafetyDrift's framing) —
   ShadowTrace reports whether a trajectory is flagged, not how many steps
   before the actual harmful action.

These are recorded as scoped-out, disclosed future work in
`docs/research/limitations.md`, not implemented in this pass, because: (a)
the approved design and implementation plan this project is executing
against does not include them, (b) retrofitting them mid-implementation
without the same TDD/calibration discipline applied to everything else
would be exactly the kind of undisciplined scope change the project's own
honesty guardrails warn against, and (c) the CPU-only, solo,
already-large-scope timeline constraint (design spec §2) is real. Listing
them here, with sources, is the mechanism for making sure they are not
silently forgotten either.
