# Limitations

Written to be read alongside `results_interpretation.md` and the paper's own
Limitations section — nothing here is softened for the write-up.

## Dataset

1. **TraceSafe-only calibration and (mostly) evaluation.** The core cascade
   is calibrated and test-scored entirely within TraceSafe-Bench
   (arXiv:2604.07223). A separate zero-retuning external check on ATBench
   (arXiv:2604.02022) was added (E6) specifically to probe this, but a
   single external dataset does not eliminate the concern — see
   `docs/research/COMPETITOR_ANALYSIS.md` for the full discussion.
2. **TraceSafe's "malicious" label blends two qualitatively different
   failure types**: genuine intent-divergence attacks (`PROMPT_INJECTION`,
   `PRIVACY_LEAKAGE`) and operational defects (`HALLUCINATION`,
   `INTERFACE_INCONSISTENCIES`). E1/E2 confirm this matters empirically —
   Tier 0's delta signal is only meaningfully discriminative on
   `PRIVACY_LEAKAGE` (AUC 0.724), and near-chance elsewhere. A blended
   overall number would misrepresent what the monitor detects; every result
   in this project is reported per-category as well as overall specifically
   because of this.
3. **Benign twins are a controlled construction, not naturally-occurring
   benign traffic.** They are the *same* trace with the mutation removed —
   ideal for isolating the divergence signal (design spec's own
   justification), but they are not independent samples of "what a normal
   user's agent does," so false-positive rate on twins should not be read
   as an estimate of real-world false-positive rate on arbitrary agent
   traffic.
4. **TraceSafe's mutations are fixed and not adaptively red-teamed against
   ShadowTrace.** MonitoringBench (arXiv:2605.09684) shows monitor catch
   rate can fall from ~95% on naturally-elicited attacks to ~60% under
   monitor-aware refined attacks. ShadowTrace's measured detection rate
   should be read as an upper bound relative to a determined,
   monitor-aware attacker, not a robustness guarantee.

## Method

5. **NLI (Tier 1) did not repair the topic-vs-goal problem** (E2: overall
   ROC-AUC 0.516, worse than chance on `PRIVACY_LEAKAGE` specifically,
   0.472). This is a genuine negative result for the design spec's stated
   hypothesis. Per-example probing (documented in
   `source/tiers/tier1_nli.py`) shows the raw entailment signal is noisy
   for premise=task/hypothesis=tool-call-sentence pairs — lexical overlap
   can outweigh semantic alignment. Consistent with the design's own risk
   register ("NLI brittle on tool-call text, not natural prose").
6. **Tier 1's input is capped at 128 tokens** for wall-clock tractability on
   a 2-core CPU host (measured ~5x speedup vs the model's default 512-token
   max, see `tiers/tier1_nli.py`). This uniformly truncates tail context on
   longer trajectories (p95 measured at 381 tokens pre-truncation) —
   applied identically regardless of label, so it is not a label-correlated
   bias, but it is a real information loss that could understate Tier 1's
   true ceiling performance.
7. **No cross-step evidence memory.** Every tier scores steps independently
   (or takes the single worst step); nothing carries structured state
   between steps the way TRACE's (arXiv:2606.07054) evidence-retention
   architecture does. A multi-step attack that only looks suspicious when
   two distant steps are connected (e.g. "read credentials at step 2" +
   "post publicly at step 9" with nothing suspicious in between) is exactly
   the class of failure this architecture is not built to catch.
8. **No monitor-collusion control.** Tier 2 (Qwen2.5-1.5B-Instruct) and
   TraceSafe's trace-generation models (`toolace-8b`, `gpt-5-mini-2025-08-07`,
   `mistral-14b`, `qwen`, `gemini-3-flash-preview`, `gpt-4o-mini-2024-07-18`)
   are not controlled for shared lineage. TraceGuard (arXiv:2604.03968)
   shows a conventional single-score monitor's safety can collapse to 0%
   under collusion conditions. Not tested here.
9. **No adversarial blind-spot stress testing.** SLEIGHT-Bench
   (alignment.anthropic.com/2026/sleight-bench/, arXiv:2605.16626)
   documents specific monitor blind spots (file/object reuse, counting,
   authorization confusion, multi-hop reasoning) that generalist LLM
   monitors — including, presumably, Tier 2 — are known to miss at a high
   rate. ShadowTrace has not been tested against transcripts constructed to
   probe these specific failure modes.
10. **No early-warning / detection-delay metric.** ShadowTrace reports
    whether a trajectory is flagged, not how many steps before a harmful
    action it fires — unlike SafetyDrift's (arXiv:2603.27148) explicitly
    predictive framing. Adding this would require step-level "which step is
    the actual pivot" ground truth beyond what the current parser extracts
    from TraceSafe.

## Statistics

11. **Single calibration/test split, single seed.** No bootstrap confidence
    intervals, no multi-seed robustness check. A single-seed AUC/detection
    number can move meaningfully on a corpus of this size; point estimates
    in this report should be read with that in mind, not as precise
    population parameters.
12. **Median trajectory length is short (~5 steps, design spec).** This
    limits how much sequence-level signal any method — including
    ShadowTrace — can extract in principle, independent of how good the
    method is.

## Infrastructure

13. **CPU-only, 2 cores.** All Tier 1/Tier 2 timing numbers reported in this
    project are specific to this hardware, not a general claim about
    cascade latency on arbitrary hardware. The relative gap between tiers
    (which is the efficiency claim) should transfer; the absolute
    millisecond numbers will not.
