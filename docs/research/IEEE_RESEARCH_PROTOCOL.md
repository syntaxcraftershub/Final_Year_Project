# ShadowTrace — IEEE Research Protocol

## Research title

**ShadowTrace: A Cost-Aware Cascade Framework for Detecting Multi-Step Intent Divergence in AI Agent Trajectories**

## Research question

Can a selective monitoring cascade combine inexpensive semantic signals, local NLI reasoning, and a local LLM adjudicator to detect multi-step agent intent divergence while reducing expensive LLM invocation?

## Core hypothesis

A trajectory monitor should not treat every action independently or invoke an LLM for every action. Cheap signals should reject clearly safe/unsafe cases, NLI should handle ambiguous goal/action relationships, and a local LLM should adjudicate only the residual gray zone.

## Formalization

Let the user task be `T` and an agent trajectory be `A = {a1, ..., an}`, where each action is represented as `(tool, parameters, result)`. The monitor estimates `f(T,A) -> {benign, malicious}` while minimizing inference cost `C(f)` subject to an acceptable detection quality `D(f)`.

For normalized embeddings `vT`, `vi`, cumulative prefix drift is:

`D_i = 1 - cos(vT, v_{1:i})`

and consecutive-step delta is:

`Delta_i = 1 - cos(v_{i-1}, v_i)`.

Tier 0 uses these inexpensive signals. Tier 1 estimates:

`P(contradiction | T, a_i)`

using a local NLI cross-encoder. Tier 2 sends the ambiguous trajectory prefix to a local Ollama-hosted Qwen model.

## Experimental rules

1. Freeze the final test set before tuning thresholds.
2. If mutation/original pairs exist, keep the complete pair in one split.
3. Never claim a metric until a script produces it.
4. Preserve negative results, including failed cumulative-drift experiments.
5. Report per-category results, not only an aggregate number.
6. Record model names, versions/tags, quantization, runtime, hardware, prompt, thresholds, seed, and latency.
7. Report LLM invocation rate as a first-class efficiency metric.
8. Compare against majority, cumulative drift, delta, Tier-0-only, Tier-1-only, full-LLM, and the full cascade.
9. Do not tune the test set.
10. Do not claim IEEE/Scopus acceptance or indexing unless verified from the target venue's official record.

## Required metrics

- ROC-AUC
- Average Precision
- Precision
- Recall
- F1
- False Positive Rate
- LLM calls per trajectory
- median and p95 latency
- per-category metrics

## Required ablations

- cumulative drift vs delta
- Tier 0 vs Tier 0+Tier 1
- Tier 0+Tier 1 vs full cascade
- threshold sensitivity
- local model size/quantization sensitivity
- with/without trajectory pairing

## Reproducibility record

Store the following with every final experiment:

- git commit SHA
- dataset source/version
- split manifest
- Python version
- dependency lock/export
- embedding model
- NLI model
- Ollama version
- exact Ollama model tag
- prompt version
- threshold configuration
- seed
- hardware
- raw metric output

## Publication integrity

The conference manuscript must contain only measured results and properly attributed prior work. A later journal manuscript must substantially extend the conference work and explicitly disclose the earlier publication if required by the target venue. Never submit the same manuscript concurrently to multiple venues.
