# ShadowTrace — Conference Paper Draft

## Title

**ShadowTrace: A Cost-Aware Cascade Framework for Detecting Multi-Step Intent Divergence in AI Agent Trajectories**

## Abstract

Autonomous AI agents can execute multi-step workflows in which unsafe behavior is distributed across individually plausible actions. Monitoring every action with a large language model can provide rich semantic judgment but introduces substantial inference cost and latency. This work proposes ShadowTrace, a three-tier trajectory-monitoring cascade that combines inexpensive embedding signals, local natural-language-inference scoring, and selective local large-language-model adjudication. The first tier computes cumulative semantic drift and consecutive-step delta; ambiguous cases are passed to an NLI model that estimates task-action contradiction; residual ambiguous trajectories are escalated to a locally hosted Qwen model through Ollama. The evaluation protocol uses leakage-aware grouped splits, matched trajectory pairs where available, category-level reporting, and explicit compute-efficiency metrics. The study is designed to quantify the trade-off among detection quality, false-positive rate, latency, and LLM invocation rate rather than optimize a single accuracy number. All final numerical claims are generated from version-controlled experiment scripts and are reported without post-hoc test-set tuning.

**Keywords:** AI agents, agent safety, trajectory monitoring, intent divergence, natural language inference, local LLM, Ollama, selective inference, cybersecurity.

## 1. Introduction

AI agents increasingly combine language reasoning with external tools and multi-step execution. A safety monitor that evaluates only isolated actions can miss a gradual change in intent. Conversely, sending every action to a large language model is computationally expensive. ShadowTrace studies whether a selective cascade can reserve expensive reasoning for cases that cheaper signals cannot resolve.

## 2. Research Gap

Existing trajectory-monitoring approaches motivate sequence-level inspection but leave an efficiency question: how much detection quality can be retained when expensive LLM judgment is selectively invoked? A second challenge is that semantic similarity can capture shared topic rather than alignment with the user's actual goal. ShadowTrace therefore treats cumulative embedding drift as an empirical baseline, introduces step-to-step delta and NLI goal/action contradiction as intermediate signals, and evaluates a local LLM as a final adjudicator rather than as the default detector.

## 3. Proposed Method

Let the task be `T` and trajectory `A = {a1, ..., an}`. For each action, the system generates a natural-language representation. Tier 0 computes normalized embedding features:

`D_i = 1 - cos(v_T, v_{1:i})`

`Delta_i = 1 - cos(v_{i-1}, v_i)`

Clearly low or high scores are classified immediately. Ambiguous cases enter Tier 1, which estimates `P(contradiction | T, a_i)` using a local NLI cross-encoder. If the NLI score remains ambiguous, Tier 2 sends the trajectory prefix to a local Qwen model served by Ollama. The local path requires no external API key.

## 4. Experimental Design

The study uses a deterministic grouped train/calibration/test protocol. When an authoritative mutation/original pair identifier is available, all members of the pair remain in one split. Otherwise the record identifier is used as the conservative grouping key and the limitation is reported. Thresholds are tuned only on the calibration split and frozen before final testing.

### Baselines

1. Majority-class baseline.
2. Whole-trajectory cumulative embedding distance.
3. Step-to-step delta.
4. Tier-0-only detector.
5. Tier-1-only detector.
6. Full LLM per action.
7. Full LLM per trajectory.
8. ShadowTrace selective cascade.

### Metrics

ROC-AUC, Average Precision, precision, recall, F1, false-positive rate, LLM calls per trajectory, median latency, p95 latency, and per-category performance.

## 5. Results

**This section must remain empty until experiments are executed. No numbers are to be inserted manually.** The experiment runner writes per-trajectory records to `result/metrics/cascade_raw.jsonl` and summary metrics to `result/metrics/cascade_summary.json`.

## 6. Ablation and Error Analysis

The paper will compare cumulative drift with consecutive-step delta, remove NLI, remove the local LLM, vary thresholds, and examine model-size/quantization effects. False positives and false negatives will be grouped by trajectory category and failure mode.

## 7. Limitations

The benchmark may mix genuine intent divergence with operational/tool-use failures. A local 4B quantized model is chosen for reproducibility on modest hardware and should not be presented as universally optimal. Thresholds are task- and benchmark-dependent until validated on additional datasets. External validity requires evaluation on additional agent environments.

## 8. Reproducibility

The repository contains the pipeline, local model configuration, split generator, evaluation runner, and research protocol. Final experiments will record the commit SHA, dataset version, model tags, prompt, thresholds, seed, hardware, and runtime.

## 9. Conclusion

ShadowTrace frames agent safety monitoring as a selective inference problem: use cheap signals whenever possible and reserve expensive semantic adjudication for uncertainty. The final contribution will be supported only by measured accuracy-efficiency trade-offs and controlled ablations.

## References

Populate this section from verified primary sources only. Do not invent bibliographic entries. Every reused algorithm, dataset, model, benchmark, and claim must have a traceable citation.
