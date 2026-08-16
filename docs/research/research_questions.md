# Research Questions and Hypotheses

## Central research question

Can a cheap tiered monitoring cascade (embeddings → NLI → local LLM) detect
trajectory-level agent divergence while substantially reducing expensive LLM
inference compared with always-on LLM judging?

## Hypotheses

- **H1 (central).** A hierarchical cascade combining cheap embedding
  signals, NLI goal-alignment, and selective local-LLM escalation can
  achieve competitive trajectory-level detection while substantially
  reducing average LLM inference relative to an always-on LLM judge.
  Status: tested by E3/E4 (cascade configs vs. per-step-LLM and
  whole-trajectory-LLM baselines on the identical test split).
- **H2.** Step-to-step embedding delta detects trajectory pivots that
  cumulative embedding drift misses. Status: **partially supported, with a
  major caveat** — E1 shows overall `max_delta` ROC-AUC 0.542 (did not clear
  the pre-registered 0.60 bar), but `PRIVACY_LEAKAGE` alone shows
  `max_delta` = 0.724 (clears the 0.70 bar). Delta is not a general-purpose
  signal; it is a real signal for one specific category.
- **H3.** NLI-based goal alignment improves detection of semantically subtle
  divergence that embedding similarity alone misses. Status: **not
  supported** — E2 shows overall NLI ROC-AUC 0.516, and 0.472 (worse than
  chance) specifically on `PRIVACY_LEAKAGE`, the one category where Tier 0
  actually works. This is a genuine negative result for the design's
  stated repair hypothesis.
- **H4.** Selective escalation preserves detection performance while
  reducing inference cost and latency relative to an always-on LLM judge.
  Status: tested by E3/E4's cost/latency/LLM-calls-per-trajectory columns.
- **H5 (added after competitor analysis, not in the original design).** The
  cascade's frozen-on-TraceSafe thresholds generalize, with measurably
  (not silently) degraded performance, to an external benchmark (ATBench)
  evaluated with zero retuning. Status: tested by E6.

## What would falsify each hypothesis

- H1 is falsified if the full cascade's detection rate / FPR is not
  competitive with the per-step-LLM baseline while using substantially
  fewer LLM calls, i.e. if the cost savings come with a detection collapse
  rather than a graceful trade-off.
- H2 is falsified (further) by low AUC across all Tier 0 features in all
  categories — already partially the case; the surviving claim is scoped to
  `PRIVACY_LEAKAGE` specifically, not trajectories in general.
- H3 is already falsified as originally stated (repairs the general
  topic-vs-goal problem); the residual, narrower claim that NLI helps on
  `PROMPT_INJECTION` specifically (E2: 0.605) is weaker and not strongly
  supported either.
- H4 is falsified if the cascade's cost savings are marginal (e.g. still
  escalates to Tier 2 on most trajectories) given how weak Tier 0/Tier 1
  turned out to be per E1/E2 — a real risk flagged going into E3/E4.
- H5 is falsified by a large, unexplained performance gap between
  TraceSafe-test and ATBench that cannot be attributed to genuinely
  different task distributions (e.g. near-chance ATBench detection while
  TraceSafe-test detection is strong).
