# Reproducibility

## Environment

- Python 3.11, CPU-only (`torch.cuda.is_available() == False`, confirmed at
  environment setup time).
- 2 CPU cores in the environment this project was built and measured in —
  every latency/cost number in `result/` is specific to that hardware.
- `env/requirements.txt`: `sentence-transformers`, `scikit-learn`,
  `fastapi`, `streamlit`, `torch`, `transformers`, `protobuf`,
  `sentencepiece`, `accelerate`, plus standard data/plotting libraries.

## Models (pinned, recorded in `database/processed/thresholds.json`)

- Encoder (Tier 0): `all-mpnet-base-v2` (sentence-transformers).
- NLI (Tier 1): `cross-encoder/nli-deberta-v3-base`, `max_seq_length=128`.
- LLM judge (Tier 2): `Qwen/Qwen2.5-1.5B-Instruct`, greedy decoding
  (`do_sample=False`), `max_new_tokens=5`.

All three download from Hugging Face Hub on first use;
`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` (set by default in
`source/encoder.py`) makes every subsequent run fully offline.

## Datasets

- TraceSafe-Bench (arXiv:2604.07223): `github.com/yenshan0530/TraceSafe`
  (git-clone-able; the GitHub REST API was not reachable from this
  environment, but `git clone` over HTTPS worked). Not committed to this
  repo (`.gitignore` — contains synthetic-but-format-valid fake credentials
  that trip GitHub push protection).
- ATBench (arXiv:2604.02022): `huggingface.co/datasets/AI45Research/ATBench`,
  file `ATBench/test.json`. Also not committed (external data, download on
  demand via `huggingface_hub.hf_hub_download`).

## Determinism

- Splitting: seed 42, fixed in `source/splits.py`.
- Tier 2 decoding: greedy (`do_sample=False`), deterministic given the same
  model weights and prompt.
- Calibration: deterministic quantile computation over the calibration
  split's scores, no randomness.
- Not fully deterministic across different hardware/BLAS backends: floating
  point embedding computation can differ in the last few decimal places
  across CPU architectures, which could in principle flip a handful of
  borderline verdicts near a threshold boundary. Not expected to change any
  headline finding's direction.

## One-command regeneration

`python result/run_all.py` — see `docs/research/experiment_protocol.md` for
the exact step order and what each step produces. E6 (ATBench) is a
separate explicit step since it depends on a second external dataset.

## What is NOT reproducible bit-for-bit

- Wall-clock latency/cost numbers (hardware-dependent).
- Hugging Face Hub download order/timing (network-dependent, does not
  affect the computed results, only how long setup takes).
