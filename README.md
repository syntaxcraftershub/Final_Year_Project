# ShadowTrace

**Cost-Aware Cascade Framework for Detecting Multi-Step Intent Divergence in AI Agent Trajectories**

Final-year B.Tech AI & Data Science research project. The repository is organized for an IEEE-quality experimental workflow: reproducible code, explicit hypotheses, leakage-aware evaluation, local inference, and no fabricated results.

## Research objective

Given a user task `T` and an agent trajectory `A = {a1, ..., an}`, estimate whether the trajectory remains aligned with the user's intent while minimizing expensive LLM inference.

The monitor is a selective cascade:

```text
User task + agent trajectory
          |
          v
   Tier 0 — cheap signals
   cumulative drift + step delta
          |
       clear? -------- yes ------> verdict
          |
          no
          v
   Tier 1 — local NLI
   task/action contradiction
          |
       clear? -------- yes ------> verdict
          |
          no
          v
   Tier 2 — local LLM
   Ollama + Qwen3 4B Q4_K_M
          |
          v
       verdict
```

### Mathematical signals

For normalized embeddings:

`D_i = 1 - cos(v_task, v_prefix_i)`

`Delta_i = 1 - cos(v_step_(i-1), v_step_i)`

Tier 1 estimates:

`P(contradiction | task, action)`

The research objective is an accuracy/efficiency trade-off, not maximum raw accuracy at any cost.

## Local-first deployment

The primary Tier-2 path uses **Ollama locally**. No OpenAI, Anthropic, Gemini, or NVIDIA API key is required for the research path.

Default configuration:

```text
Embedding: all-mpnet-base-v2
NLI:       cross-encoder/nli-MiniLM2-L6-H768
LLM:       qwen3:4b-q4_K_M
Runtime:   Ollama
Endpoint:  http://localhost:11434/api/chat
```

The exact model tag, thresholds, seed, and evaluation protocol are versioned under `research/`.

## Repository layout

```text
source/       core monitor, encoder, NLI gate
connectors/   local Ollama Tier-2 connector
backend/      FastAPI live API
frontend/     vanilla JS live dashboard
demo/         Streamlit dashboard
database/     raw + processed trajectories
result/       reproducible experiment runners and metrics
tests/        automated tests
research/     frozen configuration and experiment protocol
docs/research conference paper + publication protocol
env/          Python dependencies
```

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -r env\requirements.txt
```

Prepare the processed TraceSafe data if raw files are present:

```powershell
.venv\Scripts\python source\data_parser.py
```

Create deterministic grouped splits:

```powershell
.venv\Scripts\python result\make_split.py
```

Run tests:

```powershell
.venv\Scripts\pytest -q
```

Run a baseline evaluation without Ollama:

```powershell
.venv\Scripts\python result\evaluate_cascade.py
```

For Tier-2, install Ollama separately and pull the exact configured model tag:

```powershell
ollama pull qwen3:4b-q4_K_M
```

Then run:

```powershell
.venv\Scripts\python result\evaluate_cascade.py --enable-llm
```

## Research integrity rules

- Never insert a metric manually into the paper.
- Never tune thresholds on the final test split.
- Keep authoritative mutation/original pairs in the same split.
- Report negative results, including failed cumulative-drift experiments.
- Report per-category results as well as aggregate results.
- Record model versions, quantization, prompt, seed, hardware, latency, and git SHA.
- Treat API-backed LLMs as optional baselines, not as the core dependency.
- Do not claim IEEE/Scopus acceptance or indexing without verifying the target venue.
- A journal extension must substantially add technical content to any conference version.

See `docs/research/IEEE_RESEARCH_PROTOCOL.md` and `docs/research/CONFERENCE_PAPER_DRAFT.md`.

## Current research status

The earlier cumulative embedding-drift experiment was intentionally retained as a negative result. The next evidence-producing step is full evaluation of consecutive-step delta, local NLI, and the selective local-LLM cascade under the frozen protocol.

**No final performance number is claimed until the reproducible experiment runner produces it.**
