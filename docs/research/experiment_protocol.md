# Experiment Protocol

Order matters. Each step is run exactly once, in this order, and the test
split is touched exactly once (E3+E4).

1. **Data foundation** — `python source/data_parser.py` (requires
   `database/raw/tracesafe/golden_*.jsonl`, downloaded from
   `github.com/yenshan0530/TraceSafe`). Produces
   `database/processed/tracesafe.jsonl`.
2. **Splitting** — `python source/splits.py`. Deterministic, seed 42.
   Produces `database/processed/splits.json` (summary counts; the actual
   assignment is recomputed deterministically by `splits.load_records`,
   not stored per-record).
3. **E1** — `python result/e1_tier0_signals.py`. Calibration split only.
   Interpretation thresholds fixed in the script's docstring before running.
4. **E2** — `python result/e2_nli_signal.py`. Calibration split only.
5. **Calibration** — `python result/calibrate.py`. Calibration split only.
   Freezes `database/processed/thresholds.json`.
6. **E3+E4** — `python result/e3_e4_cascade_and_baselines.py`. **The single
   scored run on the test split.** Produces `result/e3_e4_results.csv` and
   `result/e3_e4_per_trajectory.json`.
7. **E5** — `python result/e5_published_context.py`. No data dependency;
   assembles the secondary-context table from manually verified published
   numbers.
8. **E6** (added, not in original plan) —
   `python result/e6_atbench_external.py`. Requires
   `database/raw/atbench/ATBench/test.json` (downloaded from
   `huggingface.co/datasets/AI45Research/ATBench`). Uses thresholds frozen
   in step 5, zero retuning.
9. **Results table** — `python result/build_results_table.py`.
10. **Figures** — `python result/make_figures.py` (or equivalent per-figure
    scripts).

`result/run_all.py` runs steps 1-9 in order for one-command regeneration
(E6 is external-data-dependent and kept as a separate explicit step since it
needs a second downloaded dataset not everyone reproducing this repo will
have fetched).

## Pre-registered interpretation (written before running, not after)

- E1 `max_delta` overall AUC ≥ 0.70 → delta generalises broadly. 0.60-0.70 →
  weak but usable filter. ≤ 0.60 → did not generalise as a blanket signal;
  widen Tier 0's band so it mostly escalates.
- Actual measured result: **overall 0.542** (did not clear the bar), but
  **PRIVACY_LEAKAGE alone: max_delta 0.724, mean_delta 0.754** (clears the
  0.70 bar for that category specifically). This category-level finding
  was not anticipated in the pre-registered overall-only threshold, and is
  reported as an additional, honest finding rather than retrofit into the
  original binary pass/fail framing.
- No threshold was changed after seeing either result. Tier 0/Tier 1 bands
  were calibrated exactly as specified in the design (tolerance-quantile
  rule), using whatever E1/E2 actually measured.
