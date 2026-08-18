# Journal Extension Plan

The journal manuscript must be substantially more than the conference version and must disclose the conference publication when required by the target journal.

## Additional technical contributions

1. Cross-dataset evaluation on at least one independent agent trajectory benchmark.
2. Formal threshold calibration analysis and confidence intervals.
3. Model-size and quantization sensitivity for local LLM judges.
4. Robustness to trajectory length and benign tool substitutions.
5. Detailed error taxonomy and qualitative case studies.
6. Latency distribution (median/p95/p99) and memory footprint.
7. Selective-risk / coverage analysis showing performance as escalation budget changes.
8. Additional ablations of embedding model and NLI model.
9. Reproducibility appendix with exact environment and experiment manifests.

## Conference-to-journal delta table

| Component | Conference | Journal |
|---|---|---|
| Core cascade | Yes | Expanded/formalized |
| Primary benchmark | Yes | Yes + independent benchmark |
| NLI | Yes | Model comparison |
| Local LLM | Yes | Size/quantization study |
| Ablation | Core | Extensive |
| Error analysis | Brief | Detailed |
| Robustness | Limited | Dedicated section |
| Statistical analysis | Core | Confidence intervals/significance where appropriate |
| Reproducibility | Core | Full appendix/package |

Do not submit the same manuscript concurrently to multiple venues.
