"""E2: ROC-AUC of the Tier 1 NLI entailment signal, overall and per category.

Tests the hypothesis that measuring goal alignment (entailment) rather than
topic similarity (embedding distance) repairs the Phase 3 failure. See
source/tiers/tier1_nli.py for the entailment-vs-contradiction renormalization
fix applied before this experiment was run (found via pre-calibration
diagnostics on toy examples, not tuned against calibration/test labels).

Calibration split only.

PERFORMANCE NOTE (same issue as E1, see e1_tier0_signals.py): calling
Tier1NLI.judge() once per record reloads no model (it's cached) but still
pays CrossEncoder.predict()'s per-call overhead ~900 times. This script
batches every (task, step-sentence) pair across the whole split into large
CrossEncoder.predict() calls instead -- identical math
(entailment-vs-contradiction renormalization, worst-step-wins, then
1-P(entail)), just computed via fewer, larger calls.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from encoder import step_to_sentence  # noqa: E402
from splits import load_records  # noqa: E402
from tiers.tier1_nli import _entail_given_not_neutral, _get_model  # noqa: E402

OUT = Path(__file__).resolve().parent / "e2_nli_signal.csv"


def batched_nli_scores(records, batch_size=64):
    model = _get_model()

    pairs_flat = []
    offsets = []  # (start, n_steps)
    pos = 0
    for r in records:
        steps = r["steps"]
        n = len(steps)
        for s in steps:
            pairs_flat.append((r["user_instruction"], step_to_sentence(s)))
        offsets.append((pos, n))
        pos += n

    t0 = time.time()
    logits_chunks = []
    for i in range(0, len(pairs_flat), batch_size):
        chunk = pairs_flat[i:i + batch_size]
        logits_chunks.append(np.asarray(model.predict(chunk, apply_softmax=False)))
        if (i // batch_size) % 20 == 0:
            elapsed = time.time() - t0
            done = min(i + batch_size, len(pairs_flat))
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(pairs_flat) - done) / rate / 60 if rate > 0 else float("nan")
            print(f"  NLI pairs {done}/{len(pairs_flat)}  ({rate:.1f}/s, eta {eta:.1f} min)")
    logits = np.concatenate(logits_chunks, axis=0) if logits_chunks else np.zeros((0, 3))
    entail = _entail_given_not_neutral(logits)
    print(f"  scored {len(pairs_flat)} step-pairs in {time.time() - t0:.1f}s")

    scores = []
    for start, n in offsets:
        if n == 0:
            scores.append(0.0)
            continue
        worst_entail = float(np.min(entail[start:start + n]))
        scores.append(min(1.0, max(0.0, 1.0 - worst_entail)))
    return scores


def main():
    records = load_records(split="calibration")
    print(f"calibration split: {len(records)} records")
    scores = batched_nli_scores(records)

    rows = [{
        "score": scores[i],
        "label": 1 if r["label"] == "malicious" else 0,
        "category": r["category"],
    } for i, r in enumerate(records)]

    y = [r["label"] for r in rows]
    x = [r["score"] for r in rows]
    overall = roc_auc_score(y, x)
    print(f"\nNLI overall ROC-AUC: {overall:.3f}")

    results = [{"scope": "overall", "auc": overall, "n": len(rows)}]

    benign = [r for r in rows if r["label"] == 0]
    for cat in sorted({r["category"] for r in rows if r["label"] == 1}):
        subset = benign + [r for r in rows if r["category"] == cat]
        ys = [r["label"] for r in subset]
        if len(set(ys)) < 2:
            continue
        auc = roc_auc_score(ys, [r["score"] for r in subset])
        results.append({"scope": cat, "auc": auc, "n": len(subset)})
        print(f"  {cat:28s} {auc:.3f}")

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "auc", "n"])
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
