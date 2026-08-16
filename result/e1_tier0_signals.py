"""E1: ROC-AUC of every Tier 0 feature, overall and per category.

Answers make-or-break question left open by the earlier Phase 3 gate: does
step-to-step delta generalise across the full corpus, or did it only look
good on one worked example?

Runs on the CALIBRATION split only. The test split is reserved for the single
scored run (E3+E4, result/e3_e4_cascade_and_baselines.py).

PERFORMANCE NOTE: Tier0Embedding.judge()/.features() re-embed one record at a
time, which is correct for the live per-trajectory cascade but far too slow
for a full-corpus sweep on CPU (measured ~4.4s/record => ~66 min for 902
calibration records). This script computes the identical features (max_delta,
mean_delta, final_drift, mean_drift, drift_rise) via one large batched
sentence-transformers call across the whole split instead of hundreds of tiny
calls, which is a ~10-20x wall-clock win from batching alone -- not a change
to the feature definitions or the tier's production behaviour.

INTERPRETATION -- fixed BEFORE running, so the result stays honest:
  max_delta AUC >= 0.70  -> delta generalises; Tier 0 is a real filter.
  0.60-0.70              -> weak but usable as a first-stage filter.
  <= 0.60                -> delta did NOT generalise. Legitimate finding:
                             widen Tier 0's abstention band so it escalates
                             most cases, let Tier 1 do the work. Do not
                             retune features after seeing this number.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from drift_tracker import trend_slope  # noqa: E402
from encoder import distance, encode_intent, encode_texts, step_to_sentence  # noqa: E402
from splits import load_records  # noqa: E402

OUT = Path(__file__).resolve().parent / "e1_tier0_signals.csv"
FEATURES = ["max_delta", "mean_delta", "final_drift", "mean_drift", "drift_rise"]


def batched_features(records, batch_size=256):
    """Compute Tier 0 features for every record via large batched embedding
    calls instead of one encode() call per record. Same math as
    drift_tracker.drift_line/delta_line/drift_features and
    tiers.tier0_embedding.Tier0Embedding.features -- only the call pattern
    differs.
    """
    # 1) all intents, one batch
    intents = [r["user_instruction"] for r in records]
    t0 = time.time()
    intent_vecs = encode_texts(intents)
    print(f"  encoded {len(intents)} intents in {time.time() - t0:.1f}s")

    # 2) all step sentences (flat) and all prefix strings (flat), one batch each
    step_sentences_flat = []
    prefix_strings_flat = []
    offsets = []  # (start, n_steps) into the flat arrays, per record
    pos = 0
    for r in records:
        sents = [step_to_sentence(s) for s in r["steps"]]
        running, prefixes = [], []
        for s in sents:
            running.append(s)
            prefixes.append(" ".join(running))
        step_sentences_flat.extend(sents)
        prefix_strings_flat.extend(prefixes)
        offsets.append((pos, len(sents)))
        pos += len(sents)

    t0 = time.time()
    step_vecs_flat = encode_texts(step_sentences_flat, ) if False else None
    # encode in chunks to bound peak memory / show progress
    def _chunked_encode(texts):
        out = []
        for i in range(0, len(texts), batch_size):
            out.append(encode_texts(texts[i:i + batch_size]))
            print(f"    encoded {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
        print()
        return np.concatenate(out, axis=0) if out else np.zeros((0, 768))

    step_vecs_flat = _chunked_encode(step_sentences_flat)
    print(f"  encoded {len(step_sentences_flat)} step sentences in {time.time() - t0:.1f}s")

    t0 = time.time()
    prefix_vecs_flat = _chunked_encode(prefix_strings_flat)
    print(f"  encoded {len(prefix_strings_flat)} prefixes in {time.time() - t0:.1f}s")

    rows = []
    for i, r in enumerate(records):
        start, n = offsets[i]
        step_vecs = step_vecs_flat[start:start + n]
        prefix_vecs = prefix_vecs_flat[start:start + n]
        intent_vec = intent_vecs[i]

        drifts = [distance(intent_vec, v) for v in prefix_vecs]
        if n < 2:
            deltas = [0.0] * n
        else:
            deltas = [0.0] + [distance(step_vecs[j - 1], step_vecs[j]) for j in range(1, n)]

        rows.append({
            "max_delta": max(deltas) if deltas else 0.0,
            "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
            "final_drift": drifts[-1] if drifts else 0.0,
            "mean_drift": sum(drifts) / len(drifts) if drifts else 0.0,
            "drift_rise": (drifts[-1] - drifts[0]) if drifts else 0.0,
            "label": 1 if r["label"] == "malicious" else 0,
            "category": r["category"],
        })
    return rows


def main():
    records = load_records(split="calibration")
    print(f"calibration split: {len(records)} records")
    rows = batched_features(records)

    print(f"\n{'feature':14s} {'ROC-AUC':>9s}  (n={len(rows)})")
    results = []
    for feat in FEATURES:
        y = [r["label"] for r in rows]
        x = [r[feat] for r in rows]
        auc = roc_auc_score(y, x)
        results.append({"scope": "overall", "feature": feat, "auc": auc, "n": len(rows)})
        print(f"{feat:14s} {auc:9.3f}")

    benign = [r for r in rows if r["label"] == 0]
    cats = sorted({r["category"] for r in rows if r["label"] == 1})
    for cat in cats:
        subset = benign + [r for r in rows if r["category"] == cat]
        y = [r["label"] for r in subset]
        if len(set(y)) < 2:
            continue
        print(f"\n{cat}  (n={len(subset)})")
        for feat in FEATURES:
            auc = roc_auc_score(y, [r[feat] for r in subset])
            results.append({"scope": cat, "feature": feat, "auc": auc, "n": len(subset)})
            print(f"  {feat:14s} {auc:9.3f}")

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "feature", "auc", "n"])
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
