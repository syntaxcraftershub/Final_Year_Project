"""Reproducible evaluation runner for ShadowTrace.

Usage:
  python result/evaluate_cascade.py --data database/processed/tracesafe.jsonl

The script deliberately writes raw per-trajectory records before summary
metrics so every paper table can be traced to evidence. LLM use is disabled by
default for baseline/cost runs; enable it explicitly after Ollama is installed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT))

from pipeline import Thresholds, monitor


def load(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="database/processed/tracesafe.jsonl")
    ap.add_argument("--out", default="result/metrics/cascade_raw.jsonl")
    ap.add_argument("--enable-llm", action="store_true")
    args = ap.parse_args()

    rows = load(ROOT / args.data)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = Thresholds(enable_llm=args.enable_llm)
    y_true, y_pred, scores = [], [], []
    total_llm = 0
    latencies = []

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            start = time.perf_counter()
            result = monitor(row["user_instruction"], row["steps"], cfg)
            latency_ms = (time.perf_counter() - start) * 1000
            pred = result.final_verdict
            y = 1 if row["label"] == "malicious" else 0
            score = max((s.tier0_score for s in result.states), default=0.0)
            y_true.append(y)
            y_pred.append(1 if pred == "flagged" else 0)
            scores.append(score)
            total_llm += result.llm_calls
            latencies.append(latency_ms)
            f.write(json.dumps({
                "task_id": row.get("task_id"),
                "label": row["label"],
                "category": row.get("category"),
                "prediction": pred,
                "score": score,
                "llm_calls": result.llm_calls,
                "latency_ms": latency_ms,
            }) + "\n")

    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = None
    try:
        ap_score = average_precision_score(y_true, scores)
    except ValueError:
        ap_score = None
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    metrics = {
        "n": len(rows),
        "roc_auc": auc,
        "average_precision": ap_score,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fp / max(fp + tn, 1),
        "llm_calls_per_trajectory": total_llm / max(len(rows), 1),
        "median_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "enable_llm": args.enable_llm,
    }
    print(json.dumps(metrics, indent=2))
    (ROOT / "result/metrics/cascade_summary.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
