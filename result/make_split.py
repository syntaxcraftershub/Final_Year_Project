"""Create deterministic train/calibration/test splits without row shuffling leakage.

If a record contains `pair_id`, all records in a pair stay together. Otherwise
`task_id` is used as the grouping key. This is intentionally conservative: a
future dataset adapter should populate pair_id from the benchmark's true
mutation/original relationship rather than guessing it from labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def bucket(group: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="database/processed/tracesafe.jsonl")
    ap.add_argument("--out", default="database/processed/splits")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads(x) for x in (root / args.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    groups = {}
    for row in rows:
        group = str(row.get("pair_id") or row.get("task_id"))
        groups.setdefault(group, []).append(row)
    buckets = {}
    for group in groups:
        r = bucket(group, args.seed)
        buckets[group] = "train" if r < 0.70 else "calibration" if r < 0.85 else "test"
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "calibration", "test"):
        selected = [row for group, rows2 in groups.items() if buckets[group] == split for row in rows2]
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8")
        counts[split] = len(selected)
    (out / "manifest.json").write_text(json.dumps({"seed": args.seed, "grouped": True, "counts": counts}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
