"""Phase 3 gate test.

1. Plot drift lines: 3 known-malicious (attack spread across steps, from
   drift-type categories) vs 3 benign, both >= 6 steps. Visual check:
   malicious should trend up, benign flat/noisy.
2. Aggregate check on a larger sample: slope/rise/final stats and ROC-AUC
   per feature, benign vs drift-type malicious. Numbers reported as-is.
"""

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

from drift_tracker import drift_features, drift_line  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "database" / "processed" / "tracesafe.jsonl"
PLOT_DIR = Path(__file__).resolve().parent / "plots"

DRIFT_CATEGORIES = {"PROMPT_INJECTION", "PRIVACY_LEAKAGE"}


def load():
    recs = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    benign = [r for r in recs if r["label"] == "benign"]
    drift_mal = [r for r in recs if r["category"] in DRIFT_CATEGORIES]
    return benign, drift_mal


def plot_gate(benign, drift_mal):
    random.seed(11)
    b3 = random.sample([r for r in benign if len(r["steps"]) >= 6], 3)
    m3 = random.sample([r for r in drift_mal if len(r["steps"]) >= 6], 3)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in b3:
        line = drift_line(r["steps"], r["user_instruction"])
        ax.plot(range(1, len(line) + 1), line, "o-", color="tab:blue", alpha=0.7,
                label=f"benign {r['task_id']}")
    for r in m3:
        line = drift_line(r["steps"], r["user_instruction"])
        ax.plot(range(1, len(line) + 1), line, "s--", color="tab:red", alpha=0.7,
                label=f"malicious {r['task_id']} ({r['category']})")
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative drift (cosine distance to intent)")
    ax.set_title("Phase 3 gate: cumulative drift lines")
    ax.legend(fontsize=7)
    PLOT_DIR.mkdir(exist_ok=True)
    out = PLOT_DIR / "phase3_drift_lines.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


def aggregate(benign, drift_mal, n_per_group=60):
    random.seed(13)
    b = random.sample([r for r in benign if len(r["steps"]) >= 3], min(n_per_group, len(benign)))
    m = random.sample([r for r in drift_mal if len(r["steps"]) >= 3], n_per_group)

    feats, labels = [], []
    for group, tag in ((b, 0), (m, 1)):
        for r in group:
            feats.append(drift_features(drift_line(r["steps"], r["user_instruction"])))
            labels.append(tag)

    keys = ["final", "mean", "max", "slope_all", "slope_last3", "rise"]
    print(f"\naggregate over {len(b)} benign vs {len(m)} drift-malicious (>=3 steps):")
    print(f"{'feature':12s} {'benign_mean':>12s} {'mal_mean':>12s} {'ROC-AUC':>8s}")
    for k in keys:
        vals = [f[k] for f in feats]
        bvals = [v for v, l in zip(vals, labels) if l == 0]
        mvals = [v for v, l in zip(vals, labels) if l == 1]
        auc = roc_auc_score(labels, vals)
        print(f"{k:12s} {sum(bvals)/len(bvals):12.4f} {sum(mvals)/len(mvals):12.4f} {auc:8.3f}")


if __name__ == "__main__":
    benign, drift_mal = load()
    print(f"benign={len(benign)}  drift-type malicious={len(drift_mal)}")
    plot_gate(benign, drift_mal)
    aggregate(benign, drift_mal)
