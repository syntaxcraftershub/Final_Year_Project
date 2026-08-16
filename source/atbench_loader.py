"""Loader for ATBench (external validation set, arXiv:2604.02022 /
huggingface.co/datasets/AI45Research/ATBench), used ONLY for the zero-retuning
external-generalization check (result/e6_atbench_external.py).

ATBench is not TraceSafe: it is a separate benchmark with its own trajectory
schema. This loader parses it into the SAME common record shape data_parser.py
produces ({task_id, user_instruction, steps, label, category, source}) so it
can be run through the exact same tiers/cascade code as TraceSafe, with
thresholds frozen from TraceSafe calibration and NOT retuned here -- that is
the entire point of the experiment (does the cascade generalize beyond the
benchmark it was calibrated on?).

ATBench record shape (one JSON object per list entry in ATBench/test.json):
{
  "id": int,
  "tool_used": [...],
  "contents": [[ {role: user|agent|environment, content?, thought?, action?}, ... ]],
  "label": 0 | 1,          # 0 = safe, 1 = unsafe
  "risk_source": str,
  "failure_mode": str,
  "reason": str,
  "real_world_harm": str,
}
"""

import ast
import json
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "database" / "raw" / "atbench" / "ATBench" / "test.json"


def _parse_action(action_raw):
    """ATBench's agent 'action' field is a JSON string: {"name": ..., "arguments": {...}}."""
    if not action_raw:
        return None
    if isinstance(action_raw, dict):
        return action_raw
    text = str(action_raw).strip()
    if not text:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(text)
            if isinstance(obj, dict) and "name" in obj:
                return obj
        except (ValueError, SyntaxError):
            continue
    return None


def parse_record(rec):
    """Parse one raw ATBench record into the common trajectory format."""
    conversation = rec["contents"][0] if rec.get("contents") else []
    user_parts, steps = [], []

    for i, msg in enumerate(conversation):
        role = msg.get("role")
        if role == "user":
            user_parts.append(str(msg.get("content", "")))
        elif role == "agent":
            call = _parse_action(msg.get("action"))
            result = ""
            if i + 1 < len(conversation) and conversation[i + 1].get("role") == "environment":
                result = str(conversation[i + 1].get("content", ""))
            if call:
                args = call.get("arguments") or {}
                if not isinstance(args, dict):
                    args = {"value": args}
                steps.append({"action": str(call["name"]), "params": args, "result": result})
            else:
                text = str(msg.get("content") or msg.get("thought") or "")
                steps.append({"action": "respond", "params": {"text": text}, "result": result})
        # role == "environment" is consumed as the preceding agent step's result.

    return {
        "task_id": f"atbench_{rec['id']}",
        "pair_id": f"atbench_{rec['id']}",  # ATBench has no matched-twin structure
        "user_instruction": "\n".join(user_parts),
        "steps": steps,
        "label": "malicious" if rec.get("label") == 1 else "benign",
        "category": rec.get("risk_source", "unknown") if rec.get("label") == 1 else "SAFE",
        "source": "atbench",
    }


def load_all():
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    records, skipped = [], 0
    for rec in raw:
        parsed = parse_record(rec)
        if not parsed["user_instruction"] or not parsed["steps"]:
            skipped += 1
            continue
        records.append(parsed)
    return records, skipped


if __name__ == "__main__":
    records, skipped = load_all()
    labels = {"benign": 0, "malicious": 0}
    for r in records:
        labels[r["label"]] += 1
    print(f"loaded {len(records)} ATBench trajectories, skipped {skipped}")
    print("labels:", labels)
