"""Parse TraceSafe golden JSONL files into the common trajectory format.

Common format (one JSON object per line in data/processed/tracesafe.jsonl):
{
  "task_id": str,
  "user_instruction": str,
  "steps": [ {"action": str, "params": dict, "result": str}, ... ],
  "label": "benign" | "malicious",
  "category": str,          # original TraceSafe mutation_category
  "source": "tracesafe"
}

TraceSafe raw format: each line has new_trace.trace = list of messages.
  role=user  -> content is the instruction (may be multi-turn)
  role=agent -> content is a tool call: {"name": ..., "arguments": {...}}
                (sometimes a plain-text final answer)
  role=tool  -> content is the JSON string result of the previous call
"""

import ast
import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "database" / "raw" / "tracesafe"
OUT_PATH = Path(__file__).resolve().parent.parent / "database" / "processed" / "tracesafe.jsonl"


def _as_tool_call(content):
    """Return {"name", "arguments"} if content encodes a tool call, else None."""
    if isinstance(content, dict):
        return content if "name" in content else None
    if isinstance(content, str):
        text = content.strip()
        if text.startswith("{"):
            for loader in (json.loads, ast.literal_eval):
                try:
                    obj = loader(text)
                    if isinstance(obj, dict) and "name" in obj:
                        return obj
                except (ValueError, SyntaxError):
                    continue
    return None


def parse_record(rec, task_id):
    trace = rec.get("new_trace", {}).get("trace", [])
    user_parts, steps = [], []

    for i, msg in enumerate(trace):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            user_parts.append(str(content))
        elif role == "agent":
            call = _as_tool_call(content)
            result = ""
            if i + 1 < len(trace) and trace[i + 1].get("role") == "tool":
                result = str(trace[i + 1].get("content", ""))
            if call:
                args = call.get("arguments") or {}
                if not isinstance(args, dict):
                    args = {"value": args}
                steps.append({"action": str(call["name"]), "params": args, "result": result})
            else:
                # Plain-text agent message (e.g. final answer) — keep as a step
                # so drift over natural-language actions is also tracked.
                steps.append({"action": "respond", "params": {"text": str(content)}, "result": result})

    category = rec.get("mutation_category", "Unknown")
    return {
        "task_id": task_id,
        "user_instruction": "\n".join(user_parts),
        "steps": steps,
        "label": "benign" if category.upper() == "BENIGN" else "malicious",
        "category": category,
        "source": "tracesafe",
    }


def parse_all():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    n_written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for raw_file in sorted(RAW_DIR.glob("golden_*.jsonl")):
            file_tag = raw_file.stem.replace("golden_", "")
            for line_no, line in enumerate(open(raw_file, encoding="utf-8")):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                parsed = parse_record(rec, task_id=f"{file_tag}_{line_no}")
                if not parsed["user_instruction"] or not parsed["steps"]:
                    counts["skipped_empty"] = counts.get("skipped_empty", 0) + 1
                    continue
                out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                counts[parsed["label"]] = counts.get(parsed["label"], 0) + 1
                n_written += 1
    return n_written, counts


if __name__ == "__main__":
    n, counts = parse_all()
    print(f"Wrote {n} trajectories to {OUT_PATH}")
    print("Counts:", counts)
