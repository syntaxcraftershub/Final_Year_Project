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


def parse_record(rec, task_id, trace_key="new_trace"):
    """Parse one raw TraceSafe record into the common format.

    trace_key selects which trace to read:
      "new_trace"      -> the (possibly mutated) trajectory
      "original_trace" -> its unmutated benign twin

    A twin keeps the same pair_id as its mutated counterpart, so matched pairs
    can be evaluated together and kept on the same side of a data split.
    """
    trace = rec.get(trace_key, {}).get("trace", [])
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
    is_twin = trace_key == "original_trace"
    # A twin is benign by construction; otherwise the category decides.
    label = "benign" if (is_twin or category.upper() == "BENIGN") else "malicious"

    return {
        "task_id": f"{task_id}__twin" if is_twin else task_id,
        "pair_id": task_id,
        "user_instruction": "\n".join(user_parts),
        "steps": steps,
        "label": label,
        "category": "BENIGN_TWIN" if is_twin else category,
        "source": "tracesafe",
    }


def parse_all():
    """Parse every raw TraceSafe file into the processed corpus.

    Every malicious record also emits its unmutated benign twin (parsed from
    original_trace), which roughly balances the corpus and enables the
    matched-pair evaluation design (design spec section 3.2-3.3). Pure-benign
    records (golden_0_benign.jsonl, mutation_category == "BENIGN") have no
    separate original_trace worth emitting as a second record: new_trace ==
    original_trace for those, so a twin would be an exact duplicate rather
    than a control -- skipped deliberately, not silently.

    Every skip is counted and reported, per the "log exclusions and reasons"
    requirement -- nothing is dropped silently.
    """
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    n_written = 0
    pair_ids = set()

    def _write(out, parsed):
        nonlocal n_written
        if not parsed["user_instruction"] or not parsed["steps"]:
            counts["skipped_empty"] = counts.get("skipped_empty", 0) + 1
            return
        out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
        counts[parsed["label"]] = counts.get(parsed["label"], 0) + 1
        counts[f"category:{parsed['category']}"] = counts.get(f"category:{parsed['category']}", 0) + 1
        pair_ids.add(parsed["pair_id"])
        n_written += 1

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for raw_file in sorted(RAW_DIR.glob("golden_*.jsonl")):
            file_tag = raw_file.stem.replace("golden_", "")
            for line_no, line in enumerate(open(raw_file, encoding="utf-8")):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                task_id = f"{file_tag}_{line_no}"
                _write(out, parse_record(rec, task_id, "new_trace"))
                # Malicious records carry an unmutated twin; benign files do not
                # (their new_trace already equals original_trace -- a "twin"
                # would just be a byte-identical duplicate, not a control, so
                # it is skipped and counted rather than fabricated).
                category = rec.get("mutation_category", "").upper()
                if category != "BENIGN":
                    _write(out, parse_record(rec, task_id, "original_trace"))
                else:
                    counts["skipped_benign_twin_duplicate"] = counts.get("skipped_benign_twin_duplicate", 0) + 1

    counts["n_pairs"] = len(pair_ids)
    return n_written, counts


if __name__ == "__main__":
    n, counts = parse_all()
    print(f"Wrote {n} trajectories to {OUT_PATH}")
    print("Counts:", json.dumps(counts, indent=2))
