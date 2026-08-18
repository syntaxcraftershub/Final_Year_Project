"""FastAPI backend for the live ShadowTrace trajectory monitor."""

import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "connectors"))

from pipeline import Thresholds, monitor  # noqa: E402

DATA = ROOT / "database" / "processed" / "tracesafe.jsonl"
FRONTEND = ROOT / "frontend"

CATEGORY_LABELS = {
    "BENIGN": "Benign",
    "PROMPT_INJECTION": "Malicious · prompt injection",
    "PRIVACY_LEAKAGE": "Malicious · privacy leak",
    "HALLUCINATION": "Malicious · hallucination",
    "INTERFACE_INCONSISTENCIES": "Malicious · interface bug",
}

app = FastAPI(title="ShadowTrace")
_records: dict[str, dict] = {}


def _load():
    if _records or not DATA.exists():
        return
    for line in open(DATA, encoding="utf-8"):
        rec = json.loads(line)
        _records[rec["task_id"]] = rec


@app.on_event("startup")
def startup():
    _load()
    try:
        from encoder import encode_intent
        encode_intent("warmup")
    except Exception as exc:
        print(f"[warn] embedding warmup failed: {exc}")


@app.get("/api/categories")
def categories():
    counts: dict[str, int] = {}
    for record in _records.values():
        counts[record["category"]] = counts.get(record["category"], 0) + 1
    return [{"id": c, "label": CATEGORY_LABELS.get(c, c), "count": n} for c, n in sorted(counts.items())]


@app.get("/api/trajectories")
def trajectories(category: str | None = Query(None), min_steps: int = 3, limit: int = 200):
    out = []
    for record in _records.values():
        if len(record["steps"]) < min_steps or (category and record["category"] != category):
            continue
        out.append({
            "task_id": record["task_id"], "label": record["label"],
            "category": record["category"], "n_steps": len(record["steps"]),
            "instruction_preview": record["user_instruction"][:120],
        })
    out.sort(key=lambda x: x["task_id"])
    return out[:limit]


@app.get("/api/trajectory/{task_id}")
def trajectory(task_id: str):
    record = _records.get(task_id)
    if not record:
        raise HTTPException(404, "unknown task_id")
    return {
        "task_id": record["task_id"], "label": record["label"],
        "category": record["category"], "user_instruction": record["user_instruction"],
        "steps": record["steps"],
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _run_stream(intent, steps, cfg, delay, meta):
    yield _sse("meta", meta)
    result = await run_in_threadpool(monitor, intent, steps, cfg)
    flagged_any = False
    peak_delta = 0.0
    for state in result.states:
        flagged_any = flagged_any or state.verdict == "flagged"
        peak_delta = max(peak_delta, state.delta)
        yield _sse("step", {
            "index": state.index, "step_no": state.index + 1,
            "action": state.action, "sentence": state.sentence,
            "result": state.result, "params_preview": state.sentence[:400],
            "cum_drift": round(state.cum_drift, 4), "delta": round(state.delta, 4),
            "tier0_score": round(state.tier0_score, 4),
            "nli_contradiction": None if state.nli_contradiction is None else round(state.nli_contradiction, 4),
            "tier": state.tier, "verdict": state.verdict, "reason": state.reason,
            "llm_called": state.llm_called, "flagged_so_far": flagged_any,
        })
        await asyncio.sleep(max(0.0, delay))
    final = "flagged" if flagged_any else result.final_verdict
    gt = meta.get("label")
    correct = None if gt not in ("benign", "malicious") else ((final == "flagged") == (gt == "malicious"))
    yield _sse("summary", {
        "final_verdict": final, "ground_truth": gt, "correct": correct,
        "n_steps": len(result.states), "llm_gate_calls": result.llm_calls,
        "peak_delta": round(peak_delta, 4),
    })


def _legacy_cfg(drift_high: float, delta_spike: float, gray_band: float) -> Thresholds:
    # Preserve the existing dashboard's query parameters while mapping them to
    # the new cascade thresholds. Delta is part of the same Tier-0 score.
    high = max(float(drift_high), float(delta_spike))
    low = max(0.0, high - float(gray_band))
    return Thresholds(tier0_low=low, tier0_high=high, enable_llm=True)


@app.get("/api/stream")
async def stream(task_id: str, drift_high: float = 0.62, delta_spike: float = 0.55,
                 gray_band: float = 0.05, delay: float = 0.35):
    record = _records.get(task_id)
    if not record:
        raise HTTPException(404, "unknown task_id")
    cfg = _legacy_cfg(drift_high, delta_spike, gray_band)
    meta = {
        "task_id": record["task_id"], "label": record["label"],
        "category": CATEGORY_LABELS.get(record["category"], record["category"]),
        "instruction": record["user_instruction"], "n_steps": len(record["steps"]),
    }
    return StreamingResponse(_run_stream(record["user_instruction"], record["steps"], cfg, delay, meta), media_type="text/event-stream")


@app.post("/api/stream_custom")
async def stream_custom(req: Request):
    body = await req.json()
    intent = str(body.get("intent", "")).strip()
    steps = body.get("steps", [])
    if not intent or not isinstance(steps, list) or not steps:
        raise HTTPException(400, "need non-empty 'intent' and 'steps' list")
    th = body.get("thresholds", {})
    cfg = _legacy_cfg(float(th.get("drift_high", 0.62)), float(th.get("delta_spike", 0.55)), float(th.get("gray_band", 0.05)))
    meta = {"task_id": "custom", "label": body.get("label"), "category": "Custom trajectory", "instruction": intent, "n_steps": len(steps)}
    return StreamingResponse(_run_stream(intent, steps, cfg, float(body.get("delay", 0.35)), meta), media_type="text/event-stream")


if FRONTEND.exists():
    @app.get("/")
    def index():
        return FileResponse(FRONTEND / "index.html")
    app.mount("/", StaticFiles(directory=FRONTEND), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, app_dir=str(ROOT / "backend"))
