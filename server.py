"""LoopGuard streaming server.

Exposes the same runs as the CLI, but over HTTP/WebSocket so the React Flow UI can
consume the trace live:

    GET  /graph?scenario=...    -> static graph topology (nodes + edges)
    GET  /runtime               -> selected model and embedding providers
    WS   /run?scenario=...      -> live event/signal/decision/alert stream

Run:  uvicorn server:app --reload
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from loopguard.detectors import LoopDetector
from loopguard.evals import evaluate, loop_dataset
from loopguard.runner import stream_run
from loopguard.scenarios import get_scenario

load_dotenv()

app = FastAPI(title="LoopGuard")

# The UI runs on a different origin (Vite dev server), so allow cross-origin access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# How long to pause between streamed messages, so the UI animation is watchable.
STEP_DELAY_SECONDS = 0.5
REACT_SCENARIOS = {
    "controlled_progress",
    "controlled_503",
    "controlled_401",
    "controlled_empty",
    "calc",
    "trap",
}


def react_topology() -> dict:
    return {
        "nodes": [{"id": "agent"}, {"id": "tools"}],
        "edges": [{"source": "agent", "target": "tools"}, {"source": "tools", "target": "agent"}],
    }


def graph_topology(agent) -> dict:
    """Extract nodes + edges from a compiled LangGraph for React Flow to render."""
    internal = ("__start__", "__end__")
    try:
        g = agent.get_graph()
        nodes = [{"id": n} for n in g.nodes if n not in internal]
        edges = [
            {"source": e.source, "target": e.target}
            for e in g.edges
            if e.source not in internal and e.target not in internal
        ]
        return {"nodes": nodes, "edges": edges}
    except Exception:  # noqa: BLE001 - fall back to the known shape if introspection changes
        return {
            "nodes": [{"id": "agent"}, {"id": "tools"}],
            "edges": [{"source": "agent", "target": "tools"}, {"source": "tools", "target": "agent"}],
        }


@app.get("/graph")
def graph(scenario: str = "exact") -> dict:
    if scenario in REACT_SCENARIOS:
        return react_topology()
    _, agent, _, _ = get_scenario(scenario)
    return graph_topology(agent)


@app.get("/eval")
def eval_detectors() -> dict:
    """Grade the LoopDetector on the labeled dataset and return its scorecard for the UI."""
    cases = loop_dataset()
    card = evaluate(cases, [LoopDetector(threshold=3)])
    return {
        "detector": "LoopDetector",
        "cases": len(cases),
        "tp": card.tp, "fp": card.fp, "fn": card.fn, "tn": card.tn,
        "precision": round(card.precision, 2),
        "recall": round(card.recall, 2),
        "f1": round(card.f1, 2),
    }


@app.get("/runtime")
def runtime() -> dict:
    """Return non-secret runtime provider settings for the UI."""
    provider = os.getenv("LOOPGUARD_MODEL_PROVIDER", "openai").strip().lower()
    if provider == "gemini":
        model = os.getenv("LOOPGUARD_GEMINI_MODEL", "gemini-3.6-flash")
    else:
        model = os.getenv("LOOPGUARD_OPENAI_MODEL", "gpt-4o-mini")
    return {
        "model_provider": provider,
        "model": model,
        "embedding_provider": os.getenv("LOOPGUARD_EMBEDDING_PROVIDER", "openai"),
    }


@app.websocket("/run")
async def run(websocket: WebSocket) -> None:
    await websocket.accept()
    scenario = websocket.query_params.get("scenario", "exact")
    recursion_limit = 30 if scenario in {"calc", "trap"} else 50

    try:
        _, agent, detectors, initial = get_scenario(scenario)
        for message in stream_run(agent, detectors, initial, recursion_limit):
            await websocket.send_json(message)
            await asyncio.sleep(STEP_DELAY_SECONDS)
        await websocket.close()
    except WebSocketDisconnect:
        pass  # client navigated away mid-run; nothing to clean up
