# LoopGuard

**An observability & runtime-monitoring layer for LangGraph agents.**

LoopGuard wraps a LangGraph run and gives you full visibility into what the agent is
doing: it **traces** every step, surfaces live **metrics**, and runs **detectors** that
catch pathological behavior at runtime. Catching agents stuck in **prompt loops** is the
flagship capability - but LoopGuard is the whole observability layer, not just a loop detector.

## Three pillars

| Pillar | What it does | Module |
|---|---|---|
| **Tracing** | Records every node execution as a stream of `Event`s - the ordered stream *is* the trace | `loopguard/tracer.py` |
| **Observability** | Derives metrics from the trace (steps, tool-call frequency, repeats) | `loopguard/metrics.py` |
| **Runtime monitoring** | A live `Monitor` runs pluggable **detectors** over the event stream and can interrupt the agent | `loopguard/monitor.py`, `loopguard/detectors.py` |

## What "prompt loop" means

An agent runs a cycle: think (LLM) → act (tool) → observe → think → … A *loop* is when this
cycle stops making progress and repeats. LoopGuard recognizes several flavors:

| Type | Loop | Status |
|---|---|---|
| **B** | Same tool + similar args repeated ≥3× | ✅ `LoopDetector` |
| **A** | Similar LLM decision/context repeated ("prompt loop") | ✅ `SemanticLoopDetector` (OpenAI embeddings) |
| **C** | Cyclic conversation / repeated message exchange | Powers the React Flow UI view |
| **D** | Same node-path cycle in the graph | Later |

## Detection rule (MVP, Type B)

> Normalize each tool call to `tool(name, args)`. If the same normalized signature
> appears **≥ 3 times** within a sliding window, raise a **loop alert** and interrupt the run.

Threshold and window are configurable. "Similar args" today means normalized-equal
(trim / lowercase / sorted keys); fuzzy similarity arrives with Detector A.

## Architecture

```
LangGraph agent ──stream──▶ Tracer ──events──▶ Monitor ──┬─▶ LoopDetector   (Type B)
                            (trace)            (runtime)  ├─▶ StallDetector
                                                          └─▶ ToolStormDetector
                                                              │
                              Metrics ◀──reads trace──────────┘
                                                              │
                                                   alerts ──▶ interrupt / log / (React Flow UI)
```

## Stack

- **Agent + detector + tracing:** Python, [LangGraph](https://github.com/langchain-ai/langgraph)
- **Transport (later):** FastAPI + WebSocket/SSE
- **UI (later):** React + React Flow + Vite (TypeScript)
- **Semantic detection (later):** sentence-transformers

## Run

```bash
pip install -r requirements.txt

python main.py            # Type-B demo: identical tool loop (offline, no API key)
python main.py semantic   # Type-A demo: paraphrase loop (needs OPENAI_API_KEY)

# Streaming server (Phase 2)
uvicorn server:app --reload
#   GET /graph?scenario=exact|semantic  -> graph topology (nodes + edges)
#   WS  /run?scenario=exact|semantic    -> live event/alert/metrics/done stream
```

You'll see the trace timeline, then LoopGuard catching the agent looping and interrupting it.
The semantic scenario needs an `OPENAI_API_KEY` (copy `.env.example` to `.env`).
