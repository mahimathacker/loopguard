# LoopGuard

LoopGuard detects when LangGraph agents get stuck.

It traces each run and catches repeated tool calls, semantic/paraphrase loops, and
no-progress behavior before an agent wastes time and tokens. Use it while developing,
around a live run, or later in CI to fail a change when a known task starts looping.

It works on the demo agents in this repo and on your own LangGraph agent.

## Why this exists

LLM agents run in a loop: think, act, observe, repeat. Sometimes that loop goes wrong and
the agent keeps doing the same thing without making progress. It might call the same tool
over and over, or rephrase the same failed request again and again. Left alone, it burns
tokens and time and never finishes. LoopGuard watches the agent while it runs and steps in
when this happens.

## How it works

LoopGuard stays narrow: record what the agent did, detect stuck behavior, and report the
result. It has three parts, one for each job:

| Part | Job | File |
|------|-----|------|
| Tracer | Records every step as an event. The ordered list of events is the trace. | `loopguard/tracer.py` |
| Metrics | Turns the trace into numbers: total steps, tool calls, repeat rate. | `loopguard/metrics.py` |
| Monitor | Runs detectors over the live trace and interrupts the agent when one fires. | `loopguard/monitor.py`, `loopguard/detectors.py` |

The flow is one direction:

```
LangGraph agent --stream--> Tracer --events--> Monitor --> detectors --> alert --> interrupt
```

There are two detectors today:

- `LoopDetector`: catches the same tool call with the same arguments repeated three times.
- `SemanticLoopDetector`: catches the same intent repeated in different words, using OpenAI
  embeddings. This catches loops that exact matching misses.

## The four scenarios

LoopGuard ships with four runnable scenarios. Two are scripted and offline (good for a
quick, deterministic test). Two use a real `gpt-4o-mini` agent with real tools.

### 1. Scripted: identical tool loop

A scripted agent calls the same tool with the same arguments every step. `LoopDetector`
catches it on the third call.

![Identical tool loop](ui/public/scriptedtool.png)

### 2. Scripted: paraphrase loop

A scripted agent asks the same thing in different words each step. Exact matching sees
distinct calls and misses it, but `SemanticLoopDetector` catches the repeated intent.

![Paraphrase loop](ui/public/scriptedopenai.png)

### 3. Real agent: solvable task

A real `gpt-4o-mini` agent gets a question it can answer. It uses the calculator tool,
returns the answer, and finishes. LoopGuard stays quiet and just shows the trace and
metrics of a healthy run.

![Real agent finishing](ui/public/realagentmath.png)

### 4. Real agent: impossible goal

A real agent is given a goal it cannot reach (find a source for a claim that is not true).
It searches the web on its own, again and again, with different queries. Nothing is faked,
the loop comes from the situation. `SemanticLoopDetector` catches it and stops the run.

![Real agent caught in a loop](ui/public/realagentloop.png)

## Tech stack

| Layer | Tool |
|-------|------|
| Agent runtime | Python, [LangGraph](https://github.com/langchain-ai/langgraph) |
| Real LLM agent | `gpt-4o-mini` via `langchain-openai` |
| Web search tool | DuckDuckGo via `ddgs` (no API key) |
| Semantic detection | OpenAI embeddings (`text-embedding-3-small`) |
| API server | FastAPI + WebSocket |
| UI | Next.js + React Flow + Tailwind CSS (in `ui/`) |

## Requirements

- Python 3.11 or newer. The macOS system Python 3.9 uses an old SSL library and is not
  supported, use a virtual environment on a newer Python.
- Node.js 18 or newer (for the UI).
- An OpenAI API key for the `semantic`, `calc`, and `trap` scenarios. The `exact` scenario
  runs offline with no key.

## Setup

### 1. Backend (Python)

```bash
cd agent-loop
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. API key

```bash
cp .env.example .env
# open .env and set OPENAI_API_KEY=sk-...
```

### 3. UI (Node)

```bash
cd ui
npm install
```

## Run

Run the backend and the UI in two terminals.

### Terminal 1: API server

```bash
source .venv/bin/activate
uvicorn server:app --reload --port 8000
```

### Terminal 2: UI

```bash
cd ui
npm run dev
```

Open http://localhost:3000, pick a scenario from the dropdown, and press Run.

### Command line (no UI)

You can also run any scenario straight from the terminal:

```bash
python main.py            # exact   (offline, no key)
python main.py semantic   # semantic
python main.py calc       # real agent, finishes
python main.py trap       # real agent, loops and gets caught
```

## Use LoopGuard on your own agent

LoopGuard is not tied to these demos. Wrap any compiled LangGraph agent and read the
stream of messages it produces:

```python
from loopguard import stream_run
from loopguard.detectors import LoopDetector, SemanticLoopDetector

for msg in stream_run(my_agent, [LoopDetector(), SemanticLoopDetector()], my_input):
    if msg["type"] == "alert" and msg["fatal"]:
        print("loop detected:", msg["message"])  # the run is interrupted right after
```

`stream_run` works with both classic state-dict agents and message-based ReAct agents. It
yields `event`, `alert`, `metrics`, and `done` messages that you can log, store, or render.

## Analyze an external agent's trace (offline)

You do not have to plug LoopGuard into a live agent to use it. If another team can export
their agent runs as JSON, LoopGuard can replay those runs through the same detectors and
report which ones looped. This is the lowest-effort way to try LoopGuard on someone else's
agent: no SDK, no access to their running system.

```bash
python -m loopguard.ingest examples/sample_trace.json
```

The adapter is forgiving about field names (`tool`/`tool_name`/`name`, `args`/`arguments`/
`input`, and so on), so most exports work with little or no change. See
`examples/sample_trace.json` for the accepted shape.

## Measure how good the detectors are

LoopGuard is not trying to be a full eval platform, but the detectors still need to be
measurable. The repo includes a small harness so detector quality is a number, not a
guess.

```bash
python -m loopguard.evals
```

It grades the detectors against a labeled dataset and reports precision, recall, and F1;
computes a convergence score (how often an agent takes the optimal path); compares detector
line-ups side by side; and, when an OpenAI key is set, runs an LLM-as-a-judge check for
hallucinations against human labels. See `loopguard/evals.py`.

## Roadmap

LoopGuard is deliberately not a general AI eval SDK, dataset manager, judge system, cost
platform, or full observability dashboard. The product stays focused on one painful
question: **did my agent get stuck?**

### Available now (v0)

- Tracing, live metrics, and runtime interruption for a single LangGraph agent.
- Three detectors: `LoopDetector` (exact repeats), `SemanticLoopDetector` (paraphrase
  loops), `StallDetector` (no progress).
- Four runnable scenarios, a FastAPI server, and a Next.js UI.
- Small detector-quality harness (precision/recall/F1).
- Offline trace analyzer for external agents (`loopguard/ingest.py`).

### Next (v0.x)

- **Small stuck-run report**: a concise CLI/JSON report that says clean or flagged, which
  detector fired, the repeated tool/intent/result, and the step where it happened.
- **CI check mode**: run LoopGuard against saved traces or scripted scenarios and exit
  non-zero when a known task loops, stalls, or exceeds a simple step budget.
- **Agent handoff loop detector** for multi-agent and swarm setups: catch loops that span
  several agents (A calls B calls C calls A) where no single agent looks stuck. The trace
  adapter already carries the `caller` field this needs.
- **Live SDK for LangGraph (`LoopGuardCallbackHandler`)**: a callback handler that emits an
  event per tool call and lets LoopGuard monitor and interrupt a real run in process, rather
  than after the fact.

### Later

- **GitHub check annotation**: post the small stuck-run report on a pull request when the
  CI check fails.
- **LangGraph.js / LangChain.js support** for the live path (the offline analyzer already
  works on any exported JSON regardless of language).
- **Pluggable action policies**: per-detector choices to warn, interrupt, or hand off.

## Project structure

```
agent-loop/
  loopguard/          the library
    tracer.py         records events (the trace)
    metrics.py        derives numbers from the trace
    detectors.py      LoopDetector, SemanticLoopDetector, StallDetector
    monitor.py        runs detectors over the live trace
    embeddings.py     OpenAI embeddings for semantic detection
    agent.py          demo agents (scripted) and the real gpt-4o-mini agent
    scenarios.py      the four named scenarios
    runner.py         drives a run and streams messages (the public API)
    evals.py          evaluation harness (precision/recall, convergence, judge)
    ingest.py         offline trace analyzer for external agents
  server.py           FastAPI server: /graph, /run (WebSocket), /eval
  main.py             command line runner
  examples/           sample external traces for the offline analyzer
  ui/                 Next.js + React Flow front end
  public/             README screenshots
  requirements.txt    Python dependencies
```
