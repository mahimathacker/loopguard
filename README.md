# LoopGuard

An observability and runtime-monitoring layer for LangGraph agents. LoopGuard traces
every step an agent takes, shows live metrics, and runs detectors that catch agents stuck
in loops, then stops them before they waste time and tokens.

It works on the demo agents in this repo and on your own LangGraph agent.

## Why this exists

LLM agents run in a loop: think, act, observe, repeat. Sometimes that loop goes wrong and
the agent keeps doing the same thing without making progress. It might call the same tool
over and over, or rephrase the same failed request again and again. Left alone, it burns
tokens and time and never finishes. LoopGuard watches the agent while it runs and steps in
when this happens.

## How it works

LoopGuard has three parts, one for each job:

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
  server.py           FastAPI server: /graph and /run (WebSocket)
  main.py             command line runner
  ui/                 Next.js + React Flow front end
  public/             README screenshots
  requirements.txt    Python dependencies
```
