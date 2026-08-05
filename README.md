# LoopGuard

LoopGuard detects when LangGraph agents get stuck.

It traces each run and catches repeated tool calls, semantic/paraphrase loops, and
no-progress behavior before an agent wastes time and tokens. Use it today while developing
LangGraph agents or replaying saved traces. CI check mode is planned next, so teams can
fail a change when a known task starts looping.

It works on the demo agents in this repo, includes a small `LoopGuard` live wrapper, and
can replay saved JSON traces. A cleaner callback-based SDK is planned next.

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

There are four behavior detectors today:

- `LoopDetector`: catches the same tool call with the same arguments repeated three times.
- `SemanticLoopDetector`: catches the same intent repeated in different words, using OpenAI
  embeddings. This catches loops that exact matching misses.
- `StallDetector`: warns when the agent keeps observing the same result and stops making
  progress. It is non-fatal today, so it reports a stall without interrupting the run.
- `HandoffLoopDetector`: catches repeated closed handoff cycles across agents, such as
  planner -> researcher -> reviewer -> planner.

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

LoopGuard is not tied to these demos. For live LangGraph/LangChain runs, attach the
callback handler to the run config:

```python
from loopguard import LoopGuardCallbackHandler, LoopGuardInterrupt

handler = LoopGuardCallbackHandler()

try:
    result = agent.invoke(my_input, config={"callbacks": [handler]})
except LoopGuardInterrupt as exc:
    print("loop detected:", exc)

print(handler.report())
```

The handler records tool starts, tool results, and tool errors, then raises
`LoopGuardInterrupt` when a fatal detector fires. Use `interrupt_on_fatal=False` if you
want to collect alerts without stopping the run.

For demos or cases where you want LoopGuard to drive the stream itself, wrap any compiled
LangGraph agent with the live guard and read the stream of messages it produces:

```python
from loopguard import LoopGuard
from loopguard.detectors import LoopDetector, SemanticLoopDetector, StallDetector

guard = LoopGuard(detectors=[
    LoopDetector(),
    SemanticLoopDetector(),
    StallDetector(),
], max_steps=40, max_tool_calls=15)

for msg in guard.stream(my_agent, my_input):
    if msg["type"] == "alert" and msg["fatal"]:
        print("loop detected:", msg["message"])  # the run is interrupted right after
```

`LoopGuard.stream(...)` works with both classic state-dict agents and message-based ReAct
agents. It yields `event`, `alert`, `metrics`, and `done` messages that you can log,
store, or render. The lower-level `stream_run(...)` helper is still available for callers
that want function-style control.

## Analyze an external agent's trace (offline)

You do not have to plug LoopGuard into a live agent to use it. If another team can export
their agent runs as JSON, LoopGuard can replay those runs through the same detectors and
report which ones looped. This is the lowest-effort way to try LoopGuard on someone else's
agent: no SDK, no access to their running system.

```bash
python -m loopguard.ingest examples/sample_trace.json
python -m loopguard.ingest examples/sample_trace.json --json
```

The adapter is forgiving about field names (`tool`/`tool_name`/`name`, `args`/`arguments`/
`input`, and so on), so most exports work with little or no change. See
`examples/sample_trace.json` for the accepted shape. The `--json` flag prints a local
stuck-run report with `clean`, `looping`, and `stalled` counts plus per-run alerts.

## Check saved traces

Use the check command when you want pass/fail behavior for local scripts or CI. It exits
with `1` when a selected stuck status appears, and `2` for invalid input.

```bash
python -m loopguard.check examples/sample_trace.json
python -m loopguard.check examples/sample_trace.json --fail-on stalled
python -m loopguard.check examples/sample_trace.json --max-steps 20 --max-tool-calls 10
python -m loopguard.check examples/sample_trace.json --config loopguard.yml
python -m loopguard.check examples/sample_trace.json --fail-on looping --fail-on stalled --json
```

By default, only `looping` fails the check. Use `--fail-on stalled` to fail on
no-progress warnings too, or `--fail-on alerts` to fail on any alert. Step and tool-call
budgets always fail the check when exceeded.

For CI, you can keep the same policy in a small `loopguard.yml`:

```yaml
live:
  max_steps: 40
  max_tool_calls: 15
  exact_threshold: 3
  exact_window: 12
  stall_patience: 4
  stall_fatal: false
  handoff_repeats: 2
  handoff_window: 16
  handoff_max_cycle_length: 5
  handoff_fatal: true
  semantic: false

check:
  fail_on:
    - looping
    - stalled
  max_steps: 40
  max_tool_calls: 15
```

For check-only config, top-level keys also work:

```yaml
fail_on:
  - looping
  - stalled
max_steps: 40
max_tool_calls: 15
```

Command-line flags override config values.

For live runs, `exact_threshold` and `exact_window` tune repeated-tool-call detection.
`stall_patience` controls how many repeated observations count as no progress, and
`stall_fatal` controls whether that should interrupt or only warn. `handoff_*` settings
tune multi-agent cycle detection from `caller` fields in traces. Set `semantic: true` to
enable paraphrase-loop detection with embeddings, then tune `semantic_threshold`,
`semantic_window`, and `semantic_min_repeats` if needed.

## Measure how good the detectors are

LoopGuard is not trying to be a full eval platform, but the detectors still need to be
measurable. The repo includes a small harness so detector quality is a number, not a
guess.

```bash
python -m loopguard.evals
```

It grades the loop detectors against labeled stuck-agent cases and reports precision,
recall, and F1. See `loopguard/evals.py`.

## Test

The core test suite is offline and uses fake agents/fake embeddings, so it does not need
OpenAI, web search, or a running server.

```bash
python -m unittest discover -v
```

## Roadmap

LoopGuard is deliberately not a general AI eval SDK, dataset manager, judge system, cost
platform, or full observability dashboard. The product stays focused on one painful
question: **did my agent get stuck?**

### Available now (v0)

- Tracing, live metrics, and runtime interruption for a single LangGraph agent.
- Four behavior detectors: `LoopDetector` (exact repeats), `SemanticLoopDetector`
  (paraphrase loops), `StallDetector` (no progress), and `HandoffLoopDetector`
  (multi-agent handoff cycles).
- Four runnable scenarios, a FastAPI server, and a Next.js UI.
- A small `LoopGuard` live wrapper for compiled LangGraph agents.
- A `LoopGuardCallbackHandler` for attaching LoopGuard to LangGraph/LangChain live runs.
- Local stuck-run JSON reports for saved traces.
- CI-friendly saved-trace check mode with exit codes and simple step/tool-call budgets.
- Simple `loopguard.yml` config for saved-trace checks and live detector/budget policies.
- Small detector-quality harness (precision/recall/F1).
- Offline trace analyzer for external agents (`loopguard/ingest.py`).

### Next (v0.x)

- **Real trace tuning**: calibrate thresholds and false positives against production swarm
  traces, especially paraphrase loops and handoff cycles.

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
    detectors.py      loop, semantic loop, stall, budget, and handoff detectors
    monitor.py        runs detectors over the live trace
    guard.py          small public LoopGuard wrapper for live runs
    langgraph.py      callback handler for LangGraph/LangChain live runs
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
