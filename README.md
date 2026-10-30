# LoopGuard

[![PyPI version](https://img.shields.io/pypi/v/loopguard-runtime)](https://pypi.org/project/loopguard-runtime/)
[![Python versions](https://img.shields.io/pypi/pyversions/loopguard-runtime)](https://pypi.org/project/loopguard-runtime/)
[![PyPI monthly downloads](https://img.shields.io/pypi/dm/loopguard-runtime)](https://pypi.org/project/loopguard-runtime/)

[![Docs](https://img.shields.io/badge/Docs-2563EB?style=for-the-badge)](docs/README.md)
[![PyPI](https://img.shields.io/badge/PyPI-3775A9?style=for-the-badge)](https://pypi.org/project/loopguard-runtime/)
[![Demo](https://img.shields.io/badge/Demo-FF0000?style=for-the-badge)](https://youtu.be/i5XRrvFoPA0)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge)](https://github.com/mahimathacker/loopguard)

**Runtime guardrails for AI agents that detect loops, repeated failures, and runs that stop making progress.**

Agents can keep spending time and tokens without getting closer to an answer.
LoopGuard watches LangGraph and LangChain agents as they run, checking repeated actions
against new results, progress, and failures to decide when to stop.

```text
same tool + useful new result   -> continue
same tool + same empty result   -> stop
503 -> 503 -> success           -> continue
401 -> 401 -> 401               -> stop
```

## See it in action

[![LoopGuard UI detecting a paraphrase loop and interrupting an agent pursuing an impossible goal](ui/public/realagentloop.png)](https://youtu.be/i5XRrvFoPA0)

An agent keeps rephrasing the same search while pursuing an impossible goal. LoopGuard
flags the loop and interrupts the run; the UI shows the evidence, tool calls, and verdict.

[Watch the demo on YouTube](https://youtu.be/i5XRrvFoPA0) · [Run it locally](docs/README.md#running-the-demo)

## Quick start

**Prerequisites:** Python 3.11+ and pip. The live example assumes you already have a
LangGraph/LangChain agent and its framework dependencies installed. See the
[installation guide](docs/README.md#installation) for optional dependencies.

```bash
pip install loopguard-runtime
```

For an existing LangGraph/LangChain agent (`agent`) and its input (`my_input`):

```python
from loopguard import LoopGuardCallbackHandler, LoopGuardInterrupt

handler = LoopGuardCallbackHandler()

try:
    result = agent.invoke(my_input, config={"callbacks": [handler]})
except LoopGuardInterrupt as exc:
    print("LoopGuard stopped the run:", exc)

print(handler.report())
```

LoopGuard records the run, collects detector evidence, and can interrupt execution when
the policy reaches `STOP`. Repeated tool calls alone do not trigger a stop.

[Read the getting started guide →](docs/README.md#installation)

## What LoopGuard detects

| Pattern | Evidence |
| --- | --- |
| Repeated tool calls | The same normalized tool call recurring without useful new results. |
| Semantic loops | The same intent in different words; opt-in embedding support. |
| Stalls and no progress | Stagnant observations or runs that stop advancing. |
| Repeated failures | Retryable errors versus likely permanent failures. |
| Action and handoff cycles | Repeating action sequences or closed multi-agent handoff loops. |
| Budget overruns | Configurable limits on steps and tool calls. |

[Detector details and configuration →](docs/README.md#configuration)

## How it works

```text
Agent event -> Tracer -> Detectors -> DetectionSignals -> PolicyEngine -> GuardDecision
```

Detectors collect evidence; the policy combines it into `CONTINUE`, `WARN`, `REPLAN`,
`PAUSE`, or `STOP`. Repetition with stagnant output can stop a run, while useful new
results allow it to continue. `REPLAN` and `PAUSE` are recorded recommendations today;
automatic interruption happens only at `STOP`.

LoopGuard complements tracing tools by turning recorded events into runtime decisions.

[Architecture and detector reference →](docs/README.md#architecture-and-detectors)

## Live runs and saved traces

Use the callback handler above for live runs. Reports include events, detector signals,
policy decisions, metrics, and alerts. Observation-only mode is available with
`LoopGuardCallbackHandler(interrupt_on_fatal=False)`.

LoopGuard can also inspect an exported trace without running the original agent again.
From a repository checkout:

```bash
python -m loopguard.ingest examples/sample_trace.json
```

Use check mode for pass/fail behavior in scripts or CI:

```bash
python -m loopguard.check examples/sample_trace.json
```

[Saved trace analysis →](docs/README.md#saved-trace-analysis) · [Live integration and reports →](docs/README.md#live-integration-and-reports)

## Demo scenarios

| Scenario | What to watch |
| --- | --- |
| `exact`, `semantic` | Identical calls and paraphrased requests getting caught. |
| `controlled_progress`, `controlled_503` | Useful polling and temporary failures recovering. |
| `controlled_401`, `controlled_empty` | Permanent auth failures and repeated empty results. |
| `calc`, `trap` | A real agent completing a task versus pursuing an impossible goal. |

The demo includes a FastAPI/WebSocket backend and a Next.js UI. The `exact` scenario
runs offline; the others need model or embedding credentials.

[Running the demo: installation, providers, and commands →](docs/README.md#running-the-demo)

## Evaluation

The core suite runs offline with fake agents and embeddings:

```bash
python -m unittest discover -v
```

The [v0.2 labeled traces](examples/v02_labeled_traces.json) cover healthy polling,
pagination, retry recovery, permanent failures, exact loops, and action/handoff cycles.
The evaluation harness reports detector precision, recall, and F1.

[Evaluation setup and commands →](docs/README.md#evaluation-and-tests)

## Documentation and contributing

Start with the [local guides](docs/README.md) for setup, configuration, saved traces,
and the demo. These guides will move to Mintlify when the documentation site is ready.

For bugs, ideas, or contributions, [open an issue](https://github.com/mahimathacker/loopguard/issues)
or submit a pull request. Include a reproducible trace when reporting detection problems
and run the offline test suite for code changes.

[Roadmap →](docs/README.md#roadmap) · [Project structure →](docs/README.md#project-structure)

## Package status

[loopguard-runtime v0.2.1](https://pypi.org/project/loopguard-runtime/0.2.1/) is an alpha
release. Package metadata requires Python 3.11+ and lists Python 3.11 and 3.12 classifiers.
