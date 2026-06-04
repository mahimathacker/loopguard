"""Shared orchestration: run an agent under a Monitor and emit structured messages.

This is the one place that drives a LangGraph run, translates each step into a trace
event, and runs the detectors. Both the CLI (main.py) and the WebSocket server consume
the same stream of messages, so there is no duplicated run logic.

stream_run() yields plain dicts, each tagged with a "type":
    {"type": "event",   ...Event fields}      one node executed
    {"type": "alert",   ...Alert fields}      a detector fired
    {"type": "metrics", ...Metrics fields}    final summary
    {"type": "done",    "interrupted": bool}  run finished
    {"type": "error",   "message": str}       something failed mid-run
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterator

from langgraph.errors import GraphRecursionError

from .detectors import Detector
from .metrics import Metrics
from .monitor import Monitor


def tool_signature(tool: str, args: dict) -> str:
    """Normalize a tool call into a canonical key (Type-B 'similar args' = normalized-equal)."""
    norm = ", ".join(f"{k}={str(v).strip().lower()}" for k, v in sorted(args.items()))
    return f"tool:{tool}({norm})"


def _interpret(node: str, delta: dict) -> tuple[str, str] | None:
    """Translate a LangGraph node update into (human action, machine signature)."""
    if node == "agent":
        sig = tool_signature(delta["last_tool"], delta["last_args"])
        return f"decided: {delta['last_tool']}({delta['last_args']})", sig
    if node == "tools":
        return f"observed: {delta['last_result']}", f"result:{delta['last_result']}"
    return None


def stream_run(agent, detectors: list[Detector], initial: dict | None = None) -> Iterator[dict]:
    monitor = Monitor(detectors=detectors)
    initial = initial or {"goal": "find the weather", "steps": 0}
    interrupted = False

    try:
        for chunk in agent.stream(initial, stream_mode="updates", config={"recursion_limit": 50}):
            for node, delta in chunk.items():
                interpreted = _interpret(node, delta)
                if interpreted is None:
                    continue
                action, sig = interpreted

                alerts = monitor.observe(node, action, sig, **delta)
                yield {"type": "event", **monitor.tracer.events[-1].as_dict()}
                for alert in alerts:
                    yield {"type": "alert", **asdict(alert)}

            if monitor.should_interrupt:
                interrupted = True
                break
    except GraphRecursionError:
        yield {"type": "error", "message": "LangGraph recursion_limit tripped before a detector fired."}
    except Exception as exc:  # noqa: BLE001 - surface embedding/API failures to the client
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}

    yield {"type": "metrics", **asdict(Metrics.from_events(monitor.tracer.events))}
    yield {"type": "done", "interrupted": interrupted}
