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
    """Normalize a tool call into a canonical key ('similar args' = normalized-equal)."""
    norm = ", ".join(f"{k}={str(v).strip().lower()}" for k, v in sorted(args.items()))
    return f"tool:{tool}({norm})"


def _interpret(node: str, delta: dict) -> tuple[str, str, dict] | None:
    """Translate a node update into (action, signature, payload).

    Handles both the scripted agents (last_tool/last_args/last_result deltas) and the
    real ReAct agent (message-based deltas). The payload is always a clean, JSON-
    serializable dict, never raw LangChain message objects.
    """
    if "messages" in delta:
        return _interpret_messages(delta["messages"])
    if "last_tool" in delta:
        tool, args = delta["last_tool"], delta["last_args"]
        return f"decided: {tool}({args})", tool_signature(tool, args), {
            "last_tool": tool,
            "last_args": args,
        }
    if "last_result" in delta:
        result = delta["last_result"]
        return f"observed: {result}", f"result:{result}", {"last_result": result}
    return None


def _interpret_messages(messages: list) -> tuple[str, str, dict] | None:
    """Pull the action + signature out of the newest message from a ReAct agent step."""
    if not messages:
        return None
    msg = messages[-1]

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:  # the model decided to call a tool
        call = tool_calls[0]
        name, args = call["name"], call.get("args", {})
        return f"decided: {name}({args})", tool_signature(name, args), {
            "last_tool": name,
            "last_args": args,
        }
    if getattr(msg, "type", None) == "tool":  # a tool returned a result
        content = str(getattr(msg, "content", "")).strip().replace("\n", " ")[:160]
        return f"observed: {content}", f"result:{content}", {"last_result": content}

    # otherwise it is the model's final answer (no tool calls) -> the run is finishing
    content = str(getattr(msg, "content", "")).strip().replace("\n", " ")[:200]
    return f"answered: {content}", f"final:{content}", {"final": content}
    

def stream_run(
    agent,
    detectors: list[Detector],
    initial: dict | None = None,
    recursion_limit: int = 50,
) -> Iterator[dict]:
    monitor = Monitor(detectors=detectors)
    initial = initial or {"goal": "find the weather", "steps": 0}
    interrupted = False

    try:
        for chunk in agent.stream(
            initial, stream_mode="updates", config={"recursion_limit": recursion_limit}
        ):
            for node, delta in chunk.items():
                interpreted = _interpret(node, delta)
                if interpreted is None:
                    continue
                action, sig, payload = interpreted

                alerts = monitor.observe(node, action, sig, **payload)
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