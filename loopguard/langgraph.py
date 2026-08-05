"""Callback-based live integration for LangGraph/LangChain agents.

The wrapper API in ``LoopGuard.stream(...)`` is useful for demos and controlled runs.
This callback handler is the cleaner production-facing integration: attach it to a
LangGraph/LangChain run, and it feeds tool events into the same Monitor/detectors.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ModuleNotFoundError:  # keep LoopGuard importable without LangChain installed
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass

from .config import config_budget, load_config, section
from .detectors import Alert, Detector
from .detectors import StepBudgetDetector, ToolCallBudgetDetector
from .guard import configured_detectors, default_detectors
from .metrics import Metrics
from .monitor import Monitor
from .runner import tool_signature


class LoopGuardInterrupt(RuntimeError):
    """Raised by the callback handler when a fatal detector should stop a live run."""

    def __init__(self, alert: Alert) -> None:
        self.alert = alert
        super().__init__(alert.message)


def _tool_name(serialized: dict | None, fallback: str = "tool") -> str:
    if not isinstance(serialized, dict):
        return fallback
    name = serialized.get("name")
    if name:
        return str(name)
    tool_id = serialized.get("id")
    if isinstance(tool_id, list) and tool_id:
        return str(tool_id[-1])
    if tool_id:
        return str(tool_id)
    return fallback


def _tool_args(tool_input: Any, inputs: Any = None) -> dict:
    raw = inputs if inputs is not None else tool_input
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    return {"input": raw}


def _caller(metadata: dict | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    for key in ("caller", "parent", "parent_agent", "invoked_by"):
        if metadata.get(key) is not None:
            return str(metadata[key])
    return None


class LoopGuardCallbackHandler(BaseCallbackHandler):
    """LangGraph/LangChain callback handler that detects stuck live runs.

    Attach it with ``config={"callbacks": [handler]}``. When a fatal alert fires, the
    handler raises ``LoopGuardInterrupt`` by default so the running agent stops early.
    """

    def __init__(
        self,
        detectors: list[Detector] | None = None,
        interrupt_on_fatal: bool = True,
        max_steps: int | None = None,
        max_tool_calls: int | None = None,
    ) -> None:
        super().__init__()
        self.detectors = list(detectors) if detectors is not None else default_detectors()
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        if max_steps is not None:
            self.detectors.append(StepBudgetDetector(max_steps))
        if max_tool_calls is not None:
            self.detectors.append(ToolCallBudgetDetector(max_tool_calls))
        self.interrupt_on_fatal = interrupt_on_fatal
        self.monitor = Monitor(self.detectors)
        self.messages: list[dict] = []

    @classmethod
    def from_config(
        cls,
        path: str,
        interrupt_on_fatal: bool = True,
    ) -> "LoopGuardCallbackHandler":
        """Create a callback handler from the ``live`` section of ``loopguard.yml``."""
        config = section(load_config(path), "live")
        return cls(
            detectors=configured_detectors(config),
            interrupt_on_fatal=interrupt_on_fatal,
            max_steps=config_budget(config, "max_steps"),
            max_tool_calls=config_budget(config, "max_tool_calls"),
        )

    def reset(self) -> None:
        """Reset monitor state before reusing the handler for another run."""
        for detector in self.detectors:
            detector.reset()
        self.monitor = Monitor(self.detectors)
        self.messages = []

    @property
    def alerts(self) -> list[Alert]:
        return self.monitor.alerts

    @property
    def events(self) -> list[dict]:
        return [event.as_dict() for event in self.monitor.tracer.events]

    @property
    def should_interrupt(self) -> bool:
        return self.monitor.should_interrupt

    def metrics(self) -> dict:
        return asdict(Metrics.from_events(self.monitor.tracer.events))

    def report(self) -> dict:
        return {
            "interrupted": self.should_interrupt,
            "events": self.events,
            "alerts": [asdict(alert) for alert in self.alerts],
            "metrics": self.metrics(),
        }

    def _record(self, node: str, action: str, signature: str, **payload: Any) -> None:
        alerts = self.monitor.observe(node, action, signature, **payload)
        event_message = {"type": "event", **self.monitor.tracer.events[-1].as_dict()}
        self.messages.append(event_message)
        for alert in alerts:
            self.messages.append({"type": "alert", **asdict(alert)})

        if self.interrupt_on_fatal:
            fatal = next((alert for alert in alerts if alert.fatal), None)
            if fatal is not None:
                raise LoopGuardInterrupt(fatal)

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str | None = None,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: Any = None,
        **kwargs: Any,
    ) -> None:
        """Record a tool-call decision as soon as LangChain starts the tool."""
        name = _tool_name(serialized)
        args = _tool_args(input_str, inputs)
        payload: dict[str, Any] = {
            "last_tool": name,
            "last_args": args,
            "run_id": str(run_id) if run_id is not None else None,
            "parent_run_id": str(parent_run_id) if parent_run_id is not None else None,
            "tags": tags or [],
        }
        caller = _caller(metadata)
        if caller is not None:
            payload["caller"] = caller
        self._record("agent", f"decided: {name}({args})", tool_signature(name, args), **payload)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the tool observation returned to the agent."""
        text = str(output).strip().replace("\n", " ")[:160]
        payload: dict[str, Any] = {
            "last_result": text,
            "run_id": str(run_id) if run_id is not None else None,
            "parent_run_id": str(parent_run_id) if parent_run_id is not None else None,
            "tags": tags or [],
        }
        caller = _caller(metadata)
        if caller is not None:
            payload["caller"] = caller
        self._record("tools", f"observed: {text}", f"result:{text}", **payload)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record tool failures as observations, so repeated errors can be detected."""
        text = f"{type(error).__name__}: {error}".strip().replace("\n", " ")[:160]
        payload: dict[str, Any] = {
            "last_result": text,
            "run_id": str(run_id) if run_id is not None else None,
            "parent_run_id": str(parent_run_id) if parent_run_id is not None else None,
            "tags": tags or [],
        }
        caller = _caller(metadata)
        if caller is not None:
            payload["caller"] = caller
        self._record("tools", f"error: {text}", f"result:{text}", **payload)


__all__ = ["LoopGuardCallbackHandler", "LoopGuardInterrupt"]
