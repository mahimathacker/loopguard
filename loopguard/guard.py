"""Public live guardrail wrapper.

The lower-level ``stream_run`` function still does the actual LangGraph streaming work.
LoopGuard gives callers a small, stable object for live use and sensible defaults for
the cheap detectors that should be safe to keep on during development.
"""

from __future__ import annotations

from collections.abc import Iterator

from .config import config_budget, load_config, section
from .detectors import (
    Detector,
    LoopDetector,
    StallDetector,
    StepBudgetDetector,
    ToolCallBudgetDetector,
)


def default_detectors() -> list[Detector]:
    """Cheap live defaults: exact repeated tool calls and no-progress warnings."""
    return [LoopDetector(), StallDetector()]


class LoopGuard:
    """Watch a live LangGraph agent stream and emit LoopGuard messages.

    Semantic detection is intentionally opt-in because it calls an embedding model. Pass
    a SemanticLoopDetector in ``detectors`` when you want paraphrase-loop detection.
    """

    def __init__(
        self,
        detectors: list[Detector] | None = None,
        recursion_limit: int = 50,
        max_steps: int | None = None,
        max_tool_calls: int | None = None,
    ) -> None:
        self.detectors = list(detectors) if detectors is not None else default_detectors()
        self.recursion_limit = recursion_limit
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        if max_steps is not None:
            self.detectors.append(StepBudgetDetector(max_steps))
        if max_tool_calls is not None:
            self.detectors.append(ToolCallBudgetDetector(max_tool_calls))

    @classmethod
    def from_config(
        cls,
        path: str,
        detectors: list[Detector] | None = None,
        recursion_limit: int = 50,
    ) -> "LoopGuard":
        """Create a live guard from ``loopguard.yml`` budget settings."""
        config = section(load_config(path), "live")
        return cls(
            detectors=detectors,
            recursion_limit=recursion_limit,
            max_steps=config_budget(config, "max_steps"),
            max_tool_calls=config_budget(config, "max_tool_calls"),
        )

    def stream(
        self,
        agent,
        initial: dict | None = None,
        recursion_limit: int | None = None,
    ) -> Iterator[dict]:
        """Run ``agent`` under LoopGuard and stream event/alert/metrics/done messages."""
        from .runner import stream_run

        limit = self.recursion_limit if recursion_limit is None else recursion_limit
        yield from stream_run(agent, self.detectors, initial, limit)


__all__ = ["LoopGuard", "default_detectors"]
