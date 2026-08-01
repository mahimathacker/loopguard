"""Public live guardrail wrapper.

The lower-level ``stream_run`` function still does the actual LangGraph streaming work.
LoopGuard gives callers a small, stable object for live use and sensible defaults for
the cheap detectors that should be safe to keep on during development.
"""

from __future__ import annotations

from collections.abc import Iterator

from .config import config_bool, config_budget, config_float, config_int, load_config, section
from .detectors import (
    Detector,
    HandoffLoopDetector,
    LoopDetector,
    SemanticLoopDetector,
    StallDetector,
    StepBudgetDetector,
    ToolCallBudgetDetector,
)


def default_detectors(
    exact_threshold: int = 3,
    exact_window: int = 12,
    exact_fatal: bool = True,
    stall_patience: int = 4,
    stall_fatal: bool = False,
    handoff_repeats: int = 2,
    handoff_window: int = 16,
    handoff_max_cycle_length: int = 5,
    handoff_fatal: bool = True,
) -> list[Detector]:
    """Cheap live defaults: exact repeated tool calls and no-progress warnings."""
    return [
        LoopDetector(threshold=exact_threshold, window=exact_window, fatal=exact_fatal),
        StallDetector(patience=stall_patience, fatal=stall_fatal),
        HandoffLoopDetector(
            repeats=handoff_repeats,
            window=handoff_window,
            max_cycle_length=handoff_max_cycle_length,
            fatal=handoff_fatal,
        ),
    ]


def configured_detectors(config: dict) -> list[Detector]:
    """Build live detectors from the small ``live`` config section."""
    detectors = default_detectors(
        exact_threshold=config_int(config, "exact_threshold", 3),
        exact_window=config_int(config, "exact_window", 12),
        exact_fatal=config_bool(config, "exact_fatal", True),
        stall_patience=config_int(config, "stall_patience", 4),
        stall_fatal=config_bool(config, "stall_fatal", False),
        handoff_repeats=config_int(config, "handoff_repeats", 2),
        handoff_window=config_int(config, "handoff_window", 16),
        handoff_max_cycle_length=config_int(config, "handoff_max_cycle_length", 5),
        handoff_fatal=config_bool(config, "handoff_fatal", True),
    )
    if config_bool(config, "semantic", False):
        detectors.append(
            SemanticLoopDetector(
                threshold=config_float(config, "semantic_threshold", 0.8),
                window=config_int(config, "semantic_window", 6),
                min_repeats=config_int(config, "semantic_min_repeats", 3),
                fatal=config_bool(config, "semantic_fatal", True),
            )
        )
    return detectors


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
        """Create a live guard from ``loopguard.yml`` detector and budget settings."""
        config = section(load_config(path), "live")
        return cls(
            detectors=list(detectors) if detectors is not None else configured_detectors(config),
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


__all__ = ["LoopGuard", "configured_detectors", "default_detectors"]
