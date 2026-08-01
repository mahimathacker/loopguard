"""Public live guardrail wrapper.

The lower-level ``stream_run`` function still does the actual LangGraph streaming work.
LoopGuard gives callers a small, stable object for live use and sensible defaults for
the cheap detectors that should be safe to keep on during development.
"""

from __future__ import annotations

from collections.abc import Iterator

from .detectors import Detector, LoopDetector, StallDetector


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
    ) -> None:
        self.detectors = detectors if detectors is not None else default_detectors()
        self.recursion_limit = recursion_limit

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
