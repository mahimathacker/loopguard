"""RUNTIME-MONITORING pillar - the detectors.

A Detector inspects each new Event (with the trace history available) and optionally
returns an Alert. The Monitor runs every registered detector on every event.

LoopDetector (Type B) is the MVP headliner. StallDetector and ToolStormDetector are
included to prove the framework is general - LoopGuard is an observability layer with
pluggable checks, not a single-purpose loop detector.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .tracer import Event


@dataclass
class Alert:
    """A detector's verdict that something is wrong, raised at runtime."""

    detector: str       # which detector fired
    kind: str           # short machine code, e.g. "tool_loop"
    message: str        # human-readable explanation
    fatal: bool = True  # if True, the Monitor will interrupt the run


class Detector:
    """Base class. Subclasses implement inspect()."""

    name: str = "detector"

    def inspect(self, event: Event, history: list[Event]) -> Alert | None:
        raise NotImplementedError

    def reset(self) -> None:  # called at the start of each run
        pass


class LoopDetector(Detector):
    """Type B: same tool + similar args repeated >= threshold times.

    We only look at tool-call events (signatures starting with "tool:"). Within a
    sliding window of recent events, if one tool signature recurs `threshold` times,
    the agent is stuck calling the same thing - that's a loop.
    """

    name = "LoopDetector"

    def __init__(self, threshold: int = 3, window: int = 12) -> None:
        self.threshold = threshold
        self.window = window

    def inspect(self, event: Event, history: list[Event]) -> Alert | None:
        if not event.signature.startswith("tool:"):
            return None

        recent = [e for e in history[-self.window:] if e.signature.startswith("tool:")]
        count = Counter(e.signature for e in recent)[event.signature]

        if count >= self.threshold:
            return Alert(
                detector=self.name,
                kind="tool_loop",
                message=(
                    f"Repeated tool call {count}x within the last {self.window} steps: "
                    f"{event.signature.removeprefix('tool:')}"
                ),
            )
        return None


class StallDetector(Detector):
    """The agent keeps acting but the observed result never changes (no progress).

    Included to show the Monitor framework is general. Fires when the last
    `patience` *result* events all share the same signature.
    """

    name = "StallDetector"

    def __init__(self, patience: int = 4) -> None:
        self.patience = patience

    def inspect(self, event: Event, history: list[Event]) -> Alert | None:
        if not event.signature.startswith("result:"):
            return None
        results = [e for e in history if e.signature.startswith("result:")]
        tail = results[-self.patience:]
        if len(tail) >= self.patience and len({e.signature for e in tail}) == 1:
            return Alert(
                detector=self.name,
                kind="stall",
                message=f"No new information for {self.patience} consecutive observations.",
                fatal=False,  # warn, don't necessarily kill
            )
        return None
