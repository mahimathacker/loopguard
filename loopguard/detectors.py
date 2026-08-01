"""RUNTIME-MONITORING pillar - the detectors.

A Detector inspects each new Event (with the trace history available) and optionally
returns an Alert. The Monitor runs every registered detector on every event.

LoopDetector is the flagship. StallDetector is included to prove the framework is
general - LoopGuard is an observability layer with pluggable checks, not a single-purpose
loop detector.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .embeddings import Embedder, OpenAIEmbedder, cosine
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
    """Same tool and similar args repeated >= threshold times.

    We only look at tool-call events (signatures starting with "tool:"). Within a
    sliding window of recent events, if one tool signature recurs `threshold` times,
    the agent is stuck calling the same thing - that's a loop.
    """

    name = "LoopDetector"

    def __init__(self, threshold: int = 3, window: int = 12, fatal: bool = True) -> None:
        self.threshold = threshold
        self.window = window
        self.fatal = fatal

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
                fatal=self.fatal,
            )
        return None


class StallDetector(Detector):
    """The agent keeps acting but the observed result never changes (no progress).

    Included to show the Monitor framework is general. Fires when the last
    `patience` *result* events all share the same signature.
    """

    name = "StallDetector"

    def __init__(self, patience: int = 4, fatal: bool = False) -> None:
        self.patience = patience
        self.fatal = fatal

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
                fatal=self.fatal,
            )
        return None


class StepBudgetDetector(Detector):
    """Interrupt when a run exceeds a maximum number of recorded events."""

    name = "StepBudgetDetector"

    def __init__(self, max_steps: int, fatal: bool = True) -> None:
        self.max_steps = max_steps
        self.fatal = fatal

    def inspect(self, event: Event, history: list[Event]) -> Alert | None:
        if len(history) > self.max_steps:
            return Alert(
                detector=self.name,
                kind="step_budget",
                message=f"Run exceeded step budget: {len(history)} > {self.max_steps}.",
                fatal=self.fatal,
            )
        return None


class ToolCallBudgetDetector(Detector):
    """Interrupt when a run exceeds a maximum number of tool-call decisions."""

    name = "ToolCallBudgetDetector"

    def __init__(self, max_tool_calls: int, fatal: bool = True) -> None:
        self.max_tool_calls = max_tool_calls
        self.fatal = fatal

    def inspect(self, event: Event, history: list[Event]) -> Alert | None:
        if not event.signature.startswith("tool:"):
            return None

        tool_calls = sum(1 for item in history if item.signature.startswith("tool:"))
        if tool_calls > self.max_tool_calls:
            return Alert(
                detector=self.name,
                kind="tool_call_budget",
                message=(
                    f"Run exceeded tool-call budget: {tool_calls} > "
                    f"{self.max_tool_calls}."
                ),
                fatal=self.fatal,
            )
        return None


class HandoffLoopDetector(Detector):
    """Detect repeated closed handoff cycles across multiple agents.

    A single agent can look fine locally while the system cycles globally:
    planner -> researcher -> reviewer -> planner. This detector reads the optional
    ``caller`` payload field and flags when the same closed cycle repeats consecutively.
    """

    name = "HandoffLoopDetector"

    def __init__(
        self,
        repeats: int = 2,
        window: int = 16,
        max_cycle_length: int = 5,
        fatal: bool = True,
    ) -> None:
        self.repeats = repeats
        self.window = window
        self.max_cycle_length = max_cycle_length
        self.fatal = fatal

    def _handoff_edges(self, history: list[Event]) -> list[tuple[str, str]]:
        edges: list[tuple[str, str]] = []
        for item in history[-self.window:]:
            caller = item.payload.get("caller")
            if caller is None:
                continue
            source = str(caller)
            target = item.node
            if source != target:
                edges.append((source, target))
        return edges

    def _is_closed_cycle(self, cycle: list[tuple[str, str]]) -> bool:
        if len(cycle) < 2:
            return False
        for (_, target), (next_source, _) in zip(cycle, cycle[1:]):
            if target != next_source:
                return False
        return cycle[-1][1] == cycle[0][0]

    def _cycle_text(self, cycle: list[tuple[str, str]]) -> str:
        nodes = [cycle[0][0], *[target for _, target in cycle]]
        return " -> ".join(nodes)

    def inspect(self, event: Event, history: list[Event]) -> Alert | None:
        if event.payload.get("caller") is None:
            return None

        edges = self._handoff_edges(history)
        max_length = min(self.max_cycle_length, len(edges) // self.repeats)
        for length in range(2, max_length + 1):
            cycle = edges[-length:]
            if not self._is_closed_cycle(cycle):
                continue
            expected = cycle * self.repeats
            if edges[-len(expected):] == expected:
                return Alert(
                    detector=self.name,
                    kind="handoff_loop",
                    message=(
                        f"Repeated agent handoff cycle {self.repeats}x within the last "
                        f"{self.window} steps: {self._cycle_text(cycle)}."
                    ),
                    fatal=self.fatal,
                )
        return None


class SemanticLoopDetector(Detector):
    """Same *intent* repeated, even when the wording differs.

    Strict LoopDetector misses an agent that rephrases the same failed approach
    ("weather in Paris" -> "Paris weather today" -> "current weather in Paris"): the
    signatures all differ, so the counter never climbs. This detector embeds the content
    of each tool decision and flags when recent decisions are above a similarity
    threshold, catching paraphrase loops.
    """

    name = "SemanticLoopDetector"

    def __init__(
        self,
        embedder: Embedder | None = None,
        threshold: float = 0.8,
        window: int = 6,
        min_repeats: int = 3,
        fatal: bool = True,
    ) -> None:
        self.embedder = embedder or OpenAIEmbedder()
        self.threshold = threshold
        self.window = window
        self.min_repeats = min_repeats
        self.fatal = fatal

    def _content(self, event: Event) -> str:
        """The text we embed: the variable part of the decision (the tool args),
        falling back to the human-readable action if no args are present."""
        args = event.payload.get("last_args")
        if isinstance(args, dict) and args:
            return " ".join(str(v) for v in args.values())
        return event.action

    def inspect(self, event: Event, history: list[Event]) -> Alert | None:
        if not event.signature.startswith("tool:"):
            return None

        current = self.embedder.encode(self._content(event))
        recent = [
            e for e in history[-self.window:]
            if e.signature.startswith("tool:") and e is not event
        ]
        similar = sum(
            1 for e in recent
            if cosine(current, self.embedder.encode(self._content(e))) >= self.threshold
        )

        # similar counts the prior matches; +1 for the current decision itself.
        if similar >= self.min_repeats - 1:
            return Alert(
                detector=self.name,
                kind="semantic_loop",
                message=(
                    f"{similar + 1} semantically similar tool calls within the last "
                    f"{self.window} steps (>= {self.threshold:.0%} similar) - a paraphrase loop."
                ),
                fatal=self.fatal,
            )
        return None
