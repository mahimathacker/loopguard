"""TRACING pillar.

Defines what a *trace* is in LoopGuard, from first principles:

    Trace = one full run of the agent
    Event = one observable thing that happened during the run (a node executed)

Every time a LangGraph node runs, the Tracer records an Event. The ordered list of
Events IS the trace. Both the Metrics layer and the runtime Monitor read this same
stream - and it's also the exact payload a React Flow UI would consume later.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from itertools import count
from typing import Any


# A monotonic counter for ordering. Deterministic and easy to reason about.
# Swap/augment with time.time() when you want real latency metrics.
_seq = count()


@dataclass
class Event:
    """One unit of observable work: a single node execution in the graph."""

    step: int                       # ordering index within the run
    node: str                       # which graph node ran ("agent", "tools", ...)
    action: str                     # human-readable summary of what it did
    signature: str                  # canonical key used for loop/repeat detection
    payload: dict[str, Any] = field(default_factory=dict)  # raw details

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tracer:
    """Collects Events as the graph streams. One Tracer == one run == one trace."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def record(self, node: str, action: str, signature: str, **payload: Any) -> Event:
        event = Event(
            step=next(_seq),
            node=node,
            action=action,
            signature=signature,
            payload=payload,
        )
        self.events.append(event)
        return event

    def timeline(self) -> str:
        """Pretty-print the whole trace as a vertical timeline."""
        return "\n".join(
            f"  [{e.step:>3}] {e.node:<7} | {e.action}" for e in self.events
        )
