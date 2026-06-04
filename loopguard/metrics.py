"""OBSERVABILITY pillar.

Tracing answers "what happened?" (the ordered list of Events).
Metrics answers "what does it add up to?" by deriving numbers from that same list.

Nothing here runs the agent or talks to LangGraph. It is a pure read over the trace:
give it the events the Tracer collected, get back a summary you can print, log, or
later feed to the React Flow UI as dashboard stats.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .tracer import Event


@dataclass
class Metrics:
    """A snapshot of stats derived from one trace (one run)."""

    total_steps: int = 0                       # how many events in total
    by_node: dict[str, int] = field(default_factory=dict)        # agent vs tools counts
    tool_calls: int = 0                        # number of "tool:" events
    distinct_tool_calls: int = 0               # how many *unique* tool signatures
    most_common_action: tuple[str, int] | None = None  # (signature, count) of the top repeat
    repeat_rate: float = 0.0                   # fraction of tool calls that were repeats

    @classmethod
    def from_events(cls, events: list[Event]) -> "Metrics":
        """Compute all metrics in a single pass-ish over the trace."""
        if not events:
            return cls()

        # How many times each node ran. Counter turns a list into {value: count}.
        by_node = dict(Counter(e.node for e in events))

        # Pull out just the tool-call events; their signatures are what loops are made of.
        tool_sigs = [e.signature for e in events if e.signature.startswith("tool:")]
        sig_counts = Counter(tool_sigs)

        tool_calls = len(tool_sigs)
        distinct = len(sig_counts)

        # most_common(1) -> [(signature, count)]; the single most-repeated tool call.
        most_common = sig_counts.most_common(1)[0] if sig_counts else None

        # Repeat rate: of all tool calls, how many were NOT the first time we saw them?
        # distinct calls are the "firsts"; everything beyond that is a repeat.
        repeats = tool_calls - distinct
        repeat_rate = repeats / tool_calls if tool_calls else 0.0

        return cls(
            total_steps=len(events),
            by_node=by_node,
            tool_calls=tool_calls,
            distinct_tool_calls=distinct,
            most_common_action=most_common,
            repeat_rate=repeat_rate,
        )

    def report(self) -> str:
        """Human-readable summary, the dashboard-in-text version of observability."""
        lines = [
            f"  total steps        : {self.total_steps}",
            f"  steps by node      : {self.by_node}",
            f"  tool calls         : {self.tool_calls}",
            f"  distinct tool calls: {self.distinct_tool_calls}",
            f"  repeat rate        : {self.repeat_rate:.0%}",
        ]
        if self.most_common_action:
            sig, count = self.most_common_action
            lines.append(f"  most repeated      : {sig.removeprefix('tool:')} (x{count})")
        return "\n".join(lines)
