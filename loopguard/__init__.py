"""LoopGuard - stuck-agent detection for LangGraph agents.

LoopGuard traces agent steps and runs detectors that catch repeated tool calls,
semantic loops, and no-progress behavior before an agent burns time and tokens.

Three pillars:
    tracer.Tracer    - (TRACING) records every node execution as a stream of Events
    metrics.Metrics  - (OBSERVABILITY) derives stats from the trace
    monitor.Monitor  - (RUNTIME MONITORING) runs pluggable detectors over the live stream

Detectors (in detectors.py) are the runtime checks. LoopDetector is the headliner;
SemanticLoopDetector and StallDetector catch paraphrase loops and no-progress runs.
"""

__version__ = "0.2.1"

from .monitor import Monitor
from .tracer import Tracer

__all__ = [
    "LoopGuard",
    "LoopGuardCallbackHandler",
    "LoopGuardInterrupt",
    "Monitor",
    "Tracer",
    "DetectionSignal",
    "GuardAction",
    "GuardDecision",
    "PolicyConfig",
    "PolicyEngine",
    "stream_run",
    "__version__",
]


def __getattr__(name: str):
    """Load LangGraph-facing helpers only when callers ask for them."""
    if name == "LoopGuard":
        from .guard import LoopGuard

        return LoopGuard
    if name == "LoopGuardCallbackHandler":
        from .langgraph import LoopGuardCallbackHandler

        return LoopGuardCallbackHandler
    if name == "LoopGuardInterrupt":
        from .langgraph import LoopGuardInterrupt

        return LoopGuardInterrupt
    if name in {"DetectionSignal", "GuardAction", "GuardDecision"}:
        from . import signals

        return getattr(signals, name)
    if name in {"PolicyConfig", "PolicyEngine"}:
        from . import policy

        return getattr(policy, name)
    if name == "stream_run":
        from .runner import stream_run

        return stream_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
