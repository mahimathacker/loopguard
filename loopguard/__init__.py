"""LoopGuard - an observability & runtime-monitoring layer for LangGraph agents.

LoopGuard wraps a LangGraph run and gives you full visibility into what the agent
is doing: it traces every step, surfaces live metrics, and runs detectors that catch
pathological behavior at runtime. Catching agents stuck in loops is the flagship
capability - but LoopGuard is the whole observability layer, not just a loop detector.

Three pillars:
    tracer.Tracer    - (TRACING) records every node execution as a stream of Events
    metrics.Metrics  - (OBSERVABILITY) derives stats from the trace
    monitor.Monitor  - (RUNTIME MONITORING) runs pluggable detectors over the live stream

Detectors (in detectors.py) are the runtime checks. LoopDetector is the headliner;
StallDetector / ToolStormDetector show the framework is general, not loop-only.
"""

__version__ = "0.1.0"
