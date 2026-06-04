"""RUNTIME-MONITORING pillar — the Monitor.

The Monitor is the live nerve center. As the agent streams events, the Monitor:
  1. records each Event in the Tracer (tracing),
  2. runs every registered Detector against it (runtime monitoring),
  3. collects any Alerts and decides whether to interrupt the run.

It deliberately knows nothing about LangGraph internals — it just consumes Events.
That keeps it reusable for any agent framework and for the React Flow UI later.
"""

from __future__ import annotations

from .detectors import Alert, Detector
from .tracer import Event, Tracer


class Monitor:
    def __init__(self, detectors: list[Detector], tracer: Tracer | None = None) -> None:
        self.detectors = detectors
        self.tracer = tracer or Tracer()
        self.alerts: list[Alert] = []

    def observe(self, node: str, action: str, signature: str, **payload) -> list[Alert]:
        """Record one event and run all detectors against it. Returns alerts raised."""
        event = self.tracer.record(node, action, signature, **payload)
        fired: list[Alert] = []
        for detector in self.detectors:
            alert = detector.inspect(event, self.tracer.events)
            if alert:
                fired.append(alert)
                self.alerts.append(alert)
        return fired

    @property
    def should_interrupt(self) -> bool:
        """True once any fatal alert has fired — runtime intervention signal."""
        return any(a.fatal for a in self.alerts)
