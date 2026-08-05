"""RUNTIME-MONITORING pillar - the Monitor.

The Monitor is the live nerve center. As the agent streams events, the Monitor:
  1. records each Event in the Tracer (tracing),
  2. runs every registered Detector against it (runtime monitoring),
  3. collects any Alerts and decides whether to interrupt the run.

It deliberately knows nothing about LangGraph internals - it just consumes Events.
That keeps it reusable for any agent framework and for the React Flow UI later.
"""

from __future__ import annotations

from dataclasses import dataclass

from .detectors import Alert, Detector
from .policy import PolicyEngine
from .signals import DetectionSignal, GuardAction, GuardDecision
from .tracer import Event, Tracer


@dataclass
class ObservationResult:
    event: Event
    signals: list[DetectionSignal]
    decision: GuardDecision
    alerts: list[Alert]


class Monitor:
    def __init__(
        self,
        detectors: list[Detector],
        tracer: Tracer | None = None,
        policy: PolicyEngine | None = None,
    ) -> None:
        self.detectors = detectors
        self.tracer = tracer or Tracer()
        self.policy = policy or PolicyEngine()
        self.signals: list[DetectionSignal] = []
        self.decisions: list[GuardDecision] = []
        self.alerts: list[Alert] = []

    def observe(self, node: str, action: str, signature: str, **payload) -> list[Alert]:
        """Record one event and return legacy alerts for backward compatibility."""
        return self.observe_decision(node, action, signature, **payload).alerts

    def observe_decision(
        self,
        node: str,
        action: str,
        signature: str,
        **payload,
    ) -> ObservationResult:
        """Record one event, collect signals, and produce a policy decision."""
        event = self.tracer.record(node, action, signature, **payload)
        fired: list[DetectionSignal] = []
        for detector in self.detectors:
            result = detector.inspect(event, self.tracer.events)
            if result:
                fired.append(self._as_signal(result))

        self.signals.extend(fired)
        decision_signals = self.signals[-12:] if fired else []
        decision = self.policy.decide(decision_signals)
        self.decisions.append(decision)

        alerts = self._legacy_alerts(fired, decision)
        self.alerts.extend(alerts)
        return ObservationResult(event=event, signals=fired, decision=decision, alerts=alerts)

    def _as_signal(self, result: DetectionSignal | Alert) -> DetectionSignal:
        if isinstance(result, DetectionSignal):
            return result
        return DetectionSignal(
            detector=result.detector,
            kind=result.kind,
            score=1.0 if result.fatal else 0.55,
            message=result.message,
            evidence={"fatal_hint": result.fatal},
        )

    def _legacy_alerts(
        self,
        signals: list[DetectionSignal],
        decision: GuardDecision,
    ) -> list[Alert]:
        if decision.action == GuardAction.CONTINUE:
            return []

        fatal = decision.action == GuardAction.STOP
        alerts: list[Alert] = []
        seen: set[tuple[str, str, bool]] = set()
        for signal in decision.reasons:
            key = (signal.detector, signal.kind, fatal)
            if key in seen:
                continue
            seen.add(key)
            alerts.append(
                Alert(
                    detector=signal.detector,
                    kind=signal.kind,
                    message=signal.message,
                    fatal=fatal,
                )
            )
        return alerts

    @property
    def should_interrupt(self) -> bool:
        """True once the policy has decided to stop the run."""
        return any(decision.action == GuardAction.STOP for decision in self.decisions)
