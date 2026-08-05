"""Policy engine for progress-aware runtime decisions."""

from __future__ import annotations

from dataclasses import dataclass

from .signals import DetectionSignal, GuardAction, GuardDecision


@dataclass
class PolicyConfig:
    warn_at: float = 0.50
    replan_at: float = 0.70
    pause_at: float = 0.85
    stop_at: float = 0.90


class PolicyEngine:
    """Combine detector signals into a single runtime action."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def decide(self, signals: list[DetectionSignal]) -> GuardDecision:
        risk = self._calculate_risk(signals)
        if risk >= self.config.stop_at:
            action = GuardAction.STOP
        elif risk >= self.config.pause_at:
            action = GuardAction.PAUSE
        elif risk >= self.config.replan_at:
            action = GuardAction.REPLAN
        elif risk >= self.config.warn_at:
            action = GuardAction.WARN
        else:
            action = GuardAction.CONTINUE
        return GuardDecision(action=action, risk_score=risk, reasons=signals)

    def _calculate_risk(self, signals: list[DetectionSignal]) -> float:
        if not signals:
            return 0.0

        kinds = {signal.kind for signal in signals}
        if any(kind.endswith("_budget") for kind in kinds):
            return 1.0
        if "permanent_failure" in kinds:
            return 0.95
        if (
            {"repeated_tool_call", "no_progress"} <= kinds
            or {"semantic_loop", "no_progress"} <= kinds
            or {"cycle", "no_progress"} <= kinds
        ):
            return 0.92
        repeat_kinds = {"repeated_tool_call", "semantic_loop"}
        if any(
            signal.evidence.get("fatal_hint")
            and signal.kind not in repeat_kinds
            and signal.score >= 0.70
            for signal in signals
        ):
            return 0.91
        if "handoff_loop" in kinds:
            return 0.91
        if "cycle" in kinds:
            return 0.78
        if "no_progress" in kinds:
            return 0.72
        if "retryable_failure" in kinds:
            return min(0.68, max(signal.score for signal in signals))
        return min(0.89, max(signal.score for signal in signals))
