"""Detection signals and runtime decisions.

Detectors report evidence as signals. A policy engine decides what to do with that
evidence. This keeps "what happened?" separate from "should we stop the agent?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GuardAction(str, Enum):
    CONTINUE = "continue"
    WARN = "warn"
    REPLAN = "replan"
    PAUSE = "pause"
    STOP = "stop"


@dataclass
class DetectionSignal:
    detector: str
    kind: str
    score: float
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardDecision:
    action: GuardAction
    risk_score: float
    reasons: list[DetectionSignal] = field(default_factory=list)

    @property
    def recommended_action_label(self) -> str:
        return f"Recommended action: {self.action.value.upper()}"

    @property
    def stop_reason(self) -> str:
        kinds = {reason.kind for reason in self.reasons}
        if "permanent_failure" in kinds:
            return "Run stopped: non-retryable authentication failure."
        if "step_budget" in kinds:
            return "Run stopped: step budget exceeded."
        if "tool_call_budget" in kinds:
            return "Run stopped: tool-call budget exceeded."
        if "no_progress" in kinds and (
            "repeated_tool_call" in kinds or "semantic_loop" in kinds or "cycle" in kinds
        ):
            return "Run stopped: repeated actions produced no progress."
        if "handoff_loop" in kinds:
            return "Run stopped: repeated agent handoff cycle."
        return "Run stopped: LoopGuard policy reached the stop threshold."
