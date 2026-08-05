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
