"""Named demo scenarios: which agent + which detectors.

Shared by the CLI and the WebSocket server so both expose the exact same runs.
"""

from __future__ import annotations

from .agent import build_agent, build_paraphrasing_agent
from .detectors import Detector, LoopDetector, SemanticLoopDetector, StallDetector


def get_scenario(name: str) -> tuple[str, object, list[Detector]]:
    """Return (title, compiled_agent, detectors) for a scenario name."""
    if name == "semantic":
        return (
            "Scenario: paraphrase prompt loop (Type A)",
            build_paraphrasing_agent(),
            [LoopDetector(threshold=3), SemanticLoopDetector(threshold=0.8), StallDetector(patience=4)],
        )
    return (
        "Scenario: identical tool loop (Type B)",
        build_agent(),
        [LoopDetector(threshold=3), StallDetector(patience=4)],
    )
