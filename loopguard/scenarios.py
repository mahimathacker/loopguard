"""Named demo scenarios: which agent + which detectors + the initial input.

Shared by the CLI and the WebSocket server so both expose the exact same runs.

Scripted scenarios (exact, semantic) run offline and are rigged to loop, useful as a
deterministic test of the detectors. Real scenarios (calc, trap) drive a genuine
gpt-4o-mini agent with real tools and require OPENAI_API_KEY: 'calc' finishes normally,
while 'trap' gives the agent an impossible stop-condition so it re-searches on its own.
"""

from __future__ import annotations

from .agent import build_agent, build_paraphrasing_agent, build_real_agent
from .detectors import Detector, LoopDetector, SemanticLoopDetector, StallDetector

# A solvable task: the agent uses the calculator once and finishes. LoopGuard stays quiet.
CALC_TASK = "What is 4801 multiplied by 379, then plus 1234? Use the calculator."

# An impossible stop-condition: there is no reputable source confirming this debunked
# myth, so a real search returns debunking results and the agent keeps refining its query
# and re-searching. Nothing about the tool is faked; the loop emerges from the goal.
TRAP_TASK = (
    "Find a reputable source that confirms the Great Wall of China is visible from the "
    "Moon with the unaided eye. Keep searching until you find a confirming source."
)

WEATHER_INPUT = {"goal": "find the weather", "steps": 0}


def get_scenario(name: str) -> tuple[str, object, list[Detector], dict]:
    """Return (title, compiled_agent, detectors, initial_input) for a scenario name."""
    if name == "semantic":
        return (
            "Scripted: paraphrase loop",
            build_paraphrasing_agent(),
            [LoopDetector(threshold=3), SemanticLoopDetector(threshold=0.8), StallDetector(patience=4)],
            WEATHER_INPUT,
        )
    if name == "calc":
        return (
            "Real agent: solvable task (finishes)",
            build_real_agent(),
            [LoopDetector(threshold=3), SemanticLoopDetector(threshold=0.85), StallDetector(patience=4)],
            {"messages": [{"role": "user", "content": CALC_TASK}]},
        )
    if name == "trap":
        return (
            "Real agent: impossible goal (loops)",
            build_real_agent(),
            [LoopDetector(threshold=3), SemanticLoopDetector(threshold=0.7), StallDetector(patience=4)],
            {"messages": [{"role": "user", "content": TRAP_TASK}]},
        )
    return (
        "Scripted: identical tool loop",
        build_agent(),
        [LoopDetector(threshold=3), StallDetector(patience=4)],
        WEATHER_INPUT,
    )
