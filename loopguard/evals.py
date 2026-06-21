"""EVALUATION pillar - measuring how good the detectors themselves are."""

from dataclasses import dataclass

from .detectors import Detector
from .monitor import Monitor
from .runner import tool_signature


@dataclass
class EvalCase:
    """One labeled example for grading a detector."""

    name: str                            # human label, e.g. "repeats the same search 3x"
    decisions: list[tuple[str, dict]]    # the agent's tool calls in order: (tool_name, args)
    is_loop: bool                        # ground truth: should a correct detector interrupt?


def replay(case: EvalCase, detectors: list[Detector]) -> bool:
    """Run one case through the detectors; return True if they would interrupt the run."""
    for detector in detectors:
        detector.reset()  # clear any per-run state so cases never leak into each other

    monitor = Monitor(detectors=detectors)  # fresh monitor == fresh trace, one per case
    for tool, args in case.decisions:
        signature = tool_signature(tool, args)
        monitor.observe("agent", f"decided: {tool}({args})", signature, last_tool=tool, last_args=args)
        if monitor.should_interrupt:
            return True  # a fatal alert fired -> detector predicts "loop"
    return False  # got through every decision with no fatal alert -> predicts "healthy"


if __name__ == "__main__":
    # Smoke test:  python -m loopguard.evals
    from .detectors import LoopDetector

    looping = EvalCase("same search 3x", [("search", {"q": "paris"})] * 3, is_loop=True)
    healthy = EvalCase("three distinct calls", [
        ("search", {"q": "paris"}),
        ("calculator", {"expr": "2+2"}),
        ("search", {"q": "france population"}),
    ], is_loop=False)

    for case in (looping, healthy):
        predicted = replay(case, [LoopDetector(threshold=3)])
        mark = "OK " if predicted == case.is_loop else "BAD"
        print(f"  [{mark}] {case.name}: predicted={predicted} truth={case.is_loop}")
