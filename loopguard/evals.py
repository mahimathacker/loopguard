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


@dataclass
class Scorecard:
    """Confusion-matrix counts for one detector over a dataset, plus derived rates."""

    tp: int = 0  # loop predicted, and it really was a loop      (caught it - good)
    fp: int = 0  # loop predicted, but the run was healthy        (false alarm - killed a good agent)
    fn: int = 0  # healthy predicted, but it really was a loop    (missed it - the worst case)
    tn: int = 0  # healthy predicted, and it really was healthy   (correctly stayed quiet)

    @property
    def precision(self) -> float:
        """Of the runs we flagged as loops, how many really were? Punishes false alarms."""
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        """Of the real loops, how many did we catch? Punishes misses."""
        actual_loops = self.tp + self.fn
        return self.tp / actual_loops if actual_loops else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall - one number that needs BOTH to be good."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def loop_dataset() -> list[EvalCase]:
    """A hand-labeled set of runs for grading the exact-match LoopDetector, offline."""
    return [
        # --- clear loops: a correct detector MUST catch these ---
        EvalCase("identical search x3", [("search", {"q": "paris"})] * 3, is_loop=True),
        EvalCase("identical search x5", [("search", {"q": "paris"})] * 5, is_loop=True),
        # --- healthy runs: a correct detector MUST stay quiet ---
        EvalCase("three distinct calls", [
            ("search", {"q": "paris"}),
            ("calculator", {"expr": "2+2"}),
            ("search", {"q": "france population"}),
        ], is_loop=False),
        EvalCase("same tool, different args", [
            ("search", {"q": "paris"}),
            ("search", {"q": "london"}),
            ("search", {"q": "tokyo"}),
        ], is_loop=False),
        # --- edge cases: where detectors quietly get it wrong ---
        # Repeated only twice, below threshold=3. Healthy by the detector's own rule, and a
        # good test that it does NOT trip too eagerly (guards against false positives).
        EvalCase("same search x2 only", [("search", {"q": "paris"})] * 2, is_loop=False),
        # A paraphrase loop: same intent, different wording each time. This IS a loop, but
        # the exact-match LoopDetector keys on the literal signature, so it will MISS this.
        # Labeling it True surfaces that blind spot as a real false negative (low recall),
        # which is exactly why SemanticLoopDetector exists.
        EvalCase("paraphrase loop", [
            ("search", {"q": "weather in paris"}),
            ("search", {"q": "paris weather today"}),
            ("search", {"q": "current weather in paris"}),
        ], is_loop=True),
    ]


def evaluate(cases: list[EvalCase], detectors: list[Detector]) -> Scorecard:
    """Replay every case, compare the prediction to the label, tally the confusion matrix."""
    card = Scorecard()
    for case in cases:
        predicted = replay(case, detectors)
        if predicted and case.is_loop:
            card.tp += 1
        elif predicted and not case.is_loop:
            card.fp += 1
        elif not predicted and case.is_loop:
            card.fn += 1
        else:
            card.tn += 1
    return card


if __name__ == "__main__":
    # Grade the exact-match LoopDetector on the labeled dataset:
    #   python -m loopguard.evals
    from .detectors import LoopDetector

    cases = loop_dataset()
    card = evaluate(cases, [LoopDetector(threshold=3)])

    print("LoopDetector on loop_dataset():")
    print(f"  cases     : {len(cases)}")
    print(f"  confusion : tp={card.tp} fp={card.fp} fn={card.fn} tn={card.tn}")
    print(f"  precision : {card.precision:.2f}")
    print(f"  recall    : {card.recall:.2f}")
    print(f"  f1        : {card.f1:.2f}")
