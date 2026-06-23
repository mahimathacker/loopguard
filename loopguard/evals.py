"""EVALUATION pillar - measuring how good the detectors themselves are."""

from dataclasses import dataclass

from .detectors import Detector
from .monitor import Monitor
from .runner import stream_run, tool_signature


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


def compare_detectors(
    cases: list[EvalCase], detector_sets: dict[str, list[Detector]]
) -> dict[str, Scorecard]:
    """Grade several detector line-ups on the SAME cases, for a side-by-side comparison.

    The point: prove whether adding a detector (e.g. SemanticLoopDetector) actually lifts
    recall without wrecking precision - i.e. whether it earns its extra cost.
    """
    return {name: evaluate(cases, detectors) for name, detectors in detector_sets.items()}


def convergence_score(step_counts: list[int]) -> float:
    """How close the agent stays to its own best (shortest) path across similar runs.

    Course formula:  (1/N) * sum_i  min(1, S_optimal / S_agent_i)
    S_optimal is the fewest steps any run took. 1.0 means every run was as short as the
    best one (always the optimal path); lower means some runs wandered with extra steps.
    """
    runs = [s for s in step_counts if s > 0]  # drop empty runs so we never divide by zero
    if not runs:
        return 0.0

    optimal = min(runs)                              # S_optimal: the shortest run we saw
    ratios = [min(1.0, optimal / s) for s in runs]   # each run scored 0..1 vs that best run
    return sum(ratios) / len(ratios)                 # average = overall convergence


def step_counts_for(agent, detectors: list[Detector], inputs: list[dict]) -> list[int]:
    """Run the agent once per input and collect total_steps for each run.

    These counts ARE the S_agent_i in convergence_score. We read total_steps straight from
    the 'metrics' message stream_run already emits, so this is just observing real runs.
    Requires whatever the agent needs (e.g. OPENAI_API_KEY for the real ReAct agent).
    """
    counts: list[int] = []
    for initial in inputs:
        steps = 0
        for msg in stream_run(agent, detectors, initial):
            if msg["type"] == "metrics":
                steps = msg["total_steps"]  # the final tally for this run
        counts.append(steps)
    return counts


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

    # Convergence: feed in step counts from several similar runs.
    perfect = [3, 3, 3, 3]      # every run took the optimal 3 steps
    wandering = [3, 5, 8, 3]    # two optimal, two that took extra steps
    print("\nConvergence score (1.0 = always optimal path):")
    print(f"  all runs optimal [3,3,3,3]   : {convergence_score(perfect):.2f}")
    print(f"  some runs wander [3,5,8,3]   : {convergence_score(wandering):.2f}")

    # Side-by-side: does adding the semantic detector lift recall? Needs OPENAI_API_KEY,
    # so we skip it gracefully when no key is set (keeps the rest of this demo offline).
    import os

    from dotenv import load_dotenv

    load_dotenv()  # pick up OPENAI_API_KEY from a .env file, as the server does
    if os.getenv("OPENAI_API_KEY"):
        from .detectors import SemanticLoopDetector

        print("\nDetector comparison (same dataset):")
        sets = {
            "exact only      ": [LoopDetector(threshold=3)],
            "exact + semantic": [LoopDetector(threshold=3), SemanticLoopDetector(threshold=0.8)],
        }
        for name, c in compare_detectors(cases, sets).items():
            print(f"  {name} : P={c.precision:.2f} R={c.recall:.2f} F1={c.f1:.2f}")
    else:
        print("\n(set OPENAI_API_KEY to also see the exact-vs-semantic detector comparison)")
