"""LoopGuard demo runner.

Wires the three pillars together around the looping agent:

    LangGraph agent  --stream-->  Monitor(Tracer + Detectors)  -->  interrupt + report

Run:  python main.py
"""

from __future__ import annotations

from langgraph.errors import GraphRecursionError

from loopguard.agent import build_agent
from loopguard.detectors import LoopDetector, StallDetector
from loopguard.metrics import Metrics
from loopguard.monitor import Monitor


def tool_signature(tool: str, args: dict) -> str:
    """Normalize a tool call into a canonical key (Type-B 'similar args' = normalized-equal)."""
    norm = ", ".join(f"{k}={str(v).strip().lower()}" for k, v in sorted(args.items()))
    return f"tool:{tool}({norm})"


def main() -> None:
    agent = build_agent()
    monitor = Monitor(detectors=[LoopDetector(threshold=3), StallDetector(patience=4)])

    initial = {"goal": "find the weather", "steps": 0}
    print("▶ Running agent under LoopGuard...\n")

    interrupted = False
    try:
        # stream_mode="updates" yields {node_name: state_delta} for each node that runs.
        for chunk in agent.stream(initial, stream_mode="updates", config={"recursion_limit": 50}):
            for node, delta in chunk.items():
                if node == "agent":
                    tool, args = delta["last_tool"], delta["last_args"]
                    sig = tool_signature(tool, args)
                    action = f"decided: {tool}({args})"
                elif node == "tools":
                    sig = f"result:{delta['last_result']}"
                    action = f"observed: {delta['last_result']}"
                else:
                    continue

                alerts = monitor.observe(node, action, sig, **delta)
                for a in alerts:
                    flag = "🛑" if a.fatal else "⚠️ "
                    print(f"{flag} [{a.detector}] {a.message}")

            if monitor.should_interrupt:
                interrupted = True
                break
    except GraphRecursionError:
        print("⚠️  LangGraph's built-in recursion_limit tripped (LoopGuard would've caught it sooner).")

    print("\n-- Trace timeline --")
    print(monitor.tracer.timeline())

    print("\n-- Metrics --")
    print(Metrics.from_events(monitor.tracer.events).report())

    print("\n-- Verdict --")
    if interrupted:
        print("LoopGuard interrupted the agent: prompt loop detected. ✅")
    else:
        print("Run finished without a fatal loop alert.")


if __name__ == "__main__":
    main()
