"""Minimal LoopGuard callback integration for a LangGraph/LangChain agent.

This example shows the integration shape. Replace ``agent`` and ``my_input`` with your
compiled LangGraph app and normal input payload.
"""

from __future__ import annotations

from loopguard import LoopGuardCallbackHandler, LoopGuardInterrupt


def run_with_loopguard(agent, my_input: dict) -> dict:
    handler = LoopGuardCallbackHandler.from_config("loopguard.yml")
    try:
        result = agent.invoke(my_input, config={"callbacks": [handler]})
    except LoopGuardInterrupt:
        result = None

    report = handler.report()
    return {
        "result": result,
        "loopguard": {
            "interrupted": report["interrupted"],
            "signals": report["signals"],
            "decisions": report["decisions"],
            "metrics": report["metrics"],
        },
    }
