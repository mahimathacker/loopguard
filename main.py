"""LoopGuard CLI demo.

Drives a looping agent through the shared runner and prints the streamed messages.

Usage:
    python main.py            # identical tool loop (offline, no API key)
    python main.py semantic   # paraphrase loop (uses OpenAI embeddings)
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from loopguard.metrics import Metrics
from loopguard.runner import stream_run
from loopguard.scenarios import get_scenario

load_dotenv()  # picks up OPENAI_API_KEY from a .env file if present


NEEDS_KEY = {"semantic", "calc", "trap"}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "exact"

    if name in NEEDS_KEY and not os.getenv("OPENAI_API_KEY"):
        print(f"Set OPENAI_API_KEY (e.g. in a .env file) to run the '{name}' scenario.")
        return

    title, agent, detectors, initial = get_scenario(name)
    recursion_limit = 30 if name in {"calc", "trap"} else 50
    print(f"\n=== {title} ===\nRunning agent under LoopGuard...\n")

    for msg in stream_run(agent, detectors, initial, recursion_limit):
        kind = msg["type"]
        if kind == "event":
            print(f"  [{msg['step']:>3}] {msg['node']:<7} | {msg['action']}")
        elif kind == "alert":
            flag = "STOP" if msg["fatal"] else "WARN"
            print(f"[{flag}] [{msg['detector']}] {msg['message']}")
        elif kind == "error":
            print(f"[ERROR] {msg['message']}")
        elif kind == "metrics":
            print("\n-- Metrics --")
            print(Metrics(**{k: v for k, v in msg.items() if k != "type"}).report())
        elif kind == "done":
            print("\n-- Verdict --")
            print("LoopGuard interrupted the agent: loop detected." if msg["interrupted"]
                  else "Run finished without a fatal loop alert.")


if __name__ == "__main__":
    main()
