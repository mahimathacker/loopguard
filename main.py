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


NEEDS_KEY = {
    "semantic",
    "calc",
    "trap",
    "controlled_progress",
    "controlled_503",
    "controlled_401",
    "controlled_empty",
}


def _has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _has_google_key() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def _missing_key_message(name: str) -> str | None:
    model_provider = os.getenv("LOOPGUARD_MODEL_PROVIDER", "openai").strip().lower()
    embedding_provider = os.getenv("LOOPGUARD_EMBEDDING_PROVIDER", model_provider).strip().lower()

    if name == "semantic":
        if embedding_provider == "gemini" and not _has_google_key():
            return "Set GOOGLE_API_KEY or GEMINI_API_KEY to run the 'semantic' scenario with Gemini embeddings."
        if embedding_provider == "openai" and not _has_openai_key():
            return "Set OPENAI_API_KEY to run the 'semantic' scenario with OpenAI embeddings."
        return None

    if name in NEEDS_KEY:
        if model_provider == "gemini" and not _has_google_key():
            return f"Set GOOGLE_API_KEY or GEMINI_API_KEY to run the '{name}' scenario with Gemini."
        if model_provider == "openai" and not _has_openai_key():
            return f"Set OPENAI_API_KEY to run the '{name}' scenario with OpenAI."
    return None


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "exact"

    missing_key = _missing_key_message(name)
    if missing_key:
        print(missing_key)
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
            print(msg.get("reason") if msg["interrupted"] else "Run finished without a fatal loop alert.")


if __name__ == "__main__":
    main()
