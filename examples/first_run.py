"""Exercise real LangChain tool callbacks without a model or API key.

From the repository root, install the framework extra and run:
    pip install -e ".[langgraph]"
    python examples/first_run.py
"""

from langchain_core.tools import tool

from loopguard import LoopGuardCallbackHandler, LoopGuardInterrupt


@tool
def search(query: str) -> str:
    """Search for a source; this offline example always returns no results."""
    return "no results"


def main() -> None:
    handler = LoopGuardCallbackHandler()
    handler.raise_error = True  # Let LangChain propagate callback exceptions.

    try:
        for _ in range(3):
            search.invoke({"query": "pricing page"}, config={"callbacks": [handler]})
    except LoopGuardInterrupt as exc:
        print("LoopGuard stopped the run:", exc)

    report = handler.report()
    print("Decision:", report["decisions"][-1]["action"])
    print("Tool calls:", report["metrics"]["tool_calls"])


if __name__ == "__main__":
    main()
