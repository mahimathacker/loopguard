"""Exercise the README and getting-started callback examples without model calls."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

from langchain_core.tools import tool


ROOT = Path(__file__).resolve().parents[1]


def python_blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


class FixtureAgent:
    """A deterministic invocation loop using real LangChain tool callbacks."""

    def __init__(self, results: list[str]):
        self.results = results

    def invoke(self, payload, config):
        for output in self.results:
            @tool
            def check_status(job_id: str) -> str:
                """Read a scripted job status without a network call."""
                return output

            check_status.invoke({"job_id": "42"}, config=config)
        return self.results[-1]


def verify() -> None:
    readme = ROOT / "README.md"
    code = python_blocks(readme)[0]
    for results, expected_stop in [(["pending", "processing", "completed"], False), (["no results"] * 3, True)]:
        namespace = {"agent": FixtureAgent(results), "my_input": {}}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, str(readme), "exec"), namespace)
        report = namespace["handler"].report()
        assert report["interrupted"] is expected_stop, "README stop/continue behavior changed."
        assert report["metrics"]["tool_calls"] == 3

    getting_started = ROOT / "docs/getting-started.mdx"
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(python_blocks(getting_started)[0], str(getting_started), "exec"), namespace)
    assert namespace["report"]["decisions"][-1]["action"] == "stop"
    assert namespace["report"]["metrics"]["tool_calls"] == 3

    # Exercise the documented handler replacement for observe-only mode too.
    exec(python_blocks(getting_started)[1], namespace)
    handler = namespace["handler"]
    FixtureAgent(["no results"] * 3).invoke({}, {"callbacks": [handler]})
    assert handler.report()["interrupted"] is True
    assert handler.decisions[-1]["action"] == "stop"
    print("README and getting-started examples: continue, stop, and observe-only paths passed.")


if __name__ == "__main__":
    verify()
