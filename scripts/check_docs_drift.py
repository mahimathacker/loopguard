"""Warn when developer-facing files change without a documentation update.

This is a conservative file-based reminder, not semantic API compatibility analysis.
It never rewrites files and an advisory finding does not fail the command.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
from pathlib import Path


SURFACES = {
    "Python API and detection behavior": ("loopguard/*.py",),
    "CLI commands": ("loopguard/check.py", "loopguard/ingest.py", "loopguard/evals.py"),
    "Configuration": ("loopguard/config.py", "loopguard/guard.py", "loopguard/langgraph.py"),
    "Scenarios and providers": ("loopguard/scenarios.py", "loopguard/agent.py", "loopguard/embeddings.py", "main.py", ".env.example"),
    "Demo server and frontend integration": ("server.py", "ui/app/page.tsx", "ui/app/types.ts"),
    "Package version, Python support, and extras": ("pyproject.toml",),
}


def review_changes(paths: list[str]) -> tuple[dict[str, list[str]], bool]:
    surfaces = {
        category: sorted({path for path in paths if any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)})
        for category, patterns in SURFACES.items()
    }
    surfaces = {category: matches for category, matches in surfaces.items() if matches}
    docs_changed = any(path in {"README.md", "CONTRIBUTING.md"} or
                       (path.startswith("docs/") and Path(path).suffix in {".md", ".mdx"})
                       for path in paths)
    return surfaces, docs_changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base commit or ref to compare")
    parser.add_argument("--head", default="HEAD", help="head commit or ref to compare")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]

    def commit(ref: str) -> str:
        return subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"], cwd=root, text=True).strip()

    base, head = commit(args.base), commit(args.head)
    output = subprocess.check_output(["git", "diff", "--no-renames", "--name-only", "--diff-filter=ACDMRTUXB", "-z", f"{base}...{head}"], cwd=root)
    paths = output.decode("utf-8").strip("\0").split("\0") if output else []
    surfaces, docs_changed = review_changes(paths)
    lines = ["## Documentation drift review", ""]
    if not surfaces:
        lines.append("No watched developer-facing files changed.")
    else:
        for category, matches in surfaces.items():
            lines.append(f"- {category}: " + ", ".join(f"`{path}`" for path in matches))
        if docs_changed:
            lines.extend(["", "Documentation changed too. Review whether it covers each affected surface."])
        else:
            message = "Developer-facing files changed without README/docs changes. Check public APIs, CLI flags, configuration, scenarios, providers, and version details for documentation updates."
            print("::warning title=Potential documentation drift::" + message)
            lines.extend(["", message])
        lines.extend(["", "This is a file-based reminder. Documentation edits can be unrelated; a passing check does not prove the docs are accurate."])
    report = "\n".join(lines) + "\n"
    print(report)
    if summary := os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
