"""Check documentation structure and examples without network access.

Run from any directory with ``python scripts/check_docs.py``. Use ``--release``
to turn publication placeholders from warnings into errors.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit


@dataclass
class Finding:
    path: Path
    line: int
    message: str
    severity: str = "error"


PLACEHOLDER = re.compile(
    r"\[(?:DOCS URL TO ADD[^\]\n]*|SCREENSHOT NEEDED[^\]\n]*)\]"
    r"|\b(?:DOCS_URL|YOUTUBE_OR_DEMO_URL)\b"
)


def navigation_pages(value) -> list[str]:
    pages: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "pages" and isinstance(child, list):
                pages.extend(item for item in child if isinstance(item, str))
            pages.extend(navigation_pages(child))
    elif isinstance(value, list):
        for child in value:
            pages.extend(navigation_pages(child))
    return pages


def without_code(content: str) -> str:
    """Blank fenced blocks while retaining source line numbers."""
    lines = content.splitlines(keepends=True)
    marker = None
    for index, line in enumerate(lines):
        match = re.match(r"\s*(`{3,}|~{3,})(.*)$", line)
        if match:
            fence, tail = match.groups()
            if marker is None:
                marker = fence
            elif fence[0] == marker[0] and len(fence) >= len(marker) and not tail.strip():
                marker = None
            lines[index] = "\n"
        elif marker:
            lines[index] = "\n"
    return "".join(lines)


def heading_ids(content: str) -> set[str]:
    counts: Counter[str] = Counter()
    anchors: set[str] = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", without_code(content), re.M):
        heading = re.sub(r"<[^>]+>", "", heading)
        heading = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", heading)
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        suffix = f"-{counts[slug]}" if counts[slug] else ""
        anchors.add(slug + suffix)
        counts[slug] += 1
    anchors.update(re.findall(r"\bid=[\"']([^\"']+)[\"']", content))
    return anchors


def syntax_findings(path: Path, content: str) -> list[Finding]:
    findings: list[Finding] = []
    opening = None
    body: list[str] = []
    for number, line in enumerate(content.splitlines(), 1):
        if line.rstrip() != line:
            findings.append(Finding(path, number, "Remove trailing whitespace."))
        match = re.match(r"\s*(`{3,}|~{3,})(.*)$", line)
        if opening is None:
            if match:
                marker, language = match.groups()
                language = language.strip().split(maxsplit=1)[0] if language.strip() else ""
                opening = (marker, language, number)
                body = []
                if not language:
                    findings.append(Finding(path, number, "Add a language to the code fence."))
        elif match and match[1][0] == opening[0][0] and len(match[1]) >= len(opening[0]) and not match[2].strip():
            _, language, start = opening
            source = textwrap.dedent("\n".join(body))
            try:
                if language == "python":
                    ast.parse(source)
                elif language == "json":
                    json.loads(source)
            except (SyntaxError, ValueError) as exc:
                findings.append(Finding(path, start, f"Invalid {language} example: {exc}"))
            opening = None
        else:
            body.append(line)
    if opening:
        findings.append(Finding(path, opening[2], "Close the code fence."))
    return findings


def check_repository(root: Path, *, release: bool = False) -> list[Finding]:
    root = root.resolve()
    docs = root / "docs"
    config_path = docs / "docs.json"
    findings: list[Finding] = []
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        routes = navigation_pages(config["navigation"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [Finding(config_path, 1, f"Cannot read navigation: {exc}")]

    ignore_path = docs / ".mintignore"
    patterns = ignore_path.read_text(encoding="utf-8").splitlines() if ignore_path.exists() else []

    def ignored(path: Path) -> bool:
        relative = path.relative_to(docs).as_posix()
        return any(
            fnmatch.fnmatch(relative, pattern.strip())
            or fnmatch.fnmatch(path.name, pattern.strip())
            for pattern in patterns if pattern.strip() and not pattern.lstrip().startswith("#")
        )

    navigated: set[Path] = set()
    for route in routes:
        if urlsplit(route).scheme:
            continue
        candidate = docs / route.lstrip("/")
        choices = [candidate] if candidate.suffix else [candidate.with_suffix(".mdx"), candidate.with_suffix(".md")]
        page = next((item.resolve() for item in choices if item.is_file()), None)
        if page is None or not page.is_relative_to(docs):
            findings.append(Finding(config_path, 1, f"Navigation page does not exist inside docs/: {route}"))
        elif page in navigated:
            findings.append(Finding(config_path, 1, f"Duplicate navigation page: {route}"))
        else:
            navigated.add(page)

    files = sorted([*docs.rglob("*.mdx"), *docs.rglob("*.md")])
    files.extend(path for path in (root / "README.md", root / "CONTRIBUTING.md") if path.exists())
    public = {path.resolve() for path in files if path.is_relative_to(docs) and not ignored(path)}
    for page in sorted(public - navigated):
        findings.append(Finding(page, 1, "Add this public page to docs.json or exclude it in .mintignore."))

    for path in files:
        content = path.read_text(encoding="utf-8-sig")
        findings.extend(syntax_findings(path, content))
        if path.resolve() in public:
            frontmatter = re.match(r"\A---\n(.*?)\n---\n", content, re.S)
            if not frontmatter or not all(re.search(rf"^{field}:\s*\S", frontmatter[1], re.M) for field in ("title", "description")):
                findings.append(Finding(path, 1, "Add frontmatter with a title and description."))

        visible = without_code(content)
        if path.resolve() in public or path == root / "README.md":
            for match in PLACEHOLDER.finditer(visible):
                findings.append(Finding(path, visible.count("\n", 0, match.start()) + 1,
                                        "Resolve publication placeholder: " + match[0],
                                        "error" if release else "warning"))

        for match in re.finditer(r"!\[([^\]]*)\]\(", visible):
            if not match[1].strip():
                findings.append(Finding(path, visible.count("\n", 0, match.start()) + 1, "Add image alt text."))
        for match in re.finditer(r"<img\b[^>]*>", visible, re.I):
            if not re.search(r"\balt=[\"']\s*[^\s\"'][^\"']*[\"']", match[0]):
                findings.append(Finding(path, visible.count("\n", 0, match.start()) + 1, "Add image alt text."))

        destinations = list(re.finditer(r"\]\(<?([^\s)>]+)>?(?:\s+\"[^\"]*\")?\)", visible))
        destinations += list(re.finditer(r"(?:href|src)=[\"']([^\"']+)[\"']", visible))
        destinations += list(re.finditer(r"^\s*\[[^\]]+\]:\s*<?([^\s>]+)", visible, re.M))
        for match in destinations:
            destination = match[1]
            parsed = urlsplit(destination)
            if parsed.scheme or parsed.netloc:
                continue
            target = unquote(parsed.path)
            base = docs if path.resolve() in public else root
            linked = base / target.lstrip("/") if target.startswith("/") else path.parent / target if target else path
            if path.resolve() in public and not linked.is_file():
                candidates = [linked.with_suffix(".mdx"), linked.with_suffix(".md")] if linked.suffix in {"", ".md"} else []
                linked = next((item for item in candidates if item.is_file()), linked)
            line = visible.count("\n", 0, match.start()) + 1
            if not linked.resolve().is_relative_to(root) or not linked.is_file():
                findings.append(Finding(path, line, f"Missing local link or image: {destination}"))
            elif parsed.fragment and linked.suffix in {".md", ".mdx"}:
                if unquote(parsed.fragment) not in heading_ids(linked.read_text(encoding="utf-8-sig")):
                    findings.append(Finding(path, line, f"Missing heading: {destination}"))
    return findings


def escape_annotation(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true", help="fail on publication placeholders")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    findings = check_repository(root, release=args.release)
    for finding in findings:
        path = finding.path.relative_to(root).as_posix()
        print(f"::{finding.severity} file={escape_annotation(path)},line={finding.line}::{escape_annotation(finding.message)}")
    errors = sum(item.severity == "error" for item in findings)
    warnings = len(findings) - errors
    print(f"Documentation checks: {errors} error(s), {warnings} warning(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
