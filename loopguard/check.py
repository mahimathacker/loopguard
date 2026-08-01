"""CI-friendly saved-trace checks.

This module wraps the offline ingest report with pass/fail semantics. It is intentionally
available as ``python -m loopguard.check`` until the project adds packaging metadata for
a real ``loopguard check`` console command.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

from .ingest import TraceReport, analyze_file, human_summary, json_report, status_counts

FAIL_CHOICES = ("looping", "stalled", "alerts")


def aggregate_counts(reports_by_source: list[tuple[Path, list[TraceReport]]]) -> dict[str, int]:
    """Combine status counts across all analyzed trace files."""
    all_reports = [report for _, reports in reports_by_source for report in reports]
    return status_counts(all_reports)


def matched_failures(summary: dict[str, int], fail_on: set[str]) -> list[str]:
    """Return the fail-on rules matched by this summary."""
    matched: list[str] = []
    for status in FAIL_CHOICES:
        key = "with_alerts" if status == "alerts" else status
        if status in fail_on and summary[key] > 0:
            matched.append(status)
    return matched


def budget_violations(
    reports_by_source: list[tuple[Path, list[TraceReport]]],
    max_steps: int | None = None,
    max_tool_calls: int | None = None,
) -> list[dict]:
    """Return per-run budget violations."""
    violations: list[dict] = []
    for source, reports in reports_by_source:
        for report in reports:
            if max_steps is not None and report.steps > max_steps:
                violations.append(
                    {
                        "type": "max_steps",
                        "source": str(source),
                        "run_id": report.run_id,
                        "actual": report.steps,
                        "limit": max_steps,
                    }
                )
            if max_tool_calls is not None and report.tool_calls > max_tool_calls:
                violations.append(
                    {
                        "type": "max_tool_calls",
                        "source": str(source),
                        "run_id": report.run_id,
                        "actual": report.tool_calls,
                        "limit": max_tool_calls,
                    }
                )
    return violations


def check_report(
    reports_by_source: list[tuple[Path, list[TraceReport]]],
    fail_on: set[str],
    max_steps: int | None = None,
    max_tool_calls: int | None = None,
) -> dict:
    """Build a machine-readable check report."""
    summary = aggregate_counts(reports_by_source)
    matched = matched_failures(summary, fail_on)
    violations = budget_violations(reports_by_source, max_steps, max_tool_calls)
    return {
        "summary": summary,
        "fail_on": sorted(fail_on),
        "budgets": {
            "max_steps": max_steps,
            "max_tool_calls": max_tool_calls,
        },
        "failed": bool(matched or violations),
        "matched_failures": matched,
        "budget_violations": violations,
        "sources": [
            json_report(source, reports) for source, reports in reports_by_source
        ],
    }


def run_check(
    paths: list[str],
    fail_on: set[str],
    max_steps: int | None = None,
    max_tool_calls: int | None = None,
) -> dict:
    """Analyze paths and return a combined check report."""
    reports_by_source = [(Path(path), analyze_file(path)) for path in paths]
    return check_report(reports_by_source, fail_on, max_steps, max_tool_calls)


def render_text(reports_by_source: list[tuple[Path, list[TraceReport]]], report: dict) -> str:
    """Render a readable check summary for local/CI logs."""
    lines = [f"LoopGuard check: {len(reports_by_source)} trace file(s)\n"]
    for source, reports in reports_by_source:
        path = str(source)
        lines.append(f"Analyzed {len(reports)} run(s) from {path}:\n")
        lines.extend(report.summary() for report in reports)
        lines.append("")
        lines.append(human_summary(reports))
        lines.append("")

    status = "FAIL" if report["failed"] else "PASS"
    matched = ", ".join(report["matched_failures"]) or "none"
    budget_types = sorted({item["type"] for item in report["budget_violations"]})
    budgets = ", ".join(budget_types) or "none"
    if report["budget_violations"]:
        lines.append("Budget violations:")
        for item in report["budget_violations"]:
            lines.append(
                f"  - {item['type']}: run {item['run_id']} used "
                f"{item['actual']} > {item['limit']} ({item['source']})"
            )
        lines.append("")
    lines.append(f"Check: {status} (matched fail-on: {matched}; budgets: {budgets})")
    return "\n".join(lines)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise ArgumentTypeError("must be >= 1")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Fail when saved agent traces are stuck.")
    parser.add_argument("paths", nargs="+", help="JSON trace file(s) to analyze")
    parser.add_argument(
        "--fail-on",
        action="append",
        choices=FAIL_CHOICES,
        default=None,
        help="status that should fail the check; repeatable. Default: looping",
    )
    parser.add_argument("--max-steps", type=positive_int, help="fail if any run has more steps")
    parser.add_argument(
        "--max-tool-calls",
        type=positive_int,
        help="fail if any run has more tool-call observations",
    )
    parser.add_argument("--json", action="store_true", help="print a machine-readable report")
    args = parser.parse_args(argv)

    fail_on = set(args.fail_on or ["looping"])
    try:
        reports_by_source = [(Path(path), analyze_file(path)) for path in args.paths]
        report = check_report(
            reports_by_source,
            fail_on,
            max_steps=args.max_steps,
            max_tool_calls=args.max_tool_calls,
        )
    except Exception as exc:  # noqa: BLE001 - this is a CLI boundary
        print(f"LoopGuard check error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(reports_by_source, report))

    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
