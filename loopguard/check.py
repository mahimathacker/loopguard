"""CI-friendly saved-trace checks.

This module wraps the offline ingest report with pass/fail semantics. It is intentionally
available as ``python -m loopguard.check`` until the project adds packaging metadata for
a real ``loopguard check`` console command.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
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


def check_report(
    reports_by_source: list[tuple[Path, list[TraceReport]]],
    fail_on: set[str],
) -> dict:
    """Build a machine-readable check report."""
    summary = aggregate_counts(reports_by_source)
    matched = matched_failures(summary, fail_on)
    return {
        "summary": summary,
        "fail_on": sorted(fail_on),
        "failed": bool(matched),
        "matched_failures": matched,
        "sources": [
            json_report(source, reports) for source, reports in reports_by_source
        ],
    }


def run_check(paths: list[str], fail_on: set[str]) -> dict:
    """Analyze paths and return a combined check report."""
    reports_by_source = [(Path(path), analyze_file(path)) for path in paths]
    return check_report(reports_by_source, fail_on)


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
    lines.append(f"Check: {status} (matched fail-on: {matched})")
    return "\n".join(lines)


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
    parser.add_argument("--json", action="store_true", help="print a machine-readable report")
    args = parser.parse_args(argv)

    fail_on = set(args.fail_on or ["looping"])
    try:
        reports_by_source = [(Path(path), analyze_file(path)) for path in args.paths]
        report = check_report(reports_by_source, fail_on)
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
