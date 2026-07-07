"""OFFLINE TRACE INGESTION - the replay adapter.

This is the bridge that lets LoopGuard analyze *someone else's* agent without being
plugged into their runtime. They export a trace as JSON; this module maps each step
onto the same Monitor.observe() calls the live runner makes, so every existing detector
works unchanged.

The whole point: the detectors are agent-agnostic already (they only read Events). The
only thing missing for an external user was an input port. This is that port - the
lowest-effort integration, needing no SDK and no access to their production system.

External trace shape we accept (forgiving about field names)::

    {
      "run_id": "abc",
      "steps": [
        {"step": 1, "agent": "research_agent", "tool": "web_search",
         "args": {"query": "pricing page"}, "output": "...", "caller": "supervisor"}
      ]
    }

A file may hold a single run (object) or many runs (list of objects).
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .detectors import Alert, Detector, LoopDetector, StallDetector
from .monitor import Monitor


# Each external field can arrive under several names depending on whose agent emitted
# the trace. The adapter's job is to absorb that variation, so we try aliases in order.
_TOOL_KEYS = ("tool", "tool_name", "name", "function")
_ARGS_KEYS = ("args", "arguments", "input", "parameters", "params")
_OUTPUT_KEYS = ("output", "result", "observation", "response", "content")
_AGENT_KEYS = ("agent", "node", "actor", "role")
_CALLER_KEYS = ("caller", "parent", "parent_agent", "invoked_by")


def _first(d: dict, keys: tuple[str, ...], default: Any = None) -> Any:
    """Return the first present, non-None value among the candidate keys."""
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return default


def _tool_signature(tool: str, args: dict) -> str:
    """Normalize a tool call without importing the live LangGraph runner."""
    norm = ", ".join(f"{k}={str(v).strip().lower()}" for k, v in sorted(args.items()))
    return f"tool:{tool}({norm})"


@dataclass
class TraceReport:
    """The verdict for one external run after replaying it through the detectors."""

    run_id: str
    steps: int                                   # external steps in the trace
    events: int                                  # internal Events produced (>= steps)
    alerts: list[Alert] = field(default_factory=list)

    @property
    def looped(self) -> bool:
        """Did any fatal detector (a real loop) fire on this run?"""
        return any(a.fatal for a in self.alerts)

    def summary(self) -> str:
        head = f"run {self.run_id}: {self.steps} steps -> {self.events} events"
        if not self.alerts:
            return f"  [CLEAN] {head} - no loops or stalls detected"
        label = "LOOP" if self.status == "looping" else "STALL"
        lines = [f"  [{label}] {head}"]
        for a in self.alerts:
            tag = "LOOP" if a.fatal else "warn"
            lines.append(f"      - {tag} ({a.detector}): {a.message}")
        return "\n".join(lines)

    @property
    def status(self) -> str:
        """Single report status, with fatal loops taking priority over warnings."""
        if self.looped:
            return "looping"
        if self.alerts:
            return "stalled"
        return "clean"


def _steps_to_observations(steps: list[dict]) -> list[tuple[str, str, str, dict]]:
    """Map external steps onto (node, action, signature, payload) observations.

    Each step with a tool call yields up to TWO observations - the decision and the
    result - mirroring how the live runner interprets a real agent step, so that
    LoopDetector (reads 'tool:' events) and StallDetector (reads 'result:' events) both
    have the events they need.
    """
    obs: list[tuple[str, str, str, dict]] = []
    for raw in steps:
        agent = str(_first(raw, _AGENT_KEYS, "agent"))
        tool = _first(raw, _TOOL_KEYS)
        args = _first(raw, _ARGS_KEYS, {}) or {}
        if not isinstance(args, dict):
            args = {"value": args}

        if tool is not None:
            sig = _tool_signature(str(tool), args)
            payload = {"last_tool": str(tool), "last_args": args}
            caller = _first(raw, _CALLER_KEYS)
            if caller is not None:
                payload["caller"] = caller  # carried for a future handoff-loop detector
            obs.append((agent, f"decided: {tool}({args})", sig, payload))

        output = _first(raw, _OUTPUT_KEYS)
        if output is not None:
            text = str(output).strip().replace("\n", " ")[:160]
            obs.append((agent, f"observed: {text}", f"result:{text}", {"last_result": text}))
    return obs


def analyze_trace(trace: dict, detectors: list[Detector] | None = None) -> TraceReport:
    """Replay one external trace through a fresh Monitor and return its report."""
    detectors = detectors if detectors is not None else [LoopDetector(), StallDetector()]
    for d in detectors:
        d.reset()

    monitor = Monitor(detectors=detectors)
    steps = trace.get("steps", [])
    observations = _steps_to_observations(steps)
    for node, action, sig, payload in observations:
        monitor.observe(node, action, sig, **payload)

    return TraceReport(
        run_id=str(trace.get("run_id", "unknown")),
        steps=len(steps),
        events=len(observations),
        alerts=list(monitor.alerts),
    )


def analyze_file(path: str | Path, detectors: list[Detector] | None = None) -> list[TraceReport]:
    """Load a JSON file holding one run (object) or many (list) and analyze each."""
    data = json.loads(Path(path).read_text())
    runs = data if isinstance(data, list) else [data]
    return [analyze_trace(run, detectors) for run in runs]


def _alert_type(alert: Alert) -> str:
    """Small report vocabulary: fatal detector alerts are loops; non-fatal alerts are warns."""
    return "loop" if alert.fatal else "warn"


def _report_alerts(alerts: list[Alert]) -> list[dict]:
    """Return stable, non-redundant alerts for CLI/CI output."""
    seen: set[tuple[str, str, bool]] = set()
    out: list[dict] = []
    for alert in alerts:
        key = (alert.detector, alert.kind, alert.fatal)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "type": _alert_type(alert),
                "detector": alert.detector,
                "message": alert.message,
            }
        )
    return out


def json_report(path: str | Path, reports: list[TraceReport]) -> dict:
    """Build the machine-readable stuck-run report."""
    runs = [
        {
            "run_id": report.run_id,
            "status": report.status,
            "steps": report.steps,
            "events": report.events,
            "alerts": _report_alerts(report.alerts),
        }
        for report in reports
    ]
    return {
        "source": str(path),
        "runs_analyzed": len(reports),
        "summary": status_counts(reports),
        "runs": runs,
    }


def status_counts(reports: list[TraceReport]) -> dict[str, int]:
    """Count report statuses for human and JSON summaries."""
    return {
        "clean": sum(1 for report in reports if report.status == "clean"),
        "looping": sum(1 for report in reports if report.status == "looping"),
        "stalled": sum(1 for report in reports if report.status == "stalled"),
        "with_alerts": sum(1 for report in reports if report.alerts),
    }


def human_summary(reports: list[TraceReport]) -> str:
    """Readable status summary for the text CLI output."""
    counts = status_counts(reports)
    return "\n".join(
        [
            "Summary:",
            f"  {len(reports)} run(s) analyzed",
            f"  {counts['clean']} clean",
            f"  {counts['looping']} looping",
            f"  {counts['stalled']} stalled/warned",
            f"  {counts['with_alerts']} run(s) had alerts",
        ]
    )


if __name__ == "__main__":
    # CLI:  python -m loopguard.ingest <trace.json>
    # Uses offline detectors only (no API key needed). Add SemanticLoopDetector yourself
    # when you want paraphrase-loop catching and have OPENAI_API_KEY set.
    parser = ArgumentParser(description="Analyze saved agent traces for loops and stalls.")
    parser.add_argument("path", help="JSON trace file to analyze")
    parser.add_argument("--json", action="store_true", help="print a machine-readable report")
    args = parser.parse_args()

    reports = analyze_file(args.path)
    if args.json:
        print(json.dumps(json_report(args.path, reports), indent=2))
        raise SystemExit(0)

    print(f"Analyzed {len(reports)} run(s) from {args.path}:\n")
    for r in reports:
        print(r.summary())
    print(f"\n{human_summary(reports)}")
