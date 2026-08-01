import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from loopguard import LoopGuard
from loopguard.detectors import (
    LoopDetector,
    SemanticLoopDetector,
    StallDetector,
    StepBudgetDetector,
    ToolCallBudgetDetector,
)
from loopguard.embeddings import cosine
from loopguard.ingest import analyze_file, human_summary, json_report
from loopguard.metrics import Metrics
from loopguard.monitor import Monitor
from loopguard.tracer import Event


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRACE = ROOT / "examples" / "sample_trace.json"


class FakeAgent:
    def __init__(self, chunks):
        self.chunks = chunks
        self.config_seen = None

    def stream(self, initial, stream_mode, config):
        self.config_seen = config
        yield from self.chunks


class FakeEmbedder:
    def encode(self, text):
        text = text.lower()
        if "paris" in text or "weather" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]


class DetectorTests(unittest.TestCase):
    def test_loop_detector_interrupts_on_third_identical_tool_call(self):
        monitor = Monitor([LoopDetector(threshold=3)])

        for _ in range(3):
            monitor.observe(
                "agent",
                "decided: search({'query': 'pricing page'})",
                "tool:search(query=pricing page)",
                last_tool="search",
                last_args={"query": "pricing page"},
            )

        self.assertTrue(monitor.should_interrupt)
        self.assertEqual(monitor.alerts[-1].detector, "LoopDetector")
        self.assertEqual(monitor.alerts[-1].kind, "tool_loop")

    def test_loop_detector_allows_two_retries(self):
        monitor = Monitor([LoopDetector(threshold=3)])

        for _ in range(2):
            monitor.observe(
                "agent",
                "decided: search({'query': 'pricing page'})",
                "tool:search(query=pricing page)",
                last_tool="search",
                last_args={"query": "pricing page"},
            )

        self.assertFalse(monitor.should_interrupt)
        self.assertEqual(monitor.alerts, [])

    def test_stall_detector_warns_without_interrupting(self):
        monitor = Monitor([StallDetector(patience=4)])

        for _ in range(4):
            monitor.observe("tools", "observed: no results", "result:no results")

        self.assertFalse(monitor.should_interrupt)
        self.assertEqual(monitor.alerts[-1].detector, "StallDetector")
        self.assertFalse(monitor.alerts[-1].fatal)

    def test_step_budget_detector_interrupts_after_limit(self):
        monitor = Monitor([StepBudgetDetector(max_steps=2)])

        for i in range(3):
            monitor.observe("agent", f"step {i}", f"event:{i}")

        self.assertTrue(monitor.should_interrupt)
        self.assertEqual(monitor.alerts[-1].detector, "StepBudgetDetector")
        self.assertEqual(monitor.alerts[-1].kind, "step_budget")

    def test_tool_call_budget_detector_interrupts_after_limit(self):
        monitor = Monitor([ToolCallBudgetDetector(max_tool_calls=1)])

        monitor.observe("agent", "decided: one", "tool:one()")
        monitor.observe("tools", "observed: ok", "result:ok")
        monitor.observe("agent", "decided: two", "tool:two()")

        self.assertTrue(monitor.should_interrupt)
        self.assertEqual(monitor.alerts[-1].detector, "ToolCallBudgetDetector")
        self.assertEqual(monitor.alerts[-1].kind, "tool_call_budget")

    def test_semantic_loop_detector_uses_meaning_not_exact_text(self):
        detector = SemanticLoopDetector(embedder=FakeEmbedder(), threshold=0.8)
        monitor = Monitor([detector])

        for query in ["weather in Paris", "Paris weather today", "current weather in Paris"]:
            monitor.observe(
                "agent",
                f"decided: search({{'query': {query!r}}})",
                f"tool:search(query={query.lower()})",
                last_tool="search",
                last_args={"query": query},
            )

        self.assertTrue(monitor.should_interrupt)
        self.assertEqual(monitor.alerts[-1].detector, "SemanticLoopDetector")
        self.assertEqual(monitor.alerts[-1].kind, "semantic_loop")


class RunnerWrapperTests(unittest.TestCase):
    def test_loopguard_stream_wraps_live_agent_and_interrupts(self):
        chunks = []
        for _ in range(3):
            chunks.append({"agent": {"last_tool": "search", "last_args": {"query": "pricing page"}}})
            chunks.append({"tools": {"last_result": "no results"}})
        agent = FakeAgent(chunks)

        messages = list(LoopGuard(recursion_limit=17).stream(agent, {"goal": "test"}))

        self.assertEqual(agent.config_seen, {"recursion_limit": 17})
        self.assertIn("alert", [msg["type"] for msg in messages])
        self.assertEqual(messages[-1], {"type": "done", "interrupted": True})
        alert = next(msg for msg in messages if msg["type"] == "alert")
        self.assertEqual(alert["detector"], "LoopDetector")

    def test_loopguard_stream_interrupts_on_live_step_budget(self):
        chunks = [
            {"agent": {"last_tool": "search", "last_args": {"query": "one"}}},
            {"tools": {"last_result": "one"}},
            {"agent": {"last_tool": "search", "last_args": {"query": "two"}}},
        ]

        messages = list(LoopGuard(max_steps=2).stream(FakeAgent(chunks), {"goal": "test"}))

        alert = next(msg for msg in messages if msg["type"] == "alert")
        self.assertEqual(alert["detector"], "StepBudgetDetector")
        self.assertEqual(alert["kind"], "step_budget")
        self.assertEqual(messages[-1], {"type": "done", "interrupted": True})

    def test_loopguard_stream_interrupts_on_live_tool_call_budget(self):
        chunks = [
            {"agent": {"last_tool": "search", "last_args": {"query": "one"}}},
            {"tools": {"last_result": "one"}},
            {"agent": {"last_tool": "search", "last_args": {"query": "two"}}},
        ]

        messages = list(LoopGuard(max_tool_calls=1).stream(FakeAgent(chunks), {"goal": "test"}))

        alert = next(msg for msg in messages if msg["type"] == "alert")
        self.assertEqual(alert["detector"], "ToolCallBudgetDetector")
        self.assertEqual(alert["kind"], "tool_call_budget")
        self.assertEqual(messages[-1], {"type": "done", "interrupted": True})

    def test_loopguard_from_config_uses_live_budgets(self):
        config = "\n".join(
            [
                "live:",
                "  max_steps: 2",
                "  max_tool_calls: 5",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loopguard.yml"
            path.write_text(config)
            guard = LoopGuard.from_config(str(path))

        self.assertEqual(guard.max_steps, 2)
        self.assertEqual(guard.max_tool_calls, 5)
        self.assertIn("StepBudgetDetector", [type(item).__name__ for item in guard.detectors])
        self.assertIn("ToolCallBudgetDetector", [type(item).__name__ for item in guard.detectors])


class MetricsTests(unittest.TestCase):
    def test_metrics_count_tool_repeats(self):
        events = [
            Event(1, "agent", "decided", "tool:search(query=pricing page)"),
            Event(2, "tools", "observed", "result:no results"),
            Event(3, "agent", "decided", "tool:search(query=pricing page)"),
            Event(4, "agent", "decided", "tool:summarize(text=docs)"),
        ]

        metrics = Metrics.from_events(events)

        self.assertEqual(metrics.total_steps, 4)
        self.assertEqual(metrics.by_node, {"agent": 3, "tools": 1})
        self.assertEqual(metrics.tool_calls, 3)
        self.assertEqual(metrics.distinct_tool_calls, 2)
        self.assertEqual(metrics.most_common_action, ("tool:search(query=pricing page)", 2))
        self.assertAlmostEqual(metrics.repeat_rate, 1 / 3)


class IngestTests(unittest.TestCase):
    def test_sample_trace_report_statuses(self):
        reports = analyze_file(SAMPLE_TRACE)
        report = json_report(SAMPLE_TRACE, reports)

        self.assertEqual(report["runs_analyzed"], 3)
        self.assertEqual(
            report["summary"],
            {"clean": 1, "looping": 1, "stalled": 1, "with_alerts": 2},
        )
        self.assertEqual([run["status"] for run in report["runs"]], ["looping", "stalled", "clean"])
        self.assertEqual([run["tool_calls"] for run in report["runs"]], [4, 4, 3])
        self.assertEqual(report["runs"][0]["alerts"][0]["type"], "loop")
        self.assertEqual(report["runs"][1]["alerts"][0]["type"], "warn")

    def test_human_summary_separates_looping_from_stalled(self):
        summary = human_summary(analyze_file(SAMPLE_TRACE))

        self.assertIn("1 clean", summary)
        self.assertIn("1 looping", summary)
        self.assertIn("1 stalled/warned", summary)
        self.assertIn("2 run(s) had alerts", summary)

    def test_json_cli_outputs_parseable_json(self):
        proc = subprocess.run(
            [sys.executable, "-m", "loopguard.ingest", str(SAMPLE_TRACE), "--json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        parsed = json.loads(proc.stdout)
        self.assertEqual(parsed["summary"]["looping"], 1)
        self.assertEqual(proc.stderr, "")


class CheckCliTests(unittest.TestCase):
    def test_check_fails_on_looping_by_default(self):
        proc = subprocess.run(
            [sys.executable, "-m", "loopguard.check", str(SAMPLE_TRACE)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

        self.assertEqual(proc.returncode, 1)
        self.assertIn("Check: FAIL", proc.stdout)
        self.assertIn("matched fail-on: looping", proc.stdout)
        self.assertEqual(proc.stderr, "")

    def test_check_passes_when_only_looping_fails_and_trace_is_clean(self):
        clean_trace = {
            "run_id": "clean-only",
            "steps": [
                {
                    "agent": "research_agent",
                    "tool": "web_search",
                    "args": {"query": "pricing page"},
                    "output": "found results",
                },
                {
                    "agent": "writer_agent",
                    "tool": "draft_report",
                    "args": {"topic": "pricing"},
                    "output": "draft ready",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clean.json"
            path.write_text(json.dumps(clean_trace))
            proc = subprocess.run(
                [sys.executable, "-m", "loopguard.check", str(path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

        self.assertEqual(proc.returncode, 0)
        self.assertIn("Check: PASS", proc.stdout)
        self.assertEqual(proc.stderr, "")

    def test_check_json_reports_matched_failures(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "loopguard.check",
                str(SAMPLE_TRACE),
                "--fail-on",
                "looping",
                "--fail-on",
                "stalled",
                "--json",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

        parsed = json.loads(proc.stdout)
        self.assertEqual(proc.returncode, 1)
        self.assertTrue(parsed["failed"])
        self.assertEqual(parsed["matched_failures"], ["looping", "stalled"])
        self.assertEqual(parsed["summary"]["with_alerts"], 2)
        self.assertEqual(proc.stderr, "")

    def test_check_bad_input_exits_two(self):
        proc = subprocess.run(
            [sys.executable, "-m", "loopguard.check", str(ROOT / "missing.json"), "--json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertIn("LoopGuard check error", proc.stderr)

    def test_check_fails_on_step_budget(self):
        clean_trace = {
            "run_id": "too-many-steps",
            "steps": [
                {"agent": "a", "tool": "one", "args": {}, "output": "ok"},
                {"agent": "a", "tool": "two", "args": {}, "output": "ok"},
                {"agent": "a", "tool": "three", "args": {}, "output": "ok"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "steps.json"
            path.write_text(json.dumps(clean_trace))
            proc = subprocess.run(
                [sys.executable, "-m", "loopguard.check", str(path), "--max-steps", "2", "--json"],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

        parsed = json.loads(proc.stdout)
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(parsed["matched_failures"], [])
        self.assertEqual(parsed["budget_violations"][0]["type"], "max_steps")
        self.assertEqual(parsed["budget_violations"][0]["actual"], 3)

    def test_check_fails_on_tool_call_budget(self):
        clean_trace = {
            "run_id": "too-many-tools",
            "steps": [
                {"agent": "a", "tool": "one", "args": {}, "output": "ok"},
                {"agent": "a", "tool": "two", "args": {}, "output": "done"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tools.json"
            path.write_text(json.dumps(clean_trace))
            proc = subprocess.run(
                [sys.executable, "-m", "loopguard.check", str(path), "--max-tool-calls", "1"],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

        self.assertEqual(proc.returncode, 1)
        self.assertIn("Budget violations:", proc.stdout)
        self.assertIn("max_tool_calls", proc.stdout)
        self.assertIn("Check: FAIL", proc.stdout)

    def test_check_uses_simple_yaml_config(self):
        config = "\n".join(
            [
                "check:",
                "  fail_on:",
                "    - stalled",
                "  max_steps: 10",
                "  max_tool_calls: 10",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loopguard.yml"
            path.write_text(config)
            proc = subprocess.run(
                [sys.executable, "-m", "loopguard.check", str(SAMPLE_TRACE), "--config", str(path), "--json"],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

        parsed = json.loads(proc.stdout)
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(parsed["fail_on"], ["stalled"])
        self.assertEqual(parsed["matched_failures"], ["stalled"])
        self.assertEqual(parsed["budgets"], {"max_steps": 10, "max_tool_calls": 10})
        self.assertEqual(parsed["budget_violations"], [])

    def test_check_cli_overrides_config(self):
        config = "\n".join(
            [
                "fail_on:",
                "  - stalled",
                "max_steps: 2",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loopguard.yml"
            path.write_text(config)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "loopguard.check",
                    str(SAMPLE_TRACE),
                    "--config",
                    str(path),
                    "--fail-on",
                    "looping",
                    "--max-steps",
                    "10",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

        parsed = json.loads(proc.stdout)
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(parsed["fail_on"], ["looping"])
        self.assertEqual(parsed["matched_failures"], ["looping"])
        self.assertEqual(parsed["budgets"]["max_steps"], 10)
        self.assertEqual(parsed["budget_violations"], [])

    def test_check_bad_config_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loopguard.yml"
            path.write_text("fail_on: nope")
            proc = subprocess.run(
                [sys.executable, "-m", "loopguard.check", str(SAMPLE_TRACE), "--config", str(path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

        self.assertEqual(proc.returncode, 2)
        self.assertIn("invalid fail_on", proc.stderr)


class EmbeddingMathTests(unittest.TestCase):
    def test_cosine_similarity(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([1, 0], [1]), 0.0)


if __name__ == "__main__":
    unittest.main()
