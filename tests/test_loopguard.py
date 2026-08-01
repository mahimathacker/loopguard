import json
import subprocess
import sys
import unittest
from pathlib import Path

from loopguard import LoopGuard
from loopguard.detectors import LoopDetector, SemanticLoopDetector, StallDetector
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


class EmbeddingMathTests(unittest.TestCase):
    def test_cosine_similarity(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([1, 0], [1]), 0.0)


if __name__ == "__main__":
    unittest.main()
