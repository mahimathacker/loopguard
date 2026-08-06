import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from loopguard import LoopGuard
from loopguard.detectors import (
    HandoffLoopDetector,
    LoopDetector,
    CycleDetector,
    ProgressDetector,
    RepeatedFailureDetector,
    SemanticLoopDetector,
    StallDetector,
    StepBudgetDetector,
    ToolCallBudgetDetector,
)
from loopguard.embeddings import cosine
from loopguard.ingest import analyze_file, analyze_trace, human_summary, json_report
from loopguard.langgraph import LoopGuardCallbackHandler, LoopGuardInterrupt
from loopguard.metrics import Metrics
from loopguard.monitor import Monitor
from loopguard.policy import PolicyEngine
from loopguard.signals import DetectionSignal, GuardAction
from loopguard.tracer import Event


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRACE = ROOT / "examples" / "sample_trace.json"
V02_TRACE = ROOT / "examples" / "v02_labeled_traces.json"


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
    def test_loop_detector_warns_on_third_identical_tool_call_without_progress_evidence(self):
        monitor = Monitor([LoopDetector(threshold=3)])

        for _ in range(3):
            monitor.observe(
                "agent",
                "decided: search({'query': 'pricing page'})",
                "tool:search(query=pricing page)",
                last_tool="search",
                last_args={"query": "pricing page"},
            )

        self.assertFalse(monitor.should_interrupt)
        self.assertEqual(monitor.signals[-1].detector, "LoopDetector")
        self.assertEqual(monitor.signals[-1].kind, "repeated_tool_call")
        self.assertEqual(monitor.alerts[-1].detector, "LoopDetector")
        self.assertEqual(monitor.alerts[-1].kind, "repeated_tool_call")
        self.assertFalse(monitor.alerts[-1].fatal)

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

    def test_stall_detector_can_interrupt_when_configured(self):
        monitor = Monitor([StallDetector(patience=2, fatal=True)])

        for _ in range(2):
            monitor.observe("tools", "observed: no results", "result:no results")

        self.assertTrue(monitor.should_interrupt)
        self.assertEqual(monitor.alerts[-1].detector, "StallDetector")
        self.assertTrue(monitor.alerts[-1].fatal)

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

    def test_handoff_loop_detector_interrupts_on_repeated_cycle(self):
        monitor = Monitor([HandoffLoopDetector(repeats=2, window=8)])

        for caller, agent in [
            ("planner", "researcher"),
            ("researcher", "reviewer"),
            ("reviewer", "planner"),
            ("planner", "researcher"),
            ("researcher", "reviewer"),
            ("reviewer", "planner"),
        ]:
            monitor.observe(
                agent,
                f"{caller} handed off to {agent}",
                f"handoff:{caller}->{agent}",
                caller=caller,
            )

        self.assertTrue(monitor.should_interrupt)
        self.assertEqual(monitor.alerts[-1].detector, "HandoffLoopDetector")
        self.assertEqual(monitor.alerts[-1].kind, "handoff_loop")

    def test_handoff_loop_detector_ignores_repeated_one_way_handoff(self):
        monitor = Monitor([HandoffLoopDetector(repeats=2, window=8)])

        for _ in range(4):
            monitor.observe(
                "researcher",
                "supervisor handed off to researcher",
                "handoff:supervisor->researcher",
                caller="supervisor",
            )

        self.assertFalse(monitor.should_interrupt)
        self.assertEqual(monitor.alerts, [])

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

        self.assertFalse(monitor.should_interrupt)
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

    def test_loopguard_stream_emits_signals_and_decisions(self):
        chunks = [
            {"agent": {"last_tool": "search", "last_args": {"query": "pricing page"}}},
            {"tools": {"last_result": "no results"}},
            {"agent": {"last_tool": "search", "last_args": {"query": "pricing page"}}},
            {"tools": {"last_result": "no results"}},
            {"agent": {"last_tool": "search", "last_args": {"query": "pricing page"}}},
            {"tools": {"last_result": "no results"}},
        ]

        messages = list(LoopGuard().stream(FakeAgent(chunks), {"goal": "test"}))

        self.assertIn("signal", [msg["type"] for msg in messages])
        self.assertIn("decision", [msg["type"] for msg in messages])
        self.assertEqual(
            [msg for msg in messages if msg["type"] == "decision"][-1]["action"],
            "stop",
        )

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

    def test_loopguard_from_config_uses_live_detector_policy(self):
        config = "\n".join(
            [
                "live:",
                "  exact_threshold: 5",
                "  exact_window: 20",
                "  exact_fatal: false",
                "  stall_patience: 2",
                "  stall_fatal: true",
                "  handoff_repeats: 3",
                "  handoff_window: 18",
                "  handoff_max_cycle_length: 4",
                "  handoff_fatal: false",
                "  semantic: true",
                "  semantic_threshold: 0.75",
                "  semantic_window: 8",
                "  semantic_min_repeats: 4",
                "  semantic_fatal: false",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loopguard.yml"
            path.write_text(config)
            guard = LoopGuard.from_config(str(path))

        loop = next(item for item in guard.detectors if isinstance(item, LoopDetector))
        stall = next(item for item in guard.detectors if isinstance(item, StallDetector))
        handoff = next(item for item in guard.detectors if isinstance(item, HandoffLoopDetector))
        semantic = next(item for item in guard.detectors if isinstance(item, SemanticLoopDetector))

        self.assertEqual(loop.threshold, 5)
        self.assertEqual(loop.window, 20)
        self.assertFalse(loop.fatal)
        self.assertEqual(stall.patience, 2)
        self.assertTrue(stall.fatal)
        self.assertEqual(handoff.repeats, 3)
        self.assertEqual(handoff.window, 18)
        self.assertEqual(handoff.max_cycle_length, 4)
        self.assertFalse(handoff.fatal)
        self.assertEqual(semantic.threshold, 0.75)
        self.assertEqual(semantic.window, 8)
        self.assertEqual(semantic.min_repeats, 4)
        self.assertFalse(semantic.fatal)


class CallbackHandlerTests(unittest.TestCase):
    def test_callback_handler_is_public_export(self):
        from loopguard import LoopGuardCallbackHandler as PublicHandler

        self.assertIs(PublicHandler, LoopGuardCallbackHandler)

    def test_callback_handler_records_tool_events(self):
        handler = LoopGuardCallbackHandler(interrupt_on_fatal=False)

        handler.on_tool_start({"name": "search"}, "pricing page")
        handler.on_tool_end("found results")

        self.assertEqual(
            [msg["type"] for msg in handler.messages],
            ["event", "decision", "event", "decision"],
        )
        self.assertEqual(handler.events[0]["signature"], "tool:search(input=pricing page)")
        self.assertEqual(handler.events[1]["signature"], "result:found results")
        self.assertEqual(handler.metrics()["tool_calls"], 1)

    def test_callback_handler_raises_on_fatal_loop(self):
        handler = LoopGuardCallbackHandler(interrupt_on_fatal=True)

        for _ in range(2):
            handler.on_tool_start({"name": "search"}, {"query": "pricing page"})
            handler.on_tool_end("no results")
        handler.on_tool_start({"name": "search"}, {"query": "pricing page"})
        with self.assertRaises(LoopGuardInterrupt) as caught:
            handler.on_tool_end("no results")

        self.assertEqual(caught.exception.alert.detector, "ProgressDetector")
        self.assertIn(
            "LoopDetector",
            [signal.detector for signal in caught.exception.decision.reasons],
        )
        self.assertTrue(handler.should_interrupt)

    def test_callback_handler_can_collect_alerts_without_raising(self):
        handler = LoopGuardCallbackHandler(interrupt_on_fatal=False)

        for _ in range(3):
            handler.on_tool_start({"name": "search"}, {"query": "pricing page"})
            handler.on_tool_end("no results")

        report = handler.report()

        self.assertTrue(report["interrupted"])
        self.assertEqual(report["decisions"][-1]["action"], "stop")
        self.assertEqual(report["alerts"][0]["detector"], "LoopDetector")
        self.assertEqual(report["metrics"]["tool_calls"], 3)

    def test_callback_handler_records_tool_errors_as_observations(self):
        handler = LoopGuardCallbackHandler(detectors=[StallDetector(patience=2)])

        handler.on_tool_error(RuntimeError("rate limited"))
        handler.on_tool_error(RuntimeError("rate limited"))

        self.assertEqual(handler.alerts[-1].detector, "StallDetector")
        self.assertFalse(handler.should_interrupt)

    def test_callback_handler_from_config_uses_live_policy(self):
        config = "\n".join(
            [
                "live:",
                "  exact_threshold: 5",
                "  max_tool_calls: 2",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loopguard.yml"
            path.write_text(config)
            handler = LoopGuardCallbackHandler.from_config(str(path), interrupt_on_fatal=False)

        loop = next(item for item in handler.detectors if isinstance(item, LoopDetector))

        self.assertEqual(loop.threshold, 5)
        self.assertEqual(handler.max_tool_calls, 2)
        self.assertIn("ToolCallBudgetDetector", [type(item).__name__ for item in handler.detectors])


class ProgressAwarePolicyTests(unittest.TestCase):
    def test_policy_models_are_public_exports(self):
        from loopguard import DetectionSignal as PublicSignal
        from loopguard import GuardAction as PublicAction
        from loopguard import PolicyEngine as PublicPolicy

        self.assertIs(PublicSignal, DetectionSignal)
        self.assertIs(PublicAction, GuardAction)
        self.assertIs(PublicPolicy, PolicyEngine)

    def test_policy_stops_on_repetition_plus_no_progress(self):
        decision = PolicyEngine().decide(
            [
                DetectionSignal("LoopDetector", "repeated_tool_call", 0.65, "same search"),
                DetectionSignal("ProgressDetector", "no_progress", 0.78, "same result"),
            ]
        )

        self.assertEqual(decision.action, GuardAction.STOP)
        self.assertGreaterEqual(decision.risk_score, 0.9)

    def test_policy_allows_repeated_action_without_no_progress(self):
        decision = PolicyEngine().decide(
            [DetectionSignal("LoopDetector", "repeated_tool_call", 0.65, "same polling tool")]
        )

        self.assertEqual(decision.action, GuardAction.WARN)

    def test_healthy_polling_continues_when_outputs_change(self):
        monitor = Monitor([LoopDetector(threshold=3), ProgressDetector()])

        for result in ["pending", "running", "completed"]:
            monitor.observe("agent", "decided: poll_job({})", "tool:poll_job()")
            monitor.observe("tools", f"observed: {result}", f"result:{result}")

        self.assertFalse(monitor.should_interrupt)
        self.assertNotEqual(monitor.decisions[-1].action, GuardAction.STOP)

    def test_retryable_failure_then_success_does_not_stop(self):
        monitor = Monitor([RepeatedFailureDetector(), ProgressDetector()])

        monitor.observe("tools", "error: 429 rate limit", "result:429 rate limit")
        monitor.observe("tools", "observed: success", "result:success")

        self.assertFalse(monitor.should_interrupt)
        self.assertEqual(monitor.decisions[-1].action, GuardAction.CONTINUE)

    def test_permanent_failure_stops_before_many_retries(self):
        monitor = Monitor([RepeatedFailureDetector(attempts=2)])

        monitor.observe("tools", "error: 401 unauthorized", "result:401 unauthorized")
        monitor.observe("tools", "error: 401 unauthorized", "result:401 unauthorized")

        self.assertTrue(monitor.should_interrupt)
        self.assertEqual(monitor.decisions[-1].action, GuardAction.STOP)

    def test_alternating_cycle_with_no_progress_stops(self):
        monitor = Monitor([CycleDetector(max_cycle_length=2), ProgressDetector()])

        for tool in ["search", "summarize", "search", "summarize"]:
            monitor.observe("agent", f"decided: {tool}({{}})", f"tool:{tool}()")
            monitor.observe("tools", "observed: no results", "result:no results")

        self.assertTrue(monitor.should_interrupt)
        self.assertEqual(monitor.decisions[-1].action, GuardAction.STOP)

    def test_controlled_progress_uses_looser_repeat_threshold(self):
        try:
            from loopguard.scenarios import controlled_detectors
        except ModuleNotFoundError as exc:
            self.skipTest(f"demo dependency unavailable: {exc}")

        loop = next(item for item in controlled_detectors("controlled_progress") if isinstance(item, LoopDetector))
        empty_loop = next(item for item in controlled_detectors("controlled_empty") if isinstance(item, LoopDetector))

        self.assertEqual(loop.threshold, 4)
        self.assertEqual(empty_loop.threshold, 3)


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
        self.assertIn("loop", [alert["type"] for alert in report["runs"][0]["alerts"]])
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

    def test_ingest_detects_handoff_loop_from_caller_fields(self):
        trace = {
            "run_id": "handoff-004",
            "steps": [
                {"agent": "researcher", "caller": "planner", "output": "handoff"},
                {"agent": "reviewer", "caller": "researcher", "output": "handoff"},
                {"agent": "planner", "caller": "reviewer", "output": "handoff"},
                {"agent": "researcher", "caller": "planner", "output": "handoff"},
                {"agent": "reviewer", "caller": "researcher", "output": "handoff"},
                {"agent": "planner", "caller": "reviewer", "output": "handoff"},
            ],
        }

        report = analyze_trace(trace)

        self.assertEqual(report.status, "looping")
        self.assertEqual(report.alerts[-1].detector, "HandoffLoopDetector")

    def test_v02_labeled_traces_match_expected_final_decisions(self):
        data = json.loads(V02_TRACE.read_text())
        self.assertGreaterEqual(len(data), 5)

        reports = {report.run_id: report for report in analyze_file(V02_TRACE)}

        for trace in data:
            report = reports[trace["run_id"]]
            final = report.final_decision
            self.assertIsNotNone(final)
            self.assertEqual(final.action.value, trace["expected_decision"])


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
