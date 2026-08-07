"use client";

// LoopGuard UI: renders the agent graph with React Flow, opens the /run WebSocket,
// animates the active node as events stream in, and turns the cycle red when a fatal
// loop alert fires. This is a Client Component because it uses state, effects, and the
// browser WebSocket API.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MarkerType,
  type Node,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import type {
  EvalScore,
  GuardDecisionMsg,
  MetricsMsg,
  RuntimeInfo,
  Scenario,
  ServerMessage,
} from "./types";

const API = process.env.NEXT_PUBLIC_LOOPGUARD_API ?? "http://localhost:8000";
const WS = API.replace(/^http/, "ws");

const SCENARIOS: { value: Scenario; label: string }[] = [
  { value: "controlled_progress", label: "Controlled ReAct: pending -> completed" },
  { value: "controlled_503", label: "Controlled ReAct: 503 -> success" },
  { value: "controlled_401", label: "Controlled ReAct: auth failure" },
  { value: "controlled_empty", label: "Controlled ReAct: no-results loop" },
  { value: "exact", label: "Scripted: identical tool loop" },
  { value: "semantic", label: "Scripted: paraphrase loop (embeddings)" },
  { value: "calc", label: "Real agent: solvable task (finishes)" },
  { value: "trap", label: "Real agent: impossible goal (loops)" },
];

const SCENARIO_NOTES: Record<Scenario, string> = {
  controlled_progress: "Real model controls the run; the tool returns pending, processing, then completed. Expected: continue.",
  controlled_503: "Real model controls retries; the tool returns two temporary failures, then success. Expected: retry/continue.",
  controlled_401: "Real model sees repeated auth failure. Expected: stop or replan instead of burning calls.",
  controlled_empty: "Real model receives repeated empty results. Expected: repeated/similar action plus no progress becomes stop-worthy.",
  exact: "Same tool call and same result: high stuck risk, so the policy stops.",
  semantic: "Different wording, same intent: semantic evidence catches what exact matching misses.",
  calc: "A healthy live agent path: useful result appears, so LoopGuard stays quiet.",
  trap: "A real agent keeps searching an impossible goal: repeated intent plus no progress becomes stop-worthy.",
};

const QUICKSTART = `from loopguard import LoopGuardCallbackHandler, LoopGuardInterrupt

handler = LoopGuardCallbackHandler()

try:
    agent.invoke(input, config={"callbacks": [handler]})
except LoopGuardInterrupt as exc:
    print("stopped:", exc)

print(handler.report())`;

// Fixed positions for the two known nodes; anything else gets staggered.
const POSITIONS: Record<string, { x: number; y: number }> = {
  agent: { x: 120, y: 120 },
  tools: { x: 460, y: 120 },
};

type NodeVariant = "idle" | "active" | "loop";
type LogEntryKind = "event" | "signal" | "decision" | "warning" | "error" | "success";

interface LogEntry {
  id: string;
  kind: LogEntryKind;
  label: string;
  message: string;
  step?: number;
}

function nodeStyle(variant: NodeVariant): React.CSSProperties {
  const base: React.CSSProperties = {
    padding: "12px 18px",
    borderRadius: 8,
    border: "1px solid",
    boxShadow: "0 18px 45px rgba(15, 23, 42, 0.18)",
    fontWeight: 700,
    fontSize: 14,
    width: 168,
    textAlign: "center",
  };
  if (variant === "loop")
    return { ...base, borderColor: "#f87171", background: "#fee2e2", color: "#7f1d1d" };
  if (variant === "active")
    return { ...base, borderColor: "#38bdf8", background: "#e0f2fe", color: "#0c4a6e" };
  return { ...base, borderColor: "#94a3b8", background: "#f8fafc", color: "#1e293b" };
}

const IDLE_EDGE = { stroke: "#64748b", strokeWidth: 2 };
const LOOP_EDGE = { stroke: "#ef4444", strokeWidth: 3 };

function logTone(kind: LogEntryKind): string {
  if (kind === "signal") return "border-cyan-500/30 bg-cyan-500/10 text-cyan-100";
  if (kind === "decision") return "border-violet-500/30 bg-violet-500/10 text-violet-100";
  if (kind === "warning") return "border-amber-500/30 bg-amber-500/10 text-amber-200";
  if (kind === "error") return "border-red-500/35 bg-red-500/10 text-red-200";
  if (kind === "success") return "border-emerald-500/30 bg-emerald-500/10 text-emerald-200";
  return "border-slate-700/80 bg-slate-900/60 text-slate-200";
}

function decisionTone(action: GuardDecisionMsg["action"]): string {
  if (action === "stop") return "border-red-400/50 bg-red-950/70 text-red-100";
  if (action === "pause") return "border-orange-400/50 bg-orange-950/60 text-orange-100";
  if (action === "replan") return "border-violet-400/50 bg-violet-950/60 text-violet-100";
  if (action === "warn") return "border-amber-400/50 bg-amber-950/60 text-amber-100";
  return "border-emerald-400/35 bg-emerald-950/40 text-emerald-100";
}

export default function Page() {
  const [scenario, setScenario] = useState<Scenario>("controlled_progress");
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [metrics, setMetrics] = useState<MetricsMsg | null>(null);
  const [decision, setDecision] = useState<GuardDecisionMsg | null>(null);
  const [fatalAlert, setFatalAlert] = useState<string | null>(null);
  const [verdict, setVerdict] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [evalScore, setEvalScore] = useState<EvalScore | null>(null);
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const logEndRef = useRef<HTMLDivElement | null>(null);

  const loadGraph = useCallback(async () => {
    try {
      const res = await fetch(`${API}/graph?scenario=${scenario}`);
      const data: { nodes: { id: string }[]; edges: { source: string; target: string }[] } =
        await res.json();
      setNodes(
        data.nodes.map((n, i) => ({
          id: n.id,
          position: POSITIONS[n.id] ?? { x: 120 + i * 200, y: 300 },
          data: { label: n.id },
          style: nodeStyle("idle"),
        })),
      );
      setEdges(
        data.edges.map((e) => ({
          id: `${e.source}->${e.target}`,
          source: e.source,
          target: e.target,
          animated: false,
          markerEnd: { type: MarkerType.ArrowClosed, color: IDLE_EDGE.stroke },
          style: IDLE_EDGE,
        })),
      );
    } catch {
      setFatalAlert(`Could not reach the server at ${API}. Is it running (uvicorn server:app)?`);
    }
  }, [scenario]);

  // Load (and reload) the topology whenever the scenario changes.
  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      void loadGraph();
    });
    return () => {
      cancelAnimationFrame(frame);
      wsRef.current?.close();
    };
  }, [loadGraph]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [log]);

  // Load the detector accuracy scorecard once, on mount. Independent of any run.
  useEffect(() => {
    fetch(`${API}/eval`)
      .then((res) => res.json() as Promise<EvalScore>)
      .then(setEvalScore)
      .catch(() => setEvalScore(null));
    fetch(`${API}/runtime`)
      .then((res) => res.json() as Promise<RuntimeInfo>)
      .then(setRuntime)
      .catch(() => setRuntime(null));
  }, []);

  function highlightActive(active: string) {
    setNodes((prev) =>
      prev.map((n) => ({ ...n, style: nodeStyle(n.id === active ? "active" : "idle") })),
    );
    setEdges((prev) => prev.map((e) => ({ ...e, animated: e.source === active })));
  }

  function markLoop() {
    setNodes((prev) => prev.map((n) => ({ ...n, style: nodeStyle("loop") })));
    setEdges((prev) =>
      prev.map((e) => ({
        ...e,
        animated: true,
        style: LOOP_EDGE,
        markerEnd: { type: MarkerType.ArrowClosed, color: LOOP_EDGE.stroke },
      })),
    );
  }

  function handle(msg: ServerMessage) {
    switch (msg.type) {
      case "event":
        setLog((prev) => [
          ...prev,
          {
            id: `event-${msg.step}-${prev.length}`,
            kind: "event",
            step: msg.step,
            label: msg.node,
            message: msg.action,
          },
        ]);
        highlightActive(msg.node);
        break;
      case "signal":
        setLog((prev) => [
          ...prev,
          {
            id: `signal-${prev.length}`,
            kind: "signal",
            label: `${msg.detector} / ${msg.kind}`,
            message: `${Math.round(msg.score * 100)}% risk evidence: ${msg.message}`,
          },
        ]);
        break;
      case "decision":
        setDecision(msg);
        setLog((prev) => [
          ...prev,
          {
            id: `decision-${prev.length}`,
            kind: msg.action === "stop" ? "error" : msg.action === "continue" ? "success" : "decision",
            label: msg.recommended_action,
            message: `risk ${Math.round(msg.risk_score * 100)}%${
              msg.reasons.length ? ` from ${msg.reasons.map((reason) => reason.kind).join(", ")}` : ""
            }`,
          },
        ]);
        if (msg.action === "stop") {
          setFatalAlert(msg.stop_reason ?? "Run stopped: LoopGuard policy reached the stop threshold.");
          markLoop();
        }
        break;
      case "alert":
        if (msg.fatal) {
          setFatalAlert((current) => current ?? `${msg.detector}: ${msg.message}`);
          setLog((prev) => [
            ...prev,
            {
              id: `fatal-${prev.length}`,
              kind: "error",
              label: msg.detector,
              message: msg.message,
            },
          ]);
          markLoop();
        } else {
          setLog((prev) => [
            ...prev,
            {
              id: `warn-${prev.length}`,
              kind: "warning",
              label: msg.detector,
              message: msg.message,
            },
          ]);
        }
        break;
      case "metrics":
        setMetrics(msg);
        break;
      case "error":
        setFatalAlert(`Error: ${msg.message}`);
        setLog((prev) => [
          ...prev,
          {
            id: `error-${prev.length}`,
            kind: "error",
            label: "server",
            message: msg.message,
          },
        ]);
        break;
      case "done":
        {
          const message = msg.interrupted
            ? msg.reason ?? "Run stopped: LoopGuard policy reached the stop threshold."
            : "Run finished without a fatal loop alert.";
          setVerdict(message);
          setLog((prev) => [
            ...prev,
            {
              id: `done-${prev.length}`,
              kind: msg.interrupted ? "warning" : "success",
              label: "done",
              message,
            },
          ]);
        }
        break;
    }
  }

  function run() {
    setLog([]);
    setMetrics(null);
    setDecision(null);
    setFatalAlert(null);
    setVerdict(null);
    void loadGraph();
    setRunning(true);

    const ws = new WebSocket(`${WS}/run?scenario=${scenario}`);
    wsRef.current = ws;
    ws.onmessage = (ev) => handle(JSON.parse(ev.data) as ServerMessage);
    ws.onclose = () => setRunning(false);
    ws.onerror = () => {
      setRunning(false);
      setFatalAlert(`WebSocket error. Is the server running at ${API}?`);
      setLog((prev) => [
        ...prev,
        {
          id: `socket-${prev.length}`,
          kind: "error",
          label: "websocket",
          message: `Could not connect to ${API}.`,
        },
      ]);
    };
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-slate-950 text-slate-100">
      <header className="flex min-h-20 items-center gap-4 border-b border-slate-800/90 bg-slate-950/95 px-5 py-4 shadow-2xl shadow-black/20">
        <div className="min-w-0">
          <h1 className="text-xl font-bold text-white">LoopGuard</h1>
          <p className="text-sm font-medium text-slate-500">runtime guardrail for LangGraph agents</p>
        </div>
        {runtime && (
          <div className="hidden min-w-0 rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-xs text-slate-300 lg:block">
            <span className="font-bold uppercase tracking-wide text-slate-500">model</span>{" "}
            <span className="font-mono text-sky-300">{runtime.model_provider}</span>
            <span className="mx-2 text-slate-700">/</span>
            <span className="font-mono">{runtime.model}</span>
          </div>
        )}
        <code className="hidden rounded-md border border-slate-800 bg-slate-900 px-3 py-2 font-mono text-xs text-slate-300 xl:block">
          pip install loopguard-runtime
        </code>
        <div className="ml-auto flex min-w-0 items-center gap-3">
          <label className="sr-only" htmlFor="scenario">
            Scenario
          </label>
          <div className="relative min-w-0">
            <select
              id="scenario"
              className="h-11 w-[22rem] max-w-[46vw] appearance-none rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 pr-11 text-sm font-medium text-slate-100 shadow-inner shadow-black/20 outline-none transition hover:border-slate-500 focus:border-sky-400 focus:ring-2 focus:ring-sky-400/25 disabled:cursor-not-allowed disabled:opacity-60"
              value={scenario}
              onChange={(e) => setScenario(e.target.value as Scenario)}
              disabled={running}
            >
              {SCENARIOS.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
            <svg
              aria-hidden="true"
              className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M5.23 7.21a.75.75 0 0 1 1.06.02L10 11.17l3.71-3.94a.75.75 0 1 1 1.08 1.04l-4.25 4.5a.75.75 0 0 1-1.08 0l-4.25-4.5a.75.75 0 0 1 .02-1.06Z"
                clipRule="evenodd"
              />
            </svg>
          </div>
          <button
            className="inline-flex h-11 min-w-24 items-center justify-center rounded-lg bg-sky-500 px-5 text-sm font-bold text-slate-950 shadow-lg shadow-sky-950/40 transition hover:bg-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-300 focus:ring-offset-2 focus:ring-offset-slate-950 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400 disabled:shadow-none"
            onClick={run}
            disabled={running}
          >
            {running ? "Running..." : "Run"}
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <div className="flex min-h-0 flex-1 flex-col bg-slate-950">
          <section className="grid gap-4 border-b border-slate-800 bg-slate-900/35 p-5 lg:grid-cols-[1fr_1.2fr]">
            <div className="min-w-0">
              <p className="mb-2 text-xs font-bold uppercase tracking-wide text-sky-300">SDK usage</p>
              <h2 className="text-lg font-semibold text-white">Attach LoopGuard to a live agent run.</h2>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
                The local graph below is still a demo runner. The production path is the callback SDK:
                it records tool calls and results, turns them into detection signals, then lets the
                policy decide whether to continue, warn, replan, pause, or stop.
              </p>
              <div className="mt-4 grid grid-cols-4 gap-2 text-center text-xs font-semibold">
                {["trace", "signals", "policy", "report"].map((item) => (
                  <div key={item} className="rounded-md border border-slate-800 bg-slate-950/70 px-2 py-2 text-slate-300">
                    {item}
                  </div>
                ))}
              </div>
            </div>
            <pre className="min-h-0 overflow-auto rounded-lg border border-slate-800 bg-slate-950 p-4 font-mono text-[0.72rem] leading-5 text-slate-300">
              <code>{QUICKSTART}</code>
            </pre>
          </section>

          <div className="relative min-h-0 flex-1">
            <ReactFlow
              className="loopguard-flow"
              nodes={nodes}
              edges={edges}
              fitView
              fitViewOptions={{ padding: 0.22 }}
            >
              <Background color="#334155" gap={28} size={1.2} />
              <Controls />
            </ReactFlow>
            <div className="absolute left-5 top-5 max-w-xl rounded-lg border border-slate-700/80 bg-slate-950/85 px-4 py-3 text-sm shadow-2xl shadow-black/30 backdrop-blur">
              <p className="text-xs font-bold uppercase tracking-wide text-slate-500">Local scenario</p>
              <p className="mt-1 font-semibold text-slate-100">
                {SCENARIOS.find((item) => item.value === scenario)?.label}
              </p>
              <p className="mt-1 text-slate-400">{SCENARIO_NOTES[scenario]}</p>
            </div>
            {fatalAlert && (
              <div className="absolute left-5 right-5 top-28 rounded-lg border border-red-400/50 bg-red-950/90 px-4 py-3 text-sm font-medium text-red-100 shadow-2xl shadow-red-950/30 backdrop-blur">
                {fatalAlert}
              </div>
            )}
          </div>
        </div>

        <aside className="flex min-h-0 w-[27rem] flex-col border-l border-slate-800 bg-slate-950">
          <section className="border-b border-slate-800 p-5">
            <h2 className="mb-4 text-xs font-bold uppercase tracking-wide text-slate-500">
              Offline detector eval
            </h2>
            {evalScore ? (
              <>
                <dl className="grid grid-cols-2 gap-x-5 gap-y-3 text-sm">
                  <dt className="text-slate-400">precision</dt>
                  <dd className="text-right font-mono text-emerald-300">{evalScore.precision.toFixed(2)}</dd>
                  <dt className="text-slate-400">recall</dt>
                  <dd className="text-right font-mono text-amber-300">{evalScore.recall.toFixed(2)}</dd>
                  <dt className="text-slate-400">f1</dt>
                  <dd className="text-right font-mono text-slate-100">{evalScore.f1.toFixed(2)}</dd>
                </dl>
                <p className="mt-3 font-mono text-[0.7rem] text-slate-500">
                  {evalScore.detector} · {evalScore.cases} cases · tp={evalScore.tp} fp={evalScore.fp}{" "}
                  fn={evalScore.fn} tn={evalScore.tn}
                </p>
                <p className="mt-3 text-xs leading-5 text-slate-500">
                  Static replay scorecard, not the current live scenario.
                </p>
              </>
            ) : (
              <p className="text-sm text-slate-500">Scorecard unavailable.</p>
            )}
          </section>

          <section className="border-b border-slate-800 p-5">
            <h2 className="mb-4 text-xs font-bold uppercase tracking-wide text-slate-500">
              Recommended action
            </h2>
            {decision ? (
              <div className={`rounded-lg border px-3 py-3 ${decisionTone(decision.action)}`}>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm font-bold uppercase tracking-wide">{decision.action}</span>
                  <span className="font-mono text-sm">{Math.round(decision.risk_score * 100)}%</span>
                </div>
                {decision.stop_reason && (
                  <p className="mt-2 text-xs font-semibold leading-5 text-current/90">
                    {decision.stop_reason}
                  </p>
                )}
                <p className="mt-2 text-xs leading-5 text-current/75">
                  {decision.reasons.length
                    ? decision.reasons.map((reason) => reason.kind).join(" + ")
                    : "No stuck-risk signals yet."}
                </p>
              </div>
            ) : (
              <p className="text-sm text-slate-500">Run a scenario to see policy actions.</p>
            )}
          </section>

          <section className="border-b border-slate-800 p-5">
            <h2 className="mb-4 text-xs font-bold uppercase tracking-wide text-slate-500">Metrics</h2>
            {metrics ? (
              <dl className="grid grid-cols-2 gap-x-5 gap-y-3 text-sm">
                <dt className="text-slate-400">total steps</dt>
                <dd className="text-right font-mono text-slate-100">{metrics.total_steps}</dd>
                <dt className="text-slate-400">tool calls</dt>
                <dd className="text-right font-mono text-slate-100">{metrics.tool_calls}</dd>
                <dt className="text-slate-400">distinct calls</dt>
                <dd className="text-right font-mono text-slate-100">{metrics.distinct_tool_calls}</dd>
                <dt className="text-slate-400">repeat rate</dt>
                <dd className="text-right font-mono text-slate-100">{Math.round(metrics.repeat_rate * 100)}%</dd>
              </dl>
            ) : (
              <p className="text-sm text-slate-500">Run a scenario to see metrics.</p>
            )}
          </section>

          <section className="flex min-h-0 flex-1 flex-col p-5">
            <h2 className="mb-4 text-xs font-bold uppercase tracking-wide text-slate-500">Event log</h2>
            <ul className="flex-1 space-y-2 overflow-auto pr-1 font-mono text-xs leading-relaxed">
              {log.length === 0 && (
                <li className="rounded-lg border border-dashed border-slate-800 px-3 py-4 text-slate-500">
                  Waiting for streamed events.
                </li>
              )}
              {log.map((entry) => (
                <li
                  key={entry.id}
                  className={`rounded-lg border px-3 py-2.5 ${logTone(entry.kind)}`}
                >
                  <div className="mb-1 flex items-center gap-2 text-[0.68rem] font-bold uppercase tracking-wide text-current/70">
                    {typeof entry.step === "number" && <span>#{entry.step}</span>}
                    <span>{entry.label}</span>
                  </div>
                  <p className="whitespace-pre-wrap break-words text-[0.8rem] leading-5">{entry.message}</p>
                </li>
              ))}
              <div ref={logEndRef} />
            </ul>
          </section>

          {verdict && (
            <div className="border-t border-slate-800 bg-slate-900/70 p-5 text-sm font-semibold text-slate-200">
              {verdict}
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
