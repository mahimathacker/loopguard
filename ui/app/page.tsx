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

import type { EvalScore, MetricsMsg, Scenario, ServerMessage } from "./types";

const API = process.env.NEXT_PUBLIC_LOOPGUARD_API ?? "http://localhost:8000";
const WS = API.replace(/^http/, "ws");

const SCENARIOS: { value: Scenario; label: string }[] = [
  { value: "exact", label: "Scripted: identical tool loop" },
  { value: "semantic", label: "Scripted: paraphrase loop (OpenAI)" },
  { value: "calc", label: "Real agent: solvable task (finishes)" },
  { value: "trap", label: "Real agent: impossible goal (loops)" },
];

// Fixed positions for the two known nodes; anything else gets staggered.
const POSITIONS: Record<string, { x: number; y: number }> = {
  agent: { x: 120, y: 120 },
  tools: { x: 460, y: 120 },
};

type NodeVariant = "idle" | "active" | "loop";
type LogEntryKind = "event" | "warning" | "error" | "success";

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
  if (kind === "warning") return "border-amber-500/30 bg-amber-500/10 text-amber-200";
  if (kind === "error") return "border-red-500/35 bg-red-500/10 text-red-200";
  if (kind === "success") return "border-emerald-500/30 bg-emerald-500/10 text-emerald-200";
  return "border-slate-700/80 bg-slate-900/60 text-slate-200";
}

export default function Page() {
  const [scenario, setScenario] = useState<Scenario>("exact");
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [metrics, setMetrics] = useState<MetricsMsg | null>(null);
  const [fatalAlert, setFatalAlert] = useState<string | null>(null);
  const [verdict, setVerdict] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [evalScore, setEvalScore] = useState<EvalScore | null>(null);
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
      case "alert":
        if (msg.fatal) {
          setFatalAlert(`${msg.detector}: ${msg.message}`);
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
            ? "LoopGuard interrupted the agent: loop detected."
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
          <p className="text-sm font-medium text-slate-500">agent loop observability</p>
        </div>
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
        <div className="relative flex-1 bg-slate-950">
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
          {fatalAlert && (
            <div className="absolute left-5 right-5 top-5 rounded-lg border border-red-400/50 bg-red-950/90 px-4 py-3 text-sm font-medium text-red-100 shadow-2xl shadow-red-950/30 backdrop-blur">
              {fatalAlert}
            </div>
          )}
        </div>

        <aside className="flex min-h-0 w-[27rem] flex-col border-l border-slate-800 bg-slate-950">
          <section className="border-b border-slate-800 p-5">
            <h2 className="mb-4 text-xs font-bold uppercase tracking-wide text-slate-500">
              Detector accuracy
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
              </>
            ) : (
              <p className="text-sm text-slate-500">Scorecard unavailable.</p>
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
