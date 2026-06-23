// Message contract streamed by the LoopGuard server over the /run WebSocket.
// Mirrors loopguard/runner.py (stream_run) and the dataclasses it serializes.

export interface EventMsg {
  type: "event";
  step: number;
  node: string;
  action: string;
  signature: string;
  payload: Record<string, unknown>;
}

export interface AlertMsg {
  type: "alert";
  detector: string;
  kind: string;
  message: string;
  fatal: boolean;
}

export interface MetricsMsg {
  type: "metrics";
  total_steps: number;
  by_node: Record<string, number>;
  tool_calls: number;
  distinct_tool_calls: number;
  most_common_action: [string, number] | null;
  repeat_rate: number;
}

export interface DoneMsg {
  type: "done";
  interrupted: boolean;
}

export interface ErrorMsg {
  type: "error";
  message: string;
}

export type ServerMessage = EventMsg | AlertMsg | MetricsMsg | DoneMsg | ErrorMsg;

export type Scenario = "exact" | "semantic" | "calc" | "trap";

// Detector accuracy scorecard from the /eval endpoint. Mirrors loopguard/evals.py.
export interface EvalScore {
  detector: string;
  cases: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
}
