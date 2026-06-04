"""A tiny LangGraph agent that gets stuck in a prompt loop on purpose.

This is the *subject* LoopGuard monitors. It uses no real LLM (so it runs offline and
deterministically): the "agent" node follows a scripted policy that keeps deciding to
call the same tool with the same args, and the "tool" node keeps returning "no results".
That is exactly the Type-B loop the MVP detector should catch.

Graph shape (the cycle is the `tools -> agent` edge):

        ┌───────────────┐
        ▼               │
   ┌─────────┐    ┌──────────┐
   │  agent  │──▶ │  tools   │
   └─────────┘    └──────────┘
"""

from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    goal: str
    last_tool: Optional[str]
    last_args: Optional[dict]
    last_result: Optional[str]
    steps: int


# The query the broken agent fixates on. A healthy agent would change tactics after
# seeing "no results"; ours doesn't - that's the bug LoopGuard is here to surface.
STUCK_QUERY = {"query": "weather in Paris"}


def agent_node(state: AgentState) -> dict:
    """Scripted 'LLM': always decide to search the same thing. (A real LLM goes here.)"""
    return {
        "last_tool": "search",
        "last_args": STUCK_QUERY,
        "steps": state.get("steps", 0) + 1,
    }


def tools_node(state: AgentState) -> dict:
    """Execute the chosen tool. Our search always comes back empty -> no progress."""
    return {"last_result": "no results"}


def build_agent(recursion_limit_guard: int = 50):
    """Compile and return the looping agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "tools")
    graph.add_edge("tools", "agent")  # <-- the loop
    # Note: there is no edge to END. Nothing in the agent stops it; LoopGuard must.
    return graph.compile()


# Different wordings, same intent. A strict signature check sees four *distinct* calls
# and never flags a loop; semantic detection (Detector A) sees one repeated intent.
PARAPHRASES = [
    "weather in Paris",
    "Paris weather today",
    "current weather in Paris",
    "what is the weather like in Paris right now",
]


def paraphrasing_agent_node(state: AgentState) -> dict:
    """Scripted 'LLM' that rephrases the same failed search each turn instead of
    repeating it verbatim. This is the paraphrase loop strict matching misses."""
    i = state.get("steps", 0)
    query = PARAPHRASES[i % len(PARAPHRASES)]
    return {"last_tool": "search", "last_args": {"query": query}, "steps": i + 1}


def build_paraphrasing_agent():
    """Same graph as build_agent(), but the agent paraphrases instead of repeating."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", paraphrasing_agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "tools")
    graph.add_edge("tools", "agent")
    return graph.compile()


# --------------------------------------------------------------------------------------
# Real agent: a genuine LLM (gpt-4o-mini) deciding which tool to call, via LangGraph's
# prebuilt ReAct agent. Nothing here is rigged to loop. The tools do real work (real web
# search, real arithmetic) and the agent can reach END on a solvable task. A loop only
# emerges when the *situation* makes the model keep re-calling a tool, e.g. an impossible
# stop-condition. LoopGuard watches it exactly as it watches the scripted demos.
# --------------------------------------------------------------------------------------

import ast
import operator

from langchain_core.tools import tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_arith(node):
    """Safely evaluate an arithmetic AST node (no names, calls, or attribute access)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_arith(node.left), _eval_arith(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_arith(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '4801 * 379 + 1234'."""
    try:
        return str(_eval_arith(ast.parse(expression, mode="eval").body))
    except Exception as exc:  # noqa: BLE001 - report the failure back to the model
        return f"error: {exc}"


@tool
def web_search(query: str) -> str:
    """Search the web and return the top results (title and snippet) for a query."""
    from ddgs import DDGS  # lazy import so the offline scripted demos need no network deps

    try:
        results = DDGS().text(query, max_results=3)
    except Exception as exc:  # noqa: BLE001 - surface real search failures to the model
        return f"search error: {exc}"
    if not results:
        return "No results found."
    return "\n".join(f"- {r['title']}: {r['body']}" for r in results)


SYSTEM_PROMPT = (
    "You are a research assistant. For any factual claim you MUST rely on the web_search "
    "tool rather than your own knowledge, and base your answer only on what the results "
    "say. For arithmetic, use the calculator tool. If a search does not support the claim "
    "you were asked about, refine your query and try again."
)


def build_real_agent():
    """A real ReAct agent: gpt-4o-mini + calculator + web_search. Requires OPENAI_API_KEY."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return create_react_agent(model, [calculator, web_search], prompt=SYSTEM_PROMPT)
