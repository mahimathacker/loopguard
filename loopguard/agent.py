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
