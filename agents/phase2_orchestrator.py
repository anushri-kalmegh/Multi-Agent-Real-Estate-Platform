"""LangGraph supervisor for Phase 2 conversational orchestration."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

from agents.semantic_legal_agent import answer_legal_query
from agents.llm_service import extract_requirements, generate_summary, groq_enabled
from agents.pipeline import run_property_pipeline


class ConversationState(TypedDict, total=False):
    user_query: str
    requirements: dict[str, Any]
    intent: str
    missing_fields: list[str]
    status: str
    message: str
    result_type: str
    result: dict[str, Any]
    intelligence_source: str
    response_source: str


def understand_node(state: ConversationState) -> ConversationState:
    requirements, source = extract_requirements(
        state["user_query"], state.get("requirements")
    )
    intent = requirements.get("intent", "general")
    missing = []
    if intent == "property":
        if not requirements.get("city"):
            missing.append("city")
        if not requirements.get("max_budget"):
            missing.append("budget")
    return {
        "requirements": requirements,
        "intent": intent,
        "missing_fields": missing,
        "intelligence_source": source,
    }


def route_after_understanding(state: ConversationState) -> str:
    if state.get("missing_fields"):
        return "clarify"
    if state.get("intent") == "legal":
        return "legal"
    if state.get("intent") == "property":
        return "property"
    return "general"


def clarify_node(state: ConversationState) -> ConversationState:
    missing = state["missing_fields"]
    labels = {"city": "city", "budget": "maximum budget"}
    requested = " and ".join(labels.get(field, field) for field in missing)
    return {
        "status": "needs_input",
        "result_type": "text",
        "message": (
            f"I’ve saved the requirements provided so far. What is your {requested}? "
            "You can answer naturally, and I’ll continue this same search."
        ),
    }


def property_node(state: ConversationState) -> ConversationState:
    req = state["requirements"]
    result = run_property_pipeline(
        city=req["city"],
        max_budget=req["max_budget"],
        bhk=req.get("bhk"),
        locality=req.get("locality"),
        purpose=req.get("purpose", "self-use"),
        financial_assumptions={
            key: value for key, value in (req.get("financial_assumptions") or {}).items()
            if value is not None
        },
    )
    if not result:
        return {
            "status": "no_results",
            "result_type": "text",
            "message": "No fully evaluated properties matched these requirements.",
        }
    result["query_understanding"] = req
    summary, source = generate_summary(
        state["user_query"], result, req.get("language", "English")
    )
    result["agent_trace"].insert(1, {
        "agent": "Conversation Intelligence",
        "status": "complete",
        "output": (
            f"Requirements interpreted with {state.get('intelligence_source', 'deterministic')} "
            f"and response generated with {source}"
        ),
    })
    return {
        "status": "ok",
        "result_type": "property",
        "message": summary,
        "result": result,
        "response_source": source,
    }


def legal_node(state: ConversationState) -> ConversationState:
    result = answer_legal_query(state["user_query"])
    return {
        "status": "ok" if result.get("chunk_id") != -1 else "no_results",
        "result_type": "legal",
        "message": "I found the most relevant passage in the legal knowledge base.",
        "result": result,
    }


def general_node(state: ConversationState) -> ConversationState:
    return {
        "status": "needs_input",
        "result_type": "text",
        "message": (
            "Tell me the city and budget for a property search, or ask a RERA/legal question."
        ),
    }


def build_graph():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(ConversationState)
    builder.add_node("understand", understand_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("property", property_node)
    builder.add_node("legal", legal_node)
    builder.add_node("general", general_node)
    builder.add_edge(START, "understand")
    builder.add_conditional_edges(
        "understand",
        route_after_understanding,
        {
            "clarify": "clarify",
            "property": "property",
            "legal": "legal",
            "general": "general",
        },
    )
    for node in ("clarify", "property", "legal", "general"):
        builder.add_edge(node, END)
    return builder.compile(checkpointer=InMemorySaver())


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_conversation(user_query: str, thread_id: str) -> dict:
    output = get_graph().invoke(
        {"user_query": user_query},
        {"configurable": {"thread_id": thread_id}},
    )
    return {
        "status": output.get("status"),
        "type": output.get("result_type", "text"),
        "message": output.get("message", ""),
        "data": output.get("result"),
        "requirements": output.get("requirements", {}),
        "llm_enabled": groq_enabled(),
        "intelligence_source": output.get("intelligence_source", "deterministic"),
    }
