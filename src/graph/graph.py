from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.graph.normalize import make_normalize_node
from src.graph.retrieve import make_retrieve_node
from src.graph.state import ChatGraphState
from src.graph.synthesize import make_synthesize_node


def build_chat_graph() -> Any:
    builder = StateGraph(ChatGraphState)
    builder.add_node("normalize", make_normalize_node())
    builder.add_node("classify", make_classify_node())
    builder.add_node("respond", respond_node) 
    builder.add_node("retrieve", make_retrieve_node())
    builder.add_node("synthesize", make_synthesize_node())

    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "classify")
    builder.add_conditional_edges(
        "classify",
        lambda s: "respond" if s.get("request_kind") == "smalltalk" else "retrieve",
        {"respond": "respond", "retrieve": "retrieve"},
    )
    builder.add_edge("respond", END)  
    builder.add_edge("retrieve", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()