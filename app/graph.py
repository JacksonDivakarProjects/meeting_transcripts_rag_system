# app/graph.py
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

# Use relative imports since this module lives inside the `app` package
from app.intent_classifier import classify_intent
from app.rag_engine import get_rag_answer   # re-exported below


class GraphState(TypedDict):
    question: str
    intent: Optional[str]
    answer: str
    source_documents: List[Document]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def classify_intent_node(state: GraphState) -> dict:
    return {"intent": classify_intent(state["question"])}


def handle_non_substantive(state: GraphState) -> dict:
    intent = state["intent"]
    responses = {
        "greeting": (
            "Hello! I'm your meeting analyst. "
            "Ask me about meeting transcripts — decisions, action items, or discussions."
        ),
        "off_topic": (
            "I'm specialised in meeting transcripts. "
            "Please ask about decisions, action items, or discussions from your meetings."
        ),
        "identity": (
            "I am a meeting analyst AI. I help you query meeting transcripts. "
            "How can I assist you today?"
        ),
    }
    answer = responses.get(intent, "I'm here to help with meeting transcripts.")
    return {"answer": answer, "source_documents": []}


def run_rag(state: GraphState) -> dict:
    try:
        result = get_rag_answer(state["question"])
        return {
            "answer": result["answer"],
            "source_documents": result["source_documents"],
        }
    except Exception as e:
        return {"error": str(e), "answer": f"Error during RAG: {e}", "source_documents": []}


def route_after_intent(state: GraphState) -> str:
    return "non_substantive" if state.get("intent") in {"greeting", "off_topic", "identity"} else "rag"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_rag_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("non_substantive", handle_non_substantive)
    workflow.add_node("rag", run_rag)
    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {"non_substantive": "non_substantive", "rag": "rag"},
    )
    workflow.add_edge("non_substantive", END)
    workflow.add_edge("rag", END)
    return workflow.compile()


rag_graph = build_rag_graph()
