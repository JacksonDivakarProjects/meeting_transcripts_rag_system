# main.py
import re
import traceback
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.intent_classifier import classify_intent
from app.rag_engine import get_rag_answer

app = FastAPI(title="Meeting RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = []
    hybrid: Optional[bool] = True
    bm25_weight: Optional[float] = 0.3


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_answer(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _intent_reply(intent: str) -> Optional[str]:
    """Return a canned reply for non-substantive intents, or None for meeting queries."""
    if intent == "greeting":
        return (
            "Hello! I'm your meeting analyst. "
            "Ask me about meeting transcripts — decisions, action items, or discussions."
        )
    if intent == "off_topic":
        return (
            "I'm specialised in meeting transcripts. "
            "Please ask about decisions, action items, or discussions from your meetings."
        )
    if intent == "identity":
        return (
            "I am a meeting analyst AI. I help you query meeting transcripts. "
            "If you're asking who spoke in a meeting, try: "
            "`/meeting who mentioned the budget?`"
        )
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    try:
        question = req.question.strip()

        # /meeting prefix forces RAG, bypassing intent classification
        if question.lower().startswith("/meeting"):
            clean_q = question[len("/meeting"):].strip()
            if not clean_q:
                return QueryResponse(
                    answer="Please provide a question after `/meeting`, e.g. `/meeting what decisions were made?`",
                    sources=[],
                )
            result = get_rag_answer(clean_q, hybrid=req.hybrid, bm25_weight=req.bm25_weight)
            sources = [doc.metadata for doc in result.get("source_documents", [])]
            return QueryResponse(answer=clean_answer(result["answer"]), sources=sources)

        # Intent gate
        intent = classify_intent(question)
        canned = _intent_reply(intent)
        if canned:
            return QueryResponse(answer=canned, sources=[])

        # Full RAG
        result = get_rag_answer(question, hybrid=req.hybrid, bm25_weight=req.bm25_weight)
        sources = [doc.metadata for doc in result.get("source_documents", [])]
        return QueryResponse(answer=clean_answer(result["answer"]), sources=sources)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
