from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from app.intent_classifier import classify_intent
from app.rag_engine import get_rag_answer

app = FastAPI(title="Meeting RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    hybrid: Optional[bool] = True          # Use hybrid search by default
    bm25_weight: Optional[float] = 0.3     # 30% BM25, 70% semantic

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    try:
        original_question = req.question.strip()
        
        # Check for /meeting tag – force RAG, remove tag
        if original_question.lower().startswith("/meeting"):
            clean_question = original_question[len("/meeting"):].strip()
            if not clean_question:
                answer = "Please provide a meeting-related question after /meeting, e.g., `/meeting what decisions were made?`"
                return QueryResponse(answer=answer, sources=[])
            # Force RAG without classification
            result = get_rag_answer(
                clean_question, 
                hybrid=req.hybrid, 
                bm25_weight=req.bm25_weight
            )
            sources = [doc.metadata for doc in result.get("source_documents", [])]
            return QueryResponse(answer=result["answer"], sources=sources)
        
        # No tag – use intent classifier
        intent = classify_intent(original_question)
        
        if intent == "greeting":
            answer = "Hello! I'm your meeting analyst. Ask me about meeting transcripts (decisions, action items, discussions)."
            return QueryResponse(answer=answer, sources=[])
        if intent == "off_topic":
            answer = "I'm specialized in meeting transcripts. Please ask about decisions, action items, or discussions from your meetings."
            return QueryResponse(answer=answer, sources=[])
        if intent == "identity":
            answer = "I am a meeting analyst AI. I help you query meeting transcripts. I don't have personal information about you. If you're asking about your own identity as discussed in meetings, please use `/meeting who am I?` to search transcripts."
            return QueryResponse(answer=answer, sources=[])
        
        # Meeting‑related (intent == "meeting")
        result = get_rag_answer(
            original_question, 
            hybrid=req.hybrid, 
            bm25_weight=req.bm25_weight
        )
        sources = [doc.metadata for doc in result.get("source_documents", [])]
        return QueryResponse(answer=result["answer"], sources=sources)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))