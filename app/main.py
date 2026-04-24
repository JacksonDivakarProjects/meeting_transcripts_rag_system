from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import re
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
    hybrid: Optional[bool] = True          
    bm25_weight: Optional[float] = 0.3     

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

def clean_answer(text: str) -> str:
    """Strips out <thinking>, <thought>, and any XML-style reasoning tags."""
    if not text:
        return text
    text = re.sub(r"<[^>]+>.*?</[^>]+>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>.*?(?=\[\d+\]|According to|Based on|The context|There are|The meetings|I cannot find)", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\n", "\n\n", text).strip()
    return text

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    try:
        original_question = req.question.strip()
        
        if original_question.lower().startswith("/meeting"):
            clean_question = original_question[len("/meeting"):].strip()
            if not clean_question:
                answer = "Please provide a meeting-related question after /meeting, e.g., `/meeting what decisions were made?`"
                return QueryResponse(answer=answer, sources=[])
            
            result = get_rag_answer(
                clean_question, 
                hybrid=req.hybrid, 
                bm25_weight=req.bm25_weight
            )
            sources = [doc.metadata for doc in result.get("source_documents", [])]
            return QueryResponse(answer=clean_answer(result["answer"]), sources=sources)
        
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
        
        result = get_rag_answer(
            original_question, 
            hybrid=req.hybrid, 
            bm25_weight=req.bm25_weight
        )
        sources = [doc.metadata for doc in result.get("source_documents", [])]
        return QueryResponse(answer=clean_answer(result["answer"]), sources=sources)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))