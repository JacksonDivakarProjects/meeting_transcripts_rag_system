# 📝 Meeting RAG System – Comprehensive User Guide

> **Enterprise-grade AI assistant** for querying meeting transcripts using natural language.  
> Combines semantic search (ChromaDB) + keyword search (Whoosh BM25) → feeds results to Groq LLM → delivers cited answers via FastAPI + React.

---

## What This System Does

You give it a folder of meeting transcripts (JSON format). You ask questions in plain English. It returns **accurate, source-cited answers** drawn directly from the transcripts – no hallucination, no guessing.

**Example:**  
> *“What did Sarah say about the Q3 budget?”*  
> → System finds the exact line where Sarah spoke, shows the timestamp, and answers with a citation like `[1]`.

---

##  Important Features (The Highlights)

### 1. Hybrid Retrieval – Best of Both Worlds

Most RAG systems use **only semantic search** (embeddings). That fails on exact names, IDs, or codes.  
This system uses **both**:

- **BM25 (Whoosh)** – keyword search, perfect for “meeting ID 5432” or “John’s quote”
- **Semantic (ChromaDB)** – understands meaning, perfect for “what were the main arguments?”

They are combined using **Reciprocal Rank Fusion (RRF)**.  
You can adjust the balance:  
- Default: `70% semantic + 30% keyword`  
- For code-heavy transcripts, increase keyword weight up to 50–70%.

---

### 2. Intent Classification – Smart Routing

Not every user message needs the heavy RAG pipeline. The system first classifies your query into one of four intents:

| Intent | Example | Action |
|--------|---------|--------|
| `greeting` | “Hello”, “Hi there” | Returns a friendly welcome |
| `identity` | “What are you?”, “Who made you?” | Describes the system |
| `off_topic` | “What’s the weather?” | Polite redirect back to meetings |
| `meeting` | “What decisions were made?” | Runs the full RAG pipeline |

This saves **LLM tokens** (cost) and **latency** for trivial questions.

---

### 3. Dynamic k – Right Amount of Context

Instead of always fetching the same number of documents (e.g., 5), the system adjusts `k` based on the question type:

| Question Type | k | Example |
|---------------|----|---------|
| Summary / overview | 10 | “Summarise the whole meeting” |
| Analytical | 8 | “Compare the arguments for and against” |
| Factual / specific | 4 | “Who said the project is delayed?” |
| General | 6 | (default) |

> This prevents wasting the LLM’s context window on simple queries while ensuring complex questions get enough information.

---

### 4. Disk-Cached Whoosh Index – Fast Restarts

Building a keyword index from scratch every time you restart is slow (minutes for large corpora).  
The system saves the Whoosh BM25 index to disk (`whoosh_cache/`) after the first build.  
All future restarts load from disk – **startup takes seconds**, not minutes.

---

### 5. Numbered Citations – Auditable Answers

Every retrieved chunk is prefixed with `[1]`, `[2]`, etc. The LLM is instructed to **reference these numbers** in its answer.  
The API response includes a `sources` array with:

- Speaker name
- Timestamp
- Source file name
- Meeting topic / ID

You can **trace every claim back to the original transcript**.

---

### 6. `/meeting` Prefix – Escape Hatch

Intent classifiers are good but not perfect. If the system ever misclassifies a legitimate meeting question (e.g., “who am I in this project?” → marked as `identity`), you can **force RAG** by simply starting your question with `/meeting`:

```
/meeting Who am I in this project?
```

The prefix is stripped and the query goes straight to the retrieval pipeline – bypassing intent classification.

---

### 7. LangGraph Orchestration – Future-Proof

The pipeline is built as a **LangGraph state machine**.  
Right now it does: `classify → route → RAG`.  
But because it’s a graph, you can easily add new nodes later without rewriting everything:

- Multi-meeting summarisation
- Follow-up detection (“tell me more about X”)
- Citation verification
- Routing to different vector stores (by meeting ID)

---

##  System Architecture (Simple View)

```
User (React UI or Streamlit)
        │
        ▼
   FastAPI Backend
        │
        ├── Intent Classifier (BART-MNLI)
        │         │
        │         ▼ (if meeting-related)
        │   RAG Engine
        │         │
        │         ├── Hybrid Retriever
        │         │     ├── Whoosh (keyword)
        │         │     └── ChromaDB (semantic)
        │         │
        │         └── Groq LLM (qwen3-32b)
        │
        └── JSON response (answer + sources)
```

All components can run **locally** or inside **Docker**.

---

##  Setup – Step by Step

### Prerequisites
- Python 3.11+ and Node 20+ **or** Docker + Docker Compose
- A **Groq API key** (free at [console.groq.com](https://console.groq.com))
- Your meeting transcripts (JSON) inside `./data/json_chunks/`

### 1. Clone & Install

```bash
git clone <your-repo>
cd meeting-rag-system
```

### 2. Set Environment Variables

Copy the example file and add your Groq API key:

```bash
cp .env.example .env
# Edit .env – add GROQ_API_KEY=your_key_here
```

### 3. Build the Vector Store (once)

This reads all JSON transcripts, chunks them, creates embeddings, and builds the keyword index.

```bash
pip install -r requirements.txt
python create_vector_store.py
```

> This may take a few minutes depending on the size of your transcripts. Done once.

### 4. Run the System

#### Option A – Docker (easiest)
```bash
docker compose up --build
```
- React UI: http://localhost:3000
- FastAPI docs: http://localhost:8000/docs

#### Option B – Local (development)
```bash
# Terminal 1 – Backend
uvicorn main:app --reload --port 8000

# Terminal 2 – Frontend
cd frontend
npm install
npm run dev
```

---

##  Using the API

### POST `/query` – Ask a question

**Request body:**

```json
{
  "question": "What decisions were made about the budget?",
  "chat_history": [],           // optional, for multi-turn
  "hybrid": true,               // true = BM25 + semantic
  "bm25_weight": 0.3            // 0 = pure semantic, 1 = pure keyword
}
```

**Response:**

```json
{
  "answer": "According to [1], the budget was increased by 15%.",
  "sources": [
    {
      "speaker": "Alice",
      "timestamp_str": "12:34",
      "source_file": "meeting_001.json",
      "topic": "Budget Review",
      "meeting_id": "mtg-2024-001"
    }
  ]
}
```

### GET `/health` – Check status

Returns `{"status": "ok"}` – used by Docker health checks.

---

##  Configuration Reference

All settings are in a `.env` file at the project root:

| Variable | Default | What it does |
|----------|---------|---------------|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `qwen/qwen3-32b` | Which Groq model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence transformer |
| `JSON_DIR` | `./data/json_chunks` | Folder containing your JSON transcripts |
| `VECTOR_DB_DIR` | `./meetingbank_vector_db` | Where ChromaDB persists vectors |
| `WHOOSH_DIR` | `./whoosh_cache` | Where Whoosh stores the keyword index |

Frontend (React) uses `frontend/.env`:
```ini
VITE_API_TARGET=http://localhost:8000   # or http://fastapi:8000 in Docker
```

---

##  Why These Design Decisions? 

| Decision | Why it matters |
|----------|----------------|
| **Hybrid retrieval** | Pure semantic search misses exact terms (names, IDs). BM25 catches them. RRF combines both without needing score normalisation. |
| **Whoosh over rank-bm25** | Whoosh persists to disk → instant restarts. `rank-bm25` re-indexes in memory every time → slow for large corpora. |
| **Dynamic k** | Fixed `k` wastes tokens on simple queries and under-retrieves for summaries. Dynamic k gives exactly the right amount of context. |
| **BART-MNLI for intent** | Zero-shot classification – no training data needed. Fast on CPU. Fallback defaults to `meeting` so no query is ever dropped. |
| **LangGraph** | Adding new features (follow-ups, multi-meeting, verification) becomes a matter of adding new nodes, not rewriting the pipeline. |
| **/meeting prefix** | Intent classifiers can misfire. This is a user‑controlled escape hatch that forces RAG without changing any configuration. |

---

##  Common Questions

**Q: Can I use this with other LLMs?**  
A: The system is built for Groq, but you can replace the model in `.env` with any Groq‑supported model (e.g., `llama3-70b`). For non‑Groq, you’d need to modify `qa_chain.py`.

**Q: How do I prepare my meeting transcripts?**  
A: The system expects JSON files in `JSON_DIR`. Each file should contain a list of transcripts with fields like `speaker`, `timestamp`, `text`, and metadata (`topic`, `meeting_id`). See `vector_store.py` for the exact format.

**Q: Does it support multi‑turn conversations?**  
A: Yes – the API accepts `chat_history`. The LLM receives the full conversation context.

**Q: What if I only want semantic search (no BM25)?**  
A: Set `hybrid=false` in the query body, or set `bm25_weight=0` in the request.

**Q: Can I run this without Docker?**  
A: Absolutely – see the local setup instructions above. You’ll need Python and Node running separately.

---

##  Summary

| Feature             | Benefit                                       |
| ------------------- | --------------------------------------------- |
| Hybrid retrieval    | Finds both exact terms and conceptual matches |
| Intent classifier   | Saves cost & time on greetings / off‑topic    |
| Dynamic k           | Right amount of context for every question    |
| Disk‑cached Whoosh  | Fast restarts even with huge transcripts      |
| Numbered citations  | Every answer is traceable                     |
| `/meeting` override | Never locked out by misclassification         |
| LangGraph           | Easy to extend without refactoring            |

---

*This guide covers version 1.1 of the Meeting Transcript RAG System. For advanced usage (custom embeddings, multi‑meeting routing, or adding new intent types), refer to the source code comments in `rag_engine.py` and `graph.py`.*