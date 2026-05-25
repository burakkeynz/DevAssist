# Importing required libraries for FastAPI SSE streaming backend
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional, List, Literal, cast


import ollama
from ollama import Message
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.indexer import index_codebase
from app.rag import RAGStore
from app.database import (
    init_db,
    create_session,
    save_message,
    load_session_messages,
    load_session_full,
    get_all_sessions,
    extract_memory,
    save_memory,
    load_memories,
    save_summary,
    load_summary,
    count_messages,
    delete_session,
    ROLLING_SUMMARY_THRESHOLD
)

# Configuring logging for FastAPI backend operations
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Defining application constants
LLM_MODEL = "devassist"
LOG_PATH = Path("logs/query_log.jsonl")
CODEBASE_PATH = "codebase/"
TOP_K = 5
NO_CONTEXT_THRESHOLD = 0.3

# Initializing FastAPI application
app = FastAPI(
    title="DevAssist API",
    description="Offline TreeRAG-powered code assistant with XAI attribution",
    version="1.0.0"
)

# Configuring CORS for local frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mounting frontend static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Initializing RAGStore and SQLite database as application singletons
rag_store = RAGStore()
init_db()


# Defining query request schema
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: int = TOP_K


# Defining session creation request schema
class SessionRequest(BaseModel):
    first_message: str


# Appending query audit trail to JSONL log file
def log_query(
    query: str,
    chunks: list,
    response_preview: str,
    mode: str
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "mode": mode,
        "retrieved_chunks": [
            {
                "function_name": c["metadata"].get("function_name", "N/A"),
                "file_name": c["metadata"].get("file_name", "N/A"),
                "attribution_pct": c.get("attribution_pct", 0),
                "cross_encoder_score": c.get("cross_encoder_score", 0)
            }
            for c in chunks
        ],
        "response_preview": response_preview[:200]
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    logger.info(f"Logging query audit trail: '{query[:50]}...'")


# Building context-injected prompt from retrieved TreeRAG chunks
def build_rag_prompt(query: str, chunks: list) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks):
        fname = chunk["metadata"].get("function_name", "N/A")
        file = chunk["metadata"].get("file_name", "N/A")
        pct = chunk.get("attribution_pct", 0)
        context_blocks.append(
            f"[Chunk {i+1} | {fname} | {file} | Attribution: {pct}%]\n"
            f"{chunk['content']}"
        )
    context = "\n\n".join(context_blocks)
    return (
        f"Using the following codebase context to answer the question.\n\n"
        f"--- CONTEXT ---\n{context}\n\n"
        f"--- QUESTION ---\n{query}\n\n"
        f"--- ANSWER ---"
    )


# Building system prompt with injected memories and session summary
def build_system_prompt(session_id: Optional[str]) -> str:
    parts = []

    # Injecting persistent user memories into system prompt
    memories = load_memories()
    if memories:
        memory_text = "\n".join(f"- {m}" for m in memories)
        parts.append(f"User saved notes (always remember these):\n{memory_text}")

    # Injecting rolling session summary for context compression
    if session_id:
        summary = load_summary(session_id)
        if summary:
            parts.append(f"Summary of earlier conversation:\n{summary}")

    base = (
        "You are DevAssist, an expert offline AI code assistant. "
        "When codebase context is provided, prioritize it. "
        "When no context is provided, answer using your general coding knowledge. "
        "Never fabricating code or APIs that do not exist. "
        "Always responding with precise, production-ready code."
    )

    if parts:
        return base + "\n\n" + "\n\n".join(parts)
    return base


# Generating rolling summary when message threshold is reached
async def maybe_generate_summary(session_id: str) -> None:
    msg_count = count_messages(session_id)
    if msg_count > 0 and msg_count % ROLLING_SUMMARY_THRESHOLD == 0:
        logger.info(f"Generating rolling summary at {msg_count} messages...")
        messages = load_session_messages(session_id)
        summary_prompt = (
            "Summarize the following conversation in 3-5 sentences, "
            "focusing on key technical decisions, code discussed, and user goals:\n\n"
            + "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[-10:])
        )
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[Message(role="user", content=summary_prompt)]
        )
        summary = str(response.get("message", {}).get("content", ""))  # type: ignore[union-attr]
        if summary:
            save_summary(session_id, summary)
            logger.info(f"Saving rolling summary for session: {session_id[:16]}...")


# Streaming LLM response tokens via SSE with attribution metadata
async def stream_response(
    query: str,
    session_id: Optional[str],
    top_k: int = TOP_K
) -> AsyncGenerator[str, None]:
    logger.info(f"Processing SSE query: '{query[:60]}...'")

    # Detecting and saving memory command before processing
    memory_content = extract_memory(query)
    if memory_content:
        save_memory(memory_content, session_id)
        yield json.dumps({
            "type": "memory_saved",
            "data": f"Memory saved: {memory_content}"
        })
        await asyncio.sleep(0)
        return

    # Creating new session if none provided
    if not session_id:
        session_id = create_session(query)
        yield json.dumps({"type": "session_created", "data": session_id})
        await asyncio.sleep(0)

    # Retrieving hybrid TreeRAG chunks with attribution scores
    chunks = rag_store.hybrid_retrieve(query, top_k=top_k)

    # Determining retrieval mode based on attribution threshold
    max_attribution = max((c.get("attribution_pct", 0) for c in chunks), default=0)
    mode = "rag" if max_attribution >= NO_CONTEXT_THRESHOLD * 100 else "general"

    # Emitting attribution metadata as first SSE event
    attribution_data = [
        {
            "function_name": c["metadata"].get("function_name", "N/A"),
            "file_name": c["metadata"].get("file_name", "N/A"),
            "attribution_pct": c.get("attribution_pct", 0),
            "cross_encoder_score": round(c.get("cross_encoder_score", 0), 4),
            "rrf_score": round(c.get("rrf_score", 0), 6),
            "content_preview": c["content"][:100]
        }
        for c in chunks
    ]
    yield json.dumps({
        "type": "attribution",
        "data": attribution_data,
        "mode": mode,
        "session_id": session_id
    })
    await asyncio.sleep(0)

    # Building Ollama message history with system prompt injection
    system_prompt = build_system_prompt(session_id)
    history = load_session_messages(session_id)

    # Building final query with RAG context if applicable
    final_query = build_rag_prompt(query, chunks) if mode == "rag" else query

    messages: List[Message] = [
    Message(role="system", content=system_prompt)
    ]
    for h in history:
        role = cast(Literal["user", "assistant", "system"], h["role"])
        messages.append(Message(role=role, content=h["content"]))
    messages.append(Message(role="user", content=final_query))
      

    # Saving user message to session before streaming
    save_message(session_id, "user", query, attribution_data)

    # Streaming LLM tokens via Ollama local inference
    full_response = ""
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        stream=True
    )

    for chunk_response in stream:  # type: ignore[union-attr]
        token = str(chunk_response.get("message", {}).get("content", ""))  # type: ignore[union-attr]("content", ""))
        full_response += token
        yield json.dumps({"type": "token", "data": token})
        await asyncio.sleep(0)

    # Saving assistant response to session after streaming completes
    save_message(session_id, "assistant", full_response)

    # Emitting stream completion signal
    yield json.dumps({"type": "done", "data": "[DONE]", "session_id": session_id})

    # Triggering rolling summary if threshold reached
    await maybe_generate_summary(session_id)

    # Writing query audit log entry...
    log_query(query, chunks, full_response, mode)


# Defining SSE streaming endpoint for query processing
@app.post("/query")
async def query_endpoint(request: QueryRequest) -> EventSourceResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    logger.info(f"Receiving query request: '{request.query[:60]}...'")
    return EventSourceResponse(
        stream_response(request.query, request.session_id, request.top_k)
    )


# Defining session history endpoint for UI rendering
@app.get("/sessions")
async def get_sessions() -> list:
    logger.info("Fetching all sessions for sidebar rendering...")
    return get_all_sessions()


# Defining session messages endpoint for chat history loading
@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> list:
    logger.info(f"Loading session messages: {session_id[:16]}...")
    return load_session_full(session_id)


# Defining session deletion endpoint
@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str) -> dict:
    delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


# Defining codebase indexing endpoint for ChromaDB ingestion
@app.post("/index")
async def index_endpoint() -> dict:
    logger.info("Triggering codebase re-indexing via API endpoint...")
    try:
        index = index_codebase(CODEBASE_PATH)
        rag_store.ingest_index(index)
        return {
            "status": "success",
            "indexed_files": index["stats"]["total_files"],
            "total_chunks": index["stats"]["total_chunks"]
        }
    except Exception as e:
        logger.error(f"Failing codebase indexing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Defining memories endpoint for saved notes retrieval
@app.get("/memories")
async def get_memories() -> list:
    return [{"content": m} for m in load_memories()]


# Defining health check endpoint for offline verification
@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "online",
        "model": LLM_MODEL,
        "embedding_model": "nomic-embed-text",
        "rag_store": "connected",
        "database": "connected"
    }


# Serving frontend index.html at root endpoint
@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    index_path = Path("frontend/index.html")
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>DevAssist API Running</h1>")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting DevAssist FastAPI server...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)