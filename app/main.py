"""
Smart Helpdesk Assistant — FastAPI Application
===============================================
Endpoints:
  POST /chat                   — send a message, get RAG-grounded reply
  GET  /sessions/{id}          — view conversation history
  DELETE /sessions/{id}        — clear conversation history
  POST /ingest/text            — ingest a document from text
  POST /ingest/directory       — ingest all docs from configured directory
  GET  /stats                  — index statistics
  GET  /health                 — health check
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import settings
from app.models import (
    ChatRequest, ChatResponse,
    IngestRequest, IngestResponse,
    IndexStatsResponse, SessionSummary,
)
from app.rag_engine import rag_answer, session_store
from app.ingestion import ingest_text, ingest_directory, get_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Helpdesk Assistant",
    description=(
        "Conversational RAG-powered helpdesk assistant. "
        "Answers questions grounded in your documentation with conversation memory."
    ),
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """<html><body style="font-family:sans-serif;max-width:640px;margin:40px auto">
    <h1>🤖 Smart Helpdesk Assistant</h1>
    <p>RAG-powered customer support with conversation memory.</p>
    <p><a href="/docs">📖 Swagger UI</a> &nbsp;|&nbsp; <a href="/redoc">📄 ReDoc</a></p>
    <p><b>Quick start:</b><br>
    1. POST /ingest/directory to load your docs<br>
    2. POST /chat with a session_id and message</p>
    </body></html>"""


@app.get("/health")
async def health():
    s = get_stats()
    return {"status": "ok", **s}


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Send a message to the helpdesk assistant.

    The assistant:
    - Retrieves relevant documentation chunks
    - Reformulates follow-up questions using conversation history
    - Generates grounded answers with source citations
    - Suggests follow-up questions

    Use the same session_id across requests to maintain conversation context.
    """
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty.")
    if len(req.message) > 2000:
        raise HTTPException(400, "Message too long (max 2000 chars).")
    try:
        return rag_answer(req.session_id, req.message)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(500, f"Assistant error: {str(e)}")


# ── Session management ────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}", response_model=SessionSummary)
async def get_session(session_id: str):
    """Retrieve conversation history for a session."""
    summary = session_store.get_summary(session_id)
    return SessionSummary(**summary)


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    session_store.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/sessions")
async def list_sessions():
    """List all active session IDs."""
    return {"sessions": session_store.all_sessions()}


# ── Ingestion ─────────────────────────────────────────────────────────────────

@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text_endpoint(req: IngestRequest):
    """Ingest a document from raw text content."""
    try:
        n = ingest_text(
            text=req.content,
            doc_id=req.doc_id,
            title=req.title,
            category=req.category,
            tags=req.tags,
        )
        return IngestResponse(
            doc_id=req.doc_id,
            chunks_created=n,
            status="ingested" if n > 0 else "empty",
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ingest/directory")
async def ingest_dir_endpoint():
    """
    Ingest all documents from the configured DOCS_PATH directory.
    Supports .md, .txt, and .pdf files.
    """
    try:
        results = ingest_directory(settings.DOCS_PATH)
        return {
            "status": "completed",
            "docs_path": settings.DOCS_PATH,
            "ingested": results["success"],
            "failed": results["failed"],
            "total_chunks": sum(r["chunks"] for r in results["success"]),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.get("/stats", response_model=IndexStatsResponse)
async def stats():
    return IndexStatsResponse(**get_stats())
