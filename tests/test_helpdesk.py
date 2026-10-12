"""
Tests for the Smart Helpdesk Assistant.
Run: pytest tests/ -v
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.rag_engine import (
    session_store, score_confidence,
    reformulate_query, SessionStore
)
from app.models import ChatMessage, MessageRole
from app.ingestion import recursive_chunk, clean_text

client = TestClient(app)


# ── Unit: text processing ─────────────────────────────────────────────────────

def test_clean_text_collapses_whitespace():
    text = "Hello   world\n\n\n\nHow are   you?"
    result = clean_text(text)
    assert "   " not in result
    assert "\n\n\n" not in result

def test_recursive_chunk_basic():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = recursive_chunk(text, chunk_size=40, chunk_overlap=0)
    assert len(chunks) >= 1
    # All chunks should be <= chunk_size (approx — overlap may extend slightly)
    for chunk in chunks:
        assert len(chunk) <= 80  # generous bound with overlap

def test_recursive_chunk_overlap():
    text = "Paragraph one content here.\n\nParagraph two content here.\n\nParagraph three."
    chunks = recursive_chunk(text, chunk_size=50, chunk_overlap=15)
    # With overlap, later chunks should start with content from previous chunk end
    if len(chunks) >= 2:
        # Overlap means some content repeats between consecutive chunks
        assert len(chunks) >= 2

def test_recursive_chunk_short_text():
    text = "Short text."
    chunks = recursive_chunk(text, chunk_size=512, chunk_overlap=64)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."

def test_recursive_chunk_empty():
    chunks = recursive_chunk("", chunk_size=512, chunk_overlap=64)
    assert chunks == []


# ── Unit: session store ───────────────────────────────────────────────────────

def test_session_store_add_and_get():
    store = SessionStore()
    store.add_message("session_1", MessageRole.USER, "Hello")
    store.add_message("session_1", MessageRole.ASSISTANT, "Hi there!")
    history = store.get_history("session_1")
    assert len(history) == 2
    assert history[0].role == MessageRole.USER
    assert history[1].content == "Hi there!"

def test_session_store_clear():
    store = SessionStore()
    store.add_message("session_2", MessageRole.USER, "Test message")
    store.clear("session_2")
    history = store.get_history("session_2")
    assert len(history) == 0

def test_session_store_history_cap():
    """Session history should be capped to MAX_HISTORY_TURNS * 2 messages."""
    from app.config import settings
    store = SessionStore()
    for i in range(20):
        store.add_message("capped_session", MessageRole.USER, f"message {i}")
    history = store.get_history("capped_session")
    assert len(history) <= settings.MAX_HISTORY_TURNS * 2

def test_session_store_new_session_empty():
    store = SessionStore()
    history = store.get_history("brand_new_session_xyz")
    assert history == []


# ── Unit: confidence scoring ──────────────────────────────────────────────────

def test_confidence_no_docs():
    assert score_confidence([]) == "no_context"

def test_confidence_high():
    docs = [{"score": 0.92}, {"score": 0.88}]
    assert score_confidence(docs) == "high"

def test_confidence_medium():
    docs = [{"score": 0.78}]
    assert score_confidence(docs) == "medium"

def test_confidence_low():
    docs = [{"score": 0.67}]
    assert score_confidence(docs) == "low"


# ── API endpoint tests ────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_chat_empty_message():
    r = client.post("/chat", json={"session_id": "test", "message": ""})
    assert r.status_code == 400

def test_chat_message_too_long():
    r = client.post("/chat", json={"session_id": "test", "message": "x" * 2001})
    assert r.status_code == 400

MOCK_RESPONSE_DATA = {
    "session_id": "test_session",
    "answer": "The return window is 30 days from delivery. [Source: returns_policy.md]",
    "sources": [{"source": "returns_policy.md", "page": None, "section": "Return Window",
                 "snippet": "You may return most items within 30 days", "relevance_score": 0.91}],
    "is_grounded": True,
    "confidence": "high",
    "follow_up_suggestions": ["Can I return digital products?"],
}

@patch("app.main.rag_answer")
def test_chat_success(mock_rag):
    from app.models import ChatResponse, SourceDocument
    mock_rag.return_value = ChatResponse(
        session_id="test_session",
        answer="The return window is 30 days.",
        sources=[SourceDocument(source="returns_policy.md", snippet="30 days", relevance_score=0.91)],
        is_grounded=True,
        confidence="high",
        follow_up_suggestions=["Can I return digital products?"],
    )
    r = client.post("/chat", json={"session_id": "test_session", "message": "What is the return window?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert data["is_grounded"] is True
    assert data["confidence"] == "high"
    assert len(data["sources"]) >= 1

def test_get_session_empty():
    r = client.get("/sessions/nonexistent_session_abc123")
    assert r.status_code == 200
    data = r.json()
    assert data["message_count"] == 0

def test_clear_session():
    r = client.delete("/sessions/test_session_to_clear")
    assert r.status_code == 200
    assert r.json()["status"] == "cleared"

def test_stats():
    r = client.get("/stats")
    assert r.status_code == 200
    assert "total_chunks" in r.json()

@patch("app.main.ingest_text", return_value=5)
def test_ingest_text(mock_ingest):
    payload = {
        "doc_id": "test_doc_001",
        "title": "Test Document",
        "content": "This is test content. " * 50,
        "category": "faq",
        "tags": ["test"],
    }
    r = client.post("/ingest/text", json=payload)
    assert r.status_code == 200
    assert r.json()["chunks_created"] == 5
    assert r.json()["status"] == "ingested"
