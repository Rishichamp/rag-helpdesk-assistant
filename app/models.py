"""Pydantic models for the Smart Helpdesk Assistant."""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class MessageRole(str, Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class SourceDocument(BaseModel):
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: str
    relevance_score: float


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier for conversation memory")
    message: str    = Field(..., description="User's question or message")
    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "user_123_session_1",
                "message": "What is your refund policy for digital products?"
            }
        }
    }


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceDocument]
    is_grounded: bool        = Field(..., description="True if answer is based on retrieved docs")
    confidence: str          = Field(..., description="high / medium / low / no_context")
    follow_up_suggestions: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    """Request to ingest a document from text content."""
    doc_id: str
    title: str
    content: str
    category: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    doc_id: str
    chunks_created: int
    status: str


class IndexStatsResponse(BaseModel):
    total_chunks: int
    collection_name: str
    embedding_model: str
    llm_model: str
    docs_path: str


class SessionSummary(BaseModel):
    session_id: str
    message_count: int
    messages: list[ChatMessage]
