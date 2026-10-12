"""
RAG Engine — retrieval-augmented generation with conversation memory.

Features:
  1. Conversation history — multi-turn context via session store
  2. Query reformulation — uses conversation history to make follow-up
     questions self-contained before retrieval
  3. Grounded generation — strict prompt engineering to prevent hallucination
  4. Confidence scoring — classifies answer confidence based on retrieval scores
  5. Follow-up suggestions — generates 2-3 relevant follow-up questions
"""

import json
import logging
from collections import defaultdict
from typing import Optional

from app.config import settings
from app.models import ChatMessage, MessageRole, SourceDocument, ChatResponse
from app.ingestion import retrieve

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise and helpful customer support assistant.

RULES — follow these strictly:
1. Answer ONLY using the Context provided below. Do not use any prior knowledge.
2. If the Context does not contain enough information, respond:
   "I don't have specific information about that in our documentation. 
    Please contact support@company.com or call 1-800-SUPPORT."
3. Be concise and direct. Use bullet points for multi-step answers.
4. Always cite your source: add [Source: filename] at the end of claims.
5. Never speculate, guess, or invent information.
6. If asked something outside company products/policies, politely decline.
"""

QUERY_REFORMULATION_PROMPT = """Given the conversation history and a follow-up question, 
rewrite the follow-up question to be standalone and self-contained. 
If it is already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""

GENERATION_PROMPT = """Context from our documentation:
---
{context}
---

Question: {question}

Answer (cite sources, be concise):"""

FOLLOW_UP_PROMPT = """Based on this Q&A exchange, suggest 2-3 short follow-up questions 
the user might want to ask next. Return as a JSON list of strings.
Keep each question under 12 words.

Q: {question}
A: {answer}

Follow-ups:"""


# ── Session store (in-memory) ─────────────────────────────────────────────────

class SessionStore:
    """
    In-memory conversation history store.

    In production, replace with Redis or a database for persistence across
    server restarts and horizontal scaling.

    Key design: we store the full message history per session,
    but cap it at MAX_HISTORY_TURNS to prevent context window overflow.
    """

    def __init__(self):
        self._sessions: dict[str, list[ChatMessage]] = defaultdict(list)

    def get_history(self, session_id: str) -> list[ChatMessage]:
        return self._sessions[session_id][-settings.MAX_HISTORY_TURNS * 2:]

    def add_message(self, session_id: str, role: MessageRole, content: str):
        self._sessions[session_id].append(ChatMessage(role=role, content=content))

    def get_summary(self, session_id: str) -> dict:
        msgs = self._sessions[session_id]
        return {
            "session_id":    session_id,
            "message_count": len(msgs),
            "messages":      [m.model_dump() for m in msgs],
        }

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)

    def all_sessions(self) -> list[str]:
        return list(self._sessions.keys())


session_store = SessionStore()


# ── LLM caller ────────────────────────────────────────────────────────────────

def _call_llm(messages: list[dict], temperature: float = None) -> str:
    """Route to configured LLM provider."""
    temp = temperature if temperature is not None else settings.TEMPERATURE

    if settings.LLM_PROVIDER.lower() == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.LLM_MODEL)
        # Gemini uses different message format
        history = []
        for m in messages[:-1]:
            if m["role"] == "user":
                history.append({"role": "user", "parts": [m["content"]]})
            elif m["role"] == "assistant":
                history.append({"role": "model", "parts": [m["content"]]})
        chat = model.start_chat(history=history)
        resp = chat.send_message(
            messages[-1]["content"],
            generation_config={"temperature": temp, "max_output_tokens": 1024},
        )
        return resp.text
    else:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            temperature=temp,
            max_tokens=1024,
        )
        return response.choices[0].message.content


# ── Query reformulation ───────────────────────────────────────────────────────

def reformulate_query(question: str, history: list[ChatMessage]) -> str:
    """
    If the conversation has history, reformulate the question to be standalone.

    Example:
      History: "What is the return policy?"
               "30 days from purchase."
      Follow-up: "Does it apply to digital goods too?"
      Reformulated: "Does the return policy apply to digital goods?"

    Without reformulation, the retriever would search for "digital goods"
    without the crucial context that we're asking about return policy.
    """
    if not history:
        return question  # first turn — no reformulation needed

    history_text = "\n".join(
        f"{m.role.value.capitalize()}: {m.content}" for m in history[-4:]
    )
    prompt = QUERY_REFORMULATION_PROMPT.format(
        history=history_text,
        question=question,
    )
    messages = [{"role": "user", "content": prompt}]
    reformulated = _call_llm(messages, temperature=0.0).strip()
    logger.debug(f"Reformulated: '{question}' → '{reformulated}'")
    return reformulated


# ── Confidence scoring ────────────────────────────────────────────────────────

def score_confidence(retrieved_docs: list[dict]) -> str:
    """
    Classify retrieval confidence based on similarity scores.

    Thresholds (approximate, adjust based on your embedding model):
      high   : top doc ≥ 0.85 and multiple docs ≥ 0.75
      medium : top doc ≥ 0.75
      low    : some docs found but below 0.75
      no_context: no docs above threshold
    """
    if not retrieved_docs:
        return "no_context"
    top_score = retrieved_docs[0]["score"]
    if top_score >= 0.85 and len(retrieved_docs) >= 2:
        return "high"
    elif top_score >= 0.75:
        return "medium"
    else:
        return "low"


# ── Follow-up generation ──────────────────────────────────────────────────────

def generate_follow_ups(question: str, answer: str) -> list[str]:
    """Generate 2-3 relevant follow-up question suggestions."""
    try:
        prompt = FOLLOW_UP_PROMPT.format(question=question, answer=answer)
        raw = _call_llm([{"role": "user", "content": prompt}], temperature=0.5)
        # Clean up JSON fences
        raw = raw.strip().lstrip("```json").rstrip("```").strip()
        suggestions = json.loads(raw)
        return [s for s in suggestions if isinstance(s, str)][:3]
    except Exception as e:
        logger.warning(f"Follow-up generation failed: {e}")
        return []


# ── Main RAG function ─────────────────────────────────────────────────────────

def rag_answer(session_id: str, user_message: str) -> ChatResponse:
    """
    Full RAG pipeline with conversation memory:

    1. Get conversation history for this session
    2. Reformulate the query to be self-contained (handles follow-ups)
    3. Retrieve relevant chunks from vector store
    4. Build context string from retrieved chunks
    5. Generate grounded answer using LLM
    6. Store messages in session history
    7. Generate follow-up suggestions
    8. Return structured response with sources and confidence
    """

    # Step 1: History
    history = session_store.get_history(session_id)

    # Step 2: Query reformulation
    reformulated = reformulate_query(user_message, history)

    # Step 3: Retrieve
    retrieved_docs = retrieve(reformulated)
    logger.info(f"Retrieved {len(retrieved_docs)} docs for session={session_id}")

    # Step 4: Build context
    if retrieved_docs:
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc["metadata"]
            source_label = f"[Source: {meta.get('source','unknown')}, score={doc['score']:.2f}]"
            context_parts.append(f"{source_label}\n{doc['text']}")
        context = "\n\n---\n\n".join(context_parts)
        is_grounded = True
    else:
        context = "No relevant documentation found."
        is_grounded = False

    # Step 5: Generate answer
    messages_for_llm = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    # Include recent conversation history so the LLM maintains context
    for msg in history[-6:]:
        messages_for_llm.append({
            "role": msg.role.value,
            "content": msg.content,
        })
    messages_for_llm.append({
        "role": "user",
        "content": GENERATION_PROMPT.format(context=context, question=reformulated),
    })

    answer = _call_llm(messages_for_llm)

    # Step 6: Store in session
    session_store.add_message(session_id, MessageRole.USER, user_message)
    session_store.add_message(session_id, MessageRole.ASSISTANT, answer)

    # Step 7: Build source objects
    sources = [
        SourceDocument(
            source        = doc["metadata"].get("source", "unknown"),
            section       = doc["metadata"].get("title"),
            snippet       = doc["text"][:200],
            relevance_score = doc["score"],
        )
        for doc in retrieved_docs
    ]

    # Step 8: Follow-ups (only when grounded)
    follow_ups = generate_follow_ups(user_message, answer) if is_grounded else []

    return ChatResponse(
        session_id           = session_id,
        answer               = answer,
        sources              = sources,
        is_grounded          = is_grounded,
        confidence           = score_confidence(retrieved_docs),
        follow_up_suggestions = follow_ups,
    )
