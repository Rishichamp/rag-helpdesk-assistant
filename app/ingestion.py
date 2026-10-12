"""
Document Ingestion Pipeline for the Smart Helpdesk Assistant.

Supports:
  - Plain text / markdown files
  - PDF files (via pypdf)
  - Direct text ingestion via API

Pipeline: Load → Clean → Chunk → Embed → Store in ChromaDB
"""

import os
import re
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

logger = logging.getLogger(__name__)


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove excessive whitespace, fix encoding artifacts."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


# ── Chunking ──────────────────────────────────────────────────────────────────

def recursive_chunk(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[str]:
    """
    Recursive character text splitting.
    Tries to split on paragraph → sentence → word boundaries.
    """
    chunk_size    = chunk_size    or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps or len(text) <= chunk_size:
            return [text] if text.strip() else []
        sep = seps[0]
        if sep == "":
            # No natural boundary left (str.split("") raises), so cut at the size limit.
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size) if text[i:i + chunk_size].strip()]
        splits = text.split(sep)
        chunks, current = [], ""
        for s in splits:
            candidate = current + (sep if current else "") + s
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(s) > chunk_size:
                    chunks.extend(_split(s, seps[1:]))
                    current = ""
                else:
                    current = s
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, separators)

    # Apply overlap: each chunk starts with the tail of the previous
    if chunk_overlap <= 0 or len(raw_chunks) <= 1:
        return [c for c in raw_chunks if c.strip()]

    overlapped = [raw_chunks[0]]
    for chunk in raw_chunks[1:]:
        prev_tail = overlapped[-1][-chunk_overlap:] if len(overlapped[-1]) > chunk_overlap else overlapped[-1]
        overlapped.append(prev_tail + " " + chunk)

    return [c.strip() for c in overlapped if c.strip()]


# ── Embedding helper ──────────────────────────────────────────────────────────

def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings using the configured provider."""
    if settings.LLM_PROVIDER.lower() == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        result = genai.embed_content(
            model=f"models/{settings.EMBEDDING_MODEL}",
            content=texts,
            task_type="retrieval_document",
        )
        return result["embedding"] if isinstance(texts, str) else result["embedding"]
    else:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.embeddings.create(
            input=[t if t.strip() else "empty" for t in texts],
            model=settings.EMBEDDING_MODEL,
        )
        return [d.embedding for d in response.data]


# ── ChromaDB collection ───────────────────────────────────────────────────────

_chroma_client = None
_collection = None

def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        Path(settings.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Ingestion functions ───────────────────────────────────────────────────────

def ingest_text(
    text: str,
    doc_id: str,
    title: str,
    category: Optional[str] = None,
    tags: Optional[list[str]] = None,
    source_path: Optional[str] = None,
) -> int:
    """
    Ingest a text document into the vector store.
    Returns the number of chunks created.
    """
    collection = get_collection()
    cleaned    = clean_text(text)
    chunks     = recursive_chunk(cleaned)

    if not chunks:
        logger.warning(f"No chunks created for doc_id={doc_id}")
        return 0

    # Batch embed all chunks
    embeddings = _get_embeddings(chunks)

    ids, docs, metas, embeds = [], [], [], []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{doc_id}_chunk_{i}"
        ids.append(chunk_id)
        docs.append(chunk)
        embeds.append(embedding)
        metas.append({
            "doc_id":   doc_id,
            "title":    title,
            "source":   source_path or doc_id,
            "category": category or "general",
            "tags":     json.dumps(tags or []),
            "chunk_idx": i,
            "total_chunks": len(chunks),
        })

    collection.upsert(ids=ids, embeddings=embeds, documents=docs, metadatas=metas)
    logger.info(f"Ingested '{title}' → {len(chunks)} chunks (doc_id={doc_id})")
    return len(chunks)


def ingest_markdown_file(filepath: str) -> int:
    """Ingest a markdown or text file."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    doc_id = hashlib.md5(str(path).encode()).hexdigest()[:12]
    return ingest_text(
        text=text,
        doc_id=doc_id,
        title=path.stem.replace("_", " ").title(),
        source_path=path.name,
    )


def ingest_pdf_file(filepath: str) -> int:
    """Ingest a PDF file page by page."""
    try:
        import pypdf
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")

    path = Path(filepath)
    reader = pypdf.PdfReader(str(path))
    full_text = "\n\n".join(
        page.extract_text() or "" for page in reader.pages
    )
    doc_id = hashlib.md5(str(path).encode()).hexdigest()[:12]
    return ingest_text(
        text=full_text,
        doc_id=doc_id,
        title=path.stem.replace("_", " ").title(),
        source_path=path.name,
    )


def ingest_directory(directory: str) -> dict:
    """Ingest all supported documents from a directory."""
    results = {"success": [], "failed": []}
    for path in Path(directory).rglob("*"):
        if path.suffix.lower() in (".md", ".txt"):
            try:
                n = ingest_markdown_file(str(path))
                results["success"].append({"file": path.name, "chunks": n})
            except Exception as e:
                results["failed"].append({"file": path.name, "error": str(e)})
        elif path.suffix.lower() == ".pdf":
            try:
                n = ingest_pdf_file(str(path))
                results["success"].append({"file": path.name, "chunks": n})
            except Exception as e:
                results["failed"].append({"file": path.name, "error": str(e)})
    return results


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = None,
    category_filter: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve the most semantically similar chunks to the query.
    Returns list of dicts with: text, metadata, score.
    """
    top_k = top_k or settings.TOP_K
    collection = get_collection()

    if collection.count() == 0:
        return []

    # Embed query
    query_embedding = _get_embeddings([query])[0]

    query_kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances", "metadatas"],
    )
    if category_filter:
        query_kwargs["where"] = {"category": category_filter}

    results = collection.query(**query_kwargs)

    docs = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        score    = round(1.0 - distance, 4)  # cosine similarity
        if score < settings.SIMILARITY_THRESHOLD:
            continue
        docs.append({
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score":    score,
        })

    return sorted(docs, key=lambda d: d["score"], reverse=True)


def get_stats() -> dict:
    col = get_collection()
    return {
        "total_chunks":    col.count(),
        "collection_name": settings.COLLECTION_NAME,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model":       settings.LLM_MODEL,
        "docs_path":       settings.DOCS_PATH,
    }
