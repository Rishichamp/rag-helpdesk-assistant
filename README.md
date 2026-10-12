# RAG Helpdesk Assistant

[![tests](https://github.com/Rishichamp/rag-helpdesk-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/Rishichamp/rag-helpdesk-assistant/actions/workflows/tests.yml)

A customer helpdesk assistant that answers questions from your own policy documents (shipping, returns, accounts and payments) using retrieval-augmented generation. It keeps multi-turn context and refuses to make things up when the documents don't cover a question.

## Features

- **Ingestion pipeline:** Markdown, text and PDF files go through load → clean → chunk → embed → store in ChromaDB. Recursive splitting works on paragraphs, then sentences, then words, with overlap between chunks.
- **Multi-turn memory:** a per-session conversation history.
- **Query reformulation:** follow-up questions such as "what about international orders?" are rewritten into standalone questions before retrieval.
- **Grounded answers:** strict prompting restricts answers to the retrieved context, and every answer includes its sources.
- **Confidence level:** each answer is labelled high, medium or low based on how well the retrieved passages match.
- **Provider choice:** OpenAI (`gpt-4o-mini`) or Google Gemini.
- **Session API:** view, list and clear conversations.

## Tech stack

Python 3.11 · FastAPI · ChromaDB · OpenAI / Gemini · pypdf · Pydantic · pytest

## Project structure

```
app/
  main.py         FastAPI app and endpoints
  ingestion.py    load, clean, chunk, embed and store documents
  rag_engine.py   reformulation, retrieval and grounded generation
  models.py       request/response schemas
  config.py       settings from .env
sample_docs/      shipping, returns and account/payment policies
tests/            21 tests
```

## Getting started

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # then add OPENAI_API_KEY (or GEMINI_API_KEY with LLM_PROVIDER=gemini)
python run.py                  # http://127.0.0.1:8000/docs
```

Then call `POST /ingest/directory` once to index `sample_docs/`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | Send a message and get a grounded reply with sources |
| GET | `/sessions` | List sessions |
| GET | `/sessions/{session_id}` | View a conversation |
| DELETE | `/sessions/{session_id}` | Clear a conversation |
| POST | `/ingest/text` | Ingest a document from raw text |
| POST | `/ingest/directory` | Ingest every document in `DOCS_PATH` |
| GET | `/stats` | Index statistics |
| GET | `/health` | Health check |

## Tests

```bash
pytest tests/ -q
```

21 tests covering text cleaning, chunking, session memory, confidence scoring and the API, with LLM and embedding calls mocked.

## Notes

- Sessions are held in memory, so they reset when the server restarts.
- Built as part of Module 2 of a Generative AI engineering programme.
