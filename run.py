"""Entry point. Usage: python run.py"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"\n🤖  Smart Helpdesk Assistant")
    print(f"   LLM model   : {settings.LLM_MODEL}")
    print(f"   Embed model : {settings.EMBEDDING_MODEL}")
    print(f"   Docs path   : {settings.DOCS_PATH}")
    print(f"   Chunk size  : {settings.CHUNK_SIZE} tokens, overlap {settings.CHUNK_OVERLAP}")
    print(f"   Top-k       : {settings.TOP_K}")
    print(f"   Docs        : http://127.0.0.1:{settings.APP_PORT}/docs")
    print(f"\n   Quick start:")
    print(f"   1) POST /ingest/directory  — load all docs from {settings.DOCS_PATH}")
    print(f"   2) POST /chat              — start chatting\n")
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
