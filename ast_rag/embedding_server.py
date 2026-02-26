"""
embedding_server.py - Lightweight HTTP server for computing text embeddings.

Run this on the GPU machine. AST-RAG on your main server then calls it via HTTP
instead of loading the model locally.

Usage on the GPU machine:
    pip install fastapi uvicorn sentence-transformers torch
    python -m ast_rag.embedding_server --model BAAI/bge-m3 --device cuda --port 8765

Or with uvicorn directly:
    uvicorn ast_rag.embedding_server:app --host 0.0.0.0 --port 8765

API endpoints:
    GET  /health          → {"status": "ok", "model": "...", "dim": 1024}
    POST /embed           → {"texts": [...], "normalize": true}
                            → {"embeddings": [[...], ...], "dim": 1024}

Configure AST-RAG to use this server:
    In ast_rag_config.json:
        {"embedding": {"remote_url": "http://gpu-host:8765"}}
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model holder (loaded once at startup)
# ---------------------------------------------------------------------------

_MODEL: Optional[SentenceTransformer] = None
_MODEL_NAME: str = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
_DEVICE: str = os.environ.get("EMBED_DEVICE", "cuda")
_DIM: int = 0


def _get_model() -> SentenceTransformer:
    global _MODEL, _DIM
    if _MODEL is None:
        logger.info("Loading model '%s' on device '%s'...", _MODEL_NAME, _DEVICE)
        _MODEL = SentenceTransformer(_MODEL_NAME, device=_DEVICE)
        _DIM = _MODEL.get_sentence_embedding_dimension() or 0
        logger.info("Model loaded. Embedding dimension: %d", _DIM)
    return _MODEL


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AST-RAG Embedding Server",
    description="HTTP API for computing sentence embeddings on a GPU machine.",
    version="0.1.0",
)


@app.on_event("startup")
async def _startup() -> None:
    """Pre-load the model when the server starts so first request isn't slow."""
    _get_model()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Texts to embed")
    normalize: bool = Field(True, description="L2-normalize output vectors")
    batch_size: int = Field(64, description="Encode batch size")


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int
    count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    dim: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — also confirms the model is loaded."""
    model = _get_model()
    dim = model.get_sentence_embedding_dimension() or _DIM
    return HealthResponse(
        status="ok",
        model=_MODEL_NAME,
        device=_DEVICE,
        dim=dim,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Encode a list of texts and return their embedding vectors.

    The returned vectors are float32 and optionally L2-normalized.
    Suitable for cosine similarity search with Qdrant.
    """
    if not request.texts:
        raise HTTPException(status_code=422, detail="texts must be non-empty")

    model = _get_model()

    try:
        vecs: np.ndarray = model.encode(
            request.texts,
            normalize_embeddings=request.normalize,
            batch_size=request.batch_size,
            show_progress_bar=False,
        )  # type: ignore[assignment]
    except Exception as exc:
        logger.error("Encoding failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Encoding error: {exc}") from exc

    dim = vecs.shape[1] if vecs.ndim == 2 else len(vecs)
    return EmbedResponse(
        embeddings=vecs.tolist(),
        dim=dim,
        count=len(request.texts),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the embedding server from the command line.

    Example:
        python -m ast_rag.embedding_server --model BAAI/bge-m3 --device cuda --port 8765
    """
    import argparse

    parser = argparse.ArgumentParser(description="AST-RAG embedding HTTP server")
    parser.add_argument("--model",  default="BAAI/bge-m3",  help="HuggingFace model name")
    parser.add_argument("--device", default="cuda",          help="Torch device (cuda/cpu)")
    parser.add_argument("--host",   default="0.0.0.0",       help="Bind host")
    parser.add_argument("--port",   type=int, default=8765,  help="Port to listen on")
    parser.add_argument("--workers", type=int, default=1,    help="Number of uvicorn workers")
    args = parser.parse_args()

    # Push config into module-level globals before startup
    global _MODEL_NAME, _DEVICE
    _MODEL_NAME = args.model
    _DEVICE = args.device
    os.environ["EMBED_MODEL"] = args.model
    os.environ["EMBED_DEVICE"] = args.device

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "ast_rag.embedding_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
