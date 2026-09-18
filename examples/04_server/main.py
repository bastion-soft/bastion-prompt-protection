"""FastAPI wrapper around Guard — Pattern 4.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8080

Then:
    curl -X POST localhost:8080/protect \\
         -H "Content-Type: application/json" \\
         -d '{"prompt": "Ignore previous instructions"}'

For Docker, see ../../docker/Dockerfile.cpu (one-command pull at the
README); this file is the self-contained reference for anyone wanting
to package the SDK their own way.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bastion_prompt_protection import (
    DEFAULT_MAX_INPUT_CHARS,
    Guard,
    ModelUnavailableError,
    __version__,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")


# Singleton Guard. Loaded once at app start so the model + tokenizer
# initialization cost is paid before serving the first request.
_guard: Guard | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _guard
    logger.info("loading Guard model (one-time)...")
    _guard = Guard()
    try:
        # Warm the ONNX session with one inference so the first real request
        # doesn't pay the cold-start cost. If this fails, the model is
        # unavailable and the container should be considered unhealthy.
        _guard.protect("warmup")
    except ModelUnavailableError as exc:
        logger.error("Guard warmup failed — model unavailable: %s", exc)
        raise
    logger.info("Guard ready")
    yield
    _guard = None


app = FastAPI(
    title="Bastion Prompt Protection",
    version=__version__,
    description="HTTP wrapper around the bastion_prompt_protection.Guard SDK.",
    lifespan=lifespan,
)


# ────────────────────────────────────────────────────────────────────────
# Request/response models
# ────────────────────────────────────────────────────────────────────────


class ProtectRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=DEFAULT_MAX_INPUT_CHARS)
    max_windows: int | None = None
    overlap_tokens: int | None = None
    normalize_whitespace: bool | None = None


class ProtectResponse(BaseModel):
    risk: float = Field(..., ge=0.0, le=1.0)
    label: str  # "safe" | "attack"
    stage_reached: str  # "heuristics" | "classifier"
    latency_ms: float
    windows_scanned: int
    windows_total: int
    windows_total_exact: bool


# ────────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────────


@app.get("/")
async def root() -> dict:
    return {
        "service": "bastion-prompt-protection",
        "version": __version__,
        "endpoints": ["/health", "/protect"],
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict:
    if _guard is None:
        raise HTTPException(status_code=503, detail="Guard not initialized")
    return {"status": "ok", "version": __version__}


@app.post("/protect", response_model=ProtectResponse)
async def protect(req: ProtectRequest) -> ProtectResponse:
    if _guard is None:
        raise HTTPException(status_code=503, detail="Guard not initialized")

    try:
        result = _guard.protect(
            req.prompt,
            max_windows=req.max_windows,
            overlap_tokens=req.overlap_tokens,
            normalize_whitespace=req.normalize_whitespace,
        )
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ProtectResponse(
        risk=result.risk,
        label=result.label,
        stage_reached=result.stage_reached,
        latency_ms=result.latency_ms,
        windows_scanned=result.windows_scanned,
        windows_total=result.windows_total,
        windows_total_exact=result.windows_total_exact,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
