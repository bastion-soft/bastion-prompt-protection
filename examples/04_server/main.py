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

from bastion_prompt_protection import Guard, __version__

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
    # Warm the ONNX session with one inference so the first real request
    # doesn't pay the cold-start cost.
    _guard.protect("warmup")
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
    prompt: str = Field(..., min_length=1, max_length=32_000)


class ProtectResponse(BaseModel):
    risk: float = Field(..., ge=0.0, le=1.0)
    label: str  # "safe" | "attack"
    stage_reached: str  # "heuristics" | "binary"
    latency_ms: float


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

    result = _guard.protect(req.prompt)
    return ProtectResponse(
        risk=result.risk,
        label=result.label,
        stage_reached=result.stage_reached,
        latency_ms=result.latency_ms,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
