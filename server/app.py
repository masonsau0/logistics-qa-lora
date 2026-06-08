"""FastAPI inference server.

Endpoints:
  GET  /health        - liveness check
  POST /generate      - non-streaming generation
  POST /generate/stream - server-sent-events streaming

Environment variables:
  ADAPTER_PATH        - directory with the trained LoRA adapter (default: artifacts/checkpoints/final)
  BASE_MODEL          - base HF model id (default: from src/config.py)
  LOAD_IN_4BIT        - "true" or "false" (default: true)
  USE_MOCK            - "true" returns canned responses; useful for CI / local dev (default: false)

Start:
  uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import GEN, MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------- Mock backend (for CI and key-free local dev) ----------


class MockQA:
    """Returns canned responses without loading any model. Used when USE_MOCK=true.
    Makes the server testable in CI and during frontend development without GPU."""

    def __init__(self) -> None:
        self.has_adapter = False

    def answer(self, question: str, **kwargs) -> str:
        return f"[MOCK] Received question: {question!r}. Set USE_MOCK=false and provide a real adapter to get model output."

    def stream(self, question: str, **kwargs):
        text = self.answer(question)
        for word in text.split():
            yield word + " "


# ---------- App state ----------


class AppState:
    qa = None  # populated in lifespan


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    use_mock = os.environ.get("USE_MOCK", "false").lower() == "true"
    if use_mock:
        logger.info("USE_MOCK=true - loading MockQA backend")
        state.qa = MockQA()
    else:
        # Lazy import - keeps the FastAPI app importable in environments where
        # heavy ML deps aren't installed.
        from src.inference import LogisticsQA

        adapter_path = os.environ.get("ADAPTER_PATH", "artifacts/checkpoints/final")
        base_model = os.environ.get("BASE_MODEL", MODEL.name)
        load_in_4bit = os.environ.get("LOAD_IN_4BIT", "true").lower() == "true"

        adapter = adapter_path if os.path.isdir(adapter_path) else None
        if adapter is None:
            logger.warning(
                "ADAPTER_PATH (%s) not found; running base model only",
                adapter_path,
            )
        state.qa = LogisticsQA(
            base_model_name=base_model,
            adapter_path=adapter,
            load_in_4bit=load_in_4bit,
        )
    yield
    state.qa = None


app = FastAPI(
    title="Logistics QA - LoRA fine-tuned Qwen 2.5",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------- Schemas ----------


class GenerateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    max_new_tokens: int = Field(GEN.max_new_tokens, ge=1, le=2048)
    temperature: float = Field(GEN.temperature, ge=0.0, le=2.0)
    top_p: float = Field(GEN.top_p, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    answer: str
    has_adapter: bool


# ---------- Endpoints ----------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "backend": "mock" if isinstance(state.qa, MockQA) else "model"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    if state.qa is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    answer = state.qa.answer(
        req.question,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return GenerateResponse(answer=answer, has_adapter=state.qa.has_adapter)


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest) -> StreamingResponse:
    if state.qa is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    def event_source():
        try:
            for chunk in state.qa.stream(
                req.question,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            ):
                # SSE format: each event prefixed with "data: " and ending in "\n\n"
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:  # noqa: BLE001
            logger.exception("Stream error")
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
