from __future__ import annotations

import subprocess
import sys
import schemas
import memory
import config
import uvicorn
from fastapi import FastAPI, HTTPException

from app.chat_service import ChatService
from app.config import settings
from app.inference import ModelManager
from app.memory import TranscriptStore
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ModelInfoResponse,
    TrainResponse,
)

app = FastAPI(title=settings.app_name)

store = TranscriptStore(settings.db_path)
model_manager = ModelManager(settings.model_dir)
chat_service = ChatService(model_manager=model_manager, store=store)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app_name=settings.app_name)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_loaded=model_manager.is_loaded,
        model_dir=str(settings.model_dir),
        labels=model_manager.labels,
        threshold=settings.intent_confidence_threshold,
    )


@app.post("/train", response_model=TrainResponse)
def train_model() -> TrainResponse:
    try:
        subprocess.run([sys.executable, "-m", "ml.train"], check=True)
        model_manager.reload()
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    return TrainResponse(
        status="trained",
        model_dir=str(settings.model_dir),
        labels=model_manager.labels,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Call /train or place an exported model in models/latest.",
        )

    result = chat_service.handle_message(
        session_id=request.session_id,
        message=request.message,
    )

    return ChatResponse(
        session_id=request.session_id,
        reply=result.reply,
        predicted_intent=result.predicted_intent,
        confidence=result.confidence,
        used_fallback=result.used_fallback,
        context_messages=result.context_messages,
    )


if __name__ == "__main__":
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
