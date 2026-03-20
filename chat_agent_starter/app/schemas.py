from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Conversation session id")
    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    predicted_intent: str
    confidence: float
    used_fallback: bool
    context_messages: List[str]


class TrainResponse(BaseModel):
    status: str
    model_dir: str
    labels: List[str]


class HealthResponse(BaseModel):
    status: str
    app_name: str


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_dir: str
    labels: List[str]
    threshold: float
