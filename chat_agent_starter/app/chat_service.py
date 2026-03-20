from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.config import settings
from app.inference import ModelManager
from app.memory import TranscriptStore


@dataclass
class ChatResult:
    reply: str
    predicted_intent: str
    confidence: float
    used_fallback: bool
    context_messages: List[str]


class ChatService:
    def __init__(self, model_manager: ModelManager, store: TranscriptStore) -> None:
        self.model_manager = model_manager
        self.store = store

    def handle_message(self, session_id: str, message: str) -> ChatResult:
        context_messages = self.store.format_context(
            session_id=session_id,
            limit=settings.max_context_turns,
        )

        result = self.model_manager.predict_intent(message)
        use_fallback = result.confidence < settings.intent_confidence_threshold
        reply = self.model_manager.get_response(result.intent, fallback=use_fallback)

        self.store.add_message(session_id, "user", message)
        self.store.add_message(session_id, "assistant", reply)

        return ChatResult(
            reply=reply,
            predicted_intent=result.intent,
            confidence=result.confidence,
            used_fallback=use_fallback,
            context_messages=context_messages,
        )
