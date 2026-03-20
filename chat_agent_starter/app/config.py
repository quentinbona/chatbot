from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "TensorFlow Chat Agent")
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    model_dir: Path = Path(os.getenv("MODEL_DIR", "models/latest"))
    db_path: Path = Path(os.getenv("DB_PATH", "data/transcripts/chat_agent.db"))
    max_context_turns: int = int(os.getenv("MAX_CONTEXT_TURNS", "6"))
    intent_confidence_threshold: float = float(
        os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.55")
    )


settings = Settings()
