from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


@dataclass
class InferenceResult:
    intent: str
    confidence: float
    probabilities: Dict[str, float]


class ModelManager:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model: tf.keras.Model | None = None
        self.labels: List[str] = []
        self.responses: Dict[str, List[str]] = {}
        self._load_if_available()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and bool(self.labels)

    def _load_if_available(self) -> None:
        metadata_path = self.model_dir / "metadata.json"
        if not self.model_dir.exists() or not metadata_path.exists():
            return

        self.model = tf.keras.models.load_model(self.model_dir)
        metadata = json.loads(metadata_path.read_text())
        self.labels = metadata["labels"]
        self.responses = metadata["responses"]

    def reload(self) -> None:
        self.model = None
        self.labels = []
        self.responses = {}
        self._load_if_available()

    def predict_intent(self, text: str) -> InferenceResult:
        if not self.model or not self.labels:
            raise RuntimeError("Model is not loaded. Train or export a model first.")

        probs = self.model.predict(np.array([text]), verbose=0)[0]
        best_idx = int(np.argmax(probs))
        intent = self.labels[best_idx]
        confidence = float(probs[best_idx])
        probabilities = {label: float(prob) for label, prob in zip(self.labels, probs)}
        return InferenceResult(intent=intent, confidence=confidence, probabilities=probabilities)

    def get_response(self, intent: str, fallback: bool = False) -> str:
        if fallback:
            return self.responses.get("__fallback__", ["I don't know yet."])[0]
        return self.responses.get(intent, self.responses.get("__fallback__", ["I don't know yet."]))[0]
