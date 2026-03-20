from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_training_data(path: Path) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    payload = json.loads(path.read_text())
    examples = payload["examples"]
    texts = [item["text"] for item in examples]
    labels = [item["intent"] for item in examples]
    responses = payload["responses"]
    return texts, labels, responses
