from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import tensorflow as tf


def export_model(
    model: tf.keras.Model,
    output_dir: Path,
    labels: List[str],
    responses: Dict[str, List[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.export(str(output_dir))
    metadata = {"labels": labels, "responses": responses}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
