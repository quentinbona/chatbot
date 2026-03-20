from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import classification_report


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> Dict[str, str]:
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        zero_division=0,
        output_dict=False,
    )
    return {"classification_report": report}
