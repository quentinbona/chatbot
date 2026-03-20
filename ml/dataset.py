from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def encode_labels(labels: List[str]) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = np.array([label_to_index[label] for label in labels], dtype=np.int32)
    return encoded, unique_labels, label_to_index


def train_val_split(
    texts: List[str],
    encoded_labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    return train_test_split(
        texts,
        encoded_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=encoded_labels,
    )
