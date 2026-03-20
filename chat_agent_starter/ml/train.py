from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from app.config import settings
from ml.dataset import encode_labels, train_val_split
from ml.evaluate import evaluate_predictions
from ml.export import export_model
from ml.model import build_model
from ml.preprocess import load_training_data


DATA_PATH = Path("data/raw/training_data.json")


def main() -> None:
    texts, labels, responses = load_training_data(DATA_PATH)
    y, unique_labels, _ = encode_labels(labels)
    x_train, x_val, y_train, y_val = train_val_split(texts, y)

    model = build_model(num_classes=len(unique_labels))
    model.vectorizer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(8))  # type: ignore[attr-defined]

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    model.fit(
        np.array(x_train),
        y_train,
        validation_data=(np.array(x_val), y_val),
        epochs=20,
        batch_size=8,
        callbacks=callbacks,
        verbose=1,
    )

    probs = model.predict(np.array(x_val), verbose=0)
    preds = np.argmax(probs, axis=1)
    report = evaluate_predictions(y_val, preds, unique_labels)
    print(report["classification_report"])

    export_model(
        model=model,
        output_dir=settings.model_dir,
        labels=unique_labels,
        responses=responses,
    )
    print(f"Exported model to {settings.model_dir}")


if __name__ == "__main__":
    main()
