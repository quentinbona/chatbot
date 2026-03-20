from __future__ import annotations

import tensorflow as tf


def build_model(num_classes: int) -> tf.keras.Model:
    text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="text")
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=5000,
        output_mode="int",
        output_sequence_length=32,
        standardize="lower_and_strip_punctuation",
    )
    embedding = tf.keras.layers.Embedding(input_dim=5000, output_dim=64)
    pool = tf.keras.layers.GlobalAveragePooling1D()
    dense = tf.keras.layers.Dense(64, activation="relu")
    dropout = tf.keras.layers.Dropout(0.2)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")

    x = vectorizer(text_input)
    x = embedding(x)
    x = pool(x)
    x = dense(x)
    x = dropout(x)
    y = output(x)

    model = tf.keras.Model(inputs=text_input, outputs=y)
    model.vectorizer = vectorizer  # type: ignore[attr-defined]
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
