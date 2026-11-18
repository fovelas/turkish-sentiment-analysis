from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.keras"
GRAPH_PATH = PROJECT_ROOT / "reports" / "graphic.jpeg"

LABEL_TO_ID = {"negative": 0, "positive": 1}
MAX_TOKENS = 10_000
SEQUENCE_LENGTH = 23
RANDOM_SEED = 42


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    dataset = pd.read_csv(DATASET_PATH)
    required_columns = {"text", "sentiment"}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"data/dataset.csv is missing required columns: {missing}")

    dataset = dataset.dropna(subset=["text", "sentiment"]).copy()
    dataset["text"] = dataset["text"].astype("string").str.strip()
    dataset["sentiment"] = (
        dataset["sentiment"].astype("string").str.strip().str.lower()
    )
    dataset = dataset.loc[dataset["text"].ne("")].copy()

    unknown_labels = set(dataset["sentiment"].unique()) - set(LABEL_TO_ID)
    if unknown_labels:
        labels = ", ".join(sorted(unknown_labels))
        raise ValueError(f"data/dataset.csv contains unsupported labels: {labels}")

    labels = dataset["sentiment"].map(LABEL_TO_ID).to_numpy(dtype=np.int32)
    texts = dataset["text"].to_numpy(dtype=str)
    class_counts = pd.Series(labels).value_counts()
    if class_counts.reindex([0, 1], fill_value=0).min() < 4:
        raise ValueError("Each sentiment class must contain at least four rows.")

    return texts, labels


def create_model(training_texts: np.ndarray) -> keras.Model:
    vocabulary_builder = keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        standardize="lower_and_strip_punctuation",
    )
    vocabulary_builder.adapt(tf.constant(training_texts, dtype=tf.string))
    vocabulary = vocabulary_builder.get_vocabulary()[2:]

    vectorizer = keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        standardize="lower_and_strip_punctuation",
        vocabulary=vocabulary,
    )

    text_input = keras.Input(shape=(), dtype="string", name="text")
    tokens = vectorizer(text_input)
    values = keras.layers.Embedding(
        input_dim=MAX_TOKENS,
        output_dim=50,
        mask_zero=True,
        name="embedding_layer",
    )(tokens)
    values = keras.layers.LSTM(units=16, return_sequences=True)(values)
    values = keras.layers.LSTM(units=8, return_sequences=True)(values)
    values = keras.layers.LSTM(units=4)(values)
    output = keras.layers.Dense(2, activation="softmax", name="sentiment")(values)

    model = keras.Model(inputs=text_input, outputs=output)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


def train_model() -> keras.Model:
    keras.utils.set_random_seed(RANDOM_SEED)
    texts, labels = load_dataset()

    training_texts, test_texts, training_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    training_texts, validation_texts, training_labels, validation_labels = (
        train_test_split(
            training_texts,
            training_labels,
            test_size=0.25,
            random_state=RANDOM_SEED,
            stratify=training_labels,
        )
    )

    model = create_model(training_texts)
    training_texts = tf.constant(training_texts, dtype=tf.string)
    validation_texts = tf.constant(validation_texts, dtype=tf.string)
    test_texts = tf.constant(test_texts, dtype=tf.string)
    training_labels = tf.constant(training_labels, dtype=tf.int32)
    validation_labels = tf.constant(validation_labels, dtype=tf.int32)
    test_labels = tf.constant(test_labels, dtype=tf.int32)

    history = model.fit(
        training_texts,
        training_labels,
        validation_data=(validation_texts, validation_labels),
        epochs=128,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(test_texts, test_labels, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    plt.plot(history.history["accuracy"], label="Training")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.savefig(GRAPH_PATH)
    plt.close()

    return model


if __name__ == "__main__":
    train_model()
