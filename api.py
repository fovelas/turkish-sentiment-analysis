from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.keras"


class SentimentAnalysis:
    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Run sentiment_analysis.py to create it."
            )

        self.model = keras.models.load_model(MODEL_PATH, compile=False)

    def predict(self, text: str) -> tuple[str, str, str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()
        if not text:
            raise ValueError("text must not be empty.")

        prediction = self.model.predict(
            tf.constant([text], dtype=tf.string), verbose=0
        )
        probabilities = np.asarray(prediction)[0]
        if probabilities.shape != (2,):
            raise ValueError(
                "The sentiment model must return negative and positive scores."
            )

        result = "positive" if int(np.argmax(probabilities)) == 1 else "negative"
        return (
            result,
            str(float(probabilities[0])),
            str(float(probabilities[1])),
        )

    def get_size(self) -> dict[str, int]:
        dataset = pd.read_csv(DATASET_PATH)
        if "sentiment" not in dataset.columns:
            raise ValueError("data/dataset.csv must contain a sentiment column.")

        counts = dataset["sentiment"].value_counts()
        return {
            "negative": int(counts.get("negative", 0)),
            "positive": int(counts.get("positive", 0)),
        }
