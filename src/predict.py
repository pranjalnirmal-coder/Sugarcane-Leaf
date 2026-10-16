"""Model loading and image prediction helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import CONFIDENCE_THRESHOLD, LABELS_PATH, MODEL_PATH
from src.preprocessing import ImageValidationError, preprocess_image


def load_model(model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH):
    if not model_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("Trained model files are missing. Run `python -m src.train` first.")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is not installed. Install the project requirements first.") from exc
    model = tf.keras.models.load_model(model_path)
    class_names = json.loads(labels_path.read_text(encoding="utf-8"))
    if model.output_shape[-1] != len(class_names):
        raise ValueError("Model output size does not match the saved class-label mapping.")
    return model, class_names


def interpret_probabilities(probabilities: np.ndarray, class_names: list[str],
                            threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    if scores.shape != (len(class_names),):
        raise ValueError("Prediction output shape does not match the class-label mapping.")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("Model returned invalid class probabilities.")
    index = int(np.argmax(scores))
    confidence = float(scores[index])
    uncertain = confidence < threshold
    return {
        "class_name": class_names[index], "confidence": confidence,
        "uncertain": uncertain,
        "message": ("The model is uncertain. Try a clearer sugarcane leaf photo or consult an agricultural expert."
                    if uncertain else ""),
        "probabilities": scores.tolist(),
    }


def predict_image(model, image_source, class_names: list[str],
                  threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    batch = preprocess_image(image_source)
    output = np.asarray(model.predict(batch, verbose=0))
    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError("Model prediction must have shape [1, number_of_classes].")
    return interpret_probabilities(output[0], class_names, threshold)


__all__ = ["ImageValidationError", "load_model", "interpret_probabilities", "predict_image"]
