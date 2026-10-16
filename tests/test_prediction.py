import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.predict import interpret_probabilities, load_model, predict_image
from test_preprocessing import make_image


class FixedModel:
    def predict(self, batch, verbose=0):
        self.input_shape = batch.shape
        return np.array([[0.1, 0.8, 0.1]])


class PredictionTests(unittest.TestCase):
    def test_probability_output_and_class_mapping(self):
        names = ["Healthy", "Rust", "Yellow"]
        result = interpret_probabilities(np.array([0.1, 0.8, 0.1]), names)
        self.assertEqual(result["class_name"], "Rust")
        self.assertAlmostEqual(result["confidence"], 0.8)
        self.assertFalse(result["uncertain"])

    def test_prediction_uses_model_input_shape_and_returns_one_class(self):
        model = FixedModel()
        result = predict_image(model, make_image(), ["Healthy", "Rust", "Yellow"])
        self.assertEqual(model.input_shape, (1, 224, 224, 3))
        self.assertEqual(result["class_name"], "Rust")

    def test_low_confidence_is_uncertain(self):
        result = interpret_probabilities(np.array([0.4, 0.35, 0.25]), ["Healthy", "Rust", "Yellow"], 0.6)
        self.assertTrue(result["uncertain"])
        self.assertIn("uncertain", result["message"].lower())

    def test_prediction_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            interpret_probabilities(np.array([0.7, 0.3]), ["Healthy", "Rust", "Yellow"])

    def test_missing_model_files(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with self.assertRaises(FileNotFoundError):
                load_model(Path(directory) / "missing.keras", Path(directory) / "labels.json")


if __name__ == "__main__":
    unittest.main()
