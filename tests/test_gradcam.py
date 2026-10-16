import unittest

try:
    import tensorflow as tf
except ImportError:
    tf = None


@unittest.skipIf(tf is None, "TensorFlow is not installed; Grad-CAM integration requires the trained model environment.")
class GradcamTests(unittest.TestCase):
    def test_gradcam_output_dimensions_when_model_exists(self):
        from config import LABELS_PATH, MODEL_PATH
        from src.gradcam import generate_gradcam
        from src.predict import load_model
        import numpy as np
        if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
            self.skipTest("No trained model is present yet.")
        model, _ = load_model()
        heatmap = generate_gradcam(model, np.zeros((1, 224, 224, 3), dtype=np.float32))
        self.assertEqual(heatmap.shape, (224, 224))
        self.assertTrue(np.isfinite(heatmap).all())
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
