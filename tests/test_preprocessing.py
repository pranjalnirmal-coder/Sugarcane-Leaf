import io
import unittest

import numpy as np
from PIL import Image

from src.preprocessing import ImageValidationError, open_validated_image, preprocess_image


def make_image(fmt="JPEG", size=(100, 80)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (80, 150, 60)).save(buffer, format=fmt)
    return buffer.getvalue()


class PreprocessingTests(unittest.TestCase):
    def test_valid_jpeg_and_png(self):
        self.assertEqual(open_validated_image(make_image("JPEG")).mode, "RGB")
        self.assertEqual(open_validated_image(make_image("PNG")).mode, "RGB")

    def test_unsupported_format_rejected(self):
        with self.assertRaises(ImageValidationError):
            open_validated_image(make_image("BMP"))

    def test_corrupt_image_rejected(self):
        with self.assertRaises(ImageValidationError):
            open_validated_image(b"not an image")

    def test_small_image_rejected(self):
        with self.assertRaises(ImageValidationError):
            open_validated_image(make_image(size=(20, 20)))

    def test_model_dimensions_and_normalization(self):
        result = preprocess_image(make_image())
        self.assertEqual(result.shape, (1, 224, 224, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(float(result.min()), -1.0)
        self.assertLessEqual(float(result.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
