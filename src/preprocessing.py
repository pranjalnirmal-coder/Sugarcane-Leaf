"""Image validation and preprocessing shared by training and inference."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

from config import IMAGE_SIZE, SUPPORTED_EXTENSIONS

ImageFile.LOAD_TRUNCATED_IMAGES = False
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MIN_IMAGE_SIDE = 32
MAX_IMAGE_SIDE = 12000


class ImageValidationError(ValueError):
    """Raised when an uploaded image cannot safely be used for prediction."""


def open_validated_image(source: bytes | bytearray | BinaryIO | str | Path) -> Image.Image:
    """Return a fully loaded RGB image, rejecting unsupported or malformed inputs."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ImageValidationError("Use a JPG, JPEG, or PNG image.")
        try:
            if path.stat().st_size > MAX_IMAGE_BYTES:
                raise ImageValidationError("Image is too large (maximum 20 MB).")
        except OSError as exc:
            raise ImageValidationError("The image file could not be read.") from exc
        image_source: object = path
    else:
        if hasattr(source, "getvalue"):
            raw = source.getvalue()
        elif isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
        else:
            raw = source.read()
        if len(raw) > MAX_IMAGE_BYTES:
            raise ImageValidationError("Image is too large (maximum 20 MB).")
        image_source = BytesIO(raw)

    try:
        with Image.open(image_source) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ImageValidationError("Use a JPG, JPEG, or PNG image.")
            width, height = image.size
            if min(width, height) < MIN_IMAGE_SIDE or max(width, height) > MAX_IMAGE_SIDE:
                raise ImageValidationError("Image dimensions must be between 32 and 12,000 pixels per side.")
            image.load()
            return image.convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("This image is unreadable or damaged. Please choose another file.") from exc


def preprocess_image(source: bytes | bytearray | BinaryIO | str | Path,
                     image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Prepare one image for MobileNetV2; returns a float32 [1,H,W,3] batch."""
    image = open_validated_image(source)
    image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32)
    # This is TensorFlow's MobileNetV2 preprocess_input transformation.
    pixels = pixels / 127.5 - 1.0
    return np.expand_dims(pixels, axis=0)


def preprocess_pil_for_display(image: Image.Image,
                               image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Return normalized model input from an already validated PIL image."""
    rgb = image.convert("RGB").resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    return np.expand_dims(np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0, axis=0)

def reject_obvious_documents(image: Image.Image) -> None:
    """Reject obvious white-page screenshots before the leaf classifier runs.

    This is a limited check for document-like images, not a general leaf detector.
    """
    small = np.asarray(
        image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR),
        dtype=np.int16,
    )
    channel_max = small.max(axis=2)
    channel_min = small.min(axis=2)

    bright_neutral = (channel_min >= 220) & ((channel_max - channel_min) <= 28)
    saturated = (channel_max - channel_min) >= 45
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    green = (g >= 55) & (g > r * 1.06) & (g > b * 1.03)

    if (
        float(bright_neutral.mean()) >= 0.55
        and float(saturated.mean()) <= 0.12
        and float(green.mean()) <= 0.03
    ):
        raise ImageValidationError(
            "This looks like a document or screenshot rather than a leaf photo. "
            "Please upload a clear, close-up sugarcane leaf image."
        )
