"""Grad-CAM visualization for the trained MobileNetV2 classifier."""
from __future__ import annotations

import numpy as np
from PIL import Image

from config import IMAGE_SIZE


def _backbone_and_activation_layer(model):
    """Find the MobileNetV2 backbone and its last spatial feature-map layer."""
    import tensorflow as tf
    candidates = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    if not candidates:
        raise ValueError("No nested CNN backbone was found in the trained model.")
    backbone = next((layer for layer in candidates if "mobilenet" in layer.name.lower()), candidates[-1])
    spatial = []
    for layer in backbone.layers:
        try:
            shape = tuple(layer.output.shape)
        except (AttributeError, ValueError):
            shape = ()
        if len(shape) == 4:
            spatial.append(layer)
    if not spatial:
        raise ValueError("The CNN backbone has no spatial feature map for Grad-CAM.")
    return backbone, spatial[-1].name


def generate_gradcam(model, image_batch: np.ndarray, class_index: int | None = None,
                     output_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Return a normalized float heatmap resized to (height,width).

    The selected feature layer is the last four-dimensional output in the
    MobileNetV2 backbone (normally `out_relu`). Grad-CAM explains model influence;
    it does not prove that an area is diseased.
    """
    import tensorflow as tf
    feature_layer = next((layer for layer in model.layers if layer.name == "gradcam_features"), None)
    if feature_layer is None:
        backbone, layer_name = _backbone_and_activation_layer(model)
        feature_layer = backbone.get_layer(layer_name)
    try:
        grad_model = tf.keras.Model(model.inputs, [feature_layer.output, model.output])
        with tf.GradientTape() as tape:
            feature_maps, predictions = grad_model(image_batch, training=False)
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))
            score = predictions[:, class_index]
        gradients = tape.gradient(score, feature_maps)
    except (ValueError, KeyError) as exc:
        raise RuntimeError("Could not connect the selected CNN layer to the classifier for Grad-CAM.") from exc
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients were not available for the selected layer.")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * feature_maps, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, tf.zeros_like(heatmap))
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], output_size, method="bilinear")[..., 0]
    return np.asarray(heatmap, dtype=np.float32)


def create_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.40) -> Image.Image:
    """Blend an OpenCV JET heatmap with an RGB source image."""
    import cv2
    original = np.asarray(image.convert("RGB"))
    resized = cv2.resize(np.asarray(heatmap, dtype=np.float32),
                         (original.shape[1], original.shape[0]))
    colored = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 1.0 - alpha, colored, alpha, 0)
    return Image.fromarray(overlay)
