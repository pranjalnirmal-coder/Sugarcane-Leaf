"""Train a reproducible MobileNetV2 baseline and evaluate on a held-out test set."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from config import DATA_ROOT, LABELS_PATH, MODEL_DIR, MODEL_PATH, OUTPUT_DIR, RANDOM_SEED, SPLIT_FILE
from src.dataset import audit_and_split, make_tf_datasets
from src.evaluation import evaluate_model, save_training_history


def build_model(class_count: int, image_size: int = 224, imagenet_weights: bool = True):
    import tensorflow as tf
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name="training_augmentation")
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="normalized_rgb_image")
    x = augmentation(inputs)
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet" if imagenet_weights else None,
        input_shape=(image_size, image_size, 3), pooling=None, name="mobilenetv2_backbone")
    base.trainable = False
    x = base(x, training=False)
    # This outer-graph identity keeps the CNN feature tensor connected to the
    # classifier graph for Grad-CAM after the .keras model is reloaded.
    x = tf.keras.layers.Activation("linear", name="gradcam_features")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout")(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="sugarcane_leaf_disease_classifier")
    return model, base


def train(epochs: int = 12, fine_tune_epochs: int = 5, batch_size: int = 32,
          seed: int = RANDOM_SEED, imagenet_weights: bool = True,
          data_root: Path = DATA_ROOT) -> dict:
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    audit_and_split(data_root=data_root, split_file=SPLIT_FILE, seed=seed)
    train_ds, val_ds, test_ds, class_names, class_weights = make_tf_datasets(
        data_root=data_root, split_file=SPLIT_FILE, batch_size=batch_size, seed=seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, backbone = build_model(len(class_names), imagenet_weights=imagenet_weights)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
    ]
    first = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                      class_weight=class_weights, callbacks=callbacks)
    histories = [first]
    if fine_tune_epochs > 0:
        backbone.trainable = True
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        second = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs,
                           class_weight=class_weights, callbacks=callbacks)
        histories.append(second)
    history = {key: sum((part.history.get(key, []) for part in histories), [])
               for key in set().union(*(part.history.keys() for part in histories))}
    class HistoryView:
        pass
    history_view = HistoryView()
    history_view.history = history
    save_training_history(history_view)

    # Reload the best validation checkpoint before a single, final test evaluation.
    model = tf.keras.models.load_model(MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (MODEL_DIR / "preprocessing.json").write_text(json.dumps({
        "image_size": [224, 224], "color_order": "RGB", "input_range": [-1.0, 1.0],
        "normalization": "pixel / 127.5 - 1.0", "backbone": "MobileNetV2",
    }, indent=2), encoding="utf-8")
    metrics = evaluate_model(model, test_ds, class_names)
    print(json.dumps({"classes": class_names, "test_accuracy": metrics["test_accuracy"],
                      "test_image_count": metrics["test_image_count"],
                      "metrics_file": str(OUTPUT_DIR / "evaluation_metrics.json")}, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--data", type=Path, default=DATA_ROOT,
                        help="Dataset root containing one subfolder per class.")
    parser.add_argument("--no-imagenet-weights", action="store_true",
                        help="Initialize MobileNetV2 randomly; useful if pretrained weights cannot be downloaded.")
    args = parser.parse_args()
    train(args.epochs, args.fine_tune_epochs, args.batch_size, args.seed,
          imagenet_weights=not args.no_imagenet_weights, data_root=args.data)


if __name__ == "__main__":
    main()
