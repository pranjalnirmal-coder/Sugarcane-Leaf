"""Model evaluation metrics and figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from config import OUTPUT_DIR


def evaluate_model(model, test_dataset, class_names: list[str], output_dir: Path = OUTPUT_DIR) -> dict:
    """Evaluate once on the held-out test partition and save actual metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict(test_dataset, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_dataset])
    report = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                   target_names=class_names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    metrics = {
        "test_accuracy": float(np.mean(y_true == y_pred)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
        "test_image_count": int(len(y_true)),
    }
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax)
    ax.set(xlabel="Predicted label", ylabel="True label", title="Test-set confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    return metrics


def save_training_history(history, output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    values = history.history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(values.get("accuracy", []), label="Training")
    axes[0].plot(values.get("val_accuracy", []), label="Validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].plot(values.get("loss", []), label="Training")
    axes[1].plot(values.get("val_loss", []), label="Validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)
