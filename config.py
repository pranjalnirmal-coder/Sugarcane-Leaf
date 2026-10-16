"""Shared paths and model settings for the Sugarcane Disease project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "raw" / "Sugarcane Leaf Disease Dataset"
SPLIT_FILE = ROOT / "data" / "splits.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "sugarcane_mobilenetv2.keras"
LABELS_PATH = MODEL_DIR / "class_names.json"
OUTPUT_DIR = ROOT / "outputs"
DATABASE_PATH = ROOT / "outputs" / "prediction_history.sqlite3"

IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60
RANDOM_SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
