# Sugarcane Leaf Disease Detection and Advisory System

A student project based on the uploaded Overleaf report. The local dashboard accepts a sugarcane leaf image, uses a trained CNN to predict one of the dataset classes, displays model confidence and Grad-CAM, provides preliminary source-linked guidance, and stores each signed-in user's prediction history in SQLite.

The Overleaf PDF is preserved unchanged. The report names Red Rot, Smut, Brown Rust, Leaf Scald, Red Stripe, Mosaic Disease, and Eye Spot as diseases of interest. The supplied dataset archive has five folders: `Healthy`, `Mosaic`, `RedRot`, `Rust`, and `Yellow`. Training labels are discovered from the actual folders. This dataset does not contain all report diseases. The `Yellow` folder does not identify a specific disease, so the app does not call it Yellow Leaf Disease.

## Current project status

- Implemented: image validation and MobileNetV2 preprocessing, duplicate-aware dataset audit and splits, model training/evaluation pipeline, prediction and confidence threshold, Grad-CAM, sourced preliminary advisories for the five dataset labels, Streamlit UI, local accounts, and per-user SQLite history.
- Dataset: the supplied archive was extracted to `data/raw/`, audited, and split into training, validation, and test partitions.
- Trained model: `models/sugarcane_mobilenetv2.keras` and its matching class mapping are included in this project.
- Actual held-out test result: 79.2% accuracy across 342 images. The dashboard's **Model training and evaluation** panel shows the training curves, confusion matrix, and per-class metrics from `outputs/`.
- Windows environment: a virtual environment and installed packages are local to the creator's computer. To use a shared copy, install the requirements using `run_dashboard.bat` or follow the setup instructions below.
- Login is implemented as local account sign-in/registration to honor the report's essential login feature. The report does not define role permissions or an admin console, so those are not represented as implemented features.

## Dataset source and audit

The supplied RAR follows Mendeley Data's [Sugarcane Leaf Disease Dataset, version 1](https://data.mendeley.com/datasets/9424skmnrk/1), licensed CC BY 4.0. The dataset record describes 2,569 images. The supplied archive has 2,521 image files; its folder counts are Healthy 522, Mosaic 462, RedRot 518, Rust 514, and Yellow 505. All 2,521 files decoded successfully. The initial audit found 210 exact duplicate files (same-label copies) and seven groups of visually similar leaves spanning more than one folder (30 files total); those cross-label groups were excluded from splitting because they may be mislabeled or confusing lookalikes. After filtering, 2,281 images remain: Healthy 466, Mosaic 367, RedRot 518, Rust 447, Yellow 483. Review `outputs/dataset_audit.json` for file-level details. The perceptual hash audit is a conservative heuristic, so some excluded examples may be similar but distinct leaves.

The other candidate, [Sugarcane Leaf Dataset 355y629ynj](https://data.mendeley.com/datasets/355y629ynj/1), is also available under CC BY 4.0 and lists 6,748 images across nine disease categories, including healthy and dried leaves. Its public record does not enumerate the individual class-folder labels, so its overlap with the report cannot be confirmed from the metadata alone. The supplied archive is selected for this implementation because its five actual folders are available to inspect directly and include Healthy, Mosaic, RedRot, and Rust (the report's Brown Rust example); Yellow remains ambiguous.

Expected layout:

```text
data/raw/Sugarcane Leaf Disease Dataset/
    Healthy/
    Mosaic/
    RedRot/
    Rust/
    Yellow/
```

Raw data, audit outputs, generated splits, model weights, and prediction history are excluded from Git. If using the original RAR, extract its top-level `Sugarcane Leaf Disease Dataset` folder into `data/raw/`.

Prepare and audit the dataset:

```bash
python -m src.dataset
```

This writes `data/splits.csv` and `outputs/dataset_audit.json`. It validates image readability, removes exact duplicate files from the split, groups near-duplicates (64-bit difference hash distance up to 4) so they stay within one partition, and excludes ambiguous near-duplicate groups that span different labels. Splits are reproducible (seed 42) and target 70% train, 15% validation, and 15% test per class. For another folder dataset, pass the same path to both audit and training: `python -m src.dataset --data "path/to/dataset"`, then `python -m src.train --data "path/to/dataset"`. Review the audit's actual counts before training.

## Setup

The trained model is included, so a recipient can run the dashboard without downloading the dataset or retraining. On Windows, extract the shared ZIP and double-click `run_dashboard.bat`. The first run creates a virtual environment and installs packages from PyPI, so an internet connection is needed for setup. It then launches the local dashboard in the browser. Each person should create their own local account; prediction history stays on the computer running the dashboard.

The share package includes the trained model, label mapping, source code, dashboard launch file, evaluation metrics, training curves, and confusion matrix. It does not include the raw image dataset. To retrain, obtain the dataset separately and extract its class folders into `data/raw/Sugarcane Leaf Disease Dataset/`.

To set up manually, use Python 3.10–3.12 in VS Code and create a virtual environment.

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Google Colab:

```python
!pip install -r requirements.txt
```

Training is best run in Colab if a GPU is available. On Windows, current TensorFlow supports CPU inference; native-Windows GPU support ended after TensorFlow 2.10. Use Colab or WSL2 for GPU training. See the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip).

## Train and evaluate

After dataset setup, run:

```bash
python -m src.train
```

Training uses MobileNetV2 transfer learning with ImageNet initialization, augmentation on the training path only, class weights, early stopping, learning-rate reduction, and best-validation checkpointing. If pretrained weights cannot be fetched, use `python -m src.train --no-imagenet-weights` to train from random initialization. The model output size is derived from the discovered class folders.

The test partition is kept separate from training and validation. Training saves actual outputs under `models/` and `outputs/`: the `.keras` model, `class_names.json`, preprocessing metadata, training accuracy/loss curves, test confusion matrix, and `evaluation_metrics.json` with accuracy and per-class precision/recall/F1. No result is pre-filled or represented as an experiment before training finishes.

## Launch the application

```bash
streamlit run app.py
```

Create a local account from the sign-in page. The model must exist at `models/sugarcane_mobilenetv2.keras` with its matching `models/class_names.json`. If the model files are absent, the app explains that training is required. Use JPG, JPEG, or PNG images up to 20 MB; images must be readable and at least 32 pixels per side.

## Architecture and processing flow

```text
Image upload → validation → resize/RGB/MobileNetV2 normalization
             → trained model → prediction and confidence
             → Grad-CAM heatmap/overlay → class advisory
             → display and optional per-user SQLite history
```

Key modules:

- `src/dataset.py`: folder discovery, image audit, duplicate-aware split and TensorFlow input pipeline.
- `src/preprocessing.py`: shared validation and exact inference preprocessing.
- `src/train.py`, `src/evaluation.py`: MobileNetV2 training and held-out test metrics.
- `src/predict.py`, `src/gradcam.py`: reusable inference and Grad-CAM.
- `src/advisory.py`: class-to-guidance mapping.
- `src/database.py`: local account and prediction-history tables.
- `app.py`: Streamlit dashboard.

## Advisory sources and limitations

Advice is preliminary and intentionally avoids pesticide dosages. Sources include Tamil Nadu Agricultural University guidance for [red rot and general sugarcane disease management](https://www.agritech.tnau.ac.in/govt_schemes_services/aas/sugarcane.html), [mosaic](https://agritech.tnau.ac.in/org_farm/orgfarm_agridiseases.html), [rust](https://www.agritech.tnau.ac.in/crop_protection/sugarcane_diseases/sugarcane_d3.html), and [Yellow Leaf Disease symptoms](https://agritech.tnau.ac.in/crop_protection/sugarcane_diseases/sugarcane_d1.html). The app clearly says its `Yellow` dataset class is ambiguous and should not be treated as an identified disease. Advice and image classification do not replace confirmation by an agricultural expert.

Grad-CAM highlights image regions that influenced the model output. It does not independently prove that a region is diseased. A softmax confidence threshold helps flag uncertain outputs but cannot reliably identify every non-sugarcane or out-of-distribution image. The model only learns the dataset's labels and image conditions; accuracy on unrelated fields, cameras, or varieties is unknown until evaluated on suitable external data.

The report also depicts severity and recovery-time fields, but does not define a validated method or provide severity/recovery labels. They are therefore not presented as implemented predictions.

## Troubleshooting

- **Model missing:** prepare the dataset and run `python -m src.train` first.
- **TensorFlow install fails on Windows:** use a supported 64-bit Python version, install the Microsoft Visual C++ Redistributable if requested, or train/run in Google Colab or WSL2.
- **Image rejected:** re-save it as a valid JPG or PNG, make sure it is at least 32 pixels per side, and keep it under 20 MB.
- **Dataset folder not found:** ensure the class folders are under `data/raw/Sugarcane Leaf Disease Dataset/`.
- **Pretrained weights unavailable:** training can use `--no-imagenet-weights`, though convergence may be slower and results will differ.
- **Grad-CAM error:** inference remains separate from the explanation function; verify the model is the project MobileNetV2 model and retrain if using a different architecture.
- **History is local:** SQLite history is stored in `outputs/prediction_history.sqlite3` on the machine running Streamlit. It is not a hosted account service.

## Tests

Run the available unit tests with:

```bash
python -m unittest discover -s tests -v
```

These cover image validation and preprocessing, confidence handling, class/advisory mapping, and SQLite account/history operations. Grad-CAM model-shape verification requires TensorFlow and a trained model.
