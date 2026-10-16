"""Dataset audit, duplicate-aware splitting, and TensorFlow input pipelines."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from config import DATA_ROOT, OUTPUT_DIR, RANDOM_SEED, SPLIT_FILE, SUPPORTED_EXTENSIONS


def _dhash(path: Path) -> int:
    """64-bit difference hash for conservative near-duplicate grouping."""
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        values = np.asarray(gray)
    bits = values[:, 1:] > values[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def _build_near_duplicate_groups(items: list[dict], max_distance: int = 4) -> list[list[int]]:
    """Connected components of image hashes; exact duplicates are included."""
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    hashes = [item["dhash"] for item in items]
    buckets: dict[int, list[int]] = defaultdict(list)
    for idx, value in enumerate(hashes):
        buckets[value].append(idx)
    for value, members in buckets.items():
        for idx in members[1:]:
            union(members[0], idx)
    for i, value in enumerate(hashes):
        for j in range(i + 1, len(hashes)):
            if (value ^ hashes[j]).bit_count() <= max_distance:
                union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(items)):
        groups[find(idx)].append(idx)
    return list(groups.values())


def _assign_group_splits(indices: list[int], items: list[dict], rng: random.Random) -> dict[int, str]:
    """Assign whole same-class duplicate groups to 70/15/15 partitions."""
    groups = _build_near_duplicate_groups([items[i] for i in indices])
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    target_ratio = {"train": 0.70, "validation": 0.15, "test": 0.15}
    totals = {key: 0 for key in target_ratio}
    result: dict[int, str] = {}
    total = len(indices)
    for group in groups:
        split = max(target_ratio, key=lambda key: (target_ratio[key] * total - totals[key], -list(target_ratio).index(key)))
        for local_index in group:
            original_index = indices[local_index]
            result[original_index] = split
            totals[split] += 1
    return result


def audit_and_split(data_root: Path = DATA_ROOT, split_file: Path = SPLIT_FILE,
                    seed: int = RANDOM_SEED, near_duplicate_distance: int = 4) -> dict:
    """Check folder images and write deterministic, duplicate-aware split CSV."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {data_root}\nExtract class folders under data/raw/.")
    classes = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError("Expected at least two class folders inside the dataset directory.")

    candidates: list[dict] = []
    corrupt: list[str] = []
    unsupported: list[str] = []
    for class_name in classes:
        for path in sorted((data_root / class_name).rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                unsupported.append(str(path))
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                    if width < 32 or height < 32:
                        raise ValueError("image too small")
                    dhash = _dhash(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                candidates.append({"path": path.resolve(), "class_name": class_name,
                                   "sha256": digest, "dhash": dhash, "width": width, "height": height})
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                corrupt.append(f"{path} ({exc})")

    # Keep one file per exact content hash. Record exact duplicates in the audit.
    seen_hash: dict[str, int] = {}
    exact_duplicates: list[dict] = []
    exact_cross_label_indices: set[int] = set()
    items: list[dict] = []
    for item in candidates:
        if item["sha256"] in seen_hash:
            kept_index = seen_hash[item["sha256"]]
            same_label = items[kept_index]["class_name"] == item["class_name"]
            exact_duplicates.append({"kept": str(items[kept_index]["path"]),
                                     "duplicate": str(item["path"]),
                                     "same_label": same_label})
            if not same_label:
                exact_cross_label_indices.add(kept_index)
        else:
            seen_hash[item["sha256"]] = len(items)
            items.append(item)

    rng = random.Random(seed)
    global_groups = _build_near_duplicate_groups(items, near_duplicate_distance)
    ambiguous_groups: list[list[str]] = []
    excluded_indices: set[int] = set(exact_cross_label_indices)
    for group in global_groups:
        labels = {items[i]["class_name"] for i in group}
        if len(group) > 1 and len(labels) > 1:
            ambiguous_groups.append([str(items[i]["path"]) for i in group])
            excluded_indices.update(group)

    by_class: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        if i not in excluded_indices:
            by_class[item["class_name"]].append(i)

    assignments: dict[int, str] = {}
    for class_name, indices in by_class.items():
        assignments.update(_assign_group_splits(indices, items, rng))

    rows = [{"path": str(items[i]["path"].relative_to(data_root.resolve())),
             "class_name": items[i]["class_name"], "split": assignments[i]}
            for i in range(len(items)) if i in assignments]
    split_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["split", "class_name", "path"]).to_csv(split_file, index=False)

    counts = Counter(item["class_name"] for item in candidates)
    usable_counts = Counter(items[i]["class_name"] for i in assignments)
    split_counts = {split: dict(Counter(row["class_name"] for row in rows if row["split"] == split))
                    for split in ("train", "validation", "test")}
    audit = {
        "dataset_path": str(data_root.resolve()), "seed": seed,
        "supported_classes": classes, "image_size": "variable source sizes; resized to 224x224 for model",
        "image_count_by_class_before_deduplication": dict(counts),
        "readable_supported_images_before_deduplication": len(candidates),
        "corrupt_or_too_small_images": corrupt,
        "unsupported_files": unsupported,
        "exact_duplicate_files_removed": exact_duplicates,
        "exact_duplicate_cross_label_conflicts_excluded": [str(items[i]["path"]) for i in sorted(exact_cross_label_indices)],
        "near_duplicate_hamming_threshold": near_duplicate_distance,
        "cross_class_near_duplicate_groups_excluded_as_ambiguous": ambiguous_groups,
        "usable_image_count_by_class": dict(usable_counts),
        "split_counts": split_counts,
        "split_csv": str(split_file.resolve()),
        "class_imbalance_ratio_after_filtering": (max(usable_counts.values()) / min(usable_counts.values()) if usable_counts else None),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def make_tf_datasets(data_root: Path = DATA_ROOT, split_file: Path = SPLIT_FILE,
                     batch_size: int = 32, seed: int = RANDOM_SEED):
    """Create tf.data datasets and class weights from the audited split manifest."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required. Install requirements.txt in your Python environment.") from exc
    if not split_file.exists():
        audit_and_split(data_root=data_root, split_file=split_file, seed=seed)
    frame = pd.read_csv(split_file)
    classes = sorted(frame["class_name"].unique().tolist())
    label_to_index = {name: i for i, name in enumerate(classes)}
    root = data_root.resolve()
    datasets = {}
    counts = Counter()
    for split in ("train", "validation", "test"):
        subset = frame[frame["split"] == split]
        paths = [str(root / path) for path in subset["path"]]
        labels = [label_to_index[name] for name in subset["class_name"]]
        counts.update(labels if split == "train" else [])
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def load(path, label):
            data = tf.io.read_file(path)
            image = tf.io.decode_image(data, channels=3, expand_animations=False)
            image.set_shape([None, None, 3])
            image = tf.image.resize(image, (224, 224), antialias=True)
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            return image, tf.one_hot(label, depth=len(classes))

        ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        if split == "train":
            ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
        datasets[split] = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    total = sum(counts.values())
    class_weights = {index: total / (len(classes) * count) for index, count in counts.items()}
    return datasets["train"], datasets["validation"], datasets["test"], classes, class_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    audit = audit_and_split(args.data, SPLIT_FILE, args.seed)
    summary = {
        "supported_classes": audit["supported_classes"],
        "image_count_by_class_before_deduplication": audit["image_count_by_class_before_deduplication"],
        "readable_supported_images_before_deduplication": audit["readable_supported_images_before_deduplication"],
        "corrupt_or_too_small_image_count": len(audit["corrupt_or_too_small_images"]),
        "exact_duplicate_file_count_removed": len(audit["exact_duplicate_files_removed"]),
        "cross_class_near_duplicate_group_count_excluded": len(audit["cross_class_near_duplicate_groups_excluded_as_ambiguous"]),
        "usable_image_count_by_class": audit["usable_image_count_by_class"],
        "split_counts": audit["split_counts"],
        "class_imbalance_ratio_after_filtering": audit["class_imbalance_ratio_after_filtering"],
        "full_audit_file": str(OUTPUT_DIR / "dataset_audit.json"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
