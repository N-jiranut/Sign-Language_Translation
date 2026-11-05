"""
Dataset loading, resampling, augmentation, and splitting.
โหลดข้อมูลจาก data/raw ปรับขนาดเฟรมเป็น 24 เฟรมคงที่ และแบ่ง train/val/test
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from features import resample_sequence


def load_raw_dataset(data_dir=config.DATA_RAW_DIR, target_frames=config.TARGET_FRAMES):
    """
    Scans data_dir for class subfolders and loads all .npy files.
    Returns:
        X: (N, 24, 112) numpy float32 array
        y: (N,) string label array
        groups: (N,) person IDs
        label_to_id: dict mapping label names to integer IDs
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No gesture folders found in {data_dir}.")

    label_names = [d.name for d in class_dirs]
    label_to_id = {name: idx for idx, name in enumerate(label_names)}

    X_list = []
    y_list = []
    groups_list = []

    for d in class_dirs:
        label = d.name
        files = sorted(list(d.glob("*.npy")))
        for f in files:
            try:
                seq = np.load(str(f))
                # Check dimensions
                if seq.ndim != 2 or seq.shape[1] != config.NUM_FEATURES:
                    print(f"[Skip] {f.name}: Invalid shape {seq.shape}")
                    continue

                resampled = resample_sequence(seq, target_len=target_frames)
                X_list.append(resampled)
                y_list.append(label)

                # Extract person ID from filename: <person>_<ts>.npy
                person = f.stem.split("_")[0] if "_" in f.stem else "p1"
                groups_list.append(person)
            except Exception as e:
                print(f"[Error] Failed to load {f.name}: {e}")

    if not X_list:
        raise ValueError("No valid sample files found to load.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list)
    groups = np.array(groups_list)

    return X, y, groups, label_to_id


def save_label_mapping(label_to_id, save_path=config.SAVED_LABELS):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(label_to_id, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Label mapping saved to {save_path}")


def load_label_mapping(load_path=config.SAVED_LABELS):
    with open(load_path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_dataset(X, y, groups, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.
    If multiple signers exist, splits by group (person).
    Otherwise, uses stratified random split.
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) > 1:
        # Group split by signer
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        print(f"[Split] Group split by signers {unique_groups}: {len(train_idx)} train, {len(test_idx)} test")
    else:
        # Single signer: stratified split
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=y, random_state=random_state
        )
        print(f"[Split] Stratified split: {len(train_idx)} train, {len(test_idx)} test")

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def augment_sequence(seq, noise_std=0.015):
    """
    Light augmentation for landmark sequence:
    Adds small Gaussian noise to features.
    """
    noise = np.random.normal(0, noise_std, size=seq.shape).astype(np.float32)
    # Don't add noise to the last 2 values (presence flags)
    noise[:, -2:] = 0.0
    return seq + noise
