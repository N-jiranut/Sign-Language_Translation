"""
Dataset loading, resampling, augmentation, and splitting.
โหลดข้อมูลจาก data/raw ปรับขนาดเฟรมเป็น 24 เฟรมคงที่ และแบ่ง train/val/test
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
except ImportError:
    GroupShuffleSplit = None
    train_test_split = None

_DIR = Path(__file__).resolve().parent
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from features import canonicalize_handedness, resample_sequence, trim_active_motion


def load_raw_dataset(data_dir=config.DATA_RAW_DIR, target_frames=config.TARGET_FRAMES, trim_active=True, include_classes=None):
    """
    Scans data_dir for class subfolders and loads all valid .npy files.
    Empty directories without .npy files are safely ignored.
    include_classes: optional iterable of class names to include (case-insensitive).
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
    if include_classes:
        allowed = {str(c).strip().lower() for c in include_classes if str(c).strip()}
        class_dirs = [d for d in class_dirs if d.name.lower() in allowed]

    if not class_dirs:
        raise ValueError(f"No matching gesture folders found in {data_dir}.")

    X_list = []
    y_list = []
    groups_list = []
    valid_labels = []

    for d in class_dirs:
        label = d.name
        files = sorted(list(d.glob("*.npy")))
        if not files:
            continue  # Skip empty class folders

        class_has_valid_sample = False
        for f in files:
            try:
                seq = np.load(str(f))
                if seq.ndim != 2 or seq.shape[1] != config.NUM_FEATURES:
                    print(f"[Skip] {f.name}: Invalid shape {seq.shape}")
                    continue

                if trim_active:
                    seq = trim_active_motion(seq)
                    seq = canonicalize_handedness(seq)
                resampled = resample_sequence(seq, target_len=target_frames)
                X_list.append(resampled)
                y_list.append(label)

                # Extract person ID from filename pattern: <person>_<ts>...
                stem = f.stem
                m = re.match(r"^(.*?)(?:_\d{10,})", stem)
                person = m.group(1) if m else stem.split("_")[0]
                if not person:
                    person = "p1"
                groups_list.append(person)
                class_has_valid_sample = True
            except Exception as e:
                print(f"[Error] Failed to load {f.name}: {e}")

        if class_has_valid_sample and label not in valid_labels:
            valid_labels.append(label)

    if not X_list:
        raise ValueError("No valid sample files found to load.")

    valid_labels.sort()
    label_to_id = {name: idx for idx, name in enumerate(valid_labels)}

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


def split_indices(y, groups=None, test_size=0.2, random_state=42, verbose=True):
    """
    Returns consistent (train_idx, test_idx) for dividing dataset splits safely.
    - If multiple signers (>=3) exist and group split preserves all classes in both splits,
      uses GroupShuffleSplit.
    - Otherwise, uses Stratified Split whenever each class has >= 2 samples.
    - Falls back gracefully to random split if stratification is impossible.
    """
    if train_test_split is None:
        raise ImportError(
            "scikit-learn is required for split_dataset.\n"
            "Please ensure you run using the project virtual environment:\n"
            "  .\\.venv\\Scripts\\python.exe"
        )

    n = len(y)
    indices = np.arange(n)
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = int(np.min(counts)) if len(counts) > 0 else 0

    # Try Group split only if we have sufficient independent signers
    if groups is not None and GroupShuffleSplit is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 3:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            tr_idx, te_idx = next(gss.split(indices, y, groups=groups))
            if len(np.unique(y[tr_idx])) == len(unique_classes) and len(np.unique(y[te_idx])) == len(unique_classes):
                if verbose:
                    print(f"[Split] Group split by signers {unique_groups}: {len(tr_idx)} train, {len(te_idx)} test")
                return tr_idx, te_idx

    # Stratified split
    if len(unique_classes) > 1 and min_count >= 2:
        n_classes = len(unique_classes)
        desired_test = int(np.ceil(n * test_size))
        n_test = max(n_classes, desired_test)
        if n_test <= n - n_classes:
            train_idx, test_idx = train_test_split(
                indices, test_size=n_test, stratify=y, random_state=random_state
            )
            if verbose:
                print(f"[Split] Stratified split: {len(train_idx)} train, {len(test_idx)} test")
            return train_idx, test_idx

    # Fallback random split
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    if verbose:
        print(f"[Split] Random split: {len(train_idx)} train, {len(test_idx)} test")
    return train_idx, test_idx


def split_dataset(X, y, groups=None, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets safely.
    """
    train_idx, test_idx = split_indices(y, groups=groups, test_size=test_size, random_state=random_state)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def augment_sequence(seq, noise_std=0.015):
    """
    Augmentation for landmark sequence:
    Adds small Gaussian noise to spatial landmarks, preserving presence flags.
    """
    noise = np.random.normal(0, noise_std, size=seq.shape).astype(np.float32)
    noise[:, -2:] = 0.0  # Do not corrupt binary flags
    return seq + noise


def augment_sequence_rich(seq, noise_std=0.015, rng=None):
    """
    Speed-warped + temporally shifted + scale-jittered augmentation for (T, 112) sequences.
    Makes models robust to faster/slower real-world signing and slight timing shifts.
    """
    if rng is None:
        rng = np.random.default_rng()

    seq = np.asarray(seq, dtype=np.float32)
    t_len, n_feat = seq.shape
    if t_len <= 2:
        return augment_sequence(seq, noise_std=noise_std)

    # (a) Non-linear time warp (gamma < 1 faster start, gamma > 1 faster finish)
    gamma = float(rng.uniform(0.75, 1.33))
    orig_times = np.linspace(0.0, 1.0, t_len)
    warped_times = np.linspace(0.0, 1.0, t_len) ** gamma
    warped = np.zeros_like(seq, dtype=np.float32)
    for col in range(n_feat):
        warped[:, col] = np.interp(warped_times, orig_times, seq[:, col])
    warped[:, -2:] = (warped[:, -2:] >= 0.5).astype(np.float32)

    # (b) Random boundary crop / shift (50% chance)
    if t_len >= 8 and rng.random() < 0.5:
        cut_start = int(rng.integers(0, 3))
        cut_end = int(rng.integers(0, 3))
        sub = warped[cut_start : t_len - cut_end if cut_end > 0 else t_len]
        if len(sub) >= 4:
            warped = resample_sequence(sub, target_len=t_len)

    # (c) Subtle spatial scale jitter
    scale = float(rng.uniform(0.94, 1.06))
    warped[:, :110] *= scale

    # (d) Gaussian jitter noise on coordinates only
    noise = rng.normal(0.0, noise_std, size=(t_len, 110)).astype(np.float32)
    warped[:, :110] += noise
    return warped


def augment_dataset(X, y, num_copies=2, noise_std=0.015, rich=True, seed=42):
    """
    Expands training set by applying augmentation (speed-warp + shift + noise when rich=True).
    """
    if num_copies <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    augmented_X = [X]
    augmented_y = [y]

    for _ in range(num_copies):
        if rich:
            noisy = np.array([augment_sequence_rich(s, noise_std=noise_std, rng=rng) for s in X], dtype=np.float32)
        else:
            noisy = np.array([augment_sequence(s, noise_std=noise_std) for s in X], dtype=np.float32)
        augmented_X.append(noisy)
        augmented_y.append(y)

    return np.concatenate(augmented_X, axis=0), np.concatenate(augmented_y, axis=0)
