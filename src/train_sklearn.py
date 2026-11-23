"""
Train Random Forest Classifiers (Legacy Flat & Kinematic Summary).
เทรนโมเดล Random Forest ทั้งแบบเดิม (Flat 2688) และแบบ Kinematic (1008) พร้อมเปรียบเทียบผล
"""
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

_DIR = Path(__file__).resolve().parent
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from dataset import augment_dataset, load_raw_dataset, save_label_mapping, split_indices
from features import extract_kinematic_batch


def flatten_features(X):
    """Flatten (N, T, F) into (N, T * F)."""
    return X.reshape(X.shape[0], -1)


def main():
    print("=" * 60)
    print(" Training Random Forest Classifiers (Legacy vs Kinematic) ")
    print("=" * 60)

    try:
        X_trim, y, groups, label_to_id = load_raw_dataset(trim_active=True)
        X_raw, _, _, _ = load_raw_dataset(trim_active=False)
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        print("\nTip: Collect samples first using: python src/collect_data.py")
        sys.exit(1)

    print(f"Total samples: {len(X_trim)} | Classes ({len(label_to_id)}): {list(label_to_id.keys())}")
    if len(label_to_id) < 2:
        print("[Warning] Need at least 2 distinct gesture classes to train a classifier.")
        sys.exit(1)

    tr_idx, te_idx = split_indices(y, groups=groups, test_size=0.2, random_state=42)
    X_trim_tr, X_trim_te = X_trim[tr_idx], X_trim[te_idx]
    X_raw_tr, X_raw_te = X_raw[tr_idx], X_raw[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_label_mapping(label_to_id, config.SAVED_LABELS)

    # 1. Legacy Flat RF (2688 dims)
    X_leg_tr, y_leg_tr = X_raw_tr, y_tr
    if len(X_leg_tr) < 50:
        X_leg_tr, y_leg_tr = augment_dataset(X_leg_tr, y_leg_tr, num_copies=1, noise_std=0.015, rich=False, seed=42)
    rf_legacy = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_legacy.fit(flatten_features(X_leg_tr), y_leg_tr)
    preds_leg = rf_legacy.predict(flatten_features(X_raw_te))
    acc_leg = accuracy_score(y_te, preds_leg)
    joblib.dump(rf_legacy, config.SAVED_SKLEARN_MODEL)
    print(f"[Saved] RF (Legacy Flat) -> {config.SAVED_SKLEARN_MODEL} | Test Acc: {acc_leg * 100:.2f}%")

    # 2. Kinematic RF (1008 dims, rich speed/shift augmentation + trimmed motion)
    X_aug, y_aug = X_trim_tr, y_tr
    if len(X_aug) < 300:
        X_aug, y_aug = augment_dataset(X_aug, y_aug, num_copies=3, noise_std=0.015, rich=True, seed=42)
    rf_kin = RandomForestClassifier(n_estimators=150, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_kin.fit(extract_kinematic_batch(X_aug), y_aug)
    preds_kin = rf_kin.predict(extract_kinematic_batch(X_trim_te))
    acc_kin = accuracy_score(y_te, preds_kin)
    joblib.dump(rf_kin, config.SAVED_KINEMATIC_RF_MODEL)
    print(f"[Saved] RF (Kinematic)   -> {config.SAVED_KINEMATIC_RF_MODEL} | Test Acc: {acc_kin * 100:.2f}%")

    print("\nClassification Report (RF Kinematic):")
    print(classification_report(y_te, preds_kin, zero_division=0))


if __name__ == "__main__":
    main()
