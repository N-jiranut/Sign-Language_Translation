"""
Train Random Forest Baseline Classifier.
เทรนโมเดล Random Forest เพื่อจำแนกท่าทางภาษามือ
"""
import sys
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

_DIR = Path(__file__).resolve().parent
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from dataset import load_raw_dataset, split_dataset, save_label_mapping, augment_dataset


def flatten_features(X):
    """Flatten (N, T, F) into (N, T * F)."""
    return X.reshape(X.shape[0], -1)


def main():
    print("=" * 50)
    print(" Training Random Forest Baseline Classifier ")
    print("=" * 50)

    try:
        X, y, groups, label_to_id = load_raw_dataset()
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        print("\nTip: Collect samples first using: python src/collect_data.py")
        sys.exit(1)

    print(f"Total samples: {len(X)}")
    print(f"Classes: {list(label_to_id.keys())}")
    for name in label_to_id:
        count = sum(y == name)
        print(f"  - {name}: {count} samples")

    if len(label_to_id) < 2:
        print("[Warning] Need at least 2 distinct gesture classes to train a classifier.")
        sys.exit(1)

    # Train/Test Split
    X_train, X_test, y_train, y_test = split_dataset(X, y, groups, test_size=0.2)

    # Augment compact training sets
    if len(X_train) < 50:
        X_train, y_train = augment_dataset(X_train, y_train, num_copies=1, noise_std=0.015)
        print(f"[Augmentation] Augmented training set size: {len(X_train)} samples")

    # Flatten temporal dimensions for Random Forest
    X_train_flat = flatten_features(X_train)
    X_test_flat = flatten_features(X_test)

    print(f"\nFeature vector length: {X_train_flat.shape[1]}")
    print(f"Training on {len(X_train_flat)} samples, evaluating on {len(X_test_flat)} samples...")

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_flat, y_train)

    # Evaluation
    preds = clf.predict(X_test_flat)
    acc = accuracy_score(y_test, preds)
    print("\n" + "=" * 50)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("=" * 50)
    print("Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    print("Confusion Matrix:")
    labels = sorted(list(label_to_id.keys()))
    cm = confusion_matrix(y_test, preds, labels=labels)
    print(f"Labels: {labels}")
    print(cm)

    # Save artifacts
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, config.SAVED_SKLEARN_MODEL)
    print(f"\n[Saved] Model saved to: {config.SAVED_SKLEARN_MODEL}")
    save_label_mapping(label_to_id, config.SAVED_LABELS)
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
