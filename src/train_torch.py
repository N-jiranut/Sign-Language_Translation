"""
Train PyTorch GRU Sequence Model.
เทรนโมเดล PyTorch GRU สำหรับจดจำท่วงท่าภาษามือ
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from dataset import load_raw_dataset, split_dataset, save_label_mapping
from models import SignGRU


def main():
    print("=" * 50)
    print(" Training PyTorch GRU Sequence Model ")
    print("=" * 50)

    try:
        X, y, groups, label_to_id = load_raw_dataset()
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        print("\nTip: Collect samples first using: python src/collect_data.py")
        sys.exit(1)

    num_classes = len(label_to_id)
    if num_classes < 2:
        print("[Warning] Need at least 2 distinct gesture classes to train a classifier.")
        sys.exit(1)

    # Encode string labels to integer IDs
    y_encoded = np.array([label_to_id[name] for name in y], dtype=np.int64)

    X_train, X_test, y_train, y_test = split_dataset(X, y_encoded, groups, test_size=0.2)

    # Convert to PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    batch_size = min(16, max(2, len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    model = SignGRU(
        input_size=config.NUM_FEATURES,
        hidden_size=64,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 40
    best_acc = 0.0

    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(by)
            correct += (out.argmax(dim=-1) == by).sum().item()
            total += len(by)

        train_loss = total_loss / total
        train_acc = correct / total

        # Evaluation
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                test_correct += (out.argmax(dim=-1) == by).sum().item()
                test_total += len(by)

        test_acc = test_correct / test_total if test_total > 0 else 0.0

        if test_acc >= best_acc:
            best_acc = test_acc
            # Save checkpoint
            config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_to_id": label_to_id,
                    "input_size": config.NUM_FEATURES,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "num_classes": num_classes,
                },
                config.SAVED_TORCH_MODEL,
            )

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:2d}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.1f}% | Test Acc: {test_acc * 100:.1f}%"
            )

    print("\n" + "=" * 50)
    print(f"Training Finished! Best Test Accuracy: {best_acc * 100:.2f}%")
    print(f"[Saved] Model saved to {config.SAVED_TORCH_MODEL}")
    save_label_mapping(label_to_id, config.SAVED_LABELS)


if __name__ == "__main__":
    main()
