"""
Train Deep Learning Sequence Models (PyTorch GRU & 1D-CNN) + Hybrid Ensemble Metadata.
เทรนโมเดล PyTorch GRU, 1D-CNN และสร้างตาราง Hybrid Specialist Ensemble
"""
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_all import train_all_models


def main():
    try:
        train_all_models(verbose=True)
    except Exception as e:
        print(f"[Error] Failed to train models: {e}")
        print("\nTip: Collect samples first using: python src/collect_data.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
