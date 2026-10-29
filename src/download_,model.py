"""
Download the MediaPipe model files (.task) into models/mediapipe/.
ดาวน์โหลดไฟล์โมเดล MediaPipe (รันครั้งเดียว)

    python src/download_models.py
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def download(url, path):
    if path.exists():
        print(f"[skip] {path.name} already exists / มีอยู่แล้ว")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {path.name} ...")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)
    print(f"[ok] {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    download(config.HAND_MODEL_URL, config.HAND_MODEL_PATH)
    download(config.POSE_MODEL_URL, config.POSE_MODEL_PATH)