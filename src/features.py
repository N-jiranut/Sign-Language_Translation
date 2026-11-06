"""
Feature extraction, normalization, and sequence resampling.
การแปลงจุดพิกัด MediaPipe ให้อยู่ในรูปเวกเตอร์ 112 มิติ และปรับความยาวเฟรมให้คงที่
"""
import sys
from pathlib import Path
import numpy as np

_DIR = Path(__file__).resolve().parent
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config

POSE_ORDER = [
    "nose",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
    "l_pinky", "r_pinky",
    "l_index", "r_index",
    "l_thumb", "r_thumb",
]


def extract_frame_features(pose_lms, hands_dict):
    """
    Extract 112 normalized features from one frame:
      - 26 values: 13 pose landmarks (x, y) normalized by shoulder center and width
      - 42 values: 21 left hand landmarks (x, y) normalized by left wrist and hand scale
      - 42 values: 21 right hand landmarks (x, y) normalized by right wrist and hand scale
      -  2 values: hand presence flags [left_present, right_present]
    Total: 112 dimensions.
    """
    # 1. Pose features (26 values)
    pose_feats = np.zeros(len(POSE_ORDER) * 2, dtype=np.float32)
    if pose_lms is not None:
        try:
            l_sh = pose_lms[config.POSE_KEEP["l_shoulder"]]
            r_sh = pose_lms[config.POSE_KEEP["r_shoulder"]]
            cx = (l_sh.x + r_sh.x) / 2.0
            cy = (l_sh.y + r_sh.y) / 2.0
            shoulder_w = np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
            # Guard against scale explosion when shoulders are occluded or close
            if shoulder_w < 0.05:
                shoulder_w = 0.35

            for i, name in enumerate(POSE_ORDER):
                idx = config.POSE_KEEP[name]
                lm = pose_lms[idx]
                pose_feats[i * 2] = (lm.x - cx) / shoulder_w
                pose_feats[i * 2 + 1] = (lm.y - cy) / shoulder_w
        except (IndexError, KeyError):
            pass

    # 2. Hand features (42 values each)
    def normalize_hand(lms):
        feats = np.zeros(21 * 2, dtype=np.float32)
        if lms is None or len(lms) < 21:
            return feats, 0.0

        wx, wy = lms[0].x, lms[0].y
        # Scale by distance from wrist (0) to middle finger base / MCP (9)
        scale = np.hypot(lms[9].x - wx, lms[9].y - wy)
        # Guard against division-by-near-zero explosion when pointing fingers towards camera
        if scale < 0.03:
            scale = 0.1

        for i in range(21):
            feats[i * 2] = (lms[i].x - wx) / scale
            feats[i * 2 + 1] = (lms[i].y - wy) / scale

        return feats, 1.0

    left_lms = hands_dict.get("Left") if hands_dict else None
    right_lms = hands_dict.get("Right") if hands_dict else None

    left_feats, left_flag = normalize_hand(left_lms)
    right_feats, right_flag = normalize_hand(right_lms)
    flags = np.array([left_flag, right_flag], dtype=np.float32)

    vec = np.concatenate([pose_feats, left_feats, right_feats, flags])
    return vec


def resample_sequence(frames, target_len=config.TARGET_FRAMES):
    """
    Resample a sequence of (T, 112) frame vectors to (target_len, 112).
    Uses linear interpolation across time.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError(f"Expected 2D array (T, features), got {frames.shape}")

    t_len, n_feat = frames.shape
    if t_len == target_len:
        res = frames.copy()
        res[:, -2:] = (res[:, -2:] >= 0.5).astype(np.float32)
        return res

    if t_len == 0:
        return np.zeros((target_len, n_feat), dtype=np.float32)

    if t_len == 1:
        res = np.repeat(frames, target_len, axis=0)
        res[:, -2:] = (res[:, -2:] >= 0.5).astype(np.float32)
        return res

    orig_times = np.linspace(0.0, 1.0, t_len)
    target_times = np.linspace(0.0, 1.0, target_len)

    resampled = np.zeros((target_len, n_feat), dtype=np.float32)
    for col in range(n_feat):
        resampled[:, col] = np.interp(target_times, orig_times, frames[:, col])

    # Ensure binary flags remain clean binary 0.0 or 1.0
    resampled[:, -2:] = (resampled[:, -2:] >= 0.5).astype(np.float32)
    return resampled


if __name__ == "__main__":
    # Self-test
    dummy_vec = extract_frame_features(None, {"Left": None, "Right": None})
    print(f"Feature vector shape: {dummy_vec.shape} (Expected: ({config.NUM_FEATURES},))")
    assert dummy_vec.shape == (config.NUM_FEATURES,), "Dimension mismatch!"

    dummy_seq = np.random.randn(35, config.NUM_FEATURES).astype(np.float32)
    res = resample_sequence(dummy_seq, config.TARGET_FRAMES)
    print(f"Resampled sequence shape: {res.shape} (Expected: ({config.TARGET_FRAMES}, {config.NUM_FEATURES}))")
    assert res.shape == (config.TARGET_FRAMES, config.NUM_FEATURES), "Resample shape mismatch!"
    print("Features self-test passed successfully!")
