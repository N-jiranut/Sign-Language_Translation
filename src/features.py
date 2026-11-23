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


def trim_active_motion(frames, pad_before=2, pad_after=3, min_frames=6):
    """
    Trims leading and trailing idle frames where neither hand is detected,
    so fast and slow recordings isolate the active signing gesture consistently.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or len(frames) == 0:
        return frames

    active = np.where((frames[:, -2] >= 0.5) | (frames[:, -1] >= 0.5))[0]
    if len(active) >= min_frames:
        start = max(0, int(active[0]) - pad_before)
        end = min(len(frames), int(active[-1]) + pad_after)
        return frames[start:end]
    return frames


def _masked_velocity(seq):
    """
    Computes frame-to-frame velocity for a (T, 112) sequence while zeroing out
    false spikes when a hand appears/disappears and zeroing out presence flags.
    """
    seq = np.asarray(seq, dtype=np.float32)
    vel = np.zeros_like(seq, dtype=np.float32)
    if len(seq) > 1:
        vel[1:] = seq[1:] - seq[:-1]
        l_both = (seq[1:, -2] >= 0.5) & (seq[:-1, -2] >= 0.5)
        r_both = (seq[1:, -1] >= 0.5) & (seq[:-1, -1] >= 0.5)
        vel[1:][~l_both, 26:68] = 0.0
        vel[1:][~r_both, 68:110] = 0.0
    vel[:, -2:] = 0.0
    return vel


def add_velocity_channels(seq_24x112):
    """
    Augments a (T, 112) landmark sequence with masked velocity channels -> (T, 222).
    Order: [110 spatial coords, 110 spatial velocities, 2 hand presence flags].
    """
    seq = np.asarray(seq_24x112, dtype=np.float32)
    vel = _masked_velocity(seq)[:, :110]
    return np.concatenate([seq[:, :110], vel, seq[:, 110:112]], axis=-1).astype(np.float32)


def add_velocity_batch(X):
    """Converts batch (N, T, 112) to (N, T, 222) with velocity channels."""
    if len(X) == 0:
        return np.zeros((0, X.shape[1], config.NUM_SEQ_FEATURES), dtype=np.float32)
    return np.stack([add_velocity_channels(s) for s in X], axis=0).astype(np.float32)


def extract_kinematic_features(seq_24x112):
    """
    Extracts time-shift-invariant statistical and kinematic features (1008 dims)
    from a (T, 112) sequence for the Kinematic Random Forest classifier.
    """
    seq = np.asarray(seq_24x112, dtype=np.float32)
    vel = _masked_velocity(seq)
    vel_use = vel[1:] if len(seq) > 1 else vel

    return np.concatenate(
        [
            seq.mean(axis=0),
            seq.std(axis=0),
            seq.min(axis=0),
            seq.max(axis=0),
            np.percentile(seq, 25, axis=0),
            np.percentile(seq, 75, axis=0),
            vel_use.mean(axis=0),
            vel_use.std(axis=0),
            np.abs(vel_use).max(axis=0),
        ],
        axis=0,
    ).astype(np.float32)


def extract_kinematic_batch(X):
    """Extracts kinematic summary vectors (N, 1008) from batch (N, T, 112)."""
    if len(X) == 0:
        return np.zeros((0, config.NUM_FEATURES * 9), dtype=np.float32)
    return np.stack([extract_kinematic_features(s) for s in X], axis=0).astype(np.float32)


def compute_motion_energy(seq_24x112):
    """Computes average wrist motion energy from masked pose wrist velocities."""
    seq = np.asarray(seq_24x112, dtype=np.float32)
    if len(seq) <= 1:
        return 0.0
    vel = _masked_velocity(seq)
    wrist_speed = np.linalg.norm(vel[1:, 10:12], axis=1) + np.linalg.norm(vel[1:, 12:14], axis=1)
    return float(np.mean(wrist_speed))


def mirror_sequence_horizontal(seq):
    """
    Horizontally mirrors a (T, 112) normalized sequence across the vertical shoulder axis:
      - Negates all x coordinates (even columns 0..108)
      - Swaps Left/Right pose pairs (shoulders, elbows, wrists, hand tips)
      - Swaps Left hand (26:68) <-> Right hand (68:110) and presence flags (-2 <-> -1)
    """
    m = np.asarray(seq, dtype=np.float32).copy()
    if m.ndim != 2 or m.shape[1] != config.NUM_FEATURES:
        return m

    # Negate all x coordinates (pose + both hands)
    m[:, 0:110:2] *= -1.0

    # Swap L/R pose landmark pairs
    for p_l, p_r in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)):
        tmp = m[:, p_l * 2 : p_l * 2 + 2].copy()
        m[:, p_l * 2 : p_l * 2 + 2] = m[:, p_r * 2 : p_r * 2 + 2]
        m[:, p_r * 2 : p_r * 2 + 2] = tmp

    # Swap Left hand (26:68) and Right hand (68:110)
    tmp_h = m[:, 26:68].copy()
    m[:, 26:68] = m[:, 68:110]
    m[:, 68:110] = tmp_h

    # Swap hand presence flags
    tmp_f = m[:, -2].copy()
    m[:, -2] = m[:, -1]
    m[:, -1] = tmp_f
    return m


def canonicalize_handedness(seq):
    """
    Auto-mirrors single-handed gestures signed with the opposite hand so both
    left-handed and right-handed signers map onto the same canonical side.
    Two-handed gestures remain untouched.
    """
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim != 2 or len(arr) == 0 or arr.shape[1] != config.NUM_FEATURES:
        return arr
    l_ratio = float((arr[:, -2] >= 0.5).mean())
    r_ratio = float((arr[:, -1] >= 0.5).mean())
    if r_ratio >= 0.25 and l_ratio <= 0.15:
        return mirror_sequence_horizontal(arr)
    return arr


if __name__ == "__main__":
    # Self-test
    dummy_vec = extract_frame_features(None, {"Left": None, "Right": None})
    print(f"Feature vector shape: {dummy_vec.shape} (Expected: ({config.NUM_FEATURES},))")
    assert dummy_vec.shape == (config.NUM_FEATURES,), "Dimension mismatch!"

    dummy_seq = np.random.randn(35, config.NUM_FEATURES).astype(np.float32)
    dummy_seq[:, -2:] = 0.0
    dummy_seq[8:26, -1] = 1.0
    trimmed = trim_active_motion(dummy_seq)
    assert len(trimmed) < len(dummy_seq) and len(trimmed) > 0, "Trimming failed!"

    res = resample_sequence(trimmed, config.TARGET_FRAMES)
    print(f"Resampled sequence shape: {res.shape} (Expected: ({config.TARGET_FRAMES}, {config.NUM_FEATURES}))")
    assert res.shape == (config.TARGET_FRAMES, config.NUM_FEATURES), "Resample shape mismatch!"

    vel_seq = add_velocity_channels(res)
    print(f"Velocity sequence shape: {vel_seq.shape} (Expected: ({config.TARGET_FRAMES}, {config.NUM_SEQ_FEATURES}))")
    assert vel_seq.shape == (config.TARGET_FRAMES, config.NUM_SEQ_FEATURES), "Velocity shape mismatch!"

    kin_vec = extract_kinematic_features(res)
    print(f"Kinematic vector shape: {kin_vec.shape} (Expected: ({config.NUM_FEATURES * 9},))")
    assert kin_vec.shape == (config.NUM_FEATURES * 9,), "Kinematic shape mismatch!"

    energy = compute_motion_energy(res)
    assert isinstance(energy, float) and energy >= 0.0, "Motion energy invalid!"
    print(f"Motion energy sample: {energy:.4f}")
    print("Features self-test passed successfully!")
