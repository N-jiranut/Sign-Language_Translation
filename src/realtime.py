"""
Real-time Sign Language Recognition / ระบบแปลภาษามือสดผ่านกล้อง
Captures camera frames, detects start of sign via Ready Zone, records the motion window (1.8s),
resamples to 24 frames, and predicts gesture accurately.

Usage:
    python src/realtime.py
    python src/realtime.py --model rf
    python src/realtime.py --model gru
    python src/realtime.py --mode auto       # Trigger automatically when hand raised (default)
    python src/realtime.py --mode manual     # Press SPACE to trigger recording
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from dataset import load_label_mapping
from features import extract_frame_features, resample_sequence
from landmarkers import Camera, Tracker, hands_by_side, draw_hand, draw_pose
from models import SignGRU


def load_classifier(model_type="auto"):
    """
    Loads trained model and label map.
    Supports 'rf' (Random Forest) and 'gru' (PyTorch GRU).
    """
    labels_path = config.SAVED_LABELS
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found at {labels_path}. Train a model first with train_sklearn.py or train_torch.py"
        )
    label_map = load_label_mapping(labels_path)
    id_to_label = {v: k for k, v in label_map.items()}

    if model_type == "auto":
        if config.SAVED_TORCH_MODEL.exists():
            model_type = "gru"
        elif config.SAVED_SKLEARN_MODEL.exists():
            model_type = "rf"
        else:
            raise FileNotFoundError("No trained model found in models/ (neither gru.pt nor rf.pkl)")

    if model_type == "rf":
        if not config.SAVED_SKLEARN_MODEL.exists():
            raise FileNotFoundError(f"Model file {config.SAVED_SKLEARN_MODEL} not found.")
        clf = joblib.load(config.SAVED_SKLEARN_MODEL)
        print(f"[Model] Loaded Random Forest model from {config.SAVED_SKLEARN_MODEL}")
        return "rf", clf, id_to_label

    elif model_type == "gru":
        if not config.SAVED_TORCH_MODEL.exists():
            raise FileNotFoundError(f"Model file {config.SAVED_TORCH_MODEL} not found.")
        ckpt = torch.load(config.SAVED_TORCH_MODEL, map_location="cpu", weights_only=True)
        model = SignGRU(
            input_size=ckpt["input_size"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            num_classes=ckpt["num_classes"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"[Model] Loaded PyTorch GRU model from {config.SAVED_TORCH_MODEL}")
        return "gru", model, id_to_label

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict(model_type, model, seq_24x112, id_to_label):
    """
    Predict gesture from a (24, 112) normalized sequence.
    Returns: (predicted_label, confidence_score)
    """
    if model_type == "rf":
        features_flat = seq_24x112.reshape(1, -1)
        probs = model.predict_proba(features_flat)[0]
        pred_idx = np.argmax(probs)
        conf = float(probs[pred_idx])
        pred_label = model.classes_[pred_idx]
        return str(pred_label), conf

    elif model_type == "gru":
        with torch.no_grad():
            tensor_in = torch.tensor(seq_24x112[None, ...], dtype=torch.float32)
            logits = model(tensor_in)
            probs = torch.softmax(logits, dim=-1)[0].numpy()
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            pred_label = id_to_label.get(pred_idx, f"Class {pred_idx}")
            return pred_label, conf


def get_ready_line_y(pose_lms, h):
    """
    Computes y-coordinate (pixels) of the ready line (at lower chest / rib level).
    If pose landmarks are visible, calculate relative to shoulder position and width.
    Otherwise default to 65% of screen height.
    """
    if pose_lms is not None:
        try:
            l_sh = pose_lms[config.POSE_KEEP["l_shoulder"]]
            r_sh = pose_lms[config.POSE_KEEP["r_shoulder"]]
            cy = (l_sh.y + r_sh.y) / 2.0
            sh_w = np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
            chest_y = cy + 0.65 * max(sh_w, 0.15)
            chest_y = np.clip(chest_y, 0.35, 0.85)
            return int(chest_y * h)
        except Exception:
            pass
    return int(0.65 * h)


def is_hand_in_ready_zone(hands, ready_line_y, h):
    """
    Returns True if at least one hand's wrist is above the ready line (wrist y < ready_line_y).
    """
    for side in ("Left", "Right"):
        lms = hands.get(side)
        if lms is not None and len(lms) > 0:
            wrist_y_px = int(lms[0].y * h)
            if wrist_y_px < ready_line_y:
                return True
    return False


def draw_recognition_hud(img, state, state_desc, sign_label, conf, threshold,
                         model_name, mode, fps, progress=0.0):
    """
    Draw top header HUD, status badge, prediction result, and progress/confidence bar.
    """
    h, w = img.shape[:2]
    banner_h = 115
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

    # Line 1: Header (Model, Mode, FPS, Controls)
    left_meta = f"Model: {model_name.upper()}  |  Mode: {mode.upper()}  |  FPS: {fps:4.1f}"
    right_meta = "[M] Switch Mode  |  [SPACE] Trigger"
    cv2.putText(img, left_meta, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, right_meta, (w - 300, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)

    # Line 2: Status Badge & Description
    badge_colors = {
        "idle": (140, 140, 140),
        "armed": (0, 215, 255),      # Yellow
        "recording": (0, 0, 255),    # Red
        "result": (0, 230, 0) if conf >= threshold else (0, 160, 255),
    }
    b_color = badge_colors.get(state, (200, 200, 200))
    cv2.putText(img, state_desc, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.62, b_color, 2, cv2.LINE_AA)

    # Line 3: Big Sign Output
    if state == "result":
        is_confident = conf >= threshold
        text_color = (0, 255, 0) if is_confident else (0, 160, 255)
        display_text = f"Sign: {sign_label}"
        conf_text = f"{int(conf * 100)}%"
        cv2.putText(img, display_text, (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.95, text_color, 2, cv2.LINE_AA)
        cv2.putText(img, conf_text, (w - 85, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2, cv2.LINE_AA)

    elif state == "recording":
        cv2.putText(img, "Sign: (Recording motion...)", (15, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)
        pct_text = f"{int(progress * 100)}%"
        cv2.putText(img, pct_text, (w - 85, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)

    elif state == "armed":
        cv2.putText(img, "Sign: (Hold steady... starting)", (15, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 215, 255), 2, cv2.LINE_AA)

    else:  # idle
        if sign_label and sign_label != "-":
            prev_text = f"Last Sign: {sign_label} ({int(conf * 100)}%)"
            cv2.putText(img, prev_text, (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (130, 130, 130), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Sign: (Waiting for gesture...)", (15, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.80, (130, 130, 130), 1, cv2.LINE_AA)

    # Line 4: Progress / Confidence Bar
    bar_x, bar_y, bar_w, bar_h = 15, 98, w - 30, 8
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

    if state == "recording":
        fill_w = int(bar_w * np.clip(progress, 0.0, 1.0))
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 0, 255), -1)
    elif state == "result":
        fill_w = int(bar_w * np.clip(conf, 0.0, 1.0))
        bar_color = (0, 220, 0) if conf >= threshold else (0, 160, 255)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

    # Bottom Helper Bar
    bottom_h = 28
    cv2.rectangle(img, (0, h - bottom_h), (w, h), (15, 15, 15), -1)
    help_str = "[SPACE] Record  |  [M] Mode Toggle  |  [Q/ESC] Quit"
    cv2.putText(img, help_str, (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def draw_ready_guide(img, ready_line_y, hand_inside, state):
    """
    Draw subtle ready line guide at chest level.
    """
    h, w = img.shape[:2]
    if state == "recording":
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        return

    line_color = (0, 220, 255) if hand_inside else (120, 120, 120)
    line_thickness = 2 if hand_inside else 1

    segment_w = 20
    for x in range(0, w, segment_w * 2):
        cv2.line(img, (x, ready_line_y), (min(x + segment_w, w), ready_line_y), line_color, line_thickness)

    guide_text = "READY LINE (Chest)" if not hand_inside else "HAND IN ZONE"
    cv2.putText(img, guide_text, (w - 170, ready_line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, line_color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Real-time Sign Language Recognition")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "rf", "gru"],
                        help="Model to use: 'rf', 'gru', or 'auto' (default: auto)")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "manual"],
                        help="Trigger mode: 'auto' (starts when hand raised) or 'manual' (SPACE key) (default: auto)")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Confidence threshold for recognition (default: 0.70)")
    parser.add_argument("--duration", type=float, default=1.8,
                        help="Recording duration in seconds per sign (default: 1.8)")
    parser.add_argument("--cooldown", type=float, default=2.2,
                        help="Display result cooldown in seconds (default: 2.2)")
    args = parser.parse_args()

    try:
        model_type, model, id_to_label = load_classifier(args.model)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)

    cam = Camera()
    print(f"[Ready] Camera running at {cam.actual_size()}")
    print(f"[Config] Mode: {args.mode.upper()} | Sign Duration: {args.duration}s | Threshold: {args.threshold}")
    print("[Instructions] Raise hand above chest line to trigger, or press SPACE to record. Press 'm' to toggle mode, 'q' to quit.")

    state = "idle"
    mode = args.mode

    hand_in_zone_start = None
    ARM_HOLD_DURATION = 0.35

    rec_start_time = 0.0
    recorded_frames = []

    last_label = "-"
    last_conf = 0.0
    result_start_time = 0.0

    last_id = 0
    fps = 0.0
    prev_t = time.perf_counter()

    with Tracker() as tracker:
        while True:
            frame, last_id = cam.read(last_id)
            if frame is None:
                continue

            if config.FLIP_IMAGE:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            hand_res, pose_res, _ = tracker.process(frame)
            hands = hands_by_side(hand_res)
            pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None

            frame_vec = extract_frame_features(pose_lms, hands)

            ready_line_y = get_ready_line_y(pose_lms, h)
            hand_inside = is_hand_in_ready_zone(hands, ready_line_y, h)

            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0)

            progress = 0.0
            draw_state = state

            if state == "idle":
                state_desc = "IDLE: Raise hand above chest line to begin" if mode == "auto" else "MANUAL: Press [SPACE] to record"
                if mode == "auto" and hand_inside:
                    if hand_in_zone_start is None:
                        hand_in_zone_start = now
                        draw_state = "armed"
                        state_desc = "ARMING: Hold steady at ready position..."
                    elif (now - hand_in_zone_start) >= ARM_HOLD_DURATION:
                        state = "recording"
                        rec_start_time = now
                        recorded_frames = [frame_vec]
                        hand_in_zone_start = None
                        draw_state = "recording"
                    else:
                        draw_state = "armed"
                        state_desc = "ARMING: Hold steady at ready position..."
                else:
                    hand_in_zone_start = None

            elif state == "recording":
                draw_state = "recording"
                recorded_frames.append(frame_vec)
                elapsed = now - rec_start_time
                progress = min(1.0, elapsed / args.duration)
                state_desc = f"RECORDING: Signing in progress ({elapsed:3.1f}s / {args.duration:3.1f}s)"

                if elapsed >= args.duration:
                    if len(recorded_frames) >= 10:
                        seq_24 = resample_sequence(np.array(recorded_frames), target_len=config.TARGET_FRAMES)
                        label, conf = predict(model_type, model, seq_24, id_to_label)
                        last_label = label
                        last_conf = conf
                    else:
                        last_label = "(No Motion Detected)"
                        last_conf = 0.0

                    state = "result"
                    result_start_time = now

            elif state == "result":
                draw_state = "result"
                elapsed = now - result_start_time
                state_desc = "RESULT: Translated gesture output"
                progress = 1.0

                if elapsed >= args.cooldown:
                    if mode == "auto":
                        if not hand_inside or elapsed >= (args.cooldown + 1.5):
                            state = "idle"
                            hand_in_zone_start = None
                    else:
                        state = "idle"

            # Visual Rendering
            view = frame.copy()
            if pose_lms is not None:
                draw_pose(view, pose_lms)
            for side, lms in hands.items():
                if lms is not None:
                    draw_hand(view, lms, side)

            draw_ready_guide(view, ready_line_y, hand_inside, state)
            draw_recognition_hud(view, draw_state, state_desc, last_label, last_conf,
                                 args.threshold, model_type, mode, fps, progress)

            cv2.imshow("Sign Language Real-time Recognition", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            elif key == ord(" "):
                state = "recording"
                rec_start_time = now
                recorded_frames = [frame_vec]
                hand_in_zone_start = None

            elif key in (ord("m"), ord("M")):
                mode = "manual" if mode == "auto" else "auto"
                hand_in_zone_start = None
                print(f"[Mode] Switched to {mode.upper()} mode")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
