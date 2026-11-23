"""
Real-time Sign Language Recognition / ระบบแปลภาษามือสดผ่านกล้อง
Captures camera frames, detects start of sign via Ready Zone, supports Dynamic Early-Stop
(0.55s - 1.8s) for fast signing, and classifies using Hybrid Specialist / Single Engines.

Usage:
    python src/realtime.py
    python src/realtime.py --model hybrid
    python src/realtime.py --model cnn1d
    python src/realtime.py --model gru
    python src/realtime.py --model rf_kinematic
    python src/realtime.py --model rf_legacy
    python src/realtime.py --mode auto       # Trigger automatically when hand raised (default)
    python src/realtime.py --mode manual     # Press SPACE to trigger recording
"""
import argparse
from collections import deque
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_DIR = Path(__file__).resolve().parent
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from features import extract_frame_features
from landmarkers import Camera, Tracker, draw_hand, draw_pose, hands_by_side
from models import HybridEnsemblePredictor


def load_classifier(model_type="auto", scenario=None):
    """
    Loads HybridEnsemblePredictor (optionally from a scenario-specific model folder)
    and resolves requested engine key.
    """
    model_dir = config.get_scenario_model_dir(scenario) if scenario else config.MODEL_DIR
    predictor = HybridEnsemblePredictor.load_from_disk(model_dir=model_dir, fallback_global=True)
    if predictor is None or not predictor.available_engines():
        raise FileNotFoundError(
            "No trained models found in models/. Run: python main.py --train"
        )

    avail = predictor.available_engines()
    target = model_type
    if target == "auto":
        target = "hybrid" if "hybrid" in avail else next(iter(avail))
    elif target == "rf":
        target = "rf_kinematic" if "rf_kinematic" in avail else "rf_legacy"

    if target not in avail:
        fallback = next((c for c in ("hybrid", "gru", "cnn1d", "rf_kinematic", "rf_legacy") if c in avail), None)
        if fallback is None:
            raise FileNotFoundError(f"Requested engine '{model_type}' is unavailable.")
        print(f"[Warning] Engine '{target}' not found on disk. Falling back to '{fallback}'.")
        target = fallback

    print(f"[Model] Loaded engine '{target}' ({len(predictor.id_to_label)} gestures)")
    return target, predictor, predictor.id_to_label


def predict(model_type, predictor, raw_frames, id_to_label=None):
    """
    Predicts gesture from raw (T, 112) recorded frames using the chosen engine.
    Returns: (predicted_label, confidence_score, engine_detail)
    """
    return predictor.predict(raw_frames, engine=model_type)


def get_ready_line_y(pose_lms, h):
    """
    Computes y-coordinate (pixels) of the ready line (at lower chest / rib / waist level).
    If pose landmarks are visible, calculate relative to shoulder position and width.
    Otherwise default to 74% of screen height.
    """
    if pose_lms is not None:
        try:
            l_sh = pose_lms[config.POSE_KEEP["l_shoulder"]]
            r_sh = pose_lms[config.POSE_KEEP["r_shoulder"]]
            cy = (l_sh.y + r_sh.y) / 2.0
            sh_w = np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
            mult = getattr(config, "READY_LINE_SHOULDER_MULT", 0.92)
            chest_y = cy + mult * max(sh_w, 0.15)
            chest_y = np.clip(chest_y, 0.40, 0.90)
            return int(chest_y * h)
        except Exception:
            pass
    ratio = getattr(config, "READY_LINE_RATIO", 0.74)
    return int(ratio * h)


def is_hand_in_ready_zone(hands, ready_line_y, h):
    """Returns True if at least one hand's wrist is above the ready line."""
    for side in ("Left", "Right"):
        lms = hands.get(side)
        if lms is not None and len(lms) > 0:
            if int(lms[0].y * h) < ready_line_y:
                return True
    return False


def draw_recognition_hud(img, state, state_desc, sign_label, conf, threshold,
                         model_name, mode, fps, progress=0.0, engine_detail=""):
    """Draw top header HUD, status badge, prediction result, and progress/confidence bar."""
    h, w = img.shape[:2]
    banner_h = 115
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

    # Line 1: Header (Model/Specialist, Mode, FPS, Controls)
    eng_str = engine_detail if (state == "result" and engine_detail) else model_name.upper()
    left_meta = f"{eng_str}  |  {mode.upper()}  |  {fps:4.1f} FPS"
    right_meta = "[M] Mode  |  [SPACE] Trigger"
    cv2.putText(img, left_meta, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, right_meta, (w - 235, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 160, 160), 1, cv2.LINE_AA)

    # Line 2: Status Badge & Description
    badge_colors = {
        "idle": (140, 140, 140),
        "armed": (0, 215, 255),
        "recording": (0, 0, 255),
        "result": (0, 230, 0) if conf >= threshold else (0, 160, 255),
    }
    b_color = badge_colors.get(state, (200, 200, 200))
    cv2.putText(img, state_desc, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.60, b_color, 2, cv2.LINE_AA)

    # Line 3: Big Sign Output
    if state == "result":
        is_confident = conf >= threshold
        text_color = (0, 255, 0) if is_confident else (0, 160, 255)
        display_text = f"Sign: {sign_label}"
        conf_text = f"{int(conf * 100)}%"
        cv2.putText(img, display_text, (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.92, text_color, 2, cv2.LINE_AA)
        cv2.putText(img, conf_text, (w - 85, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2, cv2.LINE_AA)

    elif state == "recording":
        cv2.putText(img, "Sign: (Recording motion...)", (15, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)
        pct_text = f"{int(progress * 100)}%"
        cv2.putText(img, pct_text, (w - 85, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)

    elif state == "armed":
        cv2.putText(img, "Sign: (Hold steady... starting)", (15, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 215, 255), 2, cv2.LINE_AA)

    else:
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
    """Draw subtle ready line guide at lower chest level."""
    h, w = img.shape[:2]
    if state == "recording":
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        return

    line_color = (0, 220, 255) if hand_inside else (120, 120, 120)
    line_thickness = 2 if hand_inside else 1

    segment_w = 20
    for x in range(0, w, segment_w * 2):
        cv2.line(img, (x, ready_line_y), (min(x + segment_w, w), ready_line_y), line_color, line_thickness)

    guide_text = "READY LINE (Lower Chest / Waist)" if not hand_inside else "HAND IN ZONE"
    cv2.putText(img, guide_text, (w - 245, ready_line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, line_color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Real-time Sign Language Recognition")
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "hybrid", "gru", "cnn1d", "rf_kinematic", "rf_legacy", "rf"],
        help="Model engine to use (default: auto -> hybrid specialist)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "manual"],
        help="Trigger mode: 'auto' or 'manual' (default: auto)",
    )
    parser.add_argument("--scenario", type=str, default=None, help="Optional scenario preset name (e.g. 'Hospital & Clinic')")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold (default: 0.60)")
    parser.add_argument("--duration", type=float, default=config.MAX_SIGN_DURATION, help="Max recording seconds (default: 1.8)")
    parser.add_argument("--cooldown", type=float, default=1.4, help="Max display cooldown in seconds (default: 1.4)")
    args = parser.parse_args()

    try:
        model_type, predictor, id_to_label = load_classifier(args.model, scenario=args.scenario)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)

    cam = Camera()
    print(f"[Ready] Camera running at {cam.actual_size()}")
    print(f"[Config] Engine: {model_type} | Mode: {args.mode.upper()} | Max Duration: {args.duration}s | Threshold: {args.threshold}")

    state = "idle"
    mode = args.mode
    hand_in_zone_start = None
    ARM_HOLD_DURATION = 0.25

    rec_start_time = 0.0
    recorded_frames = []
    pre_roll = deque(maxlen=config.PRE_ROLL_FRAMES)
    still_start_time = None
    rec_peak_speed = 0.0
    hand_dropped_frames = 0
    retrigger_move_frames = 0
    prev_wrist_xy = None

    last_label = "-"
    last_conf = 0.0
    last_detail = ""
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

            wrist_vec = frame_vec[10:14]
            wrist_speed = float(np.linalg.norm(wrist_vec - prev_wrist_xy)) if prev_wrist_xy is not None else 0.0
            prev_wrist_xy = wrist_vec.copy()

            progress = 0.0
            draw_state = state

            if state == "idle":
                pre_roll.append(frame_vec)
                state_desc = "IDLE: Raise hand above line to begin" if mode == "auto" else "MANUAL: Press [SPACE] to record"
                if mode == "auto" and hand_inside:
                    if hand_in_zone_start is None:
                        hand_in_zone_start = now
                        draw_state = "armed"
                        state_desc = "ARMING: Hold steady at ready position..."
                    elif (now - hand_in_zone_start) >= ARM_HOLD_DURATION:
                        state = "recording"
                        rec_start_time = now
                        recorded_frames = list(pre_roll)
                        still_start_time = None
                        rec_peak_speed = wrist_speed
                        hand_dropped_frames = 0
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
                state_desc = f"RECORDING: Signing ({elapsed:3.1f}s / {args.duration:3.1f}s)"

                rec_peak_speed = max(rec_peak_speed, wrist_speed)
                if wrist_speed < config.WRIST_STILL_THRESHOLD:
                    if still_start_time is None:
                        still_start_time = now
                else:
                    still_start_time = None
                hand_dropped_frames = (hand_dropped_frames + 1) if not hand_inside else 0

                can_early = (elapsed >= config.MIN_SIGN_DURATION) and (len(recorded_frames) >= 7)
                has_moved = rec_peak_speed >= config.WRIST_ACTIVE_PEAK_THRESHOLD
                still_done = has_moved and (still_start_time is not None) and ((now - still_start_time) >= config.END_HOLD_STILL_SEC)
                drop_done = hand_dropped_frames >= 4

                if (elapsed >= args.duration) or (can_early and (drop_done or still_done)):
                    if len(recorded_frames) >= 7:
                        raw_seq = np.array(recorded_frames, dtype=np.float32)
                        label, conf, detail = predict(model_type, predictor, raw_seq, id_to_label)
                        last_label, last_conf, last_detail = label, conf, detail
                    else:
                        last_label, last_conf, last_detail = "(No Motion Detected)", 0.0, ""

                    state = "result"
                    result_start_time = now
                    retrigger_move_frames = 0
                    pre_roll.clear()

            elif state == "result":
                draw_state = "result"
                pre_roll.append(frame_vec)
                cd = now - result_start_time
                state_desc = "RESULT: Lower hand or start next sign"
                progress = 1.0

                if mode == "auto":
                    if not hand_inside and cd >= 0.35:
                        state = "idle"
                        hand_in_zone_start = None
                    elif hand_inside and cd >= 0.45:
                        if wrist_speed >= 0.055:
                            retrigger_move_frames += 1
                        else:
                            retrigger_move_frames = 0
                        if retrigger_move_frames >= 2:
                            # Instant mid-air re-trigger into next sign!
                            state = "recording"
                            rec_start_time = now
                            recorded_frames = list(pre_roll)[-4:]
                            still_start_time = None
                            rec_peak_speed = wrist_speed
                            hand_dropped_frames = 0
                            retrigger_move_frames = 0
                else:
                    if cd >= 0.40:
                        state = "idle"

            view = frame.copy()
            if pose_lms is not None:
                draw_pose(view, pose_lms)
            for side, lms in hands.items():
                if lms is not None:
                    draw_hand(view, lms, side)

            draw_ready_guide(view, ready_line_y, hand_inside, state)
            draw_recognition_hud(view, draw_state, state_desc, last_label, last_conf,
                                 args.threshold, model_type, mode, fps, progress, last_detail)

            cv2.imshow("Sign Language Real-time Recognition", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                state = "recording"
                rec_start_time = now
                recorded_frames = [frame_vec]
                still_start_time = None
                hand_dropped_frames = 0
                hand_in_zone_start = None
            elif key in (ord("m"), ord("M")):
                mode = "manual" if mode == "auto" else "auto"
                hand_in_zone_start = None
                print(f"[Mode] Switched to {mode.upper()} mode")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
