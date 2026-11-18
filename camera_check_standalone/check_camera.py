"""
Standalone Camera & Posture Calibration Tool / เครื่องมือจัดตำแหน่งกล้องและสรีระแบบแยกไฟล์เดี่ยว
ใช้ตรวจสอบมุมกล้องและระยะห่างบนเครื่องอัดวิดีโอก่อนถ่ายทำจริง

Calibrated Setup Reference (ระยะมาตรฐานที่วัดจริง):
  - ความสูงฐานกล้องจากพื้น: 161 ซม. (Camera Height: 161 cm)
  - ระยะปลายเท้าถึงฐานกล้อง: 90 ซม. (Feet Distance: 90 cm)
  - มุมก้มหน้ากล้อง: 20° - 30° ก้มลงมาที่ระดับอก/เอว

Usage:
  python check_camera.py
  python check_camera.py --camera 1       # สลับกล้องถ้ามีหลายตัว
  python check_camera.py --no-flip        # กรณีใช้กล้องหลัง ไม่ต้อง Mirror
"""
import argparse
import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

POSE_KEEP = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_pinky": 17, "r_pinky": 18,
    "l_index": 19, "r_index": 20,
    "l_thumb": 21, "r_thumb": 22,
}

_POSE_EDGES = [("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"),
               ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist")]
HAND_COLORS = {"Left": (255, 160, 0), "Right": (0, 200, 255)}
_HAND_EDGES = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
SHOULDER_WIDTH_GOOD = (0.28, 0.48)


def ensure_models():
    """Ensure MediaPipe task models exist; auto-download if missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in [(HAND_MODEL_PATH, HAND_MODEL_URL), (POSE_MODEL_PATH, POSE_MODEL_URL)]:
        if not path.exists():
            print(f"[Download] Downloading {path.name} from Google Cloud...")
            try:
                urllib.request.urlretrieve(url, str(path))
                print(f"[Done] Downloaded {path.name} ({path.stat().st_size / (1024*1024):.2f} MB)")
            except Exception as e:
                print(f"[Error] Failed to download {path.name}: {e}")
                sys.exit(1)


# --------------------------------------------------------------------------
# Camera & Tracker
# --------------------------------------------------------------------------
class Camera:
    def __init__(self, index=0, width=640, height=480, fps=30):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}. Try --camera 1 or 2.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def actual_size(self):
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def release(self):
        self.cap.release()


class Tracker:
    def __init__(self):
        ensure_models()
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = vision.HandLandmarker.create_from_options(hand_opts)
        self.pose = vision.PoseLandmarker.create_from_options(pose_opts)
        self._last_ts = -1

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = time.monotonic_ns() // 1_000_000
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        t0 = time.perf_counter()
        hand_res = self.hands.detect_for_video(image, ts)
        t1 = time.perf_counter()
        pose_res = self.pose.detect_for_video(image, ts)
        t2 = time.perf_counter()
        return hand_res, pose_res, {"hands": (t1 - t0) * 1000, "pose": (t2 - t1) * 1000}

    def close(self):
        self.hands.close()
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def hands_by_side(hand_res):
    out = {"Left": None, "Right": None}
    best = {"Left": -1.0, "Right": -1.0}
    for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
        side, score = handed[0].category_name, handed[0].score
        if side in out and score > best[side]:
            out[side], best[side] = lms, score
    return out


# --------------------------------------------------------------------------
# Drawing Functions
# --------------------------------------------------------------------------
def _px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def draw_hand(img, lms, side):
    h, w = img.shape[:2]
    color = HAND_COLORS.get(side, (0, 255, 0))
    for a, b in _HAND_EDGES:
        cv2.line(img, _px(lms[a], w, h), _px(lms[b], w, h), color, 2, cv2.LINE_AA)
    for lm in lms:
        cv2.circle(img, _px(lm, w, h), 3, (255, 255, 255), -1, cv2.LINE_AA)
    x, y = _px(lms[0], w, h)
    cv2.putText(img, side, (x - 20, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_pose(img, pose_lms):
    h, w = img.shape[:2]
    for a, b in _POSE_EDGES:
        cv2.line(img, _px(pose_lms[POSE_KEEP[a]], w, h), _px(pose_lms[POSE_KEEP[b]], w, h), (120, 220, 120), 2, cv2.LINE_AA)
    for i in POSE_KEEP.values():
        cv2.circle(img, _px(pose_lms[i], w, h), 4, (80, 200, 80), -1, cv2.LINE_AA)


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_len=8, gap_len=6):
    x1, y1 = pt1
    x2, y2 = pt2
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist < 1e-3:
        return
    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist
    curr_dist = 0.0
    drawing = True
    while curr_dist < dist:
        seg_len = dash_len if drawing else gap_len
        next_dist = min(curr_dist + seg_len, dist)
        if drawing:
            p_start = (int(x1 + dx * curr_dist), int(y1 + dy * curr_dist))
            p_end = (int(x1 + dx * next_dist), int(y1 + dy * next_dist))
            cv2.line(img, p_start, p_end, color, thickness, cv2.LINE_AA)
        curr_dist = next_dist
        drawing = not drawing


def draw_corner_bracket(img, x1, y1, x2, y2, color, thickness=2, corner_len=18):
    c = corner_len
    cv2.line(img, (x1, y1), (x1 + c, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + c), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - c, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + c), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + c, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - c), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - c, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - c), color, thickness, cv2.LINE_AA)


def draw_crosshair_target(img, cx, cy, color, radius=18):
    cv2.circle(img, (cx, cy), radius, color, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, color, -1, cv2.LINE_AA)
    cv2.line(img, (cx - radius - 6, cy), (cx + radius + 6, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - radius - 6), (cx, cy + radius + 6), color, 1, cv2.LINE_AA)


def body_inside(pose_lms, margin=0.03):
    if pose_lms is None:
        return False
    keys = ("l_shoulder", "r_shoulder", "l_elbow", "r_elbow")
    for k in keys:
        lm = pose_lms[POSE_KEEP[k]]
        if not (margin <= lm.x <= 1 - margin and margin <= lm.y <= 1 - margin):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Standalone Camera & Posture Calibration Tool")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-flip", action="store_true", help="Do not flip horizontally (e.g. rear camera)")
    args = parser.parse_args()

    flip_img = not args.no_flip

    print("=" * 68)
    print(" Standalone Camera & Posture Calibration Tool ")
    print(" เครื่องมือจัดตำแหน่งกล้องและสรีระ (สำหรับเครื่องอัดวิดีโอ) ")
    print("=" * 68)
    print(" [Calibrated Setup Parameters / ค่าตำแหน่งมาตรฐานที่วัดจริง]")
    print("  • ความสูงฐานกล้องจากพื้น : 161 ซม.")
    print("  • ระยะปลายเท้าห่างจากกล้อง : 90 ซม.")
    print("  • มุมก้มหน้ากล้อง       : 20° - 30° ก้มลงที่ระดับอก/เอว")
    print("=" * 68)
    print(" ปุ่มลัด: [G] ปิด/เปิดไกด์ | [S] เซฟภาพ Snapshot | [R] รีเซ็ต | [Q] ออก")
    print("=" * 68)

    cam = Camera(index=args.camera)
    actual_w, actual_h = cam.actual_size()
    print(f"[Camera] Active at {actual_w}x{actual_h} (Index: {args.camera}, Flip: {flip_img})")

    win = deque(maxlen=90)
    fps = 0.0
    a = 0.1
    show_guides = True
    ready_streak = 0

    with Tracker() as tracker:
        while True:
            t0 = time.perf_counter()
            frame = cam.read()
            if frame is None:
                continue

            if flip_img:
                frame = cv2.flip(frame, 1)

            hand_res, pose_res, ms = tracker.process(frame)
            hands = hands_by_side(hand_res)
            pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None

            has_pose = pose_lms is not None
            has_l = hands["Left"] is not None
            has_r = hands["Right"] is not None
            has_both = has_l and has_r
            in_frame = body_inside(pose_lms)

            win.append((has_pose, has_l, has_r, has_both, in_frame))

            dt = time.perf_counter() - t0
            cur_fps = (1.0 / dt) if dt > 0 else 30.0
            fps = (1 - a) * fps + a * cur_fps

            view = frame.copy()
            h, w = view.shape[:2]

            # Draw standard skeleton landmarks
            if pose_lms is not None:
                draw_pose(view, pose_lms)
            for side, lms in hands.items():
                if lms is not None:
                    draw_hand(view, lms, side)

            # Metrics
            cx, cy = 0.5, 0.5
            shoulder_w = None
            center_diff = 0.0
            is_centered = False
            headroom_ok = False
            ready_y_px = int(0.74 * h)
            left_elbow_in_bracket = False
            right_elbow_in_bracket = False
            elbows_span_ok = False
            elbow_clipped = False
            wai_clasp_detected = False

            if pose_lms is not None:
                try:
                    l_sh = pose_lms[POSE_KEEP["l_shoulder"]]
                    r_sh = pose_lms[POSE_KEEP["r_shoulder"]]
                    nose = pose_lms[POSE_KEEP["nose"]]
                    l_elb = pose_lms[POSE_KEEP["l_elbow"]]
                    r_elb = pose_lms[POSE_KEEP["r_elbow"]]

                    cx = (l_sh.x + r_sh.x) / 2.0
                    cy = (l_sh.y + r_sh.y) / 2.0
                    shoulder_w = np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
                    center_diff = cx - 0.5
                    is_centered = abs(center_diff) <= 0.05
                    headroom_ok = (0.12 <= nose.y <= 0.42)

                    chest_y = cy + 0.92 * max(shoulder_w, 0.15)
                    ready_y_px = int(np.clip(chest_y, 0.40, 0.90) * h)

                    min_elb_x = min(l_elb.x, r_elb.x)
                    max_elb_x = max(l_elb.x, r_elb.x)
                    left_elbow_in_bracket = (0.03 <= min_elb_x <= 0.18) and (0.38 <= l_elb.y <= 0.82)
                    right_elbow_in_bracket = (0.82 <= max_elb_x <= 0.97) and (0.38 <= r_elb.y <= 0.82)
                    elbows_span_ok = left_elbow_in_bracket and right_elbow_in_bracket
                    elbow_clipped = (min_elb_x < 0.02) or (max_elb_x > 0.98)
                except Exception:
                    pass

            if has_both:
                try:
                    lw = hands["Left"][0]
                    rw = hands["Right"][0]
                    if np.hypot(lw.x - rw.x, lw.y - rw.y) < 0.18:
                        wai_clasp_detected = True
                except Exception:
                    pass

            shoulder_dist_ok = (shoulder_w is not None) and (SHOULDER_WIDTH_GOOD[0] <= shoulder_w <= SHOULDER_WIDTH_GOOD[1])
            distance_ok = elbows_span_ok or shoulder_dist_ok
            all_ready = has_pose and is_centered and distance_ok and in_frame and (not elbow_clipped)

            if all_ready:
                ready_streak += 1
            else:
                ready_streak = max(0, ready_streak - 1)
            is_calibrated = (ready_streak >= 12)

            # Guides Overlay
            if show_guides:
                center_x_px = w // 2
                c_line_color = (0, 240, 120) if is_centered else (0, 165, 255)
                draw_dashed_line(view, (center_x_px, 85), (center_x_px, h - 30), c_line_color, 1, 8, 6)

                if has_pose:
                    t_x, t_y = int(cx * w), int(cy * h)
                    t_color = (0, 240, 120) if is_centered else (0, 165, 255)
                    draw_crosshair_target(view, t_x, t_y, t_color, radius=16)
                    if not is_centered:
                        if center_diff < -0.05:
                            cv2.putText(view, "SHIFT RIGHT ->", (t_x + 22, t_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2, cv2.LINE_AA)
                        else:
                            cv2.putText(view, "<- SHIFT LEFT", (t_x - 145, t_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(view, "CENTER OK", (t_x + 20, t_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 120), 2, cv2.LINE_AA)

                headroom_y = int(0.10 * h)
                hr_color = (0, 220, 120) if headroom_ok else (0, 140, 255)
                draw_dashed_line(view, (15, headroom_y), (w - 15, headroom_y), hr_color, 1, 6, 6)
                cv2.putText(view, "HEAD MARGIN", (w - 125, headroom_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

                draw_dashed_line(view, (15, ready_y_px), (w - 15, ready_y_px), (0, 220, 255), 1, 10, 6)
                cv2.putText(view, "READY LINE (Chest Level)", (18, ready_y_px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)

                by1, by2 = int(0.42 * h), int(0.75 * h)
                bx1_l, bx2_l = int(0.03 * w), int(0.18 * w)
                color_l = (0, 240, 120) if left_elbow_in_bracket else ((0, 0, 255) if elbow_clipped else (120, 120, 120))
                draw_corner_bracket(view, bx1_l, by1, bx2_l, by2, color_l, thickness=2, corner_len=15)
                tag_l = "ELBOW OK" if left_elbow_in_bracket else "L ELBOW"
                cv2.putText(view, tag_l, (bx1_l + 4, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_l, 1, cv2.LINE_AA)

                bx1_r, bx2_r = int(0.82 * w), int(0.97 * w)
                color_r = (0, 240, 120) if right_elbow_in_bracket else ((0, 0, 255) if elbow_clipped else (120, 120, 120))
                draw_corner_bracket(view, bx1_r, by1, bx2_r, by2, color_r, thickness=2, corner_len=15)
                tag_r = "ELBOW OK" if right_elbow_in_bracket else "R ELBOW"
                cv2.putText(view, tag_r, (bx1_r + 4, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_r, 1, cv2.LINE_AA)

                if wai_clasp_detected:
                    cv2.putText(view, "✓ WAI / CLASP TEST DETECTED", (18, ready_y_px + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 120), 2, cv2.LINE_AA)

            # Top HUD
            banner_h = 82
            overlay = view.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.85, view, 0.15, 0, view)

            p1_color = (0, 240, 120) if is_centered else (0, 165, 255)
            p1_text = "[1] Center: OK" if is_centered else "[1] Center: Align"
            p2_color = (0, 240, 120) if distance_ok else (0, 80, 255)
            p2_text = "[2] Distance: OK" if distance_ok else "[2] Dist: Check"
            p3_color = (0, 240, 120) if (in_frame and not elbow_clipped) else (0, 0, 255)
            p3_text = "[3] Elbows: Inside" if (in_frame and not elbow_clipped) else "[3] Elbows: Clipped!"
            p4_color = (0, 240, 120) if has_both else ((0, 200, 255) if (has_l or has_r) else (140, 140, 140))
            p4_text = "[4] Hands: 2 Found" if has_both else "[4] Hands: Test"

            chk_x = 15
            for txt, col in [(p1_text, p1_color), (p2_text, p2_color), (p3_text, p3_color), (p4_text, p4_color)]:
                cv2.putText(view, txt, (chk_x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2, cv2.LINE_AA)
                chk_x += len(txt) * 10 + 16
            cv2.putText(view, f"{fps:4.1f} FPS", (w - 80, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

            if is_calibrated:
                cv2.putText(view, "★ ALIGNMENT READY! มุมและระยะพร้อมสำหรับอัดวิดีโอ ★", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 120), 2, cv2.LINE_AA)
            else:
                msg = "กางศอก 2 ข้างแตะกรอบขอบจอเพื่อวัดระยะ 90 ซม. แล้วพนมมือทดสอบ" if distance_ok else "กางข้อศอกออก 2 ข้างเพื่อวัดระยะห่าง (ระยะเป้าหมาย 90 ซม.)"
                cv2.putText(view, msg, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 2, cv2.LINE_AA)

            flip_tag = "Flip: ON (Mirror)" if flip_img else "Flip: OFF (Direct)"
            setup_info = f"Setup: Height 161 cm | Feet Dist 90 cm | {flip_tag} (Press [F] to toggle)"
            cv2.putText(view, setup_info, (15, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

            b_bar_h = 28
            cv2.rectangle(view, (0, h - b_bar_h), (w, h), (15, 15, 15), -1)
            b_help = "[F] Toggle Flip  |  [G] Toggle Guides  |  [S] Snapshot  |  [R] Reset  |  [Q] Quit"
            cv2.putText(view, b_help, (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Standalone Camera Check", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("f"), ord("F")):
                flip_img = not flip_img
                print(f"[Flip Mode] Toggled: {'Mirror (ON)' if flip_img else 'Direct (OFF)'}")
            elif key == ord("g"):
                show_guides = not show_guides
            elif key == ord("r"):
                win.clear()
                ready_streak = 0
            elif key == ord("s"):
                snap_dir = ROOT / "snapshots"
                snap_dir.mkdir(parents=True, exist_ok=True)
                fn = snap_dir / f"cam_check_{int(time.time())}.jpg"
                cv2.imwrite(str(fn), view)
                print(f"[Saved Snapshot] {fn}")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
