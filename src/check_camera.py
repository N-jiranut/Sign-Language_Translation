"""
Camera / position test mode.
โหมดทดสอบกล้องและตำแหน่งผู้ใช้

Shows in real time:
  - FPS and time spent in HandLandmarker / PoseLandmarker
  - % of recent frames where pose, left hand, right hand were found
  - whether shoulders + elbows are fully inside the frame
  - shoulder width (rough distance hint)

Use it to choose camera height, distance and lighting, and to test
hard signs (e.g. ไหว้) before collecting data.

    python src/check_camera.py

Keys:  q / Esc = quit   r = reset stats   s = save screenshot
On exit, a summary is printed so you can compare setups in the report.
"""
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

_DIR = Path(__file__).resolve().parent
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
from landmarkers import Camera, Tracker, hands_by_side, draw_hand, draw_pose  # noqa: E402

IDX = config.POSE_KEEP


def distance_hint(shoulder_w):
    """Classify rough distance from shoulder width in normalized coordinates (0..1).
    กะระยะห่างจากความกว้างไหล่ (0..1)"""
    if shoulder_w is None:
        return "no body", (160, 160, 160)
    lo, hi = config.SHOULDER_WIDTH_GOOD
    if shoulder_w < lo:
        return f"too far (< {lo:.2f})", (60, 60, 255)       # red
    if shoulder_w > hi:
        return f"too close (> {hi:.2f})", (60, 60, 255)     # red
    return "distance OK", (80, 220, 80)                     # green


def body_inside(pose_lms, margin=0.03):
    """True if shoulders and elbows are not clipped by the frame edges.
    ไหล่และข้อศอกสองข้างไม่ตกขอบภาพ (เหลือขอบอย่างน้อย margin)"""
    if pose_lms is None:
        return False
    keys = ("l_shoulder", "r_shoulder", "l_elbow", "r_elbow")
    for k in keys:
        lm = pose_lms[IDX[k]]
        if not (margin <= lm.x <= 1 - margin and margin <= lm.y <= 1 - margin):
            return False
    return True


def pct(iterable):
    items = list(iterable)
    return 100.0 * sum(items) / len(items) if items else 0.0


def color_for(p):
    if p >= 90:
        return (80, 220, 80)
    if p >= 60:
        return (60, 220, 220)
    return (60, 60, 255)


def main():
    print("=" * 60)
    print(" Camera & Position Check / โหมดทดสอบกล้องและตำแหน่งผู้ใช้ ")
    print("=" * 60)
    print("q / Esc = quit | r = reset stats | s = screenshot")

    cam = Camera()
    print(f"Camera opened at {cam.actual_size()}")

    win = deque(maxlen=config.STATS_WINDOW)
    all_frames = []

    fps = 0.0
    t_hands = 0.0
    t_pose = 0.0
    a = 0.1
    last_id = 0

    with Tracker() as tracker:
        while True:
            t0 = time.perf_counter()
            frame, last_id = cam.read(last_id)
            if frame is None:
                continue

            if config.FLIP_IMAGE:
                frame = cv2.flip(frame, 1)

            hand_res, pose_res, ms = tracker.process(frame)
            hands = hands_by_side(hand_res)
            pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None

            has_pose = pose_lms is not None
            has_l = hands["Left"] is not None
            has_r = hands["Right"] is not None
            has_both = has_l and has_r
            in_frame = body_inside(pose_lms)

            sample = (has_pose, has_l, has_r, has_both, in_frame)
            win.append(sample)
            all_frames.append(sample)

            dt = time.perf_counter() - t0
            cur_fps = (1.0 / dt) if dt > 0 else 30.0
            fps = (1 - a) * fps + a * cur_fps
            t_hands = (1 - a) * t_hands + a * ms["hands"]
            t_pose = (1 - a) * t_pose + a * ms["pose"]

            view = frame.copy()
            if pose_lms is not None:
                draw_pose(view, pose_lms)
            for side, lms in hands.items():
                if lms is not None:
                    draw_hand(view, lms, side)

            shoulder_w = None
            if pose_lms is not None:
                l_sh = pose_lms[IDX["l_shoulder"]]
                r_sh = pose_lms[IDX["r_shoulder"]]
                shoulder_w = np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
            hint, hint_color = distance_hint(shoulder_w)

            cols = list(zip(*win)) if win else [[]] * 5
            p_pose, p_l, p_r, p_both, p_body = (pct(c) for c in cols)
            white = (230, 230, 230)
            sw_text = f"{shoulder_w:.2f}" if shoulder_w is not None else "-"
            lines = [
                (f"FPS {fps:5.1f}   hands {t_hands:4.1f}ms  pose {t_pose:4.1f}ms",
                 (80, 220, 80) if fps >= 20 else (60, 60, 255)),
                (f"Pose found      {p_pose:5.1f}%", color_for(p_pose)),
                (f"Left hand       {p_l:5.1f}%", color_for(p_l)),
                (f"Right hand      {p_r:5.1f}%", color_for(p_r)),
                (f"Both hands      {p_both:5.1f}%", color_for(p_both)),
                (f"Body in frame   {p_body:5.1f}%", color_for(p_body)),
                (f"Shoulder width  {sw_text}  {hint}", hint_color),
                (f"last {len(win)} frames | q quit  r reset  s shot", white),
            ]

            y = 28
            for text, color in lines:
                cv2.putText(view, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
                cv2.putText(view, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1)
                y += 24

            cv2.imshow("Check Camera (q=quit)", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                win.clear()
            elif key == ord("s"):
                out_dir = Path("results/camera_check")
                out_dir.mkdir(parents=True, exist_ok=True)
                fn = out_dir / f"check_{int(time.time())}.jpg"
                cv2.imwrite(str(fn), view)
                print(f"[Saved] {fn}")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
