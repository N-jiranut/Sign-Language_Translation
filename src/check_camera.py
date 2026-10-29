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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from landmarkers import Camera, Tracker, hands_by_side, draw_hand, draw_pose  # noqa: E402

IDX = config.POSE_KEEP
BODY_CHECK = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow"]


def body_in_frame(pose_lms, margin=0.02, min_vis=0.5):
    """True if shoulders and elbows are visible and inside the image.
    ไหล่และข้อศอกทั้งสองข้างอยู่ในภาพหรือไม่"""
    for name in BODY_CHECK:
        lm = pose_lms[IDX[name]]
        if lm.visibility is not None and lm.visibility < min_vis:
            return False
        if not (margin <= lm.x <= 1 - margin and margin <= lm.y <= 1 - margin):
            return False
    return True


def pct(values):
    return 100.0 * sum(values) / len(values) if values else 0.0


def color_for(p):
    if p >= 80:
        return (80, 220, 80)    # green
    if p >= 50:
        return (0, 200, 255)    # yellow/orange
    return (60, 60, 255)        # red


def distance_hint(shoulder_w):
    # Rough guide for a seated user ~70-90 cm from a normal webcam.
    # ค่าคร่าว ๆ ปรับได้ตามกล้องที่ใช้
    if shoulder_w is None:
        return "no body", (60, 60, 255)
    if shoulder_w < 0.25:
        return "too far - move closer", (0, 200, 255)
    if shoulder_w > 0.50:
        return "too close - move back", (0, 200, 255)
    return "distance OK", (80, 220, 80)


def draw_panel(img, lines):
    """lines: list of (text, color)."""
    pad, lh = 8, 22
    w = 330
    h = pad * 2 + lh * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (pad, pad + lh * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main():
    cam = Camera()
    print(f"Camera opened: {cam.actual_size()} (requested "
          f"{config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT})")

    win = deque(maxlen=config.STATS_WINDOW)     # recent frames
    total = {"frames": 0, "pose": 0, "left": 0, "right": 0, "both": 0, "body": 0}
    t_hands = t_pose = 0.0
    fps = 0.0
    last_id = 0
    prev_t = time.perf_counter()
    shots = 0

    with Tracker() as tracker:
        while True:
            frame, last_id = cam.read(last_id)
            if frame is None:
                print("No frame from camera / ไม่ได้รับภาพจากกล้อง")
                continue
            if config.FLIP_IMAGE:
                frame = cv2.flip(frame, 1)

            hand_res, pose_res, ms = tracker.process(frame)
            hands = hands_by_side(hand_res)
            pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None

            has_pose = pose_lms is not None
            has_l = hands["Left"] is not None
            has_r = hands["Right"] is not None
            in_frame = has_pose and body_in_frame(pose_lms)
            win.append((has_pose, has_l, has_r, has_l and has_r, in_frame))
            for k, v in zip(["pose", "left", "right", "both", "body"],
                            (has_pose, has_l, has_r, has_l and has_r, in_frame)):
                total[k] += v
            total["frames"] += 1

            # smoothed timings / ค่าเฉลี่ยแบบนุ่ม
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            a = 0.1
            fps = (1 - a) * fps + a * (1.0 / dt if dt > 0 else 0)
            t_hands = (1 - a) * t_hands + a * ms["hands"]
            t_pose = (1 - a) * t_pose + a * ms["pose"]

            # draw
            view = frame.copy()
            if pose_lms is not None:
                draw_pose(view, pose_lms)
            for side, lms in hands.items():
                if lms is not None:
                    draw_hand(view, lms, side)

            shoulder_w = None
            if pose_lms is not None:
                shoulder_w = abs(pose_lms[IDX["l_shoulder"]].x - pose_lms[IDX["r_shoulder"]].x)
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
            draw_panel(view, lines)
            cv2.imshow("check_camera", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                win.clear()
                total = dict.fromkeys(total, 0)
                print("Stats reset / รีเซ็ตสถิติ")
            if key == ord("s"):
                out_dir = config.ROOT / "results" / "camera_check"
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"shot_{time.strftime('%Y%m%d_%H%M%S')}_{shots}.png"
                cv2.imwrite(str(path), view)
                shots += 1
                print(f"Saved {path}")

    cam.release()
    cv2.destroyAllWindows()

    n = total["frames"]
    if n:
        print("\n=== Summary since last reset / สรุปตั้งแต่รีเซ็ตล่าสุด ===")
        print(f"frames: {n}")
        for k in ["pose", "left", "right", "both", "body"]:
            print(f"{k:>6}: {100 * total[k] / n:5.1f}%")
        print(f"avg FPS (smoothed): {fps:.1f}, hands {t_hands:.1f} ms, pose {t_pose:.1f} ms")


if __name__ == "__main__":
    main()