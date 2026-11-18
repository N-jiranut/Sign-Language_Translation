"""
Camera & Posture Calibration Guide / ระบบจัดตำแหน่งกล้องและสรีระผู้ใช้
มีจุดมาร์คกึ่งกลางลำตัว กรอบเล็งการกางข้อศอก (Elbow Span Target) และทดสอบท่าพนมมือก่อนเริ่มเก็บข้อมูลจริง

Workflow / ลำดับขั้นตอน:
  1. จัดตัวให้อยู่กึ่งกลางหน้าจอ (สังเกตเส้น Center Guideline และเป้ากลางอก)
  2. กางข้อศอกออก 2 ข้าง ให้แตะกรอบขอบจอซ้าย-ขวา (Elbow Targets) เพื่อวัดระยะห่างที่พอดี
  3. ทดสอบยกมือ 2 ข้าง หรือพนมมือ (Wai Clasp Test)
  4. เมื่อระบบขึ้นแถบสีเขียว [★ ALIGNMENT READY ★] ให้กด [SPACEBAR] เพื่อเริ่มเก็บข้อมูลทันที!
"""
import os
import subprocess
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

import config
from landmarkers import Camera, Tracker, hands_by_side, draw_hand, draw_pose

IDX = config.POSE_KEEP


# --------------------------------------------------------------------------
# Helper Drawing Utilities / ฟังก์ชันวาดเส้นช่วยจัดตำแหน่ง
# --------------------------------------------------------------------------
def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_len=8, gap_len=6):
    """Draws a dashed line between pt1 and pt2."""
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
    """Draws sleek corner brackets around a rectangular bounding box."""
    c = corner_len
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + c, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + c), color, thickness, cv2.LINE_AA)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - c, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + c), color, thickness, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + c, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - c), color, thickness, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - c, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - c), color, thickness, cv2.LINE_AA)


def draw_crosshair_target(img, cx, cy, color, radius=18):
    """Draws a crosshair bullseye target at (cx, cy)."""
    cv2.circle(img, (cx, cy), radius, color, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, color, -1, cv2.LINE_AA)
    cv2.line(img, (cx - radius - 6, cy), (cx + radius + 6, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - radius - 6), (cx, cy + radius + 6), color, 1, cv2.LINE_AA)


def distance_hint(shoulder_w):
    """Classify rough distance from shoulder width in normalized coordinates (0..1)."""
    if shoulder_w is None:
        return "no body", (160, 160, 160)
    lo, hi = getattr(config, "SHOULDER_WIDTH_GOOD", (0.28, 0.48))
    if shoulder_w < lo:
        return f"too far (< {lo:.2f})", (60, 60, 255)
    if shoulder_w > hi:
        return f"too close (> {hi:.2f})", (60, 60, 255)
    return "distance OK", (80, 220, 80)


def body_inside(pose_lms, margin=0.03):
    """True if shoulders and elbows are not clipped by the frame edges."""
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


def launch_collector():
    """Seamlessly launches collect_data.py forwarding any CLI arguments."""
    collector_script = Path(__file__).resolve().parent / "collect_data.py"
    cmd = [sys.executable, str(collector_script)] + sys.argv[1:]
    print("\n" + "=" * 60)
    print(" [LAUNCH] Calibration verified! Starting Data Collection... ")
    print(f" Running: {' '.join(cmd)}")
    print("=" * 60 + "\n")
    subprocess.run(cmd)


def main():
    print("=" * 65)
    print(" Interactive Camera & Posture Calibration Guide ")
    print(" ระบบจัดตำแหน่งกล้องและสรีระผู้ใช้ (Visual Alignment Target) ")
    print("=" * 65)
    print("คำแนะนำบนหน้าจอ:")
    print(" 1. ขยับตัวให้อยู่กึ่งกลางเป้า Center Crosshair")
    print(" 2. กางข้อศอกออก 2 ข้าง ให้แตะกรอบขอบจอซ้าย-ขวา เพื่อวัดระยะ")
    print(" 3. ลองพนมมือ หรือยกมือ 2 ข้างเพื่อทดสอบการตรวจจับ")
    print(" 4. เมื่อขึ้น [ALIGNMENT READY] ให้กด [SPACEBAR] เริ่มเก็บข้อมูลได้ทันที!")
    print(" ปุ่มลัด: [SPACE] เริ่มเก็บข้อมูล | [G] ปิด/เปิดไกด์ | [R] รีเซ็ต | [Q] ออก")
    print("=" * 65)

    cam = Camera()
    actual_w, actual_h = cam.actual_size()
    print(f"[Camera] Active at resolution {actual_w}x{actual_h}")

    win = deque(maxlen=config.STATS_WINDOW)
    fps = 0.0
    t_hands = 0.0
    t_pose = 0.0
    a = 0.1
    last_id = 0
    show_guides = True
    ready_streak = 0
    start_collect_requested = False
    flip_img = config.FLIP_IMAGE

    with Tracker() as tracker:
        while True:
            t0 = time.perf_counter()
            frame, last_id = cam.read(last_id)
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

            sample = (has_pose, has_l, has_r, has_both, in_frame)
            win.append(sample)

            dt = time.perf_counter() - t0
            cur_fps = (1.0 / dt) if dt > 0 else 30.0
            fps = (1 - a) * fps + a * cur_fps
            t_hands = (1 - a) * t_hands + a * ms["hands"]
            t_pose = (1 - a) * t_pose + a * ms["pose"]

            view = frame.copy()
            h, w = view.shape[:2]

            # Draw standard skeleton landmarks
            if pose_lms is not None:
                draw_pose(view, pose_lms)
            for side, lms in hands.items():
                if lms is not None:
                    draw_hand(view, lms, side)

            # --------------------------------------------------------------
            # Calibration Metrics Calculation
            # --------------------------------------------------------------
            cx, cy = 0.5, 0.5
            shoulder_w = None
            center_diff = 0.0
            is_centered = False
            headroom_ok = False
            ready_y_px = int(getattr(config, "READY_LINE_RATIO", 0.74) * h)
            left_elbow_in_bracket = False
            right_elbow_in_bracket = False
            elbows_span_ok = False
            elbow_clipped = False
            wai_clasp_detected = False

            if pose_lms is not None:
                try:
                    l_sh = pose_lms[IDX["l_shoulder"]]
                    r_sh = pose_lms[IDX["r_shoulder"]]
                    nose = pose_lms[IDX["nose"]]
                    l_elb = pose_lms[IDX["l_elbow"]]
                    r_elb = pose_lms[IDX["r_elbow"]]

                    # Center & shoulder calculations
                    cx = (l_sh.x + r_sh.x) / 2.0
                    cy = (l_sh.y + r_sh.y) / 2.0
                    shoulder_w = np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
                    center_diff = cx - 0.5
                    is_centered = abs(center_diff) <= 0.05

                    # Headroom (nose position)
                    headroom_ok = (0.12 <= nose.y <= 0.42)

                    # Ready line (lower chest / rib / waist level)
                    mult = getattr(config, "READY_LINE_SHOULDER_MULT", 0.92)
                    chest_y = cy + mult * max(shoulder_w, 0.15)
                    ready_y_px = int(np.clip(chest_y, 0.40, 0.90) * h)

                    # Elbow span & edge clipping
                    min_elb_x = min(l_elb.x, r_elb.x)
                    max_elb_x = max(l_elb.x, r_elb.x)

                    # Target bracket zones at screen edges (left: 0.03-0.18, right: 0.82-0.97)
                    left_elbow_in_bracket = (0.03 <= min_elb_x <= 0.18) and (0.38 <= l_elb.y <= 0.82)
                    right_elbow_in_bracket = (0.82 <= max_elb_x <= 0.97) and (0.38 <= r_elb.y <= 0.82)
                    elbows_span_ok = left_elbow_in_bracket and right_elbow_in_bracket
                    elbow_clipped = (min_elb_x < 0.02) or (max_elb_x > 0.98)

                except Exception:
                    pass

            # Wai / Clasp detection test
            if has_both:
                try:
                    lw = hands["Left"][0]
                    rw = hands["Right"][0]
                    wrist_dist = np.hypot(lw.x - rw.x, lw.y - rw.y)
                    if wrist_dist < 0.18:
                        wai_clasp_detected = True
                except Exception:
                    pass

            # Distance classification
            hint, hint_color = distance_hint(shoulder_w)
            lo_w, hi_w = getattr(config, "SHOULDER_WIDTH_GOOD", (0.28, 0.48))
            shoulder_dist_ok = (shoulder_w is not None) and (lo_w <= shoulder_w <= hi_w)
            distance_ok = elbows_span_ok or shoulder_dist_ok

            # Overall Calibration Status
            all_ready = has_pose and is_centered and distance_ok and in_frame and (not elbow_clipped)
            if all_ready:
                ready_streak += 1
            else:
                ready_streak = max(0, ready_streak - 1)

            is_calibrated = (ready_streak >= 12)

            # --------------------------------------------------------------
            # Visual Guides Overlay (When show_guides = True)
            # --------------------------------------------------------------
            if show_guides:
                center_x_px = w // 2

                # 1. Vertical Center Line
                c_line_color = (0, 240, 120) if is_centered else (0, 165, 255)
                draw_dashed_line(view, (center_x_px, 85), (center_x_px, h - 30), c_line_color, 1, 8, 6)

                # Center Bullseye Target on signer chest
                if has_pose:
                    t_x, t_y = int(cx * w), int(cy * h)
                    t_color = (0, 240, 120) if is_centered else (0, 165, 255)
                    draw_crosshair_target(view, t_x, t_y, t_color, radius=16)

                    # Direction guidance arrow if off-center
                    if not is_centered:
                        if center_diff < -0.05:
                            cv2.putText(view, "SHIFT RIGHT ->", (t_x + 22, t_y + 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2, cv2.LINE_AA)
                        else:
                            cv2.putText(view, "<- SHIFT LEFT", (t_x - 145, t_y + 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(view, "CENTER OK", (t_x + 20, t_y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 120), 2, cv2.LINE_AA)

                # 2. Headroom Top Margin Guide
                headroom_y = int(0.10 * h)
                hr_color = (0, 220, 120) if headroom_ok else (0, 140, 255)
                draw_dashed_line(view, (15, headroom_y), (w - 15, headroom_y), hr_color, 1, 6, 6)
                cv2.putText(view, "HEAD MARGIN (เว้นระยะเหนือศีรษะ)", (w - 230, headroom_y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

                # 3. Horizontal Chest Ready Line
                draw_dashed_line(view, (15, ready_y_px), (w - 15, ready_y_px), (0, 220, 255), 1, 10, 6)
                cv2.putText(view, "READY LINE (อก: ท่าพักอยู่ใต้เส้น / เริ่มทำท่ายกพ้นเส้น)",
                            (18, ready_y_px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 255), 1, cv2.LINE_AA)

                # 4. Left & Right Elbow Calibration Brackets (ศอกแตะขอบจอ)
                by1, by2 = int(0.42 * h), int(0.75 * h)
                # Left Bracket
                bx1_l, bx2_l = int(0.03 * w), int(0.18 * w)
                color_l = (0, 240, 120) if left_elbow_in_bracket else ((0, 0, 255) if elbow_clipped else (120, 120, 120))
                draw_corner_bracket(view, bx1_l, by1, bx2_l, by2, color_l, thickness=2, corner_len=15)
                tag_l = "ELBOW OK" if left_elbow_in_bracket else "L ELBOW"
                cv2.putText(view, tag_l, (bx1_l + 4, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_l, 1, cv2.LINE_AA)

                # Right Bracket
                bx1_r, bx2_r = int(0.82 * w), int(0.97 * w)
                color_r = (0, 240, 120) if right_elbow_in_bracket else ((0, 0, 255) if elbow_clipped else (120, 120, 120))
                draw_corner_bracket(view, bx1_r, by1, bx2_r, by2, color_r, thickness=2, corner_len=15)
                tag_r = "ELBOW OK" if right_elbow_in_bracket else "R ELBOW"
                cv2.putText(view, tag_r, (bx1_r + 4, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_r, 1, cv2.LINE_AA)

                # Wai / Hands Clasp Test Badge
                if wai_clasp_detected:
                    cv2.putText(view, "✓ WAI / CLASP TEST DETECTED (ท่าพนมมือตรวจพบสมบูรณ์)",
                                (18, ready_y_px + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 120), 2, cv2.LINE_AA)

            # --------------------------------------------------------------
            # Top HUD: 4-Point Realtime Checklist & Dynamic Action Prompt
            # --------------------------------------------------------------
            banner_h = 96
            overlay = view.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.85, view, 0.15, 0, view)

            # Line 1: 4-Point Checklist
            # Point 1: Center
            p1_color = (0, 240, 120) if is_centered else (0, 165, 255)
            p1_text = "[1] Center: OK" if is_centered else "[1] Center: Align"
            # Point 2: Distance
            p2_color = (0, 240, 120) if distance_ok else (0, 80, 255)
            p2_text = "[2] Distance: OK" if distance_ok else "[2] Dist: Check"
            # Point 3: Elbows in frame
            p3_color = (0, 240, 120) if (in_frame and not elbow_clipped) else (0, 0, 255)
            p3_text = "[3] Elbows: Inside" if (in_frame and not elbow_clipped) else "[3] Elbows: Clipped!"
            # Point 4: Hands
            p4_color = (0, 240, 120) if has_both else ((0, 200, 255) if (has_l or has_r) else (140, 140, 140))
            p4_text = "[4] Hands: 2 Found" if has_both else "[4] Hands: Test"

            chk_x = 15
            for txt, col in [(p1_text, p1_color), (p2_text, p2_color), (p3_text, p3_color), (p4_text, p4_color)]:
                cv2.putText(view, txt, (chk_x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2, cv2.LINE_AA)
                chk_x += len(txt) * 10 + 16

            # Meta stats (FPS) on top-right
            cv2.putText(view, f"{fps:4.1f} FPS", (w - 85, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

            # Line 2: Big Action Status Banner
            if is_calibrated:
                banner_text = "★ ALIGNMENT READY! กด [SPACE] เริ่มเก็บข้อมูลได้ทันที ★"
                banner_color = (0, 255, 120)
                cv2.putText(view, banner_text, (15, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62, banner_color, 2, cv2.LINE_AA)
            else:
                # Provide contextual guidance
                if not has_pose:
                    msg = "รอตรวจจับสรีระ: กรุณานั่งตรงหน้ากล้อง (เห็นศีรษะถึงเอว)"
                elif not is_centered:
                    msg = "จัดกึ่งกลาง: ขยับตัวให้อยู่กึ่งกลางเส้น Center ประแนวตั้ง"
                elif elbow_clipped:
                    msg = "ระยะใกล้ไป! ศอกตกขอบจอ ให้ถอยหลังออกไปอีกนิด"
                elif not distance_ok:
                    msg = "วัดระยะห่าง: กางศอกออก 2 ข้าง ให้แตะกรอบเป้าขอบจอซ้าย-ขวา"
                elif not has_both:
                    msg = "ทดสอบมือ: ลองยกมือ 2 ข้างขึ้นมา หรือพนมมือทดสอบ"
                else:
                    msg = "รักษาร่างกายให้อยู่ในตำแหน่งอีกสักครู่..."
                cv2.putText(view, msg, (15, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2, cv2.LINE_AA)

            # Line 3: Calibrated Hardware Setup Reference
            flip_tag = "Flip: ON (Mirror)" if flip_img else "Flip: OFF (Direct)"
            setup_info = f"Setup: Height 161 cm | Feet Dist 90 cm | {flip_tag} (Press [F] to toggle)"
            cv2.putText(view, setup_info, (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

            # Bottom helper bar
            b_bar_h = 28
            cv2.rectangle(view, (0, h - b_bar_h), (w, h), (15, 15, 15), -1)
            b_help = "[SPACE] Collect  |  [F] Toggle Flip  |  [G] Guides  |  [R] Reset  |  [S] Screenshot  |  [Q] Quit"
            cv2.putText(view, b_help, (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Check Camera & Calibration (Press SPACE to Collect)", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                # Launch data collection
                start_collect_requested = True
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
                out_dir = Path("results/camera_check")
                out_dir.mkdir(parents=True, exist_ok=True)
                fn = out_dir / f"calibration_{int(time.time())}.jpg"
                cv2.imwrite(str(fn), view)
                print(f"[Saved Screenshot] {fn}")

    cam.release()
    cv2.destroyAllWindows()

    if start_collect_requested:
        launch_collector()


if __name__ == "__main__":
    main()
