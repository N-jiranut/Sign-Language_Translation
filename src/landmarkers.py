"""
Camera + MediaPipe helpers shared by check_camera, collect_data and realtime.
ตัวช่วยเรื่องกล้องและ MediaPipe ที่ใช้ร่วมกันหลายสคริปต์

Uses the MediaPipe *Tasks* API (HandLandmarker / PoseLandmarker).
The old mp.solutions API was removed from MediaPipe and must not be used.
"""
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


# --------------------------------------------------------------------------
# Camera: reads frames in a background thread so the main loop never waits
# กล้องอ่านภาพใน thread แยก ลูปหลักจะได้ไม่ต้องรอกล้อง
# --------------------------------------------------------------------------
class Camera:
    def __init__(self, index=config.CAMERA_INDEX, width=config.CAMERA_WIDTH,
                 height=config.CAMERA_HEIGHT, fps=config.CAMERA_FPS):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {index} / เปิดกล้องหมายเลข {index} ไม่ได้ "
                "(ลองเปลี่ยน CAMERA_INDEX ใน config.py)")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame = None
        self._frame_id = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._cond.notify_all()

    def read(self, last_id=0, timeout=1.0):
        """Wait for a frame newer than last_id. Returns (frame, frame_id).
        รอเฟรมใหม่กว่า last_id (ไม่ประมวลผลเฟรมเดิมซ้ำ)"""
        with self._cond:
            ok = self._cond.wait_for(lambda: self._frame_id > last_id, timeout)
            if not ok:
                return None, last_id
            return self._frame.copy(), self._frame_id

    def actual_size(self):
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# --------------------------------------------------------------------------
# Tracker: HandLandmarker + PoseLandmarker in VIDEO mode
# --------------------------------------------------------------------------
class Tracker:
    def __init__(self):
        for path in (config.HAND_MODEL_PATH, config.POSE_MODEL_PATH):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path.name} / ไม่พบไฟล์โมเดล {path.name}\n"
                    "Run: python src/download_models.py")

        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(config.HAND_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=config.HAND_DETECTION_CONF,
            min_hand_presence_confidence=config.HAND_PRESENCE_CONF,
            min_tracking_confidence=config.HAND_TRACKING_CONF,
        )
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(config.POSE_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_DETECTION_CONF,
            min_pose_presence_confidence=config.POSE_PRESENCE_CONF,
            min_tracking_confidence=config.POSE_TRACKING_CONF,
        )
        self.hands = vision.HandLandmarker.create_from_options(hand_opts)
        self.pose = vision.PoseLandmarker.create_from_options(pose_opts)
        self._last_ts = -1

    def _next_ts(self):
        # VIDEO mode needs strictly increasing timestamps (ms)
        # โหมดวิดีโอต้องใช้ timestamp ที่เพิ่มขึ้นเสมอ
        ts = time.monotonic_ns() // 1_000_000
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def process(self, frame_bgr):
        """frame_bgr must already be flipped if config.FLIP_IMAGE.
        Returns (hand_result, pose_result, timings_ms)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = self._next_ts()

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
    """Return {'Left': landmarks|None, 'Right': landmarks|None}.
    If MediaPipe reports two hands with the same label, keep the more confident one.
    แยกมือซ้าย/ขวา ถ้าได้ป้ายซ้ำกันเลือกตัวที่มั่นใจกว่า"""
    out = {"Left": None, "Right": None}
    best = {"Left": -1.0, "Right": -1.0}
    for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
        side, score = handed[0].category_name, handed[0].score
        if side in out and score > best[side]:
            out[side], best[side] = lms, score
    return out


# --------------------------------------------------------------------------
# Drawing (mp.solutions.drawing_utils no longer exists, so draw with OpenCV)
# วาดเองด้วย OpenCV เพราะ drawing_utils เดิมถูกถอดไปแล้ว
# --------------------------------------------------------------------------
HAND_COLORS = {"Left": (255, 160, 0), "Right": (0, 200, 255)}  # BGR
_HAND_EDGES = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
_POSE_EDGES = [("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"),
               ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist")]


def _px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def draw_hand(img, lms, side):
    h, w = img.shape[:2]
    color = HAND_COLORS.get(side, (0, 255, 0))
    for a, b in _HAND_EDGES:
        cv2.line(img, _px(lms[a], w, h), _px(lms[b], w, h), color, 2)
    for lm in lms:
        cv2.circle(img, _px(lm, w, h), 3, (255, 255, 255), -1)
    x, y = _px(lms[0], w, h)
    cv2.putText(img, side, (x - 20, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_pose(img, pose_lms):
    h, w = img.shape[:2]
    idx = config.POSE_KEEP
    for a, b in _POSE_EDGES:
        cv2.line(img, _px(pose_lms[idx[a]], w, h), _px(pose_lms[idx[b]], w, h), (120, 220, 120), 2)
    for i in idx.values():
        cv2.circle(img, _px(pose_lms[i], w, h), 5, (80, 200, 80), -1)