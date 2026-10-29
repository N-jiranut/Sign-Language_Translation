"""
Shared settings for every script in the project.
ค่าตั้งที่ทุกสคริปต์ใช้ร่วมกัน — แก้ที่นี่ที่เดียว
"""
from pathlib import Path

# ---------- Paths / ที่อยู่ไฟล์ ----------
ROOT = Path(__file__).resolve().parent
MP_MODEL_DIR = ROOT / "models" / "mediapipe"
HAND_MODEL_PATH = MP_MODEL_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = MP_MODEL_DIR / "pose_landmarker_lite.task"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# ---------- Camera / กล้อง ----------
CAMERA_INDEX = 0          # เปลี่ยนเป็น 1, 2 ถ้ามีหลายกล้อง
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
# Flip the image like a mirror BEFORE detection.
# MediaPipe assumes a mirrored (selfie) image when it labels Left/Right hands,
# so keep this True in every script (collect, train, realtime) — never mix.
# พลิกภาพแบบกระจกก่อนส่งเข้า MediaPipe เพื่อให้ป้าย Left/Right ตรงกับมือจริง
# ต้องตั้งเหมือนกันทุกสคริปต์
FLIP_IMAGE = True

# ---------- MediaPipe ----------
NUM_HANDS = 2
HAND_DETECTION_CONF = 0.3   # ต่ำลงเพื่อให้เจอมือที่หันด้านสันได้ง่ายขึ้น
HAND_PRESENCE_CONF = 0.5
HAND_TRACKING_CONF = 0.5

POSE_DETECTION_CONF = 0.5
POSE_PRESENCE_CONF = 0.5
POSE_TRACKING_CONF = 0.5

# Pose landmark indices we keep (13 points)
# จุด Pose ที่ใช้: จมูก ไหล่ ข้อศอก ข้อมือ นิ้วก้อย นิ้วชี้ นิ้วโป้ง
POSE_KEEP = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_pinky": 17, "r_pinky": 18,
    "l_index": 19, "r_index": 20,
    "l_thumb": 21, "r_thumb": 22,
}

# ---------- check_camera ----------
STATS_WINDOW = 90          # คิด % การเจอมือจากกี่เฟรมล่าสุด (~3 วินาที)