"""
Shared settings for every script in the project.
ค่าตั้งที่ทุกสคริปต์ใช้ร่วมกัน — แก้ที่นี่ที่เดียว
"""
import re
from pathlib import Path

# ---------- Paths / ที่อยู่ไฟล์ ----------
ROOT = Path(__file__).resolve().parent
MP_MODEL_DIR = ROOT / "models" / "mediapipe"
HAND_MODEL_PATH = MP_MODEL_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = MP_MODEL_DIR / "pose_landmarker_lite.task"

DATA_DIR = ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
PRESETS_CONFIG_PATH = DATA_DIR / "scenario_presets.json"
MODEL_DIR = ROOT / "models"
SCENARIO_MODELS_DIR = MODEL_DIR / "scenarios"
SAVED_SKLEARN_MODEL = MODEL_DIR / "rf.pkl"
SAVED_KINEMATIC_RF_MODEL = MODEL_DIR / "rf_kinematic.pkl"
SAVED_TORCH_MODEL = MODEL_DIR / "gru.pt"
SAVED_CNN_MODEL = MODEL_DIR / "cnn1d.pt"
SAVED_ENSEMBLE_META = MODEL_DIR / "ensemble_meta.json"
SAVED_LABELS = MODEL_DIR / "labels.json"
SAVED_SCALER = MODEL_DIR / "scaler.pkl"


def scenario_slug(scenario_name):
    """Converts a human-readable scenario name into a safe filesystem folder slug."""
    if not scenario_name:
        return "default"
    slug = re.sub(r"[^a-z0-9]+", "_", str(scenario_name).lower()).strip("_")
    return slug or "default"


def get_scenario_model_dir(scenario_name=None):
    """Returns the dedicated model directory for a scenario, or global MODEL_DIR if None."""
    if not scenario_name:
        return MODEL_DIR
    return SCENARIO_MODELS_DIR / scenario_slug(scenario_name)

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# ---------- Sequence & Feature Settings ----------
TARGET_FRAMES = 24        # แต่ละท่าถูก resample ให้มีความยาว 24 เฟรม
NUM_FEATURES = 112        # 13 pose (26) + 21*2 hands (84) + 2 flags = 112
NUM_SEQ_FEATURES = 222    # 110 pos + 110 velocity + 2 hand presence flags = 222
DEFAULT_PERSON = "p1"

# ---------- Fast Continuous Signing / Dynamic Early-Stop ----------
MIN_SIGN_DURATION = 0.45         # Minimum seconds before early-stop can trigger (supports fast signs)
MAX_SIGN_DURATION = 1.80         # Maximum recording duration window (and Studio sample length)
END_HOLD_STILL_SEC = 0.32        # Stop early if wrists stay still for this duration after active motion
WRIST_STILL_THRESHOLD = 0.018    # Lowered so slow deliberate signs are not prematurely cut off
WRIST_ACTIVE_PEAK_THRESHOLD = 0.045  # Gesture must show real movement before stillness early-stop fires
PRE_ROLL_FRAMES = 8              # Pre-roll buffer frames kept during arming (~0.27s at 30 FPS)

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

# ---------- check_camera & Auto-Trigger Settings ----------
STATS_WINDOW = 90          # คิด % การเจอมือจากกี่เฟรมล่าสุด (~3 วินาที)
SHOULDER_WIDTH_GOOD = (0.28, 0.48)  # Rough distance hint (ratio 0..1)
READY_LINE_RATIO = 0.74    # ระดับเส้นเริ่มอัตโนมัติ (เดิม 0.65 สูงเกินไป ปรับต่ำลงมาเป็น 0.74 ระดับชายโครง/เอวบน)
READY_LINE_SHOULDER_MULT = 0.92  # ตัวคูณระยะจากไหล่ลงมา (เดิม 0.65 อยู่แถบอกบน ปรับเป็น 0.92 ลงมาระดับลิ้นปี่/ชายโครง)
