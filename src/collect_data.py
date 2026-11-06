"""
Interactive Sign Language Data Collector / เก็บข้อมูลท่าทางภาษามือ
Press SPACE to record a 1.8-second motion sequence.
Saves extracted landmarks into data/raw/<word>/<person>_<timestamp>.npy

Usage:
    python src/collect_data.py
    python src/collect_data.py --words hello thanks yes no
    python src/collect_data.py --person user1
"""
import argparse
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
from landmarkers import Camera, Tracker, hands_by_side, draw_hand, draw_pose


def get_sample_count(raw_dir, word, person=None):
    folder = raw_dir / word
    if not folder.exists():
        return 0
    if person:
        return len(list(folder.glob(f"{person}_*.npy")))
    return len(list(folder.glob("*.npy")))


def draw_hud(img, words_list, active_idx, person, state_text, state_color, counts_map, extra_info=""):
    """Draw information overlay on the camera view with multi-gesture tabs."""
    h, w = img.shape[:2]
    banner_h = 95
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)

    active_word = words_list[active_idx]
    active_count = counts_map.get(active_word, 0)

    # Line 1: Gesture Selection Tabs (1 - N)
    tab_x = 15
    for i, word in enumerate(words_list[:8]):
        cnt = counts_map.get(word, 0)
        is_active = (i == active_idx)
        tag = f"[{i+1}] {word} ({cnt})"
        color = (0, 255, 0) if is_active else (180, 180, 180)
        thick = 2 if is_active else 1
        cv2.putText(img, tag, (tab_x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, thick, cv2.LINE_AA)
        tab_x += len(tag) * 11 + 14

    # Line 2: Active Target Info
    header = f"Target: [{active_word.upper()}]  |  Signer: [{person}]  |  Samples Collected: {active_count}"
    cv2.putText(img, header, (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    # Line 3: State line (IDLE / COUNTDOWN / RECORDING)
    cv2.putText(img, state_text, (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.68, state_color, 2, cv2.LINE_AA)

    if extra_info:
        cv2.putText(img, extra_info, (w - 320, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    # Bottom helper bar
    bottom_h = 28
    cv2.rectangle(img, (0, h - bottom_h), (w, h), (15, 15, 15), -1)
    help_text = "[SPACE] Record  |  [1-9 / TAB] Switch Gesture  |  [D] Delete Last  |  [Q] Quit"
    cv2.putText(img, help_text, (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Collect sign language motion samples.")
    parser.add_argument("--words", nargs="+", default=None,
                        help="List of gestures to collect, e.g. --words sawasdee khopkhun chai mai_chai")
    parser.add_argument("--word", type=str, default="sawasdee",
                        help="First gesture name (default: sawasdee)")
    parser.add_argument("--person", type=str, default=config.DEFAULT_PERSON,
                        help=f"Signer person ID (default: {config.DEFAULT_PERSON})")
    parser.add_argument("--duration", type=float, default=1.8,
                        help="Duration of each recording in seconds (default: 1.8)")
    args = parser.parse_args()

    words_list = args.words if args.words else [args.word]
    active_idx = 0
    person = args.person
    record_duration = args.duration

    cam = Camera()
    print(f"[Ready] Camera active at {cam.actual_size()}. Collecting gestures: {words_list}")
    print("[Controls] SPACE = Start Countdown (3s) & Record | 1..9 = Switch Gesture | D = Delete Last | Q = Quit")

    state = "idle"
    countdown_start = 0.0
    rec_start = 0.0
    current_frames = []
    last_saved_file = None
    last_id = 0

    # Cache sample counts to avoid ~210 disk IO calls per second
    counts_map = {w: get_sample_count(config.DATA_RAW_DIR, w, person) for w in words_list}

    with Tracker() as tracker:
        while True:
            frame, last_id = cam.read(last_id)
            if frame is None:
                continue

            if config.FLIP_IMAGE:
                frame = cv2.flip(frame, 1)

            hand_res, pose_res, _ = tracker.process(frame)
            hands = hands_by_side(hand_res)
            pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
            frame_vec = extract_frame_features(pose_lms, hands)

            # Draw visual landmarks
            view = frame.copy()
            if pose_lms is not None:
                draw_pose(view, pose_lms)
            for side, lms in hands.items():
                if lms is not None:
                    draw_hand(view, lms, side)

            now = time.time()
            active_word = words_list[active_idx]

            # State Machine
            if state == "idle":
                state_text = "READY - Press [SPACE] to record"
                state_color = (0, 220, 0)
                extra_info = ""

            elif state == "countdown":
                elapsed = now - countdown_start
                remaining = max(0.0, 3.0 - elapsed)
                count_num = int(np.ceil(remaining))
                if remaining <= 0.0:
                    state = "recording"
                    rec_start = time.time()
                    current_frames = []
                    state_text = "RECORDING NOW!"
                    state_color = (0, 0, 255)
                else:
                    state_text = f"GET READY: Starting in {count_num}..."
                    state_color = (0, 200, 255)
                    h, w = view.shape[:2]
                    cv2.putText(view, str(count_num), (w // 2 - 30, h // 2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 220, 255), 5, cv2.LINE_AA)
                extra_info = "Hold starting position"

            elif state == "recording":
                current_frames.append(frame_vec)
                elapsed = now - rec_start
                progress = min(1.0, elapsed / record_duration)

                h, w = view.shape[:2]
                cv2.circle(view, (w - 40, 40), 16, (0, 0, 255), -1)
                cv2.putText(view, "REC", (w - 85, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                bar_w = int((w - 40) * progress)
                cv2.rectangle(view, (20, h - 45), (20 + bar_w, h - 35), (0, 0, 255), -1)

                state_text = f"RECORDING... ({elapsed:.1f}s / {record_duration:.1f}s)"
                state_color = (0, 0, 255)
                extra_info = "Perform the gesture motion"

                if elapsed >= record_duration:
                    state = "idle"
                    if len(current_frames) >= 10:
                        save_folder = config.DATA_RAW_DIR / active_word
                        save_folder.mkdir(parents=True, exist_ok=True)
                        ts = int(time.time() * 1000)
                        file_path = save_folder / f"{person}_{ts}.npy"
                        np.save(str(file_path), np.array(current_frames, dtype=np.float32))
                        last_saved_file = file_path
                        counts_map[active_word] = counts_map.get(active_word, 0) + 1
                        print(f"[Saved] {file_path.name} to '{active_word}/' with {len(current_frames)} frames.")
                    else:
                        print(f"[Warning] Sample had too few frames ({len(current_frames)}), discarded.")
                    current_frames = []

            draw_hud(view, words_list, active_idx, person, state_text, state_color, counts_map, extra_info)
            cv2.imshow("Sign Data Collector", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                if state == "idle":
                    state = "countdown"
                    countdown_start = time.time()
            elif ord("1") <= key <= ord("9"):
                target_i = key - ord("1")
                if target_i < len(words_list):
                    active_idx = target_i
                    print(f"[Switch] Active gesture: [{words_list[active_idx]}]")
            elif key in (ord("\t"), ord("n"), ord("N")):
                active_idx = (active_idx + 1) % len(words_list)
                print(f"[Switch] Active gesture: [{words_list[active_idx]}]")
            elif key in (ord("d"), ord("D")):
                if last_saved_file and last_saved_file.exists():
                    last_saved_file.unlink()
                    counts_map[active_word] = max(0, counts_map.get(active_word, 1) - 1)
                    print(f"[Deleted] Removed last sample: {last_saved_file.name}")
                    last_saved_file = None

    cam.release()
    cv2.destroyAllWindows()
    print("[Exited] Data collector closed.")


if __name__ == "__main__":
    main()
